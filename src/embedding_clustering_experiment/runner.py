"""Orchestrate one immutable, full-dataset A-D experiment run."""

from __future__ import annotations

import importlib.metadata
import logging
import platform
import shutil
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.testing import assert_frame_equal

from .artifacts import (
    artifact_inventory,
    atomic_write_json,
    atomic_write_parquet,
    create_run_directory,
    sha256_file,
)
from .config import ExperimentConfig
from .data import load_inputs
from .encoder import CachedSentenceEncoder, SentenceEncoder
from .evaluation import run_automatic_evaluation
from .experiments import (
    build_experiments,
    build_normalization_metadata,
    validate_experiment_outputs,
)

EncoderFactory = Callable[[dict[str, Any], Path], SentenceEncoder]


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"embedding_clustering.{log_path.parent.name}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def _flush_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        handler.flush()


def _package_versions() -> dict[str, str]:
    packages = (
        "numpy",
        "pandas",
        "pyarrow",
        "PyYAML",
        "scikit-learn",
        "sentence-transformers",
        "torch",
    )
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _write_result_tables(
    run_dir: Path,
    tables: dict[str, pd.DataFrame],
    compression: str,
) -> None:
    for name, frame in tables.items():
        destination = run_dir / "results" / f"{name}.parquet"
        atomic_write_parquet(frame, destination, compression=compression)
        reread = pd.read_parquet(destination)
        assert_frame_equal(reread, frame.reset_index(drop=True), check_dtype=True)


def _write_named_tables(
    output_dir: Path,
    tables: dict[str, pd.DataFrame],
    compression: str,
) -> None:
    for name, frame in tables.items():
        destination = output_dir / f"{name}.parquet"
        atomic_write_parquet(frame, destination, compression=compression)
        reread = pd.read_parquet(destination)
        if list(reread.columns) != list(frame.columns) or len(reread) != len(frame):
            raise AssertionError(f"Serialized table failed read-back validation: {destination}")


def run(
    config: ExperimentConfig,
    *,
    encoder_factory: EncoderFactory = CachedSentenceEncoder,
) -> Path:
    values = config.values
    representative_path = config.resolve(values["inputs"]["representative_attributes"])
    opinion_path = config.resolve(values["inputs"]["opinion_units"])
    for path in (representative_path, opinion_path):
        if not path.exists():
            raise FileNotFoundError(f"Input does not exist: {path}")

    inputs = load_inputs(representative_path, opinion_path)
    results_root = config.resolve(values["project"]["results_root"])
    run_dir = create_run_directory(results_root)
    shutil.copy2(config.path, run_dir / "config.yaml")
    logger = _setup_logger(run_dir / "experiment.log")
    logger.info("Starting full-dataset run output=%s", run_dir)
    logger.info(
        "Validated separated inputs representative_rows=%d opinion_rows=%d common_reviews=%d",
        len(inputs.representative),
        len(inputs.opinion),
        len(set(inputs.representative["review_idx"]) & set(inputs.opinion["review_idx"])),
    )

    encoder = encoder_factory(values["embedding"], config.base_dir)
    logger.info(
        "Loaded embedding model=%s device=%s batch_size=%s",
        getattr(encoder, "model_id", values["embedding"]["model_id"]),
        getattr(encoder, "device", "test"),
        getattr(encoder, "batch_size", values["embedding"]["batch_size"]),
    )
    normalization_metadata = build_normalization_metadata(values, encoder, run_dir.name)
    logger.info(
        "Normalization lineage version=%s config_sha256=%s",
        normalization_metadata["normalization_version"],
        normalization_metadata["normalization_config_sha256"],
    )
    experiments = build_experiments(
        inputs.representative,
        inputs.opinion,
        encoder,
        values,
        logger,
        normalization_run_id=run_dir.name,
    )
    compression = str(values["output"]["parquet_compression"])
    _write_result_tables(run_dir, experiments.result_tables(), compression)
    _write_named_tables(run_dir / "clustering", experiments.diagnostic_tables(), compression)
    logger.info(
        "Serialized A-D outputs A/B=%d rows C/D=%d rows",
        len(experiments.experiment_a),
        len(experiments.experiment_c),
    )

    reread_results = {
        name: pd.read_parquet(run_dir / "results" / f"{name}.parquet")
        for name in experiments.result_tables()
    }
    reread_experiments = experiments.__class__(
        **reread_results,
        experiment_a_inventory=experiments.experiment_a_inventory,
        experiment_b_nodes=experiments.experiment_b_nodes,
        experiment_b_clusters=experiments.experiment_b_clusters,
        experiment_c_inventory=experiments.experiment_c_inventory,
        experiment_d_aspect_nodes=experiments.experiment_d_aspect_nodes,
        experiment_d_aspect_clusters=experiments.experiment_d_aspect_clusters,
        experiment_d_status_nodes=experiments.experiment_d_status_nodes,
        experiment_d_status_clusters=experiments.experiment_d_status_clusters,
        eligible_opinion=experiments.eligible_opinion,
    )
    validate_experiment_outputs(
        inputs.representative, experiments.eligible_opinion, reread_experiments
    )

    evaluation_summary: dict[str, Any] = {"enabled": False}
    if bool(values["evaluation"]["enabled"]):
        logger.info("Starting automatic evaluation over the full dataset")
        evaluation = run_automatic_evaluation(
            inputs.representative,
            inputs.opinion,
            experiments,
            values,
        )
        _write_named_tables(run_dir / "evaluation", evaluation.tables, compression)
        atomic_write_json(
            evaluation.summary,
            run_dir / "evaluation" / "automatic_evaluation_summary.json",
        )
        evaluation_summary = evaluation.summary
        logger.info(
            "Automatic evaluation completed checks=%d/%d common_reviews=%d "
            "human_review_tasks=%d human_cluster_tasks=%d",
            evaluation.summary["integrity_checks_passed"],
            evaluation.summary["integrity_check_count"],
            evaluation.summary["common_review_count"],
            evaluation.summary["user_evaluation_task_count"],
            evaluation.summary["cluster_evaluation_task_count"],
        )

    logger.info(
        "Completed full-dataset run cached_expressions=%s",
        getattr(encoder, "cached_expression_count", "unknown"),
    )
    _flush_logger(logger)
    manifest = {
        "created_at_local": datetime.now().astimezone().isoformat(),
        "scope": "full_input_datasets",
        "sampling_used": False,
        "python": sys.version,
        "platform": platform.platform(),
        "experiment_profile": values["project"]["experiment_profile"],
        "inputs": {
            "representative_attributes": {
                "path": str(representative_path),
                "sha256": sha256_file(representative_path),
                "rows": len(inputs.representative),
                "unique_reviews": int(inputs.representative["review_idx"].nunique()),
            },
            "opinion_units": {
                "path": str(opinion_path),
                "sha256": sha256_file(opinion_path),
                "rows": len(inputs.opinion),
                "eligible_rows": len(experiments.eligible_opinion),
                "excluded_rows": len(inputs.opinion) - len(experiments.eligible_opinion),
                "unique_reviews": int(inputs.opinion["review_idx"].nunique()),
            },
        },
        "embedding": {
            "model_id": getattr(encoder, "model_id", values["embedding"]["model_id"]),
            "local_model_path": str(
                getattr(
                    encoder,
                    "model_path",
                    config.resolve(values["embedding"]["local_model_path"]),
                )
            ),
            "device": getattr(encoder, "device", "test"),
            "batch_size": getattr(encoder, "batch_size", values["embedding"]["batch_size"]),
            "cached_expression_count": getattr(encoder, "cached_expression_count", None),
        },
        "normalization": {
            **normalization_metadata,
            "mapping_distance_metric": "cosine",
            "lineage_result_tables": ["experiment_b", "experiment_d"],
            "baseline_result_tables": ["experiment_a", "experiment_c"],
        },
        "automatic_evaluation": evaluation_summary,
        "package_versions": _package_versions(),
        "artifacts": artifact_inventory(run_dir),
    }
    atomic_write_json(manifest, run_dir / "run_manifest.json")
    return run_dir
