from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml
from conftest import StaticEncoder

from embedding_clustering_experiment.config import load_config
from embedding_clustering_experiment.runner import run


def test_runner_serializes_and_rereads_complete_fixture(
    tmp_path: Path,
    representative_frame: pd.DataFrame,
    opinion_frame: pd.DataFrame,
    static_encoder: StaticEncoder,
    config_values: dict,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    representative_frame.to_parquet(dataset_dir / "representative.parquet", index=False)
    opinion_frame.to_parquet(dataset_dir / "opinion.parquet", index=False)

    values = deepcopy(config_values)
    values["inputs"] = {
        "representative_attributes": "dataset/representative.parquet",
        "opinion_units": "dataset/opinion.parquet",
    }
    values["project"]["results_root"] = "results"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(values, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    run_dir = run(
        load_config(config_path),
        encoder_factory=lambda _config, _base_dir: static_encoder,
    )

    assert (run_dir / "results" / "experiment_a.parquet").exists()
    assert (run_dir / "results" / "experiment_d.parquet").exists()
    assert (run_dir / "evaluation" / "data_integrity_checks.parquet").exists()
    experiment_b = pd.read_parquet(run_dir / "results" / "experiment_b.parquet")
    experiment_d = pd.read_parquet(run_dir / "results" / "experiment_d.parquet")
    b_nodes = pd.read_parquet(run_dir / "clustering" / "experiment_b_nodes.parquet")
    assert {"mapping_applied", "mapping_distance"}.issubset(b_nodes.columns)
    assert experiment_b["normalization_run_id"].eq(run_dir.name).all()
    assert experiment_d["normalization_run_id"].eq(run_dir.name).all()
    review_tasks = pd.read_parquet(
        run_dir / "evaluation" / "user_evaluation_tasks.parquet"
    )
    cluster_tasks = pd.read_parquet(
        run_dir / "evaluation" / "cluster_evaluation_tasks.parquet"
    )
    assert len(review_tasks) == 2
    assert len(cluster_tasks) == 2
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["scope"] == "full_input_datasets"
    assert manifest["sampling_used"] is False
    assert manifest["normalization"]["normalization_run_id"] == run_dir.name
    assert manifest["normalization"]["normalization_version"] == "2026-08-03"
    assert manifest["normalization"]["mapping_distance_metric"] == "cosine"
    assert experiment_b["normalization_config_sha256"].iloc[0] == manifest["normalization"][
        "normalization_config_sha256"
    ]
    assert experiment_d["normalization_config_sha256"].iloc[0] == manifest["normalization"][
        "normalization_config_sha256"
    ]
    assert manifest["inputs"]["representative_attributes"]["rows"] == 4
    assert (
        manifest["automatic_evaluation"]["integrity_checks_passed"]
        == manifest["automatic_evaluation"]["integrity_check_count"]
    )
    assert manifest["automatic_evaluation"]["user_evaluation_task_count"] == 2
    assert manifest["automatic_evaluation"]["cluster_evaluation_task_count"] == 2
