"""Deterministic, evidence-carrying reports for an AI shopping agent."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import sha256_file
from .data import (
    OPINION_COLUMNS,
    SENTIMENT_VALUES,
    assert_source_columns_preserved,
    validate_opinion_input,
)
from .human_evaluation import analyze_human_evaluation

SENTIMENT_ORDER = ("positive", "negative", "mixed", "neutral", "unknown")
WILSON_Z_95 = 1.959963984540054
REPORT_SCHEMA_VERSION = "1.2.0"

# The related-product profile follows the documented MVP: all canonical aspects,
# four sentiment channels, category-prior smoothing, and an overlap penalty.
RELATED_PRODUCT_SCHEMA_VERSION = "1.0.1"
RELATED_PRODUCT_LIMIT = 3
RELATED_PRODUCT_PRIOR_STRENGTH = 5.0
RELATED_PRODUCT_EXPERIENCE_WEIGHT = 0.25
RELATED_PRODUCT_REPAIR_UTILITY_WEIGHT = 0.75
RELATED_PRODUCT_MINIMUM_WEAKNESS_SUPPORT = 2
RELATED_PRODUCT_SENTIMENTS = ("positive", "negative", "mixed", "neutral")
MOST_DEBATED_EVIDENCE_PER_SENTIMENT = 3

REQUIRED_D_COLUMNS = {
    *OPINION_COLUMNS,
    "aspect",
    "status",
    "aspect_cluster_id",
    "status_cluster_id",
    "aspect_naming_status",
    "status_naming_status",
    "aspect_mapping_applied",
    "status_mapping_applied",
    "aspect_mapping_distance",
    "status_mapping_distance",
    "embedding_model_id",
    "normalization_version",
    "normalization_run_id",
    "normalization_config_sha256",
}


@dataclass(frozen=True)
class ReportSettings:
    """Transparent thresholds shared by static and dynamic artifacts."""

    evidence_per_sentiment: int = 2
    minimum_support_reviews: int = 3
    status_near_distance: float = 0.20
    recommend_score_threshold: float = 0.15
    compare_margin: float = 0.05
    minimum_requirement_coverage: float = 0.50
    allow_near_status_match: bool = True
    decimal_places: int = 6

    def __post_init__(self) -> None:
        if self.evidence_per_sentiment < 1:
            raise ValueError("evidence_per_sentiment must be at least 1.")
        if self.minimum_support_reviews < 1:
            raise ValueError("minimum_support_reviews must be at least 1.")
        if not 0.0 < self.status_near_distance <= 2.0:
            raise ValueError("status_near_distance must be in (0, 2].")
        if not 0.0 <= self.recommend_score_threshold <= 1.0:
            raise ValueError("recommend_score_threshold must be in [0, 1].")
        if not 0.0 <= self.compare_margin <= 2.0:
            raise ValueError("compare_margin must be in [0, 2].")
        if not 0.0 <= self.minimum_requirement_coverage <= 1.0:
            raise ValueError("minimum_requirement_coverage must be in [0, 1].")
        if not isinstance(self.allow_near_status_match, bool):
            raise TypeError("allow_near_status_match must be a bool.")
        if self.decimal_places < 0:
            raise ValueError("decimal_places must be non-negative.")


@dataclass(frozen=True)
class ReportInputs:
    """Validated report inputs loaded from one immutable experiment run."""

    run_dir: Path
    manifest: dict[str, Any]
    evaluation_summary: dict[str, Any]
    human_evaluation: dict[str, Any]
    reviews: pd.DataFrame
    raw_opinion: pd.DataFrame
    experiment_d: pd.DataFrame
    joined: pd.DataFrame
    integrity: pd.DataFrame
    risky_clusters: pd.DataFrame
    status_clusters: pd.DataFrame
    reviews_path: Path
    raw_opinion_path: Path


def _required_path(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _validate_file_identity(path: Path, record: Any, label: str) -> None:
    """Fail closed when a selected file differs from its manifest record."""
    if not isinstance(record, dict):
        raise TypeError(f"run_manifest.json has no identity record for {label}.")
    expected_sha256 = str(record.get("sha256", ""))
    if len(expected_sha256) != 64:
        raise ValueError(f"run_manifest.json has no valid SHA-256 for {label}.")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 differs from run_manifest.json: "
            f"expected={expected_sha256}, actual={actual_sha256}."
        )
    expected_bytes = record.get("bytes")
    if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
        raise ValueError(f"{label} byte size differs from run_manifest.json.")


def _normalize_reviews(frame: pd.DataFrame) -> pd.DataFrame:
    if "idx" not in frame or "review" not in frame:
        raise ValueError("reviews must contain idx and review columns.")
    product_columns = [column for column in ("productName", "product_name") if column in frame]
    if len(product_columns) != 1:
        raise ValueError("reviews must contain exactly one of productName or product_name.")
    normalized = frame[["idx", "review", product_columns[0]]].rename(
        columns={"idx": "review_idx", product_columns[0]: "product_name"}
    )
    if normalized["review_idx"].isna().any() or not normalized["review_idx"].is_unique:
        raise ValueError("reviews.idx must be unique and non-null.")
    for column in ("review", "product_name"):
        if normalized[column].isna().any():
            raise ValueError(f"reviews.{column} contains null values.")
        if normalized[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"reviews.{column} contains empty strings.")
    return normalized.reset_index(drop=True)


def _validate_report_frames(
    run_dir: Path,
    manifest: dict[str, Any],
    reviews: pd.DataFrame,
    raw_opinion: pd.DataFrame,
    experiment_d: pd.DataFrame,
    joined: pd.DataFrame,
    integrity: pd.DataFrame,
    status_clusters: pd.DataFrame,
) -> None:
    missing = sorted(REQUIRED_D_COLUMNS - set(experiment_d.columns))
    if missing:
        raise ValueError(f"experiment_d is missing report columns: {missing}")
    if experiment_d.empty:
        raise ValueError("experiment_d has no rows.")
    if experiment_d["idx"].isna().any() or not experiment_d["idx"].is_unique:
        raise ValueError("experiment_d.idx must be unique and non-null.")
    invalid_sentiments = sorted(set(experiment_d["sentiment"].astype(str)) - SENTIMENT_VALUES)
    if invalid_sentiments:
        raise ValueError(f"experiment_d has invalid sentiments: {invalid_sentiments}")
    aspect_label_counts = experiment_d.groupby("aspect_cluster_id", dropna=False)["aspect"].nunique(
        dropna=False
    )
    if not aspect_label_counts.eq(1).all():
        raise ValueError("An aspect cluster ID resolves to multiple canonical labels.")
    status_rows = experiment_d.loc[experiment_d["status_cluster_id"].notna()]
    status_label_counts = status_rows.groupby("status_cluster_id", dropna=False)[
        ["aspect_cluster_id", "status"]
    ].nunique(dropna=False)
    if not status_label_counts.eq(1).all().all():
        raise ValueError("A status cluster ID resolves to multiple aspects or labels.")
    if joined["product_name"].isna().any():
        missing_review_ids = sorted(
            map(int, joined.loc[joined["product_name"].isna(), "review_idx"].unique())
        )
        raise ValueError(
            f"experiment_d has review_idx values absent from reviews: {missing_review_ids}"
        )
    if len(joined) != len(experiment_d):
        raise AssertionError("The review join changed the experiment_d row count.")

    expected_opinion_columns = set(OPINION_COLUMNS)
    missing_opinion = sorted(expected_opinion_columns - set(raw_opinion.columns))
    if missing_opinion:
        raise ValueError(f"raw opinion input is missing columns: {missing_opinion}")
    raw_review_ids = set(raw_opinion["review_idx"])
    review_ids = set(reviews["review_idx"])
    if not raw_review_ids <= review_ids:
        raise ValueError("raw opinion input contains review_idx values absent from reviews.")
    assert_source_columns_preserved(raw_opinion, experiment_d, OPINION_COLUMNS)

    bad_evidence = joined.loc[
        [
            str(excerpt) not in str(review)
            for excerpt, review in zip(joined["excerpt"], joined["review"], strict=True)
        ]
    ]
    if not bad_evidence.empty:
        raise ValueError(
            "experiment_d contains excerpts that are not contiguous review text; "
            f"first idx={int(bad_evidence.iloc[0]['idx'])}."
        )

    normalization = manifest.get("normalization", {})
    expected_run_id = str(normalization.get("normalization_run_id", ""))
    result_run_ids = set(experiment_d["normalization_run_id"].astype(str))
    if result_run_ids != {expected_run_id}:
        raise ValueError("experiment_d normalization_run_id differs from run_manifest.json.")
    if run_dir.name != expected_run_id:
        raise ValueError("run directory name differs from the manifest normalization_run_id.")
    expected_hash = str(normalization.get("normalization_config_sha256", ""))
    result_hashes = set(experiment_d["normalization_config_sha256"].astype(str))
    if result_hashes != {expected_hash}:
        raise ValueError("experiment_d normalization_config_sha256 differs from run_manifest.json.")
    if not expected_hash or len(expected_hash) != 64:
        raise ValueError("run_manifest.json has no valid normalization config hash.")

    required_integrity = {"check", "passed", "details"}
    if not required_integrity <= set(integrity.columns):
        raise ValueError("data_integrity_checks.parquet has an invalid schema.")
    if integrity.empty or not integrity["passed"].astype(bool).all():
        raise ValueError("The selected run did not pass every automatic integrity check.")

    required_status_clusters = {
        "cluster_id",
        "aspect_cluster_id",
        "canonical_label",
        "centroid_embedding",
    }
    if not required_status_clusters <= set(status_clusters.columns):
        raise ValueError("experiment_d_status_clusters.parquet has an invalid schema.")
    if not status_clusters["cluster_id"].is_unique:
        raise ValueError("D status cluster IDs must be unique.")
    if not set(status_rows["status_cluster_id"].astype(str)) <= set(
        status_clusters["cluster_id"].astype(str)
    ):
        raise ValueError("experiment_d references a status cluster absent from its cluster table.")


def load_report_inputs(
    run_dir: Path,
    reviews_path: Path,
    raw_opinion_path: Path,
    human_results_dir: Path | None = None,
) -> ReportInputs:
    """Load and validate all source tables needed by both report types."""
    selected_run = _required_path(run_dir, "run directory")
    selected_reviews = _required_path(reviews_path, "reviews input")
    selected_opinion = _required_path(raw_opinion_path, "opinion input")
    required_files = {
        "run manifest": selected_run / "run_manifest.json",
        "evaluation summary": selected_run / "evaluation/automatic_evaluation_summary.json",
        "experiment D": selected_run / "results/experiment_d.parquet",
        "integrity checks": selected_run / "evaluation/data_integrity_checks.parquet",
        "risky clusters": selected_run / "evaluation/risky_clusters.parquet",
        "review evaluation tasks": selected_run / "evaluation/user_evaluation_tasks.parquet",
        "cluster evaluation tasks": selected_run / "evaluation/cluster_evaluation_tasks.parquet",
        "B attribute clusters": selected_run / "clustering/experiment_b_clusters.parquet",
        "D aspect clusters": selected_run / "clustering/experiment_d_aspect_clusters.parquet",
        "status clusters": selected_run / "clustering/experiment_d_status_clusters.parquet",
    }
    for label, path in required_files.items():
        _required_path(path, label)

    manifest = json.loads(required_files["run manifest"].read_text(encoding="utf-8"))
    artifact_inventory = manifest.get("artifacts", {})
    for label, path in required_files.items():
        if label == "run manifest":
            continue
        relative_path = path.relative_to(selected_run).as_posix()
        _validate_file_identity(path, artifact_inventory.get(relative_path), label)
    opinion_record = manifest.get("inputs", {}).get("opinion_units")
    _validate_file_identity(selected_opinion, opinion_record, "opinion input")
    summary = json.loads(required_files["evaluation summary"].read_text(encoding="utf-8"))
    reviews = _normalize_reviews(pd.read_parquet(selected_reviews))
    raw_opinion = pd.read_parquet(selected_opinion)
    validate_opinion_input(raw_opinion)
    if len(raw_opinion) != int(opinion_record.get("rows", -1)):
        raise ValueError("opinion input row count differs from run_manifest.json.")
    if raw_opinion["review_idx"].nunique() != int(opinion_record.get("unique_reviews", -1)):
        raise ValueError("opinion input review count differs from run_manifest.json.")
    experiment_d = pd.read_parquet(required_files["experiment D"])
    if len(experiment_d) != int(opinion_record.get("eligible_rows", -1)):
        raise ValueError("experiment_d row count differs from the manifest eligible row count.")
    joined = experiment_d.merge(
        reviews,
        on="review_idx",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    integrity = pd.read_parquet(required_files["integrity checks"])
    risky = pd.read_parquet(required_files["risky clusters"])
    review_tasks = pd.read_parquet(required_files["review evaluation tasks"])
    cluster_tasks = pd.read_parquet(required_files["cluster evaluation tasks"])
    b_clusters = pd.read_parquet(required_files["B attribute clusters"])
    d_aspect_clusters = pd.read_parquet(required_files["D aspect clusters"])
    status_clusters = pd.read_parquet(required_files["status clusters"])
    _validate_report_frames(
        selected_run,
        manifest,
        reviews,
        raw_opinion,
        experiment_d,
        joined,
        integrity,
        status_clusters,
    )
    human_evaluation = analyze_human_evaluation(
        selected_run,
        human_results_dir,
        review_tasks=review_tasks,
        cluster_tasks=cluster_tasks,
        risky_clusters=risky,
        cluster_tables={
            "experiment_b_attribute": b_clusters,
            "experiment_d_aspect": d_aspect_clusters,
            "experiment_d_status": status_clusters,
        },
        random_seed=int(manifest["automatic_evaluation"]["user_evaluation_random_seed"]),
    )
    return ReportInputs(
        run_dir=selected_run,
        manifest=manifest,
        evaluation_summary=summary,
        human_evaluation=human_evaluation,
        reviews=reviews,
        raw_opinion=raw_opinion,
        experiment_d=experiment_d,
        joined=joined,
        integrity=integrity,
        risky_clusters=risky,
        status_clusters=status_clusters,
        reviews_path=selected_reviews,
        raw_opinion_path=selected_opinion,
    )


def round_float(value: Any, decimal_places: int = 6) -> float | None:
    """Round JSON floats while normalizing negative zero."""
    if value is None or pd.isna(value):
        return None
    rounded = round(float(value), decimal_places)
    return 0.0 if rounded == 0 else rounded


def collapse_sentiments(values: pd.Series) -> str:
    """Collapse one review-level grain to exactly one conservative vote."""
    known = set(map(str, values)) - {"unknown"}
    if not known:
        return "unknown"
    if len(known) == 1:
        return next(iter(known))
    return "mixed"


def wilson_interval(
    successes: int,
    total: int,
    *,
    decimal_places: int = 6,
) -> dict[str, float | None]:
    """Return a two-sided 95% Wilson score interval for a binomial share."""
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Wilson interval inputs must satisfy 0 <= successes <= total.")
    if total == 0:
        return {"lower": None, "upper": None}
    probability = successes / total
    denominator = 1 + WILSON_Z_95**2 / total
    center = (probability + WILSON_Z_95**2 / (2 * total)) / denominator
    margin = (
        WILSON_Z_95
        * math.sqrt(probability * (1 - probability) / total + WILSON_Z_95**2 / (4 * total**2))
        / denominator
    )
    return {
        "lower": round_float(max(0.0, center - margin), decimal_places),
        "upper": round_float(min(1.0, center + margin), decimal_places),
    }


def sentiment_distribution(
    counts: dict[str, int],
    *,
    decimal_places: int = 6,
) -> dict[str, Any]:
    """Build a five-class distribution with uncertainty and disagreement."""
    normalized_counts = {name: int(counts.get(name, 0)) for name in SENTIMENT_ORDER}
    total = sum(normalized_counts.values())
    shares = {
        name: round_float(normalized_counts[name] / total, decimal_places) if total else None
        for name in SENTIMENT_ORDER
    }
    probabilities = [value / total for value in normalized_counts.values() if value and total]
    entropy = (
        -sum(probability * math.log2(probability) for probability in probabilities)
        / math.log2(len(SENTIMENT_ORDER))
        if total
        else None
    )
    dominant = (
        max(
            SENTIMENT_ORDER,
            key=lambda name: (normalized_counts[name], -SENTIMENT_ORDER.index(name)),
        )
        if total
        else None
    )
    return {
        "counts": normalized_counts,
        "shares": shares,
        "dominant_sentiment": dominant,
        "dominant_share": shares[dominant] if dominant else None,
        "positive_wilson_95": wilson_interval(
            normalized_counts["positive"], total, decimal_places=decimal_places
        ),
        "negative_wilson_95": wilson_interval(
            normalized_counts["negative"], total, decimal_places=decimal_places
        ),
        "normalized_entropy": round_float(entropy, decimal_places),
    }


def mapping_stats(
    frame: pd.DataFrame,
    applied_column: str,
    distance_column: str,
    *,
    decimal_places: int = 6,
) -> dict[str, Any]:
    """Summarize row-level normalization application and mapped distances."""
    present = frame.loc[frame[applied_column].notna()]
    mapped = present.loc[present[applied_column].astype(bool), distance_column].astype(float)
    return {
        "eligible_opinion_unit_count": len(present),
        "mapped_opinion_unit_count": len(mapped),
        "mapping_rate": (
            round_float(len(mapped) / len(present), decimal_places) if len(present) else None
        ),
        "mapped_distance": {
            "p50": round_float(mapped.quantile(0.50), decimal_places) if len(mapped) else None,
            "p95": round_float(mapped.quantile(0.95), decimal_places) if len(mapped) else None,
            "max": round_float(mapped.max(), decimal_places) if len(mapped) else None,
        },
    }


def build_sentiment_votes(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Deduplicate verbose reviews at a declared product-review grain."""
    return (
        frame.groupby(
            ["product_name", "review_idx", *keys],
            observed=True,
            dropna=False,
            sort=True,
        )["sentiment"]
        .agg(collapse_sentiments)
        .rename("vote")
        .reset_index()
    )


def attach_review_votes(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Annotate every Opinion Unit with the conservative vote of its review grain."""
    annotated = frame.copy()
    annotated["review_vote_sentiment"] = annotated.groupby(
        ["product_name", "review_idx", *keys],
        observed=True,
        dropna=False,
        sort=True,
    )["sentiment"].transform(collapse_sentiments)
    return annotated


def select_evidence(
    frame: pd.DataFrame,
    sentiment: str,
    *,
    limit: int,
    vote_column: str = "sentiment",
    decimal_places: int = 6,
) -> list[dict[str, Any]]:
    """Select distinct-review evidence with the least normalization movement first."""
    if vote_column not in frame:
        raise ValueError(f"Evidence vote column is absent: {vote_column}")
    candidates = frame.loc[frame[vote_column].eq(sentiment)].copy()
    if candidates.empty:
        return []
    candidates["_sentiment_priority"] = ~candidates["sentiment"].eq(sentiment)
    candidates["_mapping_distance"] = candidates["aspect_mapping_distance"].fillna(
        0.0
    ) + candidates["status_mapping_distance"].fillna(0.0)
    candidates = (
        candidates.sort_values(["_sentiment_priority", "_mapping_distance", "idx"], kind="stable")
        .drop_duplicates("review_idx", keep="first")
        .head(limit)
    )
    rows: list[dict[str, Any]] = []
    for row in candidates.itertuples(index=False):
        if str(row.excerpt) not in str(row.review):
            raise AssertionError(f"excerpt is not contiguous review text: idx={int(row.idx)}")
        rows.append(
            {
                "opinion_unit_idx": int(row.idx),
                "review_idx": int(row.review_idx),
                "excerpt": str(row.excerpt),
                "opinion": str(row.opinion),
                "sentiment": str(row.sentiment),
                "review_vote_sentiment": str(getattr(row, vote_column)),
                "raw_aspect": str(row.raw_aspect),
                "raw_status": None if pd.isna(row.raw_status) else str(row.raw_status),
                "aspect": str(row.aspect),
                "status": None if pd.isna(row.status) else str(row.status),
                "aspect_mapping_distance": round_float(row.aspect_mapping_distance, decimal_places),
                "status_mapping_distance": round_float(row.status_mapping_distance, decimal_places),
            }
        )
    return rows


def risk_lookup(inputs: ReportInputs, decimal_places: int = 6) -> dict[str, dict[str, Any]]:
    """Index the bounded global risky-cluster table by cluster ID."""
    return {
        str(row.cluster_id): {
            "stage": str(row.stage),
            "cluster_id": str(row.cluster_id),
            "canonical_label": (None if pd.isna(row.canonical_label) else str(row.canonical_label)),
            "risk_score": round_float(row.risk_score, decimal_places),
            "member_count": int(row.member_count),
            "cluster_max_distance": round_float(row.cluster_max_distance, decimal_places),
            "cluster_silhouette_mean": round_float(row.cluster_silhouette_mean, decimal_places),
        }
        for row in inputs.risky_clusters.itertuples(index=False)
    }


def source_metadata(inputs: ReportInputs) -> dict[str, Any]:
    """Return portable run lineage without machine-local source paths."""
    human_source = inputs.human_evaluation["source"]
    return {
        "experiment": "D",
        "run_id": inputs.run_dir.name,
        "created_at_local": inputs.manifest["created_at_local"],
        "scope": inputs.manifest["scope"],
        "sampling_used": bool(inputs.manifest["sampling_used"]),
        "run_manifest_sha256": sha256_file(inputs.run_dir / "run_manifest.json"),
        "normalization": inputs.manifest["normalization"],
        "input_sha256": {
            "reviews": sha256_file(inputs.reviews_path),
            "opinion_units": inputs.manifest["inputs"]["opinion_units"]["sha256"],
        },
        "human_evaluation": {
            "status": inputs.human_evaluation["status"],
            "completed_result_file_count": human_source["completed_result_file_count"],
            "results_set_sha256": human_source["results_set_sha256"],
            "evaluator_count": inputs.human_evaluation["validation"]["evaluator_count"],
        },
    }


def product_review_counts(inputs: ReportInputs) -> dict[str, int]:
    return {
        str(product): int(count)
        for product, count in inputs.reviews.groupby("product_name", observed=True, sort=True)[
            "review_idx"
        ]
        .nunique()
        .items()
    }


def quality_flags(
    inputs: ReportInputs,
    *,
    product_name: str | None,
    decimal_places: int = 6,
) -> list[dict[str, Any]]:
    """Expose missing decision dimensions and run-level validation gaps."""
    risks = risk_lookup(inputs, decimal_places)
    if product_name is None:
        product_risk_ids = {
            cluster_id
            for cluster_id, risk in risks.items()
            if risk["stage"] in {"experiment_d_aspect", "experiment_d_status"}
        }
        risk_scope = "catalog_experiment_d"
    else:
        product = inputs.joined.loc[inputs.joined["product_name"].eq(product_name)]
        product_risk_ids = (
            set(product["aspect_cluster_id"].dropna().astype(str))
            | set(product["status_cluster_id"].dropna().astype(str))
        ) & set(risks)
        risk_scope = "product"
    human_evaluation_incomplete = inputs.human_evaluation["status"] != "completed"
    flags = [
        {
            "code": "HUMAN_EVALUATION_INCOMPLETE",
            "severity": "warning" if human_evaluation_incomplete else "info",
            "value": human_evaluation_incomplete,
            "implication": (
                "normalization_quality_not_human_validated_for_this_run"
                if human_evaluation_incomplete
                else "completed_human_evaluation_available"
            ),
        },
        {
            "code": "REVIEW_TIMESTAMPS_UNAVAILABLE",
            "severity": "warning",
            "value": True,
            "implication": "freshness_and_temporal_drift_unavailable",
        },
        {
            "code": "COMMERCE_CONTEXT_UNAVAILABLE",
            "severity": "warning",
            "value": [
                "price",
                "objective_specs",
                "inventory",
                "shipping",
                "promotion",
            ],
            "implication": "review_experience_only_not_purchase_optimality",
        },
        {
            "code": "NORMALIZATION_RISK_CLUSTERS_PRESENT",
            "severity": "warning" if product_risk_ids else "info",
            "value": len(product_risk_ids),
            "scope": risk_scope,
            "implication": "inspect_cluster_lineage_before_high_stakes_use",
        },
    ]
    if not human_evaluation_incomplete:
        d_risk_rows = [
            row
            for row in inputs.human_evaluation["cluster_evaluation"]["risky_cluster_coverage"]
            if row["stage"] in {"experiment_d_aspect", "experiment_d_status"}
        ]
        bounded_count = sum(row["bounded_risky_cluster_count"] for row in d_risk_rows)
        evaluated_count = sum(row["evaluated_risky_cluster_count"] for row in d_risk_rows)
        coverage_rate = evaluated_count / bounded_count if bounded_count else None
        flags.extend(
            [
                {
                    "code": "HUMAN_EVALUATION_RISK_COVERAGE_LIMITED",
                    "severity": (
                        "warning" if coverage_rate is not None and coverage_rate < 0.5 else "info"
                    ),
                    "value": {
                        "scope": "experiment_d_bounded_risky_clusters",
                        "evaluated_cluster_count": evaluated_count,
                        "bounded_risky_cluster_count": bounded_count,
                        "coverage_rate": round_float(coverage_rate, decimal_places),
                    },
                    "implication": "sampled_cluster_scores_do_not_validate_all_risky_clusters",
                },
                {
                    "code": "HUMAN_EVALUATION_EVALUATOR_COUNT_AT_MINIMUM",
                    "severity": (
                        "warning"
                        if inputs.human_evaluation["validation"]["evaluator_count"]
                        == inputs.human_evaluation["validation"]["minimum_evaluator_count"]
                        else "info"
                    ),
                    "value": {
                        "evaluator_count": inputs.human_evaluation["validation"]["evaluator_count"],
                        "minimum_evaluator_count": inputs.human_evaluation["validation"][
                            "minimum_evaluator_count"
                        ],
                    },
                    "implication": "agreement_and_generalization_have_limited_precision",
                },
            ]
        )
    return flags


def _summary_rows(
    inputs: ReportInputs,
    product: pd.DataFrame,
    votes: pd.DataFrame,
    *,
    product_review_count: int,
    pair: bool,
    settings: ReportSettings,
) -> list[dict[str, Any]]:
    keys = ["aspect_cluster_id", "aspect"]
    if pair:
        keys = ["aspect_cluster_id", "status_cluster_id", "aspect", "status"]
    risks = risk_lookup(inputs, settings.decimal_places)
    rows: list[dict[str, Any]] = []
    for key, group in votes.groupby(
        keys,
        observed=True,
        dropna=False,
        sort=True,
    ):
        values = key if isinstance(key, tuple) else (key,)
        record = dict(zip(keys, values, strict=True))
        counts = {name: int(group["vote"].eq(name).sum()) for name in SENTIMENT_ORDER}
        support = len(group)
        aspect_id = str(record["aspect_cluster_id"])
        payload: dict[str, Any] = {
            "aspect_cluster_id": aspect_id,
            "aspect": str(record["aspect"]),
            "supporting_review_count": support,
            "mention_rate": round_float(support / product_review_count, settings.decimal_places),
            "sentiment": sentiment_distribution(counts, decimal_places=settings.decimal_places),
            "aspect_mapping": mapping_stats(
                product.loc[product["aspect_cluster_id"].eq(record["aspect_cluster_id"])],
                "aspect_mapping_applied",
                "aspect_mapping_distance",
                decimal_places=settings.decimal_places,
            ),
            "normalization_risk": {"aspect": risks.get(aspect_id)},
        }
        if pair:
            status_id = (
                None if pd.isna(record["status_cluster_id"]) else str(record["status_cluster_id"])
            )
            status_mask = (
                product["status_cluster_id"].isna()
                if pd.isna(record["status_cluster_id"])
                else product["status_cluster_id"].eq(record["status_cluster_id"])
            )
            pair_frame = product.loc[
                product["aspect_cluster_id"].eq(record["aspect_cluster_id"]) & status_mask
            ]
            pair_frame = attach_review_votes(
                pair_frame,
                ["aspect_cluster_id", "status_cluster_id"],
            )
            payload.update(
                {
                    "status_cluster_id": status_id,
                    "status": None if pd.isna(record["status"]) else str(record["status"]),
                    "status_mapping": mapping_stats(
                        pair_frame,
                        "status_mapping_applied",
                        "status_mapping_distance",
                        decimal_places=settings.decimal_places,
                    ),
                    "normalization_risk": {
                        "aspect": risks.get(aspect_id),
                        "status": risks.get(status_id) if status_id else None,
                    },
                    "evidence": {
                        name: select_evidence(
                            pair_frame,
                            name,
                            limit=settings.evidence_per_sentiment,
                            vote_column="review_vote_sentiment",
                            decimal_places=settings.decimal_places,
                        )
                        for name in SENTIMENT_ORDER
                    },
                }
            )
        else:
            aspect_frame = product.loc[product["aspect_cluster_id"].eq(record["aspect_cluster_id"])]
            aspect_frame = attach_review_votes(aspect_frame, ["aspect_cluster_id"])
            payload["evidence"] = {
                name: select_evidence(
                    aspect_frame,
                    name,
                    limit=MOST_DEBATED_EVIDENCE_PER_SENTIMENT,
                    vote_column="review_vote_sentiment",
                    decimal_places=settings.decimal_places,
                )
                for name in SENTIMENT_ORDER
            }
        rows.append(payload)
    rows.sort(
        key=lambda row: (
            -row["supporting_review_count"],
            row["aspect"],
            str(row.get("status") or ""),
            row["aspect_cluster_id"],
            str(row.get("status_cluster_id") or ""),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _reduction_rate(raw_count: int, normalized_count: int, *, decimal_places: int) -> float | None:
    if raw_count == 0:
        return None
    return round_float((raw_count - normalized_count) / raw_count, decimal_places)


def _normalization_reduction_summary(
    product: pd.DataFrame,
    *,
    decimal_places: int,
) -> dict[str, Any]:
    """Compare raw and final D labels at their actual normalization grains."""
    raw_status_scope = product.loc[
        product["raw_status"].notna(), ["aspect_cluster_id", "raw_status"]
    ]
    raw_pair = product[["raw_aspect", "raw_status"]].copy()
    raw_pair["raw_status"] = raw_pair["raw_status"].fillna("__NULL_STATUS__")
    normalized_pair = product[["aspect_cluster_id", "status_cluster_id"]].copy()
    normalized_pair["status_cluster_id"] = normalized_pair["status_cluster_id"].fillna(
        "__NULL_STATUS__"
    )
    counts = {
        "aspect": {
            "raw_count": int(product["raw_aspect"].nunique()),
            "normalized_count": int(product["aspect_cluster_id"].nunique()),
            "comparison_grain": "unique_raw_aspect_to_unique_aspect_cluster_id",
        },
        "status": {
            "raw_count": int(raw_status_scope.drop_duplicates().shape[0]),
            "normalized_count": int(product["status_cluster_id"].dropna().nunique()),
            "comparison_grain": (
                "unique_aspect_cluster_id_raw_status_to_unique_status_cluster_id; "
                "null_status_excluded"
            ),
        },
        "aspect_status": {
            "raw_count": int(raw_pair.drop_duplicates().shape[0]),
            "normalized_count": int(normalized_pair.drop_duplicates().shape[0]),
            "comparison_grain": "unique_raw_aspect_raw_status_to_unique_aspect_cluster_id_status_cluster_id",
        },
    }
    for item in counts.values():
        item["decrease_rate"] = _reduction_rate(
            item["raw_count"],
            item["normalized_count"],
            decimal_places=decimal_places,
        )
    return counts


def _most_debated_aspect(
    aspect_rows: list[dict[str, Any]],
    *,
    top_limit: int,
    review_text_by_id: dict[int, str],
) -> dict[str, Any] | None:
    """Select the top-N aspect with the smallest positive-vs-negative share gap."""
    candidates = [
        row
        for row in aspect_rows[:top_limit]
        if row["sentiment"]["counts"]["positive"] + row["sentiment"]["counts"]["negative"] > 0
    ]
    if not candidates:
        return None
    selected = min(
        candidates,
        key=lambda row: (
            abs(
                float(row["sentiment"]["shares"]["positive"])
                - float(row["sentiment"]["shares"]["negative"])
            ),
            -row["supporting_review_count"],
            row["aspect"],
            row["aspect_cluster_id"],
        ),
    )
    result = deepcopy(selected)
    for samples in result["evidence"].values():
        for sample in samples:
            sample["review_text"] = review_text_by_id[int(sample["review_idx"])]
    result["positive_negative_rate_gap"] = round_float(
        abs(
            float(selected["sentiment"]["shares"]["positive"])
            - float(selected["sentiment"]["shares"]["negative"])
        ),
        6,
    )
    result["selection_scope"] = f"top_{top_limit}_aspect_summary"
    return result


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _related_product_profile_data(inputs: ReportInputs) -> dict[str, Any]:
    """Build all-product profiles at the documented product-review-aspect grain."""
    votes = build_sentiment_votes(inputs.joined, ["aspect_cluster_id", "aspect"])
    pair_votes = build_sentiment_votes(
        inputs.joined,
        ["aspect_cluster_id", "status_cluster_id", "aspect", "status"],
    )
    review_counts = product_review_counts(inputs)
    aspect_counts: dict[tuple[str, str], dict[str, Any]] = {}
    aspect_labels: dict[str, str] = {}
    for key, group in votes.groupby(
        ["product_name", "aspect_cluster_id", "aspect"],
        observed=True,
        sort=True,
    ):
        product_name, aspect_id, aspect = key
        count_by_sentiment = {
            sentiment: int(group["vote"].eq(sentiment).sum()) for sentiment in SENTIMENT_ORDER
        }
        aspect_id_text = str(aspect_id)
        aspect_counts[(str(product_name), aspect_id_text)] = {
            "supporting_review_count": len(group),
            "counts": count_by_sentiment,
        }
        aspect_labels[aspect_id_text] = str(aspect)

    aspect_ids = sorted(aspect_labels)
    category_priors: dict[str, dict[str, float]] = {}
    for aspect_id in aspect_ids:
        totals = {sentiment: 0 for sentiment in RELATED_PRODUCT_SENTIMENTS}
        for (_product_name, candidate_aspect_id), record in aspect_counts.items():
            if candidate_aspect_id == aspect_id:
                for sentiment in RELATED_PRODUCT_SENTIMENTS:
                    totals[sentiment] += record["counts"][sentiment]
        known_total = sum(totals.values())
        category_priors[aspect_id] = {
            sentiment: (
                totals[sentiment] / known_total
                if known_total
                else 1.0 / len(RELATED_PRODUCT_SENTIMENTS)
            )
            for sentiment in RELATED_PRODUCT_SENTIMENTS
        }

    profiles: dict[str, dict[str, Any]] = {}
    for product_name, review_count in review_counts.items():
        mention_vector: list[float] = []
        sentiment_vector: list[float] = []
        mention_rates: dict[str, float] = {}
        reliability: dict[str, float] = {}
        for aspect_id in aspect_ids:
            record = aspect_counts.get((product_name, aspect_id))
            support = int(record["supporting_review_count"]) if record else 0
            counts = record["counts"] if record else {sentiment: 0 for sentiment in SENTIMENT_ORDER}
            mention_rate = support / review_count
            known_count = sum(counts[sentiment] for sentiment in RELATED_PRODUCT_SENTIMENTS)
            unknown_share = counts["unknown"] / support if support else 0.0
            support_reliability = support / (support + RELATED_PRODUCT_PRIOR_STRENGTH)
            smoothed_distribution = {
                sentiment: (
                    (
                        counts[sentiment]
                        + RELATED_PRODUCT_PRIOR_STRENGTH * category_priors[aspect_id][sentiment]
                    )
                    / (known_count + RELATED_PRODUCT_PRIOR_STRENGTH)
                )
                for sentiment in RELATED_PRODUCT_SENTIMENTS
            }
            scale = math.sqrt(mention_rate) * support_reliability * (1.0 - unknown_share)
            mention_vector.append(math.sqrt(mention_rate))
            sentiment_vector.extend(
                scale * smoothed_distribution[sentiment] for sentiment in RELATED_PRODUCT_SENTIMENTS
            )
            mention_rates[aspect_id] = mention_rate
            reliability[aspect_id] = support_reliability
        profiles[product_name] = {
            "mention_vector": np.asarray(mention_vector, dtype=float),
            "sentiment_vector": np.asarray(sentiment_vector, dtype=float),
            "mention_rates": mention_rates,
            "support_reliability": reliability,
        }

    pair_rates: dict[tuple[str, str, str | None], float] = {}
    for key, group in pair_votes.groupby(
        ["product_name", "aspect_cluster_id", "status_cluster_id"],
        observed=True,
        dropna=False,
        sort=True,
    ):
        product_name, aspect_id, status_id = key
        pair_rates[
            (
                str(product_name),
                str(aspect_id),
                None if pd.isna(status_id) else str(status_id),
            )
        ] = len(group) / review_counts[str(product_name)]

    risk_ids_by_product: dict[str, set[str]] = {}
    risks = risk_lookup(inputs)
    for product_name, group in inputs.joined.groupby("product_name", observed=True, sort=True):
        risk_ids_by_product[str(product_name)] = (
            set(group["aspect_cluster_id"].dropna().astype(str))
            | set(group["status_cluster_id"].dropna().astype(str))
        ) & set(risks)
    return {
        "profiles": profiles,
        "aspect_ids": aspect_ids,
        "aspect_labels": aspect_labels,
        "pair_rates": pair_rates,
        "risk_ids_by_product": risk_ids_by_product,
        "review_counts": review_counts,
    }


def _related_similarity_rows(
    profile_data: dict[str, Any],
    *,
    source_product_name: str,
    decimal_places: int,
) -> list[dict[str, Any]]:
    source = profile_data["profiles"][source_product_name]
    aspect_ids = profile_data["aspect_ids"]
    pair_rates = profile_data["pair_rates"]
    pair_keys = sorted({key[1:] for key in pair_rates})
    source_risks = profile_data["risk_ids_by_product"][source_product_name]
    rows: list[dict[str, Any]] = []
    for candidate_product_name, candidate in profile_data["profiles"].items():
        if candidate_product_name == source_product_name:
            continue
        overlap_numerator = sum(
            min(source["mention_rates"][aspect_id], candidate["mention_rates"][aspect_id])
            for aspect_id in aspect_ids
        )
        overlap_denominator = sum(
            max(source["mention_rates"][aspect_id], candidate["mention_rates"][aspect_id])
            for aspect_id in aspect_ids
        )
        evidence_overlap = overlap_numerator / overlap_denominator if overlap_denominator else 0.0
        support_reliability = (
            sum(
                min(source["mention_rates"][aspect_id], candidate["mention_rates"][aspect_id])
                * math.sqrt(
                    source["support_reliability"][aspect_id]
                    * candidate["support_reliability"][aspect_id]
                )
                for aspect_id in aspect_ids
            )
            / overlap_numerator
            if overlap_numerator
            else 0.0
        )
        sentiment_similarity = _cosine_similarity(
            source["sentiment_vector"], candidate["sentiment_vector"]
        )
        experience_similarity = sentiment_similarity * math.sqrt(evidence_overlap)
        status_overlap_numerator = sum(
            min(
                pair_rates.get((source_product_name, *key), 0.0),
                pair_rates.get((candidate_product_name, *key), 0.0),
            )
            for key in pair_keys
        )
        status_overlap_denominator = sum(
            max(
                pair_rates.get((source_product_name, *key), 0.0),
                pair_rates.get((candidate_product_name, *key), 0.0),
            )
            for key in pair_keys
        )
        aspect_status_exact_overlap = (
            status_overlap_numerator / status_overlap_denominator
            if status_overlap_denominator
            else 0.0
        )
        shared_aspects = [
            {
                "aspect_cluster_id": aspect_id,
                "aspect": profile_data["aspect_labels"][aspect_id],
                "source_mention_rate": round_float(
                    source["mention_rates"][aspect_id], decimal_places
                ),
                "candidate_mention_rate": round_float(
                    candidate["mention_rates"][aspect_id], decimal_places
                ),
                "shared_mention_weight": round_float(
                    min(source["mention_rates"][aspect_id], candidate["mention_rates"][aspect_id]),
                    decimal_places,
                ),
            }
            for aspect_id in aspect_ids
            if min(source["mention_rates"][aspect_id], candidate["mention_rates"][aspect_id]) > 0
        ]
        shared_aspects.sort(
            key=lambda row: (
                -float(row["shared_mention_weight"] or 0.0),
                row["aspect"],
                row["aspect_cluster_id"],
            )
        )
        candidate_risks = profile_data["risk_ids_by_product"][candidate_product_name]
        quality_codes = []
        if evidence_overlap < 0.2:
            quality_codes.append("LOW_EVIDENCE_OVERLAP")
        if support_reliability < 0.5:
            quality_codes.append("LOW_SHARED_SUPPORT_RELIABILITY")
        if source_risks or candidate_risks:
            quality_codes.append("NORMALIZATION_RISK_CLUSTERS_PRESENT")
        rows.append(
            {
                "product_name": candidate_product_name,
                "catalog_review_count": profile_data["review_counts"][candidate_product_name],
                "experience_similarity": round_float(experience_similarity, decimal_places),
                "components": {
                    "aspect_mention_similarity": round_float(
                        _cosine_similarity(source["mention_vector"], candidate["mention_vector"]),
                        decimal_places,
                    ),
                    "aspect_sentiment_similarity": round_float(
                        sentiment_similarity, decimal_places
                    ),
                    "evidence_overlap": round_float(evidence_overlap, decimal_places),
                    "support_reliability": round_float(support_reliability, decimal_places),
                    "aspect_status_exact_overlap": round_float(
                        aspect_status_exact_overlap, decimal_places
                    ),
                },
                "shared_aspects": shared_aspects[:5],
                "normalization_risk": {
                    "source_risky_cluster_count": len(source_risks),
                    "candidate_risky_cluster_count": len(candidate_risks),
                },
                "quality_codes": quality_codes,
            }
        )
    rows.sort(key=lambda row: (-float(row["experience_similarity"] or 0.0), row["product_name"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _source_weakness_requirements(
    pair_rows: list[dict[str, Any]],
    *,
    decimal_places: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in pair_rows:
        negative_count = int(row["sentiment"]["counts"]["negative"])
        support = int(row["supporting_review_count"])
        status_id = row["status_cluster_id"]
        if not negative_count:
            continue
        if status_id is None:
            excluded.append(
                {
                    "aspect": row["aspect"],
                    "status": row["status"],
                    "reason": "STATUS_CLUSTER_ID_UNAVAILABLE",
                    "negative_review_count": negative_count,
                }
            )
            continue
        if support < RELATED_PRODUCT_MINIMUM_WEAKNESS_SUPPORT:
            excluded.append(
                {
                    "aspect": row["aspect"],
                    "status": row["status"],
                    "reason": "SOURCE_NEGATIVE_SUPPORT_BELOW_MINIMUM",
                    "negative_review_count": negative_count,
                    "supporting_review_count": support,
                }
            )
            continue
        negative_rate = negative_count / support
        importance = negative_rate * support / (support + RELATED_PRODUCT_PRIOR_STRENGTH)
        eligible.append(
            {
                "aspect": row["aspect"],
                "status": row["status"],
                "aspect_cluster_id": row["aspect_cluster_id"],
                "status_cluster_id": status_id,
                "negative_review_count": negative_count,
                "supporting_review_count": support,
                "negative_rate": round_float(negative_rate, decimal_places),
                "importance": round_float(importance, decimal_places),
            }
        )
    eligible.sort(
        key=lambda row: (
            -float(row["importance"] or 0.0),
            -row["negative_review_count"],
            -row["supporting_review_count"],
            row["aspect"],
            str(row["status"]),
        )
    )
    selected = eligible[:10]
    for position, item in enumerate(selected, start=1):
        item["requirement_id"] = f"weakness_{position}"
    return selected, excluded


def _compact_related_profile(profile: dict[str, Any] | None) -> dict[str, Any] | None:
    if profile is None:
        return None
    return {
        "aspect": profile["aspect"],
        "status": profile["status"],
        "supporting_review_count": profile["supporting_review_count"],
        "dominant_sentiment": profile["sentiment"]["dominant_sentiment"],
        "positive_wilson_lower_95": profile["sentiment"]["positive_wilson_95"]["lower"],
    }


def _compact_related_match(match: dict[str, Any]) -> dict[str, Any]:
    return {
        "requirement_id": match["requirement_id"],
        "relation": match["relation"],
        "contribution": match["contribution"],
        "matched_status": _compact_related_profile(match["matched_status_profile"]),
        "positive_alternative": _compact_related_profile(match["positive_alternative_profile"]),
    }


def _weakness_repair_alternatives(
    inputs: ReportInputs,
    *,
    source_product_name: str,
    pair_rows: list[dict[str, Any]],
    similarity_rows: list[dict[str, Any]],
    settings: ReportSettings,
) -> dict[str, Any]:
    """Rank products that have evidence for avoiding high-support source weaknesses."""
    requirements, excluded_requirements = _source_weakness_requirements(
        pair_rows,
        decimal_places=settings.decimal_places,
    )
    contract = {
        "source_negative_support_minimum": RELATED_PRODUCT_MINIMUM_WEAKNESS_SUPPORT,
        "maximum_source_requirements": 10,
        "status_match_policy": "exact_status_cluster_id_only",
        "experience_similarity_weight": RELATED_PRODUCT_EXPERIENCE_WEIGHT,
        "weakness_utility_weight": RELATED_PRODUCT_REPAIR_UTILITY_WEIGHT,
        "repair_score": (
            "0.25 * experience_similarity + 0.75 * ((weakness_utility_score + 1) / 2)"
        ),
    }
    if not requirements:
        return {
            "status": "NO_HIGH_SUPPORT_NEGATIVE_ASPECT_STATUS_EVIDENCE",
            "ranking_contract": contract,
            "source_weakness_requirements": [],
            "excluded_source_weaknesses": excluded_requirements,
            "alternatives": [],
        }

    # Import lazily to keep reporting.py and decision.py independently importable.
    from .decision import build_dynamic_decision_proposal

    repair_settings = replace(
        settings,
        minimum_support_reviews=RELATED_PRODUCT_MINIMUM_WEAKNESS_SUPPORT,
        minimum_requirement_coverage=0.0,
        allow_near_status_match=False,
    )
    request = {
        "request_id": f"static-weakness-repair:{source_product_name}",
        "requirements": [
            {
                "requirement_id": item["requirement_id"],
                "aspect": item["aspect"],
                "status": item["status"],
                "sentiment": "negative",
                "aspect_cluster_id": item["aspect_cluster_id"],
                "status_cluster_id": item["status_cluster_id"],
                "importance": item["importance"],
            }
            for item in requirements
        ],
        "excluded_products": [source_product_name],
    }
    proposal = build_dynamic_decision_proposal(inputs, request, settings=repair_settings)
    if proposal["request_validation"]["unresolved_actionable_requirement_count"]:
        raise AssertionError(
            "Static weakness requirements must resolve to canonical aspect-status pairs."
        )
    similarity_by_product = {row["product_name"]: row for row in similarity_rows}
    alternatives: list[dict[str, Any]] = []
    for candidate in proposal["candidates"]:
        evidence_coverage = float(candidate["evidence_coverage_rate"])
        if evidence_coverage == 0.0:
            continue
        weakness_utility_score = float(candidate["score"])
        normalized_utility = (weakness_utility_score + 1.0) / 2.0
        similarity = similarity_by_product[candidate["product_name"]]
        repair_score = (
            RELATED_PRODUCT_EXPERIENCE_WEIGHT * float(similarity["experience_similarity"])
            + RELATED_PRODUCT_REPAIR_UTILITY_WEIGHT * normalized_utility
        )
        alternatives.append(
            {
                "product_name": candidate["product_name"],
                "catalog_review_count": candidate["catalog_review_count"],
                "weakness_repair_score": round_float(repair_score, settings.decimal_places),
                "weakness_utility_score": round_float(
                    weakness_utility_score, settings.decimal_places
                ),
                "experience_similarity": similarity["experience_similarity"],
                "evidence_coverage_rate": candidate["evidence_coverage_rate"],
                "supported_requirement_count": candidate["supported_requirement_count"],
                "requirement_matches": [
                    _compact_related_match(match) for match in candidate["requirement_matches"]
                ],
                "quality_codes": similarity["quality_codes"],
            }
        )
    alternatives.sort(
        key=lambda row: (
            -float(row["weakness_repair_score"] or 0.0),
            -float(row["weakness_utility_score"] or 0.0),
            -float(row["experience_similarity"] or 0.0),
            row["product_name"],
        )
    )
    for rank, row in enumerate(alternatives, start=1):
        row["rank"] = rank
    return {
        "status": "COMPLETED" if alternatives else "NO_CANDIDATE_WITH_REPAIR_EVIDENCE",
        "ranking_contract": contract,
        "source_weakness_requirements": requirements,
        "excluded_source_weaknesses": excluded_requirements,
        "alternatives": alternatives[:RELATED_PRODUCT_LIMIT],
    }


def build_related_products(
    inputs: ReportInputs,
    *,
    source_product_name: str,
    pair_rows: list[dict[str, Any]],
    settings: ReportSettings,
) -> dict[str, Any]:
    """Return experience-similar and source-weakness-repair product candidates."""
    profile_data = _related_product_profile_data(inputs)
    similarity_rows = _related_similarity_rows(
        profile_data,
        source_product_name=source_product_name,
        decimal_places=settings.decimal_places,
    )
    return {
        "schema_version": RELATED_PRODUCT_SCHEMA_VERSION,
        "source_product_name": source_product_name,
        "candidate_product_count": len(similarity_rows),
        "similarity_contract": {
            "feature_scope": "all_canonical_aspect_cluster_ids_not_display_top_n",
            "vote_grain": ["product_name", "review_idx", "aspect_cluster_id"],
            "sentiment_channels": list(RELATED_PRODUCT_SENTIMENTS),
            "unknown_policy": "excluded_from_sentiment_distribution_and_quality_penalized",
            "category_prior_strength": RELATED_PRODUCT_PRIOR_STRENGTH,
            "idf_weight": 1.0,
            "profile_formula": (
                "sqrt(mention_rate) * support/(support+5) * (1-unknown_share) "
                "* smoothed_sentiment_probability"
            ),
            "experience_similarity": (
                "cosine(smoothed_multi_channel_aspect_profile) * sqrt(weighted_aspect_overlap)"
            ),
            "evidence_overlap": "sum(min(aspect_mention_rate)) / sum(max(aspect_mention_rate))",
            "aspect_status_policy": "exact_aspect_cluster_id_status_cluster_id_overlap_reported_not_scored",
        },
        "similar_products": similarity_rows[:RELATED_PRODUCT_LIMIT],
        "weakness_repair_alternatives": _weakness_repair_alternatives(
            inputs,
            source_product_name=source_product_name,
            pair_rows=pair_rows,
            similarity_rows=similarity_rows,
            settings=settings,
        ),
    }


def build_static_catalog_report(
    inputs: ReportInputs,
    product_name: str,
    *,
    settings: ReportSettings | None = None,
) -> dict[str, Any]:
    """Build one complete, machine-readable product review profile."""
    selected_settings = settings or ReportSettings()
    counts = product_review_counts(inputs)
    if product_name not in counts:
        available = ", ".join(sorted(counts))
        raise ValueError(f"Unknown product_name {product_name!r}; available: {available}")
    product = inputs.joined.loc[inputs.joined["product_name"].eq(product_name)].copy()
    if product.empty:
        raise ValueError(f"No eligible Opinion Units are available for {product_name!r}.")
    review_count = counts[product_name]
    aspect_votes = build_sentiment_votes(
        inputs.joined,
        ["aspect_cluster_id", "aspect"],
    )
    pair_votes = build_sentiment_votes(
        inputs.joined,
        ["aspect_cluster_id", "status_cluster_id", "aspect", "status"],
    )
    product_aspect_votes = aspect_votes.loc[aspect_votes["product_name"].eq(product_name)]
    product_pair_votes = pair_votes.loc[pair_votes["product_name"].eq(product_name)]
    aspect_rows = _summary_rows(
        inputs,
        product,
        product_aspect_votes,
        product_review_count=review_count,
        pair=False,
        settings=selected_settings,
    )
    pair_rows = _summary_rows(
        inputs,
        product,
        product_pair_votes,
        product_review_count=review_count,
        pair=True,
        settings=selected_settings,
    )
    normalization_reduction = _normalization_reduction_summary(
        product,
        decimal_places=selected_settings.decimal_places,
    )
    most_debated = _most_debated_aspect(
        aspect_rows,
        top_limit=10,
        review_text_by_id={
            int(row.review_idx): str(row.review)
            for row in inputs.reviews[["review_idx", "review"]].itertuples(index=False)
        },
    )
    related_products = build_related_products(
        inputs,
        source_product_name=product_name,
        pair_rows=pair_rows,
        settings=selected_settings,
    )

    raw_joined = inputs.raw_opinion.merge(
        inputs.reviews[["review_idx", "product_name"]],
        on="review_idx",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    excluded = raw_joined.loc[
        raw_joined["product_name"].eq(product_name)
        & raw_joined["raw_aspect"].eq("전반적 상품 경험")
    ]
    unit_counts = {name: int(product["sentiment"].eq(name).sum()) for name in SENTIMENT_ORDER}
    aspect_vote_counts = {
        name: int(product_aspect_votes["vote"].eq(name).sum()) for name in SENTIMENT_ORDER
    }
    risks = risk_lookup(inputs, selected_settings.decimal_places)
    risk_ids = (
        set(product["aspect_cluster_id"].dropna().astype(str))
        | set(product["status_cluster_id"].dropna().astype(str))
    ) & set(risks)
    product_risks = [risks[cluster_id] for cluster_id in sorted(risk_ids)]
    identity_payload = {
        "product_name": product_name,
        "human_evaluation_status": inputs.human_evaluation["status"],
        "human_evaluation_results_set_sha256": inputs.human_evaluation["source"][
            "results_set_sha256"
        ],
        "settings": {
            key: value
            for key, value in asdict(selected_settings).items()
            if key in {"evidence_per_sentiment", "decimal_places"}
        },
        "related_products_contract": related_products["similarity_contract"],
        "related_products_schema_version": related_products["schema_version"],
        "weakness_repair_contract": related_products["weakness_repair_alternatives"][
            "ranking_contract"
        ],
    }
    product_digest = hashlib.sha256(
        json.dumps(
            identity_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:12]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": "static_catalog_analysis",
        "report_id": (f"static_catalog_analysis:{inputs.run_dir.name}:{product_digest}"),
        "source": source_metadata(inputs),
        "human_evaluation": inputs.human_evaluation,
        "product": {
            "product_name": product_name,
            "catalog_review_count": review_count,
            "reviews_with_eligible_opinion_units": int(product["review_idx"].nunique()),
            "review_coverage_rate": round_float(
                product["review_idx"].nunique() / review_count,
                selected_settings.decimal_places,
            ),
        },
        "coverage": {
            "opinion_unit_count": len(product),
            "review_aspect_vote_count": len(product_aspect_votes),
            "review_aspect_status_vote_count": len(product_pair_votes),
            "unique_aspect_cluster_count": int(product["aspect_cluster_id"].nunique()),
            "unique_aspect_status_cluster_count": int(
                product[["aspect_cluster_id", "status_cluster_id"]].drop_duplicates().shape[0]
            ),
            "excluded_general_experience_opinion_unit_count": len(excluded),
            "excluded_general_experience_review_count": int(excluded["review_idx"].nunique()),
            "mean_opinion_units_per_covered_review": round_float(
                len(product) / product["review_idx"].nunique(),
                selected_settings.decimal_places,
            ),
        },
        "normalization_reduction": normalization_reduction,
        "aggregation_contract": {
            "aspect_vote_grain": [
                "product_name",
                "review_idx",
                "aspect_cluster_id",
            ],
            "aspect_status_vote_grain": [
                "product_name",
                "review_idx",
                "aspect_cluster_id",
                "status_cluster_id",
            ],
            "sentiment_order": list(SENTIMENT_ORDER),
            "review_vote_collapse": {
                "unknown_only": "unknown",
                "one_distinct_known_sentiment": "that_sentiment",
                "multiple_distinct_known_sentiments": "mixed",
            },
            "sentiment_denominator": "review_level_vote",
            "evidence_per_sentiment": selected_settings.evidence_per_sentiment,
            "decimal_places": selected_settings.decimal_places,
            "absence_policy": "NO_EVIDENCE",
        },
        "sentiment_distribution": {
            "opinion_unit": sentiment_distribution(
                unit_counts, decimal_places=selected_settings.decimal_places
            ),
            "review_aspect_vote": sentiment_distribution(
                aspect_vote_counts,
                decimal_places=selected_settings.decimal_places,
            ),
        },
        "aspect_summary": aspect_rows,
        "aspect_status_summary": pair_rows,
        "most_debated_aspect": most_debated,
        "related_products": related_products,
        "normalization_quality": {
            "automatic_integrity_checks": {
                "passed": int(inputs.integrity["passed"].sum()),
                "total": len(inputs.integrity),
                "all_passed": bool(inputs.integrity["passed"].all()),
            },
            "aspect_mapping": mapping_stats(
                product,
                "aspect_mapping_applied",
                "aspect_mapping_distance",
                decimal_places=selected_settings.decimal_places,
            ),
            "status_mapping": mapping_stats(
                product,
                "status_mapping_applied",
                "status_mapping_distance",
                decimal_places=selected_settings.decimal_places,
            ),
            "risky_clusters_present": product_risks,
        },
        "quality_flags": quality_flags(
            inputs,
            product_name=product_name,
            decimal_places=selected_settings.decimal_places,
        ),
        "agent_capabilities": {
            "supported": [
                "SUMMARIZE_STRUCTURED",
                "COMPARE_REVIEW_EVIDENCE",
                "RECOMMEND_REVIEW_CONDITIONED",
                "ASK_CLARIFICATION",
                "ABSTAIN",
            ],
            "unsupported_claims": [
                "best_overall_product",
                "lowest_price",
                "in_stock",
                "fastest_shipping",
                "objective_spec_superiority",
                "causal_product_effect",
            ],
            "absence_policy": "NO_EVIDENCE",
        },
    }


def markdown_cell(value: Any) -> str:
    """Escape a scalar or compact JSON value for a Markdown table cell."""
    if value is None:
        return "—"
    if isinstance(value, (bool, dict, list)):
        rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    else:
        rendered = str(value)
    return rendered.replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>")


def markdown_bold(value: Any) -> str:
    """Render a Markdown-safe scalar in bold for display-only labels."""
    rendered = markdown_cell(value).replace("*", r"\*")
    return f"**{rendered}**"


def _fixed(value: Any, places: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{places}f}"


def render_human_evaluation_markdown(human_evaluation: dict[str, Any]) -> str:
    """Render the shared A-D validation block used by both artifact views."""
    lines = ["## Human evaluation", ""]
    if human_evaluation["status"] != "completed":
        lines.append(
            "완료된 사용자 평가 파일이 없어 A–D 품질 비교 표를 생략했습니다. "
            "따라서 아래 보고서의 정규화 품질은 자동 무결성 검사까지만 검증된 상태입니다."
        )
        return "\n".join(lines)

    validation = human_evaluation["validation"]
    review = human_evaluation["review_evaluation"]
    summary_lookup = {
        (row["experiment"], row["criterion"]): row for row in review["experiment_summary"]
    }
    preference_lookup = {row["experiment"]: row for row in review["preference_summary"]}
    lines.extend(
        [
            (
                f"평가자 {validation['evaluator_count']}명이 동일한 리뷰 "
                f"{validation['review_task_count']}개에서 A–D를 1–5점으로 평가했습니다. "
                "A/B는 대표 속성, C/D는 구조화 Opinion Unit이며 B/D에 클러스터링을 적용했습니다."
            ),
            "",
            (
                "| experiment | factual_faithfulness | core_explanatory_power | "
                "information_coverage | preferred_count | preference_rate |"
            ),
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for experiment in ("A", "B", "C", "D"):
        preference = preference_lookup[experiment]
        lines.append(
            f"| {experiment} | "
            f"{_fixed(summary_lookup[(experiment, 'factual_faithfulness')]['mean'])} | "
            f"{_fixed(summary_lookup[(experiment, 'core_explanatory_power')]['mean'])} | "
            f"{_fixed(summary_lookup[(experiment, 'information_coverage')]['mean'])} | "
            f"{preference['selected_count']} | "
            f"{float(preference['selection_rate']):.1%} |"
        )

    structured = {
        row["criterion"]: row
        for row in review["factorial_effects"]
        if row["effect"] == "structured_representation_main_effect"
    }
    clustering = [
        row for row in review["factorial_effects"] if row["effect"] == "clustering_main_effect"
    ]
    d_risks = [
        row
        for row in human_evaluation["cluster_evaluation"]["risky_cluster_coverage"]
        if row["stage"] in {"experiment_d_aspect", "experiment_d_status"}
    ]
    evaluated_risks = sum(row["evaluated_risky_cluster_count"] for row in d_risks)
    bounded_risks = sum(row["bounded_risky_cluster_count"] for row in d_risks)
    d_preference = preference_lookup["D"]["selection_rate"]
    cluster_means = [row["mean"] for row in human_evaluation["cluster_evaluation"]["stage_summary"]]
    structured_supported = all(row["direction"] == "POSITIVE" for row in structured.values())
    clustering_uncertain = all(row["direction"] == "UNCERTAIN" for row in clustering)
    structured_interval_text = (
        "세 95% bootstrap 구간의 하한이 모두 0보다 큽니다"
        if structured_supported
        else "세 지표 전체에서 양의 효과가 확인되지는 않았습니다"
    )
    clustering_interval_text = (
        "세 95% bootstrap 구간이 모두 0을 포함합니다"
        if clustering_uncertain
        else "적어도 한 지표에서 0과 구분되는 효과가 검출되었습니다"
    )
    lines.extend(
        [
            "",
            (
                "구조화 표현의 주효과는 사실 충실도 "
                f"{structured['factual_faithfulness']['mean_effect_points']:+.3f}점, 핵심 설명력 "
                f"{structured['core_explanatory_power']['mean_effect_points']:+.3f}점, 정보 포괄성 "
                f"{structured['information_coverage']['mean_effect_points']:+.3f}점이며 "
                f"{structured_interval_text}. D 선택률은 {d_preference:.1%}이고, "
                f"클러스터링 주효과는 {clustering_interval_text}. 표본 클러스터 평균은 "
                f"{min(cluster_means):.3f}–{max(cluster_means):.3f}점이지만 D 위험 클러스터는 "
                f"{evaluated_risks}/{bounded_risks}개만 평가되어 전체 클러스터로 일반화할 수 없습니다."
            ),
        ]
    )
    return "\n".join(lines)


def _rate_percent(value: float | None) -> str:
    return "—" if value is None else f"{float(value) * 100:.1f}%"


def _topic_particle(value: str) -> str:
    """Choose 은/는 for a Korean-Hangul-final topic, with a safe non-Hangul default."""
    text = value.rstrip()
    if not text:
        return "는"
    codepoint = ord(text[-1])
    if 0xAC00 <= codepoint <= 0xD7A3:
        return "은" if (codepoint - 0xAC00) % 28 else "는"
    return "은"


def _directional_particle(value: str) -> str:
    """Choose 으로/로 for a Korean-Hangul-final directional phrase."""
    text = value.rstrip()
    if not text:
        return "로"
    codepoint = ord(text[-1])
    if 0xAC00 <= codepoint <= 0xD7A3:
        final_consonant = (codepoint - 0xAC00) % 28
        # No final consonant and final ㄹ both take 로.
        return "로" if final_consonant in {0, 8} else "으로"
    return "로"


def _count_and_rate(count: int, rate: float | None) -> str:
    return f"{count}({_rate_percent(rate)})"


def _sentiment_cell(row: dict[str, Any], sentiment: str) -> str:
    distribution = row["sentiment"]
    return _count_and_rate(
        int(distribution["counts"][sentiment]),
        distribution["shares"][sentiment],
    )


def _top_negative_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for row in rows if row["sentiment"]["counts"]["negative"]]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            row["sentiment"]["counts"]["negative"],
            float(row["sentiment"]["shares"]["negative"] or 0.0),
            row["supporting_review_count"],
            -row["rank"],
        ),
    )


def _render_review_samples(lines: list[str], samples: list[dict[str, Any]]) -> None:
    if not samples:
        lines.append("- 근거 없음")
        return
    for sample in samples:
        review_text = sample.get("review_text") or sample["excerpt"]
        lines.append(f"- {markdown_cell(review_text)}")


def _render_related_products_markdown(report: dict[str, Any]) -> str:
    related = report["related_products"]
    similar = related["similar_products"]
    alternatives = related["weakness_repair_alternatives"]
    lines = [
        "## 관련 상품",
        "",
        "### 유사 상품",
        "",
    ]
    if similar:
        top = similar[0]
        lines.append(
            "본 상품과 리뷰 경험이 가장 유사한 제품은 리뷰 경험 유사도 "
            f"값 {_fixed(top['experience_similarity'], 6)}을 가지는 "
            f"{markdown_bold(top['product_name'])}입니다. 이는 experience similarity를 기반으로 "
            "evidence overlap과 support reliability를 고려하고 계산되었습니다."
        )
    else:
        lines.append("본 상품과 비교 가능한 리뷰 경험 유사 상품이 없습니다.")
    lines.extend(
        [
            "",
            (
                "- **experience similarity**: 각 상품 aspect들을 긍정/부정/혼합/중립 값을 "
                "벡터화하여 코사인 유사도 값을 계산합니다."
            ),
            ("- **evidence overlap**: 공통적으로 가지는 aspect 비율."),
            ("- **support reliability**: 공통 aspect의 관측 리뷰 수가 충분한지를 반영한 신뢰도."),
            "  - 높을수록 소수 리뷰의 우연한 감성보다 반복 관측된 근거에 기반합니다.",
            "",
            "| rank | product | experience similarity | evidence overlap | support reliability |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in similar:
        components = row["components"]
        lines.append(
            f"| {row['rank']} | {markdown_bold(row['product_name'])} | "
            f"{_fixed(row['experience_similarity'], 6)} | "
            f"{_fixed(components['evidence_overlap'], 6)} | "
            f"{_fixed(components['support_reliability'], 6)} |"
        )

    lines.extend(
        [
            "",
            "### 관찰된 약점을 보완하는 대안 상품",
            "",
        ]
    )
    requirements = alternatives["source_weakness_requirements"]
    if requirements:
        requirement_text = ", ".join(
            f"'{item['aspect']} {item['status']}'" for item in requirements[:3]
        )
        lines.extend(
            [
                (
                    f"본 상품에는 {requirement_text}과 같은 부정적인 aspect-status 속성이 존재합니다."
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "본 상품에는 support 2 이상으로 확인된 부정적인 aspect-status 속성이 없습니다.",
                "",
            ]
        )
    if alternatives["alternatives"]:
        top = alternatives["alternatives"][0]
        lines.extend(
            [
                (
                    "부정 속성을 기피 조건으로 두고 약점을 보완할 수 있는 상품은 "
                    f"'{markdown_bold(top['product_name'])}'입니다. 이는 weakness utility과 "
                    "experience similarity를 3:1로 결합한 최종 점수인 weakness repair score를 "
                    "기반으로 계산되었습니다."
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "근거가 확인된 약점 보완 후보가 없어 추천을 보류했습니다.",
                "",
            ]
        )
    lines.extend(
        [
            (
                "- **weakness utility**: 원본의 부정 조건에 대해 후보가 얼마나 유리한지를 "
                "나타내는 근거 점수(-1~1로 정규화)."
            ),
            (
                "- **experience similarity**: 상품의 유사성으로 약점 보완 순위가 어느 정도의 "
                "원본 경험 유사성을 유지하는지를 평가."
            ),
            (
                "- **weakness repair score**: weakness utility과 experience similarity를 3:1로 "
                "결합한 최종 점수로 비슷한 맥락에서 약점을 해결할 근거가 됩니다."
            ),
            "",
            "| rank | product | weakness utility | experience similarity | weakness repair score |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in alternatives["alternatives"]:
        lines.append(
            f"| {row['rank']} | {markdown_bold(row['product_name'])} | "
            f"{_fixed(row['weakness_utility_score'], 6)} | "
            f"{_fixed(row['experience_similarity'], 6)} | "
            f"{_fixed(row['weakness_repair_score'], 6)} |"
        )
    return "\n".join(lines)


def render_static_markdown(report: dict[str, Any], *, row_limit: int = 10) -> str:
    """Render the requested compact catalog view from deterministic report fields."""
    if row_limit < 1:
        raise ValueError("row_limit must be at least 1.")
    product = report["product"]
    coverage = report["coverage"]
    reduction = report["normalization_reduction"]
    aspect_reduction = reduction["aspect"]
    status_reduction = reduction["status"]
    pair_reduction = reduction["aspect_status"]
    display_product_name = markdown_bold(product["product_name"])
    lines = [
        f"# {display_product_name} 상품에 대한 정적 카탈로그 분석 보고서",
        "",
        f"{display_product_name} 상품에 대한 정적 카탈로그 분석 보고서입니다.",
        "",
        (
            f"{product['catalog_review_count']}개의 리뷰에서 {coverage['opinion_unit_count']}개의 "
            "Opinion Units 속성을 추출하였습니다."
        ),
        "",
        (
            "군집화를 통해 기존 "
            f"{aspect_reduction['raw_count']}개였던 aspect의 가짓 수를 "
            f"{aspect_reduction['normalized_count']}개로 "
            f"{_rate_percent(aspect_reduction['decrease_rate'])} 감소시켰으며, aspect에 종속된 "
            f"status의 가짓 수를 {status_reduction['raw_count']}개에서 "
            f"{status_reduction['normalized_count']}개로 "
            f"{_rate_percent(status_reduction['decrease_rate'])} 감소시켰습니다. 최종적으로 "
            "aspect-status 조합의 고유 가짓 수는 "
            f"{_rate_percent(pair_reduction['decrease_rate'])} 감소했습니다."
        ),
    ]

    shown_aspects = report["aspect_summary"][:row_limit]
    top_aspect = shown_aspects[0] if shown_aspects else None
    negative_aspect = _top_negative_row(shown_aspects)
    lines.extend(
        [
            "",
            f"## 속성 감성 행렬 (상위 {row_limit}개)",
            "",
            "리뷰를 aspect 단위로 분석한 표입니다.",
            "",
            "- `etc`는 차례대로 mixed/neutral/unknown의 개수입니다.",
            "- `mention rate`의 분모는 해당 상품의 전체 카탈로그 리뷰입니다.",
            "",
            ("| rank | aspect | reviews(mention rate) | positive(rate) | negative(rate) | etc |"),
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in shown_aspects:
        counts = row["sentiment"]["counts"]
        etc = f"{counts['mixed']}/{counts['neutral']}/{counts['unknown']}"
        lines.append(
            f"| {row['rank']} | {markdown_cell(row['aspect'])} | "
            f"{_count_and_rate(row['supporting_review_count'], row['mention_rate'])} | "
            f"{_sentiment_cell(row, 'positive')} | {_sentiment_cell(row, 'negative')} | {etc} |"
        )
    if top_aspect:
        lines.extend(
            [
                "",
                (
                    f"- 가장 넓게 언급된 aspect는 '{top_aspect['aspect']}'"
                    f"{_directional_particle(top_aspect['aspect'])} "
                    f"{top_aspect['supporting_review_count']}개의 리뷰에서 언급되었습니다."
                ),
            ]
        )
    if negative_aspect:
        lines.append(
            f"- 부정 리뷰가 가장 많은 aspect는 '{negative_aspect['aspect']}'"
            f"{_directional_particle(negative_aspect['aspect'])} "
            f"{negative_aspect['sentiment']['counts']['negative']}개의 리뷰에서 언급되었습니다."
        )
    elif shown_aspects:
        lines.append("- 상위 표시 aspect에는 부정 리뷰가 없습니다.")

    shown_pairs = report["aspect_status_summary"][:row_limit]
    negative_pair = _top_negative_row(report["aspect_status_summary"])
    lines.extend(
        [
            "",
            f"## 속성-상태 행렬 (상위 {row_limit}개)",
            "",
            "리뷰를 aspect와 status 조합 단위로 분석한 표입니다.",
            "",
            "- `positive wilson lower`는 표본 수를 보수적으로 반영한 긍정 비율의 95% Wilson 하한입니다.",
            "- `etc`는 차례대로 mixed/neutral/unknown의 개수입니다.",
            "",
            (
                "| rank | aspect > status | reviews(mention rate) | positive(rate) | "
                "negative(rate) | etc | positive wilson lower |"
            ),
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in shown_pairs:
        counts = row["sentiment"]["counts"]
        etc = f"{counts['mixed']}/{counts['neutral']}/{counts['unknown']}"
        aspect_status = f"{row['aspect']} > {row['status'] if row['status'] is not None else '—'}"
        lines.append(
            f"| {row['rank']} | {markdown_cell(aspect_status)} | "
            f"{_count_and_rate(row['supporting_review_count'], row['mention_rate'])} | "
            f"{_sentiment_cell(row, 'positive')} | {_sentiment_cell(row, 'negative')} | {etc} | "
            f"{_fixed(row['sentiment']['positive_wilson_95']['lower'], 6)} |"
        )
    if shown_pairs:
        top_three = ", ".join(
            (
                f"'{row['aspect']} > {row['status'] if row['status'] is not None else '—'}' "
                f"({row['supporting_review_count']}개)"
            )
            for row in shown_pairs[:3]
        )
        lines.extend(
            [
                "",
                f"- 가장 많이 언급된 제품 속성은 {top_three}입니다.",
            ]
        )
    if negative_pair:
        negative_pair_label = (
            f"{negative_pair['aspect']} > "
            f"{negative_pair['status'] if negative_pair['status'] is not None else '—'}"
        )
        lines.append(
            "- 전체 aspect-status 조합에서 부정 리뷰가 가장 많은 항목은 "
            f"'{negative_pair_label}'{_directional_particle(negative_pair_label)} "
            f"{negative_pair['sentiment']['counts']['negative']}개의 리뷰에서 언급되었습니다."
        )
    elif shown_pairs:
        lines.append("- 전체 aspect-status 조합에 부정 리뷰가 없습니다.")

    most_debated = report["most_debated_aspect"]
    lines.extend(["", "## 가장 논쟁적인 속성", ""])
    if most_debated:
        sentiment = most_debated["sentiment"]
        lines.extend(
            [
                (
                    f"{most_debated['aspect']}{_topic_particle(most_debated['aspect'])} 긍정 "
                    f"{_rate_percent(sentiment['shares']['positive'])}, "
                    f"부정 {_rate_percent(sentiment['shares']['negative'])}로 상위 10개의 aspect 중 "
                    "가장 작은 긍정/부정 비율 차이인 "
                    f"{_rate_percent(most_debated['positive_negative_rate_gap'])}를 보이는 "
                    "가장 논쟁적인 aspect입니다."
                ),
                "",
                "### 긍정 리뷰 샘플",
                "",
            ]
        )
        _render_review_samples(lines, most_debated["evidence"]["positive"])
        lines.extend(["", "### 부정 리뷰 샘플", ""])
        _render_review_samples(lines, most_debated["evidence"]["negative"])
    else:
        lines.append(
            "상위 10개의 aspect에 긍정 또는 부정 표본이 없어 논쟁적인 aspect를 계산하지 않았습니다."
        )

    lines.extend(["", _render_related_products_markdown(report)])
    # The Markdown surface intentionally uses reader-facing spaced labels; JSON keeps snake_case.
    return "\n".join(lines).replace("_", " ") + "\n"


def atomic_write_text(text: str, destination: Path) -> None:
    """Write text atomically without partially replacing an existing artifact."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(payload: dict[str, Any], destination: Path) -> None:
    atomic_write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        destination,
    )
