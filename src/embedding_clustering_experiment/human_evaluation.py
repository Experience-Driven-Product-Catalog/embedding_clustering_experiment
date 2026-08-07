"""Validate and summarize completed blinded A-D human evaluation results."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import sha256_file

REVIEW_CRITERIA = (
    "factual_faithfulness",
    "core_explanatory_power",
    "information_coverage",
)
CLUSTER_CRITERIA = ("cohesion", "canonical_label_fit")
EXPERIMENTS = ("A", "B", "C", "D")
MINIMUM_EVALUATORS = 3
BOOTSTRAP_ITERATIONS = 5000
CONFIDENCE_LEVEL = 0.95
HUMAN_EVALUATION_SCHEMA_VERSION = "1.0.0"

REQUIRED_RESULT_COLUMNS = {
    "evaluator_uuid",
    "evaluation_level",
    "review_idx",
    "experiment",
    *REVIEW_CRITERIA,
    "preferred",
    "cluster_validation_id",
    "stage",
    "cluster_id",
    *CLUSTER_CRITERIA,
}

EXPERIMENT_CONTRACT = {
    "A": {"representation": "representative_attribute", "clustering": False},
    "B": {"representation": "representative_attribute", "clustering": True},
    "C": {"representation": "structured_opinion_unit", "clustering": False},
    "D": {"representation": "structured_opinion_unit", "clustering": True},
}


def _round_float(value: Any, places: int = 6) -> float | None:
    if value is None or pd.isna(value):
        return None
    rounded = round(float(value), places)
    return 0.0 if rounded == 0 else rounded


def _bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    *,
    iterations: int,
    confidence_level: float,
) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Bootstrap input must be a finite, non-empty one-dimensional array.")
    indices = rng.integers(0, len(array), size=(iterations, len(array)))
    means = array[indices].mean(axis=1)
    alpha = 1.0 - confidence_level
    return {
        "lower": _round_float(np.quantile(means, alpha / 2)),
        "upper": _round_float(np.quantile(means, 1.0 - alpha / 2)),
    }


def _icc_2_1(matrix: np.ndarray) -> float | None:
    """Two-way random-effects, absolute-agreement, single-measure ICC."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        return None
    if not np.isfinite(values).all():
        return None
    targets, raters = values.shape
    grand = values.mean()
    target_means = values.mean(axis=1)
    rater_means = values.mean(axis=0)
    ss_targets = raters * np.square(target_means - grand).sum()
    ss_raters = targets * np.square(rater_means - grand).sum()
    residual = values - target_means[:, None] - rater_means[None, :] + grand
    ss_error = np.square(residual).sum()
    ms_targets = ss_targets / (targets - 1)
    ms_raters = ss_raters / (raters - 1)
    ms_error = ss_error / ((targets - 1) * (raters - 1))
    denominator = ms_targets + (raters - 1) * ms_error + raters * (ms_raters - ms_error) / targets
    if math.isclose(denominator, 0.0, abs_tol=1e-15):
        return 1.0 if np.all(values == values.flat[0]) else None
    return _round_float((ms_targets - ms_error) / denominator)


def _effect_direction(interval: dict[str, float]) -> str:
    if interval["lower"] > 0:
        return "POSITIVE"
    if interval["upper"] < 0:
        return "NEGATIVE"
    return "UNCERTAIN"


def _resolve_completed_dir(
    run_dir: Path,
    human_results_dir: Path | None,
) -> tuple[Path, bool]:
    if human_results_dir is None:
        root = run_dir / "evaluation/human_results"
        explicit = False
    else:
        root = human_results_dir.expanduser().resolve()
        explicit = True
        if not root.exists():
            raise FileNotFoundError(f"human results directory does not exist: {root}")
    completed = root / "completed_evaluators" if (root / "completed_evaluators").is_dir() else root
    return completed, explicit


def _incomplete_summary(*, explicit: bool) -> dict[str, Any]:
    return {
        "schema_version": HUMAN_EVALUATION_SCHEMA_VERSION,
        "status": "not_available",
        "experiment_contract": EXPERIMENT_CONTRACT,
        "source": {
            "discovery": "explicit" if explicit else "run_default",
            "completed_result_file_count": 0,
            "results_set_sha256": None,
        },
        "validation": {
            "minimum_evaluator_count": MINIMUM_EVALUATORS,
            "evaluator_count": 0,
            "minimum_evaluator_count_reached": False,
            "all_checks_passed": False,
        },
        "review_evaluation": None,
        "cluster_evaluation": None,
        "conclusion": {
            "structured_representation": "NOT_EVALUATED",
            "clustering_review_quality": "NOT_EVALUATED",
            "sampled_cluster_quality": "NOT_EVALUATED",
            "generalization": "NOT_EVALUATED",
            "reason_codes": ["NO_COMPLETED_EVALUATOR_FILES"],
        },
    }


def _load_completed_results(completed_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    files = sorted(completed_dir.glob("*.parquet")) if completed_dir.is_dir() else []
    frames: list[pd.DataFrame] = []
    file_records: list[dict[str, Any]] = []
    for session_number, path in enumerate(files, start=1):
        frame = pd.read_parquet(path)
        missing = sorted(REQUIRED_RESULT_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"{path.name} is missing human evaluation columns: {missing}")
        evaluator_ids = frame["evaluator_uuid"].dropna().astype(str).unique()
        if len(evaluator_ids) != 1 or evaluator_ids[0] != path.stem:
            raise ValueError(f"{path.name} must contain exactly its filename evaluator UUID.")
        frames.append(frame)
        file_records.append(
            {
                "evaluator_alias": f"session_{session_number}",
                "sha256": sha256_file(path),
                "row_count": len(frame),
            }
        )
    if not frames:
        return pd.DataFrame(), []
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined, file_records


def _validate_completed_results(
    combined: pd.DataFrame,
    review_tasks: pd.DataFrame,
    cluster_tasks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if review_tasks["review_idx"].isna().any() or not review_tasks["review_idx"].is_unique:
        raise ValueError("Review evaluation task IDs must be unique and non-null.")
    if (
        cluster_tasks["cluster_validation_id"].isna().any()
        or not cluster_tasks["cluster_validation_id"].is_unique
    ):
        raise ValueError("Cluster evaluation task IDs must be unique and non-null.")
    levels = set(combined["evaluation_level"].dropna().astype(str))
    if levels != {"review", "cluster"}:
        raise ValueError(f"Human evaluation levels must be review and cluster; found {levels}.")
    evaluator_ids = sorted(combined["evaluator_uuid"].astype(str).unique())
    if len(evaluator_ids) != combined.groupby("evaluator_uuid", observed=True).ngroups:
        raise AssertionError("Evaluator group count is inconsistent.")

    review = combined.loc[combined["evaluation_level"].eq("review")].copy()
    cluster = combined.loc[combined["evaluation_level"].eq("cluster")].copy()
    review_only_null_columns = [
        "cluster_validation_id",
        "stage",
        "cluster_id",
        *CLUSTER_CRITERIA,
    ]
    if review[review_only_null_columns].notna().any().any():
        raise ValueError("Review-level human results contain cluster-only values.")
    if review["review_idx"].isna().any():
        raise ValueError("Review-level human results contain null review_idx values.")
    review_idx_values = review["review_idx"].astype(float)
    if not np.allclose(review_idx_values, np.round(review_idx_values)):
        raise ValueError("Review-level human results contain non-integer review_idx values.")
    review["review_idx"] = review_idx_values.astype("int64")
    expected_review_ids = set(review_tasks["review_idx"].astype("int64"))
    if set(review["review_idx"]) != expected_review_ids:
        raise ValueError("Human review results do not cover the selected run review task cohort.")
    expected_review_rows = len(evaluator_ids) * len(expected_review_ids) * len(EXPERIMENTS)
    if len(review) != expected_review_rows:
        raise ValueError(
            f"Expected {expected_review_rows} review rating rows; found {len(review)}."
        )
    if review.duplicated(["evaluator_uuid", "review_idx", "experiment"]).any():
        raise ValueError("Duplicate evaluator-review-experiment ratings exist.")
    review_blocks = review.groupby(["evaluator_uuid", "review_idx"], observed=True)[
        "experiment"
    ].agg(lambda values: set(map(str, values)))
    if not review_blocks.map(lambda values: values == set(EXPERIMENTS)).all():
        raise ValueError("Every evaluator-review block must contain A-D exactly once.")
    for criterion in REVIEW_CRITERIA:
        values = review[criterion]
        if values.isna().any() or not values.between(1, 5).all():
            raise ValueError(f"{criterion} ratings must be complete and within 1..5.")
        if not np.allclose(values, np.round(values)):
            raise ValueError(f"{criterion} ratings must be integer Likert scores.")
    if review["preferred"].isna().any() or not set(review["preferred"].unique()) <= {
        True,
        False,
    }:
        raise ValueError("preferred must contain complete boolean values.")
    review["preferred"] = review["preferred"].astype(bool)
    selected_per_block = review.groupby(["evaluator_uuid", "review_idx"], observed=True)[
        "preferred"
    ].sum()
    if not selected_per_block.eq(1).all():
        raise ValueError("Every evaluator-review block must select exactly one preference.")

    required_cluster_fields = [
        "cluster_validation_id",
        "stage",
        "cluster_id",
        *CLUSTER_CRITERIA,
    ]
    if cluster[required_cluster_fields].isna().any().any():
        raise ValueError("Cluster-level human results contain null required fields.")
    cluster_only_null_columns = ["review_idx", "experiment", *REVIEW_CRITERIA, "preferred"]
    if cluster[cluster_only_null_columns].notna().any().any():
        raise ValueError("Cluster-level human results contain review-only values.")
    expected_cluster_ids = set(cluster_tasks["cluster_validation_id"].astype(str))
    if set(cluster["cluster_validation_id"].astype(str)) != expected_cluster_ids:
        raise ValueError("Human cluster results do not cover the selected run cluster task cohort.")
    expected_cluster_rows = len(evaluator_ids) * len(expected_cluster_ids)
    if len(cluster) != expected_cluster_rows:
        raise ValueError(
            f"Expected {expected_cluster_rows} cluster rating rows; found {len(cluster)}."
        )
    if cluster.duplicated(["evaluator_uuid", "cluster_validation_id"]).any():
        raise ValueError("Duplicate evaluator-cluster ratings exist.")
    task_identity = cluster_tasks.set_index("cluster_validation_id")[["stage", "cluster_id"]]
    observed_identity = cluster[["cluster_validation_id", "stage", "cluster_id"]].drop_duplicates()
    observed_identity = observed_identity.set_index("cluster_validation_id").sort_index()
    expected_identity = task_identity.reindex(observed_identity.index).astype(str)
    if not observed_identity.astype(str).equals(expected_identity):
        raise ValueError("Human cluster result stage or cluster ID differs from its task.")
    for criterion in CLUSTER_CRITERIA:
        values = cluster[criterion]
        if values.isna().any() or not values.between(1, 5).all():
            raise ValueError(f"{criterion} ratings must be complete and within 1..5.")
        if not np.allclose(values, np.round(values)):
            raise ValueError(f"{criterion} ratings must be integer Likert scores.")

    return (
        review,
        cluster,
        {
            "minimum_evaluator_count": MINIMUM_EVALUATORS,
            "evaluator_count": len(evaluator_ids),
            "minimum_evaluator_count_reached": len(evaluator_ids) >= MINIMUM_EVALUATORS,
            "review_task_count": len(expected_review_ids),
            "cluster_task_count": len(expected_cluster_ids),
            "review_rating_row_count": len(review),
            "cluster_rating_row_count": len(cluster),
            "evaluator_review_block_count": len(selected_per_block),
            "preferred_selections_per_block": {
                str(int(value)): int(count)
                for value, count in selected_per_block.value_counts().sort_index().items()
            },
            "all_checks_passed": True,
        },
    )


def _agreement_rows(
    frame: pd.DataFrame,
    *,
    item_column: str,
    group_column: str,
    criteria: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, group_frame in frame.groupby(group_column, sort=True, observed=True):
        for criterion in criteria:
            pivot = group_frame.pivot(
                index=item_column,
                columns="evaluator_uuid",
                values=criterion,
            ).sort_index(axis=1)
            rows.append(
                {
                    "group": str(group),
                    "criterion": criterion,
                    "item_count": len(pivot),
                    "metric": "ICC_2_1_absolute_single",
                    "value": _icc_2_1(pivot.to_numpy()),
                }
            )
    return rows


def _review_analysis(
    review: pd.DataFrame,
    *,
    seed: int,
    iterations: int,
    confidence_level: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    item_means = review.groupby(["review_idx", "experiment"], observed=True)[
        list(REVIEW_CRITERIA)
    ].mean()
    summary_rows: list[dict[str, Any]] = []
    for experiment in EXPERIMENTS:
        for criterion in REVIEW_CRITERIA:
            values = item_means.xs(experiment, level="experiment")[criterion].to_numpy()
            summary_rows.append(
                {
                    "experiment": experiment,
                    "criterion": criterion,
                    "review_count": len(values),
                    "rating_count": int(review["experiment"].eq(experiment).sum()),
                    "mean": _round_float(values.mean()),
                    "median": _round_float(np.median(values)),
                    "standard_deviation_across_reviews": _round_float(values.std(ddof=1)),
                    "bootstrap_mean_95": _bootstrap_mean_ci(
                        values,
                        rng,
                        iterations=iterations,
                        confidence_level=confidence_level,
                    ),
                }
            )

    preference_rows: list[dict[str, Any]] = []
    item_preference = review.groupby(["review_idx", "experiment"], observed=True)[
        "preferred"
    ].mean()
    block_count = review[["evaluator_uuid", "review_idx"]].drop_duplicates().shape[0]
    for experiment in EXPERIMENTS:
        values = item_preference.xs(experiment, level="experiment").to_numpy(dtype=float)
        selected_count = int(review.loc[review["experiment"].eq(experiment), "preferred"].sum())
        preference_rows.append(
            {
                "experiment": experiment,
                "selected_count": selected_count,
                "evaluator_review_block_count": block_count,
                "selection_rate": _round_float(selected_count / block_count),
                "bootstrap_selection_rate_95": _bootstrap_mean_ci(
                    values,
                    rng,
                    iterations=iterations,
                    confidence_level=confidence_level,
                ),
            }
        )

    contrast_definitions = (
        ("B_minus_A", "B", "A"),
        ("D_minus_C", "D", "C"),
        ("C_minus_A", "C", "A"),
        ("D_minus_B", "D", "B"),
    )
    contrast_rows: list[dict[str, Any]] = []
    factorial_rows: list[dict[str, Any]] = []
    for criterion in REVIEW_CRITERIA:
        pivot = item_means[criterion].unstack("experiment")[list(EXPERIMENTS)]
        for contrast, after, before in contrast_definitions:
            values = (pivot[after] - pivot[before]).to_numpy(dtype=float)
            interval = _bootstrap_mean_ci(
                values,
                rng,
                iterations=iterations,
                confidence_level=confidence_level,
            )
            contrast_rows.append(
                {
                    "criterion": criterion,
                    "contrast": contrast,
                    "review_count": len(values),
                    "mean_difference_points": _round_float(values.mean()),
                    "bootstrap_mean_difference_95": interval,
                    "direction": _effect_direction(interval),
                }
            )
        effects = {
            "structured_representation_main_effect": (
                (pivot["C"] + pivot["D"]) - (pivot["A"] + pivot["B"])
            )
            / 2.0,
            "clustering_main_effect": ((pivot["B"] + pivot["D"]) - (pivot["A"] + pivot["C"])) / 2.0,
            "structure_clustering_interaction": (pivot["D"] - pivot["C"])
            - (pivot["B"] - pivot["A"]),
        }
        for effect, series in effects.items():
            values = series.to_numpy(dtype=float)
            interval = _bootstrap_mean_ci(
                values,
                rng,
                iterations=iterations,
                confidence_level=confidence_level,
            )
            factorial_rows.append(
                {
                    "criterion": criterion,
                    "effect": effect,
                    "review_count": len(values),
                    "mean_effect_points": _round_float(values.mean()),
                    "bootstrap_mean_95": interval,
                    "direction": _effect_direction(interval),
                }
            )

    structured = [
        row for row in factorial_rows if row["effect"] == "structured_representation_main_effect"
    ]
    clustering = [row for row in factorial_rows if row["effect"] == "clustering_main_effect"]
    preference_leader = max(preference_rows, key=lambda row: row["selection_rate"])
    interpretation_codes = []
    if structured and all(row["direction"] == "POSITIVE" for row in structured):
        interpretation_codes.append("STRUCTURED_REPRESENTATION_SUPPORTED")
    if clustering and all(row["direction"] == "UNCERTAIN" for row in clustering):
        interpretation_codes.append("NO_CLEAR_CLUSTERING_MAIN_EFFECT")
    interpretation_codes.append(f"{preference_leader['experiment']}_MOST_PREFERRED")
    return {
        "score_scale": {"minimum": 1, "maximum": 5},
        "bootstrap_contract": {
            "unit": "review_after_averaging_completed_evaluators",
            "iterations": iterations,
            "confidence_level": confidence_level,
            "random_seed": seed,
        },
        "experiment_summary": summary_rows,
        "preference_summary": preference_rows,
        "paired_contrasts": contrast_rows,
        "factorial_effects": factorial_rows,
        "agreement": _agreement_rows(
            review,
            item_column="review_idx",
            group_column="experiment",
            criteria=REVIEW_CRITERIA,
        ),
        "interpretation_codes": interpretation_codes,
    }


def _cluster_analysis(
    cluster: pd.DataFrame,
    cluster_tasks: pd.DataFrame,
    risky_clusters: pd.DataFrame,
    cluster_tables: dict[str, pd.DataFrame],
    *,
    seed: int,
    iterations: int,
    confidence_level: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 100)
    item_means = cluster.groupby(["cluster_validation_id", "stage"], observed=True)[
        list(CLUSTER_CRITERIA)
    ].mean()
    summary_rows: list[dict[str, Any]] = []
    for stage in sorted(cluster["stage"].astype(str).unique()):
        raw_stage = cluster.loc[cluster["stage"].eq(stage)]
        for criterion in CLUSTER_CRITERIA:
            values = item_means.xs(stage, level="stage")[criterion].to_numpy()
            summary_rows.append(
                {
                    "stage": stage,
                    "criterion": criterion,
                    "cluster_task_count": len(values),
                    "rating_count": len(raw_stage),
                    "mean": _round_float(values.mean()),
                    "median": _round_float(np.median(values)),
                    "rating_share_at_least_4": _round_float(raw_stage[criterion].ge(4).mean()),
                    "rating_share_at_most_2": _round_float(raw_stage[criterion].le(2).mean()),
                    "bootstrap_mean_95": _bootstrap_mean_ci(
                        values,
                        rng,
                        iterations=iterations,
                        confidence_level=confidence_level,
                    ),
                }
            )

    coverage_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    evaluated_by_stage = {
        stage: set(group["cluster_id"].astype(str))
        for stage, group in cluster_tasks.groupby("stage", observed=True)
    }
    for stage, frame in sorted(cluster_tables.items()):
        eligible = frame.loc[frame["member_count"].gt(1)]
        evaluated = evaluated_by_stage.get(stage, set())
        coverage_rows.append(
            {
                "stage": stage,
                "eligible_non_singleton_cluster_count": len(eligible),
                "evaluated_cluster_count": len(evaluated),
                "coverage_rate": (
                    _round_float(len(evaluated) / len(eligible)) if len(eligible) else None
                ),
            }
        )
        stage_risks = set(
            risky_clusters.loc[risky_clusters["stage"].eq(stage), "cluster_id"].astype(str)
        )
        evaluated_risks = evaluated & stage_risks
        risk_rows.append(
            {
                "stage": stage,
                "bounded_risky_cluster_count": len(stage_risks),
                "evaluated_risky_cluster_count": len(evaluated_risks),
                "coverage_rate": (
                    _round_float(len(evaluated_risks) / len(stage_risks)) if stage_risks else None
                ),
                "evaluated_risky_cluster_ids": sorted(evaluated_risks),
            }
        )

    five_share = _round_float(cluster[list(CLUSTER_CRITERIA)].stack().astype(float).eq(5).mean())
    interpretation_codes = []
    if all(row["mean"] >= 4.5 for row in summary_rows):
        interpretation_codes.append("SAMPLED_CLUSTER_RATINGS_HIGH")
    if any(row["coverage_rate"] is not None and row["coverage_rate"] < 0.5 for row in risk_rows):
        interpretation_codes.append("RISK_CLUSTER_COVERAGE_LIMITED")
    if five_share >= 0.7:
        interpretation_codes.append("CLUSTER_RATING_CEILING_RISK")
    return {
        "score_scale": {"minimum": 1, "maximum": 5},
        "stage_summary": summary_rows,
        "sampling_coverage": coverage_rows,
        "risky_cluster_coverage": risk_rows,
        "five_rating_share": five_share,
        "agreement": _agreement_rows(
            cluster,
            item_column="cluster_validation_id",
            group_column="stage",
            criteria=CLUSTER_CRITERIA,
        ),
        "interpretation_codes": interpretation_codes,
    }


def analyze_human_evaluation(
    run_dir: Path,
    human_results_dir: Path | None,
    *,
    review_tasks: pd.DataFrame,
    cluster_tasks: pd.DataFrame,
    risky_clusters: pd.DataFrame,
    cluster_tables: dict[str, pd.DataFrame],
    random_seed: int,
    iterations: int = BOOTSTRAP_ITERATIONS,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """Return a deterministic, fail-closed human evaluation summary."""
    completed_dir, explicit = _resolve_completed_dir(run_dir, human_results_dir)
    combined, file_records = _load_completed_results(completed_dir)
    if combined.empty:
        return _incomplete_summary(explicit=explicit)
    review, cluster, validation = _validate_completed_results(
        combined,
        review_tasks,
        cluster_tasks,
    )
    if not validation["minimum_evaluator_count_reached"]:
        raise ValueError(
            f"Need at least {MINIMUM_EVALUATORS} completed evaluators; "
            f"found {validation['evaluator_count']}."
        )
    result_hashes = [record["sha256"] for record in file_records]
    results_set_sha256 = hashlib.sha256("".join(result_hashes).encode("ascii")).hexdigest()
    review_analysis = _review_analysis(
        review,
        seed=random_seed,
        iterations=iterations,
        confidence_level=confidence_level,
    )
    cluster_analysis = _cluster_analysis(
        cluster,
        cluster_tasks,
        risky_clusters,
        cluster_tables,
        seed=random_seed,
        iterations=iterations,
        confidence_level=confidence_level,
    )
    structured_rows = [
        row
        for row in review_analysis["factorial_effects"]
        if row["effect"] == "structured_representation_main_effect"
    ]
    clustering_rows = [
        row
        for row in review_analysis["factorial_effects"]
        if row["effect"] == "clustering_main_effect"
    ]
    risk_rows = cluster_analysis["risky_cluster_coverage"]
    return {
        "schema_version": HUMAN_EVALUATION_SCHEMA_VERSION,
        "status": "completed",
        "experiment_contract": EXPERIMENT_CONTRACT,
        "source": {
            "discovery": "explicit" if explicit else "run_default",
            "completed_result_file_count": len(file_records),
            "results_set_sha256": results_set_sha256,
            "result_files": file_records,
            "review_tasks_sha256": sha256_file(
                run_dir / "evaluation/user_evaluation_tasks.parquet"
            ),
            "cluster_tasks_sha256": sha256_file(
                run_dir / "evaluation/cluster_evaluation_tasks.parquet"
            ),
        },
        "validation": validation,
        "review_evaluation": review_analysis,
        "cluster_evaluation": cluster_analysis,
        "conclusion": {
            "structured_representation": (
                "SUPPORTED"
                if all(row["direction"] == "POSITIVE" for row in structured_rows)
                else "INCONCLUSIVE"
            ),
            "clustering_review_quality": (
                "NO_CLEAR_MAIN_EFFECT"
                if all(row["direction"] == "UNCERTAIN" for row in clustering_rows)
                else "EFFECT_DETECTED"
            ),
            "sampled_cluster_quality": (
                "SUPPORTED_FOR_SAMPLED_CLUSTERS"
                if "SAMPLED_CLUSTER_RATINGS_HIGH" in cluster_analysis["interpretation_codes"]
                else "INCONCLUSIVE"
            ),
            "generalization": (
                "LIMITED_BY_EVALUATOR_COUNT_AND_RISK_COVERAGE"
                if validation["evaluator_count"] == MINIMUM_EVALUATORS
                or any(
                    row["coverage_rate"] is not None and row["coverage_rate"] < 0.5
                    for row in risk_rows
                )
                else "SUPPORTED"
            ),
            "reason_codes": [
                *review_analysis["interpretation_codes"],
                *cluster_analysis["interpretation_codes"],
                "MINIMUM_EVALUATOR_COUNT_ONLY"
                if validation["evaluator_count"] == MINIMUM_EVALUATORS
                else "EVALUATOR_COUNT_ABOVE_MINIMUM",
            ],
        },
    }
