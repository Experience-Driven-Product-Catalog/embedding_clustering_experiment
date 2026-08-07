"""Full-dataset automatic evaluation for experiments A-D."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import cosine_distances

from .data import FORBIDDEN_INTEGRATED_COLUMNS
from .experiments import ExperimentArtifacts


@dataclass(frozen=True)
class EvaluationArtifacts:
    tables: dict[str, pd.DataFrame]
    summary: dict[str, Any]


class IntegrityRecorder:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def check(self, name: str, passed: bool, details: str) -> None:
        self.rows.append({"check": name, "passed": bool(passed), "details": details})

    def finish(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.rows)
        failures = frame.loc[~frame["passed"]]
        if not failures.empty:
            names = ", ".join(failures["check"].astype(str))
            raise AssertionError(f"Automatic integrity checks failed: {names}")
        return frame


def _nullable_equal(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.eq(right).fillna(False) | (left.isna() & right.isna())


def _nullable_changed(left: pd.Series, right: pd.Series) -> pd.Series:
    return ~_nullable_equal(left, right)


def _source_column_preserved(
    result: pd.DataFrame,
    source: pd.DataFrame,
    column: str,
) -> bool:
    result_values = result.sort_values("idx", kind="stable")[column].reset_index(drop=True)
    source_values = (
        source.loc[source["idx"].isin(result["idx"])]
        .sort_values("idx", kind="stable")[column]
        .reset_index(drop=True)
    )
    return _nullable_equal(result_values, source_values).all()


def evaluate_integrity(
    representative: pd.DataFrame,
    opinion: pd.DataFrame,
    artifacts: ExperimentArtifacts,
    config: dict[str, Any],
) -> pd.DataFrame:
    recorder = IntegrityRecorder()
    eligible = artifacts.eligible_opinion
    results = artifacts.result_tables()
    recorder.check(
        "all_available_representative_rows_used",
        len(results["experiment_a"]) == len(representative)
        and len(results["experiment_b"]) == len(representative),
        f"input={len(representative)} A={len(results['experiment_a'])} "
        f"B={len(results['experiment_b'])}",
    )
    recorder.check(
        "all_eligible_opinion_rows_used",
        len(results["experiment_c"]) == len(eligible)
        and len(results["experiment_d"]) == len(eligible),
        f"eligible={len(eligible)} C={len(results['experiment_c'])} "
        f"D={len(results['experiment_d'])}",
    )
    recorder.check(
        "excluded_aspects_removed_only_from_C_D",
        set(opinion["idx"]) - set(eligible["idx"])
        == set(
            opinion.loc[
                opinion["raw_aspect"].isin(config["filters"]["excluded_raw_aspects"]),
                "idx",
            ]
        ),
        f"excluded_rows={len(opinion) - len(eligible)}",
    )
    for name, frame in results.items():
        forbidden = sorted(FORBIDDEN_INTEGRATED_COLUMNS & set(frame.columns))
        recorder.check(
            f"{name}_has_no_integrated_metadata",
            not forbidden,
            f"forbidden_columns={forbidden}",
        )
        recorder.check(
            f"{name}_idx_unique_and_nonnull",
            frame["idx"].notna().all() and frame["idx"].is_unique,
            f"rows={len(frame)} unique_idx={frame['idx'].nunique()}",
        )
        recorder.check(
            f"{name}_review_idx_is_only_external_key",
            "review_idx" in frame and frame["review_idx"].notna().all(),
            "review_idx present; product/review columns absent",
        )

    recorder.check(
        "experiment_A_is_raw_baseline",
        _nullable_equal(
            artifacts.experiment_a["raw_attribute"], artifacts.experiment_a["attribute"]
        ).all(),
        "attribute == raw_attribute for every row",
    )
    recorder.check(
        "experiment_C_is_raw_baseline",
        _nullable_equal(
            artifacts.experiment_c["raw_aspect"], artifacts.experiment_c["aspect"]
        ).all()
        and _nullable_equal(
            artifacts.experiment_c["raw_status"], artifacts.experiment_c["status"]
        ).all(),
        "aspect/status equal raw_aspect/raw_status including nulls",
    )
    recorder.check(
        "status_nullability_preserved",
        artifacts.experiment_c["status"].isna().equals(artifacts.experiment_c["raw_status"].isna())
        and artifacts.experiment_d["status"]
        .isna()
        .equals(artifacts.experiment_d["raw_status"].isna()),
        f"nullable_status_rows={int(eligible['raw_status'].isna().sum())}",
    )
    recorder.check(
        "sentiment_is_single_preserved_column",
        all(
            _source_column_preserved(frame, source, "sentiment")
            for frame, source in (
                (artifacts.experiment_a, representative),
                (artifacts.experiment_b, representative),
                (artifacts.experiment_c, eligible),
                (artifacts.experiment_d, eligible),
            )
        ),
        "no derived sentiment fields were added",
    )

    stage_specs = {
        "experiment_b": (
            artifacts.experiment_b_nodes,
            artifacts.experiment_b_clusters,
            "raw_attribute",
            (),
            config["clustering"]["experiment_b"]["distance_threshold"],
            len(representative),
        ),
        "experiment_d_aspect": (
            artifacts.experiment_d_aspect_nodes,
            artifacts.experiment_d_aspect_clusters,
            "raw_aspect",
            (),
            config["clustering"]["experiment_d"]["aspect"]["distance_threshold"],
            len(eligible),
        ),
        "experiment_d_status": (
            artifacts.experiment_d_status_nodes,
            artifacts.experiment_d_status_clusters,
            "raw_status",
            ("aspect_cluster_id",),
            config["clustering"]["experiment_d"]["status"]["distance_threshold"],
            int(eligible["raw_status"].notna().sum()),
        ),
    }
    tolerance = float(config["clustering"]["distance_tolerance"])
    for stage, (
        nodes,
        clusters,
        text_column,
        boundaries,
        threshold,
        row_count,
    ) in stage_specs.items():
        keys = [*boundaries, text_column]
        recorder.check(
            f"{stage}_node_key_unique",
            not nodes.duplicated(keys).any(),
            f"nodes={len(nodes)} keys={keys}",
        )
        recorder.check(
            f"{stage}_all_nodes_clustered",
            nodes["cluster_id"].notna().all(),
            f"assigned={int(nodes['cluster_id'].notna().sum())}/{len(nodes)}",
        )
        recorder.check(
            f"{stage}_source_row_coverage",
            int(nodes["source_row_count"].sum()) == row_count,
            f"node_source_rows={int(nodes['source_row_count'].sum())} expected={row_count}",
        )
        max_diameter = float(clusters["cluster_max_distance"].max())
        recorder.check(
            f"{stage}_complete_linkage_threshold",
            max_diameter <= float(threshold) + tolerance,
            f"max_diameter={max_diameter:.8f} threshold={float(threshold):.8f}",
        )
        recorder.check(
            f"{stage}_canonical_labels_are_observed",
            all(
                pd.isna(row.canonical_label)
                or str(row.canonical_label) in set(map(str, row.member_expressions))
                for row in clusters.itertuples(index=False)
            ),
            "every selected label is a cluster member",
        )
        recorder.check(
            f"{stage}_has_no_product_category_boundary",
            "product_category" not in nodes and "product_category" not in clusters,
            "product_category absent from nodes and clusters",
        )

    status_cluster_boundary_counts = artifacts.experiment_d_status_nodes.groupby(
        "cluster_id", observed=True
    )["aspect_cluster_id"].nunique()
    recorder.check(
        "D_status_stays_inside_aspect_cluster",
        status_cluster_boundary_counts.le(1).all(),
        f"status_clusters={len(status_cluster_boundary_counts)}",
    )
    return recorder.finish()


def _experiment_metric_row(experiment: str, frame: pd.DataFrame) -> dict[str, Any]:
    structured = experiment in {"C", "D"}
    normalized = experiment in {"B", "D"}
    if structured:
        raw_primary = ["raw_aspect"]
        final_primary = ["aspect"]
        raw_unit = ["raw_aspect", "raw_status"]
        final_unit = ["aspect", "status"]
        changed = _nullable_changed(frame["raw_aspect"], frame["aspect"]) | _nullable_changed(
            frame["raw_status"], frame["status"]
        )
    else:
        raw_primary = ["raw_attribute"]
        final_primary = ["attribute"]
        raw_unit = raw_primary
        final_unit = final_primary
        changed = _nullable_changed(frame["raw_attribute"], frame["attribute"])

    raw_primary_count = len(frame.drop_duplicates(raw_primary))
    final_primary_count = len(frame.drop_duplicates(final_primary))
    raw_unit_count = len(frame.drop_duplicates(raw_unit))
    final_unit_count = len(frame.drop_duplicates(final_unit))
    final_frequency = frame.groupby(final_unit, observed=True, dropna=False).size()
    singleton_unit_count = int(final_frequency.eq(1).sum())
    singleton_observation_count = int(final_frequency.loc[final_frequency.eq(1)].sum())
    duplicate_mask = frame.duplicated(["review_idx", *final_unit], keep=False)
    distinct_review_units = frame[["review_idx", *final_unit]].drop_duplicates()
    units_per_review = distinct_review_units.groupby("review_idx", observed=True).size()
    unique_review_count = int(frame["review_idx"].nunique())
    return {
        "experiment": experiment,
        "structured_output": structured,
        "normalized": normalized,
        "observation_count": len(frame),
        "unique_review_count": unique_review_count,
        "raw_primary_label_count": raw_primary_count,
        "final_primary_label_count": final_primary_count,
        "raw_unit_count": raw_unit_count,
        "final_unit_count": final_unit_count,
        "primary_label_compression_ratio": final_primary_count / raw_primary_count,
        "unit_compression_ratio": final_unit_count / raw_unit_count,
        "mapping_change_rate": float(changed.mean()),
        "singleton_final_unit_count": singleton_unit_count,
        "singleton_final_unit_rate": singleton_unit_count / final_unit_count,
        "singleton_observation_rate": singleton_observation_count / len(frame),
        "reused_observation_rate": 1.0 - singleton_observation_count / len(frame),
        "mean_observations_per_review": len(frame) / unique_review_count,
        "mean_unique_final_units_per_review": float(units_per_review.mean()),
        "within_review_duplicate_observation_rate": float(duplicate_mask.mean()),
    }


def build_result_metrics(
    artifacts: ExperimentArtifacts,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = {
        "A": artifacts.experiment_a,
        "B": artifacts.experiment_b,
        "C": artifacts.experiment_c,
        "D": artifacts.experiment_d,
    }
    overall = pd.DataFrame(
        [_experiment_metric_row(experiment, frame) for experiment, frame in frames.items()]
    )
    common_review_ids = set.intersection(*(set(frame["review_idx"]) for frame in frames.values()))
    paired = pd.DataFrame(
        [
            _experiment_metric_row(
                experiment, frame.loc[frame["review_idx"].isin(common_review_ids)]
            )
            for experiment, frame in frames.items()
        ]
    )
    cohort = pd.DataFrame(
        [
            {
                "representative_unique_review_count": int(
                    artifacts.experiment_a["review_idx"].nunique()
                ),
                "eligible_opinion_unique_review_count": int(
                    artifacts.experiment_c["review_idx"].nunique()
                ),
                "common_unique_review_count": len(common_review_ids),
                "representative_only_review_count": len(
                    set(artifacts.experiment_a["review_idx"]) - common_review_ids
                ),
                "eligible_opinion_only_review_count": len(
                    set(artifacts.experiment_c["review_idx"]) - common_review_ids
                ),
            }
        ]
    )

    indexed = paired.set_index("experiment")
    contrasts = {
        "normalization_direct_B_minus_A": ("B", "A"),
        "normalization_structured_D_minus_C": ("D", "C"),
        "structure_raw_C_minus_A": ("C", "A"),
        "structure_normalized_D_minus_B": ("D", "B"),
    }
    excluded = {"structured_output", "normalized"}
    numeric_metrics = [
        column
        for column in paired.columns
        if column != "experiment"
        and column not in excluded
        and pd.api.types.is_numeric_dtype(paired[column])
    ]
    effect_rows: list[dict[str, Any]] = []
    for effect, (left, right) in contrasts.items():
        for metric in numeric_metrics:
            effect_rows.append(
                {
                    "effect": effect,
                    "metric": metric,
                    "value": float(indexed.at[left, metric] - indexed.at[right, metric]),
                }
            )
    for metric in numeric_metrics:
        effect_rows.append(
            {
                "effect": "interaction_D_minus_C_minus_B_minus_A",
                "metric": metric,
                "value": float(
                    (indexed.at["D", metric] - indexed.at["C", metric])
                    - (indexed.at["B", metric] - indexed.at["A", metric])
                ),
            }
        )
    return overall, paired, pd.DataFrame(effect_rows), cohort


def _stage_specs(artifacts: ExperimentArtifacts) -> dict[str, dict[str, Any]]:
    return {
        "experiment_b": {
            "nodes": artifacts.experiment_b_nodes,
            "clusters": artifacts.experiment_b_clusters,
            "text": "raw_attribute",
            "boundaries": [],
        },
        "experiment_d_aspect": {
            "nodes": artifacts.experiment_d_aspect_nodes,
            "clusters": artifacts.experiment_d_aspect_clusters,
            "text": "raw_aspect",
            "boundaries": [],
        },
        "experiment_d_status": {
            "nodes": artifacts.experiment_d_status_nodes,
            "clusters": artifacts.experiment_d_status_clusters,
            "text": "raw_status",
            "boundaries": ["aspect_cluster_id"],
        },
    }


def _iter_boundaries(
    frame: pd.DataFrame, boundaries: list[str]
) -> list[tuple[tuple[Any, ...], pd.DataFrame]]:
    if not boundaries:
        return [((), frame)]
    key: str | list[str] = boundaries[0] if len(boundaries) == 1 else boundaries
    groups = []
    for value, group in frame.groupby(key, sort=True, observed=True, dropna=False):
        boundary = value if isinstance(value, tuple) else (value,)
        groups.append((boundary, group))
    return groups


def build_cluster_quality(
    artifacts: ExperimentArtifacts,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_frames: list[pd.DataFrame] = []
    cluster_frames: list[pd.DataFrame] = []
    for stage, spec in _stage_specs(artifacts).items():
        nodes = spec["nodes"].copy()
        nodes["silhouette"] = np.nan
        for _, group in _iter_boundaries(nodes, spec["boundaries"]):
            labels = group["cluster_id"]
            if labels.nunique() < 2 or labels.nunique() >= len(group):
                continue
            matrix = np.stack(group["embedding"].map(np.asarray))
            scores = silhouette_samples(matrix, labels.to_numpy(), metric="cosine")
            nodes.loc[group.index, "silhouette"] = scores
        node_quality = nodes[
            [*spec["boundaries"], spec["text"], "cluster_id", "source_row_count", "silhouette"]
        ].copy()
        node_quality.insert(0, "stage", stage)
        node_frames.append(node_quality)

        silhouette_by_cluster = (
            nodes.groupby("cluster_id", observed=True)["silhouette"]
            .agg(["mean", "min", "count"])
            .rename(
                columns={
                    "mean": "cluster_silhouette_mean",
                    "min": "cluster_silhouette_min",
                    "count": "silhouette_node_count",
                }
            )
            .reset_index()
        )
        quality = spec["clusters"].merge(
            silhouette_by_cluster, on="cluster_id", how="left", validate="one_to_one"
        )
        cluster_frames.append(quality)
    return pd.concat(node_frames, ignore_index=True), pd.concat(cluster_frames, ignore_index=True)


def build_cluster_stage_metrics(
    artifacts: ExperimentArtifacts,
    node_quality: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for stage, spec in _stage_specs(artifacts).items():
        nodes = spec["nodes"]
        clusters = spec["clusters"]
        normalized = nodes["canonical_label"].fillna(nodes[spec["text"]]).astype(str)
        changed = normalized.ne(nodes[spec["text"]].astype(str))
        changed_source_rows = int(nodes.loc[changed, "source_row_count"].sum())
        multi = clusters.loc[clusters["member_count"].gt(1)]
        stage_node_quality = node_quality.loc[node_quality["stage"].eq(stage)]
        rows.append(
            {
                "stage": stage,
                "unique_string_count": len(nodes),
                "cluster_count": len(clusters),
                "structural_reduction_count": len(nodes) - len(clusters),
                "structural_reduction_rate": 1.0 - len(clusters) / len(nodes),
                "multi_member_cluster_count": len(multi),
                "singleton_cluster_count": int(clusters["member_count"].eq(1).sum()),
                "source_observation_count": int(nodes["source_row_count"].sum()),
                "relabelled_source_observation_count": changed_source_rows,
                "relabelled_source_observation_rate": changed_source_rows
                / int(nodes["source_row_count"].sum()),
                "selected_cluster_count": int(
                    clusters["naming_status"].isin(["selected", "singleton_inherited"]).sum()
                ),
                "fallback_cluster_count": int(
                    (~clusters["naming_status"].isin(["selected", "singleton_inherited"])).sum()
                ),
                "mean_multi_member_diameter": (
                    float(multi["cluster_max_distance"].mean()) if len(multi) else np.nan
                ),
                "p95_multi_member_diameter": (
                    float(multi["cluster_max_distance"].quantile(0.95)) if len(multi) else np.nan
                ),
                "silhouette_node_count": int(stage_node_quality["silhouette"].notna().sum()),
                "silhouette_node_coverage": float(stage_node_quality["silhouette"].notna().mean()),
                "mean_node_silhouette": float(stage_node_quality["silhouette"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_nearest_cluster_pairs(
    artifacts: ExperimentArtifacts,
    limit_per_stage: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stage, spec in _stage_specs(artifacts).items():
        candidates: list[dict[str, Any]] = []
        clusters = spec["clusters"]
        for boundary, group in _iter_boundaries(clusters, spec["boundaries"]):
            if len(group) < 2:
                continue
            matrix = np.stack(group["centroid_embedding"].map(np.asarray))
            distances = cosine_distances(matrix)
            left_positions, right_positions = np.triu_indices(len(group), k=1)
            for left, right in zip(left_positions, right_positions, strict=True):
                left_row = group.iloc[left]
                right_row = group.iloc[right]
                candidate = {
                    "stage": stage,
                    "left_cluster_id": left_row["cluster_id"],
                    "right_cluster_id": right_row["cluster_id"],
                    "left_label": left_row["canonical_label"]
                    if pd.notna(left_row["canonical_label"])
                    else left_row["medoid_label"],
                    "right_label": right_row["canonical_label"]
                    if pd.notna(right_row["canonical_label"])
                    else right_row["medoid_label"],
                    "centroid_cosine_distance": float(distances[left, right]),
                    "boundary_json": json.dumps(boundary, ensure_ascii=False),
                }
                candidates.append(candidate)
        candidates.sort(
            key=lambda row: (
                row["centroid_cosine_distance"],
                row["left_cluster_id"],
                row["right_cluster_id"],
            )
        )
        rows.extend(candidates[:limit_per_stage])
    return pd.DataFrame(rows)


def build_risky_clusters(cluster_quality: pd.DataFrame, limit_per_stage: int) -> pd.DataFrame:
    candidates = cluster_quality.loc[cluster_quality["member_count"].gt(1)].copy()
    candidates["risk_score"] = (
        candidates["cluster_max_distance"]
        + (1.0 - candidates["cluster_silhouette_mean"].fillna(0.0)) / 2.0
    )
    return (
        candidates.sort_values(
            ["stage", "risk_score", "cluster_max_distance", "cluster_id"],
            ascending=[True, False, False, True],
            kind="stable",
        )
        .groupby("stage", observed=True, sort=True)
        .head(limit_per_stage)
        .reset_index(drop=True)
    )


def _stable_unique(values: pd.Series) -> list[str]:
    return list(dict.fromkeys(map(str, values)))


def _structured_labels(frame: pd.DataFrame) -> pd.Series:
    status = frame["status"].astype("string")
    return frame["aspect"].astype(str) + status.map(
        lambda value: "" if pd.isna(value) else f" > {value}"
    )


def _contains_named_results(value: str) -> bool:
    decoded = json.loads(str(value))
    return bool(decoded) and all(isinstance(item, str) and item.strip() for item in decoded)


def _sample_task_rows(
    candidates: pd.DataFrame,
    *,
    task_count: int,
    random_seed: int,
    task_label: str,
) -> pd.DataFrame:
    if len(candidates) < task_count:
        raise ValueError(
            f"Need {task_count} eligible {task_label} tasks; found {len(candidates)}."
        )
    generator = np.random.default_rng(random_seed)
    positions = generator.choice(len(candidates), size=task_count, replace=False)
    return candidates.iloc[positions].reset_index(drop=True)


def _build_user_evaluation_task_pool(artifacts: ExperimentArtifacts) -> pd.DataFrame:
    """Build the complete eligible review-task pool without display metadata."""
    frames = {
        "result_a": artifacts.experiment_a.assign(_display=artifacts.experiment_a["attribute"]),
        "result_b": artifacts.experiment_b.assign(_display=artifacts.experiment_b["attribute"]),
        "result_c": artifacts.experiment_c.assign(
            _display=_structured_labels(artifacts.experiment_c)
        ),
        "result_d": artifacts.experiment_d.assign(
            _display=_structured_labels(artifacts.experiment_d)
        ),
    }
    common_ids = sorted(set.intersection(*(set(frame["review_idx"]) for frame in frames.values())))
    tasks = pd.DataFrame({"review_idx": common_ids})
    for column, frame in frames.items():
        grouped = (
            frame.loc[frame["review_idx"].isin(common_ids)]
            .sort_values("idx", kind="stable")
            .groupby("review_idx", observed=True)["_display"]
            .agg(_stable_unique)
        )
        tasks[column] = (
            tasks["review_idx"]
            .map(grouped)
            .map(
                lambda values: json.dumps(
                    list(map(str, values)), ensure_ascii=False, separators=(",", ":")
                )
            )
        )
    if tasks.isna().any().any() or len(tasks) != len(common_ids):
        raise AssertionError("Full-cohort user evaluation task construction failed.")
    nonempty = pd.Series(True, index=tasks.index)
    for column in frames:
        nonempty &= tasks[column].map(_contains_named_results)
    return tasks.loc[nonempty].reset_index(drop=True)


def build_user_evaluation_tasks(
    artifacts: ExperimentArtifacts,
    *,
    task_count: int,
    random_seed: int,
) -> pd.DataFrame:
    """Sample exactly N reproducible reviews with non-empty A-D results."""
    candidates = _build_user_evaluation_task_pool(artifacts)
    return _sample_task_rows(
        candidates,
        task_count=task_count,
        random_seed=random_seed,
        task_label="review",
    )


def _member_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(member) for member in value if str(member).strip())


def build_cluster_evaluation_tasks(
    cluster_quality: pd.DataFrame,
    *,
    task_count: int,
    random_seed: int,
) -> pd.DataFrame:
    """Sample exactly N reproducible, named, multi-member cluster tasks."""
    aspect_labels = {
        str(row.cluster_id): str(
            row.canonical_label if pd.notna(row.canonical_label) else row.medoid_label
        )
        for row in cluster_quality.loc[
            cluster_quality["stage"].astype(str).eq("experiment_d_aspect")
        ].itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    ordered = cluster_quality.sort_values(["stage", "cluster_id"], kind="stable")
    for row in ordered.itertuples(index=False):
        members = _member_strings(row.member_expressions)
        canonical_label = row.canonical_label if pd.notna(row.canonical_label) else None
        if int(row.member_count) <= 1 or len(members) <= 1 or canonical_label is None:
            continue
        canonical_text = str(canonical_label).strip()
        if not canonical_text:
            continue
        parent_aspect_label = None
        aspect_cluster_id = getattr(row, "aspect_cluster_id", None)
        if pd.notna(aspect_cluster_id):
            parent_aspect_label = aspect_labels.get(str(aspect_cluster_id))
        stage = str(row.stage)
        cluster_id = str(row.cluster_id)
        rows.append(
            {
                "cluster_validation_id": f"{stage}:{cluster_id}",
                "stage": stage,
                "cluster_id": cluster_id,
                "parent_aspect_label": parent_aspect_label,
                "canonical_label": canonical_text,
                "members": json.dumps(
                    list(members), ensure_ascii=False, separators=(",", ":")
                ),
            }
        )
    candidates = pd.DataFrame(
        rows,
        columns=[
            "cluster_validation_id",
            "stage",
            "cluster_id",
            "parent_aspect_label",
            "canonical_label",
            "members",
        ],
    )
    return _sample_task_rows(
        candidates,
        task_count=task_count,
        random_seed=random_seed,
        task_label="cluster",
    )


def build_review_normalization_changes(artifacts: ExperimentArtifacts) -> pd.DataFrame:
    tasks = _build_user_evaluation_task_pool(artifacts)
    tasks["A_to_B_changed"] = [
        tuple(json.loads(left)) != tuple(json.loads(right))
        for left, right in zip(tasks["result_a"], tasks["result_b"], strict=True)
    ]
    tasks["C_to_D_changed"] = [
        tuple(json.loads(left)) != tuple(json.loads(right))
        for left, right in zip(tasks["result_c"], tasks["result_d"], strict=True)
    ]
    return tasks[["review_idx", "A_to_B_changed", "C_to_D_changed"]]


def run_automatic_evaluation(
    representative: pd.DataFrame,
    opinion: pd.DataFrame,
    artifacts: ExperimentArtifacts,
    config: dict[str, Any],
) -> EvaluationArtifacts:
    integrity = evaluate_integrity(representative, opinion, artifacts, config)
    metrics, paired_metrics, factorial_effects, cohort = build_result_metrics(artifacts)
    node_quality, cluster_quality = build_cluster_quality(artifacts)
    stage_metrics = build_cluster_stage_metrics(artifacts, node_quality)
    evaluation_config = config["evaluation"]
    nearest_pairs = build_nearest_cluster_pairs(
        artifacts, int(evaluation_config["nearest_cluster_pair_limit"])
    )
    risky = build_risky_clusters(
        cluster_quality, int(evaluation_config["risky_cluster_limit_per_stage"])
    )
    user_evaluation_config = evaluation_config["user_evaluation"]
    task_count = int(user_evaluation_config["task_count"])
    random_seed = int(user_evaluation_config["random_seed"])
    review_task_pool = _build_user_evaluation_task_pool(artifacts)
    user_tasks = build_user_evaluation_tasks(
        artifacts,
        task_count=task_count,
        random_seed=random_seed,
    )
    cluster_tasks = build_cluster_evaluation_tasks(
        cluster_quality,
        task_count=task_count,
        random_seed=random_seed,
    )
    changes = build_review_normalization_changes(artifacts)
    tables = {
        "data_integrity_checks": integrity,
        "experiment_metrics": metrics,
        "paired_experiment_metrics": paired_metrics,
        "factorial_effects": factorial_effects,
        "review_cohort_summary": cohort,
        "cluster_stage_metrics": stage_metrics,
        "cluster_node_quality": node_quality,
        "cluster_quality": cluster_quality,
        "nearest_cluster_pairs": nearest_pairs,
        "risky_clusters": risky,
        "user_evaluation_tasks": user_tasks,
        "cluster_evaluation_tasks": cluster_tasks,
        "review_normalization_changes": changes,
    }
    summary = {
        "scope": "full_input_datasets",
        "sampling_used": False,
        "integrity_check_count": len(integrity),
        "integrity_checks_passed": int(integrity["passed"].sum()),
        "representative_row_count": len(representative),
        "opinion_row_count": len(opinion),
        "eligible_opinion_row_count": len(artifacts.eligible_opinion),
        "common_review_count": len(review_task_pool),
        "user_evaluation_task_count": len(user_tasks),
        "cluster_evaluation_task_count": len(cluster_tasks),
        "user_evaluation_random_seed": random_seed,
        "user_evaluation_sampling_used": True,
        "user_evaluation_completed": False,
    }
    return EvaluationArtifacts(tables=tables, summary=summary)
