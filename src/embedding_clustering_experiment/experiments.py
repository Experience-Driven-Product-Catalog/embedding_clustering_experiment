"""Build experiments A-D without copying product or review-table columns."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .clustering import (
    build_unique_nodes,
    cluster_unique_nodes,
    public_nodes,
)
from .data import (
    FORBIDDEN_INTEGRATED_COLUMNS,
    OPINION_COLUMNS,
    REPRESENTATIVE_COLUMNS,
    assert_source_columns_preserved,
    eligible_opinion_rows,
)
from .encoder import SentenceEncoder

NORMALIZATION_METADATA_COLUMNS = (
    "embedding_model_id",
    "normalization_version",
    "normalization_run_id",
    "normalization_config_sha256",
)
EXPERIMENT_B_LINEAGE_COLUMNS = (
    "attribute_cluster_id",
    "attribute_naming_status",
    "attribute_mapping_applied",
    "attribute_mapping_distance",
    *NORMALIZATION_METADATA_COLUMNS,
)
EXPERIMENT_D_LINEAGE_COLUMNS = (
    "aspect_cluster_id",
    "status_cluster_id",
    "aspect_naming_status",
    "status_naming_status",
    "aspect_mapping_applied",
    "status_mapping_applied",
    "aspect_mapping_distance",
    "status_mapping_distance",
    *NORMALIZATION_METADATA_COLUMNS,
)


@dataclass(frozen=True)
class ExperimentArtifacts:
    experiment_a: pd.DataFrame
    experiment_b: pd.DataFrame
    experiment_c: pd.DataFrame
    experiment_d: pd.DataFrame
    experiment_a_inventory: pd.DataFrame
    experiment_b_nodes: pd.DataFrame
    experiment_b_clusters: pd.DataFrame
    experiment_c_inventory: pd.DataFrame
    experiment_d_aspect_nodes: pd.DataFrame
    experiment_d_aspect_clusters: pd.DataFrame
    experiment_d_status_nodes: pd.DataFrame
    experiment_d_status_clusters: pd.DataFrame
    eligible_opinion: pd.DataFrame

    def result_tables(self) -> dict[str, pd.DataFrame]:
        return {
            "experiment_a": self.experiment_a,
            "experiment_b": self.experiment_b,
            "experiment_c": self.experiment_c,
            "experiment_d": self.experiment_d,
        }

    def diagnostic_tables(self) -> dict[str, pd.DataFrame]:
        return {
            "experiment_a_inventory": self.experiment_a_inventory,
            "experiment_b_nodes": self.experiment_b_nodes,
            "experiment_b_clusters": self.experiment_b_clusters,
            "experiment_c_inventory": self.experiment_c_inventory,
            "experiment_d_aspect_nodes": self.experiment_d_aspect_nodes,
            "experiment_d_aspect_clusters": self.experiment_d_aspect_clusters,
            "experiment_d_status_nodes": self.experiment_d_status_nodes,
            "experiment_d_status_clusters": self.experiment_d_status_clusters,
        }


def _label_inventory(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(columns, sort=True, observed=True, dropna=False)
        .agg(
            source_row_count=("idx", "size"),
            unique_review_count=("review_idx", "nunique"),
        )
        .reset_index()
    )


def _merge_assignments(
    frame: pd.DataFrame,
    nodes: pd.DataFrame,
    keys: list[str],
    assignment_columns: list[str],
) -> pd.DataFrame:
    return frame.merge(
        nodes[[*keys, *assignment_columns]],
        on=keys,
        how="left",
        validate="many_to_one",
        sort=False,
    )


def build_normalization_metadata(
    config: dict[str, Any],
    encoder: SentenceEncoder,
    normalization_run_id: str,
) -> dict[str, str]:
    """Return row-level provenance shared by every normalized result."""
    run_id = str(normalization_run_id).strip()
    if not run_id:
        raise ValueError("normalization_run_id must be a non-empty string.")
    configured_version = config["project"].get("normalization_version")
    if not isinstance(configured_version, str) or not configured_version.strip():
        raise ValueError("project.normalization_version must be a non-empty string.")
    version = configured_version.strip()
    model_id = str(
        getattr(encoder, "model_id", None) or config["embedding"]["model_id"]
    ).strip()
    if not model_id:
        raise ValueError("embedding.model_id must be a non-empty string.")
    fingerprint_payload = {
        "normalization_version": version,
        "filters": config["filters"],
        "embedding": config["embedding"],
        "clustering": config["clustering"],
        "canonical_label": config["canonical_label"],
    }
    serialized = json.dumps(
        fingerprint_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "embedding_model_id": model_id,
        "normalization_version": version,
        "normalization_run_id": run_id,
        "normalization_config_sha256": hashlib.sha256(serialized).hexdigest(),
    }


def _append_normalization_metadata(
    frame: pd.DataFrame,
    metadata: dict[str, str],
) -> None:
    for column in NORMALIZATION_METADATA_COLUMNS:
        frame[column] = pd.Series(metadata[column], index=frame.index, dtype="string")


def build_experiments(
    representative: pd.DataFrame,
    opinion: pd.DataFrame,
    encoder: SentenceEncoder,
    config: dict[str, Any],
    logger: logging.Logger,
    *,
    normalization_run_id: str,
) -> ExperimentArtifacts:
    """Run the unchanged A-D algorithm over one category-wide dataset."""
    normalization_metadata = build_normalization_metadata(
        config,
        encoder,
        normalization_run_id,
    )
    excluded_aspects = set(map(str, config["filters"]["excluded_raw_aspects"]))
    eligible_opinion = eligible_opinion_rows(opinion, excluded_aspects)
    if eligible_opinion.empty:
        raise ValueError("No opinion-unit rows remain after raw_aspect exclusion.")
    logger.info(
        "Opinion filter kept=%d excluded=%d labels=%s",
        len(eligible_opinion),
        len(opinion) - len(eligible_opinion),
        sorted(excluded_aspects),
    )

    experiment_a = representative.copy()
    experiment_a["attribute"] = experiment_a["raw_attribute"].astype("string")

    clustering = config["clustering"]
    canonical = config["canonical_label"]

    logger.info("Experiment B: clustering %d rows without category boundaries", len(representative))
    b_node_input = build_unique_nodes(representative, "raw_attribute")
    b_clustered = cluster_unique_nodes(
        b_node_input,
        "raw_attribute",
        encoder,
        "experiment_b_attribute",
        "B",
        "aspect",
        clustering["experiment_b"],
        clustering,
        canonical,
    )
    b_rows = _merge_assignments(
        representative,
        b_clustered.nodes,
        ["raw_attribute"],
        [
            "cluster_id",
            "canonical_label",
            "naming_status",
            "mapping_applied",
            "mapping_distance",
        ],
    )
    experiment_b = b_rows.loc[:, list(REPRESENTATIVE_COLUMNS)].copy()
    experiment_b["attribute"] = (
        b_rows["canonical_label"].fillna(b_rows["raw_attribute"]).astype("string")
    )
    experiment_b["attribute_cluster_id"] = b_rows["cluster_id"].astype("string")
    experiment_b["attribute_naming_status"] = b_rows["naming_status"].astype("string")
    experiment_b["attribute_mapping_applied"] = b_rows["mapping_applied"].astype("boolean")
    experiment_b["attribute_mapping_distance"] = b_rows["mapping_distance"].astype("float64")
    _append_normalization_metadata(experiment_b, normalization_metadata)

    experiment_c = eligible_opinion.copy()
    experiment_c["aspect"] = experiment_c["raw_aspect"].astype("string")
    experiment_c["status"] = experiment_c["raw_status"].astype("string")

    logger.info(
        "Experiment D aspect: clustering %d eligible rows without category boundaries",
        len(eligible_opinion),
    )
    d_aspect_input = build_unique_nodes(eligible_opinion, "raw_aspect")
    d_aspect_clustered = cluster_unique_nodes(
        d_aspect_input,
        "raw_aspect",
        encoder,
        "experiment_d_aspect",
        "D-A",
        "aspect",
        clustering["experiment_d"]["aspect"],
        clustering,
        canonical,
    )
    d_rows = _merge_assignments(
        eligible_opinion,
        d_aspect_clustered.nodes,
        ["raw_aspect"],
        [
            "cluster_id",
            "canonical_label",
            "naming_status",
            "mapping_applied",
            "mapping_distance",
        ],
    ).rename(
        columns={
            "cluster_id": "aspect_cluster_id",
            "canonical_label": "aspect_canonical_label",
            "naming_status": "aspect_naming_status",
            "mapping_applied": "aspect_mapping_applied",
            "mapping_distance": "aspect_mapping_distance",
        }
    )
    d_rows["aspect"] = (
        d_rows["aspect_canonical_label"].fillna(d_rows["raw_aspect"]).astype("string")
    )

    forbidden_by_aspect: dict[tuple[Any, ...], set[str]] = {}
    for cluster_id, group in d_aspect_clustered.nodes.groupby(
        "cluster_id", sort=True, observed=True
    ):
        labels = set(group["raw_aspect"].astype(str))
        labels.update(group["canonical_label"].dropna().astype(str))
        forbidden_by_aspect[(cluster_id,)] = labels

    status_rows = d_rows.loc[d_rows["raw_status"].notna()].copy()
    logger.info(
        "Experiment D status: clustering %d non-null rows inside aspect clusters",
        len(status_rows),
    )
    if status_rows.empty:
        raise ValueError("No non-null status values are available for Experiment D.")
    d_status_input = build_unique_nodes(
        status_rows,
        "raw_status",
        boundary_columns=["aspect_cluster_id"],
    )
    d_status_clustered = cluster_unique_nodes(
        d_status_input,
        "raw_status",
        encoder,
        "experiment_d_status",
        "D-S",
        "status",
        clustering["experiment_d"]["status"],
        clustering,
        canonical,
        boundary_columns=["aspect_cluster_id"],
        forbidden_by_boundary=forbidden_by_aspect,
    )
    d_rows = _merge_assignments(
        d_rows,
        d_status_clustered.nodes,
        ["aspect_cluster_id", "raw_status"],
        [
            "cluster_id",
            "canonical_label",
            "naming_status",
            "mapping_applied",
            "mapping_distance",
        ],
    ).rename(
        columns={
            "cluster_id": "status_cluster_id",
            "canonical_label": "status_canonical_label",
            "naming_status": "status_naming_status",
            "mapping_applied": "status_mapping_applied",
            "mapping_distance": "status_mapping_distance",
        }
    )
    d_rows["status"] = (
        d_rows["status_canonical_label"].fillna(d_rows["raw_status"]).astype("string")
    )
    experiment_d = d_rows.loc[:, [*OPINION_COLUMNS, "aspect", "status"]].copy()
    experiment_d["aspect_cluster_id"] = d_rows["aspect_cluster_id"].astype("string")
    experiment_d["status_cluster_id"] = d_rows["status_cluster_id"].astype("string")
    experiment_d["aspect_naming_status"] = d_rows["aspect_naming_status"].astype("string")
    experiment_d["status_naming_status"] = d_rows["status_naming_status"].astype("string")
    experiment_d["aspect_mapping_applied"] = d_rows["aspect_mapping_applied"].astype("boolean")
    experiment_d["status_mapping_applied"] = d_rows["status_mapping_applied"].astype("boolean")
    experiment_d["aspect_mapping_distance"] = d_rows["aspect_mapping_distance"].astype(
        "float64"
    )
    experiment_d["status_mapping_distance"] = d_rows["status_mapping_distance"].astype(
        "float64"
    )
    _append_normalization_metadata(experiment_d, normalization_metadata)

    artifacts = ExperimentArtifacts(
        experiment_a=experiment_a,
        experiment_b=experiment_b,
        experiment_c=experiment_c,
        experiment_d=experiment_d,
        experiment_a_inventory=_label_inventory(representative, ["raw_attribute"]),
        experiment_b_nodes=public_nodes(b_clustered.nodes),
        experiment_b_clusters=b_clustered.clusters,
        experiment_c_inventory=_label_inventory(eligible_opinion, ["raw_aspect", "raw_status"]),
        experiment_d_aspect_nodes=public_nodes(d_aspect_clustered.nodes),
        experiment_d_aspect_clusters=d_aspect_clustered.clusters,
        experiment_d_status_nodes=public_nodes(d_status_clustered.nodes),
        experiment_d_status_clusters=d_status_clustered.clusters,
        eligible_opinion=eligible_opinion,
    )
    validate_experiment_outputs(representative, eligible_opinion, artifacts)
    return artifacts


def _nullable_equal(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.eq(right).fillna(False) | (left.isna() & right.isna())


def _validate_mapping_lineage(
    frame: pd.DataFrame,
    prefix: str,
    raw_column: str,
    final_column: str,
) -> None:
    cluster_column = f"{prefix}_cluster_id"
    naming_column = f"{prefix}_naming_status"
    applied_column = f"{prefix}_mapping_applied"
    distance_column = f"{prefix}_mapping_distance"
    raw_is_null = frame[raw_column].isna()
    for column in (cluster_column, naming_column, applied_column, distance_column):
        if not frame[column].isna().equals(raw_is_null):
            raise AssertionError(
                f"{column} nullability must match {raw_column} nullability."
            )

    present = ~raw_is_null
    expected_applied = ~_nullable_equal(
        frame.loc[present, raw_column],
        frame.loc[present, final_column],
    )
    actual_applied = frame.loc[present, applied_column].astype("bool")
    if not actual_applied.equals(expected_applied.astype("bool")):
        raise AssertionError(
            f"{applied_column} must indicate whether {raw_column} changed to {final_column}."
        )

    distances = frame.loc[present, distance_column]
    if not distances.between(0.0, 2.0, inclusive="both").all():
        raise AssertionError(f"{distance_column} must be a cosine distance in [0, 2].")
    unchanged_distances = distances.loc[~actual_applied]
    if not np.isclose(unchanged_distances.to_numpy(), 0.0, atol=1e-6).all():
        raise AssertionError(f"{distance_column} must be zero when no mapping was applied.")


def _validate_lineage_matches_nodes(
    frame: pd.DataFrame,
    nodes: pd.DataFrame,
    keys: list[str],
    prefix: str,
) -> None:
    fields = ("cluster_id", "naming_status", "mapping_applied", "mapping_distance")
    expected_names = {field: f"_expected_{field}" for field in fields}
    present = frame.loc[frame[keys].notna().all(axis=1)]
    comparison = present.merge(
        nodes[[*keys, *fields]].rename(columns=expected_names),
        on=keys,
        how="left",
        validate="many_to_one",
        sort=False,
    )
    for field in fields[:-1]:
        if not _nullable_equal(
            comparison[f"{prefix}_{field}"],
            comparison[expected_names[field]],
        ).all():
            raise AssertionError(f"{prefix}_{field} differs from its clustering node lineage.")
    if not np.isclose(
        comparison[f"{prefix}_mapping_distance"].to_numpy(dtype=float),
        comparison[expected_names["mapping_distance"]].to_numpy(dtype=float),
        atol=1e-8,
    ).all():
        raise AssertionError(
            f"{prefix}_mapping_distance differs from its clustering node lineage."
        )


def _validate_normalization_metadata(artifacts: ExperimentArtifacts) -> None:
    normalized_tables = {
        "experiment_b": artifacts.experiment_b,
        "experiment_d": artifacts.experiment_d,
    }
    for name, frame in normalized_tables.items():
        for column in NORMALIZATION_METADATA_COLUMNS:
            if frame[column].isna().any() or frame[column].nunique() != 1:
                raise AssertionError(f"{name}.{column} must contain one non-null run value.")
        if not frame["normalization_config_sha256"].str.fullmatch(r"[0-9a-f]{64}").all():
            raise AssertionError(
                f"{name}.normalization_config_sha256 is not a lowercase SHA-256 digest."
            )

    for column in NORMALIZATION_METADATA_COLUMNS:
        if artifacts.experiment_b[column].iloc[0] != artifacts.experiment_d[column].iloc[0]:
            raise AssertionError(f"B and D must share the same {column} value.")


def validate_experiment_outputs(
    representative: pd.DataFrame,
    eligible_opinion: pd.DataFrame,
    artifacts: ExperimentArtifacts,
) -> None:
    expected_columns = {
        "experiment_a": [*REPRESENTATIVE_COLUMNS, "attribute"],
        "experiment_b": [
            *REPRESENTATIVE_COLUMNS,
            "attribute",
            *EXPERIMENT_B_LINEAGE_COLUMNS,
        ],
        "experiment_c": [*OPINION_COLUMNS, "aspect", "status"],
        "experiment_d": [
            *OPINION_COLUMNS,
            "aspect",
            "status",
            *EXPERIMENT_D_LINEAGE_COLUMNS,
        ],
    }
    expected_rows = {
        "experiment_a": len(representative),
        "experiment_b": len(representative),
        "experiment_c": len(eligible_opinion),
        "experiment_d": len(eligible_opinion),
    }
    for name, frame in artifacts.result_tables().items():
        if list(frame.columns) != expected_columns[name]:
            raise AssertionError(f"{name} columns differ from contract: {list(frame.columns)}")
        if len(frame) != expected_rows[name]:
            raise AssertionError(f"{name} expected {expected_rows[name]} rows; found {len(frame)}.")
        if not frame["idx"].is_unique:
            raise AssertionError(f"{name}.idx is not unique.")
        forbidden = sorted(FORBIDDEN_INTEGRATED_COLUMNS & set(frame.columns))
        if forbidden:
            raise AssertionError(f"{name} copied integrated-table columns: {forbidden}")

    for name in ("experiment_a", "experiment_b"):
        assert_source_columns_preserved(
            representative,
            artifacts.result_tables()[name],
            REPRESENTATIVE_COLUMNS,
        )
        if artifacts.result_tables()[name]["attribute"].isna().any():
            raise AssertionError(f"{name}.attribute contains null values.")
    for name in ("experiment_c", "experiment_d"):
        result = artifacts.result_tables()[name]
        assert_source_columns_preserved(eligible_opinion, result, OPINION_COLUMNS)
        if result["aspect"].isna().any():
            raise AssertionError(f"{name}.aspect contains null values.")
        if not result["status"].isna().equals(result["raw_status"].isna()):
            raise AssertionError(f"{name} did not preserve raw_status nullability.")

    if not _nullable_equal(
        artifacts.experiment_a["attribute"], artifacts.experiment_a["raw_attribute"]
    ).all():
        raise AssertionError("Experiment A must be the exact raw_attribute baseline.")
    if (
        not _nullable_equal(
            artifacts.experiment_c["aspect"], artifacts.experiment_c["raw_aspect"]
        ).all()
        or not _nullable_equal(
            artifacts.experiment_c["status"], artifacts.experiment_c["raw_status"]
        ).all()
    ):
        raise AssertionError("Experiment C must be the exact raw aspect/status baseline.")

    _validate_mapping_lineage(
        artifacts.experiment_b,
        "attribute",
        "raw_attribute",
        "attribute",
    )
    _validate_mapping_lineage(
        artifacts.experiment_d,
        "aspect",
        "raw_aspect",
        "aspect",
    )
    _validate_mapping_lineage(
        artifacts.experiment_d,
        "status",
        "raw_status",
        "status",
    )
    _validate_lineage_matches_nodes(
        artifacts.experiment_b,
        artifacts.experiment_b_nodes,
        ["raw_attribute"],
        "attribute",
    )
    _validate_lineage_matches_nodes(
        artifacts.experiment_d,
        artifacts.experiment_d_aspect_nodes,
        ["raw_aspect"],
        "aspect",
    )
    _validate_lineage_matches_nodes(
        artifacts.experiment_d,
        artifacts.experiment_d_status_nodes,
        ["aspect_cluster_id", "raw_status"],
        "status",
    )
    _validate_normalization_metadata(artifacts)

    if "product_category" in artifacts.experiment_b_nodes.columns:
        raise AssertionError("Experiment B must not contain a product_category boundary.")
    if "product_category" in artifacts.experiment_d_aspect_nodes.columns:
        raise AssertionError("Experiment D aspect must not contain a product_category boundary.")
    status_boundaries = artifacts.experiment_d_status_nodes.groupby("cluster_id", observed=True)[
        "aspect_cluster_id"
    ].nunique()
    if status_boundaries.gt(1).any():
        raise AssertionError("A D-status cluster crossed its aspect_cluster_id boundary.")
