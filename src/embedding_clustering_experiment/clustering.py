"""Unique-string complete-linkage clustering and observed-label selection."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

from .encoder import SentenceEncoder

INTERNAL_NODE_COLUMNS = {"_review_ids"}


@dataclass(frozen=True)
class ClusterOutput:
    nodes: pd.DataFrame
    clusters: pd.DataFrame


def _sorted_unique_tuple(values: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(sorted(set(values)))


def build_unique_nodes(
    frame: pd.DataFrame,
    text_column: str,
    boundary_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Build one unweighted clustering node per boundary and observed string."""
    grouping = [*boundary_columns, text_column]
    if frame[text_column].isna().any():
        raise ValueError(f"{text_column} must be non-null before node construction.")
    nodes = (
        frame.groupby(grouping, sort=True, observed=True, dropna=False)
        .agg(
            source_row_count=("idx", "size"),
            _review_ids=("review_idx", _sorted_unique_tuple),
        )
        .reset_index()
    )
    nodes["source_row_count"] = nodes["source_row_count"].astype("int64")
    nodes["unique_review_count"] = nodes["_review_ids"].map(len).astype("int64")
    if nodes.duplicated(grouping).any():
        raise AssertionError(f"Unique-node construction failed for {grouping}.")
    return nodes


def public_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    return nodes.drop(columns=[column for column in INTERNAL_NODE_COLUMNS if column in nodes])


def _candidate_limit(size: int, config: Mapping[str, Any]) -> int:
    candidate_config = config["candidate_count"]
    if size == 1:
        return int(candidate_config["size_1"])
    if size == 2:
        return int(candidate_config["size_2"])
    if size < 10:
        return int(candidate_config["size_3_to_9"])
    return int(candidate_config["size_10_plus"])


def _role_candidate_allowed(
    expression: str,
    role: str,
    canonical_config: Mapping[str, Any],
    forbidden_labels: set[str],
) -> bool:
    filters = canonical_config["role_filters"]
    if expression in set(map(str, filters[f"{role}_disallowed_exact"])):
        return False
    return not (
        role == "status"
        and bool(filters["exclude_status_matching_aspect"])
        and expression in forbidden_labels
    )


def select_canonical_label(
    member_nodes: pd.DataFrame,
    distances: np.ndarray,
    text_column: str,
    role: str,
    canonical_config: Mapping[str, Any],
    naming_max_distance: float,
    forbidden_labels: set[str],
) -> dict[str, Any]:
    """Select an observed expression after medoid-first candidate filtering."""
    size = len(member_nodes)
    expressions = member_nodes[text_column].astype(str).tolist()
    if size == 1:
        average_distances = np.array([0.0], dtype=float)
        max_distances = np.array([0.0], dtype=float)
    else:
        average_distances = distances.sum(axis=1) / (size - 1)
        max_distances = distances.max(axis=1)
    medoid_position = min(
        range(size), key=lambda position: (average_distances[position], expressions[position])
    )
    medoid_label = expressions[medoid_position]
    cluster_max_distance = float(distances.max()) if size > 1 else 0.0

    if size == 1:
        details = [
            {
                "expression": expressions[0],
                "average_distance": 0.0,
                "max_distance": 0.0,
                "unique_review_count": int(member_nodes.iloc[0]["unique_review_count"]),
                "central_candidate": True,
                "role_filter_passed": True,
                "centrality": 1.0,
                "frequency_score": 1.0,
                "combined_score": 1.0,
            }
        ]
        return {
            "canonical_label": expressions[0],
            "canonical_position": 0,
            "medoid_label": medoid_label,
            "naming_status": "singleton_inherited",
            "cluster_max_distance": cluster_max_distance,
            "member_details": details,
        }

    candidate_count = min(size, _candidate_limit(size, canonical_config))
    ordered_positions = sorted(
        range(size), key=lambda position: (average_distances[position], expressions[position])
    )
    central_positions = set(ordered_positions[:candidate_count])
    eligible_positions = [
        position
        for position in ordered_positions[:candidate_count]
        if _role_candidate_allowed(expressions[position], role, canonical_config, forbidden_labels)
    ]

    scores: dict[int, tuple[float, float, float]] = {}
    if eligible_positions:
        candidate_distances = np.array(
            [average_distances[position] for position in eligible_positions], dtype=float
        )
        centralities = np.clip(1.0 - candidate_distances / 2.0, 0.0, 1.0)
        review_logs = np.array(
            [
                math.log1p(int(member_nodes.iloc[position]["unique_review_count"]))
                for position in eligible_positions
            ],
            dtype=float,
        )
        frequency_scores = review_logs / review_logs.max()
        alpha = float(canonical_config["centrality_weight_alpha"])
        combined = alpha * centralities + (1.0 - alpha) * frequency_scores
        scores = {
            position: (
                float(centralities[index]),
                float(frequency_scores[index]),
                float(combined[index]),
            )
            for index, position in enumerate(eligible_positions)
        }

    canonical_position: int | None = None
    if cluster_max_distance > naming_max_distance + 1e-12:
        naming_status = "cohesion_review_required"
    elif not eligible_positions:
        naming_status = "no_role_eligible_candidate"
    else:
        ranked = sorted(
            eligible_positions,
            key=lambda position: (
                -scores[position][2],
                average_distances[position],
                -int(member_nodes.iloc[position]["unique_review_count"]),
                expressions[position],
            ),
        )
        top_position = ranked[0]
        top_support = int(member_nodes.iloc[top_position]["unique_review_count"])
        score_margin = (
            scores[top_position][2] - scores[ranked[1]][2] if len(ranked) > 1 else float("inf")
        )
        if top_support < int(canonical_config["min_canonical_review_count"]):
            naming_status = "insufficient_review_support"
        elif score_margin < float(canonical_config["min_score_margin"]):
            naming_status = "ambiguous_score"
        else:
            canonical_position = top_position
            naming_status = "selected"

    details = []
    for position, expression in enumerate(expressions):
        score = scores.get(position)
        details.append(
            {
                "expression": expression,
                "average_distance": round(float(average_distances[position]), 8),
                "max_distance": round(float(max_distances[position]), 8),
                "unique_review_count": int(member_nodes.iloc[position]["unique_review_count"]),
                "central_candidate": position in central_positions,
                "role_filter_passed": _role_candidate_allowed(
                    expression, role, canonical_config, forbidden_labels
                ),
                "centrality": round(score[0], 8) if score else None,
                "frequency_score": round(score[1], 8) if score else None,
                "combined_score": round(score[2], 8) if score else None,
            }
        )
    return {
        "canonical_label": (
            expressions[canonical_position] if canonical_position is not None else None
        ),
        "canonical_position": canonical_position,
        "medoid_label": medoid_label,
        "naming_status": naming_status,
        "cluster_max_distance": cluster_max_distance,
        "member_details": details,
    }


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0.0 else vector / norm


def _explicit_negation_core(text: str, config: Mapping[str, Any]) -> tuple[bool, str]:
    normalized = " ".join(text.split())
    for suffix in sorted(map(str, config["suffixes"]), key=len, reverse=True):
        if normalized.endswith(suffix):
            core = normalized[: -len(suffix)].strip()
            if core:
                return True, core
    tokens = normalized.split()
    negative_tokens = set(map(str, config["whole_word_tokens"]))
    for index, token in enumerate(tokens):
        if token in negative_tokens:
            core_tokens = tokens[:index] + tokens[index + 1 :]
            if core_tokens:
                return True, " ".join(core_tokens)
    return False, normalized


def status_constrained_distances(
    texts: Sequence[str],
    embeddings: np.ndarray,
    opposition_groups: Sequence[Mapping[str, Sequence[str]]],
    opposition_suffix_pairs: Sequence[Mapping[str, str]],
    explicit_negation_config: Mapping[str, Any],
) -> tuple[np.ndarray, int]:
    """Set configured opposite status pairs to cosine distance 2 before clustering."""
    distances = np.clip(cosine_distances(embeddings), 0.0, 2.0)
    applied_pairs: set[tuple[int, int]] = set()
    for group in opposition_groups:
        side_a = set(map(str, group["side_a"]))
        side_b = set(map(str, group["side_b"]))
        a_positions = [index for index, text in enumerate(texts) if text in side_a]
        b_positions = [index for index, text in enumerate(texts) if text in side_b]
        for left in a_positions:
            for right in b_positions:
                pair = tuple(sorted((left, right)))
                applied_pairs.add(pair)
                distances[left, right] = 2.0
                distances[right, left] = 2.0

    normalized_texts = [" ".join(str(text).split()) for text in texts]
    similarity_threshold = float(explicit_negation_config["core_similarity_threshold"])
    for left in range(len(normalized_texts)):
        for right in range(left + 1, len(normalized_texts)):
            left_text = normalized_texts[left]
            right_text = normalized_texts[right]
            suffix_opposites = False
            for pair in opposition_suffix_pairs:
                side_a_suffix = str(pair["side_a_suffix"])
                side_b_suffix = str(pair["side_b_suffix"])
                if left_text.endswith(side_a_suffix) and right_text.endswith(side_b_suffix):
                    left_core = left_text[: -len(side_a_suffix)].strip()
                    right_core = right_text[: -len(side_b_suffix)].strip()
                elif left_text.endswith(side_b_suffix) and right_text.endswith(side_a_suffix):
                    left_core = left_text[: -len(side_b_suffix)].strip()
                    right_core = right_text[: -len(side_a_suffix)].strip()
                else:
                    continue
                suffix_opposites = (
                    bool(left_core)
                    and bool(right_core)
                    and SequenceMatcher(None, left_core, right_core).ratio() >= similarity_threshold
                )
                if suffix_opposites:
                    break

            explicit_opposites = False
            if bool(explicit_negation_config["enabled"]):
                left_negative, left_core = _explicit_negation_core(
                    left_text, explicit_negation_config
                )
                right_negative, right_core = _explicit_negation_core(
                    right_text, explicit_negation_config
                )
                if left_negative != right_negative:
                    negative_core = left_core if left_negative else right_core
                    positive_text = right_text if left_negative else left_text
                    explicit_opposites = (
                        SequenceMatcher(None, negative_core, positive_text).ratio()
                        >= similarity_threshold
                    )
            if suffix_opposites or explicit_opposites:
                applied_pairs.add((left, right))
                distances[left, right] = 2.0
                distances[right, left] = 2.0
    np.fill_diagonal(distances, 0.0)
    return distances, len(applied_pairs)


def _group_nodes(
    nodes: pd.DataFrame, boundary_columns: Sequence[str]
) -> Iterable[tuple[tuple[Any, ...], pd.DataFrame]]:
    if not boundary_columns:
        yield (), nodes
        return
    group_key: str | list[str] = (
        boundary_columns[0] if len(boundary_columns) == 1 else list(boundary_columns)
    )
    for boundary_value, group in nodes.groupby(group_key, sort=True, observed=True, dropna=False):
        boundary_tuple = boundary_value if isinstance(boundary_value, tuple) else (boundary_value,)
        yield boundary_tuple, group


def cluster_unique_nodes(
    nodes: pd.DataFrame,
    text_column: str,
    encoder: SentenceEncoder,
    stage_name: str,
    cluster_prefix: str,
    role: str,
    stage_config: Mapping[str, Any],
    clustering_config: Mapping[str, Any],
    canonical_config: Mapping[str, Any],
    boundary_columns: Sequence[str] = (),
    forbidden_by_boundary: Mapping[tuple[Any, ...], set[str]] | None = None,
) -> ClusterOutput:
    if nodes.empty:
        raise ValueError(f"No unique nodes available for stage {stage_name}.")
    result_nodes = nodes.copy().reset_index(drop=True)
    texts = result_nodes[text_column].astype(str).tolist()
    embeddings = encoder.encode(texts)
    if embeddings.ndim != 2 or len(embeddings) != len(result_nodes):
        raise ValueError(f"Encoder returned an invalid shape for stage {stage_name}.")
    result_nodes["embedding"] = [vector.astype(np.float32) for vector in embeddings]
    result_nodes["cluster_id"] = pd.Series(index=result_nodes.index, dtype="object")
    result_nodes["canonical_label"] = pd.Series(index=result_nodes.index, dtype="object")
    result_nodes["naming_status"] = pd.Series(index=result_nodes.index, dtype="object")
    result_nodes["mapping_applied"] = pd.Series(False, index=result_nodes.index, dtype="bool")
    result_nodes["mapping_distance"] = pd.Series(
        np.nan, index=result_nodes.index, dtype="float64"
    )

    threshold = float(stage_config["distance_threshold"])
    naming_max_distance = float(stage_config["naming_max_distance"])
    tolerance = float(clustering_config["distance_tolerance"])
    cluster_rows: list[dict[str, Any]] = []
    cluster_counter = 0

    for boundary_tuple, group in _group_nodes(result_nodes, boundary_columns):
        group_indices = group.index.to_list()
        matrix = np.stack(group["embedding"].to_list())
        constraint_pair_count = 0
        if len(group) == 1:
            raw_labels = np.array([0], dtype=int)
        else:
            if role == "status":
                fit_matrix, constraint_pair_count = status_constrained_distances(
                    group[text_column].astype(str).tolist(),
                    matrix,
                    clustering_config.get("status_opposition_groups", []),
                    clustering_config.get("status_opposition_suffix_pairs", []),
                    clustering_config["status_explicit_negation"],
                )
                metric = "precomputed"
            else:
                fit_matrix = matrix
                metric = str(clustering_config["metric"])
            model = AgglomerativeClustering(
                n_clusters=None,
                metric=metric,
                linkage=str(clustering_config["linkage"]),
                distance_threshold=threshold,
                compute_full_tree=True,
                compute_distances=True,
            )
            raw_labels = model.fit_predict(fit_matrix)

        ordered_clusters: list[tuple[str, list[int]]] = []
        for raw_label in sorted(set(raw_labels.tolist())):
            local_positions = np.flatnonzero(raw_labels == raw_label).tolist()
            minimum_expression = min(
                str(group.iloc[position][text_column]) for position in local_positions
            )
            ordered_clusters.append((minimum_expression, local_positions))
        ordered_clusters.sort(key=lambda item: item[0])
        forbidden_labels = (
            forbidden_by_boundary.get(boundary_tuple, set())
            if forbidden_by_boundary is not None
            else set()
        )

        for _, local_positions in ordered_clusters:
            cluster_counter += 1
            cluster_id = f"{cluster_prefix}-{cluster_counter:06d}"
            member_indices = [group_indices[position] for position in local_positions]
            member_nodes = result_nodes.loc[member_indices].reset_index(drop=True)
            member_matrix = np.stack(member_nodes["embedding"].to_list())
            member_distances = np.clip(cosine_distances(member_matrix), 0.0, 2.0)
            label = select_canonical_label(
                member_nodes,
                member_distances,
                text_column,
                role,
                canonical_config,
                naming_max_distance,
                forbidden_labels,
            )
            if label["cluster_max_distance"] > threshold + tolerance:
                raise AssertionError(
                    f"Complete-linkage integrity failed for {cluster_id}: "
                    f"{label['cluster_max_distance']:.8f} > {threshold:.8f}"
                )

            centroid = _normalize_vector(member_matrix.mean(axis=0)).astype(np.float32)
            canonical_position = label["canonical_position"]
            canonical_embedding: np.ndarray | None = None
            representative_average_distance: float | None = None
            representative_max_distance: float | None = None
            representative_centroid_distance: float | None = None
            if canonical_position is not None:
                canonical_embedding = member_matrix[canonical_position].astype(np.float32)
                representative_centroid_distance = float(
                    cosine_distances(canonical_embedding[None, :], centroid[None, :])[0, 0]
                )
                if len(member_nodes) == 1:
                    representative_average_distance = 0.0
                    representative_max_distance = 0.0
                else:
                    representative_distances = member_distances[canonical_position]
                    representative_average_distance = float(
                        representative_distances.sum() / (len(member_nodes) - 1)
                    )
                    representative_max_distance = float(representative_distances.max())

            for member_position, node_index in enumerate(member_indices):
                raw_label = str(result_nodes.at[node_index, text_column])
                canonical_label = label["canonical_label"]
                mapping_applied = (
                    canonical_label is not None and raw_label != str(canonical_label)
                )
                mapping_distance = (
                    float(member_distances[member_position, canonical_position])
                    if canonical_position is not None
                    else 0.0
                )
                result_nodes.at[node_index, "cluster_id"] = cluster_id
                result_nodes.at[node_index, "canonical_label"] = canonical_label
                result_nodes.at[node_index, "naming_status"] = label["naming_status"]
                result_nodes.at[node_index, "mapping_applied"] = mapping_applied
                result_nodes.at[node_index, "mapping_distance"] = mapping_distance

            review_ids = set().union(*member_nodes["_review_ids"].tolist())
            row: dict[str, Any] = {
                "stage": stage_name,
                "cluster_id": cluster_id,
                "canonical_label": label["canonical_label"],
                "medoid_label": label["medoid_label"],
                "naming_status": label["naming_status"],
                "member_count": len(member_nodes),
                "source_row_count": int(member_nodes["source_row_count"].sum()),
                "unique_review_count": len(review_ids),
                "distance_threshold": threshold,
                "naming_max_distance": naming_max_distance,
                "status_cannot_link_pair_count_in_boundary": constraint_pair_count,
                "cluster_max_distance": float(label["cluster_max_distance"]),
                "representative_average_distance": representative_average_distance,
                "representative_max_distance": representative_max_distance,
                "representative_centroid_distance": representative_centroid_distance,
                "member_expressions": member_nodes[text_column].astype(str).tolist(),
                "member_details_json": json.dumps(
                    label["member_details"], ensure_ascii=False, separators=(",", ":")
                ),
                "centroid_embedding": centroid.tolist(),
                "canonical_embedding": (
                    canonical_embedding.tolist() if canonical_embedding is not None else None
                ),
            }
            for column, value in zip(boundary_columns, boundary_tuple, strict=True):
                row[column] = value
            cluster_rows.append(row)

    if result_nodes["cluster_id"].isna().any():
        raise AssertionError(f"Some nodes were not assigned in stage {stage_name}.")
    if result_nodes["mapping_distance"].isna().any():
        raise AssertionError(f"Some mapping distances were not assigned in stage {stage_name}.")
    result_nodes["embedding"] = result_nodes["embedding"].map(lambda value: value.tolist())
    return ClusterOutput(nodes=result_nodes, clusters=pd.DataFrame(cluster_rows))
