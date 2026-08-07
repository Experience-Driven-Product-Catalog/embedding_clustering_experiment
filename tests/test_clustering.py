from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
from conftest import StaticEncoder

from embedding_clustering_experiment.clustering import (
    build_unique_nodes,
    cluster_unique_nodes,
    select_canonical_label,
)


def test_unique_nodes_use_strings_not_source_rows() -> None:
    frame = pd.DataFrame(
        {
            "idx": [1, 2, 3, 4],
            "review_idx": [10, 10, 11, 12],
            "raw_attribute": ["화질", "화질", "화질", "가격"],
        }
    )
    nodes = build_unique_nodes(frame, "raw_attribute")

    assert len(nodes) == 2
    quality = nodes.loc[nodes["raw_attribute"].eq("화질")].iloc[0]
    assert quality["source_row_count"] == 3
    assert quality["unique_review_count"] == 2


def test_complete_linkage_prevents_semantic_chaining(config_values: dict) -> None:
    angles = np.deg2rad([0.0, 20.0, 40.0])
    vectors = {
        label: [float(np.cos(angle)), float(np.sin(angle))]
        for label, angle in zip(["a", "b", "c"], angles, strict=True)
    }
    frame = pd.DataFrame(
        {"idx": [1, 2, 3], "review_idx": [1, 2, 3], "raw_attribute": ["a", "b", "c"]}
    )
    output = cluster_unique_nodes(
        build_unique_nodes(frame, "raw_attribute"),
        "raw_attribute",
        StaticEncoder(vectors),
        "test",
        "T",
        "aspect",
        {"distance_threshold": 0.10, "naming_max_distance": 0.10},
        config_values["clustering"],
        config_values["canonical_label"],
    )

    assert sorted(output.clusters["member_count"].tolist()) == [1, 2]
    assert output.clusters["cluster_max_distance"].max() <= 0.10 + 1e-6


def test_frequency_breaks_tie_only_after_central_candidates(config_values: dict) -> None:
    frame = pd.DataFrame(
        {
            "idx": range(1, 8),
            "review_idx": range(1, 8),
            "raw_attribute": ["희귀1", "희귀2"] + ["자주"] * 5,
        }
    )
    encoder = StaticEncoder({text: [1.0, 0.0] for text in ["희귀1", "희귀2", "자주"]})
    output = cluster_unique_nodes(
        build_unique_nodes(frame, "raw_attribute"),
        "raw_attribute",
        encoder,
        "test",
        "T",
        "aspect",
        {"distance_threshold": 0.10, "naming_max_distance": 0.10},
        config_values["clustering"],
        config_values["canonical_label"],
    )

    assert len(output.clusters) == 1
    assert output.clusters.iloc[0]["canonical_label"] == "자주"


def test_centrality_uses_absolute_cosine_range(config_values: dict) -> None:
    members = pd.DataFrame({"raw_attribute": ["a", "b", "c"], "unique_review_count": [1, 1, 1]})
    distances = np.array([[0.0, 0.2, 0.4], [0.2, 0.0, 0.6], [0.4, 0.6, 0.0]])
    result = select_canonical_label(
        members,
        distances,
        "raw_attribute",
        "aspect",
        config_values["canonical_label"],
        0.7,
        set(),
    )

    centrality = {item["expression"]: item["centrality"] for item in result["member_details"]}
    assert centrality == {"a": 0.85, "b": 0.8, "c": 0.75}


def test_opposite_statuses_cannot_merge(config_values: dict) -> None:
    frame = pd.DataFrame(
        {
            "idx": [1, 2],
            "review_idx": [1, 2],
            "aspect_cluster_id": ["A-1", "A-1"],
            "raw_status": ["있음", "없음"],
        }
    )
    output = cluster_unique_nodes(
        build_unique_nodes(frame, "raw_status", ["aspect_cluster_id"]),
        "raw_status",
        StaticEncoder({"있음": [1.0, 0.0], "없음": [1.0, 0.0]}),
        "status",
        "S",
        "status",
        {"distance_threshold": 0.20, "naming_max_distance": 0.20},
        config_values["clustering"],
        config_values["canonical_label"],
        boundary_columns=["aspect_cluster_id"],
    )

    assert len(output.clusters) == 2
    assert set(output.clusters["canonical_label"]) == {"있음", "없음"}


def test_node_lineage_records_fallback_without_inventing_a_mapping(
    config_values: dict,
) -> None:
    frame = pd.DataFrame(
        {
            "idx": [1, 2],
            "review_idx": [1, 2],
            "raw_attribute": ["화질", "화면 품질"],
        }
    )
    canonical = deepcopy(config_values["canonical_label"])
    canonical["min_canonical_review_count"] = 2
    output = cluster_unique_nodes(
        build_unique_nodes(frame, "raw_attribute"),
        "raw_attribute",
        StaticEncoder({"화질": [1.0, 0.0], "화면 품질": [1.0, 0.0]}),
        "test",
        "T",
        "aspect",
        {"distance_threshold": 0.10, "naming_max_distance": 0.10},
        config_values["clustering"],
        canonical,
    )

    assert output.nodes["canonical_label"].isna().all()
    assert output.nodes["naming_status"].eq("insufficient_review_support").all()
    assert not output.nodes["mapping_applied"].any()
    assert output.nodes["mapping_distance"].eq(0.0).all()
