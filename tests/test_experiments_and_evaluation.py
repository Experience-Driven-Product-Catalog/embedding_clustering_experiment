from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace

import pandas as pd
import pytest

from embedding_clustering_experiment.evaluation import (
    build_review_normalization_changes,
    run_automatic_evaluation,
)
from embedding_clustering_experiment.experiments import (
    ExperimentArtifacts,
    build_normalization_metadata,
)


def test_outputs_append_normalized_values_and_lineage_while_preserving_source_values(
    experiment_artifacts: ExperimentArtifacts,
) -> None:
    artifacts = experiment_artifacts
    assert list(artifacts.experiment_a.columns) == [
        "idx",
        "review_idx",
        "raw_attribute",
        "sentiment",
        "attribute",
    ]
    assert list(artifacts.experiment_b.columns) == [
        "idx",
        "review_idx",
        "raw_attribute",
        "sentiment",
        "attribute",
        "attribute_cluster_id",
        "attribute_naming_status",
        "attribute_mapping_applied",
        "attribute_mapping_distance",
        "embedding_model_id",
        "normalization_version",
        "normalization_run_id",
        "normalization_config_sha256",
    ]
    assert list(artifacts.experiment_d.columns) == [
        "idx",
        "review_idx",
        "raw_aspect",
        "raw_status",
        "excerpt",
        "opinion",
        "sentiment",
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
    ]
    for frame in artifacts.result_tables().values():
        assert "product_category" not in frame
        assert "product_name" not in frame
        assert "review" not in frame
        assert frame["review_idx"].notna().all()

    assert artifacts.experiment_a["sentiment"].equals(artifacts.experiment_b["sentiment"])
    c_sentiment = artifacts.experiment_c.sort_values("idx")["sentiment"].reset_index(drop=True)
    d_sentiment = artifacts.experiment_d.sort_values("idx")["sentiment"].reset_index(drop=True)
    assert c_sentiment.equals(d_sentiment)


def test_normalized_outputs_store_complete_row_level_lineage(
    experiment_artifacts: ExperimentArtifacts,
    config_values: dict,
) -> None:
    artifacts = experiment_artifacts
    for nodes in (
        artifacts.experiment_b_nodes,
        artifacts.experiment_d_aspect_nodes,
        artifacts.experiment_d_status_nodes,
    ):
        assert {"cluster_id", "naming_status", "mapping_applied", "mapping_distance"}.issubset(
            nodes.columns
        )
        assert nodes["mapping_distance"].between(0.0, 2.0).all()

    b = artifacts.experiment_b
    assert b["attribute_mapping_applied"].equals(
        b["raw_attribute"].ne(b["attribute"]).astype("boolean")
    )
    assert b["attribute_cluster_id"].str.startswith("B-").all()

    d = artifacts.experiment_d
    assert d["aspect_mapping_applied"].equals(
        d["raw_aspect"].ne(d["aspect"]).astype("boolean")
    )
    status_present = d["raw_status"].notna()
    assert d.loc[status_present, "status_mapping_applied"].equals(
        d.loc[status_present, "raw_status"]
        .ne(d.loc[status_present, "status"])
        .astype("boolean")
    )
    status_lineage = [
        "status_cluster_id",
        "status_naming_status",
        "status_mapping_applied",
        "status_mapping_distance",
    ]
    assert d.loc[~status_present, status_lineage].isna().all().all()

    for frame in (b, d):
        assert frame["embedding_model_id"].eq(config_values["embedding"]["model_id"]).all()
        assert frame["normalization_version"].eq(
            config_values["project"]["normalization_version"]
        ).all()
        assert frame["normalization_run_id"].eq("test-run").all()
        assert frame["normalization_config_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_normalization_config_fingerprint_changes_with_semantic_parameters(
    config_values: dict,
    static_encoder: object,
) -> None:
    baseline = build_normalization_metadata(config_values, static_encoder, "run-a")
    repeated = build_normalization_metadata(config_values, static_encoder, "run-b")
    changed_config = deepcopy(config_values)
    changed_config["clustering"]["experiment_b"]["distance_threshold"] = 0.23
    changed = build_normalization_metadata(changed_config, static_encoder, "run-c")

    assert baseline["normalization_config_sha256"] == repeated[
        "normalization_config_sha256"
    ]
    assert baseline["normalization_config_sha256"] != changed[
        "normalization_config_sha256"
    ]


def test_category_boundary_is_removed_but_status_stays_inside_aspect(
    experiment_artifacts: ExperimentArtifacts,
) -> None:
    artifacts = experiment_artifacts
    assert "product_category" not in artifacts.experiment_b_nodes
    assert "product_category" not in artifacts.experiment_d_aspect_nodes
    assert "product_category" not in artifacts.experiment_d_status_nodes
    assert (
        artifacts.experiment_d_status_nodes.groupby("cluster_id")["aspect_cluster_id"]
        .nunique()
        .le(1)
        .all()
    )


def test_exclusion_and_nullable_status_are_preserved(
    experiment_artifacts: ExperimentArtifacts,
) -> None:
    artifacts = experiment_artifacts
    assert 4 not in set(artifacts.experiment_c["idx"])
    assert 4 not in set(artifacts.experiment_d["idx"])
    assert artifacts.experiment_c["raw_status"].isna().sum() == 1
    assert artifacts.experiment_c["status"].isna().sum() == 1
    assert artifacts.experiment_d["status"].isna().sum() == 1
    assert artifacts.experiment_d_status_nodes["raw_status"].notna().all()


def test_automatic_metrics_use_full_data_but_human_tasks_use_configured_sample(
    representative_frame: pd.DataFrame,
    opinion_frame: pd.DataFrame,
    experiment_artifacts: ExperimentArtifacts,
    config_values: dict,
) -> None:
    evaluation = run_automatic_evaluation(
        representative_frame,
        opinion_frame,
        experiment_artifacts,
        config_values,
    )

    assert evaluation.summary["scope"] == "full_input_datasets"
    assert evaluation.summary["sampling_used"] is False
    assert evaluation.summary["common_review_count"] == 4
    assert evaluation.summary["user_evaluation_sampling_used"] is True
    assert evaluation.summary["user_evaluation_task_count"] == 2
    assert evaluation.summary["cluster_evaluation_task_count"] == 2
    assert evaluation.summary["user_evaluation_random_seed"] == 17171771
    assert (
        evaluation.summary["integrity_checks_passed"] == evaluation.summary["integrity_check_count"]
    )
    tasks = evaluation.tables["user_evaluation_tasks"]
    assert len(tasks) == 2
    assert list(tasks.columns) == ["review_idx", "result_a", "result_b", "result_c", "result_d"]
    assert "review" not in tasks
    for column in ("result_a", "result_b", "result_c", "result_d"):
        assert all(
            decoded and all(item.strip() for item in decoded)
            for decoded in map(json.loads, tasks[column])
        )

    cluster_tasks = evaluation.tables["cluster_evaluation_tasks"]
    assert len(cluster_tasks) == 2
    assert list(cluster_tasks.columns) == [
        "cluster_validation_id",
        "stage",
        "cluster_id",
        "parent_aspect_label",
        "canonical_label",
        "members",
    ]
    assert cluster_tasks["cluster_validation_id"].is_unique
    assert all(len(json.loads(value)) > 1 for value in cluster_tasks["members"])

    repeated = run_automatic_evaluation(
        representative_frame,
        opinion_frame,
        experiment_artifacts,
        config_values,
    )
    pd.testing.assert_frame_equal(tasks, repeated.tables["user_evaluation_tasks"])
    pd.testing.assert_frame_equal(
        cluster_tasks,
        repeated.tables["cluster_evaluation_tasks"],
    )
    metrics = evaluation.tables["experiment_metrics"].set_index("experiment")
    assert set(metrics.index) == {"A", "B", "C", "D"}
    assert metrics.loc["A", "observation_count"] == 4
    assert metrics.loc["C", "observation_count"] == 4


def test_human_task_generation_fails_when_exact_n_is_unavailable(
    representative_frame: pd.DataFrame,
    opinion_frame: pd.DataFrame,
    experiment_artifacts: ExperimentArtifacts,
    config_values: dict,
) -> None:
    unavailable = deepcopy(config_values)
    unavailable["evaluation"]["user_evaluation"]["task_count"] = 5

    with pytest.raises(ValueError, match="Need 5 eligible review tasks; found 4"):
        run_automatic_evaluation(
            representative_frame,
            opinion_frame,
            experiment_artifacts,
            unavailable,
        )

    cluster_unavailable = deepcopy(config_values)
    cluster_unavailable["evaluation"]["user_evaluation"]["task_count"] = 3
    with pytest.raises(ValueError, match="Need 3 eligible cluster tasks; found 2"):
        run_automatic_evaluation(
            representative_frame,
            opinion_frame,
            experiment_artifacts,
            cluster_unavailable,
        )


def test_review_change_comparison_accepts_different_list_lengths(
    experiment_artifacts: ExperimentArtifacts,
) -> None:
    artifacts = experiment_artifacts
    extra = artifacts.experiment_b.loc[artifacts.experiment_b["review_idx"].eq(10)].copy()
    extra["idx"] = 99
    extra["attribute"] = "추가 속성"
    modified = replace(
        artifacts,
        experiment_b=pd.concat([artifacts.experiment_b, extra], ignore_index=True),
    )
    changes = build_review_normalization_changes(modified)

    assert changes["A_to_B_changed"].dtype == bool
    assert changes.loc[changes["review_idx"].eq(10), "A_to_B_changed"].item() is True
