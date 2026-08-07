from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from embedding_clustering_experiment.user_evaluation import (
    CLUSTER_SCORE_COLUMNS,
    EXPERIMENTS,
    SCORE_COLUMNS,
    _task_record_at,
    apply_first_rating_sync,
    load_cluster_evaluation_tasks,
    load_evaluation_inputs,
    matching_attribute_groups,
    preference_state_key,
    rating_state_key,
    review_card_order,
    save_completed_evaluation,
)


def test_review_text_is_joined_only_at_runtime_and_not_saved(tmp_path: Path) -> None:
    tasks = pd.DataFrame(
        {
            "review_idx": [10, 11],
            "result_a": [json.dumps(["a"]), json.dumps(["a2"])],
            "result_b": [json.dumps(["b"]), json.dumps(["b2"])],
            "result_c": [json.dumps(["c"]), json.dumps(["c2"])],
            "result_d": [json.dumps(["d"]), json.dumps(["d2"])],
        }
    )
    reviews = pd.DataFrame(
        {"idx": [10, 11], "review": ["review ten", "review eleven"], "productName": ["p1", "p2"]}
    )
    tasks_path = tmp_path / "tasks.parquet"
    reviews_path = tmp_path / "reviews.parquet"
    tasks.to_parquet(tasks_path, index=False)
    reviews.to_parquet(reviews_path, index=False)

    joined = load_evaluation_inputs(tasks_path, reviews_path)
    assert joined["review"].tolist() == ["review ten", "review eleven"]
    assert "review" not in tasks.columns
    assert joined["review"].dtype == object
    assert all(joined[column].dtype == object for column in ("result_a", "result_b"))

    rows = []
    for review_idx in (10, 11):
        for experiment in EXPERIMENTS:
            rows.append(
                {
                    "evaluator_uuid": "evaluator-1",
                    "review_idx": review_idx,
                    "experiment": experiment,
                    "factual_faithfulness": 4,
                    "core_explanatory_power": 4,
                    "information_coverage": 4,
                    "preferred": experiment == "A",
                }
            )
    cluster_rows = [
        {
            "evaluator_uuid": "evaluator-1",
            "cluster_validation_id": f"cluster-{number}",
            "stage": "experiment_b",
            "cluster_id": str(number),
            "cohesion": 4,
            "canonical_label_fit": 5,
        }
        for number in (1, 2)
    ]
    destination = save_completed_evaluation(
        pd.DataFrame(rows),
        pd.DataFrame(cluster_rows),
        expected_review_ids={10, 11},
        expected_cluster_ids={"cluster-1", "cluster-2"},
        output_dir=tmp_path / "human_results",
    )
    saved = pd.read_parquet(destination)
    assert "review" not in saved
    assert "productName" not in saved
    assert "completed_at_utc" not in saved
    assert len(saved) == 10
    assert saved["evaluation_level"].value_counts().to_dict() == {
        "review": 8,
        "cluster": 2,
    }
    saved_cluster_scores = saved.loc[
        saved["evaluation_level"].eq("cluster"), list(CLUSTER_SCORE_COLUMNS)
    ]
    assert saved_cluster_scores.to_dict("list") == {
        "cohesion": [4.0, 4.0],
        "canonical_label_fit": [5.0, 5.0],
    }


def test_cluster_task_loader_rejects_singleton_members(tmp_path: Path) -> None:
    cluster_tasks = pd.DataFrame(
        {
            "cluster_validation_id": ["experiment_b:1"],
            "stage": ["experiment_b"],
            "cluster_id": ["1"],
            "parent_aspect_label": [None],
            "canonical_label": ["화질"],
            "members": [json.dumps(["화질"], ensure_ascii=False)],
        }
    )
    path = tmp_path / "cluster_evaluation_tasks.parquet"
    cluster_tasks.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="at least two named members"):
        load_cluster_evaluation_tasks(path)


def test_cluster_task_loader_detaches_arrow_strings_before_row_access(
    tmp_path: Path,
) -> None:
    cluster_tasks = pd.DataFrame(
        {
            "cluster_validation_id": ["experiment_b_attribute:B-000001"],
            "stage": ["experiment_b_attribute"],
            "cluster_id": ["B-000001"],
            "parent_aspect_label": [None],
            "canonical_label": ["화질"],
            "members": [json.dumps(["화질", "화면 품질"], ensure_ascii=False)],
        }
    )
    path = tmp_path / "cluster_evaluation_tasks.parquet"
    cluster_tasks.to_parquet(path, index=False)

    loaded = load_cluster_evaluation_tasks(path)
    assert pa.default_memory_pool().backend_name == "system"
    assert all(dtype == object for dtype in loaded.dtypes)
    record = _task_record_at(loaded, 0)
    assert pd.isna(record.pop("parent_aspect_label"))
    assert record == {
        "cluster_validation_id": "experiment_b_attribute:B-000001",
        "stage": "experiment_b_attribute",
        "cluster_id": "B-000001",
        "canonical_label": "화질",
        "members": json.dumps(["화질", "화면 품질"], ensure_ascii=False),
    }


def test_review_task_loader_rejects_an_empty_experiment_result(
    tmp_path: Path,
) -> None:
    tasks = pd.DataFrame(
        {
            "review_idx": [10],
            "result_a": [json.dumps(["화질"], ensure_ascii=False)],
            "result_b": [json.dumps([], ensure_ascii=False)],
            "result_c": [json.dumps(["화질 > 선명함"], ensure_ascii=False)],
            "result_d": [json.dumps(["화질 > 선명함"], ensure_ascii=False)],
        }
    )
    reviews = pd.DataFrame({"idx": [10], "review": ["화질이 선명합니다."]})
    tasks_path = tmp_path / "tasks.parquet"
    reviews_path = tmp_path / "reviews.parquet"
    tasks.to_parquet(tasks_path, index=False)
    reviews.to_parquet(reviews_path, index=False)

    with pytest.raises(ValueError, match="non-empty JSON arrays"):
        load_evaluation_inputs(tasks_path, reviews_path)


def test_matching_attribute_groups_use_exact_order_independent_names() -> None:
    task = {
        "result_a": json.dumps(["밝기 > 밝음", "소리 > 잘들림"]),
        "result_b": json.dumps(["소리 > 잘들림", "밝기 > 밝음"]),
        "result_c": json.dumps(["밝기 > 밝음", "소리 > 잘들림 "]),
        "result_d": json.dumps(["크기 > 큼"]),
    }

    assert matching_attribute_groups(task) == (("A", "B"),)


def test_review_card_order_is_stable_per_evaluator_and_varies_by_review() -> None:
    evaluator_uuid = "evaluator-123"
    first_order = review_card_order(evaluator_uuid, 50777)

    assert first_order == review_card_order(evaluator_uuid, 50777)
    assert set(first_order) == set(EXPERIMENTS)

    orders = {review_card_order(evaluator_uuid, review_idx) for review_idx in range(50777, 50801)}
    assert len(orders) > 1


def test_only_first_rating_selection_is_synchronized() -> None:
    review_idx = 50777
    group = ("A", "B")
    state: dict[str, object] = {}
    for experiment in group:
        for score_column in SCORE_COLUMNS:
            state[rating_state_key(review_idx, experiment, score_column)] = None

    source_key = rating_state_key(review_idx, "A", "factual_faithfulness")
    target_key = rating_state_key(review_idx, "B", "factual_faithfulness")
    state[source_key] = 4
    copied = apply_first_rating_sync(
        state,
        review_idx=review_idx,
        source_experiment="A",
        score_column="factual_faithfulness",
        matching_group=group,
    )
    assert copied == ("B",)
    assert state[target_key] == 4

    state[target_key] = 2
    copied_again = apply_first_rating_sync(
        state,
        review_idx=review_idx,
        source_experiment="B",
        score_column="factual_faithfulness",
        matching_group=group,
    )
    assert copied_again == ()
    assert state[source_key] == 4
    assert state[target_key] == 2

    second_criterion_source = rating_state_key(review_idx, "B", "core_explanatory_power")
    second_criterion_target = rating_state_key(review_idx, "A", "core_explanatory_power")
    state[second_criterion_source] = 5
    copied_second_criterion = apply_first_rating_sync(
        state,
        review_idx=review_idx,
        source_experiment="B",
        score_column="core_explanatory_power",
        matching_group=group,
    )
    assert copied_second_criterion == ("A",)
    assert state[second_criterion_target] == 5


def test_streamlit_review_sync_then_cluster_phase_and_atomic_save(
    tmp_path: Path,
) -> None:
    app_test_module = pytest.importorskip("streamlit.testing.v1")
    app_test_class = app_test_module.AppTest

    review_idx = 50777
    tasks = pd.DataFrame(
        {
            "review_idx": [review_idx],
            "result_a": [json.dumps(["그립감", "가성비"])],
            "result_b": [json.dumps(["가성비", "그립감"])],
            "result_c": [json.dumps(["그립감 > 기억에 남음"])],
            "result_d": [json.dumps(["가성비 > 좋음"])],
        }
    )
    reviews = pd.DataFrame(
        {
            "idx": [review_idx],
            "review": ["그립감이 기억에 남고 가성비가 좋습니다."],
            "productName": ["테스트 상품"],
        }
    )
    tasks_path = tmp_path / "tasks.parquet"
    reviews_path = tmp_path / "reviews.parquet"
    cluster_tasks_path = tmp_path / "cluster_evaluation_tasks.parquet"
    tasks.to_parquet(tasks_path, index=False)
    reviews.to_parquet(reviews_path, index=False)
    pd.DataFrame(
        {
            "cluster_validation_id": ["experiment_b:1"],
            "stage": ["experiment_b"],
            "cluster_id": ["1"],
            "parent_aspect_label": [None],
            "canonical_label": ["그립감"],
            "members": [json.dumps(["그립감", "손에 잡히는 느낌"], ensure_ascii=False)],
        }
    ).to_parquet(cluster_tasks_path, index=False)

    script = f"""
import sys
from embedding_clustering_experiment.user_evaluation import main
sys.argv = [
    "user_evaluation.py",
    "--tasks", {str(tasks_path)!r},
    "--reviews", {str(reviews_path)!r},
    "--output-dir", {str(tmp_path / "human_results")!r},
]
main()
"""
    app = app_test_class.from_string(script, default_timeout=20).run()
    assert not app.exception
    expected_card_order = review_card_order(app.session_state.evaluator_uuid, review_idx)
    displayed_card_order = [
        experiment
        for button in app.button
        for experiment in EXPERIMENTS
        if button.key
        == f"score_button_{review_idx}_{experiment}_factual_faithfulness_1"
    ]
    assert displayed_card_order == list(expected_card_order)

    app.button(key=f"score_button_{review_idx}_A_factual_faithfulness_4").click().run()
    source_key = rating_state_key(review_idx, "A", "factual_faithfulness")
    target_key = rating_state_key(review_idx, "B", "factual_faithfulness")
    assert app.session_state[source_key] == 4
    assert app.session_state[target_key] == 4

    app.button(key=f"score_button_{review_idx}_B_factual_faithfulness_2").click().run()
    assert not app.exception
    assert app.session_state[source_key] == 4
    assert app.session_state[target_key] == 2

    for experiment in EXPERIMENTS:
        for score_column in SCORE_COLUMNS:
            app.button(
                key=f"score_button_{review_idx}_{experiment}_{score_column}_3"
            ).click().run()
    app.checkbox(key=preference_state_key(review_idx, "A")).check().run()
    app.button(key=f"next_review_{review_idx}").click().run()
    assert not app.exception
    assert app.session_state["phase"] == "cluster"

    cluster_validation_id = "experiment_b:1"
    for score_column in CLUSTER_SCORE_COLUMNS:
        app.button(
            key=f"cluster_score_button_{cluster_validation_id}_{score_column}_4"
        ).click().run()
    app.button(key=f"next_cluster_{cluster_validation_id}").click().run()
    assert not app.exception

    completed_dir = tmp_path / "human_results" / "completed_evaluators"
    completed = list(completed_dir.glob("*.parquet"))
    assert len(completed) == 1
    saved = pd.read_parquet(completed[0])
    assert saved["evaluation_level"].value_counts().to_dict() == {
        "review": 4,
        "cluster": 1,
    }
