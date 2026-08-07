from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import yaml
from conftest import StaticEncoder

from embedding_clustering_experiment.config import load_config
from embedding_clustering_experiment.dynamic_report import (
    build_dynamic_review_decision_report,
    build_submission_from_catalog_reviews,
    render_dynamic_review_decision_markdown,
)
from embedding_clustering_experiment.reporting import (
    SENTIMENT_ORDER,
    build_static_catalog_report,
    collapse_sentiments,
    load_report_inputs,
    render_static_markdown,
    wilson_interval,
    write_json,
)
from embedding_clustering_experiment.runner import run

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _build_report_fixture(
    tmp_path: Path,
    representative_frame: pd.DataFrame,
    opinion_frame: pd.DataFrame,
    static_encoder: StaticEncoder,
    config_values: dict,
) -> tuple[Path, Path, Path]:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    representative_path = dataset_dir / "representative.parquet"
    opinion_path = dataset_dir / "opinion.parquet"
    reviews_path = dataset_dir / "reviews.parquet"
    representative_frame.to_parquet(representative_path, index=False)
    opinion_frame.to_parquet(opinion_path, index=False)
    pd.DataFrame(
        {
            "idx": pd.Series([10, 11, 12, 13], dtype="int64"),
            "review": pd.Series(
                ["첫 상품 e1", "둘째 상품 e2", "가격 의견 e3", "종합 e4, 베젤 e5"],
                dtype="string",
            ),
            "productName": pd.Series(
                ["Product A", "Product B", "Product B", "Product B"],
                dtype="string",
            ),
        }
    ).to_parquet(reviews_path, index=False)

    values = deepcopy(config_values)
    values["inputs"] = {
        "representative_attributes": "dataset/representative.parquet",
        "opinion_units": "dataset/opinion.parquet",
    }
    values["project"]["results_root"] = "results"
    values["evaluation"]["user_evaluation"]["task_count"] = 2
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(values, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    run_dir = run(
        load_config(config_path),
        encoder_factory=lambda _config, _base_dir: static_encoder,
    )
    return run_dir, reviews_path, opinion_path


def _write_completed_human_results(run_dir: Path, *, evaluator_count: int = 3) -> Path:
    review_tasks = pd.read_parquet(run_dir / "evaluation/user_evaluation_tasks.parquet")
    cluster_tasks = pd.read_parquet(run_dir / "evaluation/cluster_evaluation_tasks.parquet")
    completed_dir = run_dir / "evaluation/human_results/completed_evaluators"
    completed_dir.mkdir(parents=True, exist_ok=True)
    score_by_experiment = {
        "A": (3, 3, 2, False),
        "B": (3, 3, 2, False),
        "C": (4, 5, 5, False),
        "D": (4, 5, 5, True),
    }
    for evaluator_number in range(1, evaluator_count + 1):
        evaluator_uuid = f"00000000-0000-0000-0000-{evaluator_number:012d}"
        rows: list[dict] = []
        for review_idx in review_tasks["review_idx"]:
            for experiment, scores in score_by_experiment.items():
                factual, core, coverage, preferred = scores
                rows.append(
                    {
                        "evaluator_uuid": evaluator_uuid,
                        "evaluation_level": "review",
                        "review_idx": int(review_idx),
                        "experiment": experiment,
                        "factual_faithfulness": factual,
                        "core_explanatory_power": core,
                        "information_coverage": coverage,
                        "preferred": preferred,
                        "cluster_validation_id": None,
                        "stage": None,
                        "cluster_id": None,
                        "cohesion": None,
                        "canonical_label_fit": None,
                    }
                )
        for task in cluster_tasks.itertuples(index=False):
            rows.append(
                {
                    "evaluator_uuid": evaluator_uuid,
                    "evaluation_level": "cluster",
                    "review_idx": None,
                    "experiment": None,
                    "factual_faithfulness": None,
                    "core_explanatory_power": None,
                    "information_coverage": None,
                    "preferred": None,
                    "cluster_validation_id": str(task.cluster_validation_id),
                    "stage": str(task.stage),
                    "cluster_id": str(task.cluster_id),
                    "cohesion": 5,
                    "canonical_label_fit": 5,
                }
            )
        pd.DataFrame(rows).to_parquet(completed_dir / f"{evaluator_uuid}.parquet", index=False)
    return completed_dir


def test_static_report_is_deterministic_and_uses_review_level_votes(
    tmp_path: Path,
    representative_frame: pd.DataFrame,
    opinion_frame: pd.DataFrame,
    static_encoder: StaticEncoder,
    config_values: dict,
) -> None:
    run_dir, reviews_path, opinion_path = _build_report_fixture(
        tmp_path,
        representative_frame,
        opinion_frame,
        static_encoder,
        config_values,
    )
    inputs = load_report_inputs(run_dir, reviews_path, opinion_path)
    report = build_static_catalog_report(inputs, "Product B")

    assert report == build_static_catalog_report(inputs, "Product B")
    assert report["product"] == {
        "product_name": "Product B",
        "catalog_review_count": 3,
        "reviews_with_eligible_opinion_units": 3,
        "review_coverage_rate": 1.0,
    }
    assert report["coverage"]["excluded_general_experience_opinion_unit_count"] == 1
    assert report["normalization_quality"]["automatic_integrity_checks"]["all_passed"]
    assert report["aggregation_contract"]["sentiment_denominator"] == "review_level_vote"
    assert report["schema_version"] == "1.2.0"
    assert report["normalization_reduction"]["aspect"] == {
        "raw_count": 3,
        "normalized_count": 3,
        "comparison_grain": "unique_raw_aspect_to_unique_aspect_cluster_id",
        "decrease_rate": 0.0,
    }
    assert report["most_debated_aspect"] is not None
    assert report["most_debated_aspect"]["selection_scope"] == "top_10_aspect_summary"
    review_text_by_id = dict(
        zip(inputs.reviews["review_idx"], inputs.reviews["review"], strict=True)
    )
    for samples in report["most_debated_aspect"]["evidence"].values():
        for sample in samples:
            assert sample["review_text"] == review_text_by_id[sample["review_idx"]]
    related = report["related_products"]
    assert related["candidate_product_count"] == 1
    assert related["schema_version"] == "1.0.1"
    assert related["similarity_contract"]["feature_scope"] == (
        "all_canonical_aspect_cluster_ids_not_display_top_n"
    )
    assert related["similarity_contract"]["aspect_status_policy"] == (
        "exact_aspect_cluster_id_status_cluster_id_overlap_reported_not_scored"
    )
    assert [row["product_name"] for row in related["similar_products"]] == ["Product A"]
    assert related["weakness_repair_alternatives"]["status"] == (
        "NO_HIGH_SUPPORT_NEGATIVE_ASPECT_STATUS_EVIDENCE"
    )
    null_status_rows = [
        row for row in report["aspect_status_summary"] if row["status_cluster_id"] is None
    ]
    assert len(null_status_rows) == 1
    assert null_status_rows[0]["status"] is None
    for section in ("aspect_summary", "aspect_status_summary"):
        for row in report[section]:
            assert sum(row["sentiment"]["counts"].values()) == row["supporting_review_count"]
            assert set(row["sentiment"]["counts"]) == set(SENTIMENT_ORDER)
    for row in report["aspect_status_summary"]:
        for sentiment, evidence in row["evidence"].items():
            assert all(item["review_vote_sentiment"] == sentiment for item in evidence)

    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_json(report, first_path)
    write_json(build_static_catalog_report(inputs, "Product B"), second_path)
    assert first_path.read_bytes() == second_path.read_bytes()
    static_markdown = render_static_markdown(report)
    assert "리뷰를 aspect 단위로 분석한 표입니다." in static_markdown
    assert "| rank | aspect | reviews(mention rate) |" in static_markdown
    assert "## 가장 논쟁적인 속성" in static_markdown
    assert "### 긍정 리뷰 샘플\n\n- " in static_markdown
    assert "\n### 부정 리뷰 샘플\n\n- " in static_markdown
    assert "## 관련 상품" in static_markdown
    assert "유사 상품은 표시용 Top 10" not in static_markdown
    assert "- **experience similarity**:" in static_markdown
    assert "- **evidence overlap**: 공통적으로 가지는 aspect 비율." in static_markdown
    assert "- **support reliability**: 공통 aspect의 관측 리뷰 수가 충분한지를 반영한 신뢰도." in (
        static_markdown
    )
    assert "가중 공통도" not in static_markdown
    for old_header in (
        "## Aspect sentiment matrix (top 10)",
        "## Aspect-status matrix (top 10)",
        "## Most Debated Aspect",
        "## Related products",
        "### Similar products",
        "### Alternatives that address observed weaknesses",
    ):
        assert old_header not in static_markdown
    assert "_" not in static_markdown
    product_names = {report["product"]["product_name"]}
    product_names.update(row["product_name"] for row in related["similar_products"])
    assert all(f"**{product_name}**" in static_markdown for product_name in product_names)
    for samples in report["most_debated_aspect"]["evidence"].values():
        for sample in samples:
            assert sample["review_text"] in static_markdown
    assert "## Human evaluation" not in static_markdown
    assert "## Quality flags" not in static_markdown


def test_checked_in_static_examples_cover_related_products_and_another_product() -> None:
    expected_dir = PROJECT_ROOT / "examples/analysis_expected"
    generated_dir = PROJECT_ROOT / "examples/generated"
    for filename in (
        "static_catalog_report.json",
        "static_catalog_report.md",
        "dynamic_decision_proposal.json",
        "dynamic_decision_proposal.md",
    ):
        assert (expected_dir / filename).read_bytes() == (generated_dir / filename).read_bytes()

    report = json.loads((generated_dir / "static_catalog_report.json").read_text(encoding="utf-8"))
    related = report["related_products"]
    assert report["schema_version"] == "1.2.0"
    assert report["normalization_reduction"]["aspect"] == {
        "raw_count": 159,
        "normalized_count": 111,
        "comparison_grain": "unique_raw_aspect_to_unique_aspect_cluster_id",
        "decrease_rate": 0.301887,
    }
    assert report["most_debated_aspect"]["aspect"] == "빛샘"
    assert len(related["similar_products"]) == 3
    assert len(related["weakness_repair_alternatives"]["alternatives"]) == 3
    assert all(
        row["product_name"] != report["product"]["product_name"]
        for row in related["similar_products"]
    )
    assert (
        related["weakness_repair_alternatives"]["ranking_contract"]["status_match_policy"]
        == "exact_status_cluster_id_only"
    )
    assert all(
        "NEAR" not in match["relation"]
        for alternative in related["weakness_repair_alternatives"]["alternatives"]
        for match in alternative["requirement_matches"]
    )

    static_markdown = (generated_dir / "static_catalog_report.md").read_text(encoding="utf-8")
    for heading in (
        "## 속성 감성 행렬 (상위 10개)",
        "## 속성-상태 행렬 (상위 10개)",
        "## 가장 논쟁적인 속성",
        "## 관련 상품",
        "### 유사 상품",
        "### 관찰된 약점을 보완하는 대안 상품",
    ):
        assert heading in static_markdown
    assert (
        "| rank | product | experience similarity | evidence overlap | support reliability |"
        in (static_markdown)
    )
    assert (
        "| rank | product | weakness utility | experience similarity | weakness repair score |"
        in static_markdown
    )
    assert "_" not in static_markdown
    assert "## Human evaluation" not in static_markdown
    assert "## Quality flags" not in static_markdown
    related_section = static_markdown.split("## 관련 상품", maxsplit=1)[1]
    assert related_section.lstrip().startswith("### 유사 상품")
    assert not any(line.startswith("## ") for line in related_section.splitlines())
    assert static_markdown.rstrip().splitlines()[-1].startswith("| 3 |")
    product_names = {report["product"]["product_name"]}
    product_names.update(row["product_name"] for row in related["similar_products"])
    product_names.update(
        row["product_name"] for row in related["weakness_repair_alternatives"]["alternatives"]
    )
    assert all(f"**{product_name}**" in static_markdown for product_name in product_names)

    sample_dir = generated_dir / "random_product_smart_s32bm700"
    sample = json.loads((sample_dir / "static_catalog_report.json").read_text(encoding="utf-8"))
    assert sample["product"]["product_name"] == "SMART 삼성전자 M7 S32BM700"
    assert sample["product"]["catalog_review_count"] == 71
    assert len(sample["related_products"]["similar_products"]) == 3
    assert len(sample["related_products"]["weakness_repair_alternatives"]["alternatives"]) == 3
    sample_markdown = (sample_dir / "static_catalog_report.md").read_text(encoding="utf-8")
    assert "'화면 크기'로 14개의 리뷰" in sample_markdown
    assert "_" not in sample_markdown
    sample_product_names = {sample["product"]["product_name"]}
    sample_product_names.update(
        row["product_name"] for row in sample["related_products"]["similar_products"]
    )
    sample_product_names.update(
        row["product_name"]
        for row in sample["related_products"]["weakness_repair_alternatives"]["alternatives"]
    )
    assert all(f"**{product_name}**" in sample_markdown for product_name in sample_product_names)

    dynamic = json.loads(
        (generated_dir / "dynamic_decision_proposal.json").read_text(encoding="utf-8")
    )
    assert dynamic["proposal_type"] == "dynamic_review_decision_proposal"
    assert len(dynamic["submission"]["reviews"]) == 1
    assert len(dynamic["submitted_opinion_units"]) == 1
    assert len(dynamic["catalog_relationships"]) == 1
    assert dynamic["unmentioned_aspect_status"]
    relationship = dynamic["catalog_relationships"][0]
    assert relationship["other_status_review_count"] == 0
    assert [row["rank"] for row in relationship["other_aspect_top_statuses"]] == [1, 2, 3]
    assert relationship["other_aspect_top_statuses"][0]["status"] == "큼"
    assert relationship["other_aspect_top_statuses"][0]["supporting_review_count"] == 3
    alternatives = dynamic["alternative_recommendations"]
    assert alternatives["status"] == "COMPLETED"
    assert len(alternatives["alternatives"]) == 3
    assert all(
        row["product_name"] != dynamic["submission"]["product_name"]
        for row in alternatives["alternatives"]
    )
    assert all(
        row["weakness_repair_score"]
        == round(
            0.25 * row["experience_similarity"] + 0.75 * ((row["weakness_utility"] + 1.0) / 2.0),
            6,
        )
        for row in alternatives["alternatives"]
    )
    dynamic_markdown = (generated_dir / "dynamic_decision_proposal.md").read_text(encoding="utf-8")
    for heading in (
        "# 동적 의사결정 제안서",
        "## 다른 리뷰와의 관계",
        "### 언급되지 않는 aspect-status",
        "## 대안 상품 추천",
    ):
        assert heading in dynamic_markdown
    assert "| raw_aspect | aspect | raw_status | status | excerpt | opinion | sentiment |" in (
        dynamic_markdown
    )
    assert (
        f"{dynamic['submission']['submitted_at_local']}에 제출된 "
        f'**{dynamic["submission"]["product_name"]}**에 대한 리뷰\n\n$$"'
    ) in dynamic_markdown
    assert "\n\n를 기반으로 의사결정을 보조하는 제안서입니다." in dynamic_markdown
    assert f"**{dynamic['submission']['product_name']}**" in dynamic_markdown
    assert f"### **{relationship['aspect']}** {relationship['status']}" in dynamic_markdown
    assert (
        f"**{relationship['aspect']}** > {relationship['status']} 조합과 "
        "완전히 일치하는 aspect-status가 없습니다."
    ) in dynamic_markdown
    assert "| rank | status | reviews | positive | negative | etc |" in dynamic_markdown
    assert "| 1 | 큼 | 3 | 3 | 0 | 0/0/0 |" in dynamic_markdown
    assert (
        f"1순위 대안은 **{alternatives['alternatives'][0]['product_name']}**입니다."
    ) in dynamic_markdown
    assert all(
        f"**{row['product_name']}**" in dynamic_markdown for row in alternatives["alternatives"]
    )
    for legacy_heading in (
        "## Request",
        "## Decision",
        "## Candidate ranking",
        "## Human evaluation",
        "## Quality flags",
    ):
        assert legacy_heading not in dynamic_markdown

    multi_aspect_dir = generated_dir / "multi_aspect_review_52282"
    multi_aspect = json.loads(
        (multi_aspect_dir / "dynamic_decision_proposal.json").read_text(encoding="utf-8")
    )
    assert multi_aspect["submission"]["product_name"] == "SMART 삼성전자 M7 S32BM700"
    assert len(multi_aspect["submitted_opinion_units"]) == 8
    assert (
        len(
            {
                unit["aspect_cluster_id"]
                for unit in multi_aspect["submitted_opinion_units"]
                if unit["aspect_cluster_id"] is not None
            }
        )
        == 8
    )
    ott_relationship = next(
        row for row in multi_aspect["catalog_relationships"] if row["aspect"] == "OTT 재생"
    )
    assert ott_relationship["other_status_review_count"] == 0
    assert ott_relationship["other_aspect_top_statuses"] == [
        {
            "rank": 1,
            "status": "원활함",
            "status_cluster_id": "D-S-000039",
            "supporting_review_count": 2,
            "sentiment": {
                "counts": {
                    "positive": 2,
                    "negative": 0,
                    "mixed": 0,
                    "neutral": 0,
                    "unknown": 0,
                },
                "shares": {
                    "positive": 1.0,
                    "negative": 0.0,
                    "mixed": 0.0,
                    "neutral": 0.0,
                    "unknown": 0.0,
                },
                "dominant_sentiment": "positive",
                "dominant_share": 1.0,
                "positive_wilson_95": {"lower": 0.34238, "upper": 1.0},
                "negative_wilson_95": {"lower": 0.0, "upper": 0.65762},
                "normalized_entropy": 0.0,
            },
        }
    ]
    multi_aspect_markdown = (multi_aspect_dir / "dynamic_decision_proposal.md").read_text(
        encoding="utf-8"
    )
    assert "### **OTT 재생** 바로 재생 가능" in multi_aspect_markdown
    assert "| 1 | 원활함 | 2 | 2 | 0 | 0/0/0 |" in multi_aspect_markdown
    assert "1순위 대안은 **AOC 알파스캔 Q27G2S 게이밍 IPS 155 QHD 프리싱크 무결점**" in (
        multi_aspect_markdown
    )

    receipt = json.loads((PROJECT_ROOT / "examples/verification.json").read_text(encoding="utf-8"))
    assert receipt["check_count"] == receipt["passed_count"] == len(receipt["checks"]) == 43
    assert receipt["all_passed"] and all(receipt["checks"].values())
    for relative_path, expected_sha256 in receipt["sha256"].items():
        actual_sha256 = hashlib.sha256((generated_dir / relative_path).read_bytes()).hexdigest()
        assert actual_sha256 == expected_sha256


def test_dynamic_review_report_compares_submitted_units_with_other_reviews(
    tmp_path: Path,
    representative_frame: pd.DataFrame,
    opinion_frame: pd.DataFrame,
    static_encoder: StaticEncoder,
    config_values: dict,
) -> None:
    run_dir, reviews_path, opinion_path = _build_report_fixture(
        tmp_path,
        representative_frame,
        opinion_frame,
        static_encoder,
        config_values,
    )
    inputs = load_report_inputs(run_dir, reviews_path, opinion_path)
    source_row = inputs.joined.loc[inputs.joined["review_idx"].eq(11)].iloc[0]
    submission = {
        "submission_id": "external-negative-quality",
        "submitted_at_local": "20260804-120000",
        "product_name": "Product B",
        "reviews": [
            {
                "review": "새 사용자 e2",
                "opinion_units": [
                    {
                        "raw_aspect": "화면 품질",
                        "aspect": str(source_row.aspect),
                        "raw_status": "선명함",
                        "status": str(source_row.status),
                        "excerpt": "e2",
                        "opinion": "화면이 선명하지 않다고 느낌",
                        "sentiment": "negative",
                        "aspect_cluster_id": str(source_row.aspect_cluster_id),
                        "status_cluster_id": str(source_row.status_cluster_id),
                    }
                ],
            }
        ],
        "excluded_products": [],
    }
    proposal = build_dynamic_review_decision_report(
        inputs,
        submission,
    )

    assert proposal["proposal_type"] == "dynamic_review_decision_proposal"
    assert proposal["submitted_opinion_units"][0]["raw_aspect"] == "화면 품질"
    relationship = proposal["catalog_relationships"][0]
    assert relationship["aspect"] == str(source_row.aspect)
    assert relationship["status"] == str(source_row.status)
    assert relationship["same_status_review_count"] == 1
    assert relationship["other_status_sentiment"]["counts"]["positive"] == 1
    assert relationship["other_aspect_top_statuses"] == []
    assert relationship["relation_code"] == "CONTRADICTS_OTHER_REVIEW_MAJORITY"
    alternatives = proposal["alternative_recommendations"]
    assert alternatives["status"] == "COMPLETED"
    assert all(row["product_name"] != "Product B" for row in alternatives["alternatives"])
    assert alternatives["alternatives"][0]["requirement_matches"][0]["relation"] == (
        "EXACT_UNDESIRED_STATUS_ONLY"
    )

    markdown = render_dynamic_review_decision_markdown(proposal)
    assert "# 동적 의사결정 제안서" in markdown
    assert (
        "| raw_aspect | aspect | raw_status | status | excerpt | opinion | sentiment |" in markdown
    )
    assert "## 다른 리뷰와의 관계" in markdown
    assert "### 언급되지 않는 aspect-status" in markdown
    assert "## 대안 상품 추천" in markdown
    for legacy_heading in (
        "## Request",
        "## Decision",
        "## Candidate ranking",
        "## Human evaluation",
        "## Quality flags",
    ):
        assert legacy_heading not in markdown


def test_dynamic_catalog_submission_supports_multiple_reviews_and_validates_units(
    tmp_path: Path,
    representative_frame: pd.DataFrame,
    opinion_frame: pd.DataFrame,
    static_encoder: StaticEncoder,
    config_values: dict,
) -> None:
    run_dir, reviews_path, opinion_path = _build_report_fixture(
        tmp_path,
        representative_frame,
        opinion_frame,
        static_encoder,
        config_values,
    )
    inputs = load_report_inputs(run_dir, reviews_path, opinion_path)
    submission = build_submission_from_catalog_reviews(
        inputs,
        [11, 12],
        submitted_at_local="20260804-120001",
    )
    proposal = build_dynamic_review_decision_report(inputs, submission)

    assert [review["source_review_idx"] for review in proposal["submission"]["reviews"]] == [11, 12]
    assert proposal["submission"]["excluded_products"] == ["Product B"]
    price_relationship = next(
        row for row in proposal["catalog_relationships"] if row["aspect"] == "가격"
    )
    assert price_relationship["other_status_review_count"] == 0
    assert price_relationship["relation_code"] == "NOT_MENTIONED_BY_OTHER_REVIEWS"

    invalid_submission = deepcopy(submission)
    invalid_submission["reviews"][0]["opinion_units"][0]["raw_aspect"] = "배송 상태"
    with pytest.raises(ValueError, match="not a product attribute"):
        build_dynamic_review_decision_report(inputs, invalid_submission)

    mismatched_submission = deepcopy(submission)
    mismatched_submission["reviews"][0]["opinion_units"][0]["aspect"] = "가격"
    with pytest.raises(ValueError, match="different canonical labels"):
        build_dynamic_review_decision_report(inputs, mismatched_submission)


def test_completed_human_evaluation_is_integrated_and_explained(
    tmp_path: Path,
    representative_frame: pd.DataFrame,
    opinion_frame: pd.DataFrame,
    static_encoder: StaticEncoder,
    config_values: dict,
) -> None:
    run_dir, reviews_path, opinion_path = _build_report_fixture(
        tmp_path,
        representative_frame,
        opinion_frame,
        static_encoder,
        config_values,
    )
    _write_completed_human_results(run_dir)
    inputs = load_report_inputs(run_dir, reviews_path, opinion_path)
    report = build_static_catalog_report(inputs, "Product B")

    human = report["human_evaluation"]
    assert human["status"] == "completed"
    assert human["validation"]["evaluator_count"] == 3
    assert human["validation"]["review_task_count"] == 2
    assert human["conclusion"]["structured_representation"] == "SUPPORTED"
    assert human["conclusion"]["clustering_review_quality"] == "NO_CLEAR_MAIN_EFFECT"
    incomplete = next(
        flag for flag in report["quality_flags"] if flag["code"] == "HUMAN_EVALUATION_INCOMPLETE"
    )
    assert incomplete["value"] is False
    assert report["source"]["human_evaluation"]["results_set_sha256"]

    static_markdown = render_static_markdown(report)
    assert "리뷰를 aspect와 status 조합 단위로 분석한 표입니다." in static_markdown
    assert "## 관련 상품" in static_markdown
    assert "## Human evaluation" not in static_markdown
    assert "## Quality flags" not in static_markdown


def test_human_evaluation_fails_closed_below_minimum_evaluator_count(
    tmp_path: Path,
    representative_frame: pd.DataFrame,
    opinion_frame: pd.DataFrame,
    static_encoder: StaticEncoder,
    config_values: dict,
) -> None:
    run_dir, reviews_path, opinion_path = _build_report_fixture(
        tmp_path,
        representative_frame,
        opinion_frame,
        static_encoder,
        config_values,
    )
    _write_completed_human_results(run_dir, evaluator_count=2)

    with pytest.raises(ValueError, match="Need at least 3 completed evaluators"):
        load_report_inputs(run_dir, reviews_path, opinion_path)


def test_loader_rejects_manifest_identity_mismatches(
    tmp_path: Path,
    representative_frame: pd.DataFrame,
    opinion_frame: pd.DataFrame,
    static_encoder: StaticEncoder,
    config_values: dict,
) -> None:
    run_dir, reviews_path, opinion_path = _build_report_fixture(
        tmp_path,
        representative_frame,
        opinion_frame,
        static_encoder,
        config_values,
    )
    altered_opinion = opinion_frame.copy()
    altered_opinion.loc[0, "opinion"] = "changed"
    altered_path = tmp_path / "altered_opinion.parquet"
    altered_opinion.to_parquet(altered_path, index=False)
    with pytest.raises(ValueError, match="opinion input SHA-256 differs"):
        load_report_inputs(run_dir, reviews_path, altered_path)

    summary_path = run_dir / "evaluation/automatic_evaluation_summary.json"
    summary_path.write_text(summary_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation summary SHA-256 differs"):
        load_report_inputs(run_dir, reviews_path, opinion_path)


def test_sentiment_collapse_and_wilson_bounds() -> None:
    assert collapse_sentiments(pd.Series(["unknown", "unknown"])) == "unknown"
    assert collapse_sentiments(pd.Series(["positive", "positive"])) == "positive"
    assert collapse_sentiments(pd.Series(["positive", "negative"])) == "mixed"
    interval = wilson_interval(1, 1)
    assert interval["lower"] == 0.206549
    assert interval["upper"] == 1.0
    assert json.loads(json.dumps(interval)) == interval
