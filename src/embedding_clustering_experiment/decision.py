"""Deterministic review-conditioned comparison and recommendation proposals."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

from .reporting import (
    REPORT_SCHEMA_VERSION,
    SENTIMENT_ORDER,
    ReportInputs,
    ReportSettings,
    attach_review_votes,
    build_sentiment_votes,
    markdown_cell,
    product_review_counts,
    quality_flags,
    render_human_evaluation_markdown,
    risk_lookup,
    round_float,
    select_evidence,
    sentiment_distribution,
    source_metadata,
)

ACTIONABLE_SENTIMENTS = {"positive", "negative"}


def build_request_from_review(
    inputs: ReportInputs,
    review_idx: int,
    *,
    exclude_source_product: bool = True,
) -> dict[str, Any]:
    """Use previously extracted D Opinion Units as a deterministic demo request."""
    rows = inputs.joined.loc[inputs.joined["review_idx"].eq(review_idx)].sort_values(
        "idx", kind="stable"
    )
    if rows.empty:
        raise ValueError(f"review_idx={review_idx} has no eligible Experiment D rows.")
    product_names = rows["product_name"].astype(str).unique()
    review_texts = rows["review"].astype(str).unique()
    if len(product_names) != 1 or len(review_texts) != 1:
        raise AssertionError("A review_idx must resolve to one product and one review text.")
    requirements: list[dict[str, Any]] = []
    for number, row in enumerate(rows.itertuples(index=False), start=1):
        requirements.append(
            {
                "requirement_id": f"r{number}",
                "aspect": str(row.aspect),
                "status": None if pd.isna(row.status) else str(row.status),
                "sentiment": str(row.sentiment),
                "aspect_cluster_id": str(row.aspect_cluster_id),
                "status_cluster_id": (
                    None if pd.isna(row.status_cluster_id) else str(row.status_cluster_id)
                ),
                "importance": 1.0,
                "excerpt": str(row.excerpt),
            }
        )
    source_product = str(product_names[0])
    return {
        "request_id": f"demo-review-{review_idx}",
        "source_review_idx": int(review_idx),
        "source_review": str(review_texts[0]),
        "source_product_name": source_product,
        "requirements": requirements,
        "excluded_products": [source_product] if exclude_source_product else [],
    }


def _label_indexes(
    inputs: ReportInputs,
) -> tuple[
    dict[str, set[str]],
    dict[tuple[str, str], set[str]],
    dict[str, str],
    dict[str, dict[str, str]],
]:
    aspect_index: dict[str, set[str]] = {}
    status_index: dict[tuple[str, str], set[str]] = {}
    aspect_by_id: dict[str, str] = {}
    status_by_id: dict[str, dict[str, str]] = {}
    pairs = inputs.experiment_d[
        ["aspect_cluster_id", "status_cluster_id", "aspect", "status"]
    ].drop_duplicates()
    for row in pairs.itertuples(index=False):
        aspect_id = str(row.aspect_cluster_id)
        aspect_label = str(row.aspect)
        aspect_index.setdefault(aspect_label, set()).add(aspect_id)
        aspect_by_id[aspect_id] = aspect_label
        if pd.notna(row.status) and pd.notna(row.status_cluster_id):
            status_label = str(row.status)
            status_id = str(row.status_cluster_id)
            status_index.setdefault((aspect_id, status_label), set()).add(status_id)
            status_by_id[status_id] = {
                "aspect_cluster_id": aspect_id,
                "status": status_label,
            }
    return aspect_index, status_index, aspect_by_id, status_by_id


def _normalize_request(
    inputs: ReportInputs,
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    request_id = str(request.get("request_id", "")).strip()
    if not request_id:
        raise ValueError("request_id must be a non-empty string.")
    raw_requirements = request.get("requirements")
    if not isinstance(raw_requirements, list) or not raw_requirements:
        raise ValueError("requirements must be a non-empty list.")
    excluded = request.get("excluded_products", [])
    if not isinstance(excluded, list):
        raise TypeError("excluded_products must be a list.")
    aspect_index, status_index, aspect_by_id, status_by_id = _label_indexes(inputs)
    valid_aspect_ids = set(aspect_by_id)
    valid_status_ids = set(status_by_id)
    normalized_requirements: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for position, raw in enumerate(raw_requirements, start=1):
        if not isinstance(raw, Mapping):
            raise TypeError(f"requirements[{position - 1}] must be an object.")
        requirement_id = str(raw.get("requirement_id", f"r{position}")).strip()
        if not requirement_id or requirement_id in seen_ids:
            raise ValueError("requirement_id values must be unique and non-empty.")
        seen_ids.add(requirement_id)
        aspect = str(raw.get("aspect", "")).strip()
        status_value = raw.get("status")
        status_text = None if status_value is None else str(status_value).strip()
        status = status_text or None
        sentiment = str(raw.get("sentiment", "")).strip()
        if not aspect:
            raise ValueError(f"{requirement_id}.aspect must be non-empty.")
        if sentiment not in SENTIMENT_ORDER:
            raise ValueError(f"{requirement_id}.sentiment is invalid: {sentiment!r}")
        importance_value = raw.get("importance", 1.0)
        if isinstance(importance_value, bool):
            raise TypeError(f"{requirement_id}.importance must be a positive number.")
        importance = float(importance_value)
        if not np.isfinite(importance) or importance <= 0:
            raise ValueError(f"{requirement_id}.importance must be a positive number.")

        aspect_id_value = raw.get("aspect_cluster_id")
        aspect_id = None if aspect_id_value is None else str(aspect_id_value).strip()
        if not aspect_id:
            candidates = aspect_index.get(aspect, set())
            if len(candidates) == 1:
                aspect_id = next(iter(candidates))
            else:
                unresolved.append(
                    {
                        "requirement_id": requirement_id,
                        "reason": (
                            "ASPECT_LABEL_NOT_FOUND" if not candidates else "ASPECT_LABEL_AMBIGUOUS"
                        ),
                        "candidate_cluster_ids": sorted(candidates),
                    }
                )
        elif aspect_id not in valid_aspect_ids:
            unresolved.append(
                {
                    "requirement_id": requirement_id,
                    "reason": "ASPECT_CLUSTER_ID_NOT_FOUND",
                    "candidate_cluster_ids": [],
                }
            )
        elif aspect_by_id[aspect_id] != aspect:
            unresolved.append(
                {
                    "requirement_id": requirement_id,
                    "reason": "ASPECT_LABEL_CLUSTER_MISMATCH",
                    "candidate_cluster_ids": sorted(aspect_index.get(aspect, set())),
                    "cluster_canonical_label": aspect_by_id[aspect_id],
                }
            )

        status_id_value = raw.get("status_cluster_id")
        status_id = None if status_id_value is None else str(status_id_value).strip()
        if aspect_id and status and not status_id:
            candidates = status_index.get((aspect_id, status), set())
            if len(candidates) == 1:
                status_id = next(iter(candidates))
            else:
                unresolved.append(
                    {
                        "requirement_id": requirement_id,
                        "reason": (
                            "STATUS_LABEL_NOT_FOUND" if not candidates else "STATUS_LABEL_AMBIGUOUS"
                        ),
                        "candidate_cluster_ids": sorted(candidates),
                    }
                )
        elif status_id and status_id not in valid_status_ids:
            unresolved.append(
                {
                    "requirement_id": requirement_id,
                    "reason": "STATUS_CLUSTER_ID_NOT_FOUND",
                    "candidate_cluster_ids": [],
                }
            )
        elif status_id:
            status_record = status_by_id[status_id]
            if aspect_id and status_record["aspect_cluster_id"] != aspect_id:
                unresolved.append(
                    {
                        "requirement_id": requirement_id,
                        "reason": "STATUS_CLUSTER_ASPECT_MISMATCH",
                        "candidate_cluster_ids": [],
                        "cluster_aspect_id": status_record["aspect_cluster_id"],
                    }
                )
            if status is None:
                status = status_record["status"]
            elif status_record["status"] != status:
                unresolved.append(
                    {
                        "requirement_id": requirement_id,
                        "reason": "STATUS_LABEL_CLUSTER_MISMATCH",
                        "candidate_cluster_ids": sorted(
                            status_index.get((aspect_id or "", status), set())
                        ),
                        "cluster_canonical_label": status_record["status"],
                    }
                )
        if (
            sentiment in ACTIONABLE_SENTIMENTS
            and (not aspect_id or not status_id)
            and not any(item["requirement_id"] == requirement_id for item in unresolved)
        ):
            unresolved.append(
                {
                    "requirement_id": requirement_id,
                    "reason": "ACTIONABLE_REQUIREMENT_HAS_NO_RESOLVED_PAIR",
                    "candidate_cluster_ids": [],
                }
            )
        normalized_requirements.append(
            {
                "requirement_id": requirement_id,
                "aspect": aspect,
                "status": status,
                "sentiment": sentiment,
                "aspect_cluster_id": aspect_id,
                "status_cluster_id": status_id,
                "importance": importance,
                "excerpt": (None if raw.get("excerpt") is None else str(raw.get("excerpt"))),
            }
        )
    normalized = {
        "request_id": request_id,
        "source_review_idx": request.get("source_review_idx"),
        "source_review": request.get("source_review"),
        "source_product_name": request.get("source_product_name"),
        "requirements": normalized_requirements,
        "excluded_products": sorted({str(product) for product in excluded}),
    }
    return normalized, unresolved


def _pair_profiles(
    inputs: ReportInputs,
    settings: ReportSettings,
) -> list[dict[str, Any]]:
    votes = build_sentiment_votes(
        inputs.joined,
        ["aspect_cluster_id", "status_cluster_id", "aspect", "status"],
    )
    rows: list[dict[str, Any]] = []
    keys = [
        "product_name",
        "aspect_cluster_id",
        "status_cluster_id",
        "aspect",
        "status",
    ]
    for key, group in votes.groupby(
        keys,
        observed=True,
        dropna=False,
        sort=True,
    ):
        record = dict(zip(keys, key, strict=True))
        counts = {name: int(group["vote"].eq(name).sum()) for name in SENTIMENT_ORDER}
        support = len(group)
        rows.append(
            {
                "product_name": str(record["product_name"]),
                "aspect_cluster_id": str(record["aspect_cluster_id"]),
                "status_cluster_id": (
                    None
                    if pd.isna(record["status_cluster_id"])
                    else str(record["status_cluster_id"])
                ),
                "aspect": str(record["aspect"]),
                "status": None if pd.isna(record["status"]) else str(record["status"]),
                "supporting_review_count": support,
                "support_reliability": round_float(
                    min(1.0, support / settings.minimum_support_reviews),
                    settings.decimal_places,
                ),
                "sentiment": sentiment_distribution(
                    counts,
                    decimal_places=settings.decimal_places,
                ),
            }
        )
    return rows


def _status_vectors(inputs: ReportInputs) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    dimensions: set[int] = set()
    for row in inputs.status_clusters.itertuples(index=False):
        vector = np.asarray(row.centroid_embedding, dtype=float)
        if (
            vector.ndim != 1
            or vector.size == 0
            or not np.isfinite(vector).all()
            or np.linalg.norm(vector) == 0
        ):
            raise ValueError(f"Invalid status centroid embedding: {row.cluster_id}")
        vectors[str(row.cluster_id)] = vector
        dimensions.add(vector.size)
    if len(dimensions) != 1:
        raise ValueError("Status centroid embeddings have inconsistent dimensions.")
    return vectors


def _profile_evidence(
    inputs: ReportInputs,
    profile: dict[str, Any] | None,
    settings: ReportSettings,
) -> dict[str, Any] | None:
    if profile is None:
        return None
    frame = inputs.joined.loc[
        inputs.joined["product_name"].eq(profile["product_name"])
        & inputs.joined["aspect_cluster_id"].eq(profile["aspect_cluster_id"])
        & inputs.joined["status_cluster_id"].eq(profile["status_cluster_id"])
    ]
    frame = attach_review_votes(
        frame,
        ["aspect_cluster_id", "status_cluster_id"],
    )
    risks = risk_lookup(inputs, settings.decimal_places)
    return {
        "aspect_cluster_id": profile["aspect_cluster_id"],
        "status_cluster_id": profile["status_cluster_id"],
        "aspect": profile["aspect"],
        "status": profile["status"],
        "supporting_review_count": int(profile["supporting_review_count"]),
        "support_reliability": profile["support_reliability"],
        "sentiment": profile["sentiment"],
        "evidence": {
            name: select_evidence(
                frame,
                name,
                limit=settings.evidence_per_sentiment,
                vote_column="review_vote_sentiment",
                decimal_places=settings.decimal_places,
            )
            for name in SENTIMENT_ORDER
        },
        "normalization_risk": {
            "aspect": risks.get(profile["aspect_cluster_id"]),
            "status": risks.get(profile["status_cluster_id"]),
        },
    }


def _conflicts(requirements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sentiments_by_pair: dict[tuple[str | None, str | None], set[str]] = {}
    for item in requirements:
        pair = (item["aspect_cluster_id"], item["status_cluster_id"])
        if pair[0] is None or pair[1] is None:
            continue
        sentiments_by_pair.setdefault(pair, set()).add(item["sentiment"])
    conflicts = []
    for pair, sentiments in sentiments_by_pair.items():
        if {"positive", "negative"} <= sentiments:
            conflicts.append(
                {
                    "aspect_cluster_id": pair[0],
                    "status_cluster_id": pair[1],
                    "sentiments": sorted(sentiments),
                }
            )
    return conflicts


def _score_candidate_requirement(
    inputs: ReportInputs,
    candidate_product: str,
    item: dict[str, Any],
    profiles: list[dict[str, Any]],
    vectors: dict[str, np.ndarray],
    settings: ReportSettings,
) -> tuple[float, bool, dict[str, Any]]:
    aspect_profiles = [
        profile
        for profile in profiles
        if profile["product_name"] == candidate_product
        and profile["aspect_cluster_id"] == item["aspect_cluster_id"]
        and profile["status_cluster_id"] is not None
    ]
    target_vector = vectors.get(str(item["status_cluster_id"]))
    near_options: list[tuple[float, float, dict[str, Any]]] = []
    alternative_options: list[tuple[float, dict[str, Any]]] = []
    for profile in aspect_profiles:
        is_exact_status_match = profile["status_cluster_id"] == item["status_cluster_id"]
        candidate_vector = vectors.get(str(profile["status_cluster_id"]))
        if is_exact_status_match:
            # Exact canonical IDs remain an exact match even if an embedding vector
            # is unavailable; this is required for the static weakness-repair path.
            distance = 0.0
        elif target_vector is None or candidate_vector is None:
            distance = None
        else:
            distance = float(cosine_distances([target_vector], [candidate_vector])[0, 0])
        if is_exact_status_match or (
            settings.allow_near_status_match
            and distance is not None
            and distance <= settings.status_near_distance
        ):
            semantic_factor = max(
                0.0,
                1.0 - distance / settings.status_near_distance,
            )
            near_options.append(
                (
                    semantic_factor * float(profile["support_reliability"]),
                    distance,
                    profile,
                )
            )
        else:
            positive_lower = profile["sentiment"]["positive_wilson_95"]["lower"] or 0.0
            if profile["sentiment"]["dominant_sentiment"] == "positive":
                alternative_options.append(
                    (
                        float(profile["support_reliability"]) * float(positive_lower),
                        profile,
                    )
                )
    near_options.sort(
        key=lambda value: (
            -value[0],
            value[1],
            value[2]["status"] or "",
            value[2]["status_cluster_id"],
        )
    )
    alternative_options.sort(
        key=lambda value: (
            -value[0],
            -value[1]["supporting_review_count"],
            value[1]["status"] or "",
            value[1]["status_cluster_id"],
        )
    )
    near = near_options[0] if near_options else None
    alternative = alternative_options[0] if alternative_options else None
    contribution = 0.0
    relation = "NO_EVIDENCE"
    matched_profile = None
    alternative_profile = None
    semantic_distance = None
    supported = False
    if item["sentiment"] == "positive":
        if near:
            strength, semantic_distance, matched_profile = near
            positive_lower = matched_profile["sentiment"]["positive_wilson_95"]["lower"] or 0.0
            contribution = float(item["importance"]) * strength * float(positive_lower)
            relation = (
                "EXACT_OR_NEAR_PREFERRED_STATUS"
                if settings.allow_near_status_match
                else "EXACT_PREFERRED_STATUS"
            )
            supported = True
    else:
        penalty = near[0] if near else 0.0
        if near:
            _, semantic_distance, matched_profile = near
        alternative_score = alternative[0] if alternative else 0.0
        alternative_profile = alternative[1] if alternative else None
        contribution = float(item["importance"]) * (alternative_score - penalty)
        if near and alternative:
            relation = (
                "UNDESIRED_STATUS_AND_POSITIVE_ALTERNATIVE"
                if settings.allow_near_status_match
                else "EXACT_UNDESIRED_STATUS_AND_POSITIVE_ALTERNATIVE"
            )
            supported = True
        elif near:
            relation = (
                "UNDESIRED_STATUS_ONLY"
                if settings.allow_near_status_match
                else "EXACT_UNDESIRED_STATUS_ONLY"
            )
            supported = True
        elif alternative:
            relation = "POSITIVE_ALTERNATIVE_ONLY"
            supported = True
    return (
        contribution,
        supported,
        {
            "requirement_id": item["requirement_id"],
            "relation": relation,
            "contribution": round_float(contribution, settings.decimal_places),
            "semantic_distance_to_matched_status": round_float(
                semantic_distance,
                settings.decimal_places,
            ),
            "matched_status_profile": _profile_evidence(inputs, matched_profile, settings),
            "positive_alternative_profile": _profile_evidence(
                inputs, alternative_profile, settings
            ),
        },
    )


def build_dynamic_decision_proposal(
    inputs: ReportInputs,
    request: Mapping[str, Any],
    *,
    settings: ReportSettings | None = None,
) -> dict[str, Any]:
    """Rank catalog candidates from explicit review-conditioned evidence."""
    selected_settings = settings or ReportSettings()
    normalized_request, unresolved = _normalize_request(inputs, request)
    unresolved_ids = {item["requirement_id"] for item in unresolved}
    declared_actionable = [
        item
        for item in normalized_request["requirements"]
        if item["sentiment"] in ACTIONABLE_SENTIMENTS
    ]
    actionable = [
        item for item in declared_actionable if item["requirement_id"] not in unresolved_ids
    ]
    unresolved_actionable = [
        item for item in declared_actionable if item["requirement_id"] in unresolved_ids
    ]
    conflicts = _conflicts(normalized_request["requirements"])
    profiles = _pair_profiles(inputs, selected_settings)
    vectors = _status_vectors(inputs)
    counts = product_review_counts(inputs)
    excluded = set(normalized_request["excluded_products"])
    candidates: list[dict[str, Any]] = []
    total_importance = sum(float(item["importance"]) for item in actionable)
    for candidate_product in sorted(counts):
        if candidate_product in excluded:
            continue
        matches: list[dict[str, Any]] = []
        weighted_sum = 0.0
        supported_count = 0
        for item in actionable:
            contribution, supported, match = _score_candidate_requirement(
                inputs,
                candidate_product,
                item,
                profiles,
                vectors,
                selected_settings,
            )
            weighted_sum += contribution
            supported_count += int(supported)
            matches.append(match)
        score = weighted_sum / total_importance if total_importance else 0.0
        coverage = supported_count / len(actionable) if actionable else 0.0
        candidates.append(
            {
                "product_name": candidate_product,
                "catalog_review_count": counts[candidate_product],
                "score": round_float(score, selected_settings.decimal_places),
                "actionable_requirement_count": len(actionable),
                "supported_requirement_count": supported_count,
                "evidence_coverage_rate": round_float(coverage, selected_settings.decimal_places),
                "requirement_matches": matches,
            }
        )
    candidates.sort(key=lambda row: (-row["score"], row["product_name"]))
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank

    top = candidates[0] if candidates else None
    runner_up = candidates[1] if len(candidates) > 1 else None
    margin = top["score"] - runner_up["score"] if top and runner_up else None
    if not actionable or unresolved_actionable or conflicts:
        action = "ASK_CLARIFICATION"
        reason_codes = []
        if not actionable:
            reason_codes.append("NO_RESOLVED_ACTIONABLE_REQUIREMENT")
        if unresolved_actionable:
            reason_codes.append("UNRESOLVED_ACTIONABLE_REQUIREMENT")
        if conflicts:
            reason_codes.append("CONFLICTING_REQUIREMENTS")
    elif top is None:
        action = "ABSTAIN"
        reason_codes = ["NO_CANDIDATE"]
    elif (
        top["score"] < selected_settings.recommend_score_threshold
        or top["evidence_coverage_rate"] < selected_settings.minimum_requirement_coverage
    ):
        action = "ABSTAIN"
        reason_codes = []
        if top["score"] < selected_settings.recommend_score_threshold:
            reason_codes.append("TOP_SCORE_BELOW_THRESHOLD")
        if top["evidence_coverage_rate"] < selected_settings.minimum_requirement_coverage:
            reason_codes.append("REQUIREMENT_COVERAGE_BELOW_THRESHOLD")
    elif margin is not None and margin <= selected_settings.compare_margin:
        action = "COMPARE"
        reason_codes = [
            "TOP_SCORE_ABOVE_THRESHOLD",
            "REQUIREMENT_COVERAGE_SUFFICIENT",
            "TOP_CANDIDATES_WITHIN_COMPARE_MARGIN",
        ]
    else:
        action = "RECOMMEND"
        reason_codes = [
            "TOP_SCORE_ABOVE_THRESHOLD",
            "REQUIREMENT_COVERAGE_SUFFICIENT",
            (
                "RUNNER_UP_MARGIN_EXCEEDS_COMPARE_THRESHOLD"
                if runner_up is not None
                else "NO_RUNNER_UP_AVAILABLE"
            ),
        ]

    identity_payload = {
        "request": normalized_request,
        "human_evaluation_status": inputs.human_evaluation["status"],
        "human_evaluation_results_set_sha256": inputs.human_evaluation["source"][
            "results_set_sha256"
        ],
        "settings": asdict(selected_settings),
    }
    proposal_digest = hashlib.sha256(
        json.dumps(
            identity_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:12]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "proposal_type": "dynamic_decision_proposal",
        "proposal_id": (
            f"dynamic_decision_proposal:{inputs.run_dir.name}:"
            f"{normalized_request['request_id']}:{proposal_digest}"
        ),
        "source": source_metadata(inputs),
        "human_evaluation": inputs.human_evaluation,
        "request": normalized_request,
        "request_validation": {
            "actionable_requirement_count": len(actionable),
            "declared_actionable_requirement_count": len(declared_actionable),
            "resolved_actionable_requirement_count": len(actionable),
            "unresolved_actionable_requirement_count": len(unresolved_actionable),
            "unresolved_requirements": unresolved,
            "conflicts": conflicts,
        },
        "scoring_contract": {
            "minimum_support_reviews": selected_settings.minimum_support_reviews,
            "status_near_distance": selected_settings.status_near_distance,
            "allow_near_status_match": selected_settings.allow_near_status_match,
            "status_match_policy": (
                "exact_or_embedding_near_status"
                if selected_settings.allow_near_status_match
                else "exact_status_cluster_id_only"
            ),
            "recommend_score_threshold": selected_settings.recommend_score_threshold,
            "compare_margin": selected_settings.compare_margin,
            "minimum_requirement_coverage": (selected_settings.minimum_requirement_coverage),
            "evidence_per_sentiment": selected_settings.evidence_per_sentiment,
            "decimal_places": selected_settings.decimal_places,
            "sentiment_vote_grain": [
                "product_name",
                "review_idx",
                "aspect_cluster_id",
                "status_cluster_id",
            ],
            "positive_formula": (
                "importance * semantic_factor * support_reliability * positive_wilson_lower_95"
            ),
            "negative_formula": ("importance * (positive_alternative - undesired_penalty)"),
            "absence_policy": "NO_EVIDENCE",
        },
        "decision": {
            "action": action,
            "top_product_name": top["product_name"] if top else None,
            "top_score": top["score"] if top else None,
            "runner_up_product_name": runner_up["product_name"] if runner_up else None,
            "runner_up_score": runner_up["score"] if runner_up else None,
            "runner_up_margin": round_float(margin, selected_settings.decimal_places),
            "reason_codes": reason_codes,
        },
        "candidates": candidates,
        "quality_flags": quality_flags(
            inputs,
            product_name=None,
            decimal_places=selected_settings.decimal_places,
        ),
    }


def render_dynamic_markdown(proposal: dict[str, Any]) -> str:
    """Render decision tables with deterministic, evidence-derived interpretation."""
    validation = proposal["request_validation"]
    excluded = proposal["request"]["excluded_products"]
    lines = [
        "# 동적 의사결정 제안서 예시",
        "",
        f"`proposal_id`: `{proposal['proposal_id']}`",
        "",
        "## Request",
        "",
        (
            f"선언된 actionable 요구 {validation['declared_actionable_requirement_count']}개 중 "
            f"{validation['resolved_actionable_requirement_count']}개를 정규화된 aspect-status에 "
            f"연결했습니다. 비교 후보에서 제외한 상품은 "
            f"{', '.join(excluded) if excluded else '없습니다'}."
        ),
        "",
        "| id | aspect | status | sentiment | importance |",
        "|---|---|---|---|---:|",
    ]
    for item in proposal["request"]["requirements"]:
        lines.append(
            f"| {markdown_cell(item['requirement_id'])} | {markdown_cell(item['aspect'])} | "
            f"{markdown_cell(item['status'])} | {markdown_cell(item['sentiment'])} | "
            f"{item['importance']} |"
        )
    decision = proposal["decision"]
    scoring = proposal["scoring_contract"]
    lines.extend(
        [
            "",
            (
                "요구 사항은 positive/negative만 의사결정 점수에 반영하며, 연결되지 않은 "
                "actionable 요구나 충돌은 추천 대신 명확화 요청을 발생시킵니다."
            ),
            "",
            "## Decision",
            "",
            (
                f"추천 임계값은 {scoring['recommend_score_threshold']}, 최소 근거 커버리지는 "
                f"{scoring['minimum_requirement_coverage']}, 비교 전환 마진은 "
                f"{scoring['compare_margin']}입니다. 점수는 리뷰 근거 기반 의사결정 점수이지 "
                "구매 성공 확률이 아닙니다."
            ),
            "",
            "| action | top_product | top_score | runner_up | margin |",
            "|---|---|---:|---|---:|",
            (
                f"| {markdown_cell(decision['action'])} | "
                f"{markdown_cell(decision['top_product_name'])} | "
                f"{decision['top_score']} | {markdown_cell(decision['runner_up_product_name'])} | "
                f"{decision['runner_up_margin']} |"
            ),
            "",
            (
                f"판정은 {decision['action']}이며 근거 코드는 "
                f"{', '.join(decision['reason_codes']) if decision['reason_codes'] else '없음'}입니다."
            ),
            "",
            "## Candidate ranking",
            "",
            (
                f"후보 {len(proposal['candidates'])}개를 동일한 요구·임계값으로 정렬했습니다. "
                "evidence_coverage는 해결된 actionable 요구 중 제품 근거가 존재하는 비율입니다."
            ),
            "",
            "| rank | product | score | evidence_coverage | relation | positive_alternative |",
            "|---:|---|---:|---:|---|---|",
        ]
    )
    for candidate in proposal["candidates"]:
        relations = "; ".join(
            f"{match['requirement_id']}:{match['relation']}"
            for match in candidate["requirement_matches"]
        )
        alternatives = "; ".join(
            f"{match['requirement_id']}:"
            f"{(match.get('positive_alternative_profile') or {}).get('status')}"
            for match in candidate["requirement_matches"]
            if (match.get("positive_alternative_profile") or {}).get("status")
        )
        lines.append(
            f"| {candidate['rank']} | {markdown_cell(candidate['product_name'])} | "
            f"{candidate['score']} | {candidate['evidence_coverage_rate']} | "
            f"{markdown_cell(relations)} | {markdown_cell(alternatives or None)} |"
        )
    if proposal["candidates"]:
        top = proposal["candidates"][0]
        lines.extend(
            [
                "",
                (
                    f"1위 {top['product_name']}의 점수는 {top['score']}, 근거 커버리지는 "
                    f"{top['evidence_coverage_rate']}입니다. NO_EVIDENCE는 중립 근거가 아니라 "
                    "해당 요구를 뒷받침하거나 반박할 리뷰 근거가 없다는 뜻입니다."
                ),
            ]
        )
    lines.extend(["", render_human_evaluation_markdown(proposal["human_evaluation"])])
    warning_codes = [
        flag["code"] for flag in proposal["quality_flags"] if flag["severity"] == "warning"
    ]
    lines.extend(
        [
            "",
            "## Quality flags",
            "",
            (
                f"현재 경고 {len(warning_codes)}개는 "
                f"{', '.join(warning_codes) if warning_codes else '없음'}입니다. "
                "이 표는 추천 결과를 사용할 수 있는 근거 범위를 명시합니다."
            ),
            "",
            "| code | severity | value |",
            "|---|---|---|",
        ]
    )
    for flag in proposal["quality_flags"]:
        lines.append(
            f"| {markdown_cell(flag['code'])} | {markdown_cell(flag['severity'])} | "
            f"`{markdown_cell(flag['value'])}` |"
        )
    lines.extend(
        [
            "",
            "리뷰 시점과 가격·재고·배송·객관 사양이 없으므로 이 제안은 리뷰 경험 조건부 비교로만 해석해야 합니다.",
        ]
    )
    return "\n".join(lines) + "\n"
