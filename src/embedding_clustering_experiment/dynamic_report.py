"""Deterministic review-submission analysis for the shopping-Agent demo.

This module intentionally does not use the legacy requirement-scoring proposal.
It treats the submitted review Opinion Units as the primary observation, compares
each canonical aspect/status pair with the source product's catalog reviews, and
ranks alternatives with the static catalog report's 3:1 repair-utility/similarity
combination.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from .reporting import (
    RELATED_PRODUCT_EXPERIENCE_WEIGHT,
    RELATED_PRODUCT_LIMIT,
    RELATED_PRODUCT_MINIMUM_WEAKNESS_SUPPORT,
    RELATED_PRODUCT_REPAIR_UTILITY_WEIGHT,
    SENTIMENT_ORDER,
    ReportInputs,
    _related_product_profile_data,
    _related_similarity_rows,
    build_sentiment_votes,
    collapse_sentiments,
    markdown_bold,
    markdown_cell,
    product_review_counts,
    round_float,
    sentiment_distribution,
    source_metadata,
)

DYNAMIC_REVIEW_REPORT_SCHEMA_VERSION = "1.0.0"
DYNAMIC_REVIEW_REPORT_TYPE = "dynamic_review_decision_proposal"
OTHER_ASPECT_STATUS_TOP_N = 3
GENERAL_PRODUCT_EXPERIENCE = "전반적 상품 경험"
NON_PRODUCT_ASPECT_TERMS = (
    "배송",
    "배달",
    "포장",
    "서비스",
    "사은품",
    "증정",
    "쿠폰",
)


@dataclass(frozen=True)
class DynamicReportSettings:
    """Settings that affect the submitted-review report only."""

    decimal_places: int = 6

    def __post_init__(self) -> None:
        if self.decimal_places < 0:
            raise ValueError("decimal_places must be non-negative.")


def _optional_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _required_text(value: Any, *, field: str, context: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"{context}.{field} must be a non-empty string.")
    return text


def _catalog_submission_timestamp(inputs: ReportInputs) -> str:
    created_at = str(source_metadata(inputs)["created_at_local"])
    matched = re.match(
        r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})",
        created_at,
    )
    if not matched:
        raise ValueError(f"Cannot derive a submission timestamp from {created_at!r}.")
    year, month, day, hour, minute, second = matched.groups()
    return f"{year}{month}{day}-{hour}{minute}{second}"


def build_submission_from_catalog_reviews(
    inputs: ReportInputs,
    review_indices: Sequence[int],
    *,
    submitted_at_local: str | None = None,
) -> dict[str, Any]:
    """Create a deterministic demo submission from previously normalized D rows."""
    requested_indices = sorted({int(review_idx) for review_idx in review_indices})
    if not requested_indices:
        raise ValueError("review_indices must contain at least one review_idx.")
    rows = inputs.joined.loc[inputs.joined["review_idx"].isin(requested_indices)].copy()
    found_indices = set(map(int, rows["review_idx"].unique()))
    missing_indices = sorted(set(requested_indices) - found_indices)
    if missing_indices:
        raise ValueError(
            "No eligible Experiment D Opinion Units for review_idx values: "
            f"{', '.join(map(str, missing_indices))}."
        )
    product_names = sorted(set(map(str, rows["product_name"].unique())))
    if len(product_names) != 1:
        raise ValueError("Submitted catalog reviews must belong to exactly one product.")
    product_name = product_names[0]
    reviews: list[dict[str, Any]] = []
    for review_idx in requested_indices:
        review_rows = rows.loc[rows["review_idx"].eq(review_idx)].sort_values("idx", kind="stable")
        review_texts = review_rows["review"].astype(str).unique()
        if len(review_texts) != 1:
            raise AssertionError("A review_idx must resolve to exactly one review text.")
        opinion_units: list[dict[str, Any]] = []
        for row in review_rows.itertuples(index=False):
            opinion_units.append(
                {
                    "raw_aspect": str(row.raw_aspect),
                    "aspect": str(row.aspect),
                    "raw_status": _optional_text(row.raw_status),
                    "status": _optional_text(row.status),
                    "excerpt": str(row.excerpt),
                    "opinion": str(row.opinion),
                    "sentiment": str(row.sentiment),
                    "aspect_cluster_id": str(row.aspect_cluster_id),
                    "status_cluster_id": _optional_text(row.status_cluster_id),
                }
            )
        reviews.append(
            {
                "source_review_idx": review_idx,
                "review": str(review_texts[0]),
                "opinion_units": opinion_units,
            }
        )
    submission_id = (
        f"demo-review-{requested_indices[0]}"
        if len(requested_indices) == 1
        else "demo-reviews-" + "-".join(map(str, requested_indices))
    )
    return {
        "submission_id": submission_id,
        "submitted_at_local": submitted_at_local or _catalog_submission_timestamp(inputs),
        "product_name": product_name,
        "reviews": reviews,
        "excluded_products": [product_name],
    }


def _normalization_indexes(
    inputs: ReportInputs,
) -> tuple[
    dict[str, set[str]],
    dict[tuple[str, str], set[str]],
    dict[str, str],
    dict[str, dict[str, str]],
]:
    aspect_ids_by_label: dict[str, set[str]] = defaultdict(set)
    status_ids_by_aspect_label: dict[tuple[str, str], set[str]] = defaultdict(set)
    aspect_by_id: dict[str, str] = {}
    status_by_id: dict[str, dict[str, str]] = {}
    canonical_rows = inputs.experiment_d[
        ["aspect_cluster_id", "status_cluster_id", "aspect", "status"]
    ].drop_duplicates()
    for row in canonical_rows.itertuples(index=False):
        aspect_id = str(row.aspect_cluster_id)
        aspect = str(row.aspect)
        aspect_ids_by_label[aspect].add(aspect_id)
        aspect_by_id[aspect_id] = aspect
        status = _optional_text(row.status)
        status_id = _optional_text(row.status_cluster_id)
        if status is not None and status_id is not None:
            status_ids_by_aspect_label[(aspect_id, status)].add(status_id)
            status_by_id[status_id] = {
                "aspect_cluster_id": aspect_id,
                "status": status,
            }
    return aspect_ids_by_label, status_ids_by_aspect_label, aspect_by_id, status_by_id


def _is_non_product_aspect(raw_aspect: str) -> bool:
    normalized = raw_aspect.replace(" ", "")
    return any(term in normalized for term in NON_PRODUCT_ASPECT_TERMS)


def _resolve_opinion_unit(
    raw: Mapping[str, Any],
    *,
    review: str,
    context: str,
    aspect_ids_by_label: dict[str, set[str]],
    status_ids_by_aspect_label: dict[tuple[str, str], set[str]],
    aspect_by_id: dict[str, str],
    status_by_id: dict[str, dict[str, str]],
) -> dict[str, Any]:
    raw_aspect = _required_text(raw.get("raw_aspect"), field="raw_aspect", context=context)
    if _is_non_product_aspect(raw_aspect):
        raise ValueError(
            f"{context}.raw_aspect={raw_aspect!r} is not a product attribute and must be excluded."
        )
    raw_status = _optional_text(raw.get("raw_status"))
    excerpt = _required_text(raw.get("excerpt"), field="excerpt", context=context)
    if excerpt not in review:
        raise ValueError(f"{context}.excerpt must be a contiguous substring of its review.")
    opinion = _required_text(raw.get("opinion"), field="opinion", context=context)
    sentiment = _required_text(raw.get("sentiment"), field="sentiment", context=context)
    if sentiment not in SENTIMENT_ORDER:
        raise ValueError(f"{context}.sentiment is invalid: {sentiment!r}")
    aspect = _required_text(raw.get("aspect", raw_aspect), field="aspect", context=context)
    status = _optional_text(raw.get("status", raw_status))
    is_general_experience = aspect == GENERAL_PRODUCT_EXPERIENCE
    if is_general_experience:
        if status is not None or raw_status is not None:
            raise ValueError(f"{context} general product experience must not have a status.")
        return {
            "raw_aspect": raw_aspect,
            "aspect": aspect,
            "raw_status": None,
            "status": None,
            "excerpt": excerpt,
            "opinion": opinion,
            "sentiment": sentiment,
            "aspect_cluster_id": None,
            "status_cluster_id": None,
            "catalog_comparison_eligible": False,
        }

    aspect_cluster_id = _optional_text(raw.get("aspect_cluster_id"))
    if aspect_cluster_id is None:
        candidates = aspect_ids_by_label.get(aspect, set())
        if len(candidates) != 1:
            reason = "not found" if not candidates else "ambiguous"
            raise ValueError(f"{context}.aspect {aspect!r} is {reason} in the catalog clusters.")
        aspect_cluster_id = next(iter(candidates))
    elif aspect_cluster_id not in aspect_by_id:
        raise ValueError(
            f"{context}.aspect_cluster_id is absent from the catalog: {aspect_cluster_id!r}"
        )
    elif aspect_by_id[aspect_cluster_id] != aspect:
        raise ValueError(
            f"{context}.aspect and aspect_cluster_id resolve to different canonical labels."
        )

    status_cluster_id = _optional_text(raw.get("status_cluster_id"))
    if status is None:
        if status_cluster_id is not None:
            raise ValueError(f"{context}.status_cluster_id requires a non-null status.")
    elif status_cluster_id is None:
        candidates = status_ids_by_aspect_label.get((aspect_cluster_id, status), set())
        if len(candidates) != 1:
            reason = "not found" if not candidates else "ambiguous"
            raise ValueError(
                f"{context}.status {status!r} is {reason} under aspect {aspect!r} in catalog clusters."
            )
        status_cluster_id = next(iter(candidates))
    elif status_cluster_id not in status_by_id:
        raise ValueError(
            f"{context}.status_cluster_id is absent from the catalog: {status_cluster_id!r}"
        )
    else:
        status_record = status_by_id[status_cluster_id]
        if status_record["aspect_cluster_id"] != aspect_cluster_id:
            raise ValueError(f"{context}.status_cluster_id belongs to another aspect.")
        if status_record["status"] != status:
            raise ValueError(
                f"{context}.status and status_cluster_id resolve to different canonical labels."
            )
    return {
        "raw_aspect": raw_aspect,
        "aspect": aspect,
        "raw_status": raw_status,
        "status": status,
        "excerpt": excerpt,
        "opinion": opinion,
        "sentiment": sentiment,
        "aspect_cluster_id": aspect_cluster_id,
        "status_cluster_id": status_cluster_id,
        "catalog_comparison_eligible": True,
    }


def normalize_dynamic_submission(
    inputs: ReportInputs,
    submission: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate externally extracted Opinion Units against the active catalog clusters."""
    if not isinstance(submission, Mapping):
        raise TypeError("submission must be an object.")
    submission_id = _required_text(
        submission.get("submission_id"), field="submission_id", context="submission"
    )
    submitted_at_local = _required_text(
        submission.get("submitted_at_local"), field="submitted_at_local", context="submission"
    )
    if not re.fullmatch(r"\d{8}-\d{6}", submitted_at_local):
        raise ValueError("submission.submitted_at_local must use YYYYMMDD-HHMMSS.")
    product_name = _required_text(
        submission.get("product_name", submission.get("productName")),
        field="product_name",
        context="submission",
    )
    catalog_products = set(product_review_counts(inputs))
    if product_name not in catalog_products:
        raise ValueError(f"submission.product_name is absent from the catalog: {product_name!r}")
    raw_reviews = submission.get("reviews")
    if not isinstance(raw_reviews, list) or not raw_reviews:
        raise ValueError("submission.reviews must be a non-empty list.")
    raw_excluded_products = submission.get("excluded_products", [product_name])
    if not isinstance(raw_excluded_products, list):
        raise TypeError("submission.excluded_products must be a list.")
    excluded_products = sorted(
        {product_name, *(str(item).strip() for item in raw_excluded_products if str(item).strip())}
    )

    (
        aspect_ids_by_label,
        status_ids_by_aspect_label,
        aspect_by_id,
        status_by_id,
    ) = _normalization_indexes(inputs)
    catalog_reviews = inputs.reviews.set_index("review_idx", drop=False)
    normalized_reviews: list[dict[str, Any]] = []
    seen_source_review_indices: set[int] = set()
    opinion_unit_position = 0
    for review_position, raw_review in enumerate(raw_reviews, start=1):
        context = f"submission.reviews[{review_position - 1}]"
        if not isinstance(raw_review, Mapping):
            raise TypeError(f"{context} must be an object.")
        review_text = _required_text(raw_review.get("review"), field="review", context=context)
        source_review_idx_value = raw_review.get("source_review_idx")
        source_review_idx = None
        if source_review_idx_value is not None:
            if isinstance(source_review_idx_value, bool):
                raise TypeError(f"{context}.source_review_idx must be an integer when present.")
            source_review_idx = int(source_review_idx_value)
            if source_review_idx in seen_source_review_indices:
                raise ValueError("source_review_idx values must be unique within a submission.")
            seen_source_review_indices.add(source_review_idx)
            if source_review_idx not in catalog_reviews.index:
                raise ValueError(f"{context}.source_review_idx is absent from the catalog.")
            catalog_row = catalog_reviews.loc[source_review_idx]
            if str(catalog_row.product_name) != product_name:
                raise ValueError(f"{context}.source_review_idx belongs to another product.")
            if str(catalog_row.review) != review_text:
                raise ValueError(f"{context}.review differs from catalog source_review_idx text.")
        raw_opinion_units = raw_review.get("opinion_units")
        if not isinstance(raw_opinion_units, list) or not raw_opinion_units:
            raise ValueError(f"{context}.opinion_units must be a non-empty list.")
        opinion_units: list[dict[str, Any]] = []
        for unit_position, raw_unit in enumerate(raw_opinion_units, start=1):
            unit_context = f"{context}.opinion_units[{unit_position - 1}]"
            if not isinstance(raw_unit, Mapping):
                raise TypeError(f"{unit_context} must be an object.")
            normalized_unit = _resolve_opinion_unit(
                raw_unit,
                review=review_text,
                context=unit_context,
                aspect_ids_by_label=aspect_ids_by_label,
                status_ids_by_aspect_label=status_ids_by_aspect_label,
                aspect_by_id=aspect_by_id,
                status_by_id=status_by_id,
            )
            opinion_unit_position += 1
            normalized_unit["opinion_unit_id"] = f"u{opinion_unit_position}"
            opinion_units.append(normalized_unit)
        normalized_reviews.append(
            {
                "submission_review_id": f"review_{review_position}",
                "source_review_idx": source_review_idx,
                "review": review_text,
                "opinion_units": opinion_units,
            }
        )
    return {
        "submission_id": submission_id,
        "submitted_at_local": submitted_at_local,
        "product_name": product_name,
        "reviews": normalized_reviews,
        "excluded_products": excluded_products,
    }


def _vote_summary(votes: pd.DataFrame, *, decimal_places: int) -> dict[str, Any]:
    counts = {sentiment: int(votes["vote"].eq(sentiment).sum()) for sentiment in SENTIMENT_ORDER}
    return {
        "supporting_review_count": len(votes),
        "sentiment": sentiment_distribution(counts, decimal_places=decimal_places),
    }


def _submitted_unit_groups(submission: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str | None, str | None], dict[str, Any]] = {}
    for review in submission["reviews"]:
        for unit in review["opinion_units"]:
            if not unit["catalog_comparison_eligible"]:
                continue
            key = (unit["aspect_cluster_id"], unit["status_cluster_id"])
            record = grouped.setdefault(
                key,
                {
                    "aspect": unit["aspect"],
                    "status": unit["status"],
                    "aspect_cluster_id": unit["aspect_cluster_id"],
                    "status_cluster_id": unit["status_cluster_id"],
                    "units": [],
                    "review_votes": defaultdict(list),
                },
            )
            record["units"].append(unit)
            record["review_votes"][review["submission_review_id"]].append(unit["sentiment"])
    groups: list[dict[str, Any]] = []
    for record in grouped.values():
        votes = [
            collapse_sentiments(pd.Series(sentiments, dtype="string"))
            for sentiments in record.pop("review_votes").values()
        ]
        record["submitted_review_vote_count"] = len(votes)
        record["submitted_sentiment"] = collapse_sentiments(pd.Series(votes, dtype="string"))
        record["submitted_opinion_unit_ids"] = [unit["opinion_unit_id"] for unit in record["units"]]
        groups.append(record)
    return groups


def _other_aspect_top_statuses(
    pair_votes: pd.DataFrame,
    *,
    product_name: str,
    aspect_cluster_id: str,
    excluded_review_indices: set[int],
    decimal_places: int,
) -> list[dict[str, Any]]:
    """Return the other reviews' most frequently observed non-null statuses for one aspect."""
    aspect_mask = pair_votes["product_name"].eq(product_name) & pair_votes[
        "aspect_cluster_id"
    ].astype(str).eq(aspect_cluster_id)
    aspect_mask &= pair_votes["status_cluster_id"].astype("string").notna()
    aspect_mask &= ~pair_votes["review_idx"].isin(excluded_review_indices)
    other_aspect_votes = pair_votes.loc[aspect_mask]
    rows: list[dict[str, Any]] = []
    for (status_cluster_id, status), status_votes in other_aspect_votes.groupby(
        ["status_cluster_id", "status"],
        observed=True,
        dropna=False,
        sort=True,
    ):
        status_cluster_id_text = _optional_text(status_cluster_id)
        status_text = _optional_text(status)
        if status_cluster_id_text is None or status_text is None:
            continue
        summary = _vote_summary(status_votes, decimal_places=decimal_places)
        rows.append(
            {
                "status": status_text,
                "status_cluster_id": status_cluster_id_text,
                "supporting_review_count": summary["supporting_review_count"],
                "sentiment": summary["sentiment"],
            }
        )
    rows.sort(
        key=lambda row: (
            -int(row["supporting_review_count"]),
            str(row["status"]),
            str(row["status_cluster_id"]),
        )
    )
    for rank, row in enumerate(rows[:OTHER_ASPECT_STATUS_TOP_N], start=1):
        row["rank"] = rank
    return rows[:OTHER_ASPECT_STATUS_TOP_N]


def _catalog_relationships(
    inputs: ReportInputs,
    submission: dict[str, Any],
    *,
    settings: DynamicReportSettings,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    product_name = submission["product_name"]
    aspect_votes = build_sentiment_votes(inputs.joined, ["aspect_cluster_id", "aspect"])
    pair_votes = build_sentiment_votes(
        inputs.joined,
        ["aspect_cluster_id", "status_cluster_id", "aspect", "status"],
    )
    submitted_catalog_indices = {
        int(review["source_review_idx"])
        for review in submission["reviews"]
        if review["source_review_idx"] is not None
    }
    relationships: list[dict[str, Any]] = []
    unmentioned: list[dict[str, Any]] = []
    review_count = product_review_counts(inputs)[product_name]
    for position, group in enumerate(_submitted_unit_groups(submission), start=1):
        aspect_mask = aspect_votes["product_name"].eq(product_name) & aspect_votes[
            "aspect_cluster_id"
        ].astype(str).eq(group["aspect_cluster_id"])
        full_aspect_votes = aspect_votes.loc[aspect_mask]
        if group["status_cluster_id"] is None:
            full_pair_votes = full_aspect_votes
            comparison_grain = "aspect"
        else:
            pair_mask = pair_votes["product_name"].eq(product_name) & pair_votes[
                "aspect_cluster_id"
            ].astype(str).eq(group["aspect_cluster_id"])
            pair_mask &= (
                pair_votes["status_cluster_id"].astype("string").eq(group["status_cluster_id"])
            )
            full_pair_votes = pair_votes.loc[pair_mask]
            comparison_grain = "aspect_status"
        other_pair_votes = full_pair_votes.loc[
            ~full_pair_votes["review_idx"].isin(submitted_catalog_indices)
        ]
        full_aspect = _vote_summary(full_aspect_votes, decimal_places=settings.decimal_places)
        full_pair = _vote_summary(full_pair_votes, decimal_places=settings.decimal_places)
        other_pair = _vote_summary(other_pair_votes, decimal_places=settings.decimal_places)
        catalog_dominant = other_pair["sentiment"]["dominant_sentiment"]
        submitted_sentiment = group["submitted_sentiment"]
        if other_pair["supporting_review_count"] == 0:
            relation_code = "NOT_MENTIONED_BY_OTHER_REVIEWS"
            relation_label = "판단하기 어렵습니다"
        elif (
            submitted_sentiment in {"positive", "negative"}
            and catalog_dominant in {"positive", "negative"}
            and submitted_sentiment == catalog_dominant
        ):
            relation_code = "ALIGNS_WITH_OTHER_REVIEW_MAJORITY"
            relation_label = "일치합니다"
        elif submitted_sentiment in {"positive", "negative"} and catalog_dominant in {
            "positive",
            "negative",
        }:
            relation_code = "CONTRADICTS_OTHER_REVIEW_MAJORITY"
            relation_label = "일치하지 않습니다"
        else:
            relation_code = "INSUFFICIENT_OR_NON_DIRECTIONAL_OTHER_REVIEW_EVIDENCE"
            relation_label = "판단하기 어렵습니다"
        other_aspect_top_statuses: list[dict[str, Any]] = []
        if group["status_cluster_id"] is not None and other_pair["supporting_review_count"] == 0:
            other_aspect_top_statuses = _other_aspect_top_statuses(
                pair_votes,
                product_name=product_name,
                aspect_cluster_id=group["aspect_cluster_id"],
                excluded_review_indices=submitted_catalog_indices,
                decimal_places=settings.decimal_places,
            )
        row = {
            "rank": position,
            "comparison_grain": comparison_grain,
            "aspect": group["aspect"],
            "status": group["status"],
            "aspect_cluster_id": group["aspect_cluster_id"],
            "status_cluster_id": group["status_cluster_id"],
            "catalog_review_count": review_count,
            "aspect_review_count": full_aspect["supporting_review_count"],
            "same_status_review_count": full_pair["supporting_review_count"],
            "other_status_review_count": other_pair["supporting_review_count"],
            "other_status_sentiment": other_pair["sentiment"],
            "other_aspect_top_statuses": other_aspect_top_statuses,
            "submitted_sentiment": submitted_sentiment,
            "submitted_review_vote_count": group["submitted_review_vote_count"],
            "submitted_opinion_unit_ids": group["submitted_opinion_unit_ids"],
            "relation_code": relation_code,
            "relation_label": relation_label,
        }
        relationships.append(row)
        if group["status_cluster_id"] is not None and other_pair["supporting_review_count"] == 0:
            unmentioned.append(
                {
                    "aspect": group["aspect"],
                    "status": group["status"],
                    "aspect_cluster_id": group["aspect_cluster_id"],
                    "status_cluster_id": group["status_cluster_id"],
                    "submitted_opinion_unit_ids": group["submitted_opinion_unit_ids"],
                }
            )
    return relationships, unmentioned


def _pair_profile_indexes(
    inputs: ReportInputs,
    *,
    settings: DynamicReportSettings,
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    votes = build_sentiment_votes(
        inputs.joined,
        ["aspect_cluster_id", "status_cluster_id", "aspect", "status"],
    )
    review_counts = product_review_counts(inputs)
    exact: dict[tuple[str, str, str], dict[str, Any]] = {}
    by_product_aspect: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for key, group in votes.groupby(
        ["product_name", "aspect_cluster_id", "status_cluster_id", "aspect", "status"],
        observed=True,
        dropna=False,
        sort=True,
    ):
        product_name, aspect_cluster_id, status_cluster_id, aspect, status = key
        status_cluster_id_text = _optional_text(status_cluster_id)
        if status_cluster_id_text is None:
            continue
        product_name_text = str(product_name)
        aspect_cluster_id_text = str(aspect_cluster_id)
        summary = _vote_summary(group, decimal_places=settings.decimal_places)
        support = summary["supporting_review_count"]
        profile = {
            "product_name": product_name_text,
            "catalog_review_count": review_counts[product_name_text],
            "aspect_cluster_id": aspect_cluster_id_text,
            "status_cluster_id": status_cluster_id_text,
            "aspect": str(aspect),
            "status": _optional_text(status),
            "supporting_review_count": support,
            "mention_rate": round_float(
                support / review_counts[product_name_text], settings.decimal_places
            ),
            "support_reliability": round_float(
                min(1.0, support / RELATED_PRODUCT_MINIMUM_WEAKNESS_SUPPORT),
                settings.decimal_places,
            ),
            "sentiment": summary["sentiment"],
        }
        exact[(product_name_text, aspect_cluster_id_text, status_cluster_id_text)] = profile
        by_product_aspect[(product_name_text, aspect_cluster_id_text)].append(profile)
    for profiles in by_product_aspect.values():
        profiles.sort(
            key=lambda profile: (
                str(profile["status"]),
                str(profile["status_cluster_id"]),
            )
        )
    return exact, by_product_aspect


def _negative_submitted_conditions(
    submission: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    conditions: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for group in _submitted_unit_groups(submission):
        if group["submitted_sentiment"] != "negative":
            continue
        if group["status_cluster_id"] is None:
            excluded.append(
                {
                    "aspect": group["aspect"],
                    "status": group["status"],
                    "reason": "NEGATIVE_INPUT_HAS_NO_STATUS_CLUSTER_ID",
                    "submitted_review_vote_count": group["submitted_review_vote_count"],
                }
            )
            continue
        conditions.append(
            {
                "condition_id": f"condition_{len(conditions) + 1}",
                "aspect": group["aspect"],
                "status": group["status"],
                "aspect_cluster_id": group["aspect_cluster_id"],
                "status_cluster_id": group["status_cluster_id"],
                "submitted_review_vote_count": group["submitted_review_vote_count"],
                "importance": float(group["submitted_review_vote_count"]),
                "submitted_opinion_unit_ids": group["submitted_opinion_unit_ids"],
            }
        )
    return conditions, excluded


def _compact_profile(profile: dict[str, Any] | None) -> dict[str, Any] | None:
    if profile is None:
        return None
    return {
        "aspect": profile["aspect"],
        "status": profile["status"],
        "supporting_review_count": profile["supporting_review_count"],
        "sentiment": profile["sentiment"],
    }


def _condition_match(
    candidate_product_name: str,
    condition: dict[str, Any],
    *,
    exact_profiles: dict[tuple[str, str, str], dict[str, Any]],
    profiles_by_product_aspect: dict[tuple[str, str], list[dict[str, Any]]],
    decimal_places: int,
) -> dict[str, Any]:
    target_profile = exact_profiles.get(
        (
            candidate_product_name,
            condition["aspect_cluster_id"],
            condition["status_cluster_id"],
        )
    )
    target_negative_strength = 0.0
    if target_profile is not None:
        target_negative_upper = target_profile["sentiment"]["negative_wilson_95"]["upper"] or 0.0
        target_negative_strength = float(target_profile["support_reliability"] or 0.0) * float(
            target_negative_upper
        )
    positive_options: list[tuple[float, dict[str, Any]]] = []
    for profile in profiles_by_product_aspect.get(
        (candidate_product_name, condition["aspect_cluster_id"]), []
    ):
        if profile["status_cluster_id"] == condition["status_cluster_id"]:
            continue
        if profile["sentiment"]["dominant_sentiment"] != "positive":
            continue
        positive_lower = profile["sentiment"]["positive_wilson_95"]["lower"] or 0.0
        strength = float(profile["support_reliability"] or 0.0) * float(positive_lower)
        positive_options.append((strength, profile))
    positive_options.sort(
        key=lambda item: (
            -item[0],
            -item[1]["supporting_review_count"],
            str(item[1]["status"]),
            item[1]["status_cluster_id"],
        )
    )
    positive_alternative = positive_options[0][1] if positive_options else None
    positive_alternative_strength = positive_options[0][0] if positive_options else 0.0
    if target_profile is not None and positive_alternative is not None:
        relation = "EXACT_UNDESIRED_STATUS_AND_POSITIVE_ALTERNATIVE"
    elif target_profile is not None:
        relation = "EXACT_UNDESIRED_STATUS_ONLY"
    elif positive_alternative is not None:
        relation = "POSITIVE_ALTERNATIVE_ONLY"
    else:
        relation = "NO_EVIDENCE"
    utility = positive_alternative_strength - target_negative_strength
    reason_parts: list[str] = []
    if target_profile is None:
        reason_parts.append(f"'{condition['aspect']} {condition['status']}' 부정 상태 미관측")
    else:
        negative_count = target_profile["sentiment"]["counts"]["negative"]
        reason_parts.append(
            f"'{condition['aspect']} {condition['status']}' 부정 근거 {negative_count}개"
        )
    if positive_alternative is not None:
        positive_count = positive_alternative["sentiment"]["counts"]["positive"]
        reason_parts.append(
            f"'{condition['aspect']} {positive_alternative['status']}' 긍정 근거 {positive_count}개"
        )
    return {
        "condition_id": condition["condition_id"],
        "aspect": condition["aspect"],
        "status": condition["status"],
        "relation": relation,
        "utility": round_float(utility, decimal_places),
        "undesired_status_profile": _compact_profile(target_profile),
        "positive_alternative_profile": _compact_profile(positive_alternative),
        "reason": "; ".join(reason_parts) if reason_parts else "비교 가능한 카탈로그 근거 없음",
    }


def _alternative_recommendations(
    inputs: ReportInputs,
    submission: dict[str, Any],
    *,
    settings: DynamicReportSettings,
) -> dict[str, Any]:
    conditions, excluded_conditions = _negative_submitted_conditions(submission)
    contract = {
        "source": "submitted_negative_aspect_status_review_votes",
        "status_match_policy": "exact_status_cluster_id_for_undesired_condition",
        "positive_alternative_policy": "same_aspect_different_status_with_positive_dominant_sentiment",
        "negative_evidence_strength": "support_reliability * negative_wilson_upper_95",
        "positive_alternative_strength": "support_reliability * positive_wilson_lower_95",
        "weakness_utility": "weighted_mean(positive_alternative_strength - negative_evidence_strength)",
        "experience_similarity_weight": RELATED_PRODUCT_EXPERIENCE_WEIGHT,
        "weakness_utility_weight": RELATED_PRODUCT_REPAIR_UTILITY_WEIGHT,
        "weakness_repair_score": (
            "0.25 * experience_similarity + 0.75 * ((weakness_utility + 1) / 2)"
        ),
    }
    if not conditions:
        return {
            "status": "NO_NEGATIVE_SUBMITTED_ASPECT_STATUS",
            "ranking_contract": contract,
            "negative_conditions": [],
            "excluded_negative_conditions": excluded_conditions,
            "alternatives": [],
        }

    profile_data = _related_product_profile_data(inputs)
    similarity_rows = _related_similarity_rows(
        profile_data,
        source_product_name=submission["product_name"],
        decimal_places=settings.decimal_places,
    )
    exact_profiles, profiles_by_product_aspect = _pair_profile_indexes(inputs, settings=settings)
    alternatives: list[dict[str, Any]] = []
    total_importance = sum(condition["importance"] for condition in conditions)
    for similarity in similarity_rows:
        candidate_product_name = similarity["product_name"]
        if candidate_product_name in set(submission["excluded_products"]):
            continue
        matches = [
            _condition_match(
                candidate_product_name,
                condition,
                exact_profiles=exact_profiles,
                profiles_by_product_aspect=profiles_by_product_aspect,
                decimal_places=settings.decimal_places,
            )
            for condition in conditions
        ]
        supported_matches = [match for match in matches if match["relation"] != "NO_EVIDENCE"]
        if not supported_matches:
            continue
        weakness_utility = round_float(
            sum(
                condition["importance"] * float(match["utility"] or 0.0)
                for condition, match in zip(conditions, matches, strict=True)
            )
            / total_importance,
            settings.decimal_places,
        )
        experience_similarity = float(similarity["experience_similarity"] or 0.0)
        weakness_repair_score = (
            RELATED_PRODUCT_EXPERIENCE_WEIGHT * experience_similarity
            + RELATED_PRODUCT_REPAIR_UTILITY_WEIGHT * ((float(weakness_utility) + 1.0) / 2.0)
        )
        alternatives.append(
            {
                "product_name": candidate_product_name,
                "catalog_review_count": similarity["catalog_review_count"],
                "weakness_utility": weakness_utility,
                "experience_similarity": similarity["experience_similarity"],
                "weakness_repair_score": round_float(
                    weakness_repair_score, settings.decimal_places
                ),
                "evidence_coverage": round_float(
                    len(supported_matches) / len(conditions), settings.decimal_places
                ),
                "requirement_matches": matches,
                "recommendation_reason": " / ".join(match["reason"] for match in supported_matches),
                "quality_codes": similarity["quality_codes"],
            }
        )
    alternatives.sort(
        key=lambda row: (
            -float(row["weakness_repair_score"] or 0.0),
            -float(row["weakness_utility"] or 0.0),
            -float(row["experience_similarity"] or 0.0),
            row["product_name"],
        )
    )
    for rank, alternative in enumerate(alternatives, start=1):
        alternative["rank"] = rank
    return {
        "status": "COMPLETED" if alternatives else "NO_CANDIDATE_WITH_REPAIR_EVIDENCE",
        "ranking_contract": contract,
        "negative_conditions": conditions,
        "excluded_negative_conditions": excluded_conditions,
        "alternatives": alternatives[:RELATED_PRODUCT_LIMIT],
    }


def build_dynamic_review_decision_report(
    inputs: ReportInputs,
    submission: Mapping[str, Any],
    *,
    settings: DynamicReportSettings | None = None,
) -> dict[str, Any]:
    """Build the requested per-review relationship analysis and repair ranking."""
    selected_settings = settings or DynamicReportSettings()
    normalized_submission = normalize_dynamic_submission(inputs, submission)
    relationships, unmentioned = _catalog_relationships(
        inputs,
        normalized_submission,
        settings=selected_settings,
    )
    alternatives = _alternative_recommendations(
        inputs,
        normalized_submission,
        settings=selected_settings,
    )
    identity = {
        "schema_version": DYNAMIC_REVIEW_REPORT_SCHEMA_VERSION,
        "run_id": inputs.run_dir.name,
        "submission": normalized_submission,
        "settings": asdict(selected_settings),
        "algorithm": alternatives["ranking_contract"],
        "run_manifest_sha256": source_metadata(inputs)["run_manifest_sha256"],
    }
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()[:12]
    submitted_opinion_units = [
        {
            "submission_review_id": review["submission_review_id"],
            "source_review_idx": review["source_review_idx"],
            **unit,
        }
        for review in normalized_submission["reviews"]
        for unit in review["opinion_units"]
    ]
    return {
        "schema_version": DYNAMIC_REVIEW_REPORT_SCHEMA_VERSION,
        "proposal_type": DYNAMIC_REVIEW_REPORT_TYPE,
        "proposal_id": (
            f"{DYNAMIC_REVIEW_REPORT_TYPE}:{inputs.run_dir.name}:"
            f"{normalized_submission['submission_id']}:{digest}"
        ),
        "source": source_metadata(inputs),
        "submission": normalized_submission,
        "submitted_opinion_units": submitted_opinion_units,
        "catalog_relationships": relationships,
        "unmentioned_aspect_status": unmentioned,
        "alternative_recommendations": alternatives,
    }


def _review_display_blocks(submission: dict[str, Any]) -> list[str]:
    """Render each submitted review as its own display block."""
    return [f'$$"{markdown_cell(review["review"])}"$$' for review in submission["reviews"]]


def _relationship_heading(relationship: dict[str, Any]) -> str:
    return " ".join(
        value
        for value in (markdown_bold(relationship["aspect"]), relationship["status"])
        if value is not None
    )


def _render_other_aspect_top_statuses(lines: list[str], relationship: dict[str, Any]) -> None:
    aspect = markdown_bold(relationship["aspect"])
    top_statuses = relationship["other_aspect_top_statuses"]
    lines.extend(
        [
            (
                f"제출된 리뷰를 제외한 다른 리뷰에는 {aspect} > {relationship['status']} 조합과 "
                "완전히 일치하는 aspect-status가 없습니다."
            ),
            "",
        ]
    )
    if not top_statuses:
        lines.append(f"제출된 리뷰를 제외한 다른 리뷰에는 {aspect}의 status 관찰 자체가 없습니다.")
        return
    lines.extend(
        [
            f"대신 제출된 리뷰를 제외한 {aspect}의 가장 많이 언급된 status Top 3는 다음과 같습니다.",
            "",
            "| rank | status | reviews | positive | negative | etc |",
            "|---:|---|---:|---:|---:|---|",
        ]
    )
    for status_row in top_statuses:
        counts = status_row["sentiment"]["counts"]
        lines.append(
            f"| {status_row['rank']} | {markdown_cell(status_row['status'])} | "
            f"{status_row['supporting_review_count']} | {counts['positive']} | "
            f"{counts['negative']} | {counts['mixed']}/{counts['neutral']}/{counts['unknown']} |"
        )


def _render_relationship(lines: list[str], relationship: dict[str, Any]) -> None:
    lines.extend(["", f"### {_relationship_heading(relationship)}", ""])
    aspect = markdown_bold(relationship["aspect"])
    if relationship["comparison_grain"] == "aspect_status":
        lines.extend(
            [
                (
                    f"전체 리뷰 {relationship['catalog_review_count']}개 중 "
                    f"{aspect} 관련 언급은 {relationship['aspect_review_count']}개 리뷰에서 관찰됐습니다. "
                    f"이 중 {aspect} > {relationship['status']} 상태는 "
                    f"{relationship['same_status_review_count']}개 리뷰에서 관찰됐습니다."
                ),
                "",
            ]
        )
        if relationship["other_status_review_count"] == 0:
            _render_other_aspect_top_statuses(lines, relationship)
            lines.extend(
                [
                    "",
                    "따라서 제출된 리뷰의 평가는 다른 리뷰 다수와의 방향 일치를 판단하기 어렵습니다.",
                ]
            )
        else:
            lines.append(
                f"제출된 리뷰를 제외하고 동일한 {aspect} > {relationship['status']} 조합을 언급한 "
                "리뷰 중 "
                f"{relationship['other_status_sentiment']['counts']['negative']}개는 부정, "
                f"{relationship['other_status_sentiment']['counts']['positive']}개는 긍정, "
                f"{relationship['other_status_sentiment']['counts']['mixed']}개는 혼합 평가였습니다. "
                f"제출된 리뷰의 평가는 다수 리뷰의 평가 방향과 '{relationship['relation_label']}'."
            )
    else:
        lines.extend(
            [
                (
                    f"전체 리뷰 {relationship['catalog_review_count']}개 중 "
                    f"{aspect} 관련 언급은 {relationship['aspect_review_count']}개 리뷰에서 관찰됐습니다."
                ),
                "",
                (
                    f"제출된 Opinion Unit에 상태가 없어 {aspect} 단위로만 비교했습니다. "
                    f"제출된 리뷰의 평가는 다수 리뷰의 평가 방향과 '{relationship['relation_label']}'."
                ),
            ]
        )


def _render_alternatives(lines: list[str], alternatives: dict[str, Any]) -> None:
    lines.extend(["", "## 대안 상품 추천", ""])
    conditions = alternatives["negative_conditions"]
    ranked = alternatives["alternatives"]
    if alternatives["status"] != "COMPLETED":
        if not conditions:
            lines.append(
                "제출된 리뷰에서 상태가 정규화된 부정적인 속성-상태가 없어 대안 상품을 추천하지 않습니다."
            )
        else:
            lines.append("부정 속성을 보완할 수 있는 카탈로그 근거가 있는 대안 상품이 없습니다.")
        return
    condition_labels = ", ".join(
        f"'{condition['aspect']} {condition['status']}'" for condition in conditions
    )
    top_alternative = ranked[0]
    lines.extend(
        [
            f"제출된 리뷰에서 확인된 부정적인 속성-상태는 {condition_labels}입니다.",
            "",
            (
                "제출된 부정 속성을 기피 조건으로 둘 때, 1순위 대안은 "
                f"{markdown_bold(top_alternative['product_name'])}입니다. "
                f"**weakness utility** {top_alternative['weakness_utility']:.6f} 및 "
                f"**experience similarity** {top_alternative['experience_similarity']:.6f}를 "
                "3:1로 결합한 "
                f"**weakness repair score** {top_alternative['weakness_repair_score']:.6f}가 "
                "후보 중 가장 높기 때문입니다."
            ),
            "",
            (
                "정적 카탈로그 분석의 약점 보완 방식을 적용해, 제출된 부정 조건에 대한 "
                "weakness utility와 원본 상품의 리뷰 경험 유사도를 3:1로 결합했습니다."
            ),
            "",
            (
                "- **weakness utility**: 제출된 부정 조건에 대해 후보가 얼마나 유리한지를 "
                "나타내는 근거 점수(-1~1)."
            ),
            (
                "- **experience similarity**: 원본 상품과의 리뷰 경험 유사도로, 약점 보완이 "
                "얼마나 유사한 맥락을 유지하는지 나타냅니다."
            ),
            (
                "- **weakness repair score**: weakness utility와 experience similarity를 3:1로 "
                "결합한 최종 순위 점수입니다."
            ),
            "",
            (
                "| rank | product | weakness utility | experience similarity | "
                "weakness repair score | recommendation reason |"
            ),
            "|---:|---|---:|---:|---:|---|",
        ]
    )
    for alternative in ranked:
        lines.append(
            f"| {alternative['rank']} | {markdown_bold(alternative['product_name'])} | "
            f"{alternative['weakness_utility']:.6f} | "
            f"{alternative['experience_similarity']:.6f} | "
            f"{alternative['weakness_repair_score']:.6f} | "
            f"{markdown_cell(alternative['recommendation_reason'])} |"
        )


def render_dynamic_review_decision_markdown(proposal: dict[str, Any]) -> str:
    """Render the requested bounded Markdown view without legacy score tables."""
    submission = proposal["submission"]
    lines = [
        "# 동적 의사결정 제안서",
        "",
        (
            f"{submission['submitted_at_local']}에 제출된 "
            f"{markdown_bold(submission['product_name'])}에 대한 리뷰"
        ),
        "",
        *_review_display_blocks(submission),
        "",
        "를 기반으로 의사결정을 보조하는 제안서입니다.",
        "",
        "제출된 리뷰에서 추출 및 정규화된 속성은 다음과 같습니다.",
        "",
        "| raw_aspect | aspect | raw_status | status | excerpt | opinion | sentiment |",
        "|---|---|---|---|---|---|---|",
    ]
    for unit in proposal["submitted_opinion_units"]:
        lines.append(
            f"| {markdown_cell(unit['raw_aspect'])} | {markdown_cell(unit['aspect'])} | "
            f"{markdown_cell(unit['raw_status'])} | {markdown_cell(unit['status'])} | "
            f"{markdown_cell(unit['excerpt'])} | {markdown_cell(unit['opinion'])} | "
            f"{markdown_cell(unit['sentiment'])} |"
        )
    lines.extend(
        [
            "",
            (
                "- **raw_aspect**: 추출 시점에 정규화하지 않고 기록한, 리뷰어가 평가한 "
                "상품 속성·구성요소·사용 상황의 원본 명칭."
            ),
            (
                "- **aspect**: raw_aspect를 그대로 쓰거나 같은 product category 안의 군집 대표명으로 "
                "정규화한 최종 분석 속성."
            ),
            (
                "- **raw_status**: raw_aspect의 관찰된 상태·조건·값이며, 근거 있는 상태를 특정할 수 "
                "없을 때만 null을 가집니다."
            ),
            (
                "- **status**: raw_status를 그대로 쓰거나 해당 aspect 군집 안의 군집 대표명으로 "
                "정규화한 최종 분석 상태값."
            ),
            (
                "- **excerpt**: 해당 Opinion Unit을 뒷받침하도록 리뷰 원문에서 변경 없이 복사한 "
                "연속 구간."
            ),
            (
                "- **opinion**: 방향·정도·비교·사용 맥락을 보존해 리뷰어의 관찰 또는 평가를 "
                "간결하게 완결한 서술."
            ),
            (
                "- **sentiment**: 해당 aspect·status에 대한 리뷰어의 평가 방향으로 positive, "
                "negative, mixed, neutral, unknown 중 하나의 값을 가집니다."
            ),
            "",
            (
                "배송 상태와 같이 상품과 관련되지 않은 속성은 추출하지 않으며, 특별한 상품 속성 없는 "
                "긍정/부정 리뷰는 '전반적 상품 경험'이라는 속성값을 가집니다."
            ),
            "",
            "## 다른 리뷰와의 관계",
        ]
    )
    relationships = proposal["catalog_relationships"]
    if relationships:
        for relationship in relationships:
            _render_relationship(lines, relationship)
    else:
        lines.extend(
            [
                "",
                "제출된 Opinion Unit이 카탈로그의 정규화된 상품 속성과 연결되지 않아 비교할 수 없습니다.",
            ]
        )
    lines.extend(["", "### 언급되지 않는 aspect-status", ""])
    unmentioned = proposal["unmentioned_aspect_status"]
    if unmentioned:
        labels = ", ".join(f"{markdown_bold(row['aspect'])} {row['status']}" for row in unmentioned)
        lines.append(f"다음 속성-상태는 다른 리뷰에서 언급되지 않았습니다: {labels}.")
    else:
        lines.append("제출된 모든 속성-상태는 다른 리뷰에서도 언급되었습니다.")
    _render_alternatives(lines, proposal["alternative_recommendations"])
    return "\n".join(lines) + "\n"
