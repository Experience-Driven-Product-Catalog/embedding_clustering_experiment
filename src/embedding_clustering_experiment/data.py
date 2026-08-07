"""Strict data contracts for the separated attribute tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

SENTIMENT_VALUES = {"positive", "negative", "mixed", "neutral", "unknown"}

REPRESENTATIVE_COLUMNS = ("idx", "review_idx", "raw_attribute", "sentiment")
OPINION_COLUMNS = (
    "idx",
    "review_idx",
    "raw_aspect",
    "raw_status",
    "excerpt",
    "opinion",
    "sentiment",
)
FORBIDDEN_INTEGRATED_COLUMNS = {
    "product",
    "product_name",
    "productName",
    "product_category",
    "review",
}


@dataclass(frozen=True)
class InputTables:
    representative: pd.DataFrame
    opinion: pd.DataFrame


def load_inputs(representative_path: Path, opinion_path: Path) -> InputTables:
    representative = pd.read_parquet(representative_path)
    opinion = pd.read_parquet(opinion_path)
    validate_representative_input(representative)
    validate_opinion_input(opinion)
    return InputTables(representative=representative, opinion=opinion)


def validate_representative_input(frame: pd.DataFrame) -> None:
    _validate_exact_columns(frame, REPRESENTATIVE_COLUMNS, "representative_attributes")
    _validate_common(frame, "representative_attributes")
    _validate_nonempty_strings(frame, ("raw_attribute", "sentiment"), "representative_attributes")
    _validate_sentiment(frame, "representative_attributes")


def validate_opinion_input(frame: pd.DataFrame) -> None:
    _validate_exact_columns(frame, OPINION_COLUMNS, "opinion_units")
    _validate_common(frame, "opinion_units")
    _validate_nonempty_strings(
        frame,
        ("raw_aspect", "excerpt", "opinion", "sentiment"),
        "opinion_units",
    )
    nonnull_status = frame.loc[frame["raw_status"].notna(), "raw_status"].astype(str)
    if nonnull_status.str.strip().eq("").any():
        raise ValueError("opinion_units.raw_status contains empty strings.")
    _validate_sentiment(frame, "opinion_units")


def eligible_opinion_rows(frame: pd.DataFrame, excluded_aspects: set[str]) -> pd.DataFrame:
    return frame.loc[~frame["raw_aspect"].isin(excluded_aspects)].copy()


def assert_source_columns_preserved(
    source: pd.DataFrame,
    result: pd.DataFrame,
    source_columns: tuple[str, ...],
) -> None:
    """Prove that an augmented result did not alter any source value or dtype."""
    expected = source.loc[source["idx"].isin(result["idx"]), list(source_columns)].copy()
    expected = expected.sort_values("idx", kind="stable").reset_index(drop=True)
    actual = (
        result.loc[:, list(source_columns)].sort_values("idx", kind="stable").reset_index(drop=True)
    )
    assert_frame_equal(actual, expected, check_dtype=True, check_like=False)


def _validate_exact_columns(
    frame: pd.DataFrame,
    expected_columns: tuple[str, ...],
    source_name: str,
) -> None:
    actual = tuple(frame.columns)
    if actual != expected_columns:
        raise ValueError(
            f"{source_name} columns must be exactly {list(expected_columns)}; found {list(actual)}."
        )
    forbidden = sorted(FORBIDDEN_INTEGRATED_COLUMNS & set(frame.columns))
    if forbidden:
        raise ValueError(f"{source_name} contains integrated-table columns: {forbidden}")


def _validate_common(frame: pd.DataFrame, source_name: str) -> None:
    if frame.empty:
        raise ValueError(f"{source_name} has no rows.")
    for column in ("idx", "review_idx"):
        if frame[column].isna().any():
            raise ValueError(f"{source_name}.{column} contains null values.")
    if not frame["idx"].is_unique:
        raise ValueError(f"{source_name}.idx must be unique.")
    expected_idx = pd.Series(range(1, len(frame) + 1), dtype="int64")
    actual_idx = frame["idx"].reset_index(drop=True).astype("int64")
    if not actual_idx.equals(expected_idx):
        raise ValueError(f"{source_name}.idx must be consecutive int64 values starting at 1.")


def _validate_nonempty_strings(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    source_name: str,
) -> None:
    for column in columns:
        if frame[column].isna().any():
            raise ValueError(f"{source_name}.{column} contains null values.")
        if frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{source_name}.{column} contains empty strings.")


def _validate_sentiment(frame: pd.DataFrame, source_name: str) -> None:
    invalid = sorted(set(frame["sentiment"].astype(str)) - SENTIMENT_VALUES)
    if invalid:
        raise ValueError(f"{source_name}.sentiment contains invalid values: {invalid}")
