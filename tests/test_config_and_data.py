from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from embedding_clustering_experiment.config import load_config, validate_config
from embedding_clustering_experiment.data import (
    OPINION_COLUMNS,
    REPRESENTATIVE_COLUMNS,
    validate_opinion_input,
    validate_representative_input,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_config_has_full_dataset_only_and_preserves_algorithm_parameters() -> None:
    config = load_config(PROJECT_ROOT / "config.yaml").values

    assert "sampling" not in config
    assert config["clustering"]["metric"] == "cosine"
    assert config["clustering"]["linkage"] == "complete"
    assert config["clustering"]["experiment_b"]["distance_threshold"] == 0.22
    assert config["clustering"]["experiment_d"]["aspect"]["distance_threshold"] == 0.3591
    assert config["clustering"]["experiment_d"]["status"]["distance_threshold"] == 0.20
    assert config["evaluation"]["user_evaluation"]["task_count"] == 100
    assert config["evaluation"]["user_evaluation"]["random_seed"] == 17171771
    assert config["project"]["normalization_version"] == "2026-08-03"

    invalid = deepcopy(config)
    invalid["sampling"] = {"row_count": 100}
    with pytest.raises(ValueError, match="Sample configuration"):
        validate_config(invalid)

    invalid_version = deepcopy(config)
    invalid_version["project"]["normalization_version"] = " "
    with pytest.raises(ValueError, match="normalization_version"):
        validate_config(invalid_version)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("task_count", 0, "task_count must be a positive integer"),
        ("task_count", True, "task_count must be a positive integer"),
        ("random_seed", -1, "random_seed must be a non-negative integer"),
        ("random_seed", False, "random_seed must be a non-negative integer"),
    ],
)
def test_user_evaluation_sampling_config_is_strict(
    field: str,
    value: object,
    message: str,
) -> None:
    config = load_config(PROJECT_ROOT / "config.yaml").values
    config["evaluation"]["user_evaluation"][field] = value

    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_new_separated_input_contract_accepts_nullable_raw_status(
    representative_frame: pd.DataFrame,
    opinion_frame: pd.DataFrame,
) -> None:
    validate_representative_input(representative_frame)
    validate_opinion_input(opinion_frame)

    assert tuple(representative_frame.columns) == REPRESENTATIVE_COLUMNS
    assert tuple(opinion_frame.columns) == OPINION_COLUMNS
    assert opinion_frame["raw_status"].isna().sum() == 1


def test_integrated_product_or_review_columns_are_rejected(
    representative_frame: pd.DataFrame,
) -> None:
    integrated = representative_frame.assign(product_category="모니터", review="본문")
    with pytest.raises(ValueError, match="columns must be exactly"):
        validate_representative_input(integrated)


def test_sentiment_remains_one_valid_column(representative_frame: pd.DataFrame) -> None:
    invalid = representative_frame.copy()
    invalid.loc[0, "sentiment"] = "mostly_positive"
    with pytest.raises(ValueError, match="invalid values"):
        validate_representative_input(invalid)
