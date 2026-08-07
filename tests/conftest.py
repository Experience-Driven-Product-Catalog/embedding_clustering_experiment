from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from embedding_clustering_experiment.config import load_config
from embedding_clustering_experiment.experiments import ExperimentArtifacts, build_experiments

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StaticEncoder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = {
            text: np.asarray(vector, dtype=np.float32) / np.linalg.norm(vector)
            for text, vector in vectors.items()
        }

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.vectors[str(text)] for text in texts]).astype(np.float32)


@pytest.fixture
def config_values() -> dict:
    values = load_config(PROJECT_ROOT / "config.yaml").values
    values["embedding"]["show_progress_bar"] = False
    values["evaluation"]["user_evaluation"]["task_count"] = 2
    return values


@pytest.fixture
def representative_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "idx": pd.Series([1, 2, 3, 4], dtype="int64"),
            "review_idx": pd.Series([10, 11, 12, 13], dtype="int64"),
            "raw_attribute": pd.Series(
                ["화질", "화면 품질", "가격", "전반적 상품 경험"], dtype="string"
            ),
            "sentiment": pd.Series(["positive", "positive", "negative", "neutral"], dtype="string"),
        }
    )


@pytest.fixture
def opinion_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "idx": pd.Series([1, 2, 3, 4, 5], dtype="int64"),
            "review_idx": pd.Series([10, 11, 12, 13, 13], dtype="int64"),
            "raw_aspect": pd.Series(
                ["화질", "화면 품질", "가격", "전반적 상품 경험", "베젤"],
                dtype="string",
            ),
            "raw_status": pd.Series(
                ["선명함", "선명함", "비쌈", "만족", None], dtype="string"
            ),
            "excerpt": pd.Series(["e1", "e2", "e3", "e4", "e5"], dtype="string"),
            "opinion": pd.Series(["o1", "o2", "o3", "o4", "o5"], dtype="string"),
            "sentiment": pd.Series(
                ["positive", "positive", "negative", "positive", "neutral"],
                dtype="string",
            ),
        }
    )


@pytest.fixture
def static_encoder() -> StaticEncoder:
    return StaticEncoder(
        {
            "화질": [1.0, 0.0, 0.0, 0.0],
            "화면 품질": [1.0, 0.0, 0.0, 0.0],
            "가격": [0.0, 1.0, 0.0, 0.0],
            "전반적 상품 경험": [0.0, 0.0, 1.0, 0.0],
            "베젤": [0.0, 0.0, 0.0, 1.0],
            "선명함": [1.0, 1.0, 0.0, 0.0],
            "비쌈": [0.0, 1.0, 1.0, 0.0],
        }
    )


@pytest.fixture
def experiment_artifacts(
    representative_frame: pd.DataFrame,
    opinion_frame: pd.DataFrame,
    static_encoder: StaticEncoder,
    config_values: dict,
) -> ExperimentArtifacts:
    return build_experiments(
        representative_frame,
        opinion_frame,
        static_encoder,
        config_values,
        logging.getLogger("test"),
        normalization_run_id="test-run",
    )
