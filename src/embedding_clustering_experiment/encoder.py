"""Local sentence embedding model with a run-scoped expression cache."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class SentenceEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class CachedSentenceEncoder:
    """Load the configured local model once and encode each unique string once."""

    def __init__(self, config: Mapping[str, Any], base_dir: Path) -> None:
        configured_device = str(config["device"])
        if configured_device == "auto":
            configured_device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = configured_device
        self.batch_size = int(config["batch_size"])
        self.normalize_embeddings = bool(config["normalize_embeddings"])
        self.show_progress_bar = bool(config["show_progress_bar"])
        model_path = Path(config["local_model_path"]).expanduser()
        if not model_path.is_absolute():
            model_path = base_dir / model_path
        self.model_path = model_path.resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Configured local model does not exist: {self.model_path}")
        self.model_id = str(config["model_id"])
        self.model = SentenceTransformer(
            str(self.model_path),
            device=self.device,
            local_files_only=bool(config["local_files_only"]),
        )
        self._cache: dict[str, np.ndarray] = {}

    @property
    def cached_expression_count(self) -> int:
        return len(self._cache)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        normalized_texts = [str(text) for text in texts]
        missing = sorted(set(normalized_texts) - set(self._cache))
        if missing:
            vectors = self.model.encode(
                missing,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=self.show_progress_bar,
            ).astype(np.float32, copy=False)
            for text, vector in zip(missing, vectors, strict=True):
                self._cache[text] = vector
        if not normalized_texts:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack([self._cache[text] for text in normalized_texts]).astype(
            np.float32, copy=False
        )
