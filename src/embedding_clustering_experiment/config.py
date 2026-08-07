"""Configuration loading and validation for the full-dataset experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated YAML configuration with paths relative to its own directory."""

    path: Path
    values: dict[str, Any]

    @property
    def base_dir(self) -> Path:
        return self.path.parent

    def resolve(self, value: str | Path) -> Path:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = self.base_dir / candidate
        return candidate.resolve()


def load_config(path: Path) -> ExperimentConfig:
    config_path = path.expanduser().resolve()
    with config_path.open(encoding="utf-8") as source:
        values = yaml.safe_load(source)
    if not isinstance(values, dict):
        raise TypeError("config.yaml must contain a top-level mapping.")
    validate_config(values)
    return ExperimentConfig(path=config_path, values=values)


def validate_config(config: dict[str, Any]) -> None:
    required_sections = {
        "project",
        "inputs",
        "filters",
        "embedding",
        "clustering",
        "canonical_label",
        "output",
        "evaluation",
    }
    missing = sorted(required_sections - set(config))
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")
    if "sampling" in config:
        raise ValueError(
            "Sample configuration is not supported; this project always uses full inputs."
        )

    project = config["project"]
    normalization_version = project.get("normalization_version")
    if not isinstance(normalization_version, str) or not normalization_version.strip():
        raise ValueError("project.normalization_version must be a non-empty string.")

    inputs = config["inputs"]
    if set(inputs) != {"representative_attributes", "opinion_units"}:
        raise ValueError("inputs must contain representative_attributes and opinion_units only.")

    embedding = config["embedding"]
    if int(embedding["batch_size"]) < 1:
        raise ValueError("embedding.batch_size must be at least 1.")
    if embedding["device"] not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("embedding.device must be auto, cpu, mps, or cuda.")

    clustering = config["clustering"]
    if clustering["metric"] != "cosine":
        raise ValueError("The experiment contract requires cosine distance.")
    if clustering["linkage"] != "complete":
        raise ValueError("The experiment contract requires complete linkage.")
    stages = (
        clustering["experiment_b"],
        clustering["experiment_d"]["aspect"],
        clustering["experiment_d"]["status"],
    )
    for stage in stages:
        threshold = float(stage["distance_threshold"])
        naming_threshold = float(stage["naming_max_distance"])
        if not 0.0 < threshold <= 2.0:
            raise ValueError("Every distance_threshold must be in (0, 2].")
        if not 0.0 < naming_threshold <= threshold:
            raise ValueError("Every naming_max_distance must be in (0, distance_threshold].")

    for index, group in enumerate(clustering.get("status_opposition_groups", [])):
        if not isinstance(group, dict) or set(group) != {"side_a", "side_b"}:
            raise ValueError(
                f"status_opposition_groups[{index}] must contain side_a and side_b only."
            )
        side_a = set(map(str, group["side_a"]))
        side_b = set(map(str, group["side_b"]))
        if not side_a or not side_b or side_a & side_b:
            raise ValueError(f"status_opposition_groups[{index}] has invalid sides.")

    negation = clustering["status_explicit_negation"]
    expected_negation_keys = {
        "enabled",
        "suffixes",
        "whole_word_tokens",
        "core_similarity_threshold",
    }
    if set(negation) != expected_negation_keys:
        raise ValueError("clustering.status_explicit_negation has an invalid schema.")
    if not 0.0 <= float(negation["core_similarity_threshold"]) <= 1.0:
        raise ValueError("status explicit-negation core threshold must be in [0, 1].")

    canonical = config["canonical_label"]
    alpha = float(canonical["centrality_weight_alpha"])
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("canonical_label.centrality_weight_alpha must be in [0, 1].")
    if int(canonical["min_canonical_review_count"]) < 1:
        raise ValueError("canonical_label.min_canonical_review_count must be at least 1.")

    evaluation = config["evaluation"]
    expected_evaluation_keys = {
        "enabled",
        "nearest_cluster_pair_limit",
        "risky_cluster_limit_per_stage",
        "user_evaluation",
    }
    if set(evaluation) != expected_evaluation_keys:
        raise ValueError("evaluation has an invalid schema.")
    user_evaluation = evaluation["user_evaluation"]
    if not isinstance(user_evaluation, dict) or set(user_evaluation) != {
        "task_count",
        "random_seed",
    }:
        raise ValueError(
            "evaluation.user_evaluation must contain task_count and random_seed only."
        )
    task_count = user_evaluation["task_count"]
    random_seed = user_evaluation["random_seed"]
    if isinstance(task_count, bool) or not isinstance(task_count, int) or task_count < 1:
        raise ValueError("evaluation.user_evaluation.task_count must be a positive integer.")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise ValueError("evaluation.user_evaluation.random_seed must be a non-negative integer.")
