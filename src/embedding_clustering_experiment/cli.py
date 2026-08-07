"""Command-line entrypoint for the full-dataset experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .runner import run

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run experiments A-D and automatic evaluation on the complete inputs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="YAML configuration; relative paths resolve from the configuration directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = run(load_config(args.config))
    print(output_dir)
