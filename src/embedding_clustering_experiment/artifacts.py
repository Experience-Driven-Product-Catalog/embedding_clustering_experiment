"""Atomic artifact writes, hashing, and run-directory creation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def create_run_directory(results_root: Path) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    run_stem = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    for suffix in range(100):
        name = run_stem if suffix == 0 else f"{run_stem}-{suffix:02d}"
        candidate = results_root / name
        try:
            candidate.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(f"Could not allocate a run directory below {results_root}.")


def atomic_write_parquet(
    frame: pd.DataFrame,
    destination: Path,
    *,
    compression: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.to_parquet(
            temporary_path,
            engine="pyarrow",
            index=False,
            compression=compression,
        )
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_inventory(run_dir: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "run_manifest.json":
            continue
        inventory[str(path.relative_to(run_dir))] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return inventory
