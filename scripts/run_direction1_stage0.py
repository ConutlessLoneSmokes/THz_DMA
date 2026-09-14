#!/usr/bin/env python3
"""Run and archive the deterministic direction-one Stage 0 checks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from thz_dma import __version__  # noqa: E402
from thz_dma.stage0 import run_stage0  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_raw_metrics(path: Path, rows: list[dict]) -> None:
    fieldnames = ["category", "scenario", "frequency_hz", "distance_m", "metric", "value"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _git_state(project_root: Path) -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable: project directory is not a Git worktree\n"
    return f"commit: {commit}\ndirty: {bool(status.strip())}\n"


def _write_code_manifest(path: Path, files: list[Path]) -> None:
    records = []
    for source in sorted({item.resolve() for item in files}):
        records.append(
            {
                "path": str(source.relative_to(PROJECT_ROOT)),
                "size_bytes": source.stat().st_size,
                "sha256": _sha256(source),
            }
        )
    path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "direction1_stage0.toml",
    )
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    config_path = args.config.resolve()
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("stage0_%Y%m%dT%H%M%SZ")
    run_dir = PROJECT_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    rows, summary = run_stage0(config, PROJECT_ROOT)
    resolved = {
        "config": config,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "absorption_table_sha256": _sha256(
            PROJECT_ROOT / config["atmosphere"]["table_path"]
        ),
    }
    (run_dir / "config_resolved.json").write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "seeds.json").write_text(
        json.dumps({"experiment_seed": int(config["experiment"]["seed"])}, indent=2),
        encoding="utf-8",
    )
    environment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "package_version": __version__,
        "deep_learning_framework": "not used (NumPy-only Stage 0)",
        "cuda": "not used",
        "gpu": "not queried (CPU deterministic Stage 0)",
        "raw_metric_format": "CSV; Parquet deferred until estimator-scale runs",
        "cwd": os.getcwd(),
    }
    (run_dir / "environment.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in environment.items()) + "\n",
        encoding="utf-8",
    )
    (run_dir / "git_commit.txt").write_text(
        _git_state(PROJECT_ROOT), encoding="utf-8"
    )
    manifest_files = [
        config_path,
        PROJECT_ROOT / config["atmosphere"]["table_path"],
        Path(__file__).resolve(),
        *sorted((PROJECT_ROOT / "src").rglob("*.py")),
    ]
    _write_code_manifest(run_dir / "code_manifest.json", manifest_files)
    _write_raw_metrics(run_dir / "metrics_raw.csv", rows)
    (run_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    results_dir = PROJECT_ROOT / "results" / "direction1_stage0"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{run_id}_summary.json"
    result_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    checkpoint_dir = run_dir / "checkpoint"
    figures_dir = run_dir / "figures"
    checkpoint_dir.mkdir()
    figures_dir.mkdir()
    (checkpoint_dir / "README.txt").write_text(
        "No learned model is used in deterministic Stage 0.\n", encoding="utf-8"
    )
    (figures_dir / "README.txt").write_text(
        "Figures are deferred; metrics_raw.csv contains the plotting data.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "summary": str(result_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
