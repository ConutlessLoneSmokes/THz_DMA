#!/usr/bin/env python3
"""Run and archive the Stage 1-D bounded tensor-transfer screen."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from run_direction1_stage1c import (  # noqa: E402
    _git_state,
    _sha256,
    _write_code_manifest,
    _write_raw_metrics,
    _write_status,
)
from thz_dma import __version__  # noqa: E402
from thz_dma.analysis.stage1d import analyze_stage1d  # noqa: E402
from thz_dma.stage1d import run_stage1d  # noqa: E402


def _package_version(name: str) -> str:
    if importlib.util.find_spec(name) is None:
        return "not installed"
    module = __import__(name)
    return str(getattr(module, "__version__", "installed; version unavailable"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "direction1_stage1d_tensor_transfer.toml"
        ),
    )
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    config_path = args.config.resolve()
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    if not np.isclose(float(config["resources"]["pilot_energy_per_re"]), 1.0):
        raise ValueError("Stage 1-D currently requires pilot_energy_per_re = 1.0")

    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "stage1d_%Y%m%dT%H%M%SZ"
    )
    run_dir = PROJECT_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    status_path = run_dir / "status.json"
    _write_status(status_path, "running", run_id=run_id)

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
        json.dumps(
            {
                "replicate_seeds": config["experiment"]["replicate_seeds"],
                "scenario_seed_offset": 1_000_003,
                "repeated_pilot_noise_design_stride": 10_000_019,
                "repeated_pilot_noise_offset": 97_531,
                "bootstrap_seed": config["experiment"]["bootstrap_seed"],
                "randomization_seed": config["experiment"]["randomization_seed"],
                "dma_candidate_seed_start": config["dma"]["candidate_seed_start"],
                "data_codebook_seed_start": config["data_evaluation"][
                    "codebook_seed_start"
                ],
                "tensor_initialization": "deterministic mode-wise SVD",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix is None and (Path(sys.prefix) / "conda-meta").is_dir():
        conda_prefix = sys.prefix
    conda_name = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_name is None and conda_prefix is not None:
        conda_name = Path(conda_prefix).name
    environment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "conda_default_env": conda_name or "not detected",
        "conda_prefix": conda_prefix or "not detected",
        "numpy": np.__version__,
        "scipy": _package_version("scipy"),
        "pandas": _package_version("pandas"),
        "matplotlib": _package_version("matplotlib"),
        "package_version": __version__,
        "deep_learning_framework": "not used in Stage 1-D",
        "cuda": "not used",
        "raw_metric_format": "CSV; one row per scene/noise/design/estimator",
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
        PROJECT_ROOT / "pyproject.toml",
        config_path,
        PROJECT_ROOT / config["atmosphere"]["table_path"],
        Path(__file__).resolve(),
        PROJECT_ROOT / "scripts" / "run_direction1_stage1c.py",
        *sorted((PROJECT_ROOT / "src").rglob("*.py")),
        *sorted((PROJECT_ROOT / "tests").glob("test_*.py")),
    ]
    _write_code_manifest(run_dir / "code_manifest.json", manifest_files)
    checkpoint_dir = run_dir / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "README.txt").write_text(
        "No learned model is used in Stage 1-D.\n", encoding="utf-8"
    )

    print(f"run_id: {run_id}", flush=True)
    print(f"run_dir: {run_dir}", flush=True)
    try:
        rows, summary = run_stage1d(config, PROJECT_ROOT, progress=print)
        _write_raw_metrics(run_dir / "metrics_raw.csv", rows)
        (run_dir / "metrics_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        (run_dir / "execution_schedule.json").write_text(
            json.dumps(
                summary["execution_schedule"],
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        (run_dir / "method_manifest.json").write_text(
            json.dumps(
                summary["method_provenance"],
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        compact = analyze_stage1d(run_dir)
        results_dir = PROJECT_ROOT / "results" / "direction1_stage1d"
        results_dir.mkdir(parents=True, exist_ok=True)
        summary_result = results_dir / f"{run_id}_summary.json"
        analysis_result = results_dir / f"{run_id}_analysis_compact.json"
        summary_result.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        analysis_result.write_text(
            json.dumps(compact, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
    except Exception as error:
        _write_status(
            status_path,
            "failed",
            run_id=run_id,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise

    screen_status = compact["primary_tensor_screen"]["status"]
    _write_status(
        status_path,
        "complete",
        run_id=run_id,
        raw_metric_rows=len(rows),
        tensor_screen_status=screen_status,
        summary_path=str(summary_result),
        compact_analysis_path=str(analysis_result),
    )
    print(f"summary: {summary_result}", flush=True)
    print(f"analysis: {analysis_result}", flush=True)
    print(f"tensor_screen: {screen_status}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
