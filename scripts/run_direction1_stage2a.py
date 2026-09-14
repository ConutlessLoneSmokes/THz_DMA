#!/usr/bin/env python3
"""Run and archive Stage 2-A paired three-dimensional classical baselines."""

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
from thz_dma.analysis.stage2a import analyze_stage2a  # noqa: E402
from thz_dma.stage2a import run_stage2a  # noqa: E402


def _package_version(name: str) -> str:
    if importlib.util.find_spec(name) is None:
        return "not installed"
    module = __import__(name)
    return str(getattr(module, "__version__", "installed; version unavailable"))


def _compute_environment(config: dict) -> dict[str, str]:
    search = config["search"]
    requested = str(search.get("dictionary_compute_backend", "numpy"))
    requested_device = str(search.get("dictionary_compute_device", "cuda:0"))
    if requested == "numpy":
        return {
            "dictionary_compute_backend_requested": requested,
            "dictionary_compute_device_requested": "cpu",
            "deep_learning_framework": "not used in Stage 2-A",
            "cuda": "not used",
        }
    try:
        import torch
    except ModuleNotFoundError:
        return {
            "dictionary_compute_backend_requested": requested,
            "dictionary_compute_device_requested": requested_device,
            "deep_learning_framework": "PyTorch not installed",
            "cuda": "unavailable",
        }
    devices = [
        str(torch.cuda.get_device_name(index))
        for index in range(torch.cuda.device_count())
    ]
    return {
        "dictionary_compute_backend_requested": requested,
        "dictionary_compute_device_requested": requested_device,
        "deep_learning_framework": (
            f"PyTorch {torch.__version__}; CUDA dictionary construction only; "
            "no learning model"
        ),
        "cuda": (
            f"available={torch.cuda.is_available()}; "
            f"runtime={torch.version.cuda}; devices={devices}"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "direction1_stage2a_classical.toml",
    )
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    config_path = args.config.resolve()
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    if not np.isclose(float(config["acquisition"]["pilot_energy_per_re"]), 1.0):
        raise ValueError("Stage 2-A currently requires pilot_energy_per_re = 1.0")

    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "stage2a_%Y%m%dT%H%M%SZ"
    )
    run_dir = PROJECT_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    status_path = run_dir / "status.json"
    _write_status(status_path, "running", run_id=run_id)

    absorption_path = PROJECT_ROOT / config["atmosphere"]["table_path"]
    resolved = {
        "config": config,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "absorption_table_sha256": _sha256(absorption_path),
    }
    (run_dir / "config_resolved.json").write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    seed_manifest = {
        "replicate_seeds": config["experiment"]["replicate_seeds"],
        "randomization_seed": config["experiment"]["randomization_seed"],
        "bootstrap_seed": config["experiment"]["bootstrap_seed"],
        "noise_seed_contract": (
            "SeedSequence(replicate_seed, sample, schedule_index, 2026090921); "
            "the same standard noise draw is rescaled across SNR levels"
        ),
    }
    if str(config["search"].get("mode", "fixed")) == "beam_center_adaptive":
        seed_manifest.update(
            {
                "coarse_center_seed_contract": (
                    "SeedSequence(replicate_seed, sample, 2026091101)"
                ),
                "continuous_truth_offset_seed_contract": (
                    "SeedSequence(replicate_seed, sample, 2026091102)"
                ),
                "artificial_half_grid_offset": False,
            }
        )
    (run_dir / "seeds.json").write_text(
        json.dumps(seed_manifest, indent=2), encoding="utf-8"
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
        "pandas": _package_version("pandas"),
        "matplotlib": _package_version("matplotlib"),
        "package_version": __version__,
        **_compute_environment(config),
        "raw_metric_format": "CSV; one row per scene/schedule/SNR/estimator block",
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
        absorption_path,
        Path(__file__).resolve(),
        PROJECT_ROOT / "scripts" / "analyze_direction1_stage2a.py",
        *sorted((PROJECT_ROOT / "src").rglob("*.py")),
        *sorted((PROJECT_ROOT / "tests").glob("test_*.py")),
    ]
    _write_code_manifest(run_dir / "code_manifest.json", manifest_files)
    checkpoint_dir = run_dir / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "README.txt").write_text(
        "No learned model or checkpoint is used in Stage 2-A.\n", encoding="utf-8"
    )

    print(f"run_id: {run_id}", flush=True)
    print(f"run_dir: {run_dir}", flush=True)
    expected_scenes = (
        len(config["experiment"]["replicate_seeds"])
        * int(config["experiment"]["samples_per_seed"])
    )
    expected_rows = (
        expected_scenes
        * len(config["acquisition"]["schedules"])
        * len(config["noise"]["array_reference_snr_db"])
        * len(config["experiment"]["estimators"])
    )
    print(
        f"declared_design: {expected_scenes} scenes, {expected_rows} raw rows",
        flush=True,
    )
    try:
        rows, summary = run_stage2a(config, PROJECT_ROOT, progress=print)
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
                {
                    "scope": summary["scope"],
                    "method_provenance": summary["method_provenance"],
                    "resource_fairness": summary["resource_fairness"],
                    "schedules": summary["schedules"],
                    "search": summary["search"],
                    "data_codebook": summary["data_codebook"],
                },
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        compact = analyze_stage2a(run_dir)
        results_dir = PROJECT_ROOT / "results" / "direction1_stage2a"
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

    baseline_status = compact["primary_screen"]["baseline_gate"]["status"]
    _write_status(
        status_path,
        "complete",
        run_id=run_id,
        raw_metric_rows=len(rows),
        baseline_status=baseline_status,
        summary_path=str(summary_result),
        compact_analysis_path=str(analysis_result),
    )
    print(f"summary: {summary_result}", flush=True)
    print(f"analysis: {analysis_result}", flush=True)
    print(f"baseline_gate: {baseline_status}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
