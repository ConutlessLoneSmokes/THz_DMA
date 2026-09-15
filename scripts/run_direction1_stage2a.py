#!/usr/bin/env python3
"""Run and archive Stage 2-A paired three-dimensional classical baselines."""

from __future__ import annotations

from _run_support import (
    PROJECT_ROOT,
    create_run_archive,
    parse_run_arguments,
    require_unit_pilot_energy,
)
from thz_dma.analysis.stage2a import analyze_stage2a
from thz_dma.stage2a import run_stage2a


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
    config_path, config, requested_run_id = parse_run_arguments(
        "direction1_stage2a_classical.toml", __doc__
    )
    require_unit_pilot_energy(
        config["acquisition"]["pilot_energy_per_re"], "Stage 2-A"
    )
    seeds = {
        "replicate_seeds": config["experiment"]["replicate_seeds"],
        "randomization_seed": config["experiment"]["randomization_seed"],
        "bootstrap_seed": config["experiment"]["bootstrap_seed"],
        "noise_seed_contract": (
            "SeedSequence(replicate_seed, sample, schedule_index, 2026090921); "
            "the same standard noise draw is rescaled across SNR levels"
        ),
    }
    if str(config["search"].get("mode", "fixed")) == "beam_center_adaptive":
        seeds.update(
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
    archive = create_run_archive(
        "stage2a",
        requested_run_id,
        config_path=config_path,
        config=config,
        seeds=seeds,
        script_path=__file__,
        environment={
            **_compute_environment(config),
            "raw_metric_format": (
                "CSV; one row per scene/schedule/SNR/estimator block"
            ),
        },
        checkpoint_note="No learned model or checkpoint is used in Stage 2-A.\n",
        extra_manifest_files=(
            PROJECT_ROOT / "scripts" / "analyze_direction1_stage2a.py",
        ),
    )

    expected_scenes = len(config["experiment"]["replicate_seeds"]) * int(
        config["experiment"]["samples_per_seed"]
    )
    expected_rows = (
        expected_scenes
        * len(config["acquisition"]["schedules"])
        * len(config["noise"]["array_reference_snr_db"])
        * len(config["experiment"]["estimators"])
    )
    print(f"run_id: {archive.run_id}", flush=True)
    print(f"run_dir: {archive.directory}", flush=True)
    print(
        f"declared_design: {expected_scenes} scenes, {expected_rows} raw rows",
        flush=True,
    )
    try:
        rows, summary = run_stage2a(config, PROJECT_ROOT, progress=print)
        archive.write_metrics(rows)
        archive.write_json("metrics_summary.json", summary)
        archive.write_json("execution_schedule.json", summary["execution_schedule"])
        archive.write_json(
            "method_manifest.json",
            {
                "scope": summary["scope"],
                "method_provenance": summary["method_provenance"],
                "resource_fairness": summary["resource_fairness"],
                "schedules": summary["schedules"],
                "search": summary["search"],
                "data_codebook": summary["data_codebook"],
            },
        )
        compact = analyze_stage2a(archive.directory)
        summary_result = archive.write_result("direction1_stage2a", "summary", summary)
        analysis_result = archive.write_result(
            "direction1_stage2a", "analysis_compact", compact
        )
    except Exception as error:
        archive.write_status(
            "failed",
            run_id=archive.run_id,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise

    baseline_status = compact["primary_screen"]["baseline_gate"]["status"]
    archive.write_status(
        "complete",
        run_id=archive.run_id,
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
