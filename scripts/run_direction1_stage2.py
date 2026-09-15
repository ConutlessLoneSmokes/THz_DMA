#!/usr/bin/env python3
"""Run and archive the Stage 2 unified two-dimensional platform validation."""

from __future__ import annotations

from _run_support import (
    PROJECT_ROOT,
    create_run_archive,
    parse_run_arguments,
    require_unit_pilot_energy,
)
from thz_dma.analysis.stage2 import analyze_stage2
from thz_dma.stage2 import run_stage2_validation


def main() -> int:
    config_path, config, requested_run_id = parse_run_arguments(
        "direction1_stage2_unified_2d.toml", __doc__
    )
    require_unit_pilot_energy(
        config["acquisition"]["pilot_energy_per_re"], "Stage 2"
    )
    archive = create_run_archive(
        "stage2",
        requested_run_id,
        config_path=config_path,
        config=config,
        seeds={
            "replicate_seeds": config["experiment"]["replicate_seeds"],
            "randomization_seed": config["experiment"]["randomization_seed"],
            "bootstrap_seed": config["experiment"]["bootstrap_seed"],
            "noise_seed_contract": (
                "SeedSequence(replicate_seed, sample, physics_index, "
                "schedule_index, 2026090902)"
            ),
        },
        script_path=__file__,
        environment={
            "deep_learning_framework": "not used in Stage 2 platform validation",
            "cuda": "not used",
            "raw_metric_format": "CSV; one row per scene/physics/schedule block",
        },
        checkpoint_note="No learned model or checkpoint is used in Stage 2-0.\n",
        extra_manifest_files=(
            PROJECT_ROOT / "scripts" / "analyze_direction1_stage2.py",
        ),
    )

    print(f"run_id: {archive.run_id}", flush=True)
    print(f"run_dir: {archive.directory}", flush=True)
    try:
        rows, summary = run_stage2_validation(config, PROJECT_ROOT, progress=print)
        archive.write_metrics(rows)
        archive.write_json("metrics_summary.json", summary)
        archive.write_json("execution_schedule.json", summary["execution_schedule"])
        archive.write_json(
            "platform_manifest.json",
            {
                "scope": summary["scope"],
                "model_metadata": summary["model_metadata"],
                "resource_fairness": summary["resource_fairness"],
                "schedules": summary["schedules"],
                "method_adapter_contract": summary["method_adapter_contract"],
            },
        )
        compact = analyze_stage2(archive.directory)
        summary_result = archive.write_result("direction1_stage2", "summary", summary)
        analysis_result = archive.write_result(
            "direction1_stage2", "analysis_compact", compact
        )
    except Exception as error:
        archive.write_status(
            "failed",
            run_id=archive.run_id,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise

    platform_status = compact["platform_gate"]["status"]
    archive.write_status(
        "complete",
        run_id=archive.run_id,
        raw_metric_rows=len(rows),
        platform_status=platform_status,
        summary_path=str(summary_result),
        compact_analysis_path=str(analysis_result),
    )
    print(f"summary: {summary_result}", flush=True)
    print(f"analysis: {analysis_result}", flush=True)
    print(f"platform_gate: {platform_status}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
