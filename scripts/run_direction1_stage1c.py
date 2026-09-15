#!/usr/bin/env python3
"""Run and archive the Stage 1-C paired estimator transfer screen."""

from __future__ import annotations

from _run_support import (
    PROJECT_ROOT,
    create_run_archive,
    parse_run_arguments,
    require_unit_pilot_energy,
)
from thz_dma.analysis.stage1c import analyze_stage1c
from thz_dma.stage1c import run_stage1c


def main() -> int:
    config_path, config, requested_run_id = parse_run_arguments(
        "direction1_stage1c_method_transfer.toml", __doc__
    )
    require_unit_pilot_energy(
        config["resources"]["pilot_energy_per_re"], "Stage 1-C"
    )
    archive = create_run_archive(
        "stage1c",
        requested_run_id,
        config_path=config_path,
        config=config,
        seeds={
            "replicate_seeds": config["experiment"]["replicate_seeds"],
            "scenario_seed_offset": 1_000_003,
            "bootstrap_seed": config["experiment"]["bootstrap_seed"],
            "randomization_seed": config["experiment"]["randomization_seed"],
            "dma_candidate_seed_start": config["dma"]["candidate_seed_start"],
            "data_codebook_seed_start": config["data_evaluation"][
                "codebook_seed_start"
            ],
        },
        script_path=__file__,
        environment={
            "deep_learning_framework": "not used in Stage 1-C",
            "cuda": "not used",
            "raw_metric_format": "CSV; one row per scene/noise/design/estimator",
        },
        checkpoint_note="No learned model is used in Stage 1-C.\n",
        extra_manifest_files=(
            PROJECT_ROOT / "scripts" / "analyze_direction1_stage1c.py",
        ),
    )

    print(f"run_id: {archive.run_id}", flush=True)
    print(f"run_dir: {archive.directory}", flush=True)
    try:
        rows, summary = run_stage1c(config, PROJECT_ROOT, progress=print)
        archive.write_metrics(rows)
        archive.write_json("metrics_summary.json", summary)
        archive.write_json("execution_schedule.json", summary["execution_schedule"])
        archive.write_json("method_manifest.json", summary["method_provenance"])
        compact = analyze_stage1c(archive.directory)
        summary_result = archive.write_result("direction1_stage1c", "summary", summary)
        analysis_result = archive.write_result(
            "direction1_stage1c", "analysis_compact", compact
        )
    except Exception as error:
        archive.write_status(
            "failed",
            run_id=archive.run_id,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise

    screen_status = compact["primary_method_screen"]["status"]
    archive.write_status(
        "complete",
        run_id=archive.run_id,
        raw_metric_rows=len(rows),
        method_screen_status=screen_status,
        summary_path=str(summary_result),
        compact_analysis_path=str(analysis_result),
    )
    print(f"summary: {summary_result}", flush=True)
    print(f"analysis: {analysis_result}", flush=True)
    print(f"method_screen: {screen_status}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
