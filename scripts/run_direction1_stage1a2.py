#!/usr/bin/env python3
"""Run and archive one direction-one Stage 1-A2 experiment block."""

from __future__ import annotations

from _run_support import (
    PROJECT_ROOT,
    create_run_archive,
    parse_run_arguments,
    require_unit_pilot_energy,
)
from thz_dma.stage1a2 import run_stage1a2


def main() -> int:
    config_path, config, requested_run_id = parse_run_arguments(
        "direction1_stage1a2_gain_control.toml", __doc__
    )
    require_unit_pilot_energy(
        config["resources"]["pilot_energy_per_re"], "Stage 1-A2"
    )
    archive = create_run_archive(
        "stage1a2",
        requested_run_id,
        config_path=config_path,
        config=config,
        seeds={
            "replicate_seeds": config["experiment"]["replicate_seeds"],
            "bootstrap_seed": config["experiment"]["bootstrap_seed"],
            "randomization_seed": config["experiment"]["randomization_seed"],
            "dma_candidate_seed_start": config["dma"]["candidate_seed_start"],
        },
        script_path=__file__,
        environment={
            "deep_learning_framework": "not used in Stage 1-A2",
            "cuda": "not used",
            "raw_metric_format": "CSV; one row per scene/condition/design/model",
        },
        checkpoint_note="No learned model is used in Stage 1-A2.\n",
        figures_note="Generate figures from metrics_raw.csv after the run completes.\n",
    )

    print(f"run_id: {archive.run_id}", flush=True)
    print(f"run_dir: {archive.directory}", flush=True)
    try:
        rows, summary = run_stage1a2(config, PROJECT_ROOT, progress=print)
        archive.write_metrics(rows)
        archive.write_json("metrics_summary.json", summary)
        archive.write_json("execution_schedule.json", summary["execution_schedule"])
        result_path = archive.write_result("direction1_stage1a2", "summary", summary)
    except Exception as error:
        archive.write_status(
            "failed",
            run_id=archive.run_id,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise

    archive.write_status(
        "complete",
        run_id=archive.run_id,
        raw_metric_rows=len(rows),
        summary_path=str(result_path),
    )
    print(f"summary: {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
