#!/usr/bin/env python3
"""Run and archive the deterministic direction-one Stage 0 checks."""

from __future__ import annotations

import json

from _run_support import PROJECT_ROOT, create_run_archive, parse_run_arguments
from thz_dma.stage0 import run_stage0


def main() -> int:
    config_path, config, requested_run_id = parse_run_arguments(
        "direction1_stage0.toml", __doc__
    )
    archive = create_run_archive(
        "stage0",
        requested_run_id,
        config_path=config_path,
        config=config,
        seeds={"experiment_seed": int(config["experiment"]["seed"])},
        script_path=__file__,
        environment={
            "deep_learning_framework": "not used (NumPy-only Stage 0)",
            "cuda": "not used",
            "gpu": "not queried (CPU deterministic Stage 0)",
            "raw_metric_format": "CSV; Parquet deferred until estimator-scale runs",
        },
        checkpoint_note="No learned model is used in deterministic Stage 0.\n",
        figures_note="Figures are deferred; metrics_raw.csv contains the plotting data.\n",
    )

    try:
        rows, summary = run_stage0(config, PROJECT_ROOT)
        archive.write_metrics(
            rows,
            ["category", "scenario", "frequency_hz", "distance_m", "metric", "value"],
        )
        archive.write_json("metrics_summary.json", summary)
        result_path = archive.write_result("direction1_stage0", "summary", summary)
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
    print(
        json.dumps(
            {
                "run_id": archive.run_id,
                "run_dir": str(archive.directory),
                "summary": str(result_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
