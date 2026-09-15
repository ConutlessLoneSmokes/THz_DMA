#!/usr/bin/env python3
"""Regenerate compact Stage 1-B analysis and figures from one run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _run_support  # noqa: F401  # Adds the local src tree before package imports.
from thz_dma.analysis.stage1b import analyze_stage1b


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_stage1b(args.run_dir)
    print(
        json.dumps(
            {
                "analysis": str(args.run_dir.resolve() / "analysis"),
                "data_complete": result["data_integrity"]["complete"],
                "decision": result["primary_gate"]["status"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
