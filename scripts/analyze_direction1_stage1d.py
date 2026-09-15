#!/usr/bin/env python3
"""Regenerate Stage 1-D compact analysis and figures from one run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _run_support  # noqa: F401  # Adds the local src tree before package imports.
from thz_dma.analysis.stage1d import analyze_stage1d


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_stage1d(args.run_dir)
    print(
        json.dumps(
            {
                "analysis": str(args.run_dir.resolve() / "analysis"),
                "data_complete": result["data_integrity"]["complete"],
                "tensor_screen": result["primary_tensor_screen"]["status"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
