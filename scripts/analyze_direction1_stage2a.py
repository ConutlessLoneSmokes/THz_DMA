#!/usr/bin/env python3
"""Re-run compact Stage 2-A analysis for an immutable run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _run_support  # noqa: F401  # Adds the local src tree before package imports.
from thz_dma.analysis.stage2a import analyze_stage2a


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    compact = analyze_stage2a(args.run_dir)
    print(
        json.dumps(
            compact["primary_screen"], indent=2, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
