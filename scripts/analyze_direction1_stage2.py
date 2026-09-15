#!/usr/bin/env python3
"""Re-run compact Stage 2 analysis for an existing immutable run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _run_support  # noqa: F401  # Adds the local src tree before package imports.
from thz_dma.analysis.stage2 import analyze_stage2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    compact = analyze_stage2(args.run_dir)
    print(json.dumps(compact["platform_gate"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
