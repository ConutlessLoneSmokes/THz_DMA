#!/usr/bin/env python3
"""Re-run compact Stage 2-A analysis for an immutable run directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from thz_dma.analysis.stage2a import analyze_stage2a  # noqa: E402


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

