"""Stage 1-D bounded transfer of a tensor low-rank mechanism."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from thz_dma.stage1c import run_stage1c


def run_stage1d(
    config: dict,
    project_root: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict], dict]:
    """Run paired J1 and tensor-native acquisition estimator comparisons."""

    rows, summary = run_stage1c(config, project_root, progress=progress)
    summary["experiment"]["status"] = (
        "Stage 1-D bounded Zhang-style tensor-structure transfer screen"
    )
    summary["scope"].update(
        {
            "comparison_axis": (
                "paired estimator replacement within each acquisition design; "
                "J1 and tensor-native scans remain separate resource axes"
            ),
            "tensor_native_design_family": str(
                config["analysis"]["tensor_native_design_family"]
            ),
            "tensor_source_method": "zhang2025tensorofdm",
            "tensor_transfer_level": (
                "low-rank frequency mechanism only; not paper-exact CPD"
            ),
        }
    )
    summary["tensor_transfer_contract"] = {
        "single_configuration": (
            "rank-P block-Hankel SVD denoising followed by physical OG-OLS"
        ),
        "multiple_configurations": (
            "rank-P configuration-by-Hankel-frequency CP-ALS denoising "
            "followed by physical OG-OLS"
        ),
        "final_gain_fit": "original noisy observation",
        "paper_exact_reproduction": False,
        "excluded_source_components": [
            "transmit-DMA role",
            "microstrip-sequential four-way tensor",
            "source-paper algebraic factor extraction",
            "source-paper CP uniqueness claim",
        ],
    }
    return rows, summary
