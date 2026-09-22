"""Configured classical estimators for the Stage 1 paired comparisons."""

from __future__ import annotations

import numpy as np

from thz_dma.estimators.literature_baselines import (
    SphericalDmaTemplateOperator,
    ogols_two_path_estimate,
    screened_sbl_two_path_estimate,
)
from thz_dma.estimators.tensor_transfer import tensor_denoised_ogols_two_path_estimate
from thz_dma.estimators.twopath_omp import omp_two_path_estimate
from thz_dma.observations.frequency_diverse import MeasurementDesign


METHOD_PROVENANCE = {
    "grid_omp": {
        "label": "Grid OMP",
        "role": "internal lower-complexity sanity baseline",
        "source_key": "internal_stage1b",
        "adaptation": "unchanged two-step grid OMP with joint gain refit",
        "paper_exact_reproduction": False,
    },
    "yang_ogols_adapted": {
        "label": "Yang-style OG-OLS (adapted)",
        "role": "near-field off-grid literature-transfer baseline",
        "source_key": "yang2024xldma",
        "adaptation": (
            "single joint-frequency-vector specialization of OG-DOLS; exact "
            "spherical DMA templates, OLS atom selection, first-order off-grid "
            "refinement; no UPA EL-AZ decoupling or measurement-matrix optimization"
        ),
        "paper_exact_reproduction": False,
    },
    "sbl_jointfreq": {
        "label": "Joint-frequency SBL (adapted)",
        "role": "classical Bayesian sparse-recovery baseline",
        "source_key": "gao2024thzunfolding_parent_sbl",
        "adaptation": (
            "classical complex SBL/EM applied to a deterministic screened joint-"
            "frequency dictionary; not Gao et al.'s MMV AMP-SBL or unfolded network"
        ),
        "paper_exact_reproduction": False,
    },
    "zhang_cpd_ogols_adapted": {
        "label": "Zhang-style CPD -> OG-OLS (adapted)",
        "role": "low-rank frequency-tensor transfer baseline",
        "source_key": "zhang2025tensorofdm",
        "adaptation": (
            "rank-2 block-Hankel SVD/CP denoising of the actual receive-DMA "
            "observation, followed by exact spherical OG-OLS and an original-"
            "observation gain refit; not the source paper's transmit-DMA four-way "
            "CP model, microstrip-sequential training, algebraic parameter "
            "extraction, or uniqueness result"
        ),
        "paper_exact_reproduction": False,
    },
}


def stage1_method_names(config: dict) -> list[str]:
    names = [str(value) for value in config["experiment"]["estimators"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("Stage 1-C estimators must be non-empty and unique")
    unknown = sorted(set(names) - set(METHOD_PROVENANCE))
    if unknown:
        raise ValueError(f"unsupported Stage 1-C estimators: {unknown}")
    if "grid_omp" not in names:
        raise ValueError("grid_omp must be retained as the paired reference")
    return names


def build_stage1_template_operators(
    designs: list[MeasurementDesign],
    positions_m: np.ndarray,
    absorption,
    *,
    include_absorption: bool,
    interpolation: str,
) -> dict[str, SphericalDmaTemplateOperator]:
    operators = {}
    for design in designs:
        if include_absorption:
            coefficient = np.asarray(
                absorption.coefficient(
                    design.frequencies_hz, method=interpolation
                ),
                dtype=float,
            )
        else:
            coefficient = np.zeros(design.frequencies_hz.size, dtype=float)
        operators[design.name] = SphericalDmaTemplateOperator(
            frequencies_hz=design.frequencies_hz,
            element_positions_m=positions_m,
            weights=design.weights,
            absorption_coefficient_per_m=coefficient,
        )
    return operators


def estimate_stage1_baseline(
    method: str,
    dictionary,
    observation: np.ndarray,
    noise_variance: float | None,
    template_operator: SphericalDmaTemplateOperator,
    design: MeasurementDesign,
    config: dict,
) -> dict:
    estimator = config["estimator"]
    exclusion_range = float(estimator["exclusion_range_m"])
    exclusion_angle = float(np.deg2rad(estimator["exclusion_angle_deg"]))
    if method == "grid_omp":
        result = omp_two_path_estimate(
            dictionary,
            observation,
            exclusion_range_m=exclusion_range,
            exclusion_angle_rad=exclusion_angle,
        )
        return {
            **result,
            "iterations": 1,
            "converged": 1,
            "screened_candidate_count": dictionary.num_candidates,
        }
    if method == "yang_ogols_adapted":
        settings = estimator["yang_ogols_adapted"]
        search = config["search"]
        return ogols_two_path_estimate(
            dictionary,
            observation,
            template_operator.evaluate,
            range_bounds_m=(
                float(search["range_min_m"]),
                float(search["range_max_m"]),
            ),
            angle_bounds_rad=(
                float(np.deg2rad(search["angle_min_deg"])),
                float(np.deg2rad(search["angle_max_deg"])),
            ),
            range_grid_step_m=float(search["range_step_m"]),
            angle_grid_step_rad=float(np.deg2rad(search["angle_step_deg"])),
            exclusion_range_m=exclusion_range,
            exclusion_angle_rad=exclusion_angle,
            refinement_iterations=int(settings["refinement_iterations"]),
            derivative_fraction=float(settings["derivative_fraction"]),
            maximum_update_grid_units=float(
                settings["maximum_update_grid_units"]
            ),
            convergence_tolerance=float(settings["convergence_tolerance"]),
        )
    if method == "sbl_jointfreq":
        settings = estimator["sbl_jointfreq"]
        return screened_sbl_two_path_estimate(
            dictionary,
            observation,
            noise_variance=noise_variance,
            screening_size=int(settings["screening_size"]),
            max_iterations=int(settings["max_iterations"]),
            convergence_tolerance=float(settings["convergence_tolerance"]),
            em_damping=float(settings["em_damping"]),
            noise_floor_fraction=float(settings["noise_floor_fraction"]),
            exclusion_range_m=exclusion_range,
            exclusion_angle_rad=exclusion_angle,
        )
    if method == "zhang_cpd_ogols_adapted":
        tensor_settings = estimator["zhang_cpd_ogols_adapted"]
        offgrid_settings = estimator["yang_ogols_adapted"]
        search = config["search"]
        return tensor_denoised_ogols_two_path_estimate(
            dictionary,
            observation,
            template_operator.evaluate,
            frequencies_hz=design.frequencies_hz,
            configuration_index=design.configuration_index,
            range_bounds_m=(
                float(search["range_min_m"]),
                float(search["range_max_m"]),
            ),
            angle_bounds_rad=(
                float(np.deg2rad(search["angle_min_deg"])),
                float(np.deg2rad(search["angle_max_deg"])),
            ),
            range_grid_step_m=float(search["range_step_m"]),
            angle_grid_step_rad=float(np.deg2rad(search["angle_step_deg"])),
            exclusion_range_m=exclusion_range,
            exclusion_angle_rad=exclusion_angle,
            refinement_iterations=int(offgrid_settings["refinement_iterations"]),
            derivative_fraction=float(offgrid_settings["derivative_fraction"]),
            maximum_update_grid_units=float(
                offgrid_settings["maximum_update_grid_units"]
            ),
            convergence_tolerance=float(
                offgrid_settings["convergence_tolerance"]
            ),
            tensor_rank=int(tensor_settings["rank"]),
            hankel_rows=int(tensor_settings["hankel_rows"]),
            cp_max_iterations=int(tensor_settings["cp_max_iterations"]),
            cp_convergence_tolerance=float(
                tensor_settings["cp_convergence_tolerance"]
            ),
            cp_ridge=float(tensor_settings["cp_ridge"]),
        )
    raise ValueError(f"unsupported estimator: {method}")
