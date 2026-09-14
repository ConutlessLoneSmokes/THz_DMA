"""Variable-projection grid estimator for a single spherical-wave LoS path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from thz_dma.channels.atmosphere import AbsorptionTable, C0_M_PER_S


@dataclass(frozen=True)
class LosTemplateDictionary:
    """Known-combiner LoS templates over a range-angle grid."""

    ranges_m: np.ndarray
    angles_rad: np.ndarray
    templates: np.ndarray

    def __post_init__(self):
        ranges = np.asarray(self.ranges_m)
        angles = np.asarray(self.angles_rad)
        templates = np.asarray(self.templates)
        if ranges.ndim != 1 or angles.shape != ranges.shape:
            raise ValueError("ranges_m and angles_rad must be matching vectors")
        if templates.ndim != 2 or templates.shape[0] != ranges.size:
            raise ValueError("templates must have one row per candidate")
        if ranges.size == 0 or templates.shape[1] == 0:
            raise ValueError("dictionary cannot be empty")

    @property
    def num_candidates(self) -> int:
        return int(self.ranges_m.size)

    @property
    def num_pilots(self) -> int:
        return int(self.templates.shape[1])


def _candidate_vectors(range_grid_m, angle_grid_rad) -> tuple[np.ndarray, np.ndarray]:
    ranges = np.asarray(range_grid_m, dtype=float)
    angles = np.asarray(angle_grid_rad, dtype=float)
    if ranges.ndim != 1 or angles.ndim != 1 or ranges.size == 0 or angles.size == 0:
        raise ValueError("range and angle grids must be non-empty vectors")
    if np.any(~np.isfinite(ranges)) or np.any(ranges <= 0.0):
        raise ValueError("range grid must be finite and positive")
    if np.any(~np.isfinite(angles)):
        raise ValueError("angle grid must be finite")
    range_mesh, angle_mesh = np.meshgrid(ranges, angles, indexing="ij")
    return range_mesh.ravel(), angle_mesh.ravel()


def build_los_template_dictionaries(
    frequencies_hz,
    element_positions_m,
    range_grid_m,
    angle_grid_rad,
    weights_by_method: dict[str, np.ndarray],
    *,
    absorption_table: AbsorptionTable | None,
    include_absorption: bool,
    interpolation: str = "linear",
    chunk_size: int = 64,
    propagation_model: Literal["spherical", "far_field"] = "spherical",
) -> dict[str, LosTemplateDictionary]:
    """Build several paired dictionaries while reusing each channel chunk."""

    frequencies = np.asarray(frequencies_hz, dtype=float)
    positions = np.asarray(element_positions_m, dtype=float)
    if frequencies.ndim != 1 or positions.ndim != 1:
        raise ValueError("frequencies and element positions must be vectors")
    if np.any(frequencies <= 0.0) or positions.size == 0:
        raise ValueError("invalid frequencies or empty aperture")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not weights_by_method:
        raise ValueError("at least one combiner method is required")
    if propagation_model not in {"spherical", "far_field"}:
        raise ValueError(f"unsupported propagation_model: {propagation_model}")

    combiners: dict[str, np.ndarray] = {}
    for name, weights in weights_by_method.items():
        values = np.asarray(weights, dtype=complex)
        if values.shape != (frequencies.size, positions.size):
            raise ValueError(f"combiner {name!r} has an incompatible shape")
        combiners[name] = values

    candidate_ranges, candidate_angles = _candidate_vectors(
        range_grid_m, angle_grid_rad
    )
    template_arrays = {
        name: np.empty((candidate_ranges.size, frequencies.size), dtype=complex)
        for name in combiners
    }
    if include_absorption:
        if absorption_table is None:
            raise ValueError("absorption_table is required when absorption is enabled")
        kappa = np.asarray(
            absorption_table.coefficient(frequencies, method=interpolation), dtype=float
        )
    else:
        kappa = np.zeros(frequencies.size, dtype=float)

    for start in range(0, candidate_ranges.size, chunk_size):
        stop = min(start + chunk_size, candidate_ranges.size)
        ranges = candidate_ranges[start:stop]
        angles = candidate_angles[start:stop]
        if propagation_model == "spherical":
            distances = np.sqrt(
                ranges[:, None] ** 2
                + positions[None, :] ** 2
                - 2.0
                * ranges[:, None]
                * positions[None, :]
                * np.sin(angles[:, None])
            )
            amplitude = C0_M_PER_S / (
                4.0
                * np.pi
                * frequencies[None, :, None]
                * distances[:, None, :]
            )
            amplitude *= np.exp(
                -0.5 * kappa[None, :, None] * distances[:, None, :]
            )
            channel = amplitude * np.exp(
                -1j
                * 2.0
                * np.pi
                * frequencies[None, :, None]
                * distances[:, None, :]
                / C0_M_PER_S
            )
        else:
            amplitude = C0_M_PER_S / (
                4.0
                * np.pi
                * frequencies[None, :]
                * ranges[:, None]
            )
            amplitude *= np.exp(
                -0.5 * kappa[None, :] * ranges[:, None]
            )
            delay_phase = np.exp(
                -1j
                * 2.0
                * np.pi
                * frequencies[None, :]
                * ranges[:, None]
                / C0_M_PER_S
            )
            spatial_phase = np.exp(
                1j
                * 2.0
                * np.pi
                * frequencies[None, :, None]
                * positions[None, None, :]
                * np.sin(angles[:, None, None])
                / C0_M_PER_S
            )
            channel = (
                amplitude[:, :, None]
                * delay_phase[:, :, None]
                * spatial_phase
            )
        for name, weights in combiners.items():
            template_arrays[name][start:stop] = np.einsum(
                "kn,ckn->ck", np.conjugate(weights), channel, optimize=True
            )

    return {
        name: LosTemplateDictionary(
            ranges_m=candidate_ranges.copy(),
            angles_rad=candidate_angles.copy(),
            templates=templates,
        )
        for name, templates in template_arrays.items()
    }


def variable_projection_estimate(
    dictionary: LosTemplateDictionary,
    observations,
) -> dict[str, np.ndarray]:
    """Estimate range, angle and unknown complex gain by profile likelihood.

    For every grid template ``s(theta)``, the unknown gain is eliminated as
    ``alpha_hat = s^H y / ||s||^2``.  White-noise maximum likelihood therefore
    selects the candidate maximizing ``|s^H y|^2 / ||s||^2``.
    """

    received = np.asarray(observations, dtype=complex)
    scalar_input = received.ndim == 1
    received = np.atleast_2d(received)
    if received.ndim != 2 or received.shape[1] != dictionary.num_pilots:
        raise ValueError("observations must have one column per pilot")

    templates = np.asarray(dictionary.templates, dtype=complex)
    energy = np.sum(np.abs(templates) ** 2, axis=1)
    if np.any(energy <= 0.0):
        raise ValueError("dictionary contains a zero-energy template")
    correlation = received @ np.conjugate(templates).T
    score = np.abs(correlation) ** 2 / energy[None, :]
    best_index = np.argmax(score, axis=1)
    row = np.arange(received.shape[0])
    gain = correlation[row, best_index] / energy[best_index]
    residual_energy = np.maximum(
        0.0,
        np.sum(np.abs(received) ** 2, axis=1) - score[row, best_index],
    )
    result = {
        "candidate_index": best_index,
        "range_m": dictionary.ranges_m[best_index],
        "angle_rad": dictionary.angles_rad[best_index],
        "complex_gain": gain,
        "profile_score": score[row, best_index],
        "residual_energy": residual_energy,
    }
    if scalar_input:
        return {key: value[0] for key, value in result.items()}
    return result
