"""Frequency-dependent spreading and molecular absorption utilities.

The imported coefficient table follows the legacy MATLAB convention

    L_abs,dB = 10 log10(e) * kappa(f) * d,

so ``kappa`` is treated as a *power* attenuation coefficient in 1/m.  The
corresponding complex-channel amplitude multiplier is exp(-kappa d / 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


C0_M_PER_S = 299_792_458.0
TEN_LOG10_E = float(10.0 * np.log10(np.e))


def _return_scalar_if_scalar(value: np.ndarray, original: object):
    if np.isscalar(original):
        return float(np.asarray(value))
    return value


@dataclass(frozen=True)
class AbsorptionTable:
    """Tabulated frequency and power-absorption coefficient pairs."""

    frequency_hz: np.ndarray
    coefficient_per_m: np.ndarray
    source_path: Path

    @classmethod
    def from_txt(
        cls,
        path: str | Path,
        *,
        max_frequency_hz: float | None = None,
    ) -> "AbsorptionTable":
        source = Path(path)
        data = np.loadtxt(source, dtype=float)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("Absorption table must contain exactly two columns")

        frequency_hz = data[:, 0]
        coefficient = data[:, 1]
        valid = np.isfinite(frequency_hz) & np.isfinite(coefficient)
        valid &= frequency_hz >= 0.0
        valid &= coefficient >= 0.0
        if max_frequency_hz is not None:
            valid &= frequency_hz <= max_frequency_hz
        frequency_hz = frequency_hz[valid]
        coefficient = coefficient[valid]
        if frequency_hz.size < 2:
            raise ValueError("Absorption table has fewer than two valid rows")

        order = np.argsort(frequency_hz, kind="stable")
        frequency_hz = frequency_hz[order]
        coefficient = coefficient[order]

        unique_frequency, inverse = np.unique(frequency_hz, return_inverse=True)
        if unique_frequency.size != frequency_hz.size:
            sums = np.bincount(inverse, weights=coefficient)
            counts = np.bincount(inverse)
            coefficient = sums / counts
            frequency_hz = unique_frequency

        if not np.all(np.diff(frequency_hz) > 0.0):
            raise ValueError("Absorption frequencies must be strictly increasing")
        return cls(frequency_hz, coefficient, source.resolve())

    @property
    def frequency_range_hz(self) -> tuple[float, float]:
        return float(self.frequency_hz[0]), float(self.frequency_hz[-1])

    def coefficient(
        self,
        frequency_hz,
        *,
        method: Literal["linear", "legacy-nearest"] = "linear",
        out_of_range: Literal["raise", "clip"] = "raise",
        legacy_tolerance_hz: float = 9.894e8,
    ):
        """Return kappa(f) in 1/m.

        ``legacy-nearest`` reproduces the supplied MATLAB function: the nearest
        row is used only when its distance is strictly below the tolerance;
        otherwise the coefficient is zero.
        """

        original = frequency_hz
        query = np.asarray(frequency_hz, dtype=float)
        if np.any(~np.isfinite(query)) or np.any(query < 0.0):
            raise ValueError("Frequencies must be finite and non-negative")

        if method == "linear":
            lower, upper = self.frequency_range_hz
            if out_of_range == "raise" and np.any((query < lower) | (query > upper)):
                raise ValueError(f"Frequency outside table range [{lower}, {upper}] Hz")
            if out_of_range not in {"raise", "clip"}:
                raise ValueError(f"Unsupported out_of_range mode: {out_of_range}")
            query_used = np.clip(query, lower, upper)
            result = np.interp(query_used, self.frequency_hz, self.coefficient_per_m)
            return _return_scalar_if_scalar(result, original)

        if method != "legacy-nearest":
            raise ValueError(f"Unsupported interpolation method: {method}")
        if legacy_tolerance_hz <= 0.0:
            raise ValueError("legacy_tolerance_hz must be positive")

        insertion = np.searchsorted(self.frequency_hz, query, side="left")
        high = np.clip(insertion, 0, self.frequency_hz.size - 1)
        low = np.clip(insertion - 1, 0, self.frequency_hz.size - 1)
        choose_high = np.abs(self.frequency_hz[high] - query) < np.abs(
            self.frequency_hz[low] - query
        )
        nearest = np.where(choose_high, high, low)
        distance = np.abs(self.frequency_hz[nearest] - query)
        result = np.where(
            distance < legacy_tolerance_hz,
            self.coefficient_per_m[nearest],
            0.0,
        )
        return _return_scalar_if_scalar(result, original)

    def loss_db(self, frequency_hz, distance_m, **coefficient_kwargs):
        """Molecular absorption power loss in dB."""

        distance = np.asarray(distance_m, dtype=float)
        if np.any(~np.isfinite(distance)) or np.any(distance < 0.0):
            raise ValueError("Distances must be finite and non-negative")
        kappa = self.coefficient(frequency_hz, **coefficient_kwargs)
        return np.asarray(kappa) * distance * TEN_LOG10_E

    def power_transmission(self, frequency_hz, distance_m, **coefficient_kwargs):
        """Power transmission factor exp(-kappa d)."""

        kappa = self.coefficient(frequency_hz, **coefficient_kwargs)
        return np.exp(-np.asarray(kappa) * np.asarray(distance_m, dtype=float))

    def amplitude_transmission(self, frequency_hz, distance_m, **coefficient_kwargs):
        """Complex-channel magnitude factor exp(-kappa d / 2)."""

        kappa = self.coefficient(frequency_hz, **coefficient_kwargs)
        return np.exp(-0.5 * np.asarray(kappa) * np.asarray(distance_m, dtype=float))


def spreading_loss_db(frequency_hz, distance_m):
    """Free-space spreading loss, 20 log10(4 pi f d / c)."""

    frequency = np.asarray(frequency_hz, dtype=float)
    distance = np.asarray(distance_m, dtype=float)
    if np.any(frequency <= 0.0) or np.any(distance <= 0.0):
        raise ValueError("Frequency and distance must be positive")
    return 20.0 * np.log10(4.0 * np.pi * frequency * distance / C0_M_PER_S)


def free_space_amplitude(frequency_hz, distance_m):
    """Free-space field-amplitude factor c/(4 pi f d)."""

    frequency = np.asarray(frequency_hz, dtype=float)
    distance = np.asarray(distance_m, dtype=float)
    if np.any(frequency <= 0.0) or np.any(distance <= 0.0):
        raise ValueError("Frequency and distance must be positive")
    return C0_M_PER_S / (4.0 * np.pi * frequency * distance)


def medium_emission_noise_temperature(
    frequency_hz,
    distance_m,
    table: AbsorptionTable,
    *,
    medium_temperature_k: float = 296.0,
    **coefficient_kwargs,
):
    """Isothermal radiative-transfer emission term T(1-exp(-kappa d)).

    This is kept separate from receiver noise.  It must not be added blindly
    when a system noise temperature already includes atmospheric brightness.
    """

    if medium_temperature_k <= 0.0:
        raise ValueError("medium_temperature_k must be positive")
    transmission = table.power_transmission(
        frequency_hz, distance_m, **coefficient_kwargs
    )
    return medium_temperature_k * (1.0 - transmission)

