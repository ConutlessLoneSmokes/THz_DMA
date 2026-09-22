"""Three-dimensional geometry for a planar multi-microstrip DMA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .atmosphere import AbsorptionTable, C0_M_PER_S, free_space_amplitude


@dataclass(frozen=True)
class PlanarPath:
    """One specular path referenced to the DMA aperture centre."""

    range_m: float
    azimuth_rad: float
    elevation_rad: float
    complex_gain: complex = 1.0 + 0.0j

    def __post_init__(self) -> None:
        values = np.array(
            [self.range_m, self.azimuth_rad, self.elevation_rad], dtype=float
        )
        if not np.all(np.isfinite(values)) or self.range_m <= 0.0:
            raise ValueError("path range and angles must be finite, with positive range")
        if not np.isfinite(self.complex_gain.real) or not np.isfinite(
            self.complex_gain.imag
        ):
            raise ValueError("path gain must be finite")


def planar_dma_positions(
    num_microstrips: int,
    elements_per_microstrip: int,
    element_spacing_m: float,
    microstrip_spacing_m: float,
    *,
    centered: bool = True,
) -> np.ndarray:
    """Return positions with shape ``(microstrip, element, xyz)``.

    The aperture lies in the yz-plane.  Elements advance along y within each
    microstrip, while microstrips advance along z.  Broadside is the +x axis.
    """

    if num_microstrips < 1 or elements_per_microstrip < 1:
        raise ValueError("array dimensions must be positive")
    if element_spacing_m <= 0.0 or microstrip_spacing_m <= 0.0:
        raise ValueError("array spacings must be positive")
    y = np.arange(elements_per_microstrip, dtype=float) * element_spacing_m
    z = np.arange(num_microstrips, dtype=float) * microstrip_spacing_m
    if centered:
        y -= y.mean()
        z -= z.mean()
    positions = np.zeros((num_microstrips, elements_per_microstrip, 3), dtype=float)
    positions[:, :, 1] = y[None, :]
    positions[:, :, 2] = z[:, None]
    return positions


def direction_unit_vector(azimuth_rad: float, elevation_rad: float) -> np.ndarray:
    """Direction from aperture centre using broadside azimuth/elevation."""

    if not np.isfinite([azimuth_rad, elevation_rad]).all():
        raise ValueError("angles must be finite")
    cos_elevation = np.cos(float(elevation_rad))
    return np.array(
        [
            cos_elevation * np.cos(float(azimuth_rad)),
            cos_elevation * np.sin(float(azimuth_rad)),
            np.sin(float(elevation_rad)),
        ],
        dtype=float,
    )


def point_from_range_angles(
    range_m: float, azimuth_rad: float, elevation_rad: float
) -> np.ndarray:
    """Convert aperture-centred spherical coordinates to Cartesian position."""

    if range_m <= 0.0 or not np.isfinite(range_m):
        raise ValueError("range_m must be positive and finite")
    return float(range_m) * direction_unit_vector(azimuth_rad, elevation_rad)


def planar_distances(
    element_positions_m,
    range_m: float,
    azimuth_rad: float,
    elevation_rad: float,
) -> np.ndarray:
    """Exact distance from every planar-DMA element to a point."""

    positions = np.asarray(element_positions_m, dtype=float)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("element_positions_m must have shape (S, E, 3)")
    if not np.all(np.isfinite(positions)):
        raise ValueError("element positions must be finite")
    point = point_from_range_angles(range_m, azimuth_rad, elevation_rad)
    distance = np.linalg.norm(point[None, None, :] - positions, axis=-1)
    if np.any(distance <= 0.0):
        raise ValueError("point coincides with a DMA element")
    return distance


def planar_aperture_extents_m(element_positions_m) -> np.ndarray:
    """Cartesian side lengths of the smallest axis-aligned aperture box."""

    positions = np.asarray(element_positions_m, dtype=float)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("element_positions_m must have shape (S, E, 3)")
    flattened = positions.reshape(-1, 3)
    return np.ptp(flattened, axis=0)


def planar_aperture_diagonal_m(element_positions_m) -> float:
    """Aperture diameter defined by the rectangular corner-to-corner diagonal."""

    return float(np.linalg.norm(planar_aperture_extents_m(element_positions_m)))


def planar_rayleigh_distance_m(frequency_hz: float, element_positions_m) -> float:
    """Rayleigh distance ``2 D^2 / lambda`` using the planar diagonal D."""

    if frequency_hz <= 0.0 or not np.isfinite(frequency_hz):
        raise ValueError("frequency_hz must be positive and finite")
    diameter = planar_aperture_diagonal_m(element_positions_m)
    return 2.0 * diameter**2 * float(frequency_hz) / C0_M_PER_S


def spherical_planar_path_channel(
    frequencies_hz,
    element_positions_m,
    path: PlanarPath,
    *,
    absorption_table: AbsorptionTable | None = None,
    include_absorption: bool = True,
    interpolation: str = "linear",
) -> np.ndarray:
    """Exact spherical path channel with spreading and molecular absorption."""

    frequencies = np.atleast_1d(np.asarray(frequencies_hz, dtype=float))
    if frequencies.ndim != 1 or frequencies.size == 0 or np.any(frequencies <= 0.0):
        raise ValueError("frequencies_hz must be a non-empty positive vector")
    distance = planar_distances(
        element_positions_m,
        path.range_m,
        path.azimuth_rad,
        path.elevation_rad,
    )
    amplitude = free_space_amplitude(
        frequencies[:, None, None], distance[None, :, :]
    )
    if include_absorption:
        if absorption_table is None:
            raise ValueError("absorption_table is required when absorption is enabled")
        kappa = np.asarray(
            absorption_table.coefficient(frequencies, method=interpolation), dtype=float
        )
        amplitude *= np.exp(-0.5 * kappa[:, None, None] * distance[None, :, :])
    phase = np.exp(
        -1j
        * 2.0
        * np.pi
        * frequencies[:, None, None]
        * distance[None, :, :]
        / C0_M_PER_S
    )
    return complex(path.complex_gain) * amplitude * phase


def separable_compatibility_path_channel(
    frequencies_hz,
    element_positions_m,
    path: PlanarPath,
    *,
    carrier_frequency_hz: float,
) -> np.ndarray:
    """Plane-wave compatibility model with separable spatial/frequency factors.

    The carrier sets the spatial steering phase, while the OFDM frequency only
    sets common path delay and spreading.  This is deliberately an M0
    implementation check, not the target THz propagation model.
    """

    frequencies = np.atleast_1d(np.asarray(frequencies_hz, dtype=float))
    positions = np.asarray(element_positions_m, dtype=float)
    if frequencies.ndim != 1 or frequencies.size == 0 or np.any(frequencies <= 0.0):
        raise ValueError("frequencies_hz must be a non-empty positive vector")
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("element_positions_m must have shape (S, E, 3)")
    if carrier_frequency_hz <= 0.0:
        raise ValueError("carrier_frequency_hz must be positive")
    direction = direction_unit_vector(path.azimuth_rad, path.elevation_rad)
    projected_distance = np.einsum("sed,d->se", positions, direction)
    spatial = np.exp(
        1j
        * 2.0
        * np.pi
        * float(carrier_frequency_hz)
        * projected_distance
        / C0_M_PER_S
    )
    frequency_factor = free_space_amplitude(frequencies, path.range_m) * np.exp(
        -1j * 2.0 * np.pi * frequencies * path.range_m / C0_M_PER_S
    )
    return (
        complex(path.complex_gain)
        * frequency_factor[:, None, None]
        * spatial[None, :, :]
    )


def multipath_planar_channel(
    frequencies_hz,
    element_positions_m,
    paths: list[PlanarPath] | tuple[PlanarPath, ...],
    *,
    model: str,
    carrier_frequency_hz: float,
    absorption_table: AbsorptionTable | None = None,
    include_absorption: bool = True,
    interpolation: str = "linear",
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Sum an arbitrary number of planar-array paths."""

    if not paths:
        raise ValueError("at least one path is required")
    path_channels = []
    for path in paths:
        if model == "spherical_thz":
            channel = spherical_planar_path_channel(
                frequencies_hz,
                element_positions_m,
                path,
                absorption_table=absorption_table,
                include_absorption=include_absorption,
                interpolation=interpolation,
            )
        elif model == "separable_compatibility":
            channel = separable_compatibility_path_channel(
                frequencies_hz,
                element_positions_m,
                path,
                carrier_frequency_hz=carrier_frequency_hz,
            )
        else:
            raise ValueError(f"unsupported planar channel model: {model}")
        path_channels.append(channel)
    return np.sum(path_channels, axis=0), tuple(path_channels)


def sample_stage2_paths(config: dict, rng: np.random.Generator) -> tuple[PlanarPath, ...]:
    """Draw LoS plus specular reflections as one independent scene unit."""

    channel = config["channel"]
    paths = int(channel["num_paths"])
    if paths < 1:
        raise ValueError("num_paths must be positive")
    los_range = rng.uniform(float(channel["range_min_m"]), float(channel["range_max_m"]))
    los_azimuth = np.deg2rad(
        rng.uniform(float(channel["azimuth_min_deg"]), float(channel["azimuth_max_deg"]))
    )
    los_elevation = np.deg2rad(
        rng.uniform(
            float(channel["elevation_min_deg"]),
            float(channel["elevation_max_deg"]),
        )
    )
    los_magnitude = float(channel["los_complex_gain_magnitude"])
    if los_magnitude <= 0.0:
        raise ValueError("los_complex_gain_magnitude must be positive")
    generated = [
        PlanarPath(
            range_m=float(los_range),
            azimuth_rad=float(los_azimuth),
            elevation_rad=float(los_elevation),
            complex_gain=los_magnitude * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi)),
        )
    ]
    for _ in range(paths - 1):
        azimuth_sign = rng.choice(np.array([-1.0, 1.0]))
        elevation_sign = rng.choice(np.array([-1.0, 1.0]))
        gain_db = rng.uniform(
            float(channel["reflection_gain_relative_min_db"]),
            float(channel["reflection_gain_relative_max_db"]),
        )
        generated.append(
            PlanarPath(
                range_m=float(
                    los_range
                    + rng.uniform(
                        float(channel["reflection_excess_range_min_m"]),
                        float(channel["reflection_excess_range_max_m"]),
                    )
                ),
                azimuth_rad=float(
                    los_azimuth
                    + azimuth_sign
                    * np.deg2rad(
                        rng.uniform(
                            float(channel["reflection_azimuth_separation_min_deg"]),
                            float(channel["reflection_azimuth_separation_max_deg"]),
                        )
                    )
                ),
                elevation_rad=float(
                    los_elevation
                    + elevation_sign
                    * np.deg2rad(
                        rng.uniform(
                            float(channel["reflection_elevation_separation_min_deg"]),
                            float(channel["reflection_elevation_separation_max_deg"]),
                        )
                    )
                ),
                complex_gain=float(10.0 ** (gain_db / 20.0))
                * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi)),
            )
        )
    return tuple(generated)


def channel_from_planar_estimate(frequencies, platform, coordinates, gains, config):
    paths = tuple(
        PlanarPath(
            range_m=float(coordinates[index, 0]),
            azimuth_rad=float(coordinates[index, 1]),
            elevation_rad=float(coordinates[index, 2]),
            complex_gain=complex(gains[index]),
        )
        for index in range(coordinates.shape[0])
    )
    channel, _ = multipath_planar_channel(
        frequencies,
        platform.element_positions_m,
        paths,
        model="spherical_thz",
        carrier_frequency_hz=float(config["system"]["carrier_frequency_hz"]),
        absorption_table=platform.absorption_table,
        include_absorption=bool(config["channel"]["include_absorption"]),
        interpolation=str(config["atmosphere"]["interpolation"]),
    )
    return channel
