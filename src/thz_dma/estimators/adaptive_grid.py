"""Adaptive local candidate-grid construction for planar sparse estimation."""

from __future__ import annotations

import copy

import numpy as np

from thz_dma.channels.planar import PlanarPath, sample_stage2_paths
from thz_dma.estimators.planar_sparse import (
    PlanarDmaTemplateOperator,
    PlanarTemplateBatchEvaluator,
)


def _declared_channel_support_bounds(config: dict) -> dict[str, tuple[float, float]]:
    """Return conservative geometry bounds for every generated Stage 2 path."""

    channel = config["channel"]
    has_reflections = int(channel["num_paths"]) > 1
    range_extension = (
        float(channel["reflection_excess_range_max_m"])
        if has_reflections
        else 0.0
    )
    azimuth_extension = (
        float(channel["reflection_azimuth_separation_max_deg"])
        if has_reflections
        else 0.0
    )
    elevation_extension = (
        float(channel["reflection_elevation_separation_max_deg"])
        if has_reflections
        else 0.0
    )
    return {
        "range_m": (
            float(channel["range_min_m"]),
            float(channel["range_max_m"]) + range_extension,
        ),
        "azimuth_deg": (
            float(channel["azimuth_min_deg"]) - azimuth_extension,
            float(channel["azimuth_max_deg"]) + azimuth_extension,
        ),
        "elevation_deg": (
            float(channel["elevation_min_deg"]) - elevation_extension,
            float(channel["elevation_max_deg"]) + elevation_extension,
        ),
    }


def validate_search_contains_channel_support(config: dict) -> dict:
    """Fail before dictionary construction if generated paths can lie outside it."""

    support = _declared_channel_support_bounds(config)
    search = config["search"]
    search_bounds = {
        "range_m": (
            float(search["range_min_m"]),
            float(search["range_max_m"]),
        ),
        "azimuth_deg": (
            float(search["azimuth_min_deg"]),
            float(search["azimuth_max_deg"]),
        ),
        "elevation_deg": (
            float(search["elevation_min_deg"]),
            float(search["elevation_max_deg"]),
        ),
    }
    uncovered = []
    for dimension, truth_bounds in support.items():
        lower, upper = search_bounds[dimension]
        if truth_bounds[0] < lower - 1.0e-12 or truth_bounds[1] > upper + 1.0e-12:
            uncovered.append(dimension)
    if uncovered:
        raise ValueError(
            "search dictionary does not cover all declared path support dimensions: "
            + ", ".join(uncovered)
        )
    return {
        "declared_channel_support_bounds": {
            key: list(value) for key, value in support.items()
        },
        "fixed_search_bounds": {
            key: list(value) for key, value in search_bounds.items()
        },
        "search_contains_declared_channel_support": True,
    }


def search_mode(config: dict) -> str:
    mode = str(config["search"].get("mode", "fixed"))
    if mode not in {"fixed", "beam_center_adaptive"}:
        raise ValueError(f"unsupported Stage 2-A search mode: {mode}")
    return mode


def _search_bounds_radians(config: dict) -> tuple[np.ndarray, np.ndarray]:
    search = config["search"]
    lower = np.array(
        [
            float(search["range_min_m"]),
            float(np.deg2rad(search["azimuth_min_deg"])),
            float(np.deg2rad(search["elevation_min_deg"])),
        ],
        dtype=float,
    )
    upper = np.array(
        [
            float(search["range_max_m"]),
            float(np.deg2rad(search["azimuth_max_deg"])),
            float(np.deg2rad(search["elevation_max_deg"])),
        ],
        dtype=float,
    )
    return lower, upper


def adaptive_settings(config: dict) -> dict:
    if "adaptive_search" not in config:
        raise ValueError("beam_center_adaptive mode requires [adaptive_search]")
    raw = config["adaptive_search"]
    settings = {
        "correlation_threshold": float(raw["correlation_threshold"]),
        "window_fullwidth_multiplier": float(raw["window_fullwidth_multiplier"]),
        "grid_points_per_3db_width": float(raw["grid_points_per_3db_width"]),
        "probe_steps": np.array(
            [
                float(raw["probe_range_step_m"]),
                float(np.deg2rad(raw["probe_azimuth_step_deg"])),
                float(np.deg2rad(raw["probe_elevation_step_deg"])),
            ],
            dtype=float,
        ),
        "bisection_iterations": int(raw["bisection_iterations"]),
        "probe_batch_size": int(raw.get("probe_batch_size", 32)),
        "truth_offset_fraction": float(raw["truth_offset_halfpower_fraction"]),
        "truth_offset_caps": np.array(
            [
                float(raw["truth_offset_cap_range_m"]),
                float(np.deg2rad(raw["truth_offset_cap_azimuth_deg"])),
                float(np.deg2rad(raw["truth_offset_cap_elevation_deg"])),
            ],
            dtype=float,
        ),
        "preserve_channel_relationships": bool(
            raw.get("preserve_channel_relationships", True)
        ),
        "maximum_candidates": int(raw["maximum_candidates_per_scene"]),
    }
    if not 0.0 < settings["correlation_threshold"] < 1.0:
        raise ValueError("adaptive correlation threshold must lie in (0, 1)")
    if settings["window_fullwidth_multiplier"] < 1.0:
        raise ValueError("adaptive window multiplier must be at least one")
    if settings["grid_points_per_3db_width"] < 2.0:
        raise ValueError("adaptive grid requires at least two points per 3 dB width")
    if np.any(settings["probe_steps"] <= 0.0):
        raise ValueError("adaptive profile probe steps must be positive")
    if settings["bisection_iterations"] < 0 or settings["probe_batch_size"] < 1:
        raise ValueError("invalid adaptive profile refinement settings")
    if not 0.0 < settings["truth_offset_fraction"] <= 1.0:
        raise ValueError("truth offset fraction must lie in (0, 1]")
    if np.any(settings["truth_offset_caps"] <= 0.0):
        raise ValueError("truth offset caps must be positive")
    if settings["maximum_candidates"] < 1:
        raise ValueError("maximum adaptive candidate count must be positive")
    return settings


def _normalized_template_rho2(
    evaluator: PlanarTemplateBatchEvaluator,
    reference: np.ndarray,
    coordinates: np.ndarray,
) -> np.ndarray:
    candidates = evaluator.evaluate(
        coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
    )
    reference_energy = float(np.vdot(reference, reference).real)
    candidate_energy = np.sum(np.abs(candidates) ** 2, axis=1, dtype=float)
    if reference_energy <= 0.0 or np.any(candidate_energy <= 0.0):
        raise ValueError("adaptive profile contains a zero-energy template")
    return (
        np.abs(candidates @ np.conjugate(reference)) ** 2
        / candidate_energy
        / reference_energy
    )


def _nearest_profile_crossing(
    evaluator: PlanarTemplateBatchEvaluator,
    center: np.ndarray,
    reference: np.ndarray,
    *,
    dimension: int,
    direction: float,
    bound: float,
    probe_step: float,
    threshold: float,
    bisection_iterations: int,
    batch_size: int,
) -> tuple[float, bool]:
    maximum_distance = float(direction * (bound - center[dimension]))
    if maximum_distance < -1.0e-14:
        raise ValueError("adaptive profile bound is on the wrong side of its center")
    if maximum_distance <= 1.0e-14:
        return 0.0, True
    count = int(np.ceil(maximum_distance / probe_step))
    previous_distance = 0.0
    for first in range(1, count + 1, batch_size):
        indices = np.arange(first, min(count, first + batch_size - 1) + 1)
        distances = np.minimum(indices * probe_step, maximum_distance)
        points = np.repeat(center[None, :], distances.size, axis=0)
        points[:, dimension] += direction * distances
        rho2 = _normalized_template_rho2(evaluator, reference, points)
        crossings = np.flatnonzero(rho2 <= threshold)
        if crossings.size:
            crossing_index = int(crossings[0])
            high = float(distances[crossing_index])
            low = float(
                distances[crossing_index - 1]
                if crossing_index > 0
                else previous_distance
            )
            for _ in range(bisection_iterations):
                middle = 0.5 * (low + high)
                point = center.copy()
                point[dimension] += direction * middle
                value = float(
                    _normalized_template_rho2(
                        evaluator, reference, point[None, :]
                    )[0]
                )
                if value > threshold:
                    low = middle
                else:
                    high = middle
            return high, False
        previous_distance = float(distances[-1])
    return maximum_distance, True


def _measure_adaptive_profiles(
    operators: dict[str, PlanarDmaTemplateOperator],
    centers: np.ndarray,
    config: dict,
    evaluators: dict[str, PlanarTemplateBatchEvaluator] | None = None,
) -> dict:
    settings = adaptive_settings(config)
    lower, upper = _search_bounds_radians(config)
    schedule_profiles: dict[str, list[dict]] = {
        name: [] for name in operators
    }
    for schedule_name, operator in operators.items():
        evaluator = (
            evaluators[schedule_name]
            if evaluators is not None
            else PlanarTemplateBatchEvaluator(operator)
        )
        for center in centers:
            reference = evaluator.evaluate(
                np.array([center[0]]),
                np.array([center[1]]),
                np.array([center[2]]),
            )[0]
            negative = []
            positive = []
            truncated = []
            for dimension in range(3):
                left, left_truncated = _nearest_profile_crossing(
                    evaluator,
                    center,
                    reference,
                    dimension=dimension,
                    direction=-1.0,
                    bound=lower[dimension],
                    probe_step=float(settings["probe_steps"][dimension]),
                    threshold=float(settings["correlation_threshold"]),
                    bisection_iterations=int(settings["bisection_iterations"]),
                    batch_size=int(settings["probe_batch_size"]),
                )
                right, right_truncated = _nearest_profile_crossing(
                    evaluator,
                    center,
                    reference,
                    dimension=dimension,
                    direction=1.0,
                    bound=upper[dimension],
                    probe_step=float(settings["probe_steps"][dimension]),
                    threshold=float(settings["correlation_threshold"]),
                    bisection_iterations=int(settings["bisection_iterations"]),
                    batch_size=int(settings["probe_batch_size"]),
                )
                negative.append(left)
                positive.append(right)
                truncated.append([left_truncated, right_truncated])
            full_width = np.asarray(negative) + np.asarray(positive)
            if np.any(full_width <= 0.0):
                raise RuntimeError("adaptive 3 dB profile has zero full width")
            schedule_profiles[schedule_name].append(
                {
                    "negative_halfwidth_rad_units": list(map(float, negative)),
                    "positive_halfwidth_rad_units": list(map(float, positive)),
                    "fullwidth_rad_units": full_width.astype(float).tolist(),
                    "crossing_truncated": truncated,
                }
            )
    return {
        "settings": settings,
        "schedule_profiles": schedule_profiles,
    }


def _path_relationships_valid(coordinates: np.ndarray, config: dict) -> bool:
    channel = config["channel"]
    los = coordinates[0]
    if not (
        float(channel["range_min_m"]) <= los[0] <= float(channel["range_max_m"])
        and np.deg2rad(float(channel["azimuth_min_deg"]))
        <= los[1]
        <= np.deg2rad(float(channel["azimuth_max_deg"]))
        and np.deg2rad(float(channel["elevation_min_deg"]))
        <= los[2]
        <= np.deg2rad(float(channel["elevation_max_deg"]))
    ):
        return False
    for reflection in coordinates[1:]:
        excess_range = reflection[0] - los[0]
        azimuth_separation = abs(np.rad2deg(reflection[1] - los[1]))
        elevation_separation = abs(np.rad2deg(reflection[2] - los[2]))
        if not (
            float(channel["reflection_excess_range_min_m"])
            <= excess_range
            <= float(channel["reflection_excess_range_max_m"])
            and float(channel["reflection_azimuth_separation_min_deg"])
            <= azimuth_separation
            <= float(channel["reflection_azimuth_separation_max_deg"])
            and float(channel["reflection_elevation_separation_min_deg"])
            <= elevation_separation
            <= float(channel["reflection_elevation_separation_max_deg"])
        ):
            return False
    return True


def _sample_truth_around_centers(
    center_paths: tuple[PlanarPath, ...],
    profile: dict,
    config: dict,
    rng: np.random.Generator,
) -> tuple[tuple[PlanarPath, ...], dict]:
    centers = np.array(
        [
            [path.range_m, path.azimuth_rad, path.elevation_rad]
            for path in center_paths
        ],
        dtype=float,
    )
    settings = profile["settings"]
    schedules = list(profile["schedule_profiles"])
    negative = np.array(
        [
            [
                min(
                    profile["schedule_profiles"][schedule][path_index][
                        "negative_halfwidth_rad_units"
                    ][dimension]
                    for schedule in schedules
                )
                for dimension in range(3)
            ]
            for path_index in range(len(center_paths))
        ],
        dtype=float,
    )
    positive = np.array(
        [
            [
                min(
                    profile["schedule_profiles"][schedule][path_index][
                        "positive_halfwidth_rad_units"
                    ][dimension]
                    for schedule in schedules
                )
                for dimension in range(3)
            ]
            for path_index in range(len(center_paths))
        ],
        dtype=float,
    )
    caps = np.asarray(settings["truth_offset_caps"], dtype=float)[None, :]
    negative = np.minimum(
        negative * float(settings["truth_offset_fraction"]), caps
    )
    positive = np.minimum(
        positive * float(settings["truth_offset_fraction"]), caps
    )
    lower, upper = _search_bounds_radians(config)
    for _ in range(10_000):
        offsets = rng.uniform(-negative, positive)
        coordinates = centers + offsets
        if np.any(coordinates < lower[None, :]) or np.any(
            coordinates > upper[None, :]
        ):
            continue
        if bool(settings["preserve_channel_relationships"]) and not (
            _path_relationships_valid(coordinates, config)
        ):
            continue
        paths = tuple(
            PlanarPath(
                range_m=float(coordinates[index, 0]),
                azimuth_rad=float(coordinates[index, 1]),
                elevation_rad=float(coordinates[index, 2]),
                complex_gain=center_paths[index].complex_gain,
            )
            for index in range(len(center_paths))
        )
        return paths, {
            "offsets_m_rad_rad": offsets.astype(float).tolist(),
            "negative_offset_limits_m_rad_rad": negative.astype(float).tolist(),
            "positive_offset_limits_m_rad_rad": positive.astype(float).tolist(),
        }
    raise RuntimeError("could not sample a beam-conditioned path scene")


def _lattice_indices(
    lower: float,
    upper: float,
    *,
    anchor: float,
    step: float,
    maximum_index: int,
) -> np.ndarray:
    tolerance = 1.0e-10
    first = max(0, int(np.ceil((lower - anchor) / step - tolerance)))
    last = min(
        maximum_index, int(np.floor((upper - anchor) / step + tolerance))
    )
    if first <= last:
        return np.arange(first, last + 1, dtype=int)
    middle = int(np.clip(np.rint((0.5 * (lower + upper) - anchor) / step), 0, maximum_index))
    return np.array([middle], dtype=int)


def _adaptive_candidate_union(
    centers: np.ndarray,
    profile: dict,
    truth_coordinates: np.ndarray,
    config: dict,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict]:
    settings = profile["settings"]
    schedule_profiles = profile["schedule_profiles"]
    schedule_names = list(schedule_profiles)
    full_widths = np.array(
        [
            [
                max(
                    schedule_profiles[name][path_index]["fullwidth_rad_units"][dimension]
                    for name in schedule_names
                )
                for dimension in range(3)
            ]
            for path_index in range(centers.shape[0])
        ],
        dtype=float,
    )
    minimum_widths = np.min(
        np.array(
            [
                [item["fullwidth_rad_units"] for item in schedule_profiles[name]]
                for name in schedule_names
            ],
            dtype=float,
        ),
        axis=(0, 1),
    )
    step_caps = np.array(
        [
            float(config["search"]["range_step_m"]),
            float(np.deg2rad(config["search"]["azimuth_step_deg"])),
            float(np.deg2rad(config["search"]["elevation_step_deg"])),
        ],
        dtype=float,
    )
    steps = np.minimum(
        minimum_widths / float(settings["grid_points_per_3db_width"]),
        step_caps,
    )
    if np.any(steps <= 0.0) or not np.all(np.isfinite(steps)):
        raise RuntimeError("adaptive grid step is invalid")
    global_lower, global_upper = _search_bounds_radians(config)
    maximum_indices = np.floor(
        (global_upper - global_lower) / steps + 1.0e-10
    ).astype(int)
    half_span = (
        0.5
        * float(settings["window_fullwidth_multiplier"])
        * full_widths
    )
    window_lower = np.maximum(centers - half_span, global_lower[None, :])
    window_upper = np.minimum(centers + half_span, global_upper[None, :])
    index_blocks = []
    windows = []
    for path_index in range(centers.shape[0]):
        axes = [
            _lattice_indices(
                float(window_lower[path_index, dimension]),
                float(window_upper[path_index, dimension]),
                anchor=float(global_lower[dimension]),
                step=float(steps[dimension]),
                maximum_index=int(maximum_indices[dimension]),
            )
            for dimension in range(3)
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        indices = np.column_stack([item.ravel() for item in mesh])
        index_blocks.append(indices)
        windows.append(
            {
                "path_center_index": path_index,
                "continuous_lower_m_rad_rad": window_lower[path_index].astype(float).tolist(),
                "continuous_upper_m_rad_rad": window_upper[path_index].astype(float).tolist(),
                "lattice_shape": [int(axis.size) for axis in axes],
                "lattice_candidates_before_union": int(indices.shape[0]),
            }
        )
    union_indices = np.unique(np.vstack(index_blocks), axis=0)
    if union_indices.shape[0] > int(settings["maximum_candidates"]):
        raise RuntimeError(
            "adaptive dictionary exceeds maximum_candidates_per_scene: "
            f"{union_indices.shape[0]} > {settings['maximum_candidates']}"
        )
    coordinates = global_lower[None, :] + union_indices * steps[None, :]
    coverage = np.all(
        (truth_coordinates >= window_lower)
        & (truth_coordinates <= window_upper),
        axis=1,
    )
    if not np.all(coverage):
        raise RuntimeError("beam-conditioned truth escaped its associated local window")
    nearest_error = np.empty_like(truth_coordinates)
    for path_index, truth in enumerate(truth_coordinates):
        distance = np.abs(coordinates - truth[None, :])
        nearest_index = int(np.argmin(np.sum((distance / steps) ** 2, axis=1)))
        nearest_error[path_index] = distance[nearest_index]
    return (
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2],
    ), {
        "grid_steps_m_rad_rad": steps.astype(float).tolist(),
        "combined_fullwidths_m_rad_rad": full_widths.astype(float).tolist(),
        "windows": windows,
        "num_candidates": int(coordinates.shape[0]),
        "truth_associated_window_coverage": coverage.astype(bool).tolist(),
        "nearest_union_candidate_abs_error_m_rad_rad": nearest_error.astype(float).tolist(),
    }


def prepare_adaptive_stage2a_scene(
    config: dict,
    operators: dict[str, PlanarDmaTemplateOperator],
    *,
    center_rng: np.random.Generator,
    truth_rng: np.random.Generator,
    profile_evaluators: dict[str, PlanarTemplateBatchEvaluator] | None = None,
) -> tuple[tuple[PlanarPath, ...], tuple[np.ndarray, np.ndarray, np.ndarray], dict]:
    """Generate known coarse centers, continuous truth, and a local union grid."""

    center_paths = sample_stage2_paths(config, center_rng)
    centers = np.array(
        [
            [path.range_m, path.azimuth_rad, path.elevation_rad]
            for path in center_paths
        ],
        dtype=float,
    )
    profile = _measure_adaptive_profiles(
        operators, centers, config, profile_evaluators
    )
    paths, truth_metadata = _sample_truth_around_centers(
        center_paths, profile, config, truth_rng
    )
    truth_coordinates = np.array(
        [[path.range_m, path.azimuth_rad, path.elevation_rad] for path in paths],
        dtype=float,
    )
    candidates, grid_metadata = _adaptive_candidate_union(
        centers, profile, truth_coordinates, config
    )
    serializable_profiles = copy.deepcopy(profile["schedule_profiles"])
    return paths, candidates, {
        "center_source": "pre-observation random coarse path-beam centers",
        "centers_m_rad_rad": centers.astype(float).tolist(),
        "truth": truth_metadata,
        "schedule_profiles": serializable_profiles,
        **grid_metadata,
    }
