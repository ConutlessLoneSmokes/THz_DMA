#!/usr/bin/env python3
"""P-016 diagnostic: a fixed-RE F1 state and configuration-count curve."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
from _run_support import (  # noqa: E402
    PROJECT_ROOT as ROOT,
    create_run_archive,
    hash_file,
    read_toml,
    write_json,
    write_raw_metrics,
)

import numpy as np  # noqa: E402

from check_stage2a_f1_peak_decomposition import (  # noqa: E402
    _coordinates,
    _peak_summary,
    _spectrum,
)
from thz_dma.channels.planar import (  # noqa: E402
    channel_from_planar_estimate,
    multipath_planar_channel,
)
from thz_dma.estimators.adaptive_grid import prepare_adaptive_stage2a_scene  # noqa: E402
from thz_dma.estimators.planar_baselines import (  # noqa: E402
    _validate_accelerated_dictionary,
    build_stage2a_operators,
    estimate_planar_baseline,
    operator_for_schedule,
)
from thz_dma.estimators.planar_sparse import (  # noqa: E402
    PlanarTemplateBatchEvaluator,
    build_planar_template_dictionary_from_candidates,
)
from thz_dma.evaluation.beamforming import (  # noqa: E402
    build_planar_data_codebook,
    feasible_planar_data_metrics,
)
from thz_dma.evaluation.paths import match_planar_paths  # noqa: E402
from thz_dma.metrics.diagnostics import channel_domain_metrics  # noqa: E402
from thz_dma.observations.multistrip import (  # noqa: E402
    array_input_noise_variance,
    effective_tensor_output_snr_db,
    observe_multistrip_schedule,
)
from thz_dma.surfaces.multistrip import (  # noqa: E402
    cyclic_frequency_resonance_states,
    multistrip_dma_weights,
)


F1 = "frequency_single_shot_J1_K128"
METHODS = ("oracle_geometry_ls", "yang_ogols_3d")


def _load() -> tuple[dict, dict, Path, Path]:
    plan_path = ROOT / "configs/direction1_stage2a_p016_config_curve.toml"
    plan = read_toml(plan_path)["experiment"]
    source_path = ROOT / plan["source_config"]
    base = read_toml(source_path)
    if base["experiment"]["replicate_seeds"] != [
        2026091101, 2026091102, 2026091103, 2026091104, 2026091105
    ]:
        raise RuntimeError("D-024 evaluation seeds changed")
    if base["noise"]["array_reference_snr_db"] != [55.0, 65.0]:
        raise RuntimeError("D-024 SNR levels changed")
    counts = [int(value) for value in plan["configuration_counts"]]
    bank_size = int(plan["state_bank_size"])
    if counts != [1, 2, 4, 8, 16] or bank_size != 16:
        raise ValueError("P-016 requires its declared nested 1/2/4/8/16 curve")
    if any(128 % count or bank_size % count for count in counts):
        raise ValueError("configuration count does not divide both resources and bank")
    if float(base["acquisition"]["pilot_energy_per_re"]) != 1.0:
        raise ValueError("P-016 requires unit energy at each of 128 pilot RE")
    return plan, base, plan_path, source_path


def _state_bank(platform, base: dict, bank_size: int) -> np.ndarray:
    dma = base["dma"]
    resonances = cyclic_frequency_resonance_states(
        bank_size,
        int(dma["num_microstrips"]),
        int(dma["elements_per_microstrip"]),
        float(base["system"]["carrier_frequency_hz"]),
        float(dma["tuning_bandwidth_hz"]),
    )
    weights = multistrip_dma_weights(
        platform.frequencies_hz,
        resonances,
        platform.feed_distances_m,
        quality_factor=float(dma["quality_factor"]),
        reference_frequency_hz=float(base["system"]["carrier_frequency_hz"]),
        waveguide_effective_index=float(dma["waveguide_effective_index"]),
        waveguide_end_power_fraction=float(dma["waveguide_end_power_fraction"]),
    )
    return weights


def _packed_weights(bank: np.ndarray, anchor: int, configurations: int) -> tuple[np.ndarray, list[int]]:
    bank_size, frequencies = bank.shape[:2]
    if frequencies != 128 or bank_size % configurations or 128 % configurations:
        raise ValueError("invalid state bank or RE partition")
    state_ids = [
        (anchor + symbol * bank_size // configurations) % bank_size
        for symbol in range(configurations)
    ]
    subcarriers = np.arange(frequencies)
    packed = bank[np.asarray(state_ids)[subcarriers % configurations], subcarriers][None]
    if packed.shape != (1, 128, *bank.shape[2:]):
        raise RuntimeError("packed state shape mismatch")
    return packed, state_ids


def _output_snr(weights: np.ndarray, channel: np.ndarray, element_variance: float) -> float:
    clean = np.sum(np.conjugate(weights[0]) * channel, axis=-1)
    noise = element_variance * np.sum(np.abs(weights[0]) ** 2, axis=-1)
    return float(10.0 * np.log10(np.sum(np.abs(clean) ** 2) / np.sum(noise)))


def _scene(base: dict, operators: dict, evaluators: dict, seed: int, sample_index: int):
    paths, candidates, metadata = prepare_adaptive_stage2a_scene(
        base,
        operators,
        center_rng=np.random.default_rng(
            np.random.SeedSequence([seed, sample_index, 2_026_091_101])
        ),
        truth_rng=np.random.default_rng(
            np.random.SeedSequence([seed, sample_index, 2_026_091_102])
        ),
        profile_evaluators=evaluators,
    )
    case = copy.deepcopy(base)
    steps = np.asarray(metadata["grid_steps_m_rad_rad"], dtype=float)
    case["search"]["range_step_m"] = float(steps[0])
    case["search"]["azimuth_step_deg"] = float(np.rad2deg(steps[1]))
    case["search"]["elevation_step_deg"] = float(np.rad2deg(steps[2]))
    return paths, candidates, metadata, case


def _channel(platform, base: dict, paths):
    channel, _ = multipath_planar_channel(
        platform.frequencies_hz,
        platform.element_positions_m,
        paths,
        model="spherical_thz",
        carrier_frequency_hz=float(base["system"]["carrier_frequency_hz"]),
        absorption_table=platform.absorption_table,
        include_absorption=bool(base["channel"]["include_absorption"]),
        interpolation=str(base["atmosphere"]["interpolation"]),
    )
    return channel


def _dictionary(operator, candidates, base: dict):
    search = base["search"]
    return build_planar_template_dictionary_from_candidates(
        operator,
        *candidates,
        chunk_size=int(search["dictionary_chunk_size"]),
        storage_dtype=str(search["dictionary_storage_dtype"]),
        compute_backend=str(search["dictionary_compute_backend"]),
        compute_device=str(search["dictionary_compute_device"]),
    )


def _margins(dictionary, operator, paths, base: dict) -> list[float]:
    truths = _coordinates(paths)
    gains = np.asarray([path.complex_gain for path in paths])
    exact = operator.evaluate(*truths.T)
    return [
        float(
            _peak_summary(
                dictionary,
                _spectrum(dictionary, gains[index] * exact[index]),
                truths[index : index + 1],
                base,
            )["truth_over_false_margin_db"]
        )
        for index in range(len(paths))
    ]


def _fit_nmse(operator, dictionary, observation, paths, case, platform, base, truth_channel):
    fit = estimate_planar_baseline(
        "yang_ogols_3d", dictionary, operator, observation, paths, case
    )
    matched = match_planar_paths(fit, paths, case)
    estimate = channel_from_planar_estimate(
        platform.frequencies_hz,
        platform,
        matched["coordinates"],
        matched["gains"],
        base,
    )
    ratio = float(
        np.sum(np.abs(estimate - truth_channel) ** 2)
        / np.sum(np.abs(truth_channel) ** 2)
    )
    return float(10.0 * np.log10(max(ratio, np.finfo(float).tiny)))


def _design(plan, base, platform, bank, base_operators, evaluators, old_weights, archive):
    rows = []
    validations = {}
    f1_schedule = next(item for item in platform.schedules if item.name == F1)
    for seed in plan["design_seeds"]:
        for sample_index in range(int(plan["design_samples_per_seed"])):
            paths, candidates, _, _ = _scene(base, base_operators, evaluators, int(seed), sample_index)
            channel = _channel(platform, base, paths)
            variance = array_input_noise_variance(channel, 55.0)
            baseline_snr = _output_snr(old_weights, channel, variance)
            for state in range(int(plan["state_bank_size"])):
                weights, _ = _packed_weights(bank, state, 1)
                operator = operator_for_schedule(platform, f1_schedule, weights, base)
                dictionary = _dictionary(operator, candidates, base)
                if not validations:
                    validations = _validate_accelerated_dictionary(operator, dictionary)
                margins = _margins(dictionary, operator, paths, base)
                rows.append({
                    "scene_id": f"{seed}:{sample_index}",
                    "state": state,
                    "minimum_margin_db": min(margins),
                    "output_snr_db": _output_snr(weights, channel, variance),
                    "old_output_snr_db": baseline_snr,
                })
        print(f"design seed {seed} complete", flush=True)
    write_raw_metrics(archive.directory / "design_metrics.csv", rows)
    by_state = []
    baseline_p10 = float(np.quantile([row["old_output_snr_db"] for row in rows if row["state"] == 0], 0.1))
    for state in range(int(plan["state_bank_size"])):
        selected = [row for row in rows if row["state"] == state]
        by_state.append({
            "state": state,
            "margin_p10_db": float(np.quantile([row["minimum_margin_db"] for row in selected], 0.1)),
            "margin_median_db": float(np.median([row["minimum_margin_db"] for row in selected])),
            "output_snr_p10_db": float(np.quantile([row["output_snr_db"] for row in selected], 0.1)),
        })
    eligible = [
        row for row in by_state
        if row["output_snr_p10_db"] >= baseline_p10 - float(plan["output_snr_p10_max_drop_db"])
    ]
    chosen = (
        max(eligible, key=lambda row: (row["margin_p10_db"], row["margin_median_db"], -row["state"]))
        if eligible else
        max(by_state, key=lambda row: (row["output_snr_p10_db"], -row["state"]))
    )
    frozen = {
        "anchor_state": int(chosen["state"]),
        "single_state_output_snr_constraint_passed": bool(eligible),
        "resonance_state_bank_sha256": hashlib.sha256(
            cyclic_frequency_resonance_states(
                int(plan["state_bank_size"]),
                int(base["dma"]["num_microstrips"]),
                int(base["dma"]["elements_per_microstrip"]),
                float(base["system"]["carrier_frequency_hz"]),
                float(base["dma"]["tuning_bandwidth_hz"]),
            ).tobytes()
        ).hexdigest(),
        "design_scenes": len(rows) // int(plan["state_bank_size"]),
        "old_output_snr_p10_db": baseline_p10,
        "state_scores": by_state,
        "cuda_validation": validations,
        "state_mapping": {
            str(count): [
                (int(chosen["state"]) + symbol * int(plan["state_bank_size"]) // int(count)) % int(plan["state_bank_size"])
                for symbol in range(int(count))
            ]
            for count in plan["configuration_counts"]
        },
    }
    archive.write_json("state_design_frozen.json", frozen)
    return frozen


def _validate_source(plan, base):
    source = ROOT / plan["source_run"]
    peak = ROOT / plan["source_peak_run"]
    status = json.loads((source / "status.json").read_text(encoding="utf-8"))
    source_config = json.loads((source / "config_resolved.json").read_text(encoding="utf-8"))["config"]
    if status.get("status") != "complete" or status.get("raw_metric_rows") != 1200 or source_config != base:
        raise RuntimeError("D-024 archived evaluation does not match the declared source")
    peak_status = json.loads((peak / "status.json").read_text(encoding="utf-8"))
    if peak_status.get("status") != "complete":
        raise RuntimeError("D-026 source peak run is incomplete")
    records = json.loads((source / "metrics_summary.json").read_text(encoding="utf-8"))["search"]["scene_records"]
    if len(records) != 100:
        raise RuntimeError("D-024 scene archive is incomplete")
    with (source / "metrics_raw.csv").open(encoding="utf-8", newline="") as handle:
        f1 = [row for row in csv.DictReader(handle) if row["schedule"] == F1]
    if len(f1) != 600:
        raise RuntimeError("D-024 archived F1 comparison is incomplete")
    hashes = {
        str(path.relative_to(ROOT)): hash_file(path)
        for path in (source / "metrics_raw.csv", source / "metrics_summary.json", peak / "peak_records.json")
    }
    return {item["scene_id"]: item for item in records}, f1, hashes


def _check_scene(metadata, record):
    if int(metadata["num_candidates"]) != int(record["num_candidates"]):
        raise RuntimeError("D-024 candidate count changed")
    for key in ("centers_m_rad_rad",):
        if not np.allclose(metadata[key], record[key], rtol=0.0, atol=1.0e-12):
            raise RuntimeError(f"D-024 {key} changed")
    if not np.allclose(
        metadata["truth"]["offsets_m_rad_rad"],
        record["truth"]["offsets_m_rad_rad"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("D-024 truth offsets changed")


def _evaluate(plan, base, platform, bank, base_operators, evaluators, frozen, archive):
    records, source_f1, source_hashes = _validate_source(plan, base)
    archive.write_json("source_hashes.json", source_hashes)
    f1_schedule = next(item for item in platform.schedules if item.name == F1)
    dma, data_config = base["dma"], base["data_evaluation"]
    codebook, _ = build_planar_data_codebook(
        carrier_frequency_hz=float(base["system"]["carrier_frequency_hz"]),
        num_microstrips=int(dma["num_microstrips"]),
        elements_per_microstrip=int(dma["elements_per_microstrip"]),
        tuning_bandwidth_hz=float(dma["tuning_bandwidth_hz"]),
        feed_distances_m=platform.feed_distances_m,
        quality_factor=float(dma["quality_factor"]),
        waveguide_effective_index=float(dma["waveguide_effective_index"]),
        waveguide_end_power_fraction=float(dma["waveguide_end_power_fraction"]),
        codebook_size=int(data_config["codebook_size"]),
        seed_start=int(data_config["codebook_seed_start"]),
    )
    rows, peak_rows, noiseless_rows = [], [], []
    cuda_checks = {}
    legacy_indices = np.rint(np.linspace(0, 127, 16)).astype(int)
    symbol_seconds = platform.metadata["useful_symbol_duration_s"] + platform.metadata["cyclic_prefix_duration_s"]
    for seed in base["experiment"]["replicate_seeds"]:
        for sample_index in range(int(base["experiment"]["samples_per_seed"])):
            scene_id = f"{seed}:{sample_index}"
            profile_started = time.perf_counter()
            paths, candidates, metadata, case = _scene(base, base_operators, evaluators, seed, sample_index)
            profile_seconds = time.perf_counter() - profile_started
            _check_scene(metadata, records[scene_id])
            channel = _channel(platform, base, paths)
            truths = _coordinates(paths)
            gains = np.asarray([path.complex_gain for path in paths])
            true_center = channel_from_planar_estimate(
                np.array([float(base["system"]["carrier_frequency_hz"])]),
                platform,
                truths,
                gains,
                base,
            )[0]
            for count in plan["configuration_counts"]:
                count = int(count)
                weights, state_ids = _packed_weights(bank, frozen["anchor_state"], count)
                if state_ids != frozen["state_mapping"][str(count)]:
                    raise RuntimeError("frozen state mapping changed")
                operator = operator_for_schedule(platform, f1_schedule, weights, base)
                dictionary_started = time.perf_counter()
                dictionary = _dictionary(operator, candidates, base)
                dictionary_seconds = time.perf_counter() - dictionary_started
                if count not in cuda_checks:
                    cuda_checks[count] = _validate_accelerated_dictionary(operator, dictionary)
                exact = operator.evaluate(*truths.T)
                margins = _margins(dictionary, operator, paths, base)
                peak_rows.append({
                    "scene_id": scene_id,
                    "configurations": count,
                    "state_ids": ";".join(map(str, state_ids)),
                    "minimum_margin_db": min(margins),
                    **{f"path_{index}_margin_db": value for index, value in enumerate(margins)},
                })
                single_pass = 0
                for index, path in enumerate(paths):
                    single_case = copy.deepcopy(case)
                    single_case["channel"]["num_paths"] = 1
                    single_truth = channel_from_planar_estimate(
                        platform.frequencies_hz,
                        platform,
                        truths[index : index + 1],
                        gains[index : index + 1],
                        base,
                    )
                    nmse = _fit_nmse(
                        operator, dictionary, gains[index] * exact[index],
                        (path,), single_case, platform, base, single_truth,
                    )
                    single_pass += int(nmse <= float(plan["fullband_nmse_target_db"]))
                    noiseless_rows.append({
                        "scene_id": scene_id,
                        "configurations": count,
                        "path_index": index,
                        "kind": "single",
                        "fullband_nmse_db": nmse,
                    })
                triple_white = gains @ exact
                triple_nmse = _fit_nmse(
                    operator, dictionary, triple_white, paths, case, platform, base, channel,
                )
                noiseless_rows.append({
                    "scene_id": scene_id,
                    "configurations": count,
                    "path_index": -1,
                    "kind": "triple",
                    "fullband_nmse_db": triple_nmse,
                })
                for snr_db in base["noise"]["array_reference_snr_db"]:
                    variance = array_input_noise_variance(channel, float(snr_db))
                    rf_variance = variance * float(base["noise"]["rf_noise_variance_fraction_of_element_noise"])
                    noise_rng = np.random.default_rng(
                        np.random.SeedSequence([seed, sample_index, 0, 2_026_090_921])
                    )
                    observation = observe_multistrip_schedule(
                        weights,
                        channel,
                        pilot_energy_per_re=1.0,
                        element_noise_variance=variance,
                        rf_noise_variance=rf_variance,
                        rng=noise_rng,
                    )
                    whitened = operator.whiten(observation.observed)
                    order_rng = np.random.default_rng(
                        np.random.SeedSequence([int(base["experiment"]["randomization_seed"]), seed, sample_index, count, int(snr_db)])
                    )
                    for method in [METHODS[index] for index in order_rng.permutation(2)]:
                        started = time.perf_counter()
                        fit = estimate_planar_baseline(
                            method, dictionary, operator, whitened, paths, case
                        )
                        estimator_seconds = time.perf_counter() - started
                        matched = match_planar_paths(fit, paths, case)
                        estimated = channel_from_planar_estimate(
                            platform.frequencies_hz,
                            platform,
                            matched["coordinates"],
                            matched["gains"],
                            base,
                        )
                        domains = channel_domain_metrics(channel, estimated, np.arange(128), legacy_indices)
                        estimated_center = channel_from_planar_estimate(
                            np.array([float(base["system"]["carrier_frequency_hz"])]),
                            platform,
                            matched["coordinates"],
                            matched["gains"],
                            base,
                        )[0]
                        beam_started = time.perf_counter()
                        guard = (count - 1) * float(base["acquisition"]["switch_guard_symbol_equivalents"])
                        link = feasible_planar_data_metrics(
                            true_center,
                            estimated_center,
                            codebook,
                            element_noise_variance=variance,
                            rf_noise_variance=rf_variance,
                            coherence_symbols=float(data_config["coherence_symbols"]),
                            training_symbols=float(count),
                            switch_guard_symbol_equivalents=guard,
                        )
                        beam_seconds = time.perf_counter() - beam_started
                        errors = np.column_stack((
                            np.asarray(matched["range_error_m"]),
                            np.asarray(matched["azimuth_error_deg"]),
                            np.asarray(matched["elevation_error_deg"]),
                        ))
                        strict = np.all(
                            (np.abs(errors[:, 0]) <= 0.015)
                            & (np.abs(errors[:, 1]) <= 1.0)
                            & (np.abs(errors[:, 2]) <= 1.5)
                        )
                        rows.append({
                            "scene_id": scene_id,
                            "replicate_seed": seed,
                            "sample_index": sample_index,
                            "configurations": count,
                            "state_ids": ";".join(map(str, state_ids)),
                            "pilot_resource_elements": 128,
                            "total_pilot_energy": 128.0,
                            "pilot_frequencies_unique": 128,
                            "pilots_per_symbol": 128 // count,
                            "training_symbols": count,
                            "num_switches": count - 1,
                            "switch_guard_symbol_equivalents": guard,
                            "array_reference_snr_db": snr_db,
                            "effective_output_snr_db": effective_tensor_output_snr_db(observation.clean, observation.output_noise_variance),
                            "estimator": method,
                            **domains,
                            "strict_three_path_success": int(strict),
                            "range_rmse_m": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
                            "azimuth_rmse_deg": float(np.sqrt(np.mean(errors[:, 1] ** 2))),
                            "elevation_rmse_deg": float(np.sqrt(np.mean(errors[:, 2] ** 2))),
                            "minimum_single_margin_db": min(margins),
                            "noiseless_single_pass_count": single_pass,
                            "noiseless_triple_nmse_db": triple_nmse,
                            **link,
                            "candidate_count": int(dictionary.num_candidates),
                            "profile_seconds": profile_seconds,
                            "dictionary_seconds": dictionary_seconds,
                            "estimator_runtime_seconds": estimator_seconds,
                            "beam_selection_seconds": beam_seconds,
                            "online_compute_seconds": profile_seconds + dictionary_seconds + estimator_seconds + beam_seconds,
                            "nominal_acquisition_seconds": (count + guard) * symbol_seconds,
                        })
            if sample_index % 5 == 4:
                print(f"evaluation seed {seed}: {sample_index + 1}/20", flush=True)
    expected = 100 * len(plan["configuration_counts"]) * 2 * len(METHODS)
    if len(rows) != expected or len(peak_rows) != 500 or len(noiseless_rows) != 2000:
        raise RuntimeError("P-016 output row count mismatch")
    archive.write_metrics(rows)
    write_raw_metrics(archive.directory / "peak_metrics.csv", peak_rows)
    write_raw_metrics(archive.directory / "noiseless_metrics.csv", noiseless_rows)
    archive.write_json("cuda_checks.json", cuda_checks)
    archive.write_json("source_f1_keys.json", [
        {"scene_id": row["scene_id"], "snr_db": row["array_reference_snr_db"], "estimator": row["estimator"]}
        for row in source_f1
    ])
    return rows, peak_rows, noiseless_rows, source_f1


def _summarize(plan, rows, peaks, noiseless, source_f1, frozen, archive):
    summary = []
    target = float(plan["fullband_nmse_target_db"])
    margin_target = float(plan["peak_margin_target_db"])
    for count in plan["configuration_counts"]:
        count = int(count)
        group_peaks = [row for row in peaks if row["configurations"] == count]
        group_noiseless = [row for row in noiseless if row["configurations"] == count]
        singles = [row for row in group_noiseless if row["kind"] == "single"]
        triples = [row for row in group_noiseless if row["kind"] == "triple"]
        for snr in (55.0, 65.0):
            selected = [row for row in rows if row["configurations"] == count and row["array_reference_snr_db"] == snr and row["estimator"] == "yang_ogols_3d"]
            if len(selected) != 100:
                raise RuntimeError("P-016 Yang scene pairing is incomplete")
            linear = [row["fullband_128_nmse_linear"] for row in selected]
            summary.append({
                "configurations": count,
                "snr_db": snr,
                "margin_target_scenes": sum(row["minimum_margin_db"] > margin_target for row in group_peaks),
                "margin_median_db": float(np.median([row["minimum_margin_db"] for row in group_peaks])),
                "noiseless_single_pass": sum(row["fullband_nmse_db"] <= target for row in singles),
                "noiseless_triple_pass": sum(row["fullband_nmse_db"] <= target for row in triples),
                "noisy_fullband_pass": sum(row["fullband_128_nmse_db"] <= target for row in selected),
                "fullband_linear_mean_db": float(10.0 * np.log10(np.mean(linear))),
                "strict_path_success": sum(row["strict_three_path_success"] for row in selected),
                "beam_loss_p90_db": float(np.quantile([row["feasible_beam_loss_db"] for row in selected], 0.9)),
                "net_rate_mean_bps_hz": float(np.mean([row["net_rate_bps_hz"] for row in selected])),
                "online_compute_median_seconds": float(np.median([row["online_compute_seconds"] for row in selected])),
                "training_symbols": count,
                "switches": count - 1,
            })
    baseline = []
    for name in (F1, "tensor_structured_J8_K16_RE128"):
        source = ROOT / plan["source_run"]
        with (source / "metrics_raw.csv").open(encoding="utf-8", newline="") as handle:
            source_rows = [row for row in csv.DictReader(handle) if row["schedule"] == name and row["estimator"] == "yang_ogols_3d"]
        for snr in (55.0, 65.0):
            subset = [row for row in source_rows if float(row["array_reference_snr_db"]) == snr]
            baseline.append({
                "schedule": name,
                "snr_db": snr,
                "noisy_fullband_pass": sum(float(row["fullband_128_nmse_db"]) <= target for row in subset),
                "fullband_linear_mean_db": float(10.0 * np.log10(np.mean([float(row["fullband_128_nmse_linear"]) for row in subset]))),
                "net_rate_mean_bps_hz": float(np.mean([float(row["net_rate_bps_hz"]) for row in subset])),
            })
    archive.write_json("analysis_compact.json", {"state_design": frozen, "new_curve": summary, "archived_context": baseline})
    lines = [
        "# P-016 配置数条件诊断",
        "",
        "本报告只使用 D-024 的逐径粗中心条件；原 F1/F2 与 D-026 均从归档读取。新曲线的每组均为 128 个不同频点、128 RE、总导频能量 128。",
        "",
        f"设计集冻结起点状态：C{frozen['anchor_state']}；设计场景 {frozen['design_scenes']} 组。",
        "",
        "| J | SNR/dB | 场景最小峰裕量>0.2 dB | 无噪声单径≤−10 dB | 无噪声三径≤−10 dB | 带噪三径≤−10 dB | 全带线性均值转 dB | 严格三径路径成功 | 波束损失 P90/dB | 平均净谱效率 | 在线时间中位/s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"|{row['configurations']}|{row['snr_db']:.0f}|{row['margin_target_scenes']}/100|{row['noiseless_single_pass']}/300|{row['noiseless_triple_pass']}/100|{row['noisy_fullband_pass']}/100|{row['fullband_linear_mean_db']:.3f}|{row['strict_path_success']}/100|{row['beam_loss_p90_db']:.3f}|{row['net_rate_mean_bps_hz']:.3f}|{row['online_compute_median_seconds']:.3f}|"
        )
    lines.extend(["", "历史参照："])
    for row in baseline:
        lines.append(f"- {row['schedule']} / {row['snr_db']:.0f} dB：全带达标 {row['noisy_fullband_pass']}/100，线性均值转 dB {row['fullband_linear_mean_db']:.3f}，平均净谱效率 {row['net_rate_mean_bps_hz']:.3f}。")
    lines.extend(["", "继续线：每组至少 90/100 个场景最小峰裕量>0.2 dB，且 55/65 dB Yang 各至少 90/100 个场景全带 NMSE≤−10 dB。实际通信收益还须核对波束长尾和净谱效率。", ""])
    (archive.directory / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def _check_only(plan, base, platform, bank, base_operators, evaluators):
    seed = int(plan["design_seeds"][0])
    paths, candidates, _, case = _scene(base, base_operators, evaluators, seed, 0)
    channel = _channel(platform, base, paths)
    f1_schedule = next(item for item in platform.schedules if item.name == F1)
    for count in (1, 2, 16):
        weights, states = _packed_weights(bank, 0, count)
        for subcarrier in range(128):
            np.testing.assert_array_equal(weights[0, subcarrier], bank[states[subcarrier % count], subcarrier])
        operator = operator_for_schedule(platform, f1_schedule, weights, base)
        dictionary = _dictionary(operator, candidates, base)
        _validate_accelerated_dictionary(operator, dictionary)
        first = _coordinates(paths)[0:1]
        single_channel = _channel(platform, base, (paths[0],))
        clean = observe_multistrip_schedule(
            weights, single_channel, pilot_energy_per_re=1.0,
            element_noise_variance=0.0,
        ).clean.reshape(-1)
        np.testing.assert_allclose(operator.evaluate_raw(*first.T)[0] * paths[0].complex_gain, clean, rtol=1.0e-4, atol=1.0e-10)
        fit = estimate_planar_baseline("yang_ogols_3d", dictionary, operator, operator.whiten(observe_multistrip_schedule(weights, channel, pilot_energy_per_re=1.0, element_noise_variance=0.0).clean), paths, case)
        if not np.isfinite(fit["residual_energy"]):
            raise RuntimeError("non-finite check fit")
        print(f"check J={count}: {dictionary.num_candidates} candidates, residual={fit['residual_energy']:.3e}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="stage2a_p016_config_curve_20260926_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    plan, base, plan_path, source_path = _load()
    platform, old_weights, base_operators = build_stage2a_operators(base, ROOT)
    evaluators = {
        name: PlanarTemplateBatchEvaluator(
            operator,
            compute_backend=str(base["search"]["dictionary_compute_backend"]),
            compute_device=str(base["search"]["dictionary_compute_device"]),
        )
        for name, operator in base_operators.items()
    }
    bank = _state_bank(platform, base, int(plan["state_bank_size"]))
    if args.check_only:
        _check_only(plan, base, platform, bank, base_operators, evaluators)
        return
    if Path(args.run_id).name != args.run_id:
        raise ValueError("run ID must be a single directory name")
    config = copy.deepcopy(base)
    config["p016"] = plan
    archive = create_run_archive(
        "stage2a_p016_config_curve", args.run_id,
        config_path=plan_path, config=config,
        seeds={"design_seeds": plan["design_seeds"], "evaluation_seeds": base["experiment"]["replicate_seeds"], "paired_noise_seed_suffix": 2_026_090_921},
        script_path=__file__,
        environment={"experiment": "P-016 conditional fixed-RE configuration curve", "dictionary_backend": base["search"]["dictionary_compute_backend"]},
        checkpoint_note="No learning checkpoint. State selection is frozen before archived evaluation is read.",
        extra_manifest_files=(source_path,),
    )
    started = time.perf_counter()
    try:
        frozen = _design(plan, base, platform, bank, base_operators, evaluators, old_weights[F1], archive)
        archive.write_json("state_design_frozen_sha256.json", {"sha256": hash_file(archive.directory / "state_design_frozen.json")})
        rows, peaks, noiseless, source_f1 = _evaluate(plan, base, platform, bank, base_operators, evaluators, frozen, archive)
        _summarize(plan, rows, peaks, noiseless, source_f1, frozen, archive)
        archive.write_status("complete", raw_metric_rows=len(rows), independent_scenes=100, runtime_seconds=time.perf_counter() - started)
    except Exception as exc:
        archive.write_status("failed", error=f"{type(exc).__name__}: {exc}", runtime_seconds=time.perf_counter() - started)
        raise
    print(f"complete: {archive.directory}", flush=True)


if __name__ == "__main__":
    main()
