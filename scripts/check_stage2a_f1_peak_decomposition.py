#!/usr/bin/env python3
"""F1 matched-filter spectrum and single-/three-path decomposition."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
from _run_support import PROJECT_ROOT as ROOT, hash_file, read_toml, write_json

import numpy as np

from check_stage2a_ongrid_noiseless import traced_estimate
from thz_dma.estimators.planar_sparse import (
    PlanarTemplateBatchEvaluator,
    build_planar_template_dictionary_from_candidates,
)
from thz_dma.stage2a import (
    _channel_from_estimate,
    _match_paths,
    build_stage2a_operators,
    channel_domain_metrics,
    prepare_adaptive_stage2a_scene,
)


F1 = "frequency_single_shot_J1_K128"
METHODS = ("grid_omp_3d", "yang_ogols_3d")
FULLBAND_PASS_DB = -10.0


def _coordinates(paths) -> np.ndarray:
    return np.array(
        [[path.range_m, path.azimuth_rad, path.elevation_rad] for path in paths],
        dtype=float,
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _linear_mean_db(values) -> float:
    return float(
        10.0
        * np.log10(max(float(np.mean(values)), np.finfo(float).tiny))
    )


def _quantiles(values) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "p10": float(np.quantile(array, 0.1)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(np.max(array)),
    }


def _neighborhood_mask(dictionary, truth: np.ndarray, config: dict) -> np.ndarray:
    estimator = config["estimator"]
    return (
        (np.abs(np.asarray(dictionary.ranges_m) - truth[0]) <= float(estimator["exclusion_range_m"]))
        & (
            np.abs(np.asarray(dictionary.azimuths_rad) - truth[1])
            <= np.deg2rad(float(estimator["exclusion_azimuth_deg"]))
        )
        & (
            np.abs(np.asarray(dictionary.elevations_rad) - truth[2])
            <= np.deg2rad(float(estimator["exclusion_elevation_deg"]))
        )
    )


def _spectrum(dictionary, observation: np.ndarray) -> np.ndarray:
    templates = np.asarray(dictionary.templates)
    atom_energy = np.sum(np.abs(templates) ** 2, axis=1, dtype=float)
    observation_energy = float(np.vdot(observation, observation).real)
    if observation_energy <= 0.0 or np.any(atom_energy <= 0.0):
        raise ValueError("spectrum requires positive observation and atom energy")
    return (
        np.abs(templates @ np.conjugate(observation)) ** 2
        / atom_energy
        / observation_energy
    )


def _peak_summary(
    dictionary, score: np.ndarray, truths: np.ndarray, config: dict
) -> dict:
    masks = [_neighborhood_mask(dictionary, truth, config) for truth in truths]
    true_mask = np.logical_or.reduce(masks)
    if not np.any(true_mask) or np.all(true_mask):
        raise RuntimeError("truth/false peak partition is empty")
    true_indices = np.flatnonzero(true_mask)
    false_indices = np.flatnonzero(~true_mask)
    true_index = int(true_indices[np.argmax(score[true_indices])])
    false_index = int(false_indices[np.argmax(score[false_indices])])
    winner = int(np.argmax(score))
    true_score = float(score[true_index])
    false_score = float(score[false_index])
    margin_db = float(10.0 * np.log10(true_score / false_score))
    winner_point = np.array(
        [
            dictionary.ranges_m[winner],
            dictionary.azimuths_rad[winner],
            dictionary.elevations_rad[winner],
        ],
        dtype=float,
    )
    normalized = np.column_stack(
        (
            np.abs(truths[:, 0] - winner_point[0]) / 0.015,
            np.abs(truths[:, 1] - winner_point[1]) / np.deg2rad(1.0),
            np.abs(truths[:, 2] - winner_point[2]) / np.deg2rad(1.5),
        )
    )
    nearest_truth = int(np.argmin(np.sum(normalized**2, axis=1)))
    error = winner_point - truths[nearest_truth]
    top = np.argsort(score)[-5:][::-1]
    return {
        "winner_index": winner,
        "winner_in_truth_neighborhood": bool(true_mask[winner]),
        "winner_nearest_truth_path": nearest_truth,
        "winner_range_error_m": float(error[0]),
        "winner_azimuth_error_deg": float(np.rad2deg(error[1])),
        "winner_elevation_error_deg": float(np.rad2deg(error[2])),
        "best_truth_index": true_index,
        "best_truth_rho2": true_score,
        "best_false_index": false_index,
        "best_false_rho2": false_score,
        "truth_over_false_margin_db": margin_db,
        "top5": [
            {
                "candidate_index": int(index),
                "rho2": float(score[index]),
                "range_m": float(dictionary.ranges_m[index]),
                "azimuth_deg": float(np.rad2deg(dictionary.azimuths_rad[index])),
                "elevation_deg": float(np.rad2deg(dictionary.elevations_rad[index])),
                "in_truth_neighborhood": bool(true_mask[index]),
            }
            for index in top
        ],
    }


def _fit_case(
    method: str,
    dictionary,
    operator,
    observation: np.ndarray,
    paths,
    config: dict,
    platform,
    truth_channel: np.ndarray,
) -> tuple[dict, dict]:
    fit, trace = traced_estimate(method, dictionary, operator, observation, paths, config)
    matched = _match_paths(fit, paths, config)
    estimated_channel = _channel_from_estimate(
        platform.frequencies_hz,
        platform,
        matched["coordinates"],
        matched["gains"],
        config,
    )
    domains = channel_domain_metrics(
        truth_channel,
        estimated_channel,
        np.arange(128),
        np.rint(np.linspace(0, 127, 16)).astype(int),
    )
    return {
        "fullband_nmse_linear": float(domains["fullband_128_nmse_linear"]),
        "fullband_nmse_db": float(domains["fullband_128_nmse_db"]),
        "fullband_pass": int(domains["fullband_128_nmse_db"] <= FULLBAND_PASS_DB),
        "max_range_error_m": float(np.max(np.abs(matched["range_error_m"]))),
        "max_azimuth_error_deg": float(np.max(np.abs(matched["azimuth_error_deg"]))),
        "max_elevation_error_deg": float(np.max(np.abs(matched["elevation_error_deg"]))),
        "residual_energy_fraction": float(
            fit["residual_energy"] / np.vdot(observation, observation).real
        ),
        "selected_candidate_indices": ";".join(
            str(int(value)) for value in np.asarray(fit["candidate_index"])
        ),
    }, {
        "selected_candidate_indices": [
            int(value) for value in np.asarray(fit["candidate_index"])
        ],
        "stages": trace,
    }


def _azimuth_marginal(dictionary, score: np.ndarray, scene_id: str, kind: str) -> list[dict]:
    azimuths, inverse = np.unique(
        np.asarray(dictionary.azimuths_rad), return_inverse=True
    )
    marginal = np.full(azimuths.size, -np.inf, dtype=float)
    np.maximum.at(marginal, inverse, score)
    peak = float(np.max(marginal))
    return [
        {
            "scene_id": scene_id,
            "kind": kind,
            "azimuth_deg": float(np.rad2deg(azimuth)),
            "maximum_rho2": float(value),
            "relative_db": float(10.0 * np.log10(max(value / peak, np.finfo(float).tiny))),
        }
        for azimuth, value in zip(azimuths, marginal)
    ]


def _source_rows(run_dir: Path) -> tuple[dict, dict]:
    with (run_dir / "metrics_raw.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if row["schedule"] == F1
        and float(row["array_reference_snr_db"]) == 65.0
        and row["estimator"] in METHODS
    ]
    if len(selected) != 200:
        raise RuntimeError("source run does not contain 100 paired F1 65 dB scenes")
    lookup = {
        (row["scene_id"], row["estimator"]): {
            "fullband_nmse_db": float(row["fullband_128_nmse_db"]),
            "fullband_pass": int(float(row["fullband_128_nmse_db"]) <= FULLBAND_PASS_DB),
        }
        for row in selected
    }
    yang = [row for row in selected if row["estimator"] == "yang_ogols_3d"]
    worst = max(yang, key=lambda row: float(row["fullband_128_nmse_db"]))
    return lookup, {
        "worst_yang_scene_id": worst["scene_id"],
        "worst_yang_fullband_nmse_db": float(worst["fullband_128_nmse_db"]),
    }


def _plot(run_dir: Path, single_rows: list[dict], triple_rows: list[dict], spectrum_rows: list[dict], worst_scene: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = run_dir / "figures"
    figure_dir.mkdir()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for failed, label, color in ((0, "source 65 dB pass", "#1677ff"), (1, "source 65 dB fail", "#d4380d")):
        values = np.sort(
            [
                row["truth_over_false_margin_db"]
                for row in single_rows
                if row["estimator"] == METHODS[0]
                if row["source_yang_65db_fail"] == failed
            ]
        )
        ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=f"{label} (n={len(values)})", color=color)
    ax.set_xlabel("single-path truth/false peak margin (dB)")
    ax.set_ylabel("empirical CDF")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "single_path_peak_margin_ecdf.png", dpi=180)
    plt.close(fig)

    labels, single_rates, triple_rates = [], [], []
    for method in METHODS:
        labels.append("OMP" if method == "grid_omp_3d" else "Yang")
        block_single = [row for row in single_rows if row["estimator"] == method]
        block_triple = [row for row in triple_rows if row["estimator"] == method]
        single_rates.append(100.0 * np.mean([row["fullband_pass"] for row in block_single]))
        triple_rates.append(100.0 * np.mean([row["fullband_pass"] for row in block_triple]))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.bar(x - 0.18, single_rates, 0.36, label="dependent single paths")
    ax.bar(x + 0.18, triple_rates, 0.36, label="original three paths")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("noiseless fullband pass rate (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "single_vs_triple_pass_rate.png", dpi=180)
    plt.close(fig)

    rows = [row for row in spectrum_rows if row["scene_id"] == worst_scene]
    kinds = [f"single:{index}" for index in range(3)] + ["triple"]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2), sharex=True, sharey=True)
    truths = [
        row
        for row in single_rows
        if row["scene_id"] == worst_scene and row["estimator"] == METHODS[0]
    ]
    truth_azimuths = [row["true_azimuth_deg"] for row in sorted(truths, key=lambda row: row["path_index"])]
    for ax, kind in zip(axes.ravel(), kinds):
        block = [row for row in rows if row["kind"] == kind]
        ax.plot([row["azimuth_deg"] for row in block], [row["relative_db"] for row in block], lw=1.1)
        for truth in truth_azimuths if kind == "triple" else [truth_azimuths[int(kind.split(":")[1])]]:
            ax.axvline(truth, color="#d4380d", ls="--", lw=0.8)
        ax.set_title(kind)
        ax.grid(alpha=0.2)
        ax.set_ylim(-30, 1)
    fig.supxlabel("azimuth (deg); maximum over local range/elevation candidates")
    fig.supylabel("relative matched-filter score (dB)")
    fig.tight_layout()
    fig.savefig(figure_dir / "worst_scene_azimuth_spectrum.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="stage2a_f1_peak_decomposition_20260915_v1")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError("run ID must be one directory name")
    run_dir = ROOT / "runs" / args.run_id
    run_dir.mkdir(exist_ok=False)
    started = time.perf_counter()
    write_json(run_dir / "status.json", {"status": "running"})
    try:
        config_path = ROOT / "configs/direction1_stage2a_adaptive_300ghz.toml"
        config = read_toml(config_path)
        source_run = ROOT / "runs/stage2a_adaptive_300ghz_20260911_v1"
        source_status = json.loads((source_run / "status.json").read_text(encoding="utf-8"))
        if source_status.get("status") != "complete" or source_status.get("raw_metric_rows") != 1200:
            raise RuntimeError("source adaptive run is incomplete")
        source_config = json.loads((source_run / "config_resolved.json").read_text(encoding="utf-8"))["config"]
        if source_config != config:
            raise RuntimeError("source run and current adaptive config differ")
        protected = [config_path, *sorted(path for path in source_run.rglob("*") if path.is_file())]
        protected_before = {str(path.relative_to(ROOT)): hash_file(path) for path in protected}
        source_lookup, source_summary = _source_rows(source_run)
        design = {
            "diagnostic_only": True,
            "source_run": source_run.name,
            "fixtures": "the same 100 continuous scenes and adaptive candidate grids; each triple also split into its three dependent single paths",
            "schedule": F1,
            "noise": "none; source 65 dB triple labels are joined without rerunning noise",
            "methods": list(METHODS),
            "spectrum": "normalized squared matched-filter score over every scene's existing local 3-D candidate union",
            "truth_neighborhood": "the existing estimator exclusion box: 0.015 m / 1.0 deg / 1.5 deg around each exact path",
            "truth_false_margin_db": "10 log10(best score inside truth-neighborhood union / best score outside it)",
            "fullband_pass": "NMSE <= -10 dB; descriptive diagnostic screen retained from D-024",
            "decision_readout": "count triple failures with any dependent single failure separately from triple failures whose three singles all pass",
            "fixed": "300 GHz/30 GHz, F1 state/pilots, natural continuous truth, coarse centers, local candidates, OMP/Yang settings including two refinements",
            "not_changed": ["F1 state", "pilot locations", "dictionary sampling", "truth", "noise model", "estimator", "100 GHz"],
            "expected_single_rows": 600,
            "expected_triple_rows": 200,
        }
        write_json(run_dir / "design_pre_registered.json", design)
        sources = [
            Path(__file__).resolve(),
            ROOT / "scripts/check_stage2a_ongrid_noiseless.py",
            config_path,
            ROOT / config["atmosphere"]["table_path"],
            *sorted((ROOT / "src").rglob("*.py")),
        ]
        write_json(
            run_dir / "code_manifest.json",
            [{"path": str(path.relative_to(ROOT)), "sha256": hash_file(path)} for path in sources],
        )
        (run_dir / "environment.txt").write_text(
            f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}\n"
            f"python: {sys.version}\nexecutable: {sys.executable}\nnumpy: {np.__version__}\n"
            "device: configured CUDA for dictionary/profile construction; CPU estimators; no training\n",
            encoding="utf-8",
        )

        platform, _, operators = build_stage2a_operators(config, ROOT)
        operator = operators[F1]
        profile_evaluators = {
            name: PlanarTemplateBatchEvaluator(
                value,
                compute_backend=str(config["search"]["dictionary_compute_backend"]),
                compute_device=str(config["search"]["dictionary_compute_device"]),
            )
            for name, value in operators.items()
        }
        single_rows: list[dict] = []
        triple_rows: list[dict] = []
        peak_rows: list[dict] = []
        trace_rows: list[dict] = []
        spectrum_rows: list[dict] = []
        worst_scene = source_summary["worst_yang_scene_id"]
        for scene_number, (seed, sample_index) in enumerate(
            (
                (int(seed), sample_index)
                for seed in config["experiment"]["replicate_seeds"]
                for sample_index in range(int(config["experiment"]["samples_per_seed"]))
            ),
            start=1,
        ):
            scene_id = f"{seed}:{sample_index}"
            paths, candidates, metadata = prepare_adaptive_stage2a_scene(
                config,
                operators,
                center_rng=np.random.default_rng(np.random.SeedSequence([seed, sample_index, 2_026_091_101])),
                truth_rng=np.random.default_rng(np.random.SeedSequence([seed, sample_index, 2_026_091_102])),
                profile_evaluators=profile_evaluators,
            )
            steps = np.asarray(metadata["grid_steps_m_rad_rad"], dtype=float)
            case_config = copy.deepcopy(config)
            case_config["search"]["range_step_m"] = float(steps[0])
            case_config["search"]["azimuth_step_deg"] = float(np.rad2deg(steps[1]))
            case_config["search"]["elevation_step_deg"] = float(np.rad2deg(steps[2]))
            dictionary = build_planar_template_dictionary_from_candidates(
                operator,
                *candidates,
                chunk_size=int(config["search"]["dictionary_chunk_size"]),
                storage_dtype=str(config["search"]["dictionary_storage_dtype"]),
                compute_backend=str(config["search"]["dictionary_compute_backend"]),
                compute_device=str(config["search"]["dictionary_compute_device"]),
            )
            truths = _coordinates(paths)
            gains = np.asarray([path.complex_gain for path in paths], dtype=complex)
            exact_templates = operator.evaluate(*truths.T)
            single_method_results: dict[tuple[int, str], dict] = {}
            for path_index, path in enumerate(paths):
                observation = gains[path_index] * exact_templates[path_index]
                score = _spectrum(dictionary, observation)
                peak = _peak_summary(
                    dictionary, score, truths[path_index : path_index + 1], config
                )
                peak_rows.append({"scene_id": scene_id, "kind": f"single:{path_index}", **peak})
                if scene_id == worst_scene:
                    spectrum_rows.extend(_azimuth_marginal(dictionary, score, scene_id, f"single:{path_index}"))
                single_paths = (path,)
                single_config = copy.deepcopy(case_config)
                single_config["channel"]["num_paths"] = 1
                truth_channel = _channel_from_estimate(
                    platform.frequencies_hz,
                    platform,
                    truths[path_index : path_index + 1],
                    gains[path_index : path_index + 1],
                    config,
                )
                for method in METHODS:
                    fit, trace = _fit_case(
                        method,
                        dictionary,
                        operator,
                        observation,
                        single_paths,
                        single_config,
                        platform,
                        truth_channel,
                    )
                    row = {
                        "scene_id": scene_id,
                        "replicate_seed": seed,
                        "sample_index": sample_index,
                        "path_index": path_index,
                        "estimator": method,
                        "true_range_m": float(truths[path_index, 0]),
                        "true_azimuth_deg": float(np.rad2deg(truths[path_index, 1])),
                        "true_elevation_deg": float(np.rad2deg(truths[path_index, 2])),
                        "candidate_count": int(dictionary.num_candidates),
                        "source_yang_65db_fail": 1 - source_lookup[(scene_id, "yang_ogols_3d")]["fullband_pass"],
                        **{key: value for key, value in peak.items() if key != "top5"},
                        **fit,
                    }
                    single_rows.append(row)
                    single_method_results[(path_index, method)] = row
                    trace_rows.append({"scene_id": scene_id, "kind": f"single:{path_index}", "estimator": method, **trace})

            triple_observation = gains @ exact_templates
            triple_score = _spectrum(dictionary, triple_observation)
            triple_peak = _peak_summary(dictionary, triple_score, truths, config)
            peak_rows.append({"scene_id": scene_id, "kind": "triple", **triple_peak})
            if scene_id == worst_scene:
                spectrum_rows.extend(_azimuth_marginal(dictionary, triple_score, scene_id, "triple"))
            truth_channel = _channel_from_estimate(
                platform.frequencies_hz, platform, truths, gains, config
            )
            for method in METHODS:
                fit, trace = _fit_case(
                    method,
                    dictionary,
                    operator,
                    triple_observation,
                    paths,
                    case_config,
                    platform,
                    truth_channel,
                )
                singles_pass = all(single_method_results[(index, method)]["fullband_pass"] for index in range(3))
                row = {
                    "scene_id": scene_id,
                    "replicate_seed": seed,
                    "sample_index": sample_index,
                    "estimator": method,
                    "candidate_count": int(dictionary.num_candidates),
                    "all_three_dependent_singles_pass": int(singles_pass),
                    "source_65db_fullband_nmse_db": source_lookup[(scene_id, method)]["fullband_nmse_db"],
                    "source_65db_pass": source_lookup[(scene_id, method)]["fullband_pass"],
                    **{key: value for key, value in triple_peak.items() if key != "top5"},
                    **fit,
                }
                triple_rows.append(row)
                trace_rows.append({"scene_id": scene_id, "kind": "triple", "estimator": method, **trace})
            if scene_number % 5 == 0 or scene_number == 100:
                print(f"completed {scene_number}/100 scenes", flush=True)

        if len(single_rows) != 600 or len(triple_rows) != 200:
            raise RuntimeError("diagnostic row count mismatch")
        _write_csv(run_dir / "single_path_metrics.csv", single_rows)
        _write_csv(run_dir / "triple_path_metrics.csv", triple_rows)
        _write_csv(run_dir / "representative_azimuth_spectrum.csv", spectrum_rows)
        write_json(run_dir / "peak_records.json", peak_rows)
        write_json(run_dir / "selection_traces.json", trace_rows)

        aggregate = {}
        for method in METHODS:
            singles = [row for row in single_rows if row["estimator"] == method]
            triples = [row for row in triple_rows if row["estimator"] == method]
            aggregate[method] = {
                "dependent_single_paths": len(singles),
                "single_fullband_pass": int(sum(row["fullband_pass"] for row in singles)),
                "single_fullband_linear_mean_db": _linear_mean_db([row["fullband_nmse_linear"] for row in singles]),
                "scenes_all_three_singles_pass": int(sum(row["all_three_dependent_singles_pass"] for row in triples)),
                "triple_fullband_pass": int(sum(row["fullband_pass"] for row in triples)),
                "triple_fullband_linear_mean_db": _linear_mean_db([row["fullband_nmse_linear"] for row in triples]),
                "triple_fail_with_any_single_fail": int(sum((not row["fullband_pass"]) and (not row["all_three_dependent_singles_pass"]) for row in triples)),
                "triple_fail_while_all_singles_pass": int(sum((not row["fullband_pass"]) and row["all_three_dependent_singles_pass"] for row in triples)),
                "source_65db_fail": int(sum(not row["source_65db_pass"] for row in triples)),
                "source_65db_fail_and_noiseless_triple_fail": int(sum((not row["source_65db_pass"]) and (not row["fullband_pass"]) for row in triples)),
            }
        unique_single_peaks = [row for row in peak_rows if row["kind"].startswith("single:")]
        triple_peaks = [row for row in peak_rows if row["kind"] == "triple"]
        yang_failed_scene_ids = {
            row["scene_id"]
            for row in triple_rows
            if row["estimator"] == "yang_ogols_3d" and not row["source_65db_pass"]
        }
        summary = {
            "diagnostic_only": True,
            "source": source_summary,
            "raw_rows": {"single": len(single_rows), "triple": len(triple_rows)},
            "aggregate": aggregate,
            "matched_filter": {
                "single_winner_in_truth_neighborhood": int(sum(row["winner_in_truth_neighborhood"] for row in unique_single_peaks)),
                "single_paths": len(unique_single_peaks),
                "single_truth_over_false_margin_db": _quantiles([row["truth_over_false_margin_db"] for row in unique_single_peaks]),
                "single_margin_by_source_yang_65db": {
                    "failed_scenes": _quantiles([row["truth_over_false_margin_db"] for row in unique_single_peaks if row["scene_id"] in yang_failed_scene_ids]),
                    "passed_scenes": _quantiles([row["truth_over_false_margin_db"] for row in unique_single_peaks if row["scene_id"] not in yang_failed_scene_ids]),
                },
                "triple_first_winner_in_truth_neighborhood": int(sum(row["winner_in_truth_neighborhood"] for row in triple_peaks)),
                "triple_scenes": len(triple_peaks),
                "triple_truth_over_false_margin_db": _quantiles([row["truth_over_false_margin_db"] for row in triple_peaks]),
            },
            "protected_source_run_and_config_unchanged": protected_before
            == {str(path.relative_to(ROOT)): hash_file(path) for path in protected},
            "wall_seconds": float(time.perf_counter() - started),
        }
        if not summary["protected_source_run_and_config_unchanged"]:
            raise RuntimeError("source run or config changed during diagnostic")
        write_json(run_dir / "analysis_compact.json", summary)
        _plot(run_dir, single_rows, triple_rows, spectrum_rows, worst_scene)
        lines = [
            "# Stage 2-A F1 相关峰与单径/三径分解",
            "",
            "复用 D-024 的 100 个连续真值、粗中心和局部候选，仅分析 F1。三径按原复增益拆成三个相关单径；观测无噪声，未改变状态、导频、字典、估计器或两次细化。源 65 dB 三径标签只用于分组。",
            "",
            "| 方法 | 单径全带≤−10 dB | 三个单径均通过的场景 | 三径全带≤−10 dB | 三径失败且任一单径失败 | 三径失败但三个单径均通过 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for method in METHODS:
            item = aggregate[method]
            lines.append(
                f"|{method}|{item['single_fullband_pass']}/300|{item['scenes_all_three_singles_pass']}/100|{item['triple_fullband_pass']}/100|{item['triple_fail_with_any_single_fail']}|{item['triple_fail_while_all_singles_pass']}|"
            )
        matched = summary["matched_filter"]
        lines += [
            "",
            f"单径粗相关最高点位于真值邻域：{matched['single_winner_in_truth_neighborhood']}/300；三径首次最高点位于任一真值邻域：{matched['triple_first_winner_in_truth_neighborhood']}/100。真值邻域沿用估计器排除盒 1.5 cm/1°/1.5°。",
            "",
            f"单径真值峰相对最强邻域外错误峰的裕量中位数为 {matched['single_truth_over_false_margin_db']['median']:.3f} dB；源 65 dB Yang 失败/通过场景中的对应中位数为 {matched['single_margin_by_source_yang_65db']['failed_scenes']['median']:.3f}/{matched['single_margin_by_source_yang_65db']['passed_scenes']['median']:.3f} dB。",
            "",
            "结论只依据上述固定分解与逐场景记录；单径来自原三径，不能按 300 个独立场景计算置信区间。",
            "",
        ]
        (run_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
        write_json(
            run_dir / "status.json",
            {"status": "complete", "single_rows": 600, "triple_rows": 200, "wall_seconds": summary["wall_seconds"]},
        )
    except Exception as error:
        write_json(run_dir / "status.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
