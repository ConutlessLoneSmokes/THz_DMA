#!/usr/bin/env python3
"""Fixed-dictionary, noiseless on-grid checks, with passive selection traces.

Five existing seeded scenes are snapped to the EXISTING grid for test fixtures.
Each of their 15 paths is also tested alone. These are dependent unit fixtures,
not 20 independent research scenes. No production config/estimator is edited.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform as operating_system
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import numpy as np

import thz_dma.estimators.planar_sparse as sparse
from thz_dma.channels.planar import PlanarPath
from thz_dma.observations.multistrip import observe_multistrip_schedule
from thz_dma.stage2 import sample_stage2_paths
from thz_dma.stage2a import (
    build_stage2a_estimators, _channel_from_estimate, _estimate, _match_paths,
    channel_domain_metrics,
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coords(paths):
    return np.array([[p.range_m, p.azimuth_rad, p.elevation_rad] for p in paths])


def traced_estimate(method, dictionary, operator, received, paths, config):
    """Wrap only return values; never change a selection, update or stopping rule."""
    trace = []
    grid_fit = sparse._fit_dictionary_support
    refine = sparse._continuous_refinement
    select = sparse._select_ols_candidate
    selected_indices = []

    def traced_grid(*args, **kwargs):
        result = grid_fit(*args, **kwargs)
        trace.append({"stage": len(args[2]), "selected_indices": list(args[2]),
                      "residual_energy": result[2]})
        return result

    def traced_select(*args, **kwargs):
        index = select(*args, **kwargs)
        selected_indices.append(int(index))
        return index

    def traced_refine(*args, **kwargs):
        initial = np.asarray(args[2]).copy()
        result = refine(*args, **kwargs)
        trace.append({"stage": len(initial), "selected_indices": selected_indices.copy(),
                      "initial_coordinates_rad": initial.tolist(),
                      "refined_coordinates_rad": result["coordinates"].tolist(),
                      "initial_residual_energy": result["initial_residual_energy"],
                      "residual_energy": result["residual_energy"],
                      "iterations": result["iterations"], "converged": result["converged"]})
        return result

    with patch.object(sparse, "_fit_dictionary_support", traced_grid), \
         patch.object(sparse, "_select_ols_candidate", traced_select), \
         patch.object(sparse, "_continuous_refinement", traced_refine):
        result = _estimate(method, dictionary, operator, received, paths, config)
    return result, trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="stage2a_ongrid_noiseless_20260910_v1")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError("run ID must be one directory name")
    directory = ROOT / "runs" / args.run_id
    directory.mkdir(exist_ok=False)
    started = time.perf_counter()
    write_json(directory / "status.json", {"status": "running"})
    try:
        config_path = ROOT / "configs/direction1_stage2a_bounded_diagnostic.toml"
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        original_config = copy.deepcopy(config)
        immutable = [config_path, ROOT / "configs/direction1_stage2a_bounded_65db_diagnostic.toml",
                     ROOT / "src/thz_dma/estimators/planar_sparse.py",
                     ROOT / "src/thz_dma/surfaces/multistrip.py"]
        for run_id in ("stage2a_bounded_20260910_v1", "stage2a_bounded_65db_20260910_v1"):
            immutable.extend(p for p in (ROOT / "runs" / run_id).rglob("*") if p.is_file())
        before = {str(p.relative_to(ROOT)): sha(p) for p in immutable}
        design = {
            "diagnostic_only": True,
            "base_config": config,
            "fixture_construction": "nearest existing coordinate grid independently in r/az/el; retain seeded complex gains; split each triple into its three single-path fixtures",
            "element_noise_variance": 0.0, "rf_noise_variance": 0.0,
            "known_path_count": "1 for each single; 3 for each triple",
            "no_other_algorithm_or_acquisition_changes": True,
            "relative_whitening": "same fixed known row-energy weighting, no division by zero noise variance",
            "exact_geometry_tolerances": {"range_m": 1e-6, "azimuth_deg": 1e-4, "elevation_deg": 1e-4},
            "numerical_fullband_nmse_linear_max": 1e-8,
            "expected_rows": 120,
            "independent_parent_scenes": 5, "dependent_single_path_fixtures": 15,
        }
        write_json(directory / "config_resolved.json", design)
        sources = [Path(__file__).resolve(), config_path, ROOT / config["atmosphere"]["table_path"],
                   *sorted((ROOT / "src").rglob("*.py"))]
        write_json(directory / "code_manifest.json", [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in sources])
        (directory / "environment.txt").write_text(
            f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}\npython: {sys.version}\n"
            f"executable: {sys.executable}\nnumpy: {np.__version__}\nplatform: {operating_system.platform()}\n"
            "device: CPU; no training\n", encoding="utf-8")
        print(f"run_dir: {directory}", flush=True)
        p, weights, operators, dictionaries, search = build_stage2a_estimators(config, ROOT, progress=lambda m: print(m, flush=True))
        grids = [search["range_grid_m"], np.deg2rad(search["azimuth_grid_deg"]), np.deg2rad(search["elevation_grid_deg"])]
        reference_dictionary = next(iter(dictionaries.values()))
        dictionary_xyz = np.column_stack((reference_dictionary.ranges_m, reference_dictionary.azimuths_rad, reference_dictionary.elevations_rad))
        fixtures = []
        fixture_manifest = []
        for seed in config["experiment"]["replicate_seeds"]:
            original_paths = sample_stage2_paths(config, np.random.default_rng(seed))
            snapped = []
            for path in original_paths:
                x = [path.range_m, path.azimuth_rad, path.elevation_rad]
                nearest = [float(g[np.argmin(abs(g-v))]) for g, v in zip(grids, x)]
                snapped.append(PlanarPath(*nearest, path.complex_gain))
            fixture_manifest.append({"seed": seed, "original_coordinates_rad": coords(original_paths).tolist(),
                                     "snapped_coordinates_rad": coords(snapped).tolist(),
                                     "gains_real_imag": [[q.complex_gain.real, q.complex_gain.imag] for q in snapped]})
            for i, path in enumerate(snapped):
                fixtures.append((f"{seed}:single:{i}", "single", seed, (path,)))
            fixtures.append((f"{seed}:triple", "triple", seed, tuple(snapped)))
        write_json(directory / "fixtures.json", fixture_manifest)
        rows, traces, selection_checks = [], [], []
        max_forward_error = 0.0
        legacy = np.rint(np.linspace(0, 127, 16)).astype(int)
        for fixture_id, kind, seed, paths in fixtures:
            case_config = copy.deepcopy(config)
            case_config["channel"]["num_paths"] = len(paths)
            true_xyz = coords(paths)
            truth_indices = []
            for x in true_xyz:
                hits = np.flatnonzero(np.all(np.isclose(dictionary_xyz, x, rtol=0, atol=1e-12), axis=1))
                assert len(hits) == 1
                truth_indices.append(int(hits[0]))
            assert len(set(truth_indices)) == len(paths)
            gains = np.array([q.complex_gain for q in paths])
            channel = _channel_from_estimate(p.frequencies_hz, p, true_xyz, gains, config)
            for schedule in p.schedules:
                operator, dictionary = operators[schedule.name], dictionaries[schedule.name]
                observation = observe_multistrip_schedule(weights[schedule.name], channel[schedule.pilot_indices],
                    pilot_energy_per_re=1.0, element_noise_variance=0.0, rf_noise_variance=0.0)
                received = operator.whiten(observation.observed)
                energy = float(np.vdot(received, received).real)
                independent_templates = operator.evaluate(*true_xyz.T)
                forward_error = float(np.sum(abs(received - gains @ independent_templates)**2)/energy)
                max_forward_error = max(max_forward_error, forward_error)
                assert forward_error < 1e-20
                atoms = dictionary.templates
                norm2 = np.sum(abs(atoms)**2, axis=1, dtype=float)
                score = abs(atoms @ received.conj())**2 / norm2 / energy
                top = np.argsort(score)[-5:][::-1]
                selection_checks.append({"fixture_id": fixture_id, "schedule": schedule.name,
                    "truth_indices": truth_indices, "top5_indices": top.tolist(), "top5_normalized_scores": score[top].tolist(),
                    "true_atom_scores": score[truth_indices].tolist(),
                    "first_winner_in_truth": bool(int(top[0]) in truth_indices), "forward_nmse_linear": forward_error})
                for method in config["experiment"]["estimators"]:
                    fit, trace = traced_estimate(method, dictionary, operator, received, paths, case_config)
                    matched = _match_paths(fit, paths, case_config)
                    estimate = _channel_from_estimate(p.frequencies_hz, p, matched["coordinates"], matched["gains"], config)
                    domains = channel_domain_metrics(channel, estimate, schedule.pilot_indices, legacy)
                    maxima = [float(np.max(abs(matched[key]))) for key in ("range_error_m", "azimuth_error_deg", "elevation_error_deg")]
                    geometry_exact = all(v <= t for v, t in zip(maxima, [1e-6, 1e-4, 1e-4]))
                    original_success = all(v <= t for v, t in zip(maxima, [.03, 3, 4]))
                    selected = [int(i) for i in fit["candidate_index"]]
                    row = {"fixture_id": fixture_id, "kind": kind, "parent_seed": seed, "num_paths": len(paths),
                        "schedule": schedule.name, "estimator": method,
                        "geometry_exact": int(geometry_exact), "original_tolerance_success": int(original_success),
                        "numerical_recovery": int(geometry_exact and domains["fullband_128_nmse_linear"] <= 1e-8),
                        "max_range_error_m": maxima[0], "max_azimuth_error_deg": maxima[1], "max_elevation_error_deg": maxima[2],
                        "whitened_observation_nmse_linear": float(fit["residual_energy"] / energy),
                        "coarse_support_exact": int(set(selected) == set(truth_indices)) if method != "oracle_geometry_ls" else None,
                        "first_selected_in_truth": int(selected[0] in truth_indices) if method != "oracle_geometry_ls" else None,
                        "algorithm_iterations": int(fit["iterations"]), **domains}
                    rows.append(row)
                    for stage in trace:
                        stage["selected_last_in_truth"] = stage["selected_indices"][-1] in truth_indices
                        stage["residual_energy_fraction"] = stage["residual_energy"] / energy
                    traces.append({"fixture_id": fixture_id, "schedule": schedule.name, "estimator": method,
                        "truth_indices": truth_indices, "selected_indices": selected,
                        "estimated_coordinates_rad": matched["coordinates"].tolist(),
                        "estimated_gains_real_imag": [[float(g.real), float(g.imag)] for g in matched["gains"]], "stages": trace})
            print(f"completed {fixture_id}", flush=True)
        assert len(rows) == 120
        assert config == original_config
        assert before == {str(p.relative_to(ROOT)): sha(p) for p in immutable}
        with (directory / "metrics_raw.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        write_json(directory / "selection_traces.json", traces)
        write_json(directory / "first_selection_checks.json", selection_checks)
        aggregate = []
        for kind in ("single", "triple"):
            for schedule in p.schedules:
                for method in config["experiment"]["estimators"]:
                    block = [r for r in rows if r["kind"] == kind and r["schedule"] == schedule.name and r["estimator"] == method]
                    aggregate.append({"kind": kind, "schedule": schedule.name, "estimator": method, "fixtures": len(block),
                        "numerical_recovery": sum(r["numerical_recovery"] for r in block),
                        "geometry_exact": sum(r["geometry_exact"] for r in block),
                        "original_tolerance_success": sum(r["original_tolerance_success"] for r in block),
                        "coarse_support_exact": None if method == "oracle_geometry_ls" else sum(r["coarse_support_exact"] for r in block),
                        "first_selected_in_truth": None if method == "oracle_geometry_ls" else sum(r["first_selected_in_truth"] for r in block),
                        "mean_fullband_nmse_db": float(10*np.log10(max(np.mean([r["fullband_128_nmse_linear"] for r in block]), np.finfo(float).tiny))),
                        "max_fullband_nmse_db": max(r["fullband_128_nmse_db"] for r in block)})
        summary = {"diagnostic_only": True, "raw_rows": len(rows), "parent_scenes": 5, "single_fixtures": 15,
            "num_candidates_per_schedule": search["num_candidates"], "max_forward_nmse_linear": max_forward_error,
            "historical_runs_and_production_files_unchanged": True, "wall_seconds": time.perf_counter()-started,
            "aggregate": aggregate}
        write_json(directory / "analysis_compact.json", summary)
        lines = ["# Stage 2-A 无噪声网格内验收", "",
                 "固定原 5,082 点字典、F1/F2 状态和导频、原估计器与 2 次局部迭代。将 5 个种子场景就近投到既有网格，仅构造测试夹具；原生产信道和配置不变。每个三径夹具拆成三个单径，共 15 个相关单径夹具及 5 个三径夹具，不是 20 个独立场景。", "",
                 "数值恢复要求全部路径误差≤1e-6 m / 1e-4° / 1e-4°，且全带 NMSE≤−80 dB。另报原 3 cm / 3° / 4° 容差。失败不是运行异常，不改变判据。", "",
                 "|夹具|协议|方法|数值恢复|原容差成功|粗支持完全正确|首次选真径|全带 NMSE 均值/dB|",
                 "|---|---|---|---:|---:|---:|---:|---:|"]
        for r in aggregate:
            lines.append(f"|{r['kind']}|{r['schedule']}|{r['estimator']}|{r['numerical_recovery']}/{r['fixtures']}|{r['original_tolerance_success']}/{r['fixtures']}|{r['coarse_support_exact']}|{r['first_selected_in_truth']}|{r['mean_fullband_nmse_db']:.3f}|")
        lines += ["", "逐阶段只旁路记录选择、细化前后坐标和残差，没有修改估计器返回值。数值恢复计数不是统计置信度或正式方法排名。", "",
                  f"独立信道生成与模板前向最大相对误差：{max_forward_error:.3e}。历史运行与生产文件哈希未变。", ""]
        (directory / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
        write_json(directory / "status.json", {"status": "complete", "raw_rows": len(rows), "timestamp_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        write_json(directory / "status.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
