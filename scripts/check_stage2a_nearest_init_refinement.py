#!/usr/bin/env python3
"""Test the unchanged local refinement from oracle-nearest grid cells."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import numpy as np

from thz_dma.channels.planar import PlanarPath
from thz_dma.estimators.planar_sparse import _continuous_refinement
from thz_dma.observations.multistrip import observe_multistrip_schedule
from thz_dma.stage2 import sample_stage2_paths
from thz_dma.stage2a import (
    _channel_from_estimate,
    _match_paths,
    build_stage2a_estimators,
    channel_domain_metrics,
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def xyz(paths):
    return np.array([[p.range_m, p.azimuth_rad, p.elevation_rad] for p in paths])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="stage2a_nearest_init_20260910_v1")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError("run ID must be one directory name")
    run_dir = ROOT / "runs" / args.run_id
    run_dir.mkdir(exist_ok=False)
    write_json(run_dir / "status.json", {"status": "running"})
    try:
        config_path = ROOT / "configs/direction1_stage2a_bounded_diagnostic.toml"
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        protected = [config_path, ROOT / "src/thz_dma/estimators/planar_sparse.py"]
        for run_id in (
            "stage2a_bounded_20260910_v1", "stage2a_bounded_65db_20260910_v1",
            "stage2a_ongrid_noiseless_20260910_v1", "stage2a_offgrid_factorial_20260910_v1",
        ):
            protected.extend(p for p in (ROOT / "runs" / run_id).rglob("*") if p.is_file())
        before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
        design = {
            "diagnostic_only": True,
            "truth": "original continuous five seeded scenes, noiseless; each triple and its dependent single paths",
            "initialization": "oracle-nearest existing 1 cm / 2 deg / 3 deg grid cells",
            "refinement": config["estimator"]["yang_ogols_3d"],
            "fixed": "operators, schedules, pilots, bounds and production refinement implementation",
            "pre_stated_screens": {
                "geometry": "existing 3 cm / 3 deg / 4 deg all-path tolerances",
                "fullband": "NMSE <= -10 dB descriptive screen",
            },
            "role": "separates coarse-support acquisition from local-refinement capability; oracle initialization is not deployable",
            "expected_rows": 40,
        }
        write_json(run_dir / "design_pre_registered.json", design)
        source_paths = [Path(__file__).resolve(), config_path, *sorted((ROOT / "src").rglob("*.py"))]
        write_json(run_dir / "code_manifest.json", [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in source_paths])
        (run_dir / "environment.txt").write_text(
            f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}\npython: {sys.version}\n"
            f"executable: {sys.executable}\nnumpy: {np.__version__}\ndevice: CPU; no training\n",
            encoding="utf-8",
        )
        platform, weights, operators, _, search = build_stage2a_estimators(
            config, ROOT, progress=lambda message: print(message, flush=True)
        )
        grids = (
            search["range_grid_m"], np.deg2rad(search["azimuth_grid_deg"]),
            np.deg2rad(search["elevation_grid_deg"]),
        )
        settings = config["estimator"]["yang_ogols_3d"]
        lower = np.array([config["search"]["range_min_m"], np.deg2rad(config["search"]["azimuth_min_deg"]), np.deg2rad(config["search"]["elevation_min_deg"])] )
        upper = np.array([config["search"]["range_max_m"], np.deg2rad(config["search"]["azimuth_max_deg"]), np.deg2rad(config["search"]["elevation_max_deg"])] )
        steps = np.array([config["search"]["range_step_m"], np.deg2rad(config["search"]["azimuth_step_deg"]), np.deg2rad(config["search"]["elevation_step_deg"])] )
        rows = []
        legacy = np.rint(np.linspace(0, 127, 16)).astype(int)
        for seed in config["experiment"]["replicate_seeds"]:
            parent = sample_stage2_paths(config, np.random.default_rng(seed))
            cases = [(f"{seed}:single:{i}", "single", (path,)) for i, path in enumerate(parent)]
            cases.append((f"{seed}:triple", "triple", parent))
            for fixture_id, kind, paths in cases:
                truth_xyz = xyz(paths)
                initial_xyz = np.column_stack([
                    grid[np.argmin(abs(grid[:, None] - truth_xyz[:, dim]), axis=0)]
                    for dim, grid in enumerate(grids)
                ])
                gains = np.array([p.complex_gain for p in paths])
                channel = _channel_from_estimate(platform.frequencies_hz, platform, truth_xyz, gains, config)
                case_config = json.loads(json.dumps(config))
                case_config["channel"]["num_paths"] = len(paths)
                for schedule in platform.schedules:
                    operator = operators[schedule.name]
                    observation = observe_multistrip_schedule(
                        weights[schedule.name], channel[schedule.pilot_indices], pilot_energy_per_re=1.0,
                        element_noise_variance=0.0, rf_noise_variance=0.0,
                    )
                    received = operator.whiten(observation.clean)
                    result = _continuous_refinement(
                        operator, received, initial_xyz,
                        lower_bounds=lower, upper_bounds=upper, grid_steps=steps,
                        refinement_iterations=int(settings["refinement_iterations"]),
                        derivative_fraction=float(settings["derivative_fraction"]),
                        maximum_update_grid_units=float(settings["maximum_update_grid_units"]),
                        convergence_tolerance=float(settings["convergence_tolerance"]),
                    )
                    estimate = {
                        "range_m": result["coordinates"][:, 0],
                        "azimuth_rad": result["coordinates"][:, 1],
                        "elevation_rad": result["coordinates"][:, 2],
                        "complex_gain": result["complex_gain"],
                    }
                    matched = _match_paths(estimate, paths, case_config)
                    reconstructed = _channel_from_estimate(
                        platform.frequencies_hz, platform, matched["coordinates"], matched["gains"], config
                    )
                    metrics = channel_domain_metrics(channel, reconstructed, schedule.pilot_indices, legacy)
                    errors = [
                        np.asarray(matched["range_error_m"]), np.asarray(matched["azimuth_error_deg"]),
                        np.asarray(matched["elevation_error_deg"]),
                    ]
                    success = bool(
                        np.all(abs(errors[0]) <= .03) and np.all(abs(errors[1]) <= 3)
                        and np.all(abs(errors[2]) <= 4)
                    )
                    rows.append({
                        "fixture_id": fixture_id, "kind": kind, "parent_seed": seed,
                        "schedule": schedule.name, "num_paths": len(paths),
                        "geometry_success": int(success),
                        "basic_fullband_csi": int(metrics["fullband_128_nmse_db"] <= -10),
                        "initial_max_range_offset_m": float(np.max(abs(initial_xyz[:, 0]-truth_xyz[:, 0]))),
                        "initial_max_azimuth_offset_deg": float(np.max(abs(np.rad2deg(initial_xyz[:, 1]-truth_xyz[:, 1])))),
                        "initial_max_elevation_offset_deg": float(np.max(abs(np.rad2deg(initial_xyz[:, 2]-truth_xyz[:, 2])))),
                        "final_max_range_error_m": float(np.max(abs(errors[0]))),
                        "final_max_azimuth_error_deg": float(np.max(abs(errors[1]))),
                        "final_max_elevation_error_deg": float(np.max(abs(errors[2]))),
                        "initial_residual_fraction": float(result["initial_residual_energy"] / np.vdot(received, received).real),
                        "final_residual_fraction": float(result["residual_energy"] / np.vdot(received, received).real),
                        "iterations": int(result["iterations"]), "converged": int(result["converged"]),
                        **metrics,
                    })
            print(f"completed seed {seed}", flush=True)
        if len(rows) != design["expected_rows"]:
            raise RuntimeError(f"expected 40 rows, got {len(rows)}")
        if before != {str(p.relative_to(ROOT)): sha(p) for p in protected}:
            raise RuntimeError("protected input or historical result changed")
        with (run_dir / "metrics_raw.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        aggregate = []
        for kind in ("single", "triple"):
            for schedule in platform.schedules:
                block = [r for r in rows if r["kind"] == kind and r["schedule"] == schedule.name]
                aggregate.append({
                    "kind": kind, "schedule": schedule.name, "fixtures": len(block),
                    "geometry_success": sum(r["geometry_success"] for r in block),
                    "basic_fullband_csi": sum(r["basic_fullband_csi"] for r in block),
                    "mean_fullband_nmse_db": float(10*np.log10(np.mean([r["fullband_128_nmse_linear"] for r in block]))),
                    "median_fullband_nmse_db": float(np.median([r["fullband_128_nmse_db"] for r in block])),
                    "median_final_range_error_m": float(np.median([r["final_max_range_error_m"] for r in block])),
                    "median_final_azimuth_error_deg": float(np.median([r["final_max_azimuth_error_deg"] for r in block])),
                    "median_final_elevation_error_deg": float(np.median([r["final_max_elevation_error_deg"] for r in block])),
                })
        summary = {"diagnostic_only": True, "raw_rows": len(rows), "aggregate": aggregate,
                   "historical_runs_and_production_files_unchanged": True}
        write_json(run_dir / "analysis_compact.json", summary)
        lines = ["# Stage 2-A 最近网格真值初始化的局部细化检查", "",
                 "原连续真值、无噪声，使用真实路径对应的最近既有网格单元作为初始化；局部细化代码、2 次迭代及所有步长保持生产设置。该初始化使用真值，只作故障定位，不能部署。", "",
                 "|夹具|协议|几何成功|全带≤−10 dB|全带 NMSE 均值/dB|全带 NMSE 中位数/dB|",
                 "|---|---|---:|---:|---:|---:|"]
        for r in aggregate:
            lines.append(f"|{r['kind']}|{r['schedule']}|{r['geometry_success']}/{r['fixtures']}|{r['basic_fullband_csi']}/{r['fixtures']}|{r['mean_fullband_nmse_db']:.3f}|{r['median_fullband_nmse_db']:.3f}|")
        lines += ["", "若本检查通过而生产 Yang 失败，主因位于粗支持未进入正确局部盆地；若本检查也失败，则现有两次局部细化本身不足。原始逐样本数据保留两种情况。", ""]
        (run_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
        write_json(run_dir / "status.json", {"status": "complete", "raw_rows": len(rows), "timestamp_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        write_json(run_dir / "status.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
