#!/usr/bin/env python3
"""Noiseless 2^3 off-grid factor diagnostic for the fixed Stage 2-A setup."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
import time
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

from check_stage2a_ongrid_noiseless import traced_estimate
from thz_dma.channels.planar import PlanarPath
from thz_dma.observations.multistrip import observe_multistrip_schedule
from thz_dma.stage2 import sample_stage2_paths
from thz_dma.stage2a import (
    _channel_from_estimate,
    _estimate,
    _match_paths,
    build_stage2a_estimators,
    channel_domain_metrics,
)


VARIANTS = [
    ("grid", (0, 0, 0)),
    ("range", (1, 0, 0)),
    ("azimuth", (0, 1, 0)),
    ("elevation", (0, 0, 1)),
    ("range_azimuth", (1, 1, 0)),
    ("range_elevation", (1, 0, 1)),
    ("azimuth_elevation", (0, 1, 1)),
    ("all_continuous", (1, 1, 1)),
]
METHODS = ("grid_omp_3d", "yang_ogols_3d")


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def xyz(paths) -> np.ndarray:
    return np.array(
        [[p.range_m, p.azimuth_rad, p.elevation_rad] for p in paths], dtype=float
    )


def linear_mean_db(values) -> float:
    return float(
        10.0
        * np.log10(max(float(np.mean(values)), np.finfo(float).tiny))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="stage2a_offgrid_factorial_20260910_v1")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError("run ID must be one directory name")
    run_dir = ROOT / "runs" / args.run_id
    run_dir.mkdir(exist_ok=False)
    started = time.perf_counter()
    write_json(run_dir / "status.json", {"status": "running"})
    try:
        config_path = ROOT / "configs/direction1_stage2a_bounded_diagnostic.toml"
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        original_config = copy.deepcopy(config)
        protected = [
            config_path,
            ROOT / "configs/direction1_stage2a_bounded_65db_diagnostic.toml",
            ROOT / "src/thz_dma/estimators/planar_sparse.py",
            ROOT / "src/thz_dma/surfaces/multistrip.py",
        ]
        for run_id in (
            "stage2a_bounded_20260910_v1",
            "stage2a_bounded_65db_20260910_v1",
            "stage2a_ongrid_noiseless_20260910_v1",
        ):
            protected.extend(
                path for path in (ROOT / "runs" / run_id).rglob("*") if path.is_file()
            )
        protected_before = {str(path.relative_to(ROOT)): sha(path) for path in protected}
        design = {
            "diagnostic_only": True,
            "base_config": config,
            "factorial": {
                "factors": ["range_off_grid", "azimuth_off_grid", "elevation_off_grid"],
                "levels": {"0": "nearest existing grid coordinate", "1": "original continuous coordinate"},
                "variants": [{"name": name, "levels": list(bits)} for name, bits in VARIANTS],
            },
            "fixtures": "five original seeded triples and their 15 dependent single paths",
            "noise": "none",
            "methods": list(METHODS),
            "fixed": "existing 5082-point dictionaries, F1/F2 states and pilots, estimator settings including two Yang refinement iterations",
            "pre_stated_metrics": {
                "geometry_success": "existing 0.03 m / 3 deg / 4 deg all-path tolerances",
                "basic_fullband_csi": "NMSE <= -10 dB, i.e. error energy <= 10% of truth; descriptive diagnostic screen only",
                "nearest_cell_support": "selected coarse support equals the fully snapped nearest cells",
                "single_path_nearest_atom_rho2": "normalized squared observation-space correlation; no acceptance threshold",
                "factor_effect": "paired off-grid minus grid-coordinate result for the same fixture, schedule and method",
            },
            "expected_rows": 640,
            "not_authorized_or_claimed": [
                "formal Monte Carlo", "method ranking", "state redesign", "pilot reorder", "estimator tuning",
            ],
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
            [{"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for path in sources],
        )
        (run_dir / "environment.txt").write_text(
            f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}\n"
            f"python: {sys.version}\nexecutable: {sys.executable}\nnumpy: {np.__version__}\n"
            "device: CPU; no training\n",
            encoding="utf-8",
        )
        print(f"run_dir: {run_dir}", flush=True)
        platform, weights, operators, dictionaries, search = build_stage2a_estimators(
            config, ROOT, progress=lambda message: print(message, flush=True)
        )
        grids = (
            search["range_grid_m"],
            np.deg2rad(search["azimuth_grid_deg"]),
            np.deg2rad(search["elevation_grid_deg"]),
        )
        reference = next(iter(dictionaries.values()))
        dictionary_xyz = np.column_stack(
            (reference.ranges_m, reference.azimuths_rad, reference.elevations_rad)
        )
        parents = []
        for seed in config["experiment"]["replicate_seeds"]:
            continuous_paths = sample_stage2_paths(config, np.random.default_rng(seed))
            continuous_xyz = xyz(continuous_paths)
            snapped_xyz = np.column_stack(
                [grid[np.argmin(abs(grid[:, None] - continuous_xyz[:, dim]), axis=0)] for dim, grid in enumerate(grids)]
            )
            parents.append(
                {
                    "seed": seed,
                    "continuous_xyz_rad": continuous_xyz.tolist(),
                    "snapped_xyz_rad": snapped_xyz.tolist(),
                    "offsets_m_deg_deg": np.column_stack(
                        (
                            continuous_xyz[:, 0] - snapped_xyz[:, 0],
                            np.rad2deg(continuous_xyz[:, 1] - snapped_xyz[:, 1]),
                            np.rad2deg(continuous_xyz[:, 2] - snapped_xyz[:, 2]),
                        )
                    ).tolist(),
                    "gains_real_imag": [[p.complex_gain.real, p.complex_gain.imag] for p in continuous_paths],
                }
            )
        write_json(run_dir / "parent_fixtures.json", parents)

        rows, continuous_traces, first_checks = [], [], []
        legacy = np.rint(np.linspace(0, 127, 16)).astype(int)
        max_forward_nmse = 0.0
        for parent in parents:
            seed = parent["seed"]
            continuous_xyz = np.asarray(parent["continuous_xyz_rad"], dtype=float)
            snapped_xyz = np.asarray(parent["snapped_xyz_rad"], dtype=float)
            gains = np.array([complex(real, imag) for real, imag in parent["gains_real_imag"]])
            for variant, bits in VARIANTS:
                mask = np.asarray(bits, dtype=bool)
                variant_xyz = np.where(mask[None, :], continuous_xyz, snapped_xyz)
                variant_paths = tuple(
                    PlanarPath(*variant_xyz[index], gains[index]) for index in range(3)
                )
                cases = [(f"{seed}:single:{index}", "single", (path,)) for index, path in enumerate(variant_paths)]
                cases.append((f"{seed}:triple", "triple", variant_paths))
                for fixture_id, kind, paths in cases:
                    truth_xyz = xyz(paths)
                    truth_gains = np.array([path.complex_gain for path in paths])
                    nearest_xyz = np.column_stack(
                        [grid[np.argmin(abs(grid[:, None] - truth_xyz[:, dim]), axis=0)] for dim, grid in enumerate(grids)]
                    )
                    nearest_indices = []
                    for point in nearest_xyz:
                        hits = np.flatnonzero(
                            np.all(np.isclose(dictionary_xyz, point, rtol=0, atol=1e-12), axis=1)
                        )
                        if len(hits) != 1:
                            raise RuntimeError("nearest grid cell is not unique")
                        nearest_indices.append(int(hits[0]))
                    channel = _channel_from_estimate(
                        platform.frequencies_hz, platform, truth_xyz, truth_gains, config
                    )
                    case_config = copy.deepcopy(config)
                    case_config["channel"]["num_paths"] = len(paths)
                    for schedule in platform.schedules:
                        operator = operators[schedule.name]
                        dictionary = dictionaries[schedule.name]
                        observation = observe_multistrip_schedule(
                            weights[schedule.name],
                            channel[schedule.pilot_indices],
                            pilot_energy_per_re=1.0,
                            element_noise_variance=0.0,
                            rf_noise_variance=0.0,
                        )
                        received = operator.whiten(observation.clean)
                        energy = float(np.vdot(received, received).real)
                        exact_templates = operator.evaluate(*truth_xyz.T)
                        forward_nmse = float(
                            np.sum(abs(received - truth_gains @ exact_templates) ** 2) / energy
                        )
                        max_forward_nmse = max(max_forward_nmse, forward_nmse)
                        if forward_nmse >= 1e-20:
                            raise RuntimeError("independent channel/operator forward mismatch")
                        atom_energy = np.sum(abs(dictionary.templates) ** 2, axis=1, dtype=float)
                        first_score = abs(dictionary.templates @ received.conj()) ** 2 / atom_energy / energy
                        winner = int(np.argmax(first_score))
                        check = {
                            "fixture_id": fixture_id,
                            "variant": variant,
                            "schedule": schedule.name,
                            "nearest_indices": nearest_indices,
                            "first_winner": winner,
                            "first_winner_in_nearest_cells": winner in nearest_indices,
                            "nearest_cell_scores": first_score[nearest_indices].tolist(),
                        }
                        if kind == "single":
                            check["nearest_atom_rho2"] = float(first_score[nearest_indices[0]])
                            check["nearest_atom_loss_db"] = float(
                                -10.0 * np.log10(max(check["nearest_atom_rho2"], np.finfo(float).tiny))
                            )
                        first_checks.append(check)
                        for method in METHODS:
                            if variant == "all_continuous" and kind == "triple":
                                estimate, trace = traced_estimate(
                                    method, dictionary, operator, received, paths, case_config
                                )
                                continuous_traces.append(
                                    {
                                        "fixture_id": fixture_id,
                                        "schedule": schedule.name,
                                        "estimator": method,
                                        "nearest_indices": nearest_indices,
                                        "selected_indices": [int(value) for value in estimate["candidate_index"]],
                                        "stages": trace,
                                    }
                                )
                            else:
                                estimate = _estimate(
                                    method, dictionary, operator, received, paths, case_config
                                )
                            matched = _match_paths(estimate, paths, case_config)
                            reconstructed = _channel_from_estimate(
                                platform.frequencies_hz,
                                platform,
                                matched["coordinates"],
                                matched["gains"],
                                config,
                            )
                            metrics = channel_domain_metrics(
                                channel, reconstructed, schedule.pilot_indices, legacy
                            )
                            range_error = np.asarray(matched["range_error_m"])
                            azimuth_error = np.asarray(matched["azimuth_error_deg"])
                            elevation_error = np.asarray(matched["elevation_error_deg"])
                            selected = [int(value) for value in estimate["candidate_index"]]
                            geometry_success = bool(
                                np.all(abs(range_error) <= 0.03)
                                and np.all(abs(azimuth_error) <= 3.0)
                                and np.all(abs(elevation_error) <= 4.0)
                            )
                            rows.append(
                                {
                                    "fixture_id": fixture_id,
                                    "kind": kind,
                                    "parent_seed": seed,
                                    "variant": variant,
                                    "range_off_grid": bits[0],
                                    "azimuth_off_grid": bits[1],
                                    "elevation_off_grid": bits[2],
                                    "schedule": schedule.name,
                                    "schedule_family": schedule.family,
                                    "estimator": method,
                                    "num_paths": len(paths),
                                    "geometry_success": int(geometry_success),
                                    "basic_fullband_csi": int(metrics["fullband_128_nmse_db"] <= -10.0),
                                    "max_range_error_m": float(np.max(abs(range_error))),
                                    "max_azimuth_error_deg": float(np.max(abs(azimuth_error))),
                                    "max_elevation_error_deg": float(np.max(abs(elevation_error))),
                                    "range_rmse_m": float(np.sqrt(np.mean(range_error**2))),
                                    "azimuth_rmse_deg": float(np.sqrt(np.mean(azimuth_error**2))),
                                    "elevation_rmse_deg": float(np.sqrt(np.mean(elevation_error**2))),
                                    "coarse_support_nearest_exact": int(set(selected) == set(nearest_indices)),
                                    "first_selected_nearest": int(selected[0] in nearest_indices),
                                    "selected_indices": ";".join(map(str, selected)),
                                    "nearest_indices": ";".join(map(str, nearest_indices)),
                                    "whitened_observation_nmse_linear": float(estimate["residual_energy"] / energy),
                                    **metrics,
                                }
                            )
            print(f"completed parent seed {seed}", flush=True)
        if len(rows) != design["expected_rows"]:
            raise RuntimeError(f"expected 640 rows, got {len(rows)}")
        if config != original_config:
            raise RuntimeError("configuration mutated during diagnostic")
        protected_after = {str(path.relative_to(ROOT)): sha(path) for path in protected}
        if protected_before != protected_after:
            raise RuntimeError("a protected input or historical result changed")
        with (run_dir / "metrics_raw.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(run_dir / "first_selection_checks.json", first_checks)
        write_json(run_dir / "all_continuous_traces.json", continuous_traces)

        aggregate = []
        for variant, bits in VARIANTS:
            for kind in ("single", "triple"):
                for schedule in platform.schedules:
                    for method in METHODS:
                        block = [
                            row
                            for row in rows
                            if row["variant"] == variant
                            and row["kind"] == kind
                            and row["schedule"] == schedule.name
                            and row["estimator"] == method
                        ]
                        aggregate.append(
                            {
                                "variant": variant,
                                "factor_levels": list(bits),
                                "kind": kind,
                                "schedule": schedule.name,
                                "estimator": method,
                                "fixtures": len(block),
                                "geometry_success": sum(row["geometry_success"] for row in block),
                                "basic_fullband_csi": sum(row["basic_fullband_csi"] for row in block),
                                "coarse_support_nearest_exact": sum(row["coarse_support_nearest_exact"] for row in block),
                                "first_selected_nearest": sum(row["first_selected_nearest"] for row in block),
                                "mean_fullband_nmse_db": linear_mean_db(
                                    [row["fullband_128_nmse_linear"] for row in block]
                                ),
                                "median_fullband_nmse_db": float(
                                    np.median([row["fullband_128_nmse_db"] for row in block])
                                ),
                                "median_range_rmse_m": float(np.median([row["range_rmse_m"] for row in block])),
                                "median_azimuth_rmse_deg": float(np.median([row["azimuth_rmse_deg"] for row in block])),
                                "median_elevation_rmse_deg": float(np.median([row["elevation_rmse_deg"] for row in block])),
                            }
                        )
        single_mismatch = []
        for variant, _ in VARIANTS:
            for schedule in platform.schedules:
                block = [
                    item for item in first_checks
                    if item["variant"] == variant
                    and item["schedule"] == schedule.name
                    and ":single:" in item["fixture_id"]
                ]
                single_mismatch.append(
                    {
                        "variant": variant,
                        "schedule": schedule.name,
                        "fixtures": len(block),
                        "nearest_cell_first_wins": sum(item["first_winner_in_nearest_cells"] for item in block),
                        "median_nearest_atom_rho2": float(np.median([item["nearest_atom_rho2"] for item in block])),
                        "minimum_nearest_atom_rho2": float(np.min([item["nearest_atom_rho2"] for item in block])),
                        "median_nearest_atom_loss_db": float(np.median([item["nearest_atom_loss_db"] for item in block])),
                    }
                )
        paired_effects = []
        row_index = {
            (row["fixture_id"], row["variant"], row["schedule"], row["estimator"]): row
            for row in rows
        }
        variant_by_bits = {bits: name for name, bits in VARIANTS}
        for dimension, dimension_name in enumerate(("range", "azimuth", "elevation")):
            for kind in ("single", "triple"):
                for schedule in platform.schedules:
                    for method in METHODS:
                        deltas, success_losses, first_losses = [], [], []
                        for _, bits in VARIANTS:
                            if bits[dimension] != 0:
                                continue
                            off_bits = tuple(1 if index == dimension else value for index, value in enumerate(bits))
                            if off_bits not in variant_by_bits:
                                continue
                            base_name, off_name = variant_by_bits[bits], variant_by_bits[off_bits]
                            fixture_ids = sorted(
                                {
                                    row["fixture_id"] for row in rows
                                    if row["kind"] == kind and row["variant"] == base_name
                                }
                            )
                            for fixture_id in fixture_ids:
                                base = row_index[(fixture_id, base_name, schedule.name, method)]
                                off = row_index[(fixture_id, off_name, schedule.name, method)]
                                deltas.append(off["fullband_128_nmse_db"] - base["fullband_128_nmse_db"])
                                success_losses.append(base["geometry_success"] - off["geometry_success"])
                                first_losses.append(base["first_selected_nearest"] - off["first_selected_nearest"])
                        paired_effects.append(
                            {
                                "factor": dimension_name,
                                "kind": kind,
                                "schedule": schedule.name,
                                "estimator": method,
                                "paired_comparisons": len(deltas),
                                "median_nmse_degradation_db": float(np.median(deltas)),
                                "mean_nmse_degradation_db": float(np.mean(deltas)),
                                "geometry_success_losses": int(sum(success_losses)),
                                "first_selection_nearest_losses": int(sum(first_losses)),
                            }
                        )
        summary = {
            "diagnostic_only": True,
            "raw_rows": len(rows),
            "parent_scenes": len(parents),
            "dependent_single_path_fixtures_per_variant": 15,
            "triple_fixtures_per_variant": 5,
            "max_forward_nmse_linear": max_forward_nmse,
            "historical_runs_and_production_files_unchanged": True,
            "wall_seconds": time.perf_counter() - started,
            "aggregate": aggregate,
            "single_path_basis_mismatch": single_mismatch,
            "factor_paired_effects": paired_effects,
        }
        write_json(run_dir / "analysis_compact.json", summary)
        lines = [
            "# Stage 2-A 无噪声离网格全因子诊断",
            "",
            "距离、方位、俯仰分别取既有网格点或原连续值，共八种组合。固定字典、状态、导频、估计器及 Yang 两次迭代。5 组三径与其派生 15 个单径有依赖，只作故障定位。",
            "",
            "几何成功沿用 3 cm/3°/4°；基本全带恢复取 NMSE≤−10 dB，即误差能量不超过真值的 10%。该阈值在运行前记录，只作描述性筛查。",
            "",
            "|变体|夹具|协议|方法|几何成功|全带≤−10 dB|粗支持为最近格|首次选最近格|全带 NMSE 均值/dB|",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in aggregate:
            lines.append(
                f"|{row['variant']}|{row['kind']}|{row['schedule']}|{row['estimator']}|"
                f"{row['geometry_success']}/{row['fixtures']}|{row['basic_fullband_csi']}/{row['fixtures']}|"
                f"{row['coarse_support_nearest_exact']}/{row['fixtures']}|{row['first_selected_nearest']}/{row['fixtures']}|"
                f"{row['mean_fullband_nmse_db']:.3f}|"
            )
        lines += [
            "",
            "## 单径最近网格模板相关度",
            "",
            "|变体|协议|最近格首次胜出|相关度平方中位数|最小值|损失中位数/dB|",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in single_mismatch:
            lines.append(
                f"|{row['variant']}|{row['schedule']}|{row['nearest_cell_first_wins']}/{row['fixtures']}|"
                f"{row['median_nearest_atom_rho2']:.6f}|{row['minimum_nearest_atom_rho2']:.6f}|"
                f"{row['median_nearest_atom_loss_db']:.3f}|"
            )
        lines += [
            "",
            "全因子配对效应保存在 `analysis_compact.json`。dB 退化包含网格点数值零误差到离网格误差的跃迁，主要用于同维度排序，不解释为统计显著性。逐阶段轨迹只对原连续三径保存。",
            "",
            f"独立前向最大 NMSE={max_forward_nmse:.3e}；受保护的配置、估计器和历史运行哈希均未变化。",
            "",
        ]
        (run_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
        write_json(
            run_dir / "status.json",
            {
                "status": "complete",
                "raw_rows": len(rows),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(json.dumps({key: summary[key] for key in ("raw_rows", "max_forward_nmse_linear", "wall_seconds")}, indent=2), flush=True)
    except Exception as error:
        write_json(run_dir / "status.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
