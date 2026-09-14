"""Bounded Stage 2-A domain/SNR diagnostics; no estimator execution."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DOMAINS = ("pilot_nmse", "fullband_128_nmse", "heldout_112_nmse")
BASELINE_ID = "stage2a_bounded_20260910_v1"


def validate_domains(frame: pd.DataFrame, summary: dict) -> bool:
    if summary["scope"].get("csi_metric_version") != "frequency-domains-v2":
        return False
    for schedule in summary["schedules"]:
        block = frame[frame.schedule == schedule["name"]]
        pilots = schedule["pilot_indices"]
        heldout = schedule["heldout_indices"]
        if sorted(pilots + heldout) != list(range(128)) or set(pilots) & set(heldout):
            raise ValueError("CSI domains do not partition 128 tones")
        for name, count in zip(DOMAINS, (len(pilots), 128, len(heldout))):
            if not (block[f"{name}_subcarrier_count"] == count).all():
                raise ValueError(f"incorrect domain count: {name}")
            values = block[[f"{name}_linear", f"{name}_db"]].to_numpy(float)
            if count == 0:
                if not np.isnan(values).all():
                    raise ValueError("empty held-out domain must be missing")
            elif not np.isfinite(values).all() or np.any(values[:, 0] < 0):
                raise ValueError(f"invalid domain NMSE: {name}")
            elif not np.allclose(values[:, 1], 10 * np.log10(np.maximum(values[:, 0], np.finfo(float).tiny))):
                raise ValueError(f"linear/dB mismatch: {name}")
        if len(pilots) == 128 and not np.allclose(block.pilot_nmse_linear, block.fullband_128_nmse_linear):
            raise ValueError("F1 pilot and fullband NMSE must agree")
    return True


def _dbmean(values):
    return float(10 * np.log10(max(float(np.mean(values)), np.finfo(float).tiny)))


def domain_summary(frame):
    rows = []
    if not all(f"{name}_linear" in frame for name in DOMAINS):
        return rows
    for (schedule, snr, method), block in frame.groupby(["schedule", "array_reference_snr_db", "estimator"], sort=False):
        row = {"schedule": str(schedule), "snr_db": float(snr), "estimator": str(method), "scenes": len(block)}
        for name in DOMAINS:
            values = block[f"{name}_linear"].dropna()
            row[f"mean_{name}_db"] = _dbmean(values) if len(values) else None
        rows.append(row)
    return rows


def _config_comparable(config, baseline):
    a, b = copy.deepcopy(config), copy.deepcopy(baseline)
    for obj in (a, b):
        obj["experiment"].pop("name", None)
        obj["noise"].pop("array_reference_snr_db", None)
        obj["analysis"].pop("primary_array_reference_snr_db", None)
    return a == b


def _azimuth_stats(block, config):
    errors = np.column_stack([
        block[f"estimated_path_{i}_azimuth_deg"].to_numpy(float)
        - block[f"true_path_{i}_azimuth_deg"].to_numpy(float)
        for i in range(int(config["channel"]["num_paths"]))
    ])
    estimates = np.column_stack([block[f"estimated_path_{i}_azimuth_deg"] for i in range(errors.shape[1])])
    bounds = [config["search"][f"azimuth_{side}_deg"] for side in ("min", "max")]
    return {
        "median_azimuth_rmse_deg": float(block.azimuth_rmse_deg.median()),
        "all_azimuths_within_tolerance_scenes": int(np.sum(np.all(np.abs(errors) <= config["evaluation"]["path_success_azimuth_tolerance_deg"], axis=1))),
        "all_paths_success_scenes": int(block.all_paths_success.sum()),
        "azimuth_lower_boundary_paths": int(np.isclose(estimates, bounds[0], rtol=0, atol=1e-10).sum()),
        "azimuth_any_boundary_paths": int(np.isclose(estimates[..., None], bounds, rtol=0, atol=1e-10).any(axis=-1).sum()),
    }


def bounded_diagnostics(frame, config, directory, integrity):
    if str(config.get("search", {}).get("mode", "fixed")) == "beam_center_adaptive":
        return {
            "status": "adaptive_search_diagnostic; legacy fixed-sector 55-to-65 dB screen not applied",
            "criteria": {},
            "F1": [],
            "F2": [],
        }
    result = {
        "status": "diagnostic_only; no authorization for formal run",
        "criteria": {
            "F1_improvement": "paired azimuth RMSE decreases in >=4/5 scenes and median paired delta <0",
            "F1_recovery": "all three azimuths within existing 3-degree tolerance in >=4/5 scenes; report joint path success separately",
            "F2_pilot_good_db": -5.0,
            "F2_min_gap_db": 6.0,
            "F2_pattern": "pilot <= -5 dB and either fullband or heldout minus pilot >=6 dB in >=4/5 scenes",
            "interpretation": "descriptive five-scene thresholds, not significance or proof of pilot causality",
        },
        "F1": [], "F2": [],
    }
    if not integrity["complete"]:
        result["status"] = "invalid_data; diagnosis withheld"
        return result
    if not config["experiment"].get("diagnostic_only"):
        result["status"] = "not_bounded_diagnostic"
        return result
    high = frame[np.isclose(frame.array_reference_snr_db, 65)]
    if high.empty:
        result["status"] = "pending_65db; historical 16-tone metrics cannot fill missing domains"
        return result
    baseline_dir = Path(directory).parent / BASELINE_ID
    if not baseline_dir.exists():
        result["F1_status"] = "baseline_missing; no paired SNR conclusion"
    else:
        baseline_config = json.loads((baseline_dir / "config_resolved.json").read_text(encoding="utf-8"))["config"]
        baseline_path = baseline_dir / "metrics_raw.csv"
        baseline = pd.read_csv(baseline_path)
        baseline = baseline[np.isclose(baseline.array_reference_snr_db, 55)]
        result["baseline"] = {"run_id": BASELINE_ID, "raw_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest()}
        if not _config_comparable(config, baseline_config):
            result["F1_status"] = "config_mismatch; SNR-only comparison withheld"
        else:
            for method in ("grid_omp_3d", "yang_ogols_3d"):
                old = baseline[(baseline.schedule_family == "frequency_single_shot") & (baseline.estimator == method)].set_index("scene_id").sort_index()
                new = high[(high.schedule_family == "frequency_single_shot") & (high.estimator == method)].set_index("scene_id").sort_index()
                truth = [c for c in old if c.startswith("true_path_")]
                if len(new) != 5 or not old.index.equals(new.index) or not old.index.is_unique or not np.allclose(old[truth], new[truth], rtol=0, atol=1e-12):
                    raise ValueError("SNR comparison scene/truth pairing failed")
                if not np.allclose(new.effective_output_snr_db - old.effective_output_snr_db, 10, atol=1e-8):
                    raise ValueError("SNR comparison output SNR did not increase by 10 dB")
                delta = new.azimuth_rmse_deg - old.azimuth_rmse_deg
                old_stats, new_stats = _azimuth_stats(old, config), _azimuth_stats(new, config)
                improved = int((delta < 0).sum()) >= 4 and float(delta.median()) < 0
                recovered = new_stats["all_azimuths_within_tolerance_scenes"] >= 4
                result["F1"].append({
                    "estimator": method, "55db": old_stats, "65db": new_stats,
                    "improved_scenes": int((delta < 0).sum()), "median_paired_rmse_delta_deg": float(delta.median()),
                    "paired_rmse_delta_deg": {str(k): float(v) for k, v in delta.items()},
                    "azimuth_improved": improved, "azimuth_recovered": recovered,
                    "judgment": "方位恢复" if recovered else ("方位有改善但未恢复" if improved else "未见一致方位改善"),
                })
    for method, block in high[high.schedule_family == "tensor_structured"].groupby("estimator", sort=False):
        if not all(f"{name}_db" in block for name in DOMAINS):
            result["F2"].append({"estimator": str(method), "judgment": "缺少分域指标，不能判断"})
            continue
        good = block.pilot_nmse_db <= -5
        fullgap = block.fullband_128_nmse_db - block.pilot_nmse_db
        heldgap = block.heldout_112_nmse_db - block.pilot_nmse_db
        pattern = good & ((fullgap >= 6) | (heldgap >= 6))
        result["F2"].append({
            "estimator": str(method), "pattern_scenes": int(pattern.sum()), "scenes": len(block),
            "median_fullband_minus_pilot_db": float(fullgap.median()),
            "median_heldout_minus_pilot_db": float(heldgap.median()),
            "per_scene": [{"scene_id": str(row.scene_id), **{f"{d}_db": float(getattr(row, f"{d}_db")) for d in DOMAINS}} for row in block.itertuples()],
            "judgment": "出现导频域好、全带或未观测域差" if len(block) == 5 and pattern.sum() >= 4 else "未达到预设分域失配判据，需查看逐场景差值",
        })
    return result


def diagnostic_report_lines(compact):
    diagnostic = compact["bounded_diagnostics"]
    if str(diagnostic.get("status", "")).startswith("adaptive_search_diagnostic"):
        return [
            "",
            "## 固定扇区历史诊断",
            "",
            "本次为波束中心自适应局部网格运行，不套用旧五场景 55→65 dB 固定扇区判据。",
            "",
        ]
    lines = ["", "## 分域 CSI 与 55→65 dB 固定扇区诊断", "",
             "pilot 指导频频点上的阵元 CSI；observation NMSE 是 DMA 合并后的观测误差。旧 channel_nmse 仍为原 16 点口径。F1 未观测域不适用，留空。", ""]
    for row in compact["domain_summary"]:
        values = "，".join(f"{name}={row[f'mean_{name}_db']:.3f} dB" if row[f"mean_{name}_db"] is not None else f"{name}=不适用" for name in DOMAINS)
        lines.append(f"- {row['schedule']} / {row['snr_db']:g} dB / {row['estimator']}：{values}。")
    lines += ["", f"诊断状态：`{diagnostic['status']}`。"]
    if "F1_status" in diagnostic:
        lines.append(f"F1：`{diagnostic['F1_status']}`。")
    for row in diagnostic["F1"]:
        lines.append(f"- F1 {row['estimator']}：{row['judgment']}；配对 RMSE 中位差 {row['median_paired_rmse_delta_deg']:.3f}°，改善 {row['improved_scenes']}/5。")
        for snr in ("55db", "65db"):
            s = row[snr]
            lines.append(f"  {snr}：方位 RMSE 中位数 {s['median_azimuth_rmse_deg']:.3f}°，三径方位达标 {s['all_azimuths_within_tolerance_scenes']}/5，三径联合成功 {s['all_paths_success_scenes']}/5，方位下边界 {s['azimuth_lower_boundary_paths']}/15。")
    for row in diagnostic["F2"]:
        lines.append(f"- F2 {row['estimator']}：{row['judgment']}；匹配场景 {row.get('pattern_scenes', 'NA')}/5。")
    lines += ["", "判据：F1 至少 4/5 场景方位 RMSE 降低且配对中位差<0 记为改善；至少 4/5 场景三径方位均在原 3° 容差内记为方位恢复，三径联合成功另列。F2 导频 NMSE≤−5 dB，且全带或未观测域相对导频差≥6 dB，至少 4/5 场景满足才标记该现象。阈值仅用于本次描述性诊断。", "", "分域失配支持距离混叠解释；唯一归因于导频排布仍需下一阶段固定非均匀导频对照。状态设计、导频重排、扩大 Monte Carlo 和新论文方法均等待读取结果后再决定。", ""]
    return lines
