"""Audit D-041 pairing, frozen tuning, SNR and final trajectory endpoints."""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from _run_support import PROJECT_ROOT, hash_file, write_json
from run_zhou2025_repro import load_data, read_json, run_directory
from thz_dma.metrics.statistics import bootstrap_interval


def noiseless_diagnosis(frame, gate_db, sensitivity_db):
    """Read frozen test endpoints; never use the report to select parameters."""
    test = frame.loc[(frame.split == "test") & (frame.stage != "noisy")]
    controls = []
    for (band, operator, method), block in test.groupby(["band", "operator", "estimator"]):
        pivot = block.pivot(index="scene_id", columns="condition", values="nmse_db")
        worst = pivot.max(axis=1)
        fraction = float((worst <= gate_db).mean())
        controls.append({"band": band, "operator": operator, "estimator": method,
            "scenes": len(pivot), "gate_db": gate_db, "passed_fraction_at_both_variances": fraction,
            "all_scenes_passed": fraction == 1.,
            "worst_variance_mean_nmse_db": 10*math.log10(max(float((10**(worst/10)).mean()), 1e-30)),
            "maximum_variance_spread_db": float((pivot.max(axis=1)-pivot.min(axis=1)).max()),
            "mean_prediction_energy_ratio": float(block.prediction_energy_ratio.mean()),
            "mean_correlation": float(block.normalized_correlation.mean()),
            "mean_row_nmse": float(block.row_space_nmse_linear.mean()),
            "mean_null_nmse": float(block.null_space_nmse_linear.mean())})
    lookup = {(r["band"], r["operator"], r["estimator"]): r for r in controls}
    findings = []
    smoke = bool(frame.smoke_only.any())
    stable = all(r["maximum_variance_spread_db"] <= sensitivity_db
        for r in controls if r["estimator"] == "clean_ddpm")
    if smoke:
        findings.append({"code": "smoke_only", "interpretation":
            "模型仅作短训练连通性检查，不能据此判断DMA、宽带或交互导致正式学习恢复失败。"})
    elif not stable:
        findings.append({"code": "numerical_variance_sensitive", "interpretation":
            "数值似然方差改变无噪声结果，先处理数值敏感性，再作结构归因。"})
    else:
        full_count = int(frame.loc[frame.operator.str.startswith("dma_J"), "operator"].str.split("J").str[-1].astype(int).max())
        for band in sorted(test.band.unique()):
            digital = lookup[band, "digital", "clean_ddpm"]
            full = lookup[band, f"dma_J{full_count}", "clean_ddpm"]
            linear = lookup[band, f"dma_J{full_count}", "pinv_ls"]
            if digital["all_scenes_passed"] and not full["all_scenes_passed"] and linear["all_scenes_passed"]:
                findings.append({"code": "full_dma_recovery_failure", "band": band, "interpretation":
                    "全数字通过、全秩DMA的同观测LS通过而DDPM存在未过门槛案例：重点核对先验与在线引导，不能以该控制的信息缺失解释失败。"})
            reduced = [r["operator"] for r in controls if r["band"] == band and r["estimator"] == "clean_ddpm"
                and r["operator"].startswith("dma_J") and r["operator"] != f"dma_J{full_count}" and not r["all_scenes_passed"]]
            if full["all_scenes_passed"] and reduced:
                findings.append({"code": "configuration_reduction_failure", "band": band,
                    "operators": reduced, "interpretation":
                    "充分配置通过，减少配置后出现未过门槛案例；结合行/零空间误差继续区分先验补全不足与合法信道歧义。"})
        if {"narrow", "wide"} <= set(test.band.unique()):
            a, b, c, d = [lookup[band, op, "clean_ddpm"]["all_scenes_passed"]
                for band, op in (("narrow", "digital"), ("narrow", f"dma_J{full_count}"),
                                 ("wide", "digital"), ("wide", f"dma_J{full_count}"))]
            if a and b and not c:
                findings.append({"code": "wideband_digital_failure", "interpretation":
                    "窄带数字/DMA通过，宽带数字出现失败：先检查宽带先验、表示与恢复链路。"})
            if b and c and not d:
                findings.append({"code": "wideband_dma_joint_failure", "interpretation":
                    "窄带DMA和宽带数字通过，宽带DMA出现失败：支持检查跨频信道与频变合并的联合恢复条件；尚未隔离两者的唯一机制。"})
            if a and b and c and d:
                findings.append({"code": "full_observation_passed", "interpretation":
                    "四组充分观测均通过描述性门槛，后续重点在少配置与带噪条件；这不是等硬件或等资源的方案排名。"})
    if not findings:
        findings.append({"code": "mixed_or_incomplete_pattern", "interpretation":
            "当前控制未形成单一模式，需结合逐案例误差、去噪profile和恢复轨迹分析。"})
    return controls, {"smoke_only": smoke, "numerical_variance_stable": stable,
        "gate_db": gate_db, "candidate_findings": findings, "unique_causal_attribution": False,
        "rule": "report worst endpoint over both predeclared variances; no test-time tuning"}


def audit(prefix, scope="all"):
    if scope not in ("all", "noiseless", "narrow-noiseless"):
        raise ValueError("unknown audit scope")
    output_name = prefix if scope == "all" else f"{prefix}_{scope.replace('-', '_')}"
    output = PROJECT_ROOT/"results"/output_name
    if output.exists():
        raise FileExistsError("completed audit outputs are immutable; choose a new prefix")
    bands = (("narrow", "nb"),) if scope == "narrow-noiseless" else (("narrow", "nb"), ("wide", "wb"))
    data = {band: load_data(f"{prefix}_{tag}_data") for band, tag in bands}
    parameter_batches_paired = None
    if len(bands) == 2:
        left, right = data["narrow"], data["wide"]
        if left.counts != right.counts or left.metadata["split_seeds"] != right.metadata["split_seeds"]:
            raise ValueError("unpaired split counts/seeds")
        parameter_names = [name for name in left.metadata["file_sha256"] if name.startswith("parameters/")]
        if not parameter_names or any(left.metadata["file_sha256"][name] != right.metadata["file_sha256"].get(name)
                for name in parameter_names):
            raise ValueError("path geometry and complex gains are not byte-paired")
        parameter_batches_paired = len(parameter_names)
        with np.load(left.directory/"array.npz", allow_pickle=False) as a, np.load(right.directory/"array.npz", allow_pickle=False) as b:
            for field in ("element_positions_m", "feed_distances_m"):
                np.testing.assert_array_equal(a[field], b[field])
    inventories, frames, checks, training, profiles = [], [], {}, {}, {}
    thresholds = []
    stages = [("full-noiseless", "full"), ("reduced-noiseless", "reduced")]
    if scope == "all":
        stages.append(("noisy", "noisy"))
    for band, tag in bands:
        check_dir = run_directory(f"{prefix}_{tag}_check")
        if read_json(check_dir/"status.json")["status"] != "complete":
            raise ValueError("algebra checks must complete")
        checks[band] = read_json(check_dir/"checks.json")
        if not checks[band]["passed"]:
            raise ValueError("failed algebra/whitening check")
        train_dir = run_directory(f"{prefix}_{tag}_clean")
        train_config = read_json(train_dir/"config_resolved.json")["config"]
        expected_updates = train_config["training"]["epochs"]*math.ceil(data[band].counts["train"]/train_config["training"]["batch_size"])
        if (read_json(train_dir/"status.json")["status"] != "complete" or
                read_json(train_dir/"training_summary.json")["trained_steps"] != expected_updates):
            raise ValueError("training budget incomplete")
        training[band] = read_json(train_dir/"training_summary.json")
        profiles[band] = read_json(train_dir/"denoising_profile.json")
        thresholds.append((train_config["diagnostic"]["recovery_nmse_gate_db"],
                           train_config["diagnostic"]["variance_sensitivity_db"]))
        for stage, stage_tag in stages:
            val_dir = run_directory(f"{prefix}_{tag}_{stage_tag}_val")
            test_dir = run_directory(f"{prefix}_{tag}_{stage_tag}_test")
            tuning = read_json(val_dir/"tuning.json")
            if read_json(test_dir/"tuning.json") != tuning:
                raise ValueError("test changed frozen validation tuning")
            frozen = read_json(test_dir/"tuning_source.json")
            if frozen != {"run_id": val_dir.name, "sha256": hash_file(val_dir/"tuning.json")}:
                raise ValueError("frozen tuning source differs")
            for directory, split in ((val_dir, "validation"), (test_dir, "test")):
                if read_json(directory/"status.json")["status"] != "complete":
                    raise ValueError(f"incomplete evaluation: {directory.name}")
                config = read_json(directory/"config_resolved.json")["config"]
                if (config["band"] != band or config["job"]["stage"] != stage or config["job"]["split"] != split):
                    raise ValueError("evaluation band/stage/split differs from audit scope")
                summary = read_json(directory/"metrics_summary.json")
                frame = pd.read_csv(directory/"metrics_raw.csv")
                config_count = len(config["diagnostic"]["configuration_counts"])
                expected_operators = 2 if stage == "full-noiseless" else config_count-1 if stage == "reduced-noiseless" else config_count+1
                noise_count = len(config["evaluation_run"]["snr_db"]) if stage == "noisy" else len(config["diagnostic"]["noiseless_likelihood_variances"])
                expected_rows = expected_operators*noise_count*min(config["evaluation_run"][split+"_scenes"], data[band].counts[split])*3
                if summary["rows"] != expected_rows or len(frame) != expected_rows:
                    raise ValueError("evaluation row count differs from predeclared budget")
                if frame.duplicated(["operator", "condition", "scene_id", "estimator"]).any():
                    raise ValueError("duplicate diagnostic records")
                ref = read_json(directory/"observation_reference.json")["reference"]
                if any(r["training_scenes"] != data[band].counts["train"] for r in ref.values()):
                    raise ValueError("SNR reference did not use all training samples")
                if stage == "noisy":
                    measured = frame.likelihood_variance_normalized*10**(frame.target_whitened_snr_db/10)
                    expected = frame.operator.map({name: r["normalized_whitened_power"] for name, r in ref.items()})
                    np.testing.assert_allclose(measured, expected, rtol=1e-12, atol=1e-15)
                    np.testing.assert_allclose(frame.physical_element_noise_complex_variance,
                        frame.likelihood_variance_normalized*data[band].scale**2, rtol=1e-12, atol=1e-30)
                elif not ((frame.observation_noise_variance_normalized == 0).all()
                        and (frame.physical_element_noise_complex_variance == 0).all()):
                    raise ValueError("noiseless observations contain reported physical noise")
                if frame.error_decomposition_relative_gap.max() > 1e-5:
                    raise ValueError("row/null error decomposition failed")
                if not np.isfinite(frame[["nmse_linear", "prediction_energy_ratio", "normalized_correlation"]]).all().all():
                    raise ValueError("nonfinite primary metrics")
                terminal, trace_rows = [], 0
                for chunk in pd.read_csv(directory/"trajectories.csv", chunksize=10000):
                    trace_rows += len(chunk)
                    terminal.append(chunk.loc[(chunk.t == 0) & (chunk.phase == "updated"),
                        ["operator", "condition", "scene_id", "nmse_linear"]])
                final = pd.concat(terminal)
                ddpm = frame.query("estimator == 'clean_ddpm'")
                if trace_rows != len(ddpm)*(config["diffusion"]["steps"]+1) or len(final) != len(ddpm):
                    raise ValueError("trajectory is incomplete")
                merged = ddpm.merge(final, on=["operator", "condition", "scene_id"], validate="one_to_one", suffixes=("", "_trace"))
                np.testing.assert_allclose(merged.nmse_linear, merged.nmse_linear_trace, rtol=1e-12, atol=1e-15)
                dependencies = read_json(directory/"dependencies.json")
                if (dependencies["train_run_id"] != f"{prefix}_{tag}_clean" or dependencies["data_run_id"] != data[band].directory.name
                        or dependencies["dataset_sha256"] != hash_file(data[band].directory/"dataset.json")
                        or dependencies["checkpoint_sha256"] != hash_file(run_directory(dependencies["train_run_id"])/"checkpoint/training.pt")):
                    raise ValueError("data/model dependency changed")
                frame["run_id"] = directory.name
                frames.append(frame)
                inventories.append({"run_id": directory.name, "band": band, "stage": stage, "split": split,
                    "rows": len(frame), "trajectory_rows": trace_rows, "frozen_tuning": split == "test",
                    "variance_sensitivity": summary["numerical_variance_sensitivity"]})
    all_rows = pd.concat(frames, ignore_index=True)
    if len(set(thresholds)) != 1:
        raise ValueError("matched-band diagnostic gates differ")
    controls, diagnosis = noiseless_diagnosis(all_rows, *thresholds[0])
    rng = np.random.default_rng(2026101346)
    means, paired = [], []
    for key, block in all_rows.groupby(["band", "stage", "split", "operator", "condition", "estimator"]):
        values = block.nmse_linear.to_numpy()
        low, high = bootstrap_interval(values, np.mean, rng=rng, resamples=1000)
        means.append(dict(zip(("band", "stage", "split", "operator", "condition", "estimator"), key)) | {
            "scenes": len(block), "nmse_db": float(10*np.log10(max(values.mean(), 1e-30))),
            "nmse_ci95_low_db": 10*math.log10(max(low, 1e-30)), "nmse_ci95_high_db": 10*math.log10(max(high, 1e-30)),
            "mean_prediction_energy_ratio": float(block.prediction_energy_ratio.mean()),
            "mean_correlation": float(block.normalized_correlation.mean()),
            "mean_row_nmse": float(block.row_space_nmse_linear.mean()), "mean_null_nmse": float(block.null_space_nmse_linear.mean())})
    for key, block in all_rows.query("estimator == 'clean_ddpm' and split == 'test'").groupby(["stage", "operator", "condition"]):
        if len(bands) == 1:
            continue
        pivot = block.pivot(index="scene_id", columns="band", values="nmse_db")
        if pivot.isna().any().any():
            raise ValueError("unpaired cross-band test scenes")
        difference = (pivot.wide-pivot.narrow).to_numpy()
        paired.append({"stage": key[0], "operator": key[1], "condition": key[2], "scenes": len(pivot),
            "wide_minus_narrow_mean_db": float(difference.mean()),
            "paired_scene_ci95_db": bootstrap_interval(difference, np.mean, rng=rng, resamples=1000),
            "different_matched_priors": True, "single_training_seed": True})
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(means).to_csv(output/"group_summary.csv", index=False)
    pd.DataFrame(controls).to_csv(output/"noiseless_controls.csv", index=False)
    pd.DataFrame(inventories).drop(columns="variance_sensitivity").to_csv(output/"run_inventory.csv", index=False)
    result = {"passed": True, "prefix": prefix, "audit_scope": scope,
        "pairing_parameter_batches": parameter_batches_paired,
        "rows": len(all_rows), "trajectory_rows": sum(r["trajectory_rows"] for r in inventories),
        "smoke_only": bool(all_rows.smoke_only.all()), "diagnostic_only": True,
        "confidence_unit": "paired path scenes; not training-seed uncertainty", "checks": checks,
        "evaluations": inventories, "paired_band_differences": paired,
        "training": training, "denoising_profiles": profiles, "noiseless_diagnosis": diagnosis,
        "analysis_source_sha256": hash_file(Path(__file__)),
        "scope": "pairing/freeze/SNR/metric/trajectory audit; no unique causal claim from smoke runs"}
    write_json(output/"audit.json", result)
    (output/"README.md").write_text(
        f"# D-041 核验\n\n`{prefix}`：{len(all_rows)} 条最终指标、{result['trajectory_rows']} 条轨迹。"
        + ("两带宽路径参数与米制阵列配对通过。" if len(bands) == 2 else "本次只核验窄带，不要求宽带或带噪运行已存在。")
        + "训练集SNR标定、测试冻结和轨迹终点核对通过。\n\n"
        + ("本次是 4 次训练更新的小检查，不作为正式学习性能或唯一原因证据。\n" if result["smoke_only"] else
           "这是单训练种子的诊断对照；场景配对区间不表示训练种子不确定度。\n")
        + "\n[无噪声控制表](noiseless_controls.csv)按预声明门槛报告两数值方差下的最坏端点。\n\n"
        + "\n".join(f"- {finding['interpretation']}" for finding in diagnosis["candidate_findings"])
        + "\n\n上述模式只是候选排查方向，不作唯一归因，也不据测试结果重新选参。\n", encoding="utf-8")
    (output/"analysis_source.py").write_text(Path(__file__).read_text(), encoding="utf-8")
    print(f"audit passed: {len(all_rows)} metric rows, {result['trajectory_rows']} trajectory rows; {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--scope", choices=["all", "noiseless", "narrow-noiseless"], default="all")
    args = parser.parse_args()
    run_directory(args.prefix)  # validate before creating any output
    audit(args.prefix, args.scope)


if __name__ == "__main__":
    main()
