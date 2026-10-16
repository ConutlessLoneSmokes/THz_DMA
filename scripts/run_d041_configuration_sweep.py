"""D-041 integer-J and fixed-128-RE sweeps using existing matched Clean DDPMs."""
import argparse
import copy
import hashlib
import math
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import torch

import run_d041_diagnostic as diagnostic
from _run_support import PROJECT_ROOT, RunArchive, create_run_archive, hash_file, read_toml, relative_path
from run_zhou2025_repro import read_json, run_directory
from thz_dma.metrics.statistics import bootstrap_interval
from thz_dma.surfaces.multistrip import cyclic_frequency_resonance_states
from zhou2025_repro.configuration_sweep import ScheduledDmaOperator, re128_mask
from zhou2025_repro.d041 import DiagnosticDigitalOperator, SpectralDmaOperator, hardware_weights, paired_array, state_bank

ORIGINAL_SIGNATURE = diagnostic.signature
ORIGINAL_OPERATOR_RECORD = diagnostic.operator_record


def plans(config):
    sweep = config["configuration_sweep"]
    if sweep["experiment"] == "curve":
        yield "digital", None, None, {"state_mode": "digital"}
        bank = state_bank(config)
        for count in sweep["counts"]:
            yield f"dma_J{count}", bank[:count], None, {"configurations": count,
                "state_mode": "distinct", "state_bank": "D041_random", "state_indices": list(range(count))}
        for count in sweep["repeat_counts"]:
            yield f"repeat_J{count}", np.repeat(bank[:1], count, axis=0), None, {
                "configurations": count, "state_mode": "repeat", "state_bank": "D041_random", "state_indices": [0]*count}
    else:
        bank = cyclic_frequency_resonance_states(16, 1, 160, config["system"]["carrier_frequency_hz"],
            config["dma"]["tuning_bandwidth_hz"])
        for count in sweep["re128_configuration_counts"]:
            indices = (2+np.arange(count)*(16//count)) % 16
            for mode in (["distinct"] if count == 1 else ["distinct", "repeat", "interleaved"]):
                selected = np.repeat(indices[:1], count) if mode == "repeat" else indices
                label = f"re128_J{count}_K{128//count}_{mode}"
                yield label, bank[selected], re128_mask(count, interleaved=mode == "interleaved"), {
                    "configurations": count, "state_mode": mode, "state_bank": "cyclic16_C2_origin",
                    "state_indices": selected.tolist(), "pilots_per_configuration": 128//count,
                    "frequency_layout": "interleaved_all128" if mode == "interleaved" else "repeated_regular_grid"}


def operator_groups(config, stage, *, dtype=torch.complex64):
    array = paired_array(config)
    shape = (len(array.frequencies), *array.positions.shape[:2])
    selected = config["job"].get("frozen_operator_labels")
    master = (state_bank(config) if config["configuration_sweep"]["experiment"] == "curve" else
        cyclic_frequency_resonance_states(16, 1, 160, config["system"]["carrier_frequency_hz"],
            config["dma"]["tuning_bandwidth_hz"]))
    master_weights = hardware_weights(config, array, master)
    for label, bank, active, metadata in plans(config):
        if selected is not None and label not in selected:
            continue
        if bank is None:
            yield label, DiagnosticDigitalOperator(shape, device=config["device"]), None
            continue
        weights = master_weights[metadata["state_indices"]]
        kwargs = {"rf_noise_fraction": config["noise"]["rf_noise_variance_fraction_of_element_noise"],
            "rtol": config["diagnostic"]["svd_relative_tolerance"], "device": config["device"], "dtype": dtype}
        op = (SpectralDmaOperator(weights, shape[1], **kwargs) if active is None else
              ScheduledDmaOperator(weights, shape[1], active, **kwargs))
        op.sweep_metadata = {**metadata,
            "switch_guard_per_transition": config["acquisition"]["switch_guard_symbol_equivalents"]}
        yield label, op, weights


def operator_record(archive, label, operator, weights):
    if not isinstance(operator, ScheduledDmaOperator):
        result = ORIGINAL_OPERATOR_RECORD(archive, label, operator, weights)
    else:
        tones, strips, elements = operator.channel_shape
        s, ranks, keep = operator.singular_values.numpy(), operator.ranks.numpy(), operator.keep.numpy()
        conditions = [float(row[0]/row[mask][-1]) for row, mask in zip(s, keep) if mask.any()]
        active = operator.active
        count = weights.shape[0]
        np.savez(archive.directory/f"spectrum_{label}.npz", singular_values=s, ranks=ranks,
            active_re_mask=active, raw_row_energy=np.sum(abs(weights)**2, axis=-1),
            relative_noise_variance=operator.relative_variance.cpu().numpy())
        result = {"label": label, "frontend": "dma", "configurations": count,
            "complex_observations": int(active.sum())*strips, "complex_unknowns": tones*strips*elements,
            "rank": int(ranks.sum())*strips, "rank_per_frequency_min": int(ranks.min()),
            "rank_per_frequency_max": int(ranks.max()), "condition_max": max(conditions),
            "singular_value_min": float(s[keep].min()), "singular_value_max": float(s.max()),
            "pilot_re": int(active.sum()), "total_pilot_energy": int(active.sum()),
            "training_symbols": count, "switches": count-1,
            "weights_sha256": hashlib.sha256(np.ascontiguousarray(weights).tobytes()).hexdigest(),
            "active_mask_sha256": hashlib.sha256(active.tobytes()).hexdigest(),
            "pilot_indices": [np.flatnonzero(row).tolist() for row in active],
            "observed_frequency_count": int(active.any(axis=0).sum()),
            "spectrum_file": f"spectrum_{label}.npz", "svd_precision": "complex128 CPU",
            "same_weights_on_all_strips": True, "raw_weights_not_normalized": True}
    result.update(getattr(operator, "sweep_metadata", {"state_mode": "digital"}))
    result["unique_dma_configurations"] = (0 if weights is None else len(set(result["state_indices"])))
    result["state_switches"] = (0 if result["state_mode"] in ("repeat", "digital") else result["switches"])
    # The shared evaluator uses this field for the actual state-switch cost.
    result["switches"] = result["state_switches"]
    result["switch_guard_symbol_equivalents"] = result["switches"]*result.pop("switch_guard_per_transition", 0.)
    return result


def save_hardware(archive, config):
    array = paired_array(config)
    payload = {"frequencies_hz": array.frequencies, "element_positions_m": array.positions,
        "feed_distances_m": array.feed_distances, "D041_random_resonances_hz": state_bank(config),
        "cyclic16_resonances_hz": cyclic_frequency_resonance_states(16, 1, 160,
            config["system"]["carrier_frequency_hz"], config["dma"]["tuning_bandwidth_hz"])}
    for label, bank, active, metadata in plans(config):
        if bank is not None:
            payload[label+"_state_indices"] = metadata["state_indices"]
        if active is not None:
            payload[label+"_active_re_mask"] = active
    np.savez_compressed(archive.directory/"hardware.npz", **payload)


def summarize(frame, operators, config):
    ddpm = frame.loc[frame.estimator == "clean_ddpm"]
    if ddpm.duplicated(["operator", "condition", "scene_id"]).any():
        raise ValueError("duplicate DDPM endpoints")
    worst = ddpm.loc[ddpm.groupby(["operator", "scene_id"]).nmse_linear.idxmax()]
    sweep = config["configuration_sweep"]
    reference = worst.loc[worst.operator == "dma_J192"].set_index("scene_id").nmse_linear
    rows = []
    for record in operators:
        label = record["label"]
        block = worst.loc[worst.operator == label].set_index("scene_id")
        spread = ddpm.loc[ddpm.operator == label].groupby("scene_id").nmse_db.agg(lambda x: x.max()-x.min())
        row = {"operator": label, "configurations": record["configurations"], "state_mode": record["state_mode"],
            "pilot_re": record["pilot_re"], "complex_observations": record["complex_observations"],
            "rank": record["rank"], "switches": record["switches"], "scenes": len(block),
            "unique_dma_configurations": record.get("unique_dma_configurations"),
            "worst_variance_mean_nmse_db": 10*math.log10(max(float(block.nmse_linear.mean()), 1e-30)),
            "recovery_scene_fraction": float((block.nmse_db <= sweep["recovery_nmse_db"]).mean()),
            "maximum_variance_spread_db": float(spread.max()),
            "numerical_variance_stable": bool((spread <= config["diagnostic"]["variance_sensitivity_db"]).all()),
            "guidance_at_search_boundary": any(entry["selected"] in (entry["candidates"][0], entry["candidates"][-1])
                for key, entry in config.get("frozen_tuning", {}).items() if key.startswith(label+":"))}
        for key in ("observation_residual_relative", "prediction_energy_ratio", "normalized_correlation",
                    "row_space_nmse_linear", "null_space_nmse_linear"):
            row[key] = float(block[key].mean())
        if len(reference):
            aligned = reference.reindex(block.index)
            if aligned.isna().any():
                raise ValueError("unpaired reference scenes")
            losses = 10*np.log10(block.nmse_linear.clip(lower=1e-30)/aligned.clip(lower=1e-30))
            row["mean_nmse_loss_vs_J192_db"] = 10*math.log10(float(block.nmse_linear.mean()/aligned.mean()))
            row["near_reference_scene_fraction"] = float((losses <= sweep["near_reference_loss_db"]).mean())
            row["paired_mean_scene_loss_db"] = float(losses.mean())
            low, high = bootstrap_interval(losses.to_numpy(), np.mean,
                rng=np.random.default_rng(config["evaluation_run"]["bootstrap_seed"]), resamples=2000)
            row["paired_mean_scene_loss_ci95_low_db"], row["paired_mean_scene_loss_ci95_high_db"] = low, high
        rows.append(row)
    return pd.DataFrame(rows)


def select_test_operators(summary, config):
    sweep = config["configuration_sweep"]
    all_labels = summary.operator.tolist()
    if sweep["experiment"] != "curve":
        return {"operator_labels": all_labels, "selection_split": "validation", "minimum_recovery_J": None,
            "minimum_near_reference_J": None, "scope": "all predeclared fixed-RE controls"}
    distinct = summary.loc[summary.state_mode == "distinct"]
    recovered = distinct.recovery_scene_fraction >= sweep["required_scene_fraction"]
    near = ((distinct.mean_nmse_loss_vs_J192_db <= sweep["near_reference_loss_db"]) &
            (distinct.near_reference_scene_fraction >= sweep["required_scene_fraction"]))
    minimum_recovery = int(distinct.loc[recovered].configurations.min()) if recovered.any() else None
    minimum_near = int(distinct.loc[near].configurations.min()) if near.any() else None
    if config["job"]["smoke"]:
        minimum_recovery = minimum_near = None
    chosen = {"digital", "dma_J192", "dma_J160", "dma_J32", "dma_J8", "dma_J1", "repeat_J192", "repeat_J32", "repeat_J8"}
    for minimum in (minimum_recovery, minimum_near):
        if minimum is not None:
            chosen.update(f"dma_J{value}" for value in (minimum-1, minimum, minimum+1))
    return {"operator_labels": [label for label in all_labels if label in chosen], "selection_split": "validation",
        "minimum_recovery_J": minimum_recovery, "minimum_near_reference_J": minimum_near,
        "near_reference_loss_db": sweep["near_reference_loss_db"], "required_scene_fraction": sweep["required_scene_fraction"],
        "tested_J": sorted(distinct.configurations.tolist()), "smoke_only": config["job"]["smoke"],
        "scope": "minimum on validation for this bank/prior/guidance grid; test verifies frozen candidates"}


def report(archive, config):
    frame = pd.read_csv(archive.directory/"metrics_raw.csv")
    operators = read_json(archive.directory/"operators.json")
    reporting = {**config, "frozen_tuning": read_json(archive.directory/"tuning.json")}
    summary = summarize(frame, operators, reporting)
    summary.to_csv(archive.directory/"sweep_summary.csv", index=False)
    if config["job"]["split"] == "validation":
        archive.write_json("selection.json", select_test_operators(summary, config))
    else:
        selection = read_json(run_directory(config["job"]["tuning_run_id"])/"selection.json")
        archive.write_json("validation_selection.json", selection)
        confirmation = {}
        for criterion, key in (("recovery", "minimum_recovery_J"), ("near_reference", "minimum_near_reference_J")):
            minimum = selection[key]
            if minimum is not None:
                row = summary.set_index("operator").loc[f"dma_J{minimum}"]
                sweep = config["configuration_sweep"]
                passed = (row.recovery_scene_fraction >= sweep["required_scene_fraction"] if criterion == "recovery" else
                    row.mean_nmse_loss_vs_J192_db <= sweep["near_reference_loss_db"] and
                    row.near_reference_scene_fraction >= sweep["required_scene_fraction"])
                confirmation[criterion] = {"validation_selected_J": minimum, "test_passed": bool(passed),
                    "numerical_variance_stable": bool(row.numerical_variance_stable)}
        archive.write_json("test_confirmation.json", confirmation)
    worst = frame.loc[frame.estimator == "clean_ddpm"].groupby(["scene_id", "operator"]).nmse_linear.max().unstack()
    contrasts = []
    for record in operators:
        if record["state_mode"] == "digital":
            continue
        count, label = record["configurations"], record["label"]
        if config["configuration_sweep"]["experiment"] == "curve":
            control, name = f"repeat_J{count}", "distinct_minus_repeat"
            if record["state_mode"] != "distinct":
                continue
        else:
            control = label.replace("_distinct", "_repeat").replace("_interleaved", "_distinct")
            name = "interleaved_minus_repeated_frequency_grid" if record["state_mode"] == "interleaved" else "distinct_minus_repeat"
            if record["state_mode"] == "repeat" or count == 1:
                continue
        if control not in worst:
            continue
        loss = 10*np.log10(worst[label].clip(lower=1e-30)/worst[control].clip(lower=1e-30))
        low, high = bootstrap_interval(loss.to_numpy(), np.mean,
            rng=np.random.default_rng(config["evaluation_run"]["bootstrap_seed"]), resamples=2000)
        contrasts.append({"contrast": name, "configurations": count, "operator": label, "control": control,
            "paired_scene_mean_delta_db": float(loss.mean()), "ci95_low_db": low, "ci95_high_db": high,
            "mean_nmse_delta_db": 10*math.log10(float(worst[label].mean()/worst[control].mean())),
            "negative_favors_operator": True})
    pd.DataFrame(contrasts).to_csv(archive.directory/"paired_contrasts.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for mode, block in summary.groupby("state_mode"):
        block = block.sort_values("configurations")
        if mode == "digital":
            axes[0].axhline(block.worst_variance_mean_nmse_db.iloc[0], color="gray", linestyle="--", label="digital")
            continue
        axes[0].plot(block.configurations, block.worst_variance_mean_nmse_db, ".-", label=mode)
        axes[1].plot(block.configurations, block.null_space_nmse_linear, ".-", label=mode)
    for ax in axes:
        ax.set_xlabel("DMA configurations J")
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=.25)
        ax.legend()
    axes[0].set_ylabel("CSI NMSE (dB), worst numerical variance")
    axes[1].set_ylabel("Null-space error / truth energy")
    fig.savefig(archive.directory/"configuration_curve.png", dpi=160)
    plt.close(fig)
    (archive.directory/"README.md").write_text(
        f"# {archive.run_id}\n\n{config['band']} / {config['configuration_sweep']['experiment']} / {config['job']['split']}\n\n"
        "sweep_summary.csv给出最坏数值方差下的最终指标；configuration_curve.png为配置曲线。\n"
        "selection.json只由validation生成；test只验证冻结的配置与引导，不重新寻找最小J。\n"
        "最小J限于本状态库、匹配先验和引导候选；数值敏感及边界选参需结合结果报告。\n"
        "重用此前测试场景，不能称为新的独立测试集。smoke结果不作性能结论。\n", encoding="utf-8")


def resolved_settings(args):
    config = copy.deepcopy(read_json(run_directory(args.data_run_id)/"config_resolved.json")["config"])
    if config["band"] != args.band or bool(config["job"]["smoke"]) != args.smoke:
        raise ValueError("band/smoke must match the reused D-041 data and model")
    if args.experiment == "re128" and args.band != "wide":
        raise ValueError("RE128 frequency/configuration control requires wideband K=128")
    sweep = read_toml(args.config)
    if (sweep["configuration_min"] != 1 or sweep["configuration_max"] != 192 or
            sweep["re128_configuration_counts"] != [1, 2, 4, 8, 16] or
            any(type(n) is not int or not 2 <= n <= 192 for n in sweep["repeat_configuration_counts"]) or
            len(set(sweep["repeat_configuration_counts"])) != len(sweep["repeat_configuration_counts"]) or
            not math.isfinite(sweep["recovery_nmse_db"]) or not math.isfinite(sweep["near_reference_loss_db"]) or
            sweep["near_reference_loss_db"] < 0 or not 0 < sweep["required_scene_fraction"] <= 1):
        raise ValueError("invalid predeclared sweep counts/gates")
    sweep.update(experiment=args.experiment, counts=list(range(192, 0, -1)),
        repeat_counts=sweep["repeat_configuration_counts"])
    if args.smoke:
        sweep.update(counts=[192, 160, 8, 1], repeat_counts=[8])
        config["evaluation_run"].update(validation_scenes=1, test_scenes=1, guidance=[.5, 1.])
    config["configuration_sweep"] = sweep
    config["device"] = args.device
    config["job"] = {**vars(args), "config": str(args.config.relative_to(PROJECT_ROOT)),
        "stage": "full-noiseless", "sure_variant": "corrected"}
    if args.split == "test":
        prior = run_directory(args.tuning_run_id)
        selection = read_json(prior/"selection.json")
        config["job"].update(frozen_operator_labels=selection["operator_labels"],
            selection_sha256=hash_file(prior/"selection.json"))
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT/"configs/d041_configuration_sweep.toml")
    parser.add_argument("--experiment", choices=["curve", "re128"], required=True)
    parser.add_argument("--band", choices=["narrow", "wide"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-run-id", required=True)
    parser.add_argument("--train-run-id", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--tuning-run-id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.config = args.config.resolve()
    if args.split == "test" and not args.tuning_run_id:
        parser.error("test requires --tuning-run-id")
    config = resolved_settings(args)
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; use --device cpu for a smoke check")
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    directory = run_directory(args.run_id)
    if args.resume:
        archive = RunArchive(args.run_id, directory)
        previous = read_json(directory/"config_resolved.json")["config"]
        previous["job"]["resume"] = True
        if previous != config:
            raise ValueError("resume configuration/selection differs")
        for item in read_json(directory/"code_manifest.json"):
            if hash_file(PROJECT_ROOT/relative_path(item["path"])) != item["sha256"]:
                raise ValueError(f"resume source changed: {item['path']}")
        if read_json(archive.status_path)["status"] == "complete":
            raise ValueError("completed runs are immutable; use a new run ID")
        archive.write_status("running", resumed=True)
    else:
        archive = create_run_archive("d041_sweep", args.run_id, config_path=args.config, config=config,
            script_path=__file__, seeds={"data": config["data"], "state": config["diagnostic"]["state_seed"],
                "posterior": config["evaluation_run"]["posterior_seed"]},
            environment={"torch": str(torch.__version__), "cuda": str(torch.version.cuda), "device": args.device},
            checkpoint_note="New acquisition evaluations only; reused data/models are read-only.\n",
            extra_manifest_files=[PROJECT_ROOT/"scripts/run_d041_diagnostic.py", PROJECT_ROOT/"scripts/run_zhou2025_repro.py",
                PROJECT_ROOT/"scripts/run_direction1_diffusion.py", PROJECT_ROOT/"scripts/run_d041_sweep_linux.sh"])
        with zipfile.ZipFile(directory/"source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as snapshot:
            for item in read_json(directory/"code_manifest.json"):
                snapshot.write(PROJECT_ROOT/relative_path(item["path"]), item["path"])
    # Bind acquisition hooks only in this separate runner; retain the original D-041 files.
    diagnostic.operator_groups, diagnostic.operator_record = operator_groups, operator_record
    diagnostic.save_hardware = save_hardware
    diagnostic.signature = lambda c: {**ORIGINAL_SIGNATURE(c), "configuration_sweep": c["configuration_sweep"]}
    try:
        diagnostic.evaluate(archive, config)
        report(archive, config)
        archive.write_status("complete", experiment=args.experiment, split=args.split, smoke_only=args.smoke)
    except BaseException as error:
        archive.write_status("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
