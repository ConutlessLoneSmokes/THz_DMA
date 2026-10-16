"""Evaluate fixed-RE budgets and matched DMA controls without changing old runners."""
import argparse
import copy
import math
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import torch

import run_d041_configuration_sweep as sweep
import run_d041_diagnostic as diagnostic
from _run_support import PROJECT_ROOT, RunArchive, create_run_archive, hash_file, read_toml, relative_path
from run_zhou2025_repro import read_json, run_directory
from thz_dma.metrics.statistics import bootstrap_interval
from thz_dma.observations.pilots import full_band_pilot_indices
from thz_dma.surfaces.multistrip import cyclic_frequency_resonance_states
from zhou2025_repro.configuration_sweep import ScheduledDmaOperator
from zhou2025_repro.d041 import DiagnosticDigitalOperator, SpectralDmaOperator, hardware_weights, paired_array, state_bank


def budget_mask(configurations, pilot_frequencies, *, interleaved=False):
    if (type(configurations) is not int or not 1 <= configurations <= 192 or
            type(pilot_frequencies) is not int or pilot_frequencies not in (8, 16, 32, 64, 128)):
        raise ValueError("J=1..192 and Kp=8/16/32/64/128 required")
    active = np.zeros((configurations, 128), dtype=bool)
    stride = 128//pilot_frequencies
    if interleaved and configurations % stride:
        raise ValueError("balanced full-frequency control requires J*Kp divisible by 128")
    for index in range(configurations):
        tones = (index % stride + stride*np.arange(pilot_frequencies) if interleaved else
                 full_band_pilot_indices(128, pilot_frequencies))
        active[index, tones] = True
    return active


def plans(config):
    settings = config["configuration_sweep"]
    budget = settings["budget_re"]
    bank = state_bank(config)
    if budget == 0:
        yield "digital", None, None, {"state_mode": "digital", "budget_re": 0, "is_main": False}
        yield "dma_J192", bank, None, {"configurations": 192, "state_mode": "distinct",
            "state_bank": "D041_random", "state_indices": list(range(192)), "budget_re": 0,
            "pilots_per_configuration": 128, "frequency_layout": "all128", "is_main": False}
        return
    frequency_counts = settings["pilot_frequency_counts"][:2] if config["job"]["smoke"] else settings["pilot_frequency_counts"]
    for tones in frequency_counts:
        count = budget//tones
        if budget % tones or count > settings["maximum_configurations"]:
            continue
        modes = ["distinct"] + (["repeat"] if count > 1 else [])
        if tones < 128 and count > 1:
            modes.append("interleaved")
        for mode in modes:
            indices = [0]*count if mode == "repeat" else list(range(count))
            yield f"re{budget}_J{count}_K{tones}_{mode}", bank[indices], budget_mask(
                count, tones, interleaved=mode == "interleaved"), {
                "configurations": count, "state_mode": mode, "state_bank": "D041_random",
                "state_indices": indices, "pilots_per_configuration": tones, "budget_re": budget,
                "frequency_layout": "balanced_interleaved_all128" if mode == "interleaved" else "repeated_regular_grid",
                "is_main": mode == "distinct"}
    if budget == 128:
        legacy = copy.deepcopy(config)
        legacy["configuration_sweep"].update(experiment="re128", re128_configuration_counts=[1, 2] if config["job"]["smoke"] else [1, 2, 4, 8, 16])
        for label, states, active, metadata in sweep.plans(legacy):
            yield "legacy_"+label, states, active, {**metadata, "budget_re": 128, "is_main": False}


def operator_groups(config, stage, *, dtype=torch.complex64):
    array = paired_array(config)
    shape = (len(array.frequencies), *array.positions.shape[:2])
    banks = {"D041_random": state_bank(config),
        "cyclic16_C2_origin": cyclic_frequency_resonance_states(16, 1, 160,
            config["system"]["carrier_frequency_hz"], config["dma"]["tuning_bandwidth_hz"])}
    weights_by_bank = {name: hardware_weights(config, array, bank) for name, bank in banks.items()}
    selected = config["job"].get("frozen_operator_labels")
    for label, bank, active, metadata in plans(config):
        if selected is not None and label not in selected:
            continue
        if bank is None:
            op, weights = DiagnosticDigitalOperator(shape, device=config["device"]), None
        else:
            weights = weights_by_bank[metadata["state_bank"]][metadata["state_indices"]]
            kwargs = {"rf_noise_fraction": config["noise"]["rf_noise_variance_fraction_of_element_noise"],
                "rtol": config["diagnostic"]["svd_relative_tolerance"], "device": config["device"], "dtype": dtype}
            op = (SpectralDmaOperator(weights, shape[1], **kwargs) if active is None else
                  ScheduledDmaOperator(weights, shape[1], active, **kwargs))
        op.sweep_metadata = {**metadata,
            "switch_guard_per_transition": config["acquisition"]["switch_guard_symbol_equivalents"]}
        yield label, op, weights


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


def contrast_pairs(operators):
    labels = {record["label"] for record in operators}
    for record in operators:
        label, mode = record["label"], record["state_mode"]
        control = label.replace("_interleaved", "_distinct") if mode == "interleaved" else label.replace("_distinct", "_repeat")
        if mode in ("distinct", "interleaved") and control != label and control in labels:
            yield label, control, "coverage" if mode == "interleaved" else "state_diversity"
        if label.startswith("legacy_"):
            control = label.removeprefix("legacy_")
            if control in labels:
                yield control, label, "random_minus_legacy_bank"


def report(archive, config):
    frame = pd.read_csv(archive.directory/"metrics_raw.csv")
    operators = read_json(archive.directory/"operators.json")
    reporting = {**config, "frozen_tuning": read_json(archive.directory/"tuning.json")}
    summary = sweep.summarize(frame, operators, reporting)
    metadata = pd.DataFrame([{key: record.get(key) for key in ("label", "budget_re", "pilots_per_configuration",
        "state_bank", "frequency_layout", "observed_frequency_count", "is_main", "training_symbols")}
        for record in operators]).rename(columns={"label": "operator"})
    summary = summary.merge(metadata, on="operator", validate="one_to_one")
    summary.to_csv(archive.directory/"sweep_summary.csv", index=False)
    worst = frame.loc[frame.groupby(["operator", "estimator", "scene_id"]).nmse_linear.idxmax()]
    estimator_rows = []
    for (label, estimator), block in worst.groupby(["operator", "estimator"]):
        estimator_rows.append({"operator": label, "estimator": estimator, "scenes": len(block),
            "worst_variance_mean_nmse_db": 10*math.log10(max(float(block.nmse_linear.mean()), 1e-30)),
            "recovery_scene_fraction": float((block.nmse_db <= config["configuration_sweep"]["recovery_nmse_db"]).mean()),
            **{key: float(block[key].mean()) for key in ("observation_residual_relative", "prediction_energy_ratio",
                "normalized_correlation", "row_space_nmse_linear", "null_space_nmse_linear")}})
    pd.DataFrame(estimator_rows).merge(metadata, on="operator", validate="many_to_one").to_csv(
        archive.directory/"estimator_summary.csv", index=False)
    ddpm = worst.loc[worst.estimator == "clean_ddpm"].pivot(index="scene_id", columns="operator", values="nmse_linear")
    contrasts = []
    for label, control, kind in contrast_pairs(operators):
        if ddpm[[label, control]].isna().any().any():
            raise ValueError("contrasts require paired scenes")
        delta = 10*np.log10(ddpm[label].clip(lower=1e-30)/ddpm[control].clip(lower=1e-30))
        low, high = bootstrap_interval(delta.to_numpy(), np.mean,
            rng=np.random.default_rng(config["evaluation_run"]["bootstrap_seed"]), resamples=2000)
        contrasts.append({"contrast": kind, "operator": label, "control": control,
            "paired_scene_mean_delta_db": float(delta.mean()), "ci95_low_db": low, "ci95_high_db": high,
            "mean_nmse_delta_db": 10*math.log10(float(ddpm[label].mean()/ddpm[control].mean())),
            "negative_favors_operator": True})
    pd.DataFrame(contrasts, columns=["contrast", "operator", "control", "paired_scene_mean_delta_db",
        "ci95_low_db", "ci95_high_db", "mean_nmse_delta_db", "negative_favors_operator"]).to_csv(
        archive.directory/"paired_contrasts.csv", index=False)
    if config["job"]["split"] == "validation":
        archive.write_json("selection.json", {"operator_labels": [r["label"] for r in operators],
            "selection_split": "validation", "budget_re": config["configuration_sweep"]["budget_re"],
            "scope": "all predeclared controls; guidance selected on validation; no test reselection",
            "smoke_only": config["job"]["smoke"]})
    else:
        archive.write_json("validation_selection.json", read_json(
            run_directory(config["job"]["tuning_run_id"])/"selection.json"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for (bank, mode), block in summary.groupby(["state_bank", "state_mode"], dropna=False):
        if block.budget_re.iloc[0] == 0:
            continue
        block = block.sort_values("configurations")
        label = f"{bank} / {mode}"
        axes[0].plot(block.configurations, block.worst_variance_mean_nmse_db, ".-", label=label)
        axes[1].plot(block.configurations, block.null_space_nmse_linear, ".-", label=label)
    for ax in axes:
        ax.set_xlabel("DMA configuration uses J")
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=.25)
        if ax.lines:
            ax.legend(fontsize=7)
    axes[0].axhline(config["configuration_sweep"]["recovery_nmse_db"], color="gray", linestyle="--")
    axes[0].set_ylabel("CSI NMSE (dB), worst numerical variance")
    axes[1].set_ylabel("Null-space error / truth energy")
    fig.savefig(archive.directory/"budget_controls.png", dpi=160)
    plt.close(fig)
    (archive.directory/"README.md").write_text(
        f"# {archive.run_id}\n\n预算={config['configuration_sweep']['budget_re']}，0表示非等资源参照。\n\n"
        "sweep_summary.csv为DDPM最坏数值方差指标，estimator_summary.csv包含三种估计器。\n"
        "paired_contrasts.csv给出状态、覆盖及128 RE状态库的配对差值。\n"
        "oracle_geometry_ls读取额外真实几何；参照组不作等资源排名。\n"
        "复用既有测试场景；smoke只验证连通性，不能作为性能结论。\n", encoding="utf-8")


def resolved_settings(args):
    config = copy.deepcopy(read_json(run_directory(args.data_run_id)/"config_resolved.json")["config"])
    if config["band"] != "wide" or bool(config["job"]["smoke"]) != args.smoke:
        raise ValueError("matched wideband D-041 data/model and matching smoke flag required")
    settings = read_toml(args.config)
    if (settings["pilot_re_budgets"] != [128, 512, 1024, 2048, 4096, 8192] or
            settings["pilot_frequency_counts"] != [128, 64, 32, 16, 8] or settings["maximum_configurations"] != 192 or
            any(type(n) is not int for n in [*settings["pilot_re_budgets"], *settings["pilot_frequency_counts"]]) or
            args.budget_re not in [0, *settings["pilot_re_budgets"]] or
            not math.isfinite(settings["recovery_nmse_db"]) or not 0 < settings["required_scene_fraction"] <= 1 or
            not math.isfinite(settings["near_reference_loss_db"]) or settings["near_reference_loss_db"] < 0):
        raise ValueError("invalid predeclared budgets, frequency counts or recovery gate")
    if args.smoke:
        config["evaluation_run"].update(validation_scenes=1, test_scenes=1, guidance=[.5, 1.])
    config["configuration_sweep"] = {**settings, "experiment": "budget", "budget_re": args.budget_re}
    config["device"] = args.device
    config["job"] = {**vars(args), "config": args.config.relative_to(PROJECT_ROOT).as_posix(),
        "stage": "full-noiseless", "sure_variant": "corrected"}
    if args.split == "test":
        prior = run_directory(args.tuning_run_id)
        selection = read_json(prior/"selection.json")
        expected = [label for label, *_ in plans(config)]
        if selection["operator_labels"] != expected or selection["budget_re"] != args.budget_re:
            raise ValueError("frozen validation operators/budget differ")
        config["job"].update(frozen_operator_labels=expected, selection_sha256=hash_file(prior/"selection.json"))
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT/"configs/d041_budget_sweep.toml")
    parser.add_argument("--budget-re", type=int, choices=[0, 128, 512, 1024, 2048, 4096, 8192], required=True,
        help="0 evaluates full-digital/J192 references; positive values are fixed pilot-RE budgets")
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
    if args.resume and directory.exists():
        archive = RunArchive(args.run_id, directory)
        previous = read_json(directory/"config_resolved.json")["config"]
        previous["job"]["resume"] = True
        if previous != config:
            raise ValueError("resume configuration/selection differs")
        for item in read_json(directory/"code_manifest.json"):
            if hash_file(PROJECT_ROOT/relative_path(item["path"])) != item["sha256"]:
                raise ValueError(f"resume source changed: {item['path']}")
        if (directory/"dependencies.json").exists() and read_json(directory/"dependencies.json") != diagnostic.evaluation_dependencies(config, diagnostic.matched_data(config)):
            raise ValueError("resume data/model dependencies changed")
        if read_json(archive.status_path)["status"] == "complete":
            print(f"Completed budget run retained: {args.run_id}", flush=True)
            return
        archive.write_status("running", resumed=True)
    else:
        archive = create_run_archive("d041_budget", args.run_id, config_path=args.config, config=config,
            script_path=__file__, seeds={"data": config["data"], "state": config["diagnostic"]["state_seed"],
                "posterior": config["evaluation_run"]["posterior_seed"]},
            environment={"torch": str(torch.__version__), "cuda": str(torch.version.cuda), "device": args.device},
            checkpoint_note="Budget/control evaluations only; existing data/models/results remain read-only.\n",
            extra_manifest_files=[PROJECT_ROOT/"scripts"/name for name in ["run_d041_configuration_sweep.py",
                "run_d041_diagnostic.py", "run_zhou2025_repro.py", "run_direction1_diffusion.py",
                "run_d041_budget_linux.sh", "analyze_d041_budget_sweep.py"]])
        with zipfile.ZipFile(directory/"source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as snapshot:
            for item in read_json(directory/"code_manifest.json"):
                snapshot.write(PROJECT_ROOT/relative_path(item["path"]), item["path"])
    diagnostic.operator_groups, diagnostic.operator_record = operator_groups, sweep.operator_record
    diagnostic.save_hardware = save_hardware
    diagnostic.signature = lambda c: {**sweep.ORIGINAL_SIGNATURE(c), "configuration_sweep": c["configuration_sweep"]}
    try:
        diagnostic.evaluate(archive, config)
        report(archive, config)
        archive.write_status("complete", budget_re=args.budget_re, split=args.split, smoke_only=args.smoke)
    except BaseException as error:
        archive.write_status("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
