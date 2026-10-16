"""Combine completed budget blocks; freeze validation workpoints before test."""
import argparse
from datetime import datetime, timezone
import re
from pathlib import Path

import pandas as pd

from _run_support import PROJECT_ROOT, hash_file, read_toml, write_json
from run_zhou2025_repro import read_json, run_directory


def select_workpoint(summary, *, smoke, required_fraction):
    if smoke:
        return None
    eligible = summary.loc[(summary.is_main == True) & (summary.budget_re > 0) &
        (summary.recovery_scene_fraction >= required_fraction)]
    if eligible.empty:
        return None
    row = eligible.sort_values(["pilot_re", "training_symbols", "operator"]).iloc[0]
    return {"operator": row.operator, "pilot_re": int(row.pilot_re),
        "configurations": int(row.configurations), "pilot_frequencies": int(row.pilots_per_configuration),
        "validation_run_id": row.run_id}


def verify_report(directory):
    for name, digest in read_json(directory/"manifest.json").items():
        if hash_file(directory/name) != digest:
            raise ValueError(f"saved report changed: {directory/name}")
    inputs = read_json(directory/"inputs.json")
    for name, digest in inputs["files"].items():
        if hash_file(PROJECT_ROOT/name) != digest:
            raise ValueError(f"report input changed: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.prefix):
        parser.error("prefix must be a plain run identifier")
    settings = read_toml(PROJECT_ROOT/"configs/d041_budget_sweep.toml")
    inputs, summaries, estimators, contrasts, configs = {}, [], [], [], []
    for budget in [0, *settings["pilot_re_budgets"]]:
        tag = "references" if budget == 0 else f"re{budget}"
        run_id = f"{args.prefix}_{tag}_{args.split}"
        directory = run_directory(run_id)
        if read_json(directory/"status.json")["status"] != "complete":
            raise ValueError(f"incomplete budget run: {run_id}")
        config = read_json(directory/"config_resolved.json")["config"]
        if config["configuration_sweep"]["budget_re"] != budget or config["job"]["split"] != args.split:
            raise ValueError(f"budget/split differs: {run_id}")
        configs.append(config)
        names = ["status.json", "config_resolved.json", "dependencies.json", "tuning.json",
            "sweep_summary.csv", "estimator_summary.csv", "paired_contrasts.csv"]
        names += ["selection.json"] if args.split == "validation" else ["tuning_source.json", "validation_selection.json"]
        for name in names:
            p = directory/name
            inputs[p.relative_to(PROJECT_ROOT).as_posix()] = hash_file(p)
        summaries.append(pd.read_csv(directory/"sweep_summary.csv").assign(run_id=run_id))
        estimators.append(pd.read_csv(directory/"estimator_summary.csv").assign(run_id=run_id))
        block = pd.read_csv(directory/"paired_contrasts.csv")
        if not block.empty:
            contrasts.append(block.assign(run_id=run_id, budget_re=budget))
    base = configs[0]
    for config in configs[1:]:
        for key in ("band", "system", "channel", "dma", "noise", "atmosphere", "acquisition", "diffusion", "diagnostic", "evaluation_run", "device", "data"):
            if config[key] != base[key]:
                raise ValueError(f"cross-budget {key} differs")
        for key in ("data_run_id", "train_run_id", "smoke"):
            if config["job"][key] != base["job"][key]:
                raise ValueError(f"cross-budget source/{key} differs")
    for name in [Path(__file__), PROJECT_ROOT/"scripts/run_d041_budget_sweep.py",
            PROJECT_ROOT/"configs/d041_budget_sweep.toml"]:
        inputs[name.relative_to(PROJECT_ROOT).as_posix()] = hash_file(name)
    validation_directory = PROJECT_ROOT/"results/d041_budget_sweep"/f"{args.prefix}_validation"
    if args.split == "test":
        verify_report(validation_directory)
        for name in ["selection.json", "manifest.json"]:
            p = validation_directory/name
            inputs[p.relative_to(PROJECT_ROOT).as_posix()] = hash_file(p)
    receipt = {"files": inputs, "prefix": args.prefix, "split": args.split}
    directory = PROJECT_ROOT/"results/d041_budget_sweep"/f"{args.prefix}_{args.split}"
    if directory.exists():
        if not args.resume or read_json(directory/"inputs.json") != receipt:
            raise ValueError("report exists or inputs changed; use a new prefix")
        if (directory/"manifest.json").exists():
            verify_report(directory)
            print(f"Completed budget report retained: {directory}", flush=True)
            return
    directory.mkdir(parents=True, exist_ok=args.resume)
    write_json(directory/"inputs.json", receipt)
    summary = pd.concat(summaries, ignore_index=True)
    if summary.operator.duplicated().any():
        raise ValueError("duplicate operators across budget blocks")
    summary.to_csv(directory/"budget_summary.csv", index=False)
    pd.concat(estimators, ignore_index=True).to_csv(directory/"estimator_summary.csv", index=False)
    pd.concat(contrasts, ignore_index=True).to_csv(directory/"paired_contrasts.csv", index=False)
    if args.split == "validation":
        selection = {"selection_split": "validation", "created_utc": datetime.now(timezone.utc).isoformat(),
            "smoke_only": base["job"]["smoke"], "recovery_nmse_db": settings["recovery_nmse_db"],
            "required_scene_fraction": settings["required_scene_fraction"],
            "rule": "among main random-bank groups meeting the gate, minimum RE then minimum training symbols",
            "workpoint": select_workpoint(summary, smoke=base["job"]["smoke"], required_fraction=settings["required_scene_fraction"]),
            "test_scope": "all predeclared operators; no test-based resource selection",
            "numerical_stability_is_separate": True}
        write_json(directory/"selection.json", selection)
    else:
        selection = read_json(validation_directory/"selection.json")
        workpoint = selection["workpoint"]
        confirmation = {"validation_selection": selection, "reused_previous_test_scenes": True}
        if workpoint is not None:
            row = summary.set_index("operator").loc[workpoint["operator"]]
            confirmation.update(test_mean_nmse_db=float(row.worst_variance_mean_nmse_db),
                test_recovery_fraction=float(row.recovery_scene_fraction),
                test_passed=bool(row.recovery_scene_fraction >= selection["required_scene_fraction"]),
                numerical_variance_stable=bool(row.numerical_variance_stable))
        write_json(directory/"test_confirmation.json", confirmation)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for tones, block in summary.loc[summary.is_main == True].groupby("pilots_per_configuration"):
        block = block.sort_values("pilot_re")
        axes[0].plot(block.pilot_re, block.worst_variance_mean_nmse_db, ".-", label=f"Kp={int(tones)}")
        axes[1].plot(block.pilot_re, block.recovery_scene_fraction, ".-", label=f"Kp={int(tones)}")
    for ax in axes:
        ax.set_xlabel("Pilot resource elements")
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=.25)
        ax.legend()
    axes[0].axhline(settings["recovery_nmse_db"], color="gray", linestyle="--")
    axes[1].axhline(settings["required_scene_fraction"], color="gray", linestyle="--")
    axes[0].set_ylabel("CSI NMSE (dB), worst numerical variance")
    axes[1].set_ylabel("Fraction reaching recovery gate")
    axes[1].set_ylim(-.03, 1.03)
    fig.savefig(directory/"budget_performance.png", dpi=160)
    fig.savefig(directory/"budget_performance.pdf")
    plt.close(fig)
    (directory/"README.md").write_text(
        f"# {args.prefix} / {args.split}\n\n"
        "budget_summary.csv含全部DDPM配置/状态/覆盖控制；estimator_summary.csv含三种估计器。\n"
        "budget_performance.png只画共同随机库主组；对照曲线另存每个运行的budget_controls.png。\n"
        "工作点只从验证集选择，测试只核验冻结工作点。预算0的数字/J192是非等资源参照。\n"
        "零物理噪声；单位RE能量固定；测试复用旧场景。smoke不报告性能门槛。\n", encoding="utf-8")
    write_json(directory/"manifest.json", {p.name: hash_file(p) for p in sorted(directory.iterdir()) if p.is_file()})
    print(f"Saved budget report: {directory}; operators={len(summary)}", flush=True)


if __name__ == "__main__":
    main()
