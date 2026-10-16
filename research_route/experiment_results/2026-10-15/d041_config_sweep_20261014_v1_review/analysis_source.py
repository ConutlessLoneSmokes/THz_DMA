"""Independent endpoint summaries and plots; never select new test parameters."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).parent
PREFIX = "d041_config_sweep_20261014_v1"
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
from thz_dma.metrics.statistics import bootstrap_interval


def main():
    summaries, thresholds, sensitivity, trajectory = [], [], [], []
    raw_tests = []
    for folder in sorted((ROOT/"runs").glob(PREFIX+"_*")):
        config = json.loads((folder/"config_resolved.json").read_text())["config"]
        frame = pd.read_csv(folder/"metrics_raw.csv")
        operators = {r["label"]: r for r in json.loads((folder/"operators.json").read_text())}
        split, experiment, band = config["job"]["split"], config["job"]["experiment"], config["band"]
        worst = frame.loc[frame.groupby(["operator", "estimator", "scene_id"]).nmse_linear.idxmax()]
        for (label, method), block in worst.groupby(["operator", "estimator"]):
            summaries.append({"band": band, "experiment": experiment, "split": split, "operator": label,
                "estimator": method, "configurations": operators[label]["configurations"],
                "state_mode": operators[label]["state_mode"], "rank": operators[label]["rank"],
                "pilot_re": operators[label]["pilot_re"], "scenes": len(block),
                "nmse_db": 10*np.log10(max(float(block.nmse_linear.mean()), 1e-30)),
                "pass_fraction_minus10db": float((block.nmse_db <= -10).mean()),
                **{key: float(block[key].mean()) for key in ("observation_residual_relative", "prediction_energy_ratio",
                    "normalized_correlation", "row_space_nmse_linear", "null_space_nmse_linear")}})
        ddpm = frame.loc[frame.estimator == "clean_ddpm"]
        for label, block in ddpm.groupby("operator"):
            pivot = block.pivot(index="scene_id", columns="condition", values="nmse_db")
            spread = pivot.max(axis=1)-pivot.min(axis=1)
            sensitivity.append({"band": band, "experiment": experiment, "split": split, "operator": label,
                "scenes": len(pivot), "max_spread_db": float(spread.max()), "cases_over_0p1db": int((spread > .1).sum()),
                "recovery_pass_flips": int(((pivot <= -10).nunique(axis=1) > 1).sum()),
                "max_linear_nmse_difference": float((10**(pivot/10)).max(axis=1).sub((10**(pivot/10)).min(axis=1)).max())})
        if experiment == "curve" and split == "validation":
            errors = ddpm.groupby(["scene_id", "operator"]).nmse_linear.max().unstack()
            reference = errors.dma_J192
            eligible_recovery, eligible_near = [], []
            for count in range(1, 193):
                error = errors[f"dma_J{count}"]
                recovered = float((10*np.log10(error) <= -10).mean()) >= .95
                losses = 10*np.log10(error/reference)
                near = 10*np.log10(error.mean()/reference.mean()) <= .5 and float((losses <= .5).mean()) >= .95
                if recovered:
                    eligible_recovery.append(count)
                if near:
                    eligible_near.append(count)
            selected = json.loads((folder/"selection.json").read_text())
            assert selected["minimum_recovery_J"] == min(eligible_recovery)
            assert selected["minimum_near_reference_J"] == min(eligible_near)
            thresholds.append({"band": band, "minimum_recovery_J_validation": min(eligible_recovery),
                "minimum_near_reference_J_validation": min(eligible_near),
                "full_J_grid": len(errors.columns)-10, "recovery_resource_reduction_vs_J192": 1-min(eligible_recovery)/192,
                "near_reference_resource_reduction_vs_J192": 1-min(eligible_near)/192})
        if split == "test":
            raw_tests.append(frame.assign(experiment=experiment))
            columns = ["operator", "condition", "t", "nmse_linear", "observation_residual_relative", "prediction_energy_ratio",
                "normalized_correlation", "row_space_nmse_linear", "null_space_nmse_linear"]
            for chunk in pd.read_csv(folder/"trajectories.csv", usecols=columns, chunksize=50000):
                selected = chunk.loc[chunk.t.isin([100, 75, 50, 25, 10, 5, 1, 0])]
                for (label, condition, time), block in selected.groupby(["operator", "condition", "t"]):
                    trajectory.append({"band": band, "experiment": experiment, "operator": label, "condition": condition,
                        "t": time, "scenes": len(block), **{key: float(block[key].sum()) for key in columns[3:]}})
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT/"endpoint_summary.csv", index=False)
    pd.DataFrame(thresholds).to_csv(OUT/"validation_thresholds.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/"variance_sensitivity.csv", index=False)
    trace = pd.DataFrame(trajectory).groupby(["band", "experiment", "operator", "condition", "t"], as_index=False).sum()
    for key in columns[3:]:
        trace[key] /= trace.scenes
    trace["nmse_db"] = 10*np.log10(trace.nmse_linear.clip(lower=1e-30))
    trace.to_csv(OUT/"trajectory_summary.csv", index=False)
    assert len(thresholds) == 2 and (trace.scenes == 100).all()
    test = pd.concat(raw_tests)
    paired = []
    errors = test.loc[test.estimator == "clean_ddpm"].groupby(["band", "experiment", "scene_id", "operator"]).nmse_linear.max()
    for (band, experiment), block in errors.groupby(level=["band", "experiment"]):
        block = block.droplevel(["band", "experiment"]).unstack()
        for label in block:
            control = (label.replace("dma_", "repeat_") if label.startswith("dma_") else
                label.replace("_distinct", "_repeat").replace("_interleaved", "_distinct"))
            if control == label or control not in block or label.endswith("_repeat"):
                continue
            delta = 10*np.log10(block[label]/block[control])
            low, high = bootstrap_interval(delta.to_numpy(), np.mean, rng=np.random.default_rng(2026101346), resamples=10000)
            paired.append({"band": band, "experiment": experiment, "operator": label, "control": control,
                "mean_delta_db": float(delta.mean()), "ci95_low_db": low, "ci95_high_db": high,
                "difference_of_mean_linear_nmse_db": float(10*np.log10(block[label].mean()/block[control].mean()))})
    pd.DataFrame(paired).to_csv(OUT/"paired_test_contrasts.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    colors = {"narrow": "#0072B2", "wide": "#D55E00"}
    for band in ("narrow", "wide"):
        base = summary.loc[(summary.band == band) & (summary.experiment == "curve") &
            (summary.estimator == "clean_ddpm") & (summary.state_mode == "distinct")]
        validation = base.loc[base.split == "validation"].sort_values("configurations")
        frozen = base.loc[base.split == "test"]
        axes[0].plot(validation.configurations, validation.nmse_db, color=colors[band], label=band+" validation (32)")
        axes[0].scatter(frozen.configurations, frozen.nmse_db, color=colors[band], s=23,
            marker="o" if band == "narrow" else "x", label=band+" frozen test (100)")
    axes[0].set(xlabel="DMA configuration uses J; all tones, budget varies", ylabel="CSI NMSE (dB)",
        title="Integer-J scan; different matched priors")
    protocols = summary.loc[(summary.experiment == "re128") & (summary.split == "test") & (summary.estimator == "clean_ddpm")]
    for mode, marker in (("distinct", "o"), ("repeat", "s"), ("interleaved", "^")):
        block = protocols.loc[protocols.state_mode == mode].sort_values("configurations")
        axes[1].plot(block.configurations, block.nmse_db, marker=marker, label=mode)
    axes[1].set(xlabel="DMA configuration uses J; J x Kp = 128 RE", ylabel="CSI NMSE (dB)",
        title="Wideband frozen test: all 13 controls near 0 dB", ylim=(-10.8, 1), xscale="log")
    axes[1].set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    for ax in axes:
        ax.axhline(-10, color="gray", linestyle="--", linewidth=.8, label="-10 dB recovery gate")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(OUT/"configuration_overview.png", dpi=180)
    fig.savefig(OUT/"configuration_overview.pdf")
    plt.close(fig)
    print("independent summaries, thresholds, trajectories and paired intervals completed")


if __name__ == "__main__":
    main()
