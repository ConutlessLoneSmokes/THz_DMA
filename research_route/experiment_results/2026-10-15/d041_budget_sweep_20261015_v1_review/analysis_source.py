"""Recompute descriptive results from audited endpoints without changing frozen choices."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).parent
ROOT = OUT.parents[3]
PREFIX = "d041_budget_sweep_20261015_v1"
sys.path.insert(0, str(ROOT/"src"))
from thz_dma.metrics.statistics import bootstrap_interval

METRICS = ["observation_residual_relative", "prediction_energy_ratio", "normalized_correlation",
           "row_space_nmse_linear", "null_space_nmse_linear"]


def main():
    assert json.loads((OUT/"verification.json").read_text())["passed"]
    summaries, sensitivities, endpoints, comparisons = [], [], [], []
    for split in ("validation", "test"):
        saved = ROOT/"results/d041_budget_sweep"/f"{PREFIX}_{split}"
        expected = pd.read_csv(saved/"estimator_summary.csv").set_index(["operator", "estimator"])
        for budget in (0, 128, 512, 1024, 2048, 4096, 8192):
            tag = "references" if budget == 0 else f"re{budget}"
            folder = ROOT/"runs"/f"{PREFIX}_{tag}_{split}"
            frame = pd.read_csv(OUT/f"{tag}_{split}_endpoints.csv")
            operators = {r["label"]: r for r in json.loads((folder/"operators.json").read_text())}
            worst = frame.loc[frame.groupby(["operator", "estimator", "scene_id"]).nmse_linear.idxmax()].copy()
            for (label, estimator), block in worst.groupby(["operator", "estimator"]):
                record = operators[label]
                summary = {"split": split, "operator": label, "estimator": estimator,
                    "budget_re": budget, "configurations": record["configurations"], "rank": record["rank"],
                    "unique_states": record.get("unique_dma_configurations", 0),
                    "Kp": record.get("pilots_per_configuration", 0), "is_main": record.get("is_main", False),
                    "scenes": len(block), "nmse_db": float(10*np.log10(block.nmse_linear.mean())),
                    "pass_fraction": float((block.nmse_db <= -10).mean()),
                    **{key: float(block[key].mean()) for key in METRICS}}
                original = expected.loc[(label, estimator)]
                np.testing.assert_allclose([summary["nmse_db"], summary["pass_fraction"], *[summary[k] for k in METRICS]],
                    original[["worst_variance_mean_nmse_db", "recovery_scene_fraction", *METRICS]].to_numpy(dtype=float),
                    rtol=1e-9, atol=1e-12)
                summaries.append(summary)
            ddpm = worst.loc[worst.estimator == "clean_ddpm"].assign(budget_re=budget)
            endpoints.append(ddpm)
            for label, block in frame.loc[frame.estimator == "clean_ddpm"].groupby("operator"):
                values = block.pivot(index="scene_id", columns="condition", values="nmse_db")
                spread = values.max(axis=1)-values.min(axis=1)
                sensitivities.append({"split": split, "operator": label, "maximum_spread_db": float(spread.max()),
                                      "cases_over_0p1db": int((spread > .1).sum())})
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT/"endpoint_summary.csv", index=False)
    pd.DataFrame(sensitivities).to_csv(OUT/"variance_sensitivity.csv", index=False)
    endpoints = pd.concat(endpoints, ignore_index=True)
    endpoints.to_csv(OUT/"worst_variance_scene_endpoints.csv", index=False)
    assert len(summary) == 480 and len(endpoints) == 10560
    test = summary.loc[(summary.split == "test") & (summary.estimator == "clean_ddpm")]
    errors = endpoints.loc[endpoints.split == "test"].pivot(index="scene_id", columns="operator", values="nmse_linear")
    pairs = []
    for label in errors:
        if label.startswith("legacy_") or not label.startswith("re"):
            continue
        control = label.replace("_distinct", "_repeat").replace("_interleaved", "_distinct")
        if control != label and control in errors:
            pairs.append((label, control))
    pairs += [("re4096_J128_K32_distinct", "re4096_J64_K64_distinct"),
              ("re8192_J128_K64_distinct", "re8192_J64_K128_distinct")]
    for label, control in pairs:
        delta = 10*np.log10(errors[label]/errors[control])
        low, high = bootstrap_interval(delta.to_numpy(), np.mean, rng=np.random.default_rng(2026101546), resamples=10000)
        comparisons.append({"operator": label, "control": control, "mean_scene_delta_db": float(delta.mean()),
            "ci95_low_db": low, "ci95_high_db": high,
            "difference_of_mean_linear_nmse_db": float(10*np.log10(errors[label].mean()/errors[control].mean()))})
    pd.DataFrame(comparisons).to_csv(OUT/"paired_test_contrasts.csv", index=False)
    val = summary.loc[(summary.split == "validation") & (summary.estimator == "clean_ddpm") & summary.is_main]
    descriptive = val.loc[val.groupby("budget_re").nmse_db.idxmin()].copy()
    descriptive["test_nmse_db"] = descriptive.operator.map(test.set_index("operator").nmse_db)
    descriptive["test_pass_fraction"] = descriptive.operator.map(test.set_index("operator").pass_fraction)
    descriptive.to_csv(OUT/"per_budget_validation_best_mean.csv", index=False)
    labels = ["re8192_J128_K64_distinct", "re8192_J128_K64_interleaved"]
    tails = endpoints.loc[(endpoints.split == "test") & endpoints.operator.isin(labels)]
    tails.to_csv(OUT/"8192_frequency_coverage_scene_pairs.csv", index=False)
    # The full-tone designs reuse identical inputs and seeds from the preceding J scan.
    anchor_differences = []
    for split in ("validation", "test"):
        old = pd.read_csv(ROOT/"runs"/f"d041_config_sweep_20261014_v1_wb_curve_{split}/metrics_raw.csv")
        old = old.set_index(["operator", "condition", "scene_id", "estimator"])
        for count in (1, 4, 8, 16, 32, 64):
            if f"dma_J{count}" not in old.index.get_level_values(0):
                continue
            budget = 128*count
            new = pd.read_csv(OUT/f"re{budget}_{split}_endpoints.csv")
            new = new.loc[new.operator == f"re{budget}_J{count}_K128_distinct"]
            new = new.set_index(["condition", "scene_id", "estimator"]).sort_index()
            before = old.loc[f"dma_J{count}"].sort_index()
            delta = float(np.max(abs(new[["nmse_linear", *METRICS]].to_numpy()-before[["nmse_linear", *METRICS]].to_numpy())))
            np.testing.assert_allclose(new[["nmse_linear", *METRICS]], before[["nmse_linear", *METRICS]],
                                       rtol=5e-5, atol=1e-10)
            ddpm = new.index.get_level_values("estimator") == "clean_ddpm"
            anchor_differences.append({"split": split, "J": count, "maximum_metric_difference": delta,
                "maximum_ddpm_nmse_db_difference": float(abs(new.loc[ddpm, "nmse_db"]-before.loc[ddpm, "nmse_db"]).max())})
    pd.DataFrame(anchor_differences).to_csv(OUT/"previous_full_frequency_anchor_checks.csv", index=False)
    plot(test, errors)
    print(descriptive[["budget_re", "operator", "test_nmse_db", "test_pass_fraction"]].to_string(index=False))
    print("8192 main 128x64 failures:", tails.loc[(tails.operator == labels[0]) & (tails.nmse_db > -10),
        ["scene_index", "nmse_db", "prediction_energy_ratio", "normalized_correlation"]].to_json(orient="records"))
    print("Validated 480 estimator summaries and", len(anchor_differences), "numerically consistent full-frequency anchors.")


def plot(test, errors):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                        "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), constrained_layout=True)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#666666"]
    groups = sorted(test.loc[test.is_main].groupby("Kp"), key=lambda item: -item[0])
    for (tones, group), color, marker in zip(groups, colors, ["o", "s", "^", "D", "v"]):
        group = group.sort_values("budget_re")
        axes[0].plot(group.budget_re, group.nmse_db, marker=marker, color=color, label=f"Kp={int(tones)}", lw=1.5)
    axes[0].set(xscale="log", xlabel="Pilot budget R (RE)", ylabel="CSI NMSE (dB)",
                title="(a) 24 random-bank main designs", ylim=(-16.3, 1))
    axes[0].set_xticks([128, 512, 1024, 2048, 4096, 8192], ["128", "512", "1024", "2048", "4096", "8192"])
    axes[0].minorticks_off()
    axes[0].legend(loc="lower left", fontsize=9)
    selected = ["re8192_J64_K128_repeat", "re8192_J64_K128_distinct", "re8192_J128_K64_repeat",
                "re8192_J128_K64_distinct", "re8192_J128_K64_interleaved"]
    labels = ["64x128\nrepeat", "64x128\ndistinct", "128x64\nrepeat", "128x64\ndistinct", "128x64\ninterleaved"]
    rng = np.random.default_rng(20261015)
    for index, label in enumerate(selected):
        axes[1].scatter(index+rng.uniform(-.12, .12, len(errors)), 10*np.log10(errors[label]), s=10,
                        color="#888888", alpha=.4, edgecolors="none", label="Each scene" if index == 0 else None)
        value = test.set_index("operator").loc[label]
        axes[1].scatter(index, value.nmse_db, marker="D", s=45, color="#0072B2", zorder=3,
                        label="Linear mean -> dB" if index == 0 else None)
        axes[1].text(index, .55, f"{int(round(value.pass_fraction*100))}/100", ha="center", fontsize=9)
    axes[1].set(xticks=range(5), xticklabels=labels, ylim=(-20, 1.5), ylabel="CSI NMSE (dB)",
                title="(b) Same 8192 RE: states and coverage")
    axes[1].legend(loc="lower left", fontsize=9)
    for ax in axes:
        ax.axhline(-10, color="#333333", linestyle="--", lw=1)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("300 GHz / 30 GHz bandwidth / 8x160 DMA / zero physical noise\n"
                 "100 reused test scenes; each endpoint takes the worse of two numerical variances", fontsize=11)
    fig.savefig(OUT/"budget_review.png", dpi=220)
    fig.savefig(OUT/"budget_review.pdf")
    plt.close(fig)
    test.loc[test.is_main].to_csv(OUT/"figure_a_source.csv", index=False)
    errors[selected].reset_index().to_csv(OUT/"figure_b_source.csv", index=False)


if __name__ == "__main__":
    main()
