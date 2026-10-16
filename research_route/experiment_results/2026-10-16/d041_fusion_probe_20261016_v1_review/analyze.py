"""Paired screening contrasts and descriptive figures; never selects test parameters."""
import numpy as np
import pandas as pd

from audit import OUT, folder, read
from _run_support import write_json
from thz_dma.metrics.statistics import bootstrap_interval

LABELS = ["re8192_J64_K128_distinct", "re4096_J64_K64_distinct", "re4096_J128_K32_distinct",
          "re4096_J128_K32_interleaved", "re128_J16_K8_distinct"]
SHORT = ["8192 RE\n64 x 128", "4096 RE\n64 x 64", "4096 RE\n128 x 32", "4096 RE\n128 x 32\ninterleaved", "128 RE\n16 x 8"]
METHODS = ["zhou", "scalar_fusion", "pinv_ls"]
COLORS = ["#0072B2", "#D55E00", "#999999"]


def main():
    table = pd.read_csv(folder("screen")/"metrics_raw.csv")
    powers = read(folder("screen")/"training_reference_power.json")
    assert powers == read(folder("tune")/"training_reference_power.json")
    operators = {r["label"]: r for r in read(folder("screen")/"operators.json")}
    assert operators == {r["label"]: r for r in read(folder("tune")/"operators.json")}
    for label in LABELS:
        budget = operators[label]["budget_re"]
        previous = folder("screen").parent/f"d041_budget_sweep_20261015_v1_re{budget}_validation"
        old = {r["label"]: r for r in read(previous/"operators.json")}[label]
        for name in ("weights_sha256", "active_mask_sha256", "state_indices", "rank", "pilot_re", "unique_dma_configurations"):
            assert operators[label][name] == old[name]
        old_power = read(previous/"observation_reference.json")["reference"][label]
        assert powers[label]["training_scenes"] == old_power["training_scenes"] == 2048
        np.testing.assert_allclose([powers[label][k] for k in ("normalized_whitened_power", "physical_whitened_power")],
            [old_power[k] for k in ("normalized_whitened_power", "physical_whitened_power")], rtol=2e-9)
    write_json(OUT/"legacy_metadata.json", {"passed": True, "matched_operators": 5, "matched_training_power_calibrations": 5,
        "training_scenes_per_operator": 2048, "tuning_screening_hardware_and_power_identical": True})
    table.loc[(table.method == "zhou") & (table.condition == "snr_20")].groupby("operator")[[
        "actual_whitened_snr_db", "realized_whitened_snr_db"]].agg(["mean", "min", "max"]).to_csv(OUT/"snr_summary.csv")
    noiseless = table.loc[table.condition.str.startswith("noiseless_")].sort_values("nmse_linear").groupby(
        ["operator", "method", "scene_index"], sort=False).tail(1).copy()
    noiseless["condition"] = "noiseless"
    collapsed = pd.concat([noiseless, table.loc[table.condition == "snr_20"]], ignore_index=True)
    collapsed.to_csv(OUT/"screen_worst_variance_rows.csv", index=False)
    metrics = ["row_space_nmse_linear", "null_space_nmse_linear", "observation_residual_relative",
               "clean_observation_residual_relative", "prediction_energy_ratio", "normalized_correlation"]
    rows, pairs = [], []
    rng = np.random.default_rng(2026101602)
    for (label, condition, method), block in collapsed.groupby(["operator", "condition", "method"]):
        assert len(block) == 64 and (block.status == "success").all()
        rows.append({"operator": label, "condition": condition, "method": method,
            "scenes": len(block), "mean_nmse_db": 10*np.log10(block.nmse_linear.mean()),
            "pass_minus10db_count": int((block.nmse_linear <= .1).sum()), **block[metrics].mean().to_dict()})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT/"screen_summary.csv", index=False)
    for (label, condition), block in collapsed.groupby(["operator", "condition"]):
        paired = block.pivot(index="scene_index", columns="method", values="nmse_linear")
        assert paired.shape == (64, 3) and not paired.isna().any().any()
        delta = 10*np.log10(paired.scalar_fusion/paired.zhou)
        ci = bootstrap_interval(delta, np.mean, rng=rng, resamples=10000)
        pairs.append({"operator": label, "condition": condition, "scenes": 64,
            "aggregate_delta_db": 10*np.log10(paired.scalar_fusion.mean()/paired.zhou.mean()),
            "mean_paired_delta_db": delta.mean(), "ci95_low": ci[0], "ci95_high": ci[1],
            "fusion_better_scenes": int((delta < 0).sum()), "maximum_scene_delta_db": delta.max(),
            "minimum_scene_delta_db": delta.min()})
    contrasts = pd.DataFrame(pairs)
    contrasts.to_csv(OUT/"paired_effects.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    for ax, condition, labels, names in [(axes[0], "noiseless", LABELS, SHORT),
                                        (axes[1], "snr_20", LABELS[:2], SHORT[:2])]:
        for i, method in enumerate(METHODS):
            values = [summary.loc[(summary.operator == label) & (summary.condition == condition) &
                                 (summary.method == method)].iloc[0].mean_nmse_db for label in labels]
            ax.bar(np.arange(len(labels))+(i-1)*.25, values, width=.24, color=COLORS[i],
                   label=["Zhou (validation-tuned)", "Scalar fusion", "Pseudoinverse LS"][i])
        ax.axhline(-10, color="black", linestyle="--", linewidth=.8)
        ax.set_xticks(range(len(labels)), names, fontsize=8)
        ax.set_ylim(-16, .6)
        ax.set_ylabel("Full CSI NMSE (dB)")
        ax.set_title("No observation noise" if condition == "noiseless" else "20 dB training-calibrated receiver SNR")
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8, loc="lower right")
    ordered = pd.concat([contrasts.loc[(contrasts.operator == label) & (contrasts.condition == condition)]
                         for condition, labels in [("noiseless", LABELS), ("snr_20", LABELS[:2])] for label in labels])
    y = np.arange(len(ordered))
    delta = ordered.mean_paired_delta_db.to_numpy()
    axes[2].errorbar(delta, y, xerr=[delta-ordered.ci95_low.to_numpy(), ordered.ci95_high.to_numpy()-delta],
                     fmt="o", color=COLORS[1], capsize=3)
    axes[2].axvline(0, color="black", linewidth=.8)
    axes[2].set_yticks(y, ["8192 / 64x128", "4096 / 64x64", "4096 / 128x32", "4096 / interleaved", "128 / 16x8",
                          "20 dB / 8192", "20 dB / 4096"], fontsize=8)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Fusion minus Zhou: paired NMSE (dB)")
    axes[2].set_title("64 scenes; descriptive bootstrap 95% intervals")
    axes[2].grid(axis="x", alpha=.2)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT/("screen_comparison."+suffix), dpi=180)
    plt.close(fig)
    shifts = pd.read_csv(OUT/"variance_shift.csv")
    profile = read(folder("calibrate")/"calibration_profile.json")["variances_real"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, label, title in zip(axes, LABELS[:2], ["8192 RE / 64x128", "4096 RE / 64x64"]):
        ax.semilogy(range(1, 101), profile, color="black", linestyle="--", label="Offline training MSE used as variance")
        for condition, style in [("noiseless_v1e-08", "-"), ("snr_20", ":")]:
            for i, method in enumerate(METHODS[:2]):
                block = shifts.loc[(shifts.operator == label) & (shifts.condition == condition) & (shifts.method == method)].sort_values("t")
                ax.semilogy(block.t, block.online_real_mse, style, marker="o", color=COLORS[i],
                    label=f'{"Zhou" if i == 0 else "Fusion"}, {"noiseless" if style == "-" else "20 dB"}')
        ax.invert_xaxis()
        ax.set_xlabel("Reverse-time index t")
        ax.set_ylabel("Mean squared error per real coordinate")
        ax.set_title(title)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT/("variance_alignment."+suffix), dpi=180)
    plt.close(fig)
    print(summary[["operator", "condition", "method", "mean_nmse_db", "pass_minus10db_count"]].to_string(index=False))
    print(contrasts.to_string(index=False))


if __name__ == "__main__":
    main()
