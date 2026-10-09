"""Audit a completed SBL run and compare matched archived baselines; no estimation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _run_support import PROJECT_ROOT, hash_file, write_json
from thz_dma.metrics.statistics import bootstrap_interval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="stage2a_sbl_snr_20261002_v1")
    args = parser.parse_args()
    directory = PROJECT_ROOT / "runs" / args.run_id
    old = PROJECT_ROOT / "runs/stage2a_somp_snr_20260929_v1"
    linear = PROJECT_ROOT / "runs/stage2a_linear_20260929_v1"
    config = json.loads((directory / "config_resolved.json").read_text())["config"]
    frame = pd.read_csv(directory / "metrics_raw.csv")
    key = ["scene_id", "schedule", "array_reference_snr_db", "estimator"]
    assert json.loads((directory / "status.json").read_text())["status"] == "complete"
    assert len(frame) == 3600 and frame.scene_id.nunique() == 100
    assert not frame.duplicated(key).any()
    expected = {(s["name"], snr, method) for s in config["acquisition"]["schedules"]
                for snr in config["noise"]["array_reference_snr_db"]
                for method in config["experiment"]["estimators"]}
    assert all(set(g[key[1:]].itertuples(index=False, name=None)) == expected
               for _, g in frame.groupby("scene_id"))
    assert np.isfinite(frame.fullband_128_nmse_db).all()
    saved = [row for path in sorted((directory / "checkpoint").glob("*.json"))
             for row in json.loads(path.read_text(encoding="utf-8"))["rows"]]
    pd.testing.assert_frame_equal(frame.sort_values(key).reset_index(drop=True),
        pd.DataFrame(saved).sort_values(key).reset_index(drop=True).replace("", np.nan),
        check_dtype=False, rtol=1e-12, atol=1e-12)
    previous = pd.read_csv(old / "metrics_raw.csv")
    paired = frame[frame.estimator != "sbl_jointfreq_3d"].merge(previous, on=key,
        suffixes=("_new", "_old"), validate="one_to_one")
    assert len(paired) == 2400
    unchanged = {}
    for col in ("fullband_128_nmse_db", "effective_output_snr_db", "net_rate_bps_hz",
                "adaptive_candidate_count"):
        unchanged[col] = float(np.max(np.abs(paired[col+"_new"] - paired[col+"_old"])))
        assert unchanged[col] == 0
    prior_config = json.loads((old / "config_resolved.json").read_text())["config"]
    physical_sections = [k for k in config if k not in ("experiment", "estimator", "analysis")]
    assert all(config[k] == prior_config[k] for k in physical_sections)

    sbl = frame[frame.estimator == "sbl_jointfreq_3d"].copy()
    states = sbl.sbl_state_json.map(json.loads)
    sbl["active_atoms"] = states.map(lambda x: len(x["indices"]))
    sbl["max_evidence_improvement"] = states.map(lambda x: x["max_evidence_improvement"])
    sbl["objective_increases_gt_1e_minus6"] = states.map(
        lambda x: int(np.sum(np.diff(x["objective_history"]) > 1e-6)))
    assert all(len(state["objective_history"]) == row.algorithm_iterations + 1
               for (_, row), state in zip(sbl.iterrows(), states))
    assert (sbl.algorithm_converged == (sbl.max_evidence_improvement <= 1e-4)).all()
    output = directory / "analysis"
    fields = key + ["algorithm_converged", "algorithm_iterations", "active_atoms",
        "max_evidence_improvement", "objective_increases_gt_1e_minus6",
        "fullband_128_nmse_db", "estimator_runtime_seconds", "channel_reconstruction_seconds"]
    sbl[fields].to_csv(output / "sbl_diagnostics.csv", index=False)
    sbl.groupby(["schedule", "array_reference_snr_db"]).agg(
        converged=("algorithm_converged", "sum"),
        iterations_median=("algorithm_iterations", "median"),
        iterations_max=("algorithm_iterations", "max"),
        active_median=("active_atoms", "median"), active_max=("active_atoms", "max"),
        solve_seconds_median=("estimator_runtime_seconds", "median"),
        synthesis_seconds_median=("channel_reconstruction_seconds", "median")
    ).to_csv(output / "sbl_convergence_table.csv")

    archived = pd.read_csv(linear / "metrics_raw.csv")
    matched_linear = frame[frame.estimator != "sbl_jointfreq_3d"].merge(archived,
        on=key, suffixes=("_new", "_old"), validate="one_to_one")
    assert len(matched_linear) == 2400
    linear_delta = float(np.max(np.abs(matched_linear.fullband_128_nmse_db_new
                                     - matched_linear.fullband_128_nmse_db_old)))
    assert linear_delta < 1e-3
    lmmse = archived[archived.estimator == "lmmse"]
    assert set(lmmse[key[:-1]].itertuples(index=False, name=None)) == set(sbl[key[:-1]].itertuples(index=False, name=None))
    combined = pd.concat([frame, lmmse], ignore_index=True)
    rng = np.random.default_rng(config["experiment"]["bootstrap_seed"])
    records = []
    for (schedule, snr, method), group in combined.groupby(key[1:]):
        values = group.fullband_128_nmse_linear.to_numpy()
        ci = bootstrap_interval(values, lambda v: float(10*np.log10(np.mean(v))),
                                rng=rng, resamples=1000)
        records.append(dict(schedule=schedule, snr_db=snr, estimator=method,
            nmse_db=10*np.log10(values.mean()), ci_low=ci[0], ci_high=ci[1]))
    curves = pd.DataFrame(records)
    curves.to_csv(output / "sbl_current_baselines.csv", index=False)
    styles = {"oracle_geometry_ls": ("Oracle geometry + LS", "#009E73", "^", "--"),
              "grid_omp_3d": ("OMP", "#0072B2", "s", "-"),
              "sbl_jointfreq_3d": ("SBL posterior mean", "#CC79A7", "o", "-"),
              "lmmse": ("Conditional MC-128 LMMSE (archive)", "#D55E00", "D", ":")}
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
        for ax, (schedule, group), title in zip(axes, curves.groupby("schedule"),
                ["F1: 1 configuration, 128 pilot tones", "F2: 8 configurations, 16 pilot tones"]):
            for method, (label, color, marker, line) in styles.items():
                g = group[group.estimator == method].sort_values("snr_db")
                ax.plot(g.snr_db, g.nmse_db, label=label, color=color, marker=marker, ls=line)
                ax.fill_between(g.snr_db, g.ci_low, g.ci_high, color=color, alpha=.12)
            ax.set(title=title, xlabel="Input reference SNR (dB)", xticks=[45, 50, 55, 60, 65, 70])
            ax.grid(alpha=.25)
        axes[0].set_ylabel("Fullband 128-tone CSI NMSE (dB)")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
                   bbox_to_anchor=(.5, 1.01), frameon=False)
        fig.text(.5, .015, "100 paired scenes; mean linear NMSE converted to dB; shading: 95% scene bootstrap CI.\n"
                 "Known coarse centers. LMMSE additionally uses matched conditional covariance. Oracle uses true geometry.",
                 ha="center", fontsize=9)
        fig.subplots_adjust(top=.78, bottom=.21, left=.085, right=.98, wspace=.10)
        figure = output / "figures/04_sbl_current_baselines"
        for suffix in (".png", ".pdf"):
            fig.savefig(figure.with_suffix(suffix), dpi=240, facecolor="white")
        plt.close(fig)
    manifest = {
        "source_sha256": {str(p.relative_to(PROJECT_ROOT)): hash_file(p) for p in
            [directory / "metrics_raw.csv", old / "metrics_raw.csv", linear / "metrics_raw.csv"]},
        "analysis_script_sha256": hash_file(Path(__file__)),
        "rows": len(frame), "independent_scenes": 100, "checkpoint_rows_equal": True,
        "physical_config_sections_equal": physical_sections,
        "old_baseline_max_abs_delta": unchanged,
        "linear_archive_baseline_max_abs_nmse_delta_db": linear_delta,
        "sbl_converged": int(sbl.algorithm_converged.sum()),
        "sbl_iterations_range": [int(sbl.algorithm_iterations.min()), int(sbl.algorithm_iterations.max())],
        "objective_increases_above_1e_minus6": int(sbl.objective_increases_gt_1e_minus6.sum()),
        "figure_statistic": "10*log10(mean(linear NMSE)); 1000 scene bootstrap draws; no smoothing or exclusions",
        "bootstrap_seed": config["experiment"]["bootstrap_seed"],
        "figure_sha256": {suffix: hash_file(figure.with_suffix(suffix)) for suffix in (".png", ".pdf")},
    }
    write_json(output / "sbl_audit.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
