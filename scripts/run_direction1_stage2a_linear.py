#!/usr/bin/env python3
"""Paired conditional LMMSE with archived OMP and Oracle baselines."""
from __future__ import annotations

import json
import os
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import pandas as pd

from _run_support import PROJECT_ROOT, create_run_archive, hash_file, parse_run_arguments
from thz_dma.channels.planar import sample_stage2_paths, channel_from_planar_estimate
from thz_dma.estimators.planar_baselines import build_stage2a_operators
from thz_dma.estimators.linear_channel import (
    conditional_planar_covariance_basis, lmmse_coefficients,
)
from thz_dma.evaluation.beamforming import build_planar_data_codebook, feasible_planar_data_metrics
from thz_dma.metrics.diagnostics import channel_domain_metrics
from thz_dma.metrics.statistics import bootstrap_interval, power_db
from thz_dma.observations.multistrip import array_input_noise_variance, observe_multistrip_schedule, effective_tensor_output_snr_db


def project_basis(basis, weights, operator, indices):
    raw = np.einsum("jkse,nkse->nsjk", np.conjugate(weights), basis[:, indices], optimize=True)
    return (raw / np.sqrt(operator.relative_output_noise_variance)[None]).reshape(len(basis), -1)


def decomposition(truth, estimate):
    power = float(np.vdot(truth, truth).real)
    estimate_power = float(np.vdot(estimate, estimate).real)
    cross = np.vdot(truth, estimate)
    return {"estimate_energy_ratio": estimate_power / power,
            "real_cross_ratio": float(cross.real / power),
            "channel_correlation_squared": float(abs(cross) ** 2 / max(power * estimate_power, np.finfo(float).tiny))}


def analyze(archive, rows, setup):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(rows)
    rng = np.random.default_rng(setup["bootstrap_seed"])
    summaries = []
    for (schedule, snr, method), block in frame.groupby(["schedule", "array_reference_snr_db", "estimator"]):
        values = block.fullband_128_nmse_linear.to_numpy()
        interval = bootstrap_interval(values, np.mean, rng=rng, resamples=setup["bootstrap_resamples"])
        reference = frame[(frame.schedule == schedule) & (frame.array_reference_snr_db == snr) & (frame.estimator == "grid_omp_3d")].set_index("scene_id")
        difference = block.set_index("scene_id").fullband_128_nmse_db - reference.fullband_128_nmse_db
        pair_ci = bootstrap_interval(difference.to_numpy(), np.mean, rng=rng, resamples=setup["bootstrap_resamples"])
        summaries.append({"schedule": schedule, "snr_db": float(snr), "estimator": method,
                          "scenes": len(block), "nmse_db": power_db(np.mean(values)),
                          "ci_low_db": power_db(interval[0]), "ci_high_db": power_db(interval[1]),
                          "paired_delta_db": float(difference.mean()), "paired_low_db": pair_ci[0], "paired_high_db": pair_ci[1],
                          "median_pilot_beam_loss_db": float(block.pilot_beam_loss_db.median()),
                          "mean_pilot_net_rate": float(block.pilot_net_rate.mean()),
                          "median_estimator_seconds": float(block.estimator_seconds.median())})
    table = pd.DataFrame(summaries)
    table.to_csv(archive.directory / "summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    labels = {"oracle_geometry_ls": "Oracle geometry + LS", "grid_omp_3d": "OMP",
              "lmmse": "Conditional MC-LMMSE"}
    plot_table = table[table.estimator.isin(labels)]
    for ax, schedule in zip(axes, table.schedule.unique()):
        for method, block in plot_table[plot_table.schedule == schedule].groupby("estimator"):
            ax.plot(block.snr_db, block.nmse_db, "o-", label=labels[method])
            ax.fill_between(block.snr_db, block.ci_low_db, block.ci_high_db, alpha=.12)
        ax.set(title=schedule, xlabel="Ideal-array reference SNR (dB)")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Fullband CSI NMSE (dB; linear mean then dB)")
    fig.tight_layout()
    fig.savefig(archive.directory / "nmse_vs_snr.png", dpi=180)
    plt.close(fig)
    lines = ["# Stage 2-A 条件 LMMSE 与 OMP/Oracle 对照", "",
             f"固定 {frame.scene_id.nunique()} 场景、两协议、六 SNR 点；当前均为已知逐径粗中心的条件实验。", "",
             "LMMSE 使用逐场景粗中心、已知条件分布和独立随机数做协方差积分，保留跨频相关；均值为零，路径增益二阶矩解析计算。128 次几何抽样对应 384 个协方差因子。它不是已知测试 CSI 的 Oracle，也不是精确非线性后验均值。",
             "通信指标在两协议共有且最接近载频的导频上，使用同一可行 DMA 码本重新计算；不能与旧载频通信指标拼接。协方差构建耗时单独归档；当前矩阵系数求解及 CSI 合成为估计时间。", "",
             "|协议|SNR|方法|全带 NMSE/dB|相对 OMP 的配对差值/dB，95% CI|", "|---|---:|---|---:|---|" ]
    for item in summaries:
        if item["snr_db"] in (55, 65):
            lines.append(f"|{item['schedule']}|{item['snr_db']:g}|{item['estimator']}|{item['nmse_db']:.3f}|{item['paired_delta_db']:+.3f} [{item['paired_low_db']:+.3f}, {item['paired_high_db']:+.3f}]|")
    lines += ["", "2026-10-01 起不再运行或展示 SOMP 与 Tikhonov；历史运行保留。"]
    (archive.directory / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summaries


def main():
    config_path, plan, run_id = parse_run_arguments("direction1_stage2a_linear.toml", __doc__)
    settings = plan["experiment"]
    source = PROJECT_ROOT / settings["source_run"]
    config = json.loads((source / "config_resolved.json").read_text(encoding="utf-8"))["config"]
    source_summary = json.loads((source / "metrics_summary.json").read_text(encoding="utf-8"))
    source_rows = pd.read_csv(source / "metrics_raw.csv")
    source_rows = source_rows[source_rows.estimator.isin(["oracle_geometry_ls", "grid_omp_3d"])]
    config["experiment"]["estimators"] = ["oracle_geometry_ls", "grid_omp_3d", "lmmse"]
    resolved = {**config, "linear_experiment": settings}
    hashes = {name: hash_file(source / name) for name in ("metrics_raw.csv", "metrics_summary.json", "config_resolved.json")}
    archive = create_run_archive("stage2a_linear", run_id, config_path=config_path, config=resolved,
        seeds={"prior": settings["prior_seed"], "evaluation": config["experiment"]["replicate_seeds"], "source_hashes": hashes},
        script_path=__file__, environment={"linear_algebra_threads": 4},
        checkpoint_note="No neural model. Conditional prior integration uses independent seeds and analytic gain moments.\n")
    try:
        run(archive, config, settings, source_rows, source_summary)
        assert hashes == {name: hash_file(source / name) for name in hashes}
        archive.write_status("complete", run_id=archive.run_id)
    except Exception as error:
        archive.write_status("failed", error=str(error))
        raise


def run(archive, config, settings, source_rows, source_summary):
    platform, weights, operators = build_stage2a_operators(config, PROJECT_ROOT)
    common = sorted(set.intersection(*(set(s.pilot_indices) for s in platform.schedules)))
    beam_index = min(common, key=lambda i: abs(platform.frequencies_hz[i] - config["system"]["carrier_frequency_hz"]))
    dma = config["dma"]
    codebook, _ = build_planar_data_codebook(
        carrier_frequency_hz=float(platform.frequencies_hz[beam_index]),
        feed_distances_m=platform.feed_distances_m,
        **{key: dma[key] for key in ("num_microstrips", "elements_per_microstrip", "tuning_bandwidth_hz", "quality_factor", "waveguide_effective_index", "waveguide_end_power_fraction")},
        codebook_size=config["data_evaluation"]["codebook_size"], seed_start=config["data_evaluation"]["codebook_seed_start"])
    rows, preparation, convergence = [], [], []
    source_error = 0.0
    snrs = config["noise"]["array_reference_snr_db"]
    records = source_summary["search"]["scene_records"][:settings["scene_limit"]]
    if not records or len(records) != settings["scene_limit"]:
        raise ValueError("invalid scene limit")
    for scene_number, record in enumerate(records):
        scene_id = record["scene_id"]
        seed, sample = map(int, scene_id.split(":"))
        centers = sample_stage2_paths(config, np.random.default_rng(np.random.SeedSequence([seed, sample, 2_026_091_101])))
        np.testing.assert_allclose([[p.range_m, p.azimuth_rad, p.elevation_rad] for p in centers], record["centers_m_rad_rad"], atol=1e-14)
        coordinates = np.asarray(record["centers_m_rad_rad"]) + record["truth"]["offsets_m_rad_rad"]
        truth = channel_from_planar_estimate(platform.frequencies_hz, platform, coordinates, np.array([p.complex_gain for p in centers]), config)
        started = time.perf_counter()
        basis = conditional_planar_covariance_basis(platform, config, record, draws=settings["prior_draws"],
            rng=np.random.default_rng(np.random.SeedSequence([settings["prior_seed"], seed, sample])))
        preparation.append({"scene_id": scene_id, "covariance_basis_seconds": time.perf_counter() - started, "basis_bytes": basis.nbytes})
        for schedule_index, schedule in enumerate(platform.schedules):
            operator, combiner = operators[schedule.name], weights[schedule.name]
            started = time.perf_counter()
            observed_basis = project_basis(basis, combiner, operator, schedule.pilot_indices)
            observations, variances, whites = [], [], []
            for snr in snrs:
                variance = array_input_noise_variance(truth, snr)
                observation = observe_multistrip_schedule(combiner, truth[schedule.pilot_indices], pilot_energy_per_re=1,
                    element_noise_variance=variance, rf_noise_variance=variance * config["noise"]["rf_noise_variance_fraction_of_element_noise"],
                    rng=np.random.default_rng(np.random.SeedSequence([seed, sample, schedule_index, 2_026_090_921])))
                observations.append(observation)
                variances.append(variance)
                whites.append(operator.whiten(observation.observed))
            preparation[-1][schedule.name + "_projection_seconds"] = time.perf_counter() - started
            lmmse, lmmse_seconds = [], []
            for white, variance in zip(whites, variances):
                started = time.perf_counter()
                coefficients = lmmse_coefficients(observed_basis, white, variance)
                lmmse.append((basis.reshape(len(basis), -1).T @ coefficients.astype(np.complex64)).reshape(truth.shape))
                lmmse_seconds.append(time.perf_counter() - started)
            lmmse = np.asarray(lmmse)
            if scene_number < settings["covariance_check_scenes"]:
                count = settings["covariance_check_draws"] * config["channel"]["num_paths"]
                scale = np.sqrt(settings["prior_draws"] / settings["covariance_check_draws"])
                small_coeff = lmmse_coefficients(observed_basis[:count] * scale, np.asarray(whites).T, np.asarray(variances))
                small = (basis[:count].reshape(count, -1).T @ (small_coeff * scale).astype(np.complex64)).T.reshape(len(snrs), *truth.shape)
                for i, snr in enumerate(snrs):
                    convergence.append({"scene_id": scene_id, "schedule": schedule.name, "snr_db": snr,
                        "small_nmse_db": power_db(np.linalg.norm(small[i] - truth) ** 2 / np.linalg.norm(truth) ** 2),
                        "large_nmse_db": power_db(np.linalg.norm(lmmse[i] - truth) ** 2 / np.linalg.norm(truth) ** 2)})
            for index, snr in enumerate(snrs):
                observation, variance, white = observations[index], variances[index], whites[index]
                estimates = {"lmmse": (lmmse[index], lmmse_seconds[index])}
                block = source_rows[(source_rows.scene_id == scene_id) & (source_rows.schedule == schedule.name) & (source_rows.array_reference_snr_db == snr)]
                for old in block.itertuples():
                    points = np.array([[getattr(old, f"estimated_path_{p}_range_m"), np.deg2rad(getattr(old, f"estimated_path_{p}_azimuth_deg")), np.deg2rad(getattr(old, f"estimated_path_{p}_elevation_deg"))] for p in range(3)])
                    templates = operator.evaluate(*points.T)
                    gains = np.linalg.lstsq(templates.T, white, rcond=None)[0]
                    estimate = channel_from_planar_estimate(platform.frequencies_hz, platform, points, gains, config)
                    estimates[old.estimator] = (estimate, old.estimator_runtime_seconds)
                    discrepancy = abs(power_db(np.linalg.norm(estimate - truth) ** 2 / np.linalg.norm(truth) ** 2) - old.fullband_128_nmse_db)
                    source_error = max(source_error, discrepancy)
                    assert discrepancy < 0.02, (scene_id, old.estimator, discrepancy)
                    assert abs(effective_tensor_output_snr_db(observation.clean, observation.output_noise_variance) - old.effective_output_snr_db) < 1e-9
                for method, (estimate, seconds) in estimates.items():
                    link = feasible_planar_data_metrics(truth[beam_index], estimate[beam_index], codebook,
                        element_noise_variance=variance, rf_noise_variance=variance * config["noise"]["rf_noise_variance_fraction_of_element_noise"],
                        coherence_symbols=config["data_evaluation"]["coherence_symbols"], training_symbols=schedule.training_symbols,
                        switch_guard_symbol_equivalents=schedule.num_switches * config["acquisition"]["switch_guard_symbol_equivalents"])
                    rows.append({"scene_id": scene_id, "schedule": schedule.name, "array_reference_snr_db": snr, "estimator": method,
                        **channel_domain_metrics(truth, estimate, schedule.pilot_indices, np.arange(128)),
                        **decomposition(truth, estimate), "pilot_beam_loss_db": link["feasible_beam_loss_db"],
                        "pilot_net_rate": link["net_rate_bps_hz"], "estimator_seconds": seconds})
        print(f"{scene_number + 1}/{len(records)} {scene_id}, rows={len(rows)}", flush=True)
    assert len(rows) == len(records) * len(platform.schedules) * len(snrs) * 3
    assert not pd.DataFrame(rows).duplicated(["scene_id", "schedule", "array_reference_snr_db", "estimator"]).any()
    archive.write_metrics(rows)
    archive.write_json("preparation.json", preparation)
    archive.write_json("covariance_convergence.json", convergence)
    summary = analyze(archive, rows, settings)
    archive.write_json("metrics_summary.json", {"rows": len(rows), "scene_count": len(preparation),
        "beam_subcarrier_index": int(beam_index), "beam_frequency_hz": float(platform.frequencies_hz[beam_index]),
        "source_reconstruction_max_nmse_delta_db": source_error, "summary": summary})


if __name__ == "__main__":
    main()
