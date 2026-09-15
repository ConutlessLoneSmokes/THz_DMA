#!/usr/bin/env python3
"""Minimal metric/analysis validation. No dictionary build or OMP/Yang run."""
from __future__ import annotations

import ast
import copy
import json
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
from _run_support import PROJECT_ROOT as ROOT, hash_file, read_toml

import numpy as np
import pandas as pd
from thz_dma.stage2a import channel_domain_metrics, _channel_from_estimate, _operator_for_schedule
from thz_dma.stage2 import build_stage2_platform, sample_stage2_paths, stage2_schedule_weights
from thz_dma.observations.multistrip import array_input_noise_variance, observe_multistrip_schedule
from thz_dma.estimators.planar_sparse import oracle_geometry_ls_estimate
from thz_dma.analysis.stage2a import analyze_stage2a
from thz_dma.analysis.stage2a_domains import BASELINE_ID, DOMAINS, bounded_diagnostics, _config_comparable


def main():
    for path in [ROOT / "src/thz_dma/stage2a.py", *sorted((ROOT / "src/thz_dma/analysis").glob("stage2a*.py")), Path(__file__)]:
        ast.parse(path.read_text(encoding="utf-8"))
    baseline = ROOT / "runs" / BASELINE_ID
    before = {str(p): hash_file(p) for p in baseline.rglob("*") if p.is_file()}
    config = read_toml(
        ROOT / "configs/direction1_stage2a_bounded_65db_diagnostic.toml"
    )
    old_config = json.loads((baseline / "config_resolved.json").read_text(encoding="utf-8"))["config"]
    assert _config_comparable(config, old_config)
    assert config["noise"]["array_reference_snr_db"] == [65.0]
    assert config["analysis"]["primary_array_reference_snr_db"] == 65.0
    platform = build_stage2_platform(config, ROOT)
    raw = pd.read_csv(baseline / "metrics_raw.csv")
    legacy = np.rint(np.linspace(0, 127, 16)).astype(int)
    f2pilots = platform.schedules[1].pilot_indices
    assert np.array_equal(legacy, f2pilots)
    # Unequal truth energy catches an incorrect arithmetic average of domain NMSE.
    truth = np.arange(1, 129, dtype=float)[:, None, None] * np.ones((128, 2, 3), complex)
    estimate = truth.copy()
    held = np.setdiff1d(np.arange(128), f2pilots)
    estimate[held] *= 2
    metric = channel_domain_metrics(truth, estimate, f2pilots, legacy)
    assert metric["pilot_nmse_linear"] == 0 and metric["heldout_112_nmse_linear"] == 1
    assert np.isclose(metric["fullband_128_nmse_linear"], np.sum(abs(truth[held])**2) / np.sum(abs(truth)**2))
    f1metric = channel_domain_metrics(truth, estimate, np.arange(128), legacy)
    assert f1metric["pilot_nmse_linear"] == f1metric["fullband_128_nmse_linear"]
    assert f1metric["heldout_112_nmse_linear"] is None
    # Check all five original geometries without building dictionaries or estimating them.
    for seed in config["experiment"]["replicate_seeds"]:
        paths = sample_stage2_paths(config, np.random.default_rng(seed))
        record = raw[raw.scene_id == f"{seed}:0"].iloc[0]
        for i, path in enumerate(paths):
            assert np.isclose(path.range_m, record[f"true_path_{i}_range_m"], rtol=0, atol=1e-12)
            assert np.isclose(np.rad2deg(path.azimuth_rad), record[f"true_path_{i}_azimuth_deg"], rtol=0, atol=1e-12)
            assert np.isclose(np.rad2deg(path.elevation_rad), record[f"true_path_{i}_elevation_deg"], rtol=0, atol=1e-12)
    # Two Oracle fits on a single unchanged physical scene exercise actual 128x8x160 CSI.
    seed = config["experiment"]["replicate_seeds"][0]
    paths = sample_stage2_paths(config, np.random.default_rng(seed))
    coords = np.array([[p.range_m, p.azimuth_rad, p.elevation_rad] for p in paths])
    channel = _channel_from_estimate(platform.frequencies_hz, platform, coords, np.array([p.complex_gain for p in paths]), config)
    oracle_checks = []
    for index, schedule in enumerate(platform.schedules):
        weights = stage2_schedule_weights(platform, schedule, config, "thz_physical")
        operator = _operator_for_schedule(platform, schedule, weights, config)
        observations = []
        for snr in (55, 65):
            observations.append(observe_multistrip_schedule(weights, channel[schedule.pilot_indices], pilot_energy_per_re=1,
                element_noise_variance=array_input_noise_variance(channel, snr), rf_noise_variance=0,
                rng=np.random.default_rng(np.random.SeedSequence([seed, 0, index, 2026090921]))))
        low, high = observations
        assert np.allclose((low.observed-low.clean)/np.sqrt(10), high.observed-high.clean, rtol=1e-10, atol=1e-20)
        fit = oracle_geometry_ls_estimate(operator, operator.whiten(high.observed), coords[:, 0], coords[:, 1], coords[:, 2])
        fitted = _channel_from_estimate(platform.frequencies_hz, platform, coords, fit["complex_gain"], config)
        metrics = channel_domain_metrics(channel, fitted, schedule.pilot_indices, legacy)
        direct = np.sum(abs(fitted[legacy]-channel[legacy])**2)/np.sum(abs(channel[legacy])**2)
        assert np.isclose(metrics["channel_nmse_linear"], direct)
        oracle_checks.append({"schedule": schedule.name, "finite_fullband": bool(np.isfinite(metrics["fullband_128_nmse_linear"]))})
    # Synthetic report fixture, explicitly not a simulation result; deleted on exit.
    with tempfile.TemporaryDirectory(prefix="stage2a_metric_check_") as temporary:
        temp = Path(temporary)
        old = temp / BASELINE_ID
        old.mkdir()
        for name in ("metrics_raw.csv", "config_resolved.json", "metrics_summary.json"):
            (old/name).write_bytes((baseline/name).read_bytes())
        legacy_analysis = analyze_stage2a(old)
        assert not legacy_analysis["domain_metrics_available"]
        fixture = temp / "synthetic_65db"
        fixture.mkdir()
        frame = raw.copy()
        frame["array_reference_snr_db"] = 65.0
        frame["effective_output_snr_db"] += 10
        for name in DOMAINS:
            frame[f"{name}_linear"] = .01 if name == "pilot_nmse" else 1.0
            frame[f"{name}_db"] = -20.0 if name == "pilot_nmse" else 0.0
            frame[f"{name}_subcarrier_count"] = {"pilot_nmse": 16, "fullband_128_nmse": 128, "heldout_112_nmse":112}[name]
        f1 = frame.schedule_family == "frequency_single_shot"
        frame.loc[f1, "pilot_nmse_subcarrier_count"] = 128
        frame.loc[f1, "pilot_nmse_linear"] = 1
        frame.loc[f1, "pilot_nmse_db"] = 0
        frame.loc[f1, "heldout_112_nmse_subcarrier_count"] = 0
        frame.loc[f1, ["heldout_112_nmse_linear", "heldout_112_nmse_db"]] = np.nan
        summary = json.loads((baseline/"metrics_summary.json").read_text(encoding="utf-8"))
        summary["experiment"]["array_reference_snr_db"] = [65.0]
        summary["scope"]["csi_metric_version"] = "frequency-domains-v2"
        for saved, schedule in zip(summary["schedules"], platform.schedules):
            saved["pilot_indices"] = schedule.pilot_indices.tolist()
            saved["heldout_indices"] = np.setdiff1d(np.arange(128), schedule.pilot_indices).tolist()
        frame.to_csv(fixture/"metrics_raw.csv", index=False)
        (fixture/"metrics_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (fixture/"config_resolved.json").write_text(json.dumps({"config": config}), encoding="utf-8")
        compact = analyze_stage2a(fixture)
        assert compact["domain_metrics_available"]
        assert all(row["pattern_scenes"] == 5 for row in compact["bounded_diagnostics"]["F2"])
        assert all(not row["azimuth_improved"] for row in compact["bounded_diagnostics"]["F1"])
        frame.loc[f1, "azimuth_rmse_deg"] = 0
        for i in range(3):
            frame.loc[f1, f"estimated_path_{i}_azimuth_deg"] = frame.loc[f1, f"true_path_{i}_azimuth_deg"]
        positive = bounded_diagnostics(frame, config, fixture, {"complete": True})
        assert all(row["azimuth_improved"] and row["azimuth_recovered"] for row in positive["F1"])
        invalid = copy.deepcopy(config)
        invalid["estimator"]["yang_ogols_3d"]["refinement_iterations"] += 1
        assert bounded_diagnostics(frame, invalid, fixture, {"complete": True})["F1_status"].startswith("config_mismatch")
    assert before == {str(p): hash_file(p) for p in baseline.rglob("*") if p.is_file()}
    print(json.dumps({"status": "passed", "static_parse": True, "config_only_snr_and_labels_changed": True,
        "five_original_geometries_match": True, "standard_noise_scaling": True, "energy_weighted_domains": True,
        "legacy_and_synthetic_analysis": True, "historical_run_unchanged": True,
        "physical_oracle_checks": oracle_checks, "dictionary_builds": 0, "OMP_Yang_calls": 0,
        "full_diagnostic_or_formal_run_started": False}, indent=2))


if __name__ == "__main__":
    main()
