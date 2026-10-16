"""Independent saved-record checks; reads old results without changing them."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
from _run_support import hash_file, write_json
from run_d041_fusion_probe import verify_run

OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def check_prefix(prefix, steps):
    result = {"prefix": prefix, "steps": steps, "endpoints": 0, "trajectories": 0, "denoiser_predictions": 0,
              "source_members": 0, "failed_cases": 0}
    calibration = ROOT/"runs"/(prefix+"_calibrate")
    profile = read(calibration/"calibration_profile.json")
    prior = pd.read_csv(calibration/"prior_denoising.csv")
    assert len(prior) == 3*steps and set(prior.split) == {"train", "validation"}
    assert profile["training_scenes"] == 2 and profile["validation_not_used_for_variance"]
    for t, value in enumerate(profile["variances_real"], 1):
        training = prior.loc[(prior.t == t) & (prior.split == "train")]
        assert len(training) == 2
        np.testing.assert_allclose(value, max(training.real_coordinate_mse.mean(), 1e-8), rtol=1e-12)
    assert read(calibration/"gaussian_checks.json")["passed"]
    tuning = None
    for stage, count, scene in [("calibrate", None, None), ("tune", 48, 0), ("screen", 36, 1)]:
        folder = ROOT/"runs"/(prefix+"_"+stage)
        verify_run(folder)
        manifest = read(folder/"code_manifest.json")
        with zipfile.ZipFile(folder/"source_snapshot.zip") as snapshot:
            for item in manifest:
                assert hashlib.sha256(snapshot.read(item["path"])).hexdigest() == item["sha256"]
                result["source_members"] += 1
        if stage == "calibrate":
            continue
        parents = read(folder/"parents.json")
        assert parents["calibrate"]["manifest_sha256"] == hash_file(calibration/"manifest.json")
        table = pd.read_csv(folder/"metrics_raw.csv")
        assert len(table) == count and set(table.scene_index) == {scene} and set(table.data_split) == {"validation"}
        records = [read(p) for p in (folder/"checkpoint").glob("case_*.json")]
        assert len(records) == count
        result["endpoints"] += count
        reference = read(folder/"training_reference_power.json")
        if stage == "tune":
            tuning = read(folder/"tuning.json")
            for (label, condition), block in table.loc[table.method == "zhou"].groupby(["operator", "condition"]):
                finite = block.loc[block.status == "success"]
                selected = finite.sort_values("nmse_linear", kind="stable").iloc[0].guidance
                assert selected == tuning[f"{label}:{condition}"]["selected"]
        else:
            assert tuning == read(folder/"tuning.json")
            assert parents["tune"]["manifest_sha256"] == hash_file(ROOT/"runs"/(prefix+"_tune")/"manifest.json")
            for row in table.loc[table.method == "zhou"].itertuples():
                assert row.guidance == tuning[f"{row.operator}:{row.condition}"]["selected"]
        for _, block in table.groupby(["operator", "condition", "scene_index"]):
            assert block.observation_sha256.nunique() == 1 and block.posterior_seed.nunique() == 1
        for row in table.itertuples():
            if row.condition == "snr_20":
                np.testing.assert_allclose(reference[row.operator]["normalized_whitened_power"]/
                    row.observation_complex_variance, 100., rtol=1e-12)
            else:
                assert row.observation_complex_variance == 0. and row.physical_element_noise_complex_variance == 0.
        raw_trace = 0
        for case in records:
            row, trace, means = case["row"], case["trajectory"], case["denoiser"]
            result["denoiser_predictions"] += len(means)
            assert row["data_split"] == "validation"
            if row["status"] != "success":
                result["failed_cases"] += 1
                continue
            assert row["network_calls"] == (0 if row["method"] == "pinv_ls" else steps)
            if row["method"] == "pinv_ls":
                assert not trace and not means
                continue
            assert len(trace) == steps+1 and trace[0]["network_calls"] == 0 and trace[-1]["t"] == 0
            assert trace[-1]["state_alpha_bar"] == 1. and trace[-1]["network_calls"] == steps
            np.testing.assert_allclose(row["nmse_linear"], trace[-1]["nmse_linear"], rtol=1e-12)
            for state in trace:
                np.testing.assert_allclose(state["nmse_linear"],
                    state["row_space_nmse_linear"]+state["null_space_nmse_linear"], rtol=1e-4, atol=1e-6)
            raw_trace += len(trace)
        csv_trace = pd.read_csv(folder/"trajectories.csv")
        assert len(csv_trace) == sum(len(c["trajectory"]) for c in records)
        result["trajectories"] += len(csv_trace)
        for _, block in csv_trace.loc[csv_trace.phase == "initial"].groupby(["operator", "condition", "scene_index"]):
            for key in ("nmse_linear", "prediction_energy_ratio", "row_space_nmse_linear", "null_space_nmse_linear"):
                np.testing.assert_allclose(block[key], block[key].iloc[0], rtol=0, atol=0)
    return result


def compare_frozen_baselines(prefix):
    folder = ROOT/"runs"/(prefix+"_tune")
    new = pd.read_csv(folder/"metrics_raw.csv")
    records = read(folder/"operators.json")
    powers = read(folder/"training_reference_power.json")
    matched = []
    for resource in (128, 4096, 8192):
        original = ROOT/"runs"/f"d041_budget_sweep_20261015_v1_re{resource}_validation"
        old = pd.read_csv(original/"metrics_raw.csv")
        old_operators = {r["label"]: r for r in read(original/"operators.json")}
        old_power = read(original/"observation_reference.json")["reference"]
        for record in records:
            if record["budget_re"] != resource:
                continue
            label = record["label"]
            for key in ("weights_sha256", "active_mask_sha256", "rank", "pilot_re", "state_indices", "unique_dma_configurations"):
                assert record[key] == old_operators[label][key]
            assert powers[label]["training_scenes"] == 2048
            np.testing.assert_allclose(powers[label]["normalized_whitened_power"],
                old_power[label]["normalized_whitened_power"], rtol=1e-6)
            for condition in ("noiseless_v1e-08", "noiseless_v1e-10"):
                first = new.loc[(new.operator == label) & (new.condition == condition) &
                                (new.method == "zhou") & (new.guidance == 4.)].iloc[0]
                second = old.loc[(old.operator == label) & (old.condition == condition) &
                                 (old.estimator == "clean_ddpm") & (old.scene_index == 0)].iloc[0]
                assert second.guidance == 4.
                keys = ["nmse_linear", "observation_residual_relative", "prediction_energy_ratio",
                        "normalized_correlation", "row_space_nmse_linear", "null_space_nmse_linear"]
                np.testing.assert_allclose(first[keys].to_numpy(float), second[keys].to_numpy(float), rtol=2e-5, atol=1e-7)
                matched.append({"operator": label, "condition": condition,
                                "nmse_absolute_difference": abs(first.nmse_linear-second.nmse_linear)})
    assert len(matched) == 10
    return matched


def check_rejection():
    identifier = "d041_fusion_missing_parent_guard_20261016"
    command = [sys.executable, str(ROOT/"scripts/run_d041_fusion_probe.py"), "screen", "--run-id", identifier,
        "--data-run-id", "d041_noabs_smoke_20261013_v3_wb_data", "--train-run-id", "d041_noabs_smoke_20261013_v3_wb_clean",
        "--device", "cpu", "--smoke"]
    rejected = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert rejected.returncode == 2 and "--calibration-run-id" in rejected.stderr
    assert not (ROOT/"runs"/identifier).exists()
    original = ROOT/"runs/d041_fusion_smoke_20261016_v1_calibrate"
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)/"copy"
        shutil.copytree(original, folder)
        profile = folder/"calibration_profile.json"
        profile.write_text("{}")
        try:
            verify_run(folder)
            raise AssertionError("changed calibration was accepted")
        except ValueError as error:
            assert "saved output changed" in str(error)
        shutil.copyfile(original/profile.name, profile)
        sources = read(folder/"code_manifest.json")
        sources[0]["sha256"] = "0"*64
        write_json(folder/"code_manifest.json", sources)
        manifest = read(folder/"manifest.json")
        manifest["code_manifest.json"] = hash_file(folder/"code_manifest.json")
        write_json(folder/"manifest.json", manifest)
        try:
            verify_run(folder)
            raise AssertionError("changed source was accepted")
        except ValueError as error:
            assert "source changed" in str(error)
    return {"missing_parent_rejected_without_run_creation": True, "changed_profile_rejected": True,
            "changed_source_rejected": True}


def main():
    results = [check_prefix("d041_fusion_smoke_20261016_v1", 8),
               check_prefix("d041_fusion_gpu_check_20261016_v1", 100)]
    paired = compare_frozen_baselines("d041_fusion_gpu_check_20261016_v1")
    rejections = check_rejection()
    protected = read(OUT/"protected_files.json")
    for name, digest in protected.items():
        assert hash_file(ROOT/name) == digest, name
    inventories = {}
    cache = {}
    for name in ("d041_budget_sweep_20261015_v1_review", "d041_config_sweep_20261014_v1_review"):
        path = ROOT/"research_route/experiment_results/2026-10-15"/name/"source_inputs.json"
        inventory = read(path)
        for key, value in inventory.items():
            source = ROOT/key
            stat = source.stat()
            token = (str(source), stat.st_size, stat.st_mtime_ns)
            if token not in cache:
                cache[token] = hash_file(source)
            assert stat.st_size == value["size_bytes"] and cache[token] == value["sha256"], key
        inventories[name] = len(inventory)
        print(f"preserved {name}: {len(inventory)} files", flush=True)
    write_json(OUT/"verification.json", {"passed": True, "runs": results, "frozen_baseline_replays": paired,
        "protected_files": len(protected), "rejection_controls": rejections, "historical_inventories": inventories,
        "unique_historical_files": len(cache), "scope": "small train/validation checks; no performance acceptance"})
    print(json.dumps({"passed": True, "runs": results, "protected_files": len(protected)}, indent=2))


if __name__ == "__main__":
    main()
