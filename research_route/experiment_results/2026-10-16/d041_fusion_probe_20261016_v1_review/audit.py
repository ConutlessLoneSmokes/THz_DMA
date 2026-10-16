"""Read-only formal fusion audit; all derived evidence stays in this directory."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
from _run_support import write_json
from audit_zhou2025_migration import cached_hash
import run_d041_fusion_probe as probe
import run_zhou2025_repro as reference
from thz_dma.metrics.statistics import bootstrap_interval
from zhou2025_repro.moment_recovery import gaussian_control

PREFIX = "d041_fusion_probe_20261016_v1"
KEY = ["operator", "condition", "scene_index", "method", "guidance"]
METRICS = ["nmse_linear", "nmse_db", "observation_residual_relative", "clean_observation_residual_relative",
           "prediction_energy_ratio", "normalized_correlation", "coherent_correlation_real",
           "row_space_nmse_linear", "null_space_nmse_linear", "truth_energy_normalized", "prediction_energy_normalized"]


def read(path):
    return json.loads(path.read_text())


def folder(stage):
    return ROOT/"runs"/(PREFIX+"_"+stage)


def key(row):
    return tuple(row[name] if name != "guidance" else (row[name] or 0.) for name in KEY)


def close(first, second):
    np.testing.assert_allclose(first, second, rtol=2e-9, atol=2e-12)


def audit():
    torch.set_num_threads(4)
    probe.hash_file = reference.hash_file = cached_hash
    configs, inputs, members = {}, {}, 0
    for stage in ("calibrate", "tune", "screen"):
        path = folder(stage)
        probe.verify_run(path)
        config = read(path/"config_resolved.json")["config"]
        configs[stage] = config
        assert not config["job"]["smoke"] and config["job"]["stage"] == stage
        assert config["band"] == "wide" and not config["channel"]["include_absorption"]
        assert config["diffusion"]["steps"] == 100
        with zipfile.ZipFile(path/"source_snapshot.zip") as snapshot:
            for item in read(path/"code_manifest.json"):
                assert hashlib.sha256(snapshot.read(item["path"])).hexdigest() == item["sha256"]
                members += 1
        for name, digest in read(path/"manifest.json").items():
            file = path/name
            inputs[str(file.relative_to(ROOT))] = {"size_bytes": file.stat().st_size, "sha256": digest}
        inputs[str((path/"manifest.json").relative_to(ROOT))] = {
            "size_bytes": (path/"manifest.json").stat().st_size, "sha256": cached_hash(path/"manifest.json")}
    assert probe.signature(configs["calibrate"]) == probe.signature(configs["tune"]) == probe.signature(configs["screen"])
    dependencies = read(folder("calibrate")/"dependencies.json")
    data = reference.load_data(dependencies["data_run_id"])
    assert cached_hash(data.directory/"dataset.json") == dependencies["dataset_sha256"]
    assert cached_hash(ROOT/"runs"/dependencies["train_run_id"]/"checkpoint/training.pt") == dependencies["checkpoint_sha256"]
    assert data.counts == {"train": 2048, "validation": 256, "test": 100} and data.shape == (128, 8, 160)
    for stage in ("tune", "screen"):
        assert read(folder(stage)/"dependencies.json") == dependencies
        parents = read(folder(stage)/"parents.json")
        assert parents["calibrate"] == {"run_id": PREFIX+"_calibrate", "manifest_sha256": cached_hash(folder("calibrate")/"manifest.json")}
        if stage == "screen":
            assert parents["tune"] == {"run_id": PREFIX+"_tune", "manifest_sha256": cached_hash(folder("tune")/"manifest.json")}
        started = (folder(stage)/"environment.txt").read_text().splitlines()[0].split(": ", 1)[1]
        assert started > read(folder("calibrate")/"status.json")["timestamp_utc"]
    started = (folder("screen")/"environment.txt").read_text().splitlines()[0].split(": ", 1)[1]
    assert started > read(folder("tune")/"status.json")["timestamp_utc"]
    settings = configs["calibrate"]["fusion_probe"]
    assert [settings[n] for n in ["calibration_train_scenes", "prior_validation_scenes", "tuning_scenes", "screening_start", "screening_scenes"]] == [256, 32, 32, 32, 64]
    assert settings["guidance"] == [.5, 1., 2., 4., 8., 16.]
    prior = pd.read_csv(folder("calibrate")/"prior_denoising.csv")
    decomposition = pd.read_csv(folder("calibrate")/"prior_decomposition.csv")
    profile = read(folder("calibrate")/"calibration_profile.json")
    assert len(prior) == 28800 and len(decomposition) == 1120
    assert not prior.duplicated(["split", "scene_index", "t"]).any()
    assert profile["training_scenes"] == 256 and profile["validation_not_used_for_variance"]
    for t in range(1, 101):
        saved = read(folder("calibrate")/"checkpoint"/f"calibration_t{t:03d}.json")
        block = prior.loc[prior.t == t]
        assert len(saved["prior_rows"]) == 288 and saved["train_scenes"] == 256
        assert set(block.loc[block.split == "train", "scene_index"]) == set(range(256))
        assert set(block.loc[block.split == "validation", "scene_index"]) == set(range(32))
        close(block.real_coordinate_mse.to_numpy(), [r["real_coordinate_mse"] for r in saved["prior_rows"]])
        value = max(float(block.loc[block.split == "train", "real_coordinate_mse"].mean()), 1e-8)
        close(value, profile["variances_real"][t-1])
        close(value, saved["variance_per_real_component"])
    assert read(folder("calibrate")/"gaussian_checks.json")["passed"] and gaussian_control()["passed"]
    prior_summary = prior.groupby(["split", "t"])[["nmse_linear", "real_coordinate_mse", "prediction_energy_ratio", "normalized_correlation"]].mean().reset_index()
    prior_summary["nmse_db"] = 10*np.log10(prior_summary.nmse_linear)
    prior_summary.to_csv(OUT/"prior_summary.csv", index=False)
    tables, counts, summaries = {}, {}, 0
    tuning = read(folder("tune")/"tuning.json")
    assert tuning == read(folder("screen")/"tuning.json")
    for stage, size, scenes in [("tune", 3072, range(32)), ("screen", 2304, range(32, 96))]:
        path = folder(stage)
        table = pd.read_csv(path/"metrics_raw.csv").fillna({"guidance": 0.})
        assert len(table) == size and not table.duplicated(KEY).any()
        assert set(table.scene_index) == set(scenes) and set(table.data_split) == {"validation"}
        assert (table.status == "success").all() and np.isfinite(table[METRICS].to_numpy()).all()
        expected_calls = np.where(table.method == "pinv_ls", 0, 100)
        np.testing.assert_array_equal(table.network_calls, expected_calls)
        # The physical projections are complex64, unlike the CSV/JSON identity checks.
        np.testing.assert_allclose(table.nmse_linear, table.row_space_nmse_linear+table.null_space_nmse_linear,
                                   rtol=1e-5, atol=1e-7)
        for _, block in table.groupby(["operator", "condition", "scene_index"]):
            assert block.observation_sha256.nunique() == block.posterior_seed.nunique() == 1
            assert block.likelihood_complex_variance.nunique() == 1
        powers = read(path/"training_reference_power.json")
        for row in table.itertuples():
            assert row.scene_id == data.record("validation", row.scene_index)["scene_id"]
            assert powers[row.operator]["training_scenes"] == 2048
            if row.condition == "snr_20":
                close(powers[row.operator]["normalized_whitened_power"]/row.observation_complex_variance, 100.)
                close(row.physical_element_noise_complex_variance, row.observation_complex_variance*data.scale**2)
            else:
                assert row.observation_complex_variance == row.physical_element_noise_complex_variance == 0.
        trajectories = pd.read_csv(path/"trajectories.csv").fillna({"guidance": 0.})
        means = pd.read_csv(path/"denoiser_predictions.csv").fillna({"guidance": 0.})
        np.testing.assert_allclose(trajectories.nmse_linear,
            trajectories.row_space_nmse_linear+trajectories.null_space_nmse_linear, rtol=1e-5, atol=1e-7)
        trace_groups, mean_groups = trajectories.groupby(KEY).indices, means.groupby(KEY).indices
        row_groups = {tuple(row[n] for n in KEY): row for row in table.to_dict("records")}
        cases = list((path/"checkpoint").glob("case_*.json"))
        assert len(cases) == size
        for file in cases:
            case = read(file)
            row, trace, predicted = case["row"], case["trajectory"], case["denoiser"]
            identifier = key(row)
            close([row[n] for n in METRICS], [row_groups[identifier][n] for n in METRICS])
            if row["method"] == "pinv_ls":
                assert not trace and not predicted
                continue
            assert len(trace) == 101 and len(predicted) == 7
            assert [r["t"] for r in trace] == list(range(100, -1, -1))
            assert [r["network_calls"] for r in trace] == list(range(101))
            close([row[n] for n in METRICS], [trace[-1][n] for n in METRICS])
            close(trajectories.iloc[trace_groups[identifier]][METRICS], [[r[n] for n in METRICS] for r in trace])
            close(means.iloc[mean_groups[identifier]][METRICS], [[r[n] for n in METRICS] for r in predicted])
        initial = trajectories.loc[trajectories.phase == "initial"]
        for _, block in initial.groupby(["operator", "condition", "scene_index"]):
            for name in METRICS:
                close(block[name], np.repeat(block[name].iloc[0], len(block)))
        assert len(trajectories) == int((table.method != "pinv_ls").sum())*101
        assert len(means) == int((table.method != "pinv_ls").sum())*7
        for summary in pd.read_csv(path/"summary.csv").itertuples():
            block = table.loc[(table.operator == summary.operator) & (table.condition == summary.condition) & (table.method == summary.method)]
            if summary.method == "zhou":
                block = block.loc[block.guidance == tuning[f"{summary.operator}:{summary.condition}"]["selected"]]
            assert len(block) == len(scenes) == summary.scenes and summary.failed_scenes == 0
            close(10*np.log10(block.nmse_linear.mean()), summary.mean_nmse_db)
            close((block.nmse_linear <= .1).mean(), summary.fraction_below_minus10db)
            summaries += 1
        for saved in pd.read_csv(path/"worst_variance_summary.csv").itertuples():
            block = table.loc[(table.operator == saved.operator) & (table.method == saved.method) & table.condition.str.startswith("noiseless_")]
            if saved.method == "zhou":
                block = block.loc[[r.guidance == tuning[f"{r.operator}:{r.condition}"]["selected"] for r in block.itertuples()]]
            paired = block.pivot(index="scene_index", columns="condition", values="nmse_linear")
            assert paired.shape == (len(scenes), 2)
            close(10*np.log10(paired.max(axis=1).mean()), saved.worst_variance_mean_nmse_db)
            close((paired.max(axis=1) <= .1).mean(), saved.fraction_below_minus10db)
            close((10*np.log10(paired).max(axis=1)-10*np.log10(paired).min(axis=1)).max(), saved.maximum_numerical_variance_gap_db)
            summaries += 1
        tables[stage] = table
        counts[stage] = {"endpoints": len(table), "cases": len(cases), "trajectories": len(trajectories), "denoiser_predictions": len(means)}
        if stage == "screen":
            means["online_real_mse"] = means.nmse_linear*means.truth_energy_normalized/(2*163840)
            means["offline_variance_real"] = [profile["variances_real"][t-1] for t in means.t]
            shift = means.groupby(["operator", "condition", "method", "t"])[["online_real_mse", "offline_variance_real"]].mean().reset_index()
            shift["online_error_to_offline_variance"] = shift.online_real_mse/shift.offline_variance_real
            shift.to_csv(OUT/"variance_shift.csv", index=False)
            trajectories.loc[trajectories.condition.isin(["noiseless_v1e-08", "snr_20"])].groupby(
                ["operator", "condition", "method", "t"])[METRICS].mean().reset_index().to_csv(OUT/"mean_trajectories.csv", index=False)
    choices = []
    for label, record in tuning.items():
        operator, condition = label.split(":")
        block = tables["tune"].loc[(tables["tune"].operator == operator) & (tables["tune"].condition == condition) & (tables["tune"].method == "zhou")]
        grouped = block.groupby("guidance").nmse_linear.mean()
        assert grouped.idxmin() == record["selected"]
        assert record["at_upper_boundary"] == (record["selected"] == 16.)
        for candidate in record["candidates"]:
            assert candidate["valid"]
            close(grouped[candidate["guidance"]], candidate["mean_nmse_linear"])
            choices.append({"operator": operator, "condition": condition, **candidate, "selected": record["selected"],
                            "mean_nmse_db": 10*np.log10(candidate["mean_nmse_linear"])})
        screen = tables["screen"].loc[(tables["screen"].operator == operator) & (tables["screen"].condition == condition) & (tables["screen"].method == "zhou")]
        assert set(screen.guidance) == {record["selected"]}
    pd.DataFrame(choices).to_csv(OUT/"guidance_grid.csv", index=False)
    matched = 0
    for budget in (128, 4096, 8192):
        original = ROOT/"runs"/f"d041_budget_sweep_20261015_v1_re{budget}_validation"
        old = pd.read_csv(original/"metrics_raw.csv")
        first = tables["tune"].loc[(tables["tune"].pilot_re == budget) & (tables["tune"].method == "zhou") & (tables["tune"].guidance == 4.) & tables["tune"].condition.str.startswith("noiseless_")]
        second = old.loc[(old.estimator == "clean_ddpm") & old.operator.isin(first.operator.unique())]
        merged = first.merge(second, on=["operator", "condition", "scene_index"], suffixes=("_new", "_old"), validate="one_to_one")
        assert len(merged) == len(first) and (merged.guidance_old == 4.).all()
        close(merged[[n+"_new" for n in METRICS]], merged[[n+"_old" for n in METRICS]])
        matched += len(merged)
    assert matched == 320
    write_json(OUT/"source_inputs.json", inputs)
    write_json(OUT/"verification.json", {"passed": True, "counts": counts, "calibration_rows": len(prior),
        "calibration_decomposition_rows": len(decomposition), "source_members": members,
        "old_baseline_endpoints_matched": matched, "summaries_recomputed": summaries,
        "data_files_hashed": len(data.metadata["file_sha256"]), "train_validation_disjoint": True,
        "tuning_screening_disjoint": True, "independent_test": False, "failed_cases": 0})
    print(json.dumps(read(OUT/"verification.json"), indent=2), flush=True)
    data.close()


def preserve():
    inventories = {}
    for name in ("d041_budget_sweep_20261015_v1_review", "d041_config_sweep_20261014_v1_review"):
        inventory = read(ROOT/"research_route/experiment_results/2026-10-15"/name/"source_inputs.json")
        for name_in_run, record in inventory.items():
            path = ROOT/name_in_run
            assert path.stat().st_size == record["size_bytes"] and cached_hash(path) == record["sha256"], name_in_run
        inventories[name] = len(inventory)
        print(f"historical inventory retained: {name}, {len(inventory)} files", flush=True)
    delivery = ROOT/"research_route/experiment_results/2026-10-16/d041_fusion_probe_delivery"
    protected = read(delivery/"protected_files.json")
    for name, digest in protected.items():
        assert cached_hash(ROOT/name) == digest, name
    for name, digest in read(delivery/"delivery_manifest.json").items():
        assert cached_hash(delivery/name) == digest, name
    for name, record in read(OUT/"protected_inputs.json").items():
        assert cached_hash(ROOT/name) == record["sha256"], name
    current = read(OUT/"source_inputs.json")
    for name, record in current.items():
        assert cached_hash(ROOT/name) == record["sha256"], name
    write_json(OUT/"preservation.json", {"passed": True, "historical_inventories": inventories,
        "protected_implementation_files": len(protected), "new_run_files": len(current),
        "previous_delivery_manifest_retained": True, "original_log_retained": True})


if __name__ == "__main__":
    audit()
    preserve()
