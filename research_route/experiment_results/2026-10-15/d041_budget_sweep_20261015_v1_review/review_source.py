"""Independent audit and fixed GPU replay of the completed D-041 budget sweeps."""
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT/"scripts"), str(ROOT/"src")]
from _run_support import hash_file
from audit_zhou2025_migration import cached_hash
from analyze_d041_budget_sweep import verify_report, select_workpoint
import run_d041_diagnostic as diagnostic
from run_d041_budget_sweep import plans, operator_groups
from run_d041_configuration_sweep import summarize, ORIGINAL_SIGNATURE
from run_zhou2025_repro import load_data, load_model, read_json
from zhou2025_repro.data import encode, decode
from zhou2025_repro.ddpm import recover
from zhou2025_repro.d041 import diagnostics, paired_array, hardware_weights

OUT = Path(__file__).parent
PREFIX = "d041_budget_sweep_20261015_v1"
KEY = ["operator", "condition", "scene_id"]
METRICS = ["nmse_linear", "observation_residual_relative", "prediction_energy_ratio",
    "normalized_correlation", "row_space_nmse_linear", "null_space_nmse_linear"]


def audit():
    assert not (OUT/"verification.json").exists(), "retain completed audit"
    inventory, inputs, frames, trace_sums = [], {}, {}, defaultdict(lambda: np.zeros(1+len(METRICS)))
    members, selected_checks, max_selection_relative_gap = 0, 0, 0.
    validation_report = ROOT/"results/d041_budget_sweep"/f"{PREFIX}_validation"
    frozen = read_json(validation_report/"selection.json")
    frozen_time = datetime.fromisoformat(frozen["created_utc"])
    shared_dependencies = None
    for split in ("validation", "test"):
        for budget in (0, 128, 512, 1024, 2048, 4096, 8192):
            tag = "references" if budget == 0 else f"re{budget}"
            folder = ROOT/"runs"/f"{PREFIX}_{tag}_{split}"
            config = read_json(folder/"config_resolved.json")["config"]
            job = config["job"]
            assert read_json(folder/"status.json")["status"] == "complete" and not job["smoke"]
            assert config["band"] == "wide" and config["channel"]["include_absorption"] is False
            assert config["diffusion"]["steps"] == 100 and config["evaluation_run"]["guidance"] == [.5, 1., 2., 4.]
            assert config["configuration_sweep"]["budget_re"] == budget
            count = 32 if split == "validation" else 100
            records = read_json(folder/"operators.json")
            schedules = list(plans(config))
            assert [r["label"] for r in records] == [p[0] for p in schedules]
            with zipfile.ZipFile(folder/"source_snapshot.zip") as snapshot:
                for item in read_json(folder/"code_manifest.json"):
                    assert cached_hash(ROOT/item["path"]) == item["sha256"]
                    assert hashlib.sha256(snapshot.read(item["path"])).hexdigest() == item["sha256"]
                    members += 1
            dependencies = read_json(folder/"dependencies.json")
            if shared_dependencies is None:
                shared_dependencies = dependencies
            assert dependencies == shared_dependencies
            data_folder = ROOT/"runs"/job["data_run_id"]
            assert cached_hash(data_folder/"dataset.json") == dependencies["dataset_sha256"]
            data = read_json(data_folder/"dataset.json")
            assert data["counts"] == {"train": 2048, "validation": 256, "test": 100}
            for name, digest in data["file_sha256"].items():
                assert cached_hash(data_folder/name) == digest
            checkpoint = ROOT/"runs"/job["train_run_id"]/"checkpoint/training.pt"
            assert cached_hash(checkpoint) == dependencies["checkpoint_sha256"]
            model = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert model["step"] == 256000 and model["data_run_id"] == job["data_run_id"]
            tuning = read_json(folder/"tuning.json")
            if split == "test":
                prior = ROOT/"runs"/job["tuning_run_id"]
                previous = read_json(prior/"config_resolved.json")["config"]
                assert ORIGINAL_SIGNATURE(config) == ORIGINAL_SIGNATURE(previous)
                assert config["configuration_sweep"] == previous["configuration_sweep"]
                assert read_json(prior/"dependencies.json") == dependencies
                assert cached_hash(prior/"selection.json") == job["selection_sha256"]
                assert cached_hash(prior/"tuning.json") == read_json(folder/"tuning_source.json")["sha256"]
                assert tuning == read_json(prior/"tuning.json")
                assert read_json(prior/"selection.json") == read_json(folder/"validation_selection.json")
                started = next(line.split(": ", 1)[1] for line in (folder/"environment.txt").read_text().splitlines()
                    if line.startswith("timestamp_utc: "))
                assert datetime.fromisoformat(started) > frozen_time
            else:
                assert read_json(folder/"selection.json")["operator_labels"] == [r["label"] for r in records]
                for value in tuning.values():
                    assert value["validation_scenes"] == 32 and value["candidates"] == [.5, 1., 2., 4.]
                    assert value["selected"] == value["candidates"][int(np.argmin(value["validation_nmse_linear"]))]
            with np.load(folder/"hardware.npz") as hardware:
                np.testing.assert_array_equal(hardware["element_positions_m"], paired_array(config).positions)
                for record, (label, bank, active, metadata) in zip(records, schedules):
                    if bank is None:
                        continue
                    np.testing.assert_array_equal(hardware[label+"_state_indices"], metadata["state_indices"])
                    weights = hardware_weights(config, paired_array(config), bank)
                    assert hashlib.sha256(np.ascontiguousarray(weights).tobytes()).hexdigest() == record["weights_sha256"]
                    if active is not None:
                        np.testing.assert_array_equal(hardware[label+"_active_re_mask"], active)
                        assert active.sum() == budget == record["pilot_re"]
                        assert record["complex_observations"] == 8*budget
                        assert record["observed_frequency_count"] == int(active.any(0).sum())
                        if metadata["state_mode"] == "interleaved":
                            np.testing.assert_array_equal(active.sum(0), budget//128)
                        with np.load(folder/record["spectrum_file"]) as spectrum:
                            np.testing.assert_array_equal(spectrum["active_re_mask"], active)
                            assert int(spectrum["ranks"].sum())*8 == record["rank"]
                    if record["state_mode"] == "repeat":
                        assert record["unique_dma_configurations"] == 1 and record["switches"] == 0
            frame = pd.read_csv(folder/"metrics_raw.csv")
            assert len(frame) == len(records)*count*2*3
            assert not frame.duplicated(KEY+["estimator"]).any()
            assert np.isfinite(frame[METRICS].to_numpy()).all()
            assert frame.observation_noise_variance_normalized.eq(0).all()
            assert frame.physical_element_noise_complex_variance.eq(0).all()
            assert frame.likelihood_variance_normalized.isin([1e-8, 1e-10]).all()
            assert frame.observation_residual_relative.eq(frame.clean_observation_residual_relative).all()
            assert set(frame.scene_index) == set(range(count))
            for record in records:
                block = frame.loc[frame.operator == record["label"]]
                assert len(block) == count*6
                for field in ("rank", "complex_observations", "pilot_re", "total_pilot_energy", "switches", "training_symbols"):
                    assert block[field].eq(record[field]).all()
                for condition, group in block.groupby("condition"):
                    key = record["label"]+":"+condition
                    entry = tuning[key]
                    assert group.guidance.eq(entry["selected"]).all()
                    if split == "validation":
                        actual = float(group.loc[group.estimator == "clean_ddpm"].nmse_linear.mean())
                        expected = entry["validation_nmse_linear"][entry["candidates"].index(entry["selected"])]
                        gap = abs(actual-expected)/max(expected, 1e-30)
                        assert gap < 1e-6
                        selected_checks += 1
                        max_selection_relative_gap = max(max_selection_relative_gap, gap)
            recalculated = summarize(frame, records, {**config, "frozen_tuning": tuning})
            saved = pd.read_csv(folder/"sweep_summary.csv").set_index("operator")
            pd.testing.assert_frame_equal(recalculated.set_index("operator"), saved[recalculated.columns.drop("operator")],
                check_dtype=False, check_like=True, rtol=1e-9, atol=1e-12)
            cases, endpoints = [], []
            paths = sorted((folder/"checkpoint").glob("case_*.json"))
            assert len(paths) == len(records)*count*2
            for path in paths:
                content = path.read_bytes()
                case = json.loads(content)
                assert len(case["trajectory"]) == 101 and case["trajectory"][0]["t"] == 100 and case["trajectory"][-1]["t"] == 0
                cases.extend(case["rows"])
                endpoints.append(case["trajectory"][-1])
                inputs[path.relative_to(ROOT).as_posix()] = {"size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            pd.testing.assert_frame_equal(pd.DataFrame(cases).set_index(KEY+["estimator"]).sort_index(),
                frame.set_index(KEY+["estimator"]).sort_index(), check_dtype=False, check_like=True, rtol=1e-10, atol=1e-12)
            ddpm = frame.loc[frame.estimator == "clean_ddpm"].set_index(KEY).sort_index()
            final = pd.DataFrame(endpoints).set_index(KEY).sort_index()
            np.testing.assert_allclose(final[METRICS], ddpm[METRICS], rtol=1e-10, atol=1e-12)
            total_trace, finals = 0, []
            columns = KEY+["t", "network_calls", "phase", *METRICS, "error_decomposition_relative_gap"]
            for chunk in pd.read_csv(folder/"trajectories.csv", usecols=columns, chunksize=50000):
                step = np.arange(total_trace, total_trace+len(chunk)) % 101
                assert np.array_equal(chunk.t, 100-step) and np.array_equal(chunk.network_calls, step)
                assert np.isfinite(chunk[METRICS].to_numpy()).all()
                assert (chunk.error_decomposition_relative_gap/(1+chunk.nmse_linear) < 5e-6).all()
                finals.append(chunk.loc[chunk.t == 0])
                sample = chunk.loc[chunk.t.isin([100, 75, 50, 25, 10, 5, 1, 0])]
                for key, group in sample.groupby(["operator", "condition", "t"]):
                    trace_sums[(split, *key)] += np.r_[len(group), group[METRICS].sum().to_numpy()]
                total_trace += len(chunk)
            assert total_trace == len(records)*count*2*101
            final_csv = pd.concat(finals).set_index(KEY).sort_index()
            assert final_csv.index.equals(ddpm.index)
            np.testing.assert_allclose(final_csv[METRICS], ddpm[METRICS], rtol=1e-10, atol=1e-12)
            for path in folder.rglob("*"):
                if path.is_file() and path.relative_to(ROOT).as_posix() not in inputs:
                    inputs[path.relative_to(ROOT).as_posix()] = {"size_bytes": path.stat().st_size, "sha256": cached_hash(path)}
            frame.to_csv(OUT/(tag+"_"+split+"_endpoints.csv"), index=False)
            frames[(budget, split)] = frame
            inventory.append({"run_id": folder.name, "budget_re": budget, "split": split, "operators": len(records),
                "rows": len(frame), "trajectories": total_trace, "cases": len(paths)})
            print("verified", inventory[-1], flush=True)
    for split in ("validation", "test"):
        verify_report(ROOT/"results/d041_budget_sweep"/f"{PREFIX}_{split}")
    combined = pd.read_csv(validation_report/"budget_summary.csv")
    assert select_workpoint(combined, smoke=False, required_fraction=.95) == frozen["workpoint"]
    assert frozen["workpoint"]["operator"] == "re8192_J64_K128_distinct"
    prior_inputs = read_json(ROOT/"research_route/experiment_results/2026-10-15/d041_config_sweep_20261014_v1_review/source_inputs.json")
    for name, item in prior_inputs.items():
        assert (ROOT/name).stat().st_size == item["size_bytes"] and cached_hash(ROOT/name) == item["sha256"]
    old = pd.read_csv(ROOT/"runs/d041_config_sweep_20261014_v1_wb_re128_test/metrics_raw.csv")
    new = frames[(128, "test")].loc[lambda f: f.operator.str.startswith("legacy_")].copy()
    new["operator"] = new.operator.str.removeprefix("legacy_")
    a, b = old.set_index(KEY+["estimator"]).sort_index(), new.set_index(KEY+["estimator"]).sort_index()
    np.testing.assert_allclose(a[METRICS], b[METRICS], rtol=1e-8, atol=1e-12)
    trajectory_rows = []
    for (split, label, condition, t), sums in trace_sums.items():
        trajectory_rows.append({"split": split, "operator": label, "condition": condition, "t": t,
            "scenes": int(sums[0]), **dict(zip(METRICS, sums[1:]/sums[0]))})
    pd.DataFrame(trajectory_rows).to_csv(OUT/"trajectory_summary.csv", index=False)
    result = {"passed": True, "runs": inventory, "rows": sum(i["rows"] for i in inventory),
        "trajectories": sum(i["trajectories"] for i in inventory), "cases": sum(i["cases"] for i in inventory),
        "source_snapshot_members": members, "selected_validation_objectives_checked": selected_checks,
        "maximum_selected_objective_relative_gap": max_selection_relative_gap,
        "original_files_unchanged": len(prior_inputs), "source_input_files": len(inputs),
        "legacy_128_re_metric_maximum_absolute_delta": float(np.max(abs(a[METRICS].to_numpy()-b[METRICS].to_numpy()))),
        "validation_workpoint_frozen_before_all_tests": True}
    assert (result["rows"], result["trajectories"], result["cases"]) == (63360, 2133120, 21120)
    (OUT/"source_inputs.json").write_text(json.dumps(inputs, ensure_ascii=False, indent=2)+"\n")
    (OUT/"verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


@torch.no_grad()
def replay():
    assert not (OUT/"replay.json").exists() and torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    blocks = [("references", ["digital", "dma_J192"]),
        ("re128", ["re128_J16_K8_distinct", "legacy_re128_J16_K8_distinct"]),
        ("re4096", ["re4096_J64_K64_distinct", "re4096_J128_K32_distinct", "re4096_J128_K32_interleaved"]),
        ("re8192", ["re8192_J64_K128_distinct", "re8192_J64_K128_repeat", "re8192_J128_K64_distinct", "re8192_J128_K64_interleaved"])]
    results = []
    for tag, labels in blocks:
        folder = ROOT/"runs"/f"{PREFIX}_{tag}_test"
        config = read_json(folder/"config_resolved.json")["config"]
        config["job"]["frozen_operator_labels"] = labels
        model, schedule, _, data = load_model(config["job"]["train_run_id"], "cuda:0")
        frame = pd.read_csv(folder/"metrics_raw.csv")
        for label, op, _ in operator_groups(config, "full-noiseless"):
            for index in (0, 99):
                row = frame.loc[(frame.operator == label) & (frame.scene_index == index) &
                    (frame.condition == "noiseless_v1e-08") & (frame.estimator == "clean_ddpm")].iloc[0]
                truth = torch.as_tensor(data.physical("test", [index]), device="cuda:0")/data.scale
                clean = op.forward(encode(truth, "dma"))
                image, _ = recover(model, schedule, op, clean, float(row.likelihood_variance_normalized),
                    scale=float(row.guidance), seed=diagnostic.posterior_seed(config, label, index, "test"), refinement=1)
                checked = diagnostics(op, truth, decode(image, "dma"), clean, clean)
                np.testing.assert_allclose([checked[k] for k in METRICS], row[METRICS].to_numpy(dtype=float), rtol=5e-5, atol=1e-10)
                results.append({"operator": label, "scene_index": index, "nmse_db_difference": checked["nmse_db"]-float(row.nmse_db)})
        print("replayed", tag, flush=True)
    assert len(results) == 22
    result = {"passed": True, "fixed_cases": len(results), "max_nmse_db_difference": max(abs(r["nmse_db_difference"]) for r in results),
        "scope": "fixed frozen endpoints at scenes 0/99; no selection or tuning", "results": results}
    (OUT/"replay.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}), flush=True)


if __name__ == "__main__":
    replay() if "--gpu" in sys.argv else audit()
