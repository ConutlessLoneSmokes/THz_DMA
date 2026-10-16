"""Read-only audit of the six completed D-041 configuration sweeps."""
from pathlib import Path
import hashlib
import json
import math
import sys
import zipfile

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT/"scripts"), str(ROOT/"src")]
from _run_support import hash_file
from audit_zhou2025_migration import cached_hash
import run_d041_diagnostic as diagnostic
from run_d041_configuration_sweep import operator_groups, select_test_operators, summarize, ORIGINAL_SIGNATURE
from run_zhou2025_repro import load_data, load_model, read_json
from zhou2025_repro.data import decode, encode
from zhou2025_repro.ddpm import recover
from zhou2025_repro.d041 import diagnostics

OUT = Path(__file__).parent
PREFIX = "d041_config_sweep_20261014_v1"
KEY = ["operator", "condition", "scene_id"]
METRICS = ["nmse_linear", "observation_residual_relative", "prediction_energy_ratio",
    "normalized_correlation", "row_space_nmse_linear", "null_space_nmse_linear"]


def audit():
    assert not (OUT/"verification.json").exists(), "retain previous audit"
    inventory, preserved, frames, source_members = [], {}, {}, 0
    for folder in sorted((ROOT/"runs").glob(PREFIX+"_*")):
        config = read_json(folder/"config_resolved.json")["config"]
        job, split = config["job"], config["job"]["split"]
        assert read_json(folder/"status.json")["status"] == "complete" and not job["smoke"]
        assert config["channel"]["include_absorption"] is False
        assert config["diffusion"]["steps"] == 100 and config["evaluation_run"]["guidance"] == [.5, 1., 2., 4.]
        count = 32 if split == "validation" else 100
        records = read_json(folder/"operators.json")
        with zipfile.ZipFile(folder/"source_snapshot.zip") as snapshot:
            for item in read_json(folder/"code_manifest.json"):
                assert cached_hash(ROOT/item["path"]) == item["sha256"]
                assert hashlib.sha256(snapshot.read(item["path"])).hexdigest() == item["sha256"]
                source_members += 1
        dependencies = read_json(folder/"dependencies.json")
        assert dependencies["data_run_id"] == job["data_run_id"] and dependencies["train_run_id"] == job["train_run_id"]
        assert cached_hash(ROOT/"runs"/job["data_run_id"]/"dataset.json") == dependencies["dataset_sha256"]
        assert cached_hash(ROOT/"runs"/job["train_run_id"]/"checkpoint/training.pt") == dependencies["checkpoint_sha256"]
        data = read_json(ROOT/"runs"/job["data_run_id"]/"dataset.json")
        assert data["counts"] == {"train": 2048, "validation": 256, "test": 100}
        for name, digest in data["file_sha256"].items():
            assert cached_hash(ROOT/"runs"/job["data_run_id"]/name) == digest
        model = torch.load(ROOT/"runs"/job["train_run_id"]/"checkpoint/training.pt", map_location="cpu", weights_only=True)
        assert model["step"] == 256000 and model["data_run_id"] == job["data_run_id"]
        tuning = read_json(folder/"tuning.json")
        if split == "test":
            prior = ROOT/"runs"/job["tuning_run_id"]
            previous = read_json(prior/"config_resolved.json")["config"]
            assert read_json(prior/"status.json")["status"] == "complete"
            assert ORIGINAL_SIGNATURE(config) == ORIGINAL_SIGNATURE(previous)
            assert config["configuration_sweep"] == previous["configuration_sweep"]
            assert read_json(prior/"dependencies.json") == dependencies
            assert hash_file(prior/"selection.json") == job["selection_sha256"]
            assert hash_file(prior/"tuning.json") == read_json(folder/"tuning_source.json")["sha256"]
            assert {r["label"] for r in records} == set(job["frozen_operator_labels"])
            assert tuning == {k: v for k, v in read_json(prior/"tuning.json").items() if k.split(":")[0] in job["frozen_operator_labels"]}
            assert read_json(prior/"selection.json") == read_json(folder/"validation_selection.json")
        else:
            assert len(records) == (202 if job["experiment"] == "curve" else 13)
            if job["experiment"] == "curve":
                assert sorted(r["configurations"] for r in records if r["state_mode"] == "distinct") == list(range(1, 193))
            for value in tuning.values():
                assert value["validation_scenes"] == 32 and value["candidates"] == [.5, 1., 2., 4.]
                assert value["selected"] == value["candidates"][int(np.argmin(value["validation_nmse_linear"]))]
        frame = pd.read_csv(folder/"metrics_raw.csv")
        assert len(frame) == len(records)*count*2*3
        assert not frame.duplicated(KEY+["estimator"]).any()
        assert np.isfinite(frame[METRICS].to_numpy()).all()
        assert (frame.observation_noise_variance_normalized == 0).all()
        assert (frame.physical_element_noise_complex_variance == 0).all()
        assert frame.likelihood_variance_normalized.isin([1e-8, 1e-10]).all()
        assert (frame.observation_residual_relative == frame.clean_observation_residual_relative).all()
        assert set(frame.scene_index) == set(range(count)) and frame.model_run_id.eq(job["train_run_id"]).all()
        for record in records:
            block = frame.loc[frame.operator == record["label"]]
            assert len(block) == count*2*3
            for column in ("rank", "complex_observations", "pilot_re", "total_pilot_energy", "switches", "training_symbols"):
                assert block[column].eq(record[column]).all()
            assert (record["frontend"] == "digital" or record["complex_observations"] == 8*record["pilot_re"])
            for condition, group in block.groupby("condition"):
                assert group.guidance.eq(tuning[record["label"]+":"+condition]["selected"]).all()
            if record["state_mode"] == "repeat":
                assert record["unique_dma_configurations"] == 1 and record["switches"] == 0
            if job["experiment"] == "re128":
                assert record["pilot_re"] == 128 and record["complex_observations"] == 1024
                active = np.zeros((record["configurations"], 128), dtype=bool)
                for j, indices in enumerate(record["pilot_indices"]):
                    active[j, indices] = True
                assert active.sum() == 128
                expected_rank = 8*(128//record["configurations"]) if record["state_mode"] == "repeat" else 1024
                assert record["rank"] == expected_rank
        recalculated = summarize(frame, records, {**config, "frozen_tuning": tuning})
        saved = pd.read_csv(folder/"sweep_summary.csv")
        pd.testing.assert_frame_equal(recalculated, saved, check_dtype=False, rtol=1e-10, atol=1e-12)
        if split == "validation":
            assert select_test_operators(recalculated, config) == read_json(folder/"selection.json")
        endpoints, case_rows = [], []
        case_paths = sorted((folder/"checkpoint").glob("case_*.json"))
        assert len(case_paths) == len(records)*count*2
        for number, path in enumerate(case_paths):
            raw = path.read_bytes()
            preserved[path.relative_to(ROOT).as_posix()] = {"size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            case = json.loads(raw)
            assert len(case["rows"]) == 3 and len(case["trajectory"]) == 101
            assert case["trajectory"][0]["t"] == 100 and case["trajectory"][-1]["t"] == 0
            case_rows.extend(case["rows"])
            endpoints.append(case["trajectory"][-1])
            if number and number % 4000 == 0:
                print(folder.name, "cases", number, flush=True)
        case_frame = pd.DataFrame(case_rows).set_index(KEY+["estimator"]).sort_index()
        pd.testing.assert_frame_equal(case_frame, frame.set_index(KEY+["estimator"]).sort_index(),
            check_dtype=False, check_like=True, rtol=1e-10, atol=1e-12)
        ddpm = frame.loc[frame.estimator == "clean_ddpm"].set_index(KEY).sort_index()
        final = pd.DataFrame(endpoints).set_index(KEY).sort_index()
        np.testing.assert_allclose(final[METRICS], ddpm[METRICS], rtol=1e-10, atol=1e-12)
        rows, final_parts = 0, []
        columns = KEY+["t", "network_calls", "phase", *METRICS, "error_decomposition_relative_gap"]
        for chunk in pd.read_csv(folder/"trajectories.csv", usecols=columns, chunksize=50000):
            step = np.arange(rows, rows+len(chunk)) % 101
            assert np.array_equal(chunk.t, 100-step) and np.array_equal(chunk.network_calls, step)
            assert np.array_equal(chunk.phase, np.where(step == 0, "initial", "updated"))
            assert np.isfinite(chunk[METRICS].to_numpy()).all()
            assert (chunk.error_decomposition_relative_gap/(1+chunk.nmse_linear) < 5e-6).all()
            final_parts.append(chunk.loc[chunk.t == 0])
            rows += len(chunk)
        assert rows == len(records)*count*2*101
        final_csv = pd.concat(final_parts).set_index(KEY).sort_index()
        assert final_csv.index.equals(ddpm.index)
        np.testing.assert_allclose(final_csv[METRICS], ddpm[METRICS], rtol=1e-10, atol=1e-12)
        for path in folder.rglob("*"):
            if path.is_file() and path.relative_to(ROOT).as_posix() not in preserved:
                preserved[path.relative_to(ROOT).as_posix()] = {"size_bytes": path.stat().st_size, "sha256": cached_hash(path)}
        recalculated.to_csv(OUT/(folder.name.removeprefix(PREFIX+"_")+"_summary.csv"), index=False)
        frames[folder.name] = frame
        inventory.append({"run_id": folder.name, "band": config["band"], "split": split, "operators": len(records),
            "rows": len(frame), "trajectories": rows, "cases": len(case_paths), "model_updates": model["step"],
            "data_run_id": job["data_run_id"], "model_run_id": job["train_run_id"], "best_step": model["best_step"]})
        print("verified", inventory[-1], flush=True)
    assert len(inventory) == 6 and sum(r["rows"] for r in inventory) == 105264
    assert sum(r["trajectories"] for r in inventory) == 3543888
    old = read_json(ROOT/"research_route/experiment_results/2026-10-14/d041_noabs_20261013_v2_noiseless_review/source_inputs.json")
    for name, item in old.items():
        assert (ROOT/name).stat().st_size == item["size_bytes"] and cached_hash(ROOT/name) == item["sha256"]
    result = {"passed": True, "runs": inventory, "source_snapshot_members": source_members,
        "original_files_unchanged": len(old), "source_input_files": len(preserved), "rows": 105264,
        "trajectories": 3543888, "cases": 35088, "scope": "complete raw metrics/cases/trajectories, frozen choices and source/dependency hashes"}
    (OUT/"source_inputs.json").write_text(json.dumps(preserved, ensure_ascii=False, indent=2)+"\n")
    (OUT/"verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


@torch.no_grad()
def replay():
    assert not (OUT/"replay.json").exists()
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    results = []
    for suffix, labels in (("nb_curve_test", ["dma_J67", "dma_J161", "dma_J160"]),
            ("wb_curve_test", ["dma_J56", "dma_J163", "dma_J162"]),
            ("wb_re128_test", ["re128_J1_K128_distinct", "re128_J8_K16_distinct",
                "re128_J16_K8_distinct", "re128_J16_K8_interleaved"])):
        folder = ROOT/"runs"/(PREFIX+"_"+suffix)
        config = read_json(folder/"config_resolved.json")["config"]
        config["job"]["frozen_operator_labels"] = labels
        frame = pd.read_csv(folder/"metrics_raw.csv")
        model, schedule, _, data = load_model(config["job"]["train_run_id"], "cuda:0")
        for label, op, _ in operator_groups(config, "full-noiseless"):
            for index in (0, 99):
                row = frame.loc[(frame.operator == label) & (frame.scene_index == index) &
                    (frame.condition == "noiseless_v1e-08") & (frame.estimator == "clean_ddpm")].iloc[0]
                truth = torch.as_tensor(data.physical("test", [index]), device="cuda:0")/data.scale
                clean = op.forward(encode(truth, "dma"))
                image, _ = recover(model, schedule, op, clean, float(row.likelihood_variance_normalized),
                    scale=float(row.guidance), seed=diagnostic.posterior_seed(config, label, index, "test"), refinement=1)
                checked = diagnostics(op, truth, decode(image, "dma"), clean, clean)
                np.testing.assert_allclose([checked[key] for key in METRICS], row[METRICS].to_numpy(dtype=float),
                    rtol=5e-5, atol=1e-10)
                results.append({"run_id": folder.name, "operator": label, "scene_index": index,
                    "variance": float(row.likelihood_variance_normalized), "guidance": float(row.guidance),
                    "nmse_db_difference": checked["nmse_db"]-float(row.nmse_db)})
        print("replayed", suffix, flush=True)
    assert len(results) == 20
    output = {"passed": True, "fixed_cases": 20, "max_nmse_db_difference": max(abs(r["nmse_db_difference"]) for r in results),
        "results": results, "scope": "fixed frozen endpoints; no test-time tuning or selection"}
    (OUT/"replay.json").write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({key: value for key, value in output.items() if key != "results"}), flush=True)


if __name__ == "__main__":
    replay() if "--gpu" in sys.argv else audit()
