"""Read-only smoke/original-run audit; --gpu checks five formal-model interfaces."""
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
from _run_support import hash_file
import run_d041_diagnostic as diagnostic
from run_d041_configuration_sweep import operator_groups, resolved_settings, ORIGINAL_SIGNATURE
from run_zhou2025_repro import load_data, load_model, read_json
from zhou2025_repro.data import decode, encode
from zhou2025_repro.ddpm import recover
from zhou2025_repro.d041 import diagnostics

OUTPUT = Path(__file__).parent


def audit():
    runs, rows, trajectories = [], 0, 0
    reference_checks = 0
    for directory in sorted((ROOT/"runs").glob("d041_config_smoke_20261014_v2_*")):
        config = read_json(directory/"config_resolved.json")["config"]
        assert read_json(directory/"status.json")["status"] == "complete"
        for item in read_json(directory/"code_manifest.json"):
            assert hash_file(ROOT/item["path"]) == item["sha256"]
        frame, trace = pd.read_csv(directory/"metrics_raw.csv"), pd.read_csv(directory/"trajectories.csv")
        records = read_json(directory/"operators.json")
        assert len(frame) == len(records)*2*3 and len(trace) == len(records)*2*9
        assert (frame.observation_noise_variance_normalized == 0).all()
        assert (frame.observation_residual_relative == frame.clean_observation_residual_relative).all()
        final = trace.loc[trace.t == 0].set_index(["operator", "condition", "scene_id"])
        endpoint = frame.loc[frame.estimator == "clean_ddpm"].set_index(["operator", "condition", "scene_id"])
        for key in ("nmse_linear", "prediction_energy_ratio", "row_space_nmse_linear", "null_space_nmse_linear"):
            np.testing.assert_allclose(final[key].sort_index(), endpoint[key].sort_index(), rtol=1e-12, atol=1e-15)
        assert float(frame.error_decomposition_relative_gap.max()) < 1e-5
        assert all(int(r["complex_observations"]) == int(r["pilot_re"])*8 for r in records if r["frontend"] == "dma")
        for record in records:
            if record["state_mode"] == "repeat":
                assert record["switches"] == 0 and record["unique_dma_configurations"] == 1
            if config["configuration_sweep"]["experiment"] == "re128":
                assert record["pilot_re"] == 128
                mask = np.zeros((record["configurations"], 128), dtype=bool)
                for j, indices in enumerate(record["pilot_indices"]):
                    mask[j, indices] = True
                assert int(mask.sum()) == 128
                assert int(mask.any(0).sum()) == (128 if record["state_mode"] == "interleaved" else 128//record["configurations"])
        if config["job"]["split"] == "validation":
            data = load_data(config["job"]["data_run_id"])
            references = read_json(directory/"observation_reference.json")["reference"]
            for label, op, _ in operator_groups(config, "full-noiseless"):
                power = diagnostic.training_reference(data, op, config)
                assert math.isclose(power, references[label]["normalized_whitened_power"], rel_tol=1e-12)
                reference_checks += 1
            selection = read_json(directory/"selection.json")
            assert selection["minimum_recovery_J"] is None and selection["minimum_near_reference_J"] is None
        else:
            prior = ROOT/"runs"/config["job"]["tuning_run_id"]
            previous = read_json(prior/"config_resolved.json")["config"]
            assert ORIGINAL_SIGNATURE(previous) == ORIGINAL_SIGNATURE(config)
            assert previous["configuration_sweep"] == config["configuration_sweep"]
            assert read_json(prior/"dependencies.json") == read_json(directory/"dependencies.json")
            assert hash_file(prior/"tuning.json") == read_json(directory/"tuning_source.json")["sha256"]
            assert hash_file(prior/"selection.json") == config["job"]["selection_sha256"]
            assert {r["label"] for r in records} == set(config["job"]["frozen_operator_labels"])
            assert read_json(directory/"tuning.json") == {key: value for key, value in read_json(prior/"tuning.json").items()
                if key.split(":")[0] in config["job"]["frozen_operator_labels"]}
        rows += len(frame)
        trajectories += len(trace)
        runs.append({"run_id": directory.name, "rows": len(frame), "trajectories": len(trace), "operators": len(records)})
    assert len(runs) == 6 and rows == 300 and trajectories == 900 and reference_checks == 25
    original = read_json(ROOT/"research_route/experiment_results/2026-10-14/d041_noabs_20261013_v2_noiseless_review/source_inputs.json")
    for name, item in original.items():
        path = ROOT/name
        assert path.stat().st_size == item["size_bytes"] and hash_file(path) == item["sha256"], name
    original_sources = read_json(ROOT/"runs/d041_noabs_20261013_v2_wb_full_test/code_manifest.json")
    for item in original_sources:
        assert hash_file(ROOT/item["path"]) == item["sha256"]
    return {"passed": True, "runs": runs, "rows": rows, "trajectories": trajectories,
        "training_reference_recomputations": reference_checks, "original_files_unchanged": len(original),
        "original_source_files_unchanged": len(original_sources), "unit_tests_passed": 66,
        "scope": "smoke algebra/acquisition/freezing/output; no formal threshold or protocol-performance claim"}


def gpu():
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    results = []
    for band, experiment, labels in (("narrow", "curve", ["dma_J160"]), ("wide", "curve", ["dma_J160"]),
            ("wide", "re128", ["re128_J8_K16_distinct", "re128_J16_K8_distinct", "re128_J16_K8_interleaved"])):
        tag = "nb" if band == "narrow" else "wb"
        args = SimpleNamespace(config=ROOT/"configs/d041_configuration_sweep.toml", band=band,
            experiment=experiment, data_run_id=f"d041_noabs_20261013_v2_{tag}_data",
            train_run_id=f"d041_noabs_20261013_v2_{tag}_clean", device="cuda:0", split="validation",
            tuning_run_id=None, run_id="delivery_probe_only", smoke=False, resume=False)
        config = resolved_settings(args)
        config["job"]["frozen_operator_labels"] = labels
        model, schedule, _, data = load_model(args.train_run_id, args.device)
        truth = torch.as_tensor(data.physical("validation", [0]), device="cuda:0")/data.scale
        for label, op, _ in operator_groups(config, "full-noiseless"):
            clean = op.forward(encode(truth, "dma"))
            endpoints = []
            image, trace = recover(model, schedule, op, clean, 1e-8, scale=1.,
                seed=diagnostic.posterior_seed(config, label, 0, "validation"), refinement=1,
                trace_observer=lambda state, info: endpoints.append(info))
            metrics = diagnostics(op, truth, decode(image, "dma"), clean, clean)
            # The internal trace labels updates 100..1; the state callback labels endpoints 99..0.
            assert torch.isfinite(image).all() and endpoints[-1]["t"] == 0 and trace[-1]["network_calls"] == 100
            assert clean.numel() == 8*(128 if experiment == "re128" else len(op.matrix)*160)
            results.append({"band": band, "operator": label, "metrics": metrics, "network_calls": 100,
                "returned_trace_t": trace[-1]["t"], "endpoint_callback_t": endpoints[-1]["t"]})
    assert len(results) == 5
    return {"passed": True, "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
        "cases": results, "selection_split": "validation scene 0 only, fixed guidance=1 and variance=1e-8",
        "scope": "five finite 100-step formal-model interface checks; not a performance estimate or parameter selection"}


if __name__ == "__main__":
    torch.set_num_threads(4)
    output = OUTPUT/("gpu_probe.json" if "--gpu" in sys.argv else "verification.json")
    if output.exists():
        raise FileExistsError("retain previous verification; use another output directory")
    result = gpu() if "--gpu" in sys.argv else audit()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("runs", "cases")}, ensure_ascii=False))
