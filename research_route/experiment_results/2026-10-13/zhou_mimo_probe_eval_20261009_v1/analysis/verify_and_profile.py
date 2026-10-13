"""Verify the implemented short pipelines and profile frozen DDPM weights."""
import ast
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_zhou2025_repro as runner
from _run_support import hash_file
from zhou2025_repro.ddpm import recover
from zhou2025_repro.data import decode, angular
from zhou2025_repro.baselines import lasso, real_operator


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


torch.set_num_threads(4)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
out = Path(__file__).resolve().parent
runs = sorted(p for p in (ROOT / "runs").glob("zhou_*20261009_v1")
              if p.name.startswith(("zhou_mimo_", "zhou_dma_")))
assert len(runs) == 35
checks = {"complete_runs": 0, "verified_source_snapshots": 0, "verified_snapshot_files": 0,
          "verified_datasets": 0, "verified_trained_models": 0, "evaluation_rows": {},
          "evaluation_metrics_sha256": {}}
profiles, sure_rows, inventories = [], [], []
data = {}
for path in runs:
    assert read(path / "status.json")["status"] == "complete", path
    checks["complete_runs"] += 1
    config = read(path / "config_resolved.json")["config"]
    inventories.append({"run_id": path.name, "command": config["job"]["command"],
                        "scenario": config["scenario"], "smoke_only": config["job"]["smoke"]})
    snapshot = path / "source_snapshot.zip"
    if snapshot.exists():
        with zipfile.ZipFile(snapshot) as archive:
            for entry in read(path / "code_manifest.json"):
                payload = archive.read(entry["path"].replace("\\", "/"))
                assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
                checks["verified_snapshot_files"] += 1
        checks["verified_source_snapshots"] += 1
    if (path / "dataset.json").exists():
        data[path.name] = runner.load_data(path.name)
        checks["verified_datasets"] += 1
    if (path / "training_summary.json").exists():
        model, schedule, checkpoint, source = runner.load_model(path.name, config["device"])
        checks["verified_trained_models"] += 1
        if checkpoint["kind"] == "dm" and "probe" in path.name:
            for row in runner.validation_profile(model, schedule, source, config):
                profiles.append({"model": path.name, **row})
    if (path / "metrics_raw.csv").exists():
        frame = pd.read_csv(path / "metrics_raw.csv")
        assert not frame.duplicated(["scene_id", "schedule", "snr_db", "estimator"]).any()
        assert np.isfinite(frame.nmse_linear).all() and (frame.nmse_linear > 0).all()
        assert set(frame.split) == {"validation"}
        expected = 192 if "probe" in path.name else (56 if config["scenario"] == "mimo" else 48)
        assert len(frame) == expected
        by_observation = frame.groupby(["scene_id", "schedule", "snr_db"])
        for column in ("complex_noise_variance_phys", "actual_whitened_snr_db"):
            assert by_observation[column].nunique().eq(1).all()
        if config["scenario"] == "dma":
            assert frame.loc[frame.heldout_112_nmse_subcarrier_count == 0, "heldout_112_nmse_linear"].isna().all()
            assert frame.loc[frame.heldout_112_nmse_subcarrier_count == 112, "heldout_112_nmse_linear"].notna().all()
            assert np.isfinite(frame.net_rate_bps_hz).all()
        checks["evaluation_rows"][path.name] = len(frame)
        checks["evaluation_metrics_sha256"][path.name] = hash_file(path / "metrics_raw.csv")

clean = data["zhou_mimo_probe_data_20261009_v1"].physical("validation", range(32))
for name, item in data.items():
    if "probe_denoised" not in name:
        continue
    noisy = data[item.metadata["parent_run_id"]].physical("validation", range(32))
    estimates = item.physical("validation", range(32))
    for label, values in (("raw_noisy", noisy), ("sure_denoised", estimates)):
        errors = [runner.nmse(h, e) for h, e in zip(clean, values)]
        sure_rows.append({"dataset": name, "variant": item.metadata["sure_variant"],
            "noise_variance": data[item.metadata["parent_run_id"]].metadata["requested_canonical_real_variance"],
            "estimator": label, "scenes": len(errors), "nmse_db": float(10*np.log10(np.mean(errors)))})

recovery_errors = []
for identifier in ("zhou_mimo_probe_eval_20261009_v1", "zhou_dma_smoke_eval_20261009_v1"):
    path = ROOT / "runs" / identifier
    config = read(path / "config_resolved.json")["config"]
    target = data[config["job"]["data_run_id"]]
    model_id = config["job"]["train_run_id"][0]
    model, schedule, _, model_data = runner.load_model(model_id, config["device"])
    frame = pd.read_csv(path / "metrics_raw.csv")
    operators, _ = runner.build_operators(config, target)
    for op_index, (label, operator, _) in enumerate(operators):
        with torch.no_grad():
            image = target.images("validation", range(config["evaluation_run"]["scene_limit"]), config["device"])
            observed_clean = operator.forward(image)
        noise = runner.standard_noise(observed_clean, config["evaluation_run"], op_index, "validation")
        for snr in config["evaluation_run"]["snr_db"]:
            row = frame[(frame.schedule == label) & (frame.snr_db == snr) & (frame.estimator == model_id)].iloc[0]
            variance = row.complex_noise_variance_phys / target.scale**2
            ratio = target.scale/model_data.scale
            seed = int(np.random.SeedSequence([config["evaluation_run"]["posterior_seed"], op_index, 0, 0]).generate_state(1)[0])
            prediction, _ = recover(model, schedule, operator,
                (observed_clean[:1]+np.sqrt(variance)*noise[:1])*ratio, variance*ratio**2,
                scale=row.guidance, seed=seed, refinement=config["evaluation_run"]["refinement"])
            prediction = (decode(prediction, target.scenario)*model_data.scale)[0].cpu().numpy()
            actual = runner.nmse(target.physical("validation", [0])[0], prediction)
            recovery_errors.append(abs(10*np.log10(actual)-row.nmse_db))
assert max(recovery_errors) < 1e-5
checks["current_recovery_rechecks"] = len(recovery_errors)
checks["max_recovery_difference_db"] = max(recovery_errors)

old = subprocess.check_output(["git", "show", "3a767ff368e121cd59be271706f33e2e5cc2db10:scripts/check_zhou_reference_curves.py"], cwd=ROOT, text=True)
nodes = [node for node in ast.parse(old).body if isinstance(node, ast.FunctionDef)
         and node.name in ("angular", "real_operator", "lasso")]
original = {"np": np, "torch": torch}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "archived_reference_helpers", "exec"), original)
torch.manual_seed(912)
y, b = torch.randn(2, 3, 4, dtype=torch.complex128), torch.randn(6, 4, dtype=torch.complex128)
for penalty in (0., .1, 1.):
    before, after = original["lasso"](y, b, penalty), lasso(y, b, penalty)
    torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)
    assert before[1:] == after[1:]
checks["extracted_lasso_exact_matches"] = 3
for name, function in (("angular", angular), ("real_operator", real_operator)):
    torch.testing.assert_close(original[name](b), function(b), rtol=0, atol=0)
pd.DataFrame(profiles).to_csv(out / "denoising_profiles.csv", index=False)
pd.DataFrame(sure_rows).to_csv(out / "sure_denoising.csv", index=False)
pd.DataFrame(inventories).to_csv(out / "run_inventory.csv", index=False)
checks["denoising_profile_rows"] = len(profiles)
checks["sure_denoising_rows"] = len(sure_rows)
assert len(profiles) == 35 and len(sure_rows) == 8
for identifier, digest in checks["evaluation_metrics_sha256"].items():
    assert hash_file(ROOT / "runs" / identifier / "metrics_raw.csv") == digest
(out / "integrity.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
for item in data.values():
    item.close()
print(json.dumps(checks, indent=2))
print(pd.DataFrame(sure_rows).to_string(index=False))
