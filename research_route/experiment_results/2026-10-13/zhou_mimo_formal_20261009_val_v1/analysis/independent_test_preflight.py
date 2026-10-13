"""Check D-038 frozen inputs and old validation outputs without evaluating test CSI."""
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/"scripts"), str(ROOT/"src")]
import run_zhou2025_repro as r
from zhou2025_repro.data import decode


def main():
    folder = Path(__file__).resolve().parent.parent
    config = r.read_json(folder/"config_resolved.json")["config"]
    settings, job = config["evaluation_run"], config["job"]
    supplied = r.read_toml(ROOT/"configs/zhou2025_parametric_mimo.toml")
    assert all(config[key] == value for key, value in supplied.items())
    assert r.read_json(folder/"status.json")["status"] == "complete"
    assert job["split"] == "validation" and len(job["train_run_id"]) == 7
    assert settings["pilot_counts"] == [64, 38]
    assert settings["snr_db"] == list(np.arange(-10., 31., 5.))
    assert settings["scene_limit"] == 100 and settings["refinement"] == 1
    test_id = "zhou_mimo_independent_test_20261010_v1"
    assert not (ROOT/"runs"/test_id).exists()
    original = {name:r.hash_file(folder/name) for name in
                ("metrics_raw.csv", "tuning.json", "dependencies.json", "config_resolved.json")}
    assert original["metrics_raw.csv"] == "d49daea869ad243e86bb9df21030d1fbdc124fc9479a2b233b15ef7576bafc20"
    assert original["tuning.json"] == "51b09c7d4e66ba8c0020ce9b8a538a9469b9b9a8abd9beb33759eafb0d76bbe0"
    tuning = r.read_json(folder/"tuning.json")
    keys = {f"QPSK_Np{n}:{snr:g}:{name}" for n in settings["pilot_counts"]
            for snr in settings["snr_db"] for name in job["train_run_id"]+["lasso"]}
    assert set(tuning) == keys and len(keys) == 144
    for item in tuning.values():
        assert np.isfinite(item["validation_nmse"]).all()
        assert item["selected"] == item["candidates"][int(np.argmin(item["validation_nmse"]))]
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    target = r.load_data(job["data_run_id"])
    assert target.counts == {"train":4096, "validation":256, "test":100}
    assert len(set(target.metadata["split_seeds"].values())) == 3
    models = {name:r.load_model(name, config["device"]) for name in job["train_run_id"]}
    datasets = {item[3].directory.name:item[3] for item in models.values()}
    datasets[target.directory.name] = target
    covariance_data = {target.directory.name:target}
    for model, schedule, checkpoint, source in models.values():
        r.verify_physics(config, source)
        assert checkpoint["diffusion"] == config["diffusion"] and checkpoint["kind"] == "dm"
        if source.metadata["kind"] == "noisy":
            covariance_data[source.directory.name] = source
    dependencies = r.dependency_record(target, models, covariance_data)
    assert dependencies == r.read_json(folder/"dependencies.json")
    assert len(datasets) == 7 and sum(len(d.metadata["file_sha256"]) for d in datasets.values()) == 300
    manifest = r.read_json(folder/"code_manifest.json")
    changes = [item["path"].replace("\\", "/") for item in manifest
               if r.hash_file(ROOT/item["path"]) != item["sha256"]]
    assert changes == ["scripts/run_zhou2025_repro.py", "tests/test_zhou2025_repro.py"]
    frame = pd.read_csv(folder/"metrics_raw.csv")
    operators, _ = r.build_operators(config, target)
    checks = []
    with torch.no_grad():
        # Match the archived observation batch to preserve GPU transform/matmul rounding.
        images = target.images("validation", range(100), config["device"])
        truth = target.physical("validation", [0])[0]
        for index, (label, operator, _) in enumerate(operators):
            signal = operator.forward(images)
            noise = r.standard_noise(signal, settings, index, "validation")
            variance = target.shape[-1]/100.
            seed = int(np.random.SeedSequence([settings["posterior_seed"], index, 0, 0]).generate_state(1)[0])
            for name in (job["train_run_id"][0], "zhou_mimo_formal_20261009_suredm1_paper_v1"):
                model, schedule, _, source = models[name]
                ratio = target.scale/source.scale
                image, _ = r.recover(model, schedule, operator,
                    (signal[:1]+math.sqrt(variance)*noise[:1])*ratio,
                    variance*ratio**2, scale=tuning[f"{label}:20:{name}"]["selected"],
                    seed=seed, refinement=1)
                predicted = (decode(image, "mimo")*source.scale)[0].cpu().numpy()
                actual = 10*np.log10(r.nmse(truth, predicted))
                saved = frame[(frame.schedule==label)&(frame.snr_db==20)&(frame.estimator==name)
                              &(frame.scene_id==f"{target.metadata['split_seeds']['validation']}:0")].iloc[0]
                checks.append({"schedule":label, "model":name, "absolute_difference_db":abs(actual-saved.nmse_db)})
    assert len(checks) == 4 and max(c["absolute_difference_db"] for c in checks) < 1e-9, checks
    assert original == {name:r.hash_file(folder/name) for name in original}
    result = {"checked_at_utc":datetime.now(timezone.utc).isoformat(), "status":"ready_not_run",
        "test_run_id":test_id, "validation_run_id":folder.name, "expected_test_rows":21600,
        "test_truth_evaluated":False, "models":list(models), "dependencies":dependencies,
        "counts":target.counts, "split_seeds":target.metadata["split_seeds"],
        "data_files_verified":300, "frozen_tuning_entries":len(tuning),
        "evaluation_run":settings, "diffusion":config["diffusion"], "original_validation_sha256":original,
        "current_source_changes":changes,
        "source_change_reason":"Reject changed validation tuning across independent-test resume; estimator equations unchanged.",
        "current_runner_sha256":r.hash_file(ROOT/"scripts/run_zhou2025_repro.py"),
        "validation_rechecks":checks}
    r.atomic_json(folder/"analysis/independent_test_preflight.json", result)
    for data in datasets.values():
        data.close()
    print(json.dumps({"status":result["status"], "models":len(models), "data_files":300,
        "tuning_entries":len(tuning), "validation_rechecks":len(checks),
        "max_difference_db":max(c["absolute_difference_db"] for c in checks),
        "test_run_created":False}))


if __name__ == "__main__":
    main()
