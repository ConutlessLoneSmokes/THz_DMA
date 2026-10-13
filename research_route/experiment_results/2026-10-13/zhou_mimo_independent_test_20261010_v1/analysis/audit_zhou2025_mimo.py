"""Audit D-037 MIMO training, paired validation, and frozen-penalty LASSO sensitivity."""
import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_zhou2025_repro as r
from zhou2025_repro.data import decode
from zhou2025_repro.ddpm import recover
from zhou2025_repro.baselines import lasso

PREFIX = "zhou_mimo_formal_20261009"
ROOT = r.PROJECT_ROOT
KEYS = ["scene_id", "schedule", "snr_db", "estimator"]
TITLES = {
    f"{PREFIX}_clean_v1": "Clean DDPM",
    f"{PREFIX}_naive01_v1": "Naive DM (0.1)",
    f"{PREFIX}_naive1_v1": "Naive DM (1)",
    **{f"{PREFIX}_suredm{tag}_{variant}_v1": f"SURE-DM {variant} ({noise})"
       for tag, noise in (("01", "0.1"), ("1", "1")) for variant in ("paper", "corrected")},
    f"lmmse_{PREFIX}_data_v1": "Clean-data LMMSE",
    f"lmmse_{PREFIX}_noisy01_v1": "Noisy-data LMMSE (0.1)",
    f"lmmse_{PREFIX}_noisy1_v1": "Noisy-data LMMSE (1)",
    "ls": "LS", "lasso": "LASSO",
}


def read_evaluation(folder, config, expected):
    frame = pd.read_csv(folder / "metrics_raw.csv")
    assert len(frame) == expected and frame.scene_id.nunique() == 100
    assert not frame.duplicated(KEYS).any()
    assert set(frame.split) == {config["job"]["split"]} and not frame.smoke_only.any()
    assert np.isfinite(frame.nmse_linear).all() and frame.nmse_linear.gt(0).all()
    np.testing.assert_allclose(frame.nmse_db, 10*np.log10(frame.nmse_linear), atol=1e-10, rtol=0)
    settings = config["evaluation_run"]
    methods = set(frame.estimator)
    groups = {(f"QPSK_Np{n}", snr, method) for n in settings["pilot_counts"]
              for snr in settings["snr_db"] for method in methods}
    for _, block in frame.groupby("scene_id"):
        assert set(block[KEYS[1:]].itertuples(index=False, name=None)) == groups
    for field in ("complex_noise_variance_phys", "actual_whitened_snr_db"):
        assert frame.groupby(KEYS[:3])[field].nunique().eq(1).all()
    saved = pd.DataFrame([row for path in sorted((folder / "checkpoint").glob("eval_*.json"))
                          for row in r.read_json(path)])
    saved = saved.where(saved.notna(), np.nan)
    pd.testing.assert_frame_equal(frame.sort_values(KEYS).reset_index(drop=True),
        saved[frame.columns].sort_values(KEYS).reset_index(drop=True),
        check_dtype=False, atol=1e-12, rtol=1e-12)
    means = frame.groupby(KEYS[1:]).nmse_linear.mean()
    summary = pd.read_csv(folder / "comparison.csv")
    assert len(summary) == len(means)
    for row in summary.itertuples():
        assert abs(row.nmse_db-10*np.log10(means.loc[(row.schedule, row.snr_db, row.estimator)])) < 1e-10
    tuning = r.read_json(folder / "tuning.json")
    assert len(tuning) == len(config["job"]["train_run_id"]+["lasso"])*18
    for key, entry in tuning.items():
        assert entry["selected"] == entry["candidates"][int(np.argmin(entry["validation_nmse"]))]
        label, snr, method = key.split(":")
        block = frame[(frame.schedule == label) & (frame.snr_db == float(snr)) & (frame.estimator == method)]
        field = "lasso_penalty" if method == "lasso" else "guidance"
        np.testing.assert_allclose(block[field], entry["selected"], atol=1e-12, rtol=0)
        if method != "lasso":
            assert block.network_calls.eq(100).all() and block.model_run_id.eq(method).all()
    return frame, tuning


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    joint = ROOT / "runs" / f"{PREFIX}_val_v1"
    first = ROOT / "runs" / f"{PREFIX}_clean_val_v1"
    out = joint / "analysis"
    out.mkdir(exist_ok=True)
    checks = {"stage_runs": 0, "source_members": 0, "current_source_members": 0,
              "dataset_files": 0, "training_models": 0, "evaluation_hashes": {}}
    inventories, profiles, history_rows = [], [], []
    stages = sorted((ROOT / "runs").glob(f"{PREFIX}_*"))
    assert len(stages) == 20
    datasets = {}
    for folder in stages:
        assert r.read_json(folder / "status.json")["status"] == "complete"
        resolved = r.read_json(folder / "config_resolved.json")
        config = resolved["config"]
        assert not config["job"]["smoke"] and config["scenario"] == "mimo"
        assert r.hash_file(Path(resolved["config_path"])) == resolved["config_sha256"]
        with zipfile.ZipFile(folder / "source_snapshot.zip") as bundle:
            for entry in r.read_json(folder / "code_manifest.json"):
                assert hashlib.sha256(bundle.read(entry["path"].replace("\\", "/"))).hexdigest() == entry["sha256"]
                assert r.hash_file(ROOT / entry["path"]) == entry["sha256"], entry["path"]
                checks["source_members"] += 1
                checks["current_source_members"] += 1
        checks["stage_runs"] += 1
        item = {"run_id": folder.name, "command": config["job"]["command"], "status": "complete"}
        if (folder / "dataset.json").exists():
            data = r.load_data(folder.name)
            r.verify_physics(config, data)
            assert data.counts == {"train":4096, "validation":256, "test":100}
            assert data.shape == (16,64) and len(set(data.metadata["split_seeds"].values())) == 3
            datasets[folder.name] = data
            checks["dataset_files"] += len(data.metadata["file_sha256"])
        if (folder / "training_summary.json").exists():
            summary = r.read_json(folder / "training_summary.json")
            checkpoint = torch.load(folder / "checkpoint/training.pt", map_location="cpu", weights_only=True)
            source = r.load_data(checkpoint["data_run_id"])
            assert checkpoint["dataset_sha256"] == r.hash_file(source.directory / "dataset.json")
            assert summary["dataset_sha256"] == checkpoint["dataset_sha256"]
            epochs = config["sure"]["epochs"] if checkpoint["kind"] == "sure" else config["training"]["epochs"]
            steps = epochs*math.ceil(source.counts["train"]/config["training"]["batch_size"])
            assert summary["trained_steps"] == checkpoint["step"] == steps
            history = r.read_json(folder / "training_history.json")
            assert history == checkpoint["history"] and history[-1]["step"] == steps
            best = min(history, key=lambda x:x["validation_loss"])
            assert best["step"] == summary["best_step"] == checkpoint["best_step"]
            assert best["validation_loss"] == summary["best_validation_loss"] == checkpoint["best_loss"]
            assert all(torch.isfinite(x).all() for x in checkpoint["best_model"].values())
            assert summary["validation_uses_clean_labels"] == (checkpoint["data_kind"] == "clean")
            assert summary["precision"] == ("float64" if checkpoint["kind"] == "sure" else "float32")
            assert config["training"]["seed"] == 2026100911
            checks["training_models"] += 1
            item.update({k:summary[k] for k in ("epochs", "trained_steps", "best_step", "best_validation_loss", "precision")})
            history_rows.extend({"model":folder.name, **row} for row in history)
            if checkpoint["kind"] == "dm":
                profile = r.read_json(folder / "denoising_profile.json")
                assert [x["t"] for x in profile] == [1,25,50,75,100]
                assert all(x["samples"] == 64 for x in profile)
                profiles.extend({"model":folder.name, **row} for row in profile)
            source.close()
        inventories.append(item)
    assert checks["training_models"] == 11 and len(datasets) == 7
    config = r.read_json(joint / "config_resolved.json")["config"]
    first_config = r.read_json(first / "config_resolved.json")["config"]
    assert config["evaluation_run"] == first_config["evaluation_run"]
    frame, tuning = read_evaluation(joint, config, 21600)
    original, original_tuning = read_evaluation(first, first_config, 7200)
    common = frame[frame.estimator.isin(original.estimator.unique())]
    timing_fields = ["online_seconds", "covariance_preparation_seconds"]
    pd.testing.assert_frame_equal(common.drop(columns=timing_fields).sort_values(KEYS).reset_index(drop=True),
        original.drop(columns=timing_fields).sort_values(KEYS).reset_index(drop=True), check_dtype=False, rtol=0, atol=0)
    assert all(tuning[key] == value for key, value in original_tuning.items())
    checks["identical_common_rows"] = len(common)
    checks["evaluation_rows"] = {first.name:len(original), joint.name:len(frame)}
    checks["evaluation_hashes"] = {folder.name:r.hash_file(folder / "metrics_raw.csv") for folder in (first,joint)}
    target = datasets[config["job"]["data_run_id"]]
    models = {name:r.load_model(name, config["device"]) for name in config["job"]["train_run_id"]}
    covariance = {f"{PREFIX}_{tag}_v1":datasets[f"{PREFIX}_{tag}_v1"] for tag in ("data","noisy01","noisy1")}
    assert r.dependency_record(target, models, covariance) == r.read_json(joint / "dependencies.json")
    for data in datasets.values():
        array = data.physical("train", range(4096)).astype(np.complex128)
        variance = data.metadata["known_noise_real_variance_phys"]
        if data.metadata["kind"] != "denoised":
            scale = r.signal_scale(float(np.sum(abs(array)**2)), array.size, variance)
            assert abs(scale/data.scale-1) < 1e-12
    clean = target.physical("validation", range(256))
    noise01 = datasets[f"{PREFIX}_noisy01_v1"].physical("validation", range(256))-clean
    noise1 = datasets[f"{PREFIX}_noisy1_v1"].physical("validation", range(256))-clean
    noise_gap = float(np.linalg.norm(noise1-np.sqrt(10)*noise01)/np.linalg.norm(noise1))
    assert noise_gap < 2e-6
    checks["paired_training_noise_relative_error"] = noise_gap
    denoising = []
    for tag in ("01","1"):
        for variant in ("paper","corrected"):
            source = datasets[f"{PREFIX}_noisy{tag}_v1"]
            denoised = datasets[f"{PREFIX}_denoised{tag}_{variant}_v1"]
            for label, values in (("noisy",source.physical("validation",range(256))),
                                  (variant,denoised.physical("validation",range(256)))):
                errors = [r.nmse(h,e) for h,e in zip(clean,values)]
                denoising.append({"noise":source.metadata["requested_canonical_real_variance"],
                    "variant":variant, "estimator":label, "scenes":256,
                    "nmse_db":float(10*np.log10(np.mean(errors)))})
    pd.DataFrame(inventories).to_csv(out / "run_inventory.csv", index=False)
    pd.DataFrame(history_rows).to_csv(out / "training_history.csv", index=False)
    pd.DataFrame(profiles).to_csv(out / "denoising_profiles.csv", index=False)
    pd.DataFrame(denoising).to_csv(out / "sure_denoising.csv", index=False)
    pairs = [(f"{PREFIX}_clean_v1",f"lmmse_{PREFIX}_data_v1"), (f"{PREFIX}_clean_v1","lasso")]
    for tag in ("01","1"):
        pairs.extend((f"{PREFIX}_suredm{tag}_{variant}_v1",f"{PREFIX}_naive{tag}_v1") for variant in ("paper","corrected"))
        pairs.append((f"{PREFIX}_suredm{tag}_corrected_v1",f"{PREFIX}_suredm{tag}_paper_v1"))
    rng = np.random.default_rng(2026101001)
    paired, values = [], []
    for (label,snr), block in frame.groupby(["schedule","snr_db"]):
        pivot = block.pivot(index="scene_id",columns="estimator",values="nmse_db")
        for method, reference in pairs:
            delta = (pivot[method]-pivot[reference]).to_numpy()
            assert np.isfinite(delta).all() and len(delta) == 100
            interval = r.bootstrap_interval(delta,np.mean,rng=rng,resamples=10000)
            paired.append({"schedule":label,"snr_db":snr,"method":method,"reference":reference,
                "mean_delta_db":float(delta.mean()),"ci95_low":interval[0],"ci95_high":interval[1],
                "improved_cases":int(sum(delta<0)),"cases":100})
            values.extend({"scene_id":scene,"schedule":label,"snr_db":snr,"method":method,
                "reference":reference,"delta_db":float(value)} for scene,value in zip(pivot.index,delta))
    assert len(paired) == 144
    pd.DataFrame(paired).to_csv(out / "paired_comparisons.csv", index=False)
    pd.DataFrame(values).to_csv(out / "paired_per_scene.csv", index=False)
    bad = frame[(frame.estimator == "lasso") & ~frame.lasso_converged.astype(bool)]
    assert len(bad) == 209
    checks["original_lasso_nonconverged"] = len(bad)
    operators, _ = r.build_operators(config,target)
    observed = {}
    with torch.no_grad():
        images = target.images("validation",range(100),config["device"])
        for index,(label,op,_) in enumerate(operators):
            signal = op.forward(images)
            observed[label] = (index,op,signal,r.standard_noise(signal,config["evaluation_run"],index,"validation"))
    repeats = []
    for label in frame.schedule.unique():
        index,op,signal,noise = observed[label]
        for scene_index in (0,99):
            for snr in (0.,20.,30.):
                for name in (f"{PREFIX}_clean_v1",f"{PREFIX}_suredm1_corrected_v1"):
                    row = frame[(frame.schedule==label)&(frame.snr_db==snr)&(frame.estimator==name)
                                &(frame.scene_id==f"{target.metadata['split_seeds']['validation']}:{scene_index}")].iloc[0]
                    variance = row.complex_noise_variance_phys/target.scale**2
                    model,schedule,_,source = models[name]
                    ratio = target.scale/source.scale
                    seed = int(np.random.SeedSequence([config["evaluation_run"]["posterior_seed"],index,scene_index,0]).generate_state(1)[0])
                    image,_ = recover(model,schedule,op,(signal[scene_index:scene_index+1]+math.sqrt(variance)*noise[scene_index:scene_index+1])*ratio,
                        variance*ratio**2,scale=row.guidance,seed=seed,refinement=1)
                    prediction = (decode(image,"mimo")*source.scale)[0].cpu().numpy()
                    actual = 10*np.log10(r.nmse(clean[scene_index],prediction))
                    repeats.append(abs(actual-row.nmse_db))
    assert len(repeats) == 24 and max(repeats) < 1e-9
    checks["frozen_recovery_rechecks"] = len(repeats)
    checks["max_recovery_difference_db"] = max(repeats)
    extended = []
    for number,row in enumerate(bad.sort_values(KEYS).itertuples(),1):
        index,op,signal,noise = observed[row.schedule]
        scene_index = int(row.scene_id.split(":")[-1])
        variance = row.complex_noise_variance_phys/target.scale**2
        y = signal[scene_index:scene_index+1]+math.sqrt(variance)*noise[scene_index:scene_index+1]
        previous,_,_ = lasso(y,op.b,row.lasso_penalty,steps=1000)
        previous_image = torch.stack((previous.real,previous.imag),dim=1)
        previous_channel = (decode(previous_image,"mimo")*target.scale)[0].cpu().numpy()
        assert abs(10*np.log10(r.nmse(clean[scene_index],previous_channel))-row.nmse_db) < 1e-9
        angle,iterations,converged = lasso(y,op.b,row.lasso_penalty,steps=10000)
        prediction = (decode(torch.stack((angle.real,angle.imag),dim=1),"mimo")*target.scale)[0].cpu().numpy()
        error = r.nmse(clean[scene_index],prediction)
        old_objective = float(.5*(previous@op.b-y).abs().square().sum()+row.lasso_penalty*previous.abs().sum())
        objective = float(.5*(angle@op.b-y).abs().square().sum()+row.lasso_penalty*angle.abs().sum())
        assert objective <= old_objective*(1+1e-5)
        extended.append({"scene_id":row.scene_id,"schedule":row.schedule,"snr_db":row.snr_db,
            "old_nmse_db":row.nmse_db,"new_nmse_db":float(10*np.log10(error)),"new_nmse_linear":error,
            "penalty":row.lasso_penalty,"iterations":iterations,"converged":bool(converged),
            "old_objective":old_objective,"new_objective":objective})
        if number%20 == 0 or number == len(bad):
            pd.DataFrame(extended).to_csv(out / "lasso_extended.csv",index=False)
            print(f"LASSO sensitivity {number}/{len(bad)}",flush=True)
    checks["extended_lasso_converged"] = int(sum(row["converged"] for row in extended))
    for name,digest in checks["evaluation_hashes"].items():
        assert r.hash_file(ROOT/"runs"/name/"metrics_raw.csv") == digest
    r.atomic_json(out / "integrity.json",checks)
    shutil.copyfile(__file__,out / Path(__file__).name)
    print(json.dumps(checks,indent=2),flush=True)
    print(pd.DataFrame(denoising).to_string(index=False),flush=True)
    for data in datasets.values():
        data.close()


if __name__ == "__main__":
    main()
