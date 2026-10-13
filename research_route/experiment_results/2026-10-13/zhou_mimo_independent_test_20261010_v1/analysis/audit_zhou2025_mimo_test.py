"""Audit D-038 independent test, reusing D-037 checks and fixed-penalty controls."""
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
from audit_zhou2025_mimo import PREFIX, KEYS, TITLES, read_evaluation
from zhou2025_repro.data import decode
from zhou2025_repro.baselines import lasso


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    folder = r.PROJECT_ROOT/"runs/zhou_mimo_independent_test_20261010_v1"
    validation = r.PROJECT_ROOT/"runs"/f"{PREFIX}_val_v1"
    out = folder/"analysis"
    out.mkdir(exist_ok=True)
    original = {str(path.relative_to(r.PROJECT_ROOT)):r.hash_file(path) for path in
        (folder/"metrics_raw.csv", folder/"tuning.json", validation/"metrics_raw.csv", validation/"tuning.json")}
    resolved = r.read_json(folder/"config_resolved.json")
    config = resolved["config"]
    prior = r.read_json(validation/"config_resolved.json")["config"]
    assert r.read_json(folder/"status.json")["status"] == "complete"
    assert config["job"]["split"] == "test" and config["job"]["tuning_run_id"] == validation.name
    assert {k:v for k,v in config.items() if k!="job"} == {k:v for k,v in prior.items() if k!="job"}
    for key in ("data_run_id", "train_run_id"):
        assert config["job"][key] == prior["job"][key]
    assert r.hash_file(Path(resolved["config_path"])) == resolved["config_sha256"]
    assert r.read_json(folder/"dependencies.json") == r.read_json(validation/"dependencies.json")
    assert r.read_json(folder/"tuning_source.json") == {
        "run_id":validation.name, "sha256":r.hash_file(validation/"tuning.json")}
    assert r.read_json(folder/"observation_reference.json") == r.read_json(validation/"observation_reference.json")
    frame, tuning = read_evaluation(folder, config, 21600)
    previous, previous_tuning = read_evaluation(validation, prior, 21600)
    assert tuning == previous_tuning and not set(frame.scene_id)&set(previous.scene_id)
    summary = r.read_json(folder/"metrics_summary.json")
    assert summary["rows"] == 21600 and summary["split"] == "test" and not summary["smoke_only"]
    checks = {"evaluation_rows":len(frame), "test_scenes":100, "frozen_tuning_entries":len(tuning),
        "observations_match_across_methods":True, "tuning_matches_validation":True,
        "original_sha256":original, "source_members":0, "dataset_files":0}
    manifest = r.read_json(folder/"code_manifest.json")
    with zipfile.ZipFile(folder/"source_snapshot.zip") as snapshot:
        for entry in manifest:
            assert hashlib.sha256(snapshot.read(entry["path"].replace("\\", "/"))).hexdigest() == entry["sha256"]
            assert r.hash_file(r.PROJECT_ROOT/entry["path"]) == entry["sha256"], entry["path"]
            checks["source_members"] += 1
    target = r.load_data(config["job"]["data_run_id"])
    assert target.counts == {"train":4096, "validation":256, "test":100}
    assert target.shape == (16,64)
    sets = {split:{target.record(split, i)["scene_id"] for i in range(count)}
            for split,count in target.counts.items()}
    assert len(set.union(*sets.values())) == 4452
    assert set(frame.scene_id) == sets["test"]
    models = {name:r.load_model(name, config["device"]) for name in config["job"]["train_run_id"]}
    datasets = {loaded[3].directory.name:loaded[3] for loaded in models.values()}
    datasets[target.directory.name] = target
    covariance = {f"{PREFIX}_{tag}_v1":datasets[f"{PREFIX}_{tag}_v1"] for tag in ("data", "noisy01", "noisy1")}
    assert r.dependency_record(target, models, covariance) == r.read_json(folder/"dependencies.json")
    assert set(frame.estimator) == set(models)|{f"lmmse_{name}" for name in covariance}|{"ls", "lasso"}
    for source in datasets.values():
        r.verify_physics(config, source)
        checks["dataset_files"] += len(source.metadata["file_sha256"])
    assert checks["dataset_files"] == 300 and len(datasets) == 7
    checks["ddpm_models"] = 7
    checks["sure_data_dependencies"] = sum(d.metadata["kind"]=="denoised" for d in datasets.values())
    operators, _ = r.build_operators(config,target)
    observed = {}
    with torch.no_grad():
        images = target.images("test",range(100),config["device"])
        truth = target.physical("test",range(100))
        for index,(label,operator,_) in enumerate(operators):
            signal = operator.forward(images)
            observed[label] = index,operator,signal,r.standard_noise(signal,config["evaluation_run"],index,"test")
    repeats = []
    for label,(index,operator,signal,noise) in observed.items():
        for scene in (0,99):
            for snr in (0.,20.,30.):
                variance = target.shape[-1]/10**(snr/10)
                for name in (f"{PREFIX}_clean_v1",f"{PREFIX}_suredm1_corrected_v1"):
                    row = frame[(frame.schedule==label)&(frame.snr_db==snr)&(frame.estimator==name)
                                &(frame.scene_id==target.record("test",scene)["scene_id"])].iloc[0]
                    assert abs(row.complex_noise_variance_phys/(variance*target.scale**2)-1) < 1e-12
                    actual_snr = 10*np.log10(float(signal[scene:scene+1].abs().square().mean())/variance)
                    assert abs(actual_snr-row.actual_whitened_snr_db) < 1e-9
                    model,schedule,_,source = models[name]
                    ratio = target.scale/source.scale
                    seed = int(np.random.SeedSequence([config["evaluation_run"]["posterior_seed"],index,scene,1]).generate_state(1)[0])
                    image,_ = r.recover(model,schedule,operator,
                        (signal[scene:scene+1]+math.sqrt(variance)*noise[scene:scene+1])*ratio,
                        variance*ratio**2,scale=row.guidance,seed=seed,refinement=1)
                    prediction = (decode(image,"mimo")*source.scale)[0].cpu().numpy()
                    repeats.append(abs(10*np.log10(r.nmse(truth[scene],prediction))-row.nmse_db))
    assert len(repeats) == 24 and max(repeats) < 1e-9
    checks["frozen_recovery_rechecks"] = len(repeats)
    checks["max_recovery_difference_db"] = max(repeats)
    pairs = [(f"{PREFIX}_clean_v1",f"lmmse_{PREFIX}_data_v1"),(f"{PREFIX}_clean_v1","lasso")]
    for tag in ("01","1"):
        pairs.extend((f"{PREFIX}_suredm{tag}_{variant}_v1",f"{PREFIX}_naive{tag}_v1") for variant in ("paper","corrected"))
        pairs.append((f"{PREFIX}_suredm{tag}_corrected_v1",f"{PREFIX}_suredm{tag}_paper_v1"))
    rng = np.random.default_rng(2026101001)
    paired,values = [],[]
    for (label,snr),block in frame.groupby(["schedule","snr_db"]):
        pivot = block.pivot(index="scene_id",columns="estimator",values="nmse_db")
        for method,reference in pairs:
            delta = (pivot[method]-pivot[reference]).to_numpy()
            assert len(delta) == 100 and np.isfinite(delta).all()
            interval = r.bootstrap_interval(delta,np.mean,rng=rng,resamples=10000)
            paired.append({"schedule":label,"snr_db":snr,"method":method,"reference":reference,
                "mean_delta_db":float(delta.mean()),"ci95_low":interval[0],"ci95_high":interval[1],
                "improved_cases":int(sum(delta<0)),"cases":100})
            values.extend({"scene_id":scene,"schedule":label,"snr_db":snr,"method":method,
                "reference":reference,"delta_db":float(value)} for scene,value in zip(pivot.index,delta))
    assert len(paired) == 144
    pd.DataFrame(paired).to_csv(out/"paired_comparisons.csv",index=False)
    pd.DataFrame(values).to_csv(out/"paired_per_scene.csv",index=False)
    curves = frame.groupby(KEYS[1:]).nmse_linear.mean().reset_index()
    curves["nmse_db"] = 10*np.log10(curves.nmse_linear)
    curves.to_csv(out/"curves.csv",index=False)
    old_curves = previous.groupby(KEYS[1:]).nmse_linear.mean().reset_index()
    old_curves["nmse_db"] = 10*np.log10(old_curves.nmse_linear)
    comparison = curves.merge(old_curves,on=KEYS[1:],suffixes=("_test","_validation"),validate="one_to_one")
    comparison["test_minus_validation_db"] = comparison.nmse_db_test-comparison.nmse_db_validation
    comparison.to_csv(out/"validation_test_curves.csv",index=False)
    steps = []
    for (scene,label,method),block in frame[frame.estimator.isin(models)].groupby(["scene_id","schedule","estimator"]):
        block = block.sort_values("snr_db")
        for first,second in zip(block.iloc[:-1].itertuples(),block.iloc[1:].itertuples()):
            steps.append({"scene_id":scene,"schedule":label,"estimator":method,"from_snr":first.snr_db,
                "to_snr":second.snr_db,"change_db":second.nmse_db-first.nmse_db})
    steps = pd.DataFrame(steps)
    assert len(steps) == 11200
    steps.to_csv(out/"per_scene_snr_steps.csv",index=False)
    steps[steps.from_snr>=15].groupby("estimator").change_db.agg(
        steps="size",increases=lambda x:int(sum(x>0)),max_change_db="max").to_csv(out/"high_snr_tails.csv")
    bad = frame[frame.estimator.eq("lasso")]
    assert bad.lasso_converged.notna().all()
    bad = bad[~bad.lasso_converged.astype(bool)]
    checks["original_lasso_nonconverged"] = len(bad)
    extended = []
    for number,row in enumerate(bad.sort_values(KEYS).itertuples(),1):
        _,operator,signal,noise = observed[row.schedule]
        scene = int(row.scene_id.split(":")[-1])
        variance = target.shape[-1]/10**(row.snr_db/10)
        y = signal[scene:scene+1]+math.sqrt(variance)*noise[scene:scene+1]
        previous_angle,_,_ = lasso(y,operator.b,row.lasso_penalty,steps=1000)
        old_channel = (decode(torch.stack((previous_angle.real,previous_angle.imag),dim=1),"mimo")*target.scale)[0].cpu().numpy()
        assert abs(10*np.log10(r.nmse(truth[scene],old_channel))-row.nmse_db) < 1e-9
        angle,iterations,converged = lasso(y,operator.b,row.lasso_penalty,steps=10000)
        prediction = (decode(torch.stack((angle.real,angle.imag),dim=1),"mimo")*target.scale)[0].cpu().numpy()
        error = r.nmse(truth[scene],prediction)
        old_objective = float(.5*(previous_angle@operator.b-y).abs().square().sum()+row.lasso_penalty*previous_angle.abs().sum())
        objective = float(.5*(angle@operator.b-y).abs().square().sum()+row.lasso_penalty*angle.abs().sum())
        assert objective <= old_objective*(1+1e-5)
        extended.append({"scene_id":row.scene_id,"schedule":row.schedule,"snr_db":row.snr_db,
            "old_nmse_db":row.nmse_db,"new_nmse_db":float(10*np.log10(error)),"new_nmse_linear":error,
            "penalty":row.lasso_penalty,"iterations":iterations,"converged":bool(converged),
            "old_objective":old_objective,"new_objective":objective})
        if number%20 == 0 or number==len(bad):
            pd.DataFrame(extended).to_csv(out/"lasso_extended.csv",index=False)
            print(f"LASSO sensitivity {number}/{len(bad)}",flush=True)
    checks["extended_lasso_converged"] = sum(row["converged"] for row in extended)
    sensitivity = frame[frame.estimator.eq("lasso")].merge(pd.DataFrame(extended)[
        ["scene_id","schedule","snr_db","new_nmse_linear"]],on=KEYS[:3],how="left",validate="one_to_one")
    sensitivity["extended_nmse_linear"] = sensitivity.new_nmse_linear.fillna(sensitivity.nmse_linear)
    means = sensitivity.groupby(["schedule","snr_db"])[["nmse_linear","extended_nmse_linear"]].mean()
    means["old_nmse_db"] = 10*np.log10(means.nmse_linear)
    means["extended_nmse_db"] = 10*np.log10(means.extended_nmse_linear)
    means["delta_db"] = means.extended_nmse_db-means.old_nmse_db
    means.to_csv(out/"lasso_curve_sensitivity.csv")
    for name,digest in original.items():
        assert r.hash_file(r.PROJECT_ROOT/name) == digest
    r.atomic_json(out/"integrity.json",checks)
    shutil.copyfile(__file__,out/Path(__file__).name)
    shutil.copyfile(r.PROJECT_ROOT/"scripts/audit_zhou2025_mimo.py",out/"audit_zhou2025_mimo.py")
    print(json.dumps(checks,indent=2),flush=True)
    print(pd.DataFrame(paired).groupby(["method","reference"]).ci95_high.apply(lambda x:int(sum(x<0))).to_string(),flush=True)
    print(means[means.delta_db.ne(0)].to_string(),flush=True)
    for data in datasets.values():
        data.close()


if __name__ == "__main__":
    main()
