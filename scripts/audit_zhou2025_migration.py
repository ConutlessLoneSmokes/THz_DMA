"""Read-only D-039 run, provenance, frozen-test and paired-result audit."""
import hashlib
import itertools
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_zhou2025_repro as r

ROOT = r.PROJECT_ROOT
OUT = ROOT/"runs/zhou_wideband_formal_20261011_dma_test_v1/analysis"
KEYS = ["scene_id","schedule","snr_db","estimator"]
FAMILIES = ["zhou_mimo_extended_20261011", *[f"zhou_carrier_{factor}_20261011" for factor in
            ("40ghz_noabs","300ghz_noabs","300ghz_abs")],
            "zhou_wideband_formal_20261011_digital","zhou_wideband_formal_20261011_dma"]
EXPECTED = [25200,10800,10800,10800,3600,10800]
FRAMES, CONFIGS, DATA = {}, {}, {}
hash_cache = {}
original_hash = r.hash_file


def cached_hash(path):
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path),stat.st_size,stat.st_mtime_ns)
    if key not in hash_cache:
        hash_cache[key] = original_hash(path)
    return hash_cache[key]


def get_data(name):
    if name not in DATA:
        DATA[name] = r.load_data(name)
    return DATA[name]


def paired(first,second,label,relation,rng):
    merged = first[KEYS[:3]+["nmse_db"]].merge(second[KEYS[:3]+["nmse_db"]],
        on=KEYS[:3],suffixes=("_first","_second"),validate="one_to_one")
    assert len(merged)==1800
    results = []
    for (schedule,snr),block in merged.groupby(["schedule","snr_db"]):
        delta = (block.nmse_db_first-block.nmse_db_second).to_numpy()
        assert len(delta)==100 and np.isfinite(delta).all()
        ci = r.bootstrap_interval(delta,np.mean,rng=rng,resamples=10000)
        results.append({"family":label,"relation":relation,"schedule":schedule,"snr_db":snr,
            "mean_delta_db":delta.mean(),"ci95_low":ci[0],"ci95_high":ci[1],
            "improved_cases":int(sum(delta<0)),"cases":100})
    return results


def main():
    OUT.mkdir(exist_ok=True)
    r.hash_file = cached_hash
    inventory,curve_rows,training,convergence = [],[],[],[]
    stages = [ROOT/"runs"/(family+"_"+split+"_v1") for family in FAMILIES for split in ("val","test")]
    stages += [ROOT/"runs"/(family+"_"+stage+"_v1") for family in FAMILIES[1:4] for stage in ("data","clean")]
    stages += [ROOT/"runs"/("zhou_wideband_formal_20261011_"+stage+"_v1") for stage in ("data","clean")]
    assert len(stages)==20
    source_members = 0
    preserved = {}
    for folder in stages:
        status = r.read_json(folder/"status.json")
        assert status["status"]=="complete",folder.name
        resolved = r.read_json(folder/"config_resolved.json")
        config = resolved["config"]
        assert not config["job"]["smoke"]
        assert cached_hash(Path(resolved["config_path"]))==resolved["config_sha256"]
        with zipfile.ZipFile(folder/"source_snapshot.zip") as archive:
            for item in r.read_json(folder/"code_manifest.json"):
                assert hashlib.sha256(archive.read(item["path"].replace("\\","/"))).hexdigest()==item["sha256"]
                assert cached_hash(ROOT/item["path"])==item["sha256"],item["path"]
                source_members += 1
        CONFIGS[folder.name] = config
        count = 0
        if config["job"]["command"]=="prepare":
            data = get_data(folder.name)
        elif config["job"]["command"]=="train":
            checkpoint = torch.load(folder/"checkpoint/training.pt",map_location="cpu",weights_only=True)
            summary = r.read_json(folder/"training_summary.json")
            data = get_data(checkpoint["data_run_id"])
            r.verify_physics(config,data)
            assert cached_hash(data.directory/"dataset.json")==checkpoint["dataset_sha256"]
            expected_steps = 16000 if data.scenario=="mimo" else 256000
            assert summary["trained_steps"]==checkpoint["step"]==expected_steps
            assert summary["best_step"]==checkpoint["best_step"]
            assert summary["best_validation_loss"]==checkpoint["best_loss"]
            history = checkpoint["history"]
            assert min(row["validation_loss"] for row in history)==summary["best_validation_loss"]
            training.append({"run_id":folder.name,**summary})
        else:
            family = next(f for f in FAMILIES if folder.name.startswith(f+"_"))
            frame = pd.read_csv(folder/"metrics_raw.csv")
            count = len(frame)
            assert count==EXPECTED[FAMILIES.index(family)]
            assert frame.scene_id.nunique()==100 and not frame.duplicated(KEYS).any()
            assert not frame.smoke_only.any() and set(frame.split)=={config["job"]["split"]}
            assert np.isfinite(frame.nmse_linear).all() and frame.nmse_linear.gt(0).all()
            np.testing.assert_allclose(frame.nmse_db,10*np.log10(frame.nmse_linear),atol=1e-10,rtol=0)
            methods = set(frame.estimator)
            schedules = set(frame.schedule)
            groups = set(itertools.product(schedules,config["evaluation_run"]["snr_db"],methods))
            for _,block in frame.groupby("scene_id"):
                assert set(block[KEYS[1:]].itertuples(index=False,name=None))==groups
            for field in ("complex_noise_variance_phys","actual_whitened_snr_db"):
                assert frame.groupby(KEYS[:3])[field].nunique().eq(1).all()
            saved = [row for path in (folder/"checkpoint").glob("*.json")
                     if path.name.startswith(("eval_","coarse_")) for row in r.read_json(path)]
            saved = pd.DataFrame(saved).where(pd.DataFrame(saved).notna(),np.nan)
            pd.testing.assert_frame_equal(frame.sort_values(KEYS).reset_index(drop=True),
                saved[frame.columns].sort_values(KEYS).reset_index(drop=True),check_dtype=False,atol=1e-12,rtol=1e-12)
            summary = r.read_json(folder/"metrics_summary.json")
            assert summary["rows"]==count and summary["split"]==config["job"]["split"]
            computed = frame.groupby(KEYS[1:]).nmse_linear.mean()
            for row in pd.read_csv(folder/"comparison.csv").itertuples():
                assert abs(row.nmse_db-10*np.log10(computed.loc[(row.schedule,row.snr_db,row.estimator)]))<1e-10
            tuning = r.read_json(folder/"tuning.json")
            for key,entry in tuning.items():
                assert entry["selected"]==entry["candidates"][int(np.argmin(entry["validation_nmse"]))]
                label,snr,method = key.split(":")
                block = frame[(frame.schedule==label)&(frame.snr_db==float(snr))&(frame.estimator==method)]
                field = "lasso_penalty" if method=="lasso" else "guidance"
                np.testing.assert_allclose(block[field],entry["selected"],atol=1e-12,rtol=0)
            nn = frame[frame.estimator.isin(config["job"]["train_run_id"])]
            assert nn.network_calls.eq(100).all()
            data = get_data(config["job"]["data_run_id"])
            r.verify_physics(config,data)
            assert set(frame.scene_id)=={data.record(config["job"]["split"],i)["scene_id"] for i in range(100)}
            dependencies = r.read_json(folder/"dependencies.json")
            assert cached_hash(data.directory/"dataset.json")==dependencies["target"]["dataset_sha256"]
            for model,digest in dependencies["models"].items():
                assert cached_hash(ROOT/"runs"/model/"checkpoint/training.pt")==digest
                cp = torch.load(ROOT/"runs"/model/"checkpoint/training.pt",map_location="cpu",weights_only=True)
                source = get_data(cp["data_run_id"])
                assert cached_hash(source.directory/"dataset.json")==cp["dataset_sha256"]
                r.verify_physics(config,source)
            for name,digest in dependencies["covariance_data"].items():
                assert cached_hash(get_data(name).directory/"dataset.json")==digest
            if "noise_scale_reference" in dependencies:
                source = dependencies["noise_scale_reference"]
                noise_data = get_data(source["run_id"])
                assert noise_data.scale==source["complex_scale"]
                assert cached_hash(noise_data.directory/"dataset.json")==source["dataset_sha256"]
            if config["job"]["split"]=="test":
                val = ROOT/"runs"/config["job"]["tuning_run_id"]
                previous = CONFIGS[val.name]
                assert previous["evaluation_run"]==config["evaluation_run"]
                assert r.read_json(val/"dependencies.json")==dependencies
                assert r.read_json(val/"tuning.json")==tuning
                assert r.read_json(folder/"tuning_source.json")=={"run_id":val.name,"sha256":cached_hash(val/"tuning.json")}
            for method,block in frame.groupby("estimator"):
                if method=="lasso" or method=="conditional_sbl_jointfreq_3d":
                    field = "lasso_converged" if method=="lasso" else "algorithm_converged"
                    convergence.append({"run_id":folder.name,"method":method,"rows":len(block),
                        "not_converged":int(block[field].astype(str).str.lower().eq("false").sum())})
            for (label,snr,method),block in frame.groupby(KEYS[1:]):
                curve_rows.append({"family":family,"split":config["job"]["split"],"schedule":label,
                    "snr_db":snr,"estimator":method,"nmse_db":10*np.log10(block.nmse_linear.mean()),
                    "mean_actual_whitened_snr_db":block.actual_whitened_snr_db.mean(),
                    "pilot_nmse_db":10*np.log10(block.pilot_nmse_linear.mean()) if "pilot_nmse_linear" in block else None,
                    "heldout_nmse_db":10*np.log10(block.heldout_112_nmse_linear.mean()) if
                      "heldout_112_nmse_linear" in block and block.heldout_112_nmse_linear.notna().any() else None,
                    "net_rate":block.net_rate_bps_hz.mean() if "net_rate_bps_hz" in block else None})
            FRAMES[folder.name]=frame
            preserved[folder.name]={name:cached_hash(folder/name) for name in ("metrics_raw.csv","tuning.json")}
        inventory.append({"run_id":folder.name,"command":config["job"]["command"],"rows":count})
        print("checked",folder.name,count,flush=True)
    assert sum(row["rows"] for row in inventory)==144000
    for data in DATA.values():
        ids = [{data.record(split,i)["scene_id"] for i in range(count)} for split,count in data.counts.items()]
        assert len(set.union(*ids))==sum(data.counts.values())
    # Supplemental methods must leave the old 12-method results numerically unchanged.
    for split,prior in (("val","zhou_mimo_formal_20261009_val_v1"),("test","zhou_mimo_independent_test_20261010_v1")):
        old = pd.read_csv(ROOT/"runs"/prior/"metrics_raw.csv")
        new = FRAMES[f"{FAMILIES[0]}_{split}_v1"]
        merged = old.merge(new,on=KEYS,suffixes=("_old","_new"),validate="one_to_one")
        assert len(merged)==21600
        for field in ("nmse_linear","nmse_db","guidance","lasso_penalty","complex_noise_variance_phys"):
            np.testing.assert_allclose(merged[field+"_old"],merged[field+"_new"],atol=1e-10,rtol=1e-12,equal_nan=True)
    # All carrier configurations use the same path records and physical noise.
    records = []
    for family in FAMILIES[1:4]:
        data = get_data(family+"_data_v1")
        records.append([data.record(split,i) for split,count in data.counts.items() for i in range(count)])
    assert records[0]==records[1]==records[2] and len(records[0])==4452
    for split in ("val","test"):
        frames = [FRAMES[family+"_"+split+"_v1"] for family in FAMILIES[1:4]]
        for other in frames[1:]:
            first = frames[0].groupby(KEYS[:3]).complex_noise_variance_phys.first()
            second = other.groupby(KEYS[:3]).complex_noise_variance_phys.first()
            np.testing.assert_allclose(first,second,rtol=1e-12,atol=0)
        digital = FRAMES[FAMILIES[4]+"_"+split+"_v1"].copy()
        digital["schedule"]=digital.schedule.str.replace("digital_","",regex=False)
        dma = FRAMES[FAMILIES[5]+"_"+split+"_v1"]
        np.testing.assert_allclose(digital.groupby(KEYS[:3]).complex_noise_variance_phys.first(),
                                   dma.groupby(KEYS[:3]).complex_noise_variance_phys.first(),rtol=1e-12)
    comparisons = []
    rng = np.random.default_rng(2026101301)
    for family in FAMILIES:
        frame = FRAMES[family+"_test_v1"]
        config = CONFIGS[family+"_test_v1"]
        model = config["job"]["train_run_id"][0]
        references = ["lmmse_"+config["job"]["data_run_id"]]
        references += [name for name in ("lasso","angular_omp","conditional_conditional_lmmse",
                       "conditional_grid_omp_3d","conditional_sbl_jointfreq_3d") if name in set(frame.estimator)]
        for reference in references:
            comparisons += paired(frame[frame.estimator==model],frame[frame.estimator==reference],family,
                                  "Clean DDPM minus "+reference,rng)
    for first,second,relation in ((FAMILIES[3],FAMILIES[2],"300abs minus 300noabs"),
                                   (FAMILIES[2],FAMILIES[1],"300noabs minus 40noabs")):
        for method in ("clean","lasso","angular_omp","oracle_geometry_ls"):
            frames = []
            for family in (first,second):
                frame = FRAMES[family+"_test_v1"]
                selected = CONFIGS[family+"_test_v1"]["job"]["train_run_id"][0] if method=="clean" else method
                frames.append(frame[frame.estimator==selected])
            comparisons += paired(*frames,first,relation+" "+method,rng)
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(OUT/"curves.csv",index=False)
    pd.DataFrame(comparisons).to_csv(OUT/"paired_comparisons.csv",index=False)
    pd.DataFrame(inventory).to_csv(OUT/"run_inventory.csv",index=False)
    pd.DataFrame(training).to_csv(OUT/"training_inventory.csv",index=False)
    pd.DataFrame(convergence).to_csv(OUT/"solver_convergence.csv",index=False)
    result = {"status":"archive_and_frozen_test_checks_passed","stage_runs":20,"evaluation_rows":144000,
        "source_members":source_members,"unique_datasets":len(DATA),
        "dataset_manifest_members":sum(len(data.metadata["file_sha256"]) for data in DATA.values()),
        "frozen_test_runs":6,"extended_common_rows_unchanged":43200,
        "paired_carrier_records":4452,"physical_noise_control_passed":True,"original_files":preserved,
        "paired_comparison_groups":len(comparisons),"bootstrap_resamples":10000,"bootstrap_seed":2026101301}
    (OUT/"integrity.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="original_files"}),flush=True)
    print(curves[(curves.split=="test")&curves.snr_db.isin([20.,30.])].to_string(index=False),flush=True)


if __name__=="__main__":
    main()
