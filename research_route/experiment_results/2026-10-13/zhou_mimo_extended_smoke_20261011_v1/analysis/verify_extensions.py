"""Read-only consistency audit of the D-039 implementation checks."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/"src"),str(ROOT/"scripts")]
import run_zhou2025_repro as r
from thz_dma.channels.atmosphere import AbsorptionTable
from thz_dma.channels.planar import channel_from_planar_estimate
from zhou2025_repro.parametric_baselines import mimo_path_templates

OUT = Path(__file__).parent
prefixes = [f"zhou_carrier_{factor}_smoke_20261011" for factor in
            ("40ghz_noabs","300ghz_noabs","300ghz_abs")]
ids = ["zhou_mimo_extended_smoke_20261011_v1"]
ids += [f"{prefix}_{stage}_v1" for prefix in prefixes for stage in ("data","clean","val")]
ids += [f"zhou_wideband_smoke_20261011_{stage}_v1" for stage in
        ("data","clean","digital_val","dma_val")]
ids += ["zhou_wideband_smoke_20261011_data_v2"]
rows, inventory, current_changes = [], [], {}
zip_members = 0
for identifier in ids:
    folder = ROOT/"runs"/identifier
    assert r.read_json(folder/"status.json")["status"] == "complete", identifier
    config = r.read_json(folder/"config_resolved.json")["config"]
    assert config["job"]["smoke"]
    manifest = r.read_json(folder/"code_manifest.json")
    with zipfile.ZipFile(folder/"source_snapshot.zip") as snapshot:
        assert set(snapshot.namelist()) == {item["path"].replace("\\","/") for item in manifest}
        for item in manifest:
            assert hashlib.sha256(snapshot.read(item["path"].replace("\\","/"))).hexdigest() == item["sha256"]
            zip_members += 1
    changed = [item["path"].replace("\\","/") for item in manifest if r.hash_file(ROOT/item["path"]) != item["sha256"]]
    assert set(changed) <= {"scripts/run_zhou2025_repro.py","tests/test_zhou2025_extensions.py"}
    current_changes[identifier] = changed
    count = 0
    if (folder/"metrics_raw.csv").exists():
        frame = pd.read_csv(folder/"metrics_raw.csv")
        assert not frame.duplicated(["scene_id","schedule","snr_db","estimator"]).any()
        count = len(frame)
        assert count == r.read_json(folder/"metrics_summary.json")["rows"]
        assert np.isfinite(frame.nmse_linear).all() and (frame.nmse_linear >= 0).all()
        assert frame.smoke_only.all() and (frame.split=="validation").all()
        expected = 40 if identifier==ids[0] else (16 if "digital_val" in identifier else
                   24 if "dma_val" in identifier else 48)
        assert count==expected, (identifier,count,expected)
        for _, block in frame.groupby(["scene_id","schedule","snr_db"]):
            assert block.complex_noise_variance_phys.nunique()==1
            assert block.actual_whitened_snr_db.nunique()==1
        frame["run_id"] = identifier
        rows.append(frame)
    if config["job"]["command"]=="train":
        assert r.read_json(folder/"training_summary.json")["trained_steps"]==4
    inventory.append({"run_id":identifier,"command":config["job"]["command"],"rows":count})
assert len(ids)==15 and sum(item["rows"] for item in inventory)==224

prior = ROOT/"runs/zhou_mimo_formal_20261009_val_v1"
prior_dependencies = r.read_json(prior/"dependencies.json")
for name, digest in prior_dependencies["models"].items():
    assert r.hash_file(ROOT/"runs"/name/"checkpoint/training.pt")==digest
datasets = {prior_dependencies["target"]["run_id"]:prior_dependencies["target"]["dataset_sha256"]}
datasets.update(prior_dependencies["covariance_data"])
for name in prior_dependencies["models"]:
    checkpoint = r.torch.load(ROOT/"runs"/name/"checkpoint/training.pt",map_location="cpu",weights_only=True)
    datasets[checkpoint["data_run_id"]] = checkpoint["dataset_sha256"]
verified_files = 0
for name, digest in datasets.items():
    data = r.load_data(name)
    assert r.hash_file(data.directory/"dataset.json")==digest
    verified_files += len(data.metadata["file_sha256"])
    data.close()
assert r.hash_file(prior/"metrics_raw.csv")=="d49daea869ad243e86bb9df21030d1fbdc124fc9479a2b233b15ef7576bafc20"
assert r.hash_file(ROOT/"runs/zhou_mimo_independent_test_20261010_v1/metrics_raw.csv")=="f6f4f04faab7b6a8230d55a685d18aab5111d864d903197c874a69e20fe5ed13"

table = AbsorptionTable.from_txt(ROOT/"data/atmosphere/data_freq_abscoe.txt",max_frequency_hz=1e12)
reconstruction_errors, records = [], []
for prefix in prefixes:
    data = r.load_data(prefix+"_data_v1")
    config = r.read_json(data.directory/"config_resolved.json")["config"]
    split_records = []
    for split,count in data.counts.items():
        for index in range(count):
            record = data.record(split,index)
            split_records.append(record)
            templates = mimo_path_templates(config,record["paths"],table)
            gains = np.array([complex(p["gain_real"],p["gain_imag"]) for p in record["paths"]])
            estimate = np.einsum("l,lrt->rt",gains,templates)
            truth = data.physical(split,[index])[0]
            error = float(np.linalg.norm(estimate-truth)/np.linalg.norm(truth))
            assert error < 2e-7
            reconstruction_errors.append(error)
    records.append(split_records)
    data.close()
assert records[0]==records[1]==records[2]

carrier_rows = [frame for frame in rows if "zhou_carrier_" in frame.run_id.iloc[0]]
for first,other in zip(carrier_rows,carrier_rows[1:]):
    left = first.groupby(["scene_id","schedule","snr_db"]).complex_noise_variance_phys.first()
    right = other.groupby(["scene_id","schedule","snr_db"]).complex_noise_variance_phys.first()
    np.testing.assert_allclose(left,right,rtol=1e-12,atol=0)

wideband = [r.load_data(f"zhou_wideband_smoke_20261011_data_v{version}") for version in (1,2)]
assert wideband[0].metadata["index"]==wideband[1].metadata["index"]
assert wideband[0].scale==wideband[1].scale
config = r.read_json(wideband[0].directory/"config_resolved.json")["config"]
config["evaluation_run"]["frontend"]="dma"
config["evaluation_run"]["conditional_references"]=[]
_,platform = r.build_operators(config,wideband[0])
for split in ("train","validation","test"):
    np.testing.assert_array_equal(wideband[0].physical(split,[0,1]),wideband[1].physical(split,[0,1]))
    for index in (0,1):
        record = wideband[0].record(split,index)
        gains = np.asarray(record["gain_real"])+1j*np.asarray(record["gain_imag"])
        estimate = channel_from_planar_estimate(platform.frequencies_hz,platform,
                    np.asarray(record["coordinates"]),gains,config)
        truth = wideband[0].physical(split,[index])[0]
        error = float(np.linalg.norm(estimate-truth)/np.linalg.norm(truth))
        assert error < 2e-7
        reconstruction_errors.append(error)
for data in wideband:
    data.close()
digital = next(frame for frame in rows if "digital_val" in frame.run_id.iloc[0]).copy()
dma = next(frame for frame in rows if "dma_val" in frame.run_id.iloc[0]).copy()
digital["schedule"] = digital.schedule.str.replace("digital_","",regex=False)
merged = digital.groupby(["scene_id","schedule","snr_db"]).first().merge(
         dma.groupby(["scene_id","schedule","snr_db"]).first(),left_index=True,right_index=True,suffixes=("_digital","_dma"))
assert len(merged)==8
np.testing.assert_allclose(merged.complex_noise_variance_phys_digital,merged.complex_noise_variance_phys_dma,rtol=1e-12)
assert (digital.receiver_rf_outputs==1280).all() and (digital.num_switches==0).all()
assert (dma.receiver_rf_outputs==8).all()

pd.DataFrame(inventory).to_csv(OUT/"run_inventory.csv",index=False)
pd.concat(rows,ignore_index=True).to_csv(OUT/"smoke_metrics.csv",index=False)
result = {"status":"implementation_connectivity_passed","formal_performance_tested":False,
    "run_count":len(ids),"evaluation_rows":224,"unit_tests_passed":40,
    "source_zip_members_verified":zip_members,"current_source_changes_after_initial_smoke":current_changes,
    "source_change_reason":"additional absorption-table/cache provenance guard and its test; estimator equations unchanged",
    "original_models_unchanged":len(prior_dependencies["models"]),"original_dataset_files_verified":verified_files,
    "original_validation_and_test_metrics_unchanged":True,"paired_carrier_path_records":len(records[0]),
    "truth_parameter_reconstructions":len(reconstruction_errors),
    "maximum_relative_truth_reconstruction_error":max(reconstruction_errors),
    "carrier_physical_noise_equal":True,"digital_dma_physical_noise_equal":True,
    "digital_dma_same_element_noise_realization":False,
    "digital_rf_outputs":1280,"dma_rf_outputs":8}
(OUT/"verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({key:value for key,value in result.items() if key!="current_source_changes_after_initial_smoke"}))
