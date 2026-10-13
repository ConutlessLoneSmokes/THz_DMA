"""Supplementary digital LS and noiseless covariance-span checks, fixed data."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
import run_zhou2025_repro as r
import audit_zhou2025_migration as audit
from zhou2025_repro.data import decode
from thz_dma.metrics.diagnostics import channel_domain_metrics

torch.set_num_threads(4)
r.hash_file=audit.cached_hash
folder=ROOT/"runs/zhou_wideband_formal_20261011_digital_test_v1"
config=r.read_json(folder/"config_resolved.json")["config"]
data=audit.get_data(config["job"]["data_run_id"])
operators,_=r.build_operators(config,data)
rows=[]
with torch.no_grad():
    images=data.images("test",range(100),config["device"])
    truths=data.physical("test",range(100))
    for op_index,(label,operator,acquisition) in enumerate(operators):
        clean=operator.forward(images)
        noise=r.standard_noise(clean,config["evaluation_run"],op_index,"test")
        noiseless=decode(operator.adjoint(clean)/operator.repetitions,"dma")
        expected=torch.zeros_like(noiseless)
        expected[:,operator.indices]=torch.as_tensor(truths[:,acquisition.pilot_indices],device=noiseless.device)/data.scale
        torch.testing.assert_close(noiseless,expected,rtol=3e-6,atol=3e-6)
        for snr in config["evaluation_run"]["snr_db"]:
            variance=r.observation_variance(data,config["evaluation_run"],0.,snr)
            estimated=(decode(operator.adjoint(clean+variance**.5*noise)/operator.repetitions,"dma")*data.scale).cpu().numpy()
            for index in range(100):
                error=r.nmse(truths[index],estimated[index])
                rows.append({"scene_id":data.record("test",index)["scene_id"],"schedule":label,
                    "snr_db":snr,"estimator":"supplemental_digital_minimum_norm_ls","nmse_linear":error,
                    "nmse_db":10*np.log10(error),**channel_domain_metrics(truths[index],estimated[index],
                    acquisition.pilot_indices,np.arange(0,128,8))})
        print("checked direct LS",label,flush=True)
    # Full-tone digital A is unitary: the noise-free LMMSE limit tests the training span.
    operator=operators[0][1]
    fitted,count,_=r.fit_lmmse(data,operator,data.scale,config)
    assert count==256
    prediction=r.lmmse_estimate(fitted,data,count,operator,operator.forward(images),1e-8,data.scale,config).cpu().numpy()
    projection=[{"scene_id":data.record("test",i)["scene_id"],
                 "noise_free_256_basis_nmse":r.nmse(truths[i],prediction[i])} for i in range(100)]
    frame=pd.DataFrame(rows)
    assert len(frame)==1800 and not frame.duplicated(["scene_id","schedule","snr_db"]).any()
    frame.to_csv(audit.OUT/"digital_ls_per_scene.csv",index=False)
    curves=frame.groupby(["schedule","snr_db"]).nmse_linear.mean().reset_index()
    curves["nmse_db"]=10*np.log10(curves.nmse_linear)
    curves.to_csv(audit.OUT/"digital_ls_curves.csv",index=False)
    pd.DataFrame(projection).to_csv(audit.OUT/"covariance_span_projection.csv",index=False)
    result={"supplemental_ls_rows":1800,"noiseless_ls_identity_check_passed":True,
        "covariance_draws":256,"test_scenes":100,"noiseless_projection_variance":1e-8,
        "noiseless_projection_nmse_db":10*np.log10(np.mean([row["noise_free_256_basis_nmse"] for row in projection])),
        "test_tuning_performed":False,"original_metrics_overwritten":False}
    (audit.OUT/"digital_control_integrity.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result),flush=True)
    print(curves.to_string(index=False),flush=True)
