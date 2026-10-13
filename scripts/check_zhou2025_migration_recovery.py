"""Fixed-weight replay and projection diagnostics; no training or test tuning."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_zhou2025_repro as r
import audit_zhou2025_migration as audit
from zhou2025_repro.data import decode
from zhou2025_repro.parametric_baselines import angular_omp,mimo_path_templates,oracle_true_geometry
from thz_dma.channels.atmosphere import AbsorptionTable


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    r.hash_file=audit.cached_hash
    records,details=[],[]
    for family in audit.FAMILIES:
        folder=r.PROJECT_ROOT/"runs"/(family+"_test_v1")
        config=r.read_json(folder/"config_resolved.json")["config"]
        frame=pd.read_csv(folder/"metrics_raw.csv")
        data=audit.get_data(config["job"]["data_run_id"])
        name=config["job"]["train_run_id"][0]
        model,schedule,checkpoint,source=r.load_model(name,config["device"])
        operators,_=r.build_operators(config,data)
        table=AbsorptionTable.from_txt(r.PROJECT_ROOT/config["atmosphere"]["table_path"],max_frequency_hz=1e12)
        noise_scale=(audit.get_data(config["evaluation_run"]["noise_scale_run_id"]).scale
                     if config["evaluation_run"].get("noise_scale_run_id") else None)
        with torch.no_grad():
            images=data.images("test",range(100),config["device"])
            for op_index,(label,operator,acquisition) in enumerate(operators):
                clean=operator.forward(images)
                noise=r.standard_noise(clean,config["evaluation_run"],op_index,"test")
                methods=[name]+(["angular_omp","oracle_geometry_ls"] if family==audit.FAMILIES[0] else [])
                for index in (0,99):
                    truth=data.physical("test",[index])[0].astype(np.complex128)
                    scene=data.record("test",index)["scene_id"]
                    for snr in (20.,30.):
                        variance=r.observation_variance(data,config["evaluation_run"],0.,snr,noise_scale)
                        y=clean[index:index+1]+math.sqrt(variance)*noise[index:index+1]
                        for method in methods:
                            row=frame[(frame.scene_id==scene)&(frame.schedule==label)&(frame.snr_db==snr)&(frame.estimator==method)].iloc[0]
                            if method==name:
                                seed=int(np.random.SeedSequence([config["evaluation_run"]["posterior_seed"],op_index,index,1]).generate_state(1)[0])
                                ratio=data.scale/source.scale
                                estimate,trace=r.recover(model,schedule,operator,y*ratio,variance*ratio**2,
                                    scale=row.guidance,seed=seed,refinement=config["evaluation_run"]["refinement"])
                                prediction=(decode(estimate,data.scenario)*source.scale)[0].cpu().numpy().astype(np.complex128)
                                canonical=estimate/ratio
                            else:
                                if method=="angular_omp":
                                    angle,_=angular_omp(y,operator.b,config["evaluation_run"]["omp_max_atoms"])
                                else:
                                    templates=mimo_path_templates(config,data.record("test",index)["paths"],table)/data.scale
                                    angle,_=oracle_true_geometry(y,operator.b,templates)
                                canonical=torch.stack((angle.real,angle.imag),dim=1)
                                prediction=(decode(canonical,"mimo")*data.scale)[0].cpu().numpy().astype(np.complex128)
                            error=10*np.log10(r.nmse(truth,prediction))
                            difference=abs(error-row.nmse_db)
                            assert difference<2e-6,(family,method,label,snr,difference)
                            records.append({"family":family,"scene_id":scene,"schedule":label,"snr_db":snr,
                                            "method":method,"recomputed_nmse_db":error,"difference_db":difference})
                            if family in audit.FAMILIES[4:] and method==name:
                                predicted=operator.forward(canonical)
                                true_energy=float(np.sum(abs(truth)**2))
                                predicted_energy=float(np.sum(abs(prediction)**2))
                                info={"frontend":config["evaluation_run"]["frontend"],"scene_id":scene,
                                    "schedule":label,"snr_db":snr,"nmse_db":error,
                                    "prediction_energy_fraction":predicted_energy/true_energy,
                                    "truth_prediction_correlation_squared":float(abs(np.vdot(truth,prediction))**2/
                                      max(true_energy*predicted_energy,1e-300)),
                                    "noisy_observation_residual_ratio":float((predicted-y).abs().square().sum()/y.abs().square().sum()),
                                    "clean_observation_error_ratio":float((predicted-clean[index:index+1]).abs().square().sum()/clean[index:index+1].abs().square().sum()),
                                    "final_state_real_rms":trace[-1]["real_rms"]}
                                if config["evaluation_run"]["frontend"]=="dma":
                                    gram=operator.physical.gram.to(torch.complex128)
                                    selected=clean[index:index+1].permute(0,3,1,2).to(gram)
                                    dual=(torch.linalg.pinv(gram,rtol=1e-10)@selected[...,None])[...,0].permute(0,2,3,1)
                                    projection=operator.adjoint(dual.to(clean))
                                    info["true_rowspace_energy_fraction"]=float(projection.square().sum()/images[index:index+1].square().sum())
                                details.append(info)
                print("replayed",family,label,flush=True)
            del images
        source.close()
        del model,operators
        torch.cuda.empty_cache()
    assert len(records)==64 and len(details)==16
    pd.DataFrame(records).to_csv(audit.OUT/"recovery_rechecks.csv",index=False)
    pd.DataFrame(details).to_csv(audit.OUT/"fixed_weight_diagnostics.csv",index=False)
    result={"frozen_replays":64,"max_difference_db":max(row["difference_db"] for row in records),
            "diagnostic_cases":[0,99],"diagnostic_snr_db":[20,30],"test_tuning_performed":False,
            "new_training_performed":False}
    (audit.OUT/"replay_integrity.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result),flush=True)
    print(pd.DataFrame(details).to_string(index=False),flush=True)


if __name__=="__main__":
    main()
