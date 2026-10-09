import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path.cwd()/"scripts"))
import run_direction1_diffusion as r
p=Path("runs/diffusion_observation_snr_endpoint_20261008_v1")
config=json.loads((p/"config_resolved.json").read_text())["config"]
settings=config["diffusion"]
data_dir=r.PROJECT_ROOT/"runs/diffusion_data_20261002_v1"
data=json.loads((data_dir/"dataset.json").read_text())
_,record=r.read_example(data_dir,data["index"]["validation"][0])
seed,sample=map(int,record["scene_id"].split(":"))
platform,weights,operators=r.build_stage2a_operators(config,r.PROJECT_ROOT)
truth=r.record_channel(record,platform,config);scale=data["real_component_scale"]
basis=r.conditional_planar_covariance_basis(platform,config,record,draws=128,rng=np.random.default_rng(np.random.SeedSequence([2026092901,seed,sample])))
prior=r.LowRankChannelPrior(basis/scale,device="cuda:0",relative_cutoff=1e-5)
reference=json.loads((p/"observation_snr_calibration.json").read_text())["training_reference_power"]
f=pd.read_csv(p/"metrics_raw.csv");rows=[]
for si,schedule in enumerate(platform.schedules):
 op=operators[schedule.name];variance=reference[schedule.name]/1000
 obs=r.observe_multistrip_schedule(weights[schedule.name],truth[schedule.pilot_indices],pilot_energy_per_re=1.,element_noise_variance=variance,rf_noise_variance=0.,rng=np.random.default_rng(np.random.SeedSequence([seed,sample,si,2026090921])))
 y=torch.tensor((op.whiten(obs.observed)/scale).reshape(op.observation_shape),device="cuda:0",dtype=torch.complex64)[None]
 tensor_op=r.TorchDmaOperator(op,schedule.pilot_indices,device="cuda:0")
 mean,_=prior.posterior_moments(tensor_op,y,variance/scale**2)
 estimate=(prior.u.to(torch.complex128)@mean).reshape(truth.shape).cpu().numpy()*scale
 nmse=float(10*np.log10(np.sum(abs(estimate-truth)**2)/np.sum(abs(truth)**2)))
 g=f[(f.schedule==schedule.name)&(f.estimator=="gaussian_diffusion_mean")].fullband_128_nmse_db.item()
 rows.append(dict(schedule=schedule.name,target_snr_db=30,truncated_lmmse_db=nmse,gaussian_diffusion_db=g,difference_db=g-nmse))
pd.DataFrame(rows).to_csv(p/"analysis/gaussian_truncation_30db.csv",index=False)
print(pd.DataFrame(rows).to_string(index=False))
