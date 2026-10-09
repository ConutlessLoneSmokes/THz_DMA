"""Audit completed D-033 validation and resolve the largest Gaussian control gaps."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import run_direction1_diffusion as r
import numpy as np
import pandas as pd
import torch


def main():
    root=r.PROJECT_ROOT/'runs'
    folder=root/'diffusion_v4_val_20261004_v1'
    output=folder/'analysis'
    output.mkdir(exist_ok=True)
    config=json.loads((folder/'config_resolved.json').read_text())['config']
    train=root/config['job']['train_run_id']
    data_dir=root/config['job']['data_run_id']
    checks={}
    for directory in (folder,train,data_dir):
        assert json.loads((directory/'status.json').read_text())['status']=='complete'
        manifest=json.loads((directory/'code_manifest.json').read_text())
        with zipfile.ZipFile(directory/'source_snapshot.zip') as z:
            for item in manifest:
                assert hashlib.sha256(z.read(item['path'].replace('\\','/'))).hexdigest()==item['sha256']
                if directory!=data_dir:
                    assert r.hash_file(r.PROJECT_ROOT/item['path'])==item['sha256']
        checks[directory.name+'_snapshot_files']=len(manifest)
    data_dir,data=r.require_dataset(data_dir.name,config)
    dep=json.loads((folder/'dependencies.json').read_text())
    assert dep['model_sha256']==r.hash_file(train/'checkpoint/training.pt')
    assert dep['dataset_sha256']==r.hash_file(data_dir/'dataset.json')
    checks['dataset_file_hashes_verified']=len(data['file_sha256'])
    frame=pd.read_csv(folder/'metrics_raw.csv')
    keys=['scene_id','schedule','array_reference_snr_db','estimator']
    methods={'oracle_geometry_ls','grid_omp_3d','sbl_jointfreq_3d','lmmse','diffusion_mean','gaussian_diffusion_mean'}
    expected={(s['name'],snr,m) for s in config['acquisition']['schedules']
              for snr in config['noise']['array_reference_snr_db'] for m in methods}
    assert len(frame)==1152 and frame.scene_id.nunique()==16 and not frame.duplicated(keys).any()
    assert all(set(g[keys[1:]].itertuples(index=False,name=None))==expected for _,g in frame.groupby('scene_id'))
    selected={r.read_example(data_dir,n)[1]['scene_id'] for n in data['index']['validation'][:16]}
    assert set(frame.scene_id)==selected
    saved=[row for p in (folder/'checkpoint').glob('*.json') for row in json.loads(p.read_text())]
    pd.testing.assert_frame_equal(frame.sort_values(keys).reset_index(drop=True),
        pd.DataFrame(saved).sort_values(keys).reset_index(drop=True).replace('',np.nan),
        check_dtype=False,rtol=1e-12,atol=1e-12)
    assert frame.groupby(keys[:3]).effective_output_snr_db.nunique().eq(1).all()
    for col in ['fullband_128_nmse_linear','pilot_nmse_linear','pilot_beam_loss_db','pilot_net_rate','estimation_and_reconstruction_seconds']:
        assert np.isfinite(frame[col]).all()
    for trace in frame[frame.estimator.str.contains('diffusion')].sampler_trace:
        last=json.loads(trace)[-1]
        assert last['initialization']=='conditional_gaussian' and last['antithetic']
    old=pd.read_csv(root/'diffusion_val_20261002_v1/metrics_raw.csv')
    common=frame[~frame.estimator.str.contains('diffusion')].merge(old,on=keys,suffixes=('_new','_old'),validate='one_to_one')
    differences={c:float(abs(common[c+'_new']-common[c+'_old']).max()) for c in
                 ['fullband_128_nmse_linear','effective_output_snr_db','pilot_net_rate']}
    assert len(common)==768 and all(v==0 for v in differences.values())
    checks.update(rows=1152,common_baseline_rows=768,baseline_max_differences=differences,
                  sbl_converged=int(frame[frame.estimator=='sbl_jointfreq_3d'].algorithm_converged.sum()))
    history=pd.read_json(train/'training_history.json')
    checkpoint=torch.load(train/'checkpoint/training.pt',map_location='cuda:0',weights_only=True)
    assert checkpoint['step']==10000 and len(history)==40
    assert checkpoint['best_step']==history.loc[history.validation_edm_loss.idxmin(),'step']
    checks.update(best_step=checkpoint['best_step'],best_weighted_edm_loss=checkpoint['best_loss'])
    summary=pd.read_csv(folder/'comparison.csv')
    for _,row in summary.iterrows():
        g=frame[(frame.schedule==row.schedule)&(frame.array_reference_snr_db==row.snr_db)&(frame.estimator==row.estimator)]
        assert abs(10*np.log10(g.fullband_128_nmse_linear.mean())-row.mean_linear_nmse_db)<1e-10
    pair=frame[frame.estimator=='gaussian_diffusion_mean'].merge(frame[frame.estimator=='lmmse'],on=keys[:3],suffixes=('_g','_l'))
    pair['gaussian_minus_lmmse_db']=pair.fullband_128_nmse_db_g-pair.fullband_128_nmse_db_l
    pair[keys[:3]+['gaussian_minus_lmmse_db']].to_csv(output/'gaussian_gaps.csv',index=False)
    r.atomic_json(output/'integrity.json',checks)
    print(json.dumps(checks),flush=True)
    # Paired full-CSI denoising only; does not alter or retrain either network.
    models={}
    for label,run in [('steps1000','diffusion_v2_train_probe_20261004_v1'),('steps10000',train.name)]:
        ck=torch.load(root/run/'checkpoint/training.pt',map_location='cuda:0',weights_only=True)
        models[label]=r.build_score_model(config['diffusion']['network']).to('cuda:0').eval()
        models[label].load_state_dict(ck['best_model'])
    denoising=[]
    generator=torch.Generator(device='cuda:0').manual_seed(2026100406)
    with torch.no_grad():
        for name in data['index']['validation'][:4]:
            x,c=r.load_batch(data_dir,[name],data['real_component_scale'],config,'cuda:0')
            for sigma in (.01,.1,1.,3.,10.,80.):
                noisy=x+sigma*torch.randn(x.shape,device=x.device,generator=generator)
                for label,model in models.items():
                    d=noisy-sigma*model(noisy,x.new_full((1,),sigma),c)
                    denoising.append(dict(name=name,sigma=sigma,model=label,
                        denoised_nmse=float((d-x).square().sum()/x.square().sum())))
    pd.DataFrame(denoising).to_csv(output/'paired_denoising.csv',index=False)
    del models
    # Analytic posterior mean with the actual truncated prior, for the two largest gaps.
    platform,weights,operators=r.build_stage2a_operators(config,r.PROJECT_ROOT)
    rows=[]
    for _,row in pair.nlargest(2,'gaussian_minus_lmmse_db').iterrows():
        seed,sample=map(int,row.scene_id.split(':'))
        name=next(n for n in data['index']['validation'] if r.read_example(data_dir,n)[1]['scene_id']==row.scene_id)
        _,record=r.read_example(data_dir,name)
        truth=r.record_channel(record,platform,config)
        scale=data['real_component_scale']
        basis=r.conditional_planar_covariance_basis(platform,config,record,draws=128,
            rng=np.random.default_rng(np.random.SeedSequence([2026092901,seed,sample])))
        prior=r.LowRankChannelPrior(basis/scale,device='cuda:0',relative_cutoff=1e-5)
        si,schedule=next((i,s) for i,s in enumerate(platform.schedules) if s.name==row.schedule)
        operator=operators[schedule.name]
        variance=r.array_input_noise_variance(truth,float(row.array_reference_snr_db))
        obs=r.observe_multistrip_schedule(weights[schedule.name],truth[schedule.pilot_indices],
            pilot_energy_per_re=1.,element_noise_variance=variance,
            rf_noise_variance=variance*config['noise']['rf_noise_variance_fraction_of_element_noise'],
            rng=np.random.default_rng(np.random.SeedSequence([seed,sample,si,2026090921])))
        y=torch.tensor((operator.whiten(obs.observed)/scale).reshape(operator.observation_shape),device='cuda:0',dtype=torch.complex64)[None]
        op=r.TorchDmaOperator(operator,schedule.pilot_indices,device='cuda:0')
        mean,_=prior.posterior_moments(op,y,variance/scale**2)
        estimate=(prior.u.to(torch.complex128)@mean).reshape(truth.shape).cpu().numpy()*scale
        nmse=10*np.log10(np.sum(abs(estimate-truth)**2)/np.sum(abs(truth)**2))
        rows.append(dict(scene_id=row.scene_id,schedule=row.schedule,snr_db=row.array_reference_snr_db,
            full_lmmse_db=row.fullband_128_nmse_db_l,truncated_lmmse_db=nmse,
            gaussian_diffusion_db=row.fullband_128_nmse_db_g,
            gaussian_minus_truncated_db=row.fullband_128_nmse_db_g-nmse,
            retained_energy=prior.retained_energy,rank=prior.rank))
    pd.DataFrame(rows).to_csv(output/'gaussian_truncation_check.csv',index=False)
    shutil.copyfile(__file__,output/Path(__file__).name)
    print('Denoising and Gaussian truncation checks saved',flush=True)


if __name__=='__main__':
    main()
