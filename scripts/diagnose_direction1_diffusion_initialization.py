"""Physical Gaussian moment, finite-chain and Fourier denoising controls for D-033."""
import argparse
import json
import zipfile

import numpy as np
import pandas as pd
import torch

import run_direction1_diffusion as r
from thz_dma.estimators.diffusion import LowRankChannelPrior, fourier_channel, pack_channel, unpack_channel


def fourier_probe(data_dir, data, model, config):
    rows = []
    rng = torch.Generator(device='cuda:0').manual_seed(2026100405)
    with torch.no_grad():
        for split in ('train','validation'):
            for name in data['index'][split][:4]:
                x, condition = r.load_batch(data_dir,[name],data['real_component_scale'],config,'cuda:0')
                spectrum = unpack_channel(fourier_channel(x))
                power = spectrum.abs().square().flatten()
                top = float(power.topk(max(1,len(power)//100)).values.sum()/power.sum())
                for sigma in (.3,1.,3.):
                    noisy = x+sigma*torch.randn(x.shape,device=x.device,generator=rng)
                    learned = noisy-sigma*model(noisy,noisy.new_full((1,),sigma),condition)
                    z = unpack_channel(fourier_channel(noisy))
                    soft = z*(1-3*sigma/z.abs().clamp_min(1e-12)).clamp_min(0)
                    estimate = fourier_channel(pack_channel(soft),inverse=True)
                    rows.append(dict(split=split,name=name,sigma=sigma,top_one_percent_energy=top,
                        old_denoiser_nmse=float((learned-x).square().sum()/x.square().sum()),
                        soft_threshold_nmse=float((estimate-x).square().sum()/x.square().sum())))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    if not r.re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id):
        raise ValueError('invalid run id')
    settings=r.read_toml(r.PROJECT_ROOT/'configs/direction1_diffusion_v4_probe.toml')
    config=r.read_toml(r.PROJECT_ROOT/settings['base_config'])
    config['diffusion']=settings
    config['diagnostic']=dict(scenes=2,snr_db=55,data_run='diffusion_data_20261002_v1',
                              old_model='diffusion_train_20261002_v1',fourier_seed=2026100405)
    archive=r.create_run_archive('diffusion_initialization',args.run_id,
        config_path=r.PROJECT_ROOT/'configs/direction1_diffusion_v4_probe.toml',config=config,
        seeds=config['diagnostic'],script_path=__file__,environment={'torch':str(torch.__version__)},
        checkpoint_note='Diagnostic only; analytic latent Gaussian moments separate error sources.\n',
        extra_manifest_files=[r.PROJECT_ROOT/'scripts/run_direction1_diffusion.py'])
    with zipfile.ZipFile(archive.directory/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as bundle:
        for row in json.loads((archive.directory/'code_manifest.json').read_text()):
            bundle.write(r.PROJECT_ROOT/row['path'],row['path'])
    try:
        data_dir=r.PROJECT_ROOT/'runs'/config['diagnostic']['data_run']
        data=json.loads((data_dir/'dataset.json').read_text())
        model_path=r.PROJECT_ROOT/'runs'/config['diagnostic']['old_model']/'checkpoint/training.pt'
        checkpoint=torch.load(model_path,map_location='cuda:0',weights_only=True)
        assert checkpoint['dataset_sha256']==r.hash_file(data_dir/'dataset.json')
        model=r.ConditionalScoreNet().to('cuda:0').eval()
        model.load_state_dict(checkpoint['best_model'])
        archive.write_json('dependencies.json',dict(dataset_sha256=r.hash_file(data_dir/'dataset.json'),
                                                   old_model_sha256=r.hash_file(model_path)))
        pd.DataFrame(fourier_probe(data_dir,data,model,config)).to_csv(archive.directory/'fourier_probe.csv',index=False)
        del model
        platform,weights,operators=r.build_stage2a_operators(config,r.PROJECT_ROOT)
        rows=[]
        for name in data['index']['validation'][:2]:
            assert r.hash_file(data_dir/'data'/name)==data['file_sha256'][name]
            _,record=r.read_example(data_dir,name)
            seed,sample=map(int,record['scene_id'].split(':'))
            truth=r.record_channel(record,platform,config)
            scale=data['real_component_scale']
            h=torch.tensor(truth/scale,device='cuda:0').flatten().to(torch.complex128)
            basis=r.conditional_planar_covariance_basis(platform,config,record,
                draws=128,rng=np.random.default_rng(np.random.SeedSequence([2026092901,seed,sample])))
            prior=LowRankChannelPrior(basis/scale,device='cuda:0')
            condition=torch.tensor(r.center_condition(record['centers_m_rad_rad'],config),device='cuda:0')[None]
            for si,schedule in enumerate(platform.schedules):
                operator=operators[schedule.name]
                op=r.TorchDmaOperator(operator,schedule.pilot_indices,device='cuda:0')
                variance=r.array_input_noise_variance(truth,55)
                obs=r.observe_multistrip_schedule(weights[schedule.name],truth[schedule.pilot_indices],
                    pilot_energy_per_re=1.,element_noise_variance=variance,
                    rf_noise_variance=variance*config['noise']['rf_noise_variance_fraction_of_element_noise'],
                    rng=np.random.default_rng(np.random.SeedSequence([seed,sample,si,2026090921])))
                white=operator.whiten(obs.observed)
                y=torch.tensor((white/scale).reshape(operator.observation_shape),device='cuda:0',dtype=torch.complex64)[None]
                mean,cov=prior.posterior_moments(op,y,variance/scale**2)
                values,q=torch.linalg.eigh(cov)
                observed_basis=r.project_basis(basis,weights[schedule.name],operator,schedule.pilot_indices)
                coefficients=r.lmmse_coefficients(observed_basis,white,variance)
                lmmse=torch.tensor(basis.reshape(len(basis),-1).T@coefficients.astype(np.complex64)/scale,device='cuda:0').to(torch.complex128)
                u=prior.u.to(torch.complex128)
                physical_mean=u@mean
                power=float(h.abs().square().sum())
                def nmse(z):return float((z-h).abs().square().sum())/power
                for hi in (80.,800.):
                    levels=np.geomspace(hi,.01,64)
                    transfer=torch.ones_like(values)
                    for a,b in zip(levels,list(levels[1:])+[0.]):
                        first=2*a/(values+2*a*a)
                        second=2*b/(values+2*b*b)
                        transfer*=1+(b-a)*first if b==0 else 1+(b-a)*.5*(first+second*(1+(b-a)*first))
                    exact_transfer=torch.sqrt(values/(values+2*hi*hi))
                    for initialization in ('sigma_gaussian','conditional_gaussian'):
                        startmean=torch.zeros_like(mean) if initialization=='sigma_gaussian' else mean
                        finishmean=mean+q@(transfer*(q.H@(startmean-mean)))
                        finishcov=transfer.square()*(2*hi*hi+(values if initialization=='conditional_gaussian' else 0))
                        row=dict(scene_id=record['scene_id'],schedule=schedule.name,initialization=initialization,
                            sigma_max=hi,rank=prior.rank,retained_energy=prior.retained_energy,
                            lmmse_nmse=nmse(lmmse),truncated_lmmse_nmse=nmse(physical_mean),
                            truncation_relative_difference=float((physical_mean-lmmse).abs().square().sum()/lmmse.abs().square().sum()),
                            expected_mean_nmse=nmse(u@finishmean),
                            expected_four_chain_nmse=nmse(u@finishmean)+float(finishcov.sum())/(4*power),
                            covariance_relative_error=float(torch.linalg.norm(finishcov-values)/torch.linalg.norm(values)),
                            ode_transfer_relative_error=float(torch.linalg.norm(transfer-exact_transfer)/torch.linalg.norm(exact_transfer)))
                        rows.append(row)
                # Match the old 4-chain seed and compare guided physical score to exact latent flow.
                ps=int(np.random.SeedSequence([settings['posterior']['seed'],seed,sample,si,2]).generate_state(1)[0])
                generator=torch.Generator(device='cuda:0').manual_seed(ps)
                start=torch.randn((4,16,160,128),device='cuda:0',generator=generator)*80
                z=prior.coefficients(start).to(torch.complex128)
                exact=mean+((z-mean)@q.conj()*torch.sqrt(values/(values+2*80**2)))@q.T
                expected_estimate=u@exact.mean(0)
                old_settings={**settings['posterior'],'initialization':'sigma_gaussian','antithetic':False}
                samples,_=r.posterior_samples(None,op,y,condition,variance/scale**2,old_settings,seed=ps,gaussian_prior=prior)
                actual=samples.mean(0).flatten().to(torch.complex128)
                rows[-4].update(fixed_four_chain_exact_nmse=nmse(expected_estimate),
                    fixed_four_chain_guided_nmse=nmse(actual),
                    numerical_relative_error=float((actual-expected_estimate).abs().square().sum()/expected_estimate.abs().square().sum()))
                pd.DataFrame(rows).to_csv(archive.directory/'physical_gaussian_controls.csv',index=False)
                print(record['scene_id'],schedule.name,'Gaussian control done',flush=True)
        archive.write_status('complete',run_id=args.run_id,diagnostic_only=True)
    except BaseException as error:
        archive.write_status('failed',error=str(error));raise


if __name__=='__main__':
    main()
