"""Audit D-034 curves, actual reference powers, and signal/noise scaling equivalence."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import torch

import run_direction1_diffusion as r
import check_zhou_reference_curves as z


def scaling_control():
    rng=np.random.default_rng(2026100802)
    a=rng.normal(size=(4,8))+1j*rng.normal(size=(4,8))
    h=rng.normal(size=8)+1j*rng.normal(size=8)
    noise=(rng.normal(size=4)+1j*rng.normal(size=4))/np.sqrt(2)
    clean=a@h;power=np.mean(abs(clean)**2);gamma=100.;variance=power/gamma
    original=clean+np.sqrt(variance)*noise
    unit_noise=np.sqrt(gamma)*clean/np.sqrt(power)+noise
    np.testing.assert_allclose(original/np.sqrt(variance),unit_noise,rtol=1e-12,atol=1e-12)
    covariance=np.eye(8)
    def lmmse(matrix,y,nu):
        return covariance@matrix.conj().T@np.linalg.solve(matrix@covariance@matrix.conj().T+nu*np.eye(4),y)
    first=lmmse(a,original,variance)
    second=lmmse(a/np.sqrt(variance),unit_noise,1.)
    np.testing.assert_allclose(first,second,rtol=1e-12,atol=1e-12)
    return dict(target_snr_db=20,amplitude_factor=10.,power_factor=100.,
        observation_max_difference=float(abs(original/np.sqrt(variance)-unit_noise).max()),
        lmmse_max_difference=float(abs(first-second).max()))


def main():
    names=['zhou_reference_loader_20261008_v1','zhou_reference_global_20261008_v1','diffusion_observation_snr_20261008_v1']
    output=r.PROJECT_ROOT/'runs'/names[-1]/'analysis'
    output.mkdir(exist_ok=True)
    checks={};hashes={};frames={}
    for name in names:
        folder=r.PROJECT_ROOT/'runs'/name
        assert json.loads((folder/'status.json').read_text())['status']=='complete'
        config=json.loads((folder/'config_resolved.json').read_text())['config']
        manifest=json.loads((folder/'code_manifest.json').read_text())
        with zipfile.ZipFile(folder/'source_snapshot.zip') as bundle:
            for item in manifest:
                assert hashlib.sha256(bundle.read(item['path'].replace('\\','/'))).hexdigest()==item['sha256']
                assert r.hash_file(r.PROJECT_ROOT/item['path'])==item['sha256'],item['path']
        dependencies=json.loads((folder/'dependencies.json').read_text())
        if name.startswith('zhou'):
            for entry in dependencies.values():
                path=Path(entry['path'])
                if path not in hashes:hashes[path]=r.hash_file(path)
                assert hashes[path]==entry['sha256']
            key=['scene_id','pilots','snr_db','estimator'];expected_rows=3600
            expected={(p,s,m) for p in config['pilot_counts'] for s in config['snr_db'] for m in ('lasso','lmmse')}
            summary=pd.read_csv(folder/'comparison.csv')
            value_column='nmse_linear'
        else:
            data_dir,data=r.require_dataset(config['job']['data_run_id'],config)
            assert dependencies['dataset_sha256']==r.hash_file(data_dir/'dataset.json')
            assert dependencies['model_sha256']==r.hash_file(r.PROJECT_ROOT/'runs'/config['job']['train_run_id']/'checkpoint/training.pt')
            key=['scene_id','schedule','observation_snr_target_db','estimator'];expected_rows=1728
            methods=['oracle_geometry_ls','grid_omp_3d','sbl_jointfreq_3d','lmmse','diffusion_mean','gaussian_diffusion_mean']
            expected={(s['name'],n,m) for s in config['acquisition']['schedules'] for n in config['job']['observation_snr_db'] for m in methods}
            summary=pd.read_csv(folder/'comparison.csv');value_column='fullband_128_nmse_linear'
        f=pd.read_csv(folder/'metrics_raw.csv');frames[name]=f
        assert len(f)==expected_rows and not f.duplicated(key).any()
        assert all(set(g[key[1:]].itertuples(index=False,name=None))==expected for _,g in f.groupby('scene_id'))
        assert np.isfinite(f[value_column]).all()
        curves=f.groupby(key[1:])[value_column].mean().map(lambda x:10*np.log10(x))
        for _,row in summary.iterrows():
            lookup=(row.pilots,row.snr_db,row.estimator) if name.startswith('zhou') else (row.schedule,row.snr_db,row.estimator)
            assert abs(curves.loc[lookup]-(row.nmse_db if name.startswith('zhou') else row.mean_linear_nmse_db))<1e-10
        if name.startswith('diffusion'):
            saved=[row for p in (folder/'checkpoint').glob('*.json') for row in json.loads(p.read_text())]
            pd.testing.assert_frame_equal(f.sort_values(key).reset_index(drop=True),
                pd.DataFrame(saved).sort_values(key).reset_index(drop=True).replace('',np.nan),check_dtype=False,atol=1e-12,rtol=1e-12)
            assert f.groupby(key[:3]).effective_output_snr_db.nunique().eq(1).all()
        checks[name]=dict(rows=len(f),scenes=f.scene_id.nunique(),source_snapshot_files=len(manifest),dependencies_verified=True)
    dma=frames[names[-1]]
    checks['sbl_converged']=int(dma[dma.estimator=='sbl_jointfreq_3d'].algorithm_converged.sum())
    nonconverged=[]
    for name in names[:2]:
        f=frames[name];bad=f[(f.estimator=='lasso')&f.lasso_converged.eq(False)]
        for (pilots,snr),g in bad.groupby(['pilots','snr_db']):
            nonconverged.append(dict(run=name,pilots=int(pilots),snr_db=float(snr),rows=len(g),iterations=int(g.lasso_iterations.max())))
    checks['lasso_batches_reaching_iteration_limit']=nonconverged
    r.atomic_json(output/'integrity.json',checks)
    r.atomic_json(output/'scaling_equivalence.json',scaling_control())
    print(json.dumps(checks),flush=True)

    # Recreate each fixed pilot exactly, including the noise RNG draws before the next pilot.
    rng=np.random.default_rng(2026100801);pilots={}
    for count in (64,38):
        p=(2*rng.integers(0,2,(64,count))-1+1j*(2*rng.integers(0,2,(64,count))-1))/np.sqrt(2)
        pilots[count]=torch.tensor(p,dtype=torch.complex64,device='cuda:0')
        for _ in range(4):rng.normal(size=(100,16,count))
    global_scale=json.loads((r.PROJECT_ROOT/'runs'/names[1]/'normalization.json').read_text())['training_global_scale']
    dataset=r.PROJECT_ROOT/'sources/datasets/zhou2025'
    totals={(mode,count):0. for mode in ('author_loader','paper_global') for count in pilots}
    with torch.no_grad():
        for start in range(0,100000,1024):
            raw=z.read_channels(dataset/'quadriga_Nt64_Nr16_ULA0.50_train.mat',start,min(start+1024,100000))
            for mode in ('author_loader','paper_global'):
                h=z.normalize(raw,mode,global_scale)
                for count,p in pilots.items():
                    totals[mode,count]+=float((h@p).abs().square().double().sum())/(16*count)
    powers=[]
    for (mode,count),total in totals.items():
        measured=total/100000
        powers.append(dict(normalization=mode,pilots=count,training_receive_power=measured,
            paper_nominal_receive_power=64.,measured_minus_nominal_snr_db=10*np.log10(measured/64)))
    pd.DataFrame(powers).to_csv(output/'reference_receive_power.csv',index=False)
    print(pd.DataFrame(powers).to_string(index=False),flush=True)
    raw=z.read_channels(dataset/'quadriga_Nt64_Nr16_ULA0.50_test.mat',0,100)
    powers=(abs(raw.astype(np.complex128))**2).mean((1,2))/global_scale**2
    r.atomic_json(output/'global_test_power.json',{'min':float(powers.min()),'median':float(np.median(powers)),
        'mean':float(powers.mean()),'max':float(powers.max()),
        'entry_power_5_95_percentiles':np.quantile(powers,[.05,.95]).tolist(),
        'relative_snr_5_95_percentiles_db':(10*np.log10(np.quantile(powers,[.05,.95]))).tolist()})
    rows=[]
    for name in names[:2]:
        f=frames[name]
        for snr in (-10.,20.):
            g=f[(f.estimator=='lmmse')&(f.pilots==64)&(f.snr_db==snr)].sort_values('scene_id')
            for lo,hi in ((0,25),(25,75),(75,100)):
                subset=np.argsort(powers)[lo:hi]
                rows.append(dict(normalization='author_loader' if name==names[0] else 'paper_global',snr_db=snr,
                    power_quantile_group=f'{lo}-{hi}%',nmse_db=float(10*np.log10(g.nmse_linear.to_numpy()[subset].mean()))))
    pd.DataFrame(rows).to_csv(output/'reference_power_strata.csv',index=False)
    # Aggregate by means in linear power as well as by mean dB; these are different statistics.
    rows=[]
    for (schedule,target),g in dma[dma.estimator=='lmmse'].groupby(['schedule','observation_snr_target_db']):
        rows.append(dict(schedule=schedule,target_db=target,
            mean_sample_db=float(g.whitened_observation_snr_db.mean()),
            linear_mean_then_db=float(10*np.log10(np.mean(10**(g.whitened_observation_snr_db/10)))),
            rf_mean_sample_db=float(g.effective_output_snr_db.mean())))
    pd.DataFrame(rows).to_csv(output/'actual_snr_summary.csv',index=False)
    shared_axis_plots(output)
    (output/'audit_direction1_snr_curves.py').write_text(Path(__file__).read_text(),encoding='utf-8')


def shared_axis_plots(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    powers=pd.read_csv(output/'reference_receive_power.csv')
    curves=[]
    for mode,run in [('author_loader','zhou_reference_loader_20261008_v1'),('paper_global','zhou_reference_global_20261008_v1')]:
        f=pd.read_csv(r.PROJECT_ROOT/'runs'/run/'comparison.csv')
        offsets=powers[powers.normalization==mode].set_index('pilots').measured_minus_nominal_snr_db
        f['shared_receive_snr_db']=f.snr_db+f.pilots.map(offsets)
        f['normalization']=mode
        curves.append(f)
    combined=pd.concat(curves,ignore_index=True)
    combined.to_csv(output/'reference_shared_snr_curves.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True,constrained_layout=True)
    for ax,pilots in zip(axes,[64,38]):
        g=combined[combined.pilots==pilots]
        for (mode,method),c in g.groupby(['normalization','estimator']):
            ax.plot(c.shared_receive_snr_db,c.nmse_db,'o-' if method=='lmmse' else 'x--',
                color='tab:blue' if mode=='author_loader' else 'tab:orange',
                label=method.upper()+(', per sample' if mode=='author_loader' else ', training global'))
        ax.set(title=f'Zhou public data: Np={pilots}',xlabel='Training-reference received SNR (dB)',xlim=(-11,31),ylim=(-32,5))
        ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[0].set_ylabel('NMSE (dB; mean per-channel ratio)')
    fig.savefig(output/'reference_normalization_comparison.png',dpi=180)
    fig.savefig(output/'reference_normalization_comparison.pdf');plt.close(fig)
    f=pd.read_csv(r.PROJECT_ROOT/'runs/diffusion_observation_snr_20261008_v1/comparison.csv')
    labels={'grid_omp_3d':'OMP','sbl_jointfreq_3d':'SBL','lmmse':'LMMSE',
        'diffusion_mean':'Diffusion v4','oracle_geometry_ls':'Oracle geometry + LS','gaussian_diffusion_mean':'Gaussian control'}
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True,constrained_layout=True)
    for ax,(schedule,g) in zip(axes,f.groupby('schedule')):
        for method,c in g.groupby('estimator'):
            c=c.sort_values('snr_db');ax.plot(c.snr_db,c.mean_linear_nmse_db,'o-',label=labels[method])
        ax.set(title='F1: 1 config x 128 tones' if 'J1' in schedule else 'F2: 8 configs x 16 tones',
            xlabel='Training-reference whitened received SNR (dB)',xlim=(-10,30),ylim=(-60,5))
        ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[0].set_ylabel('Fullband NMSE (dB; mean per-channel ratio)')
    fig.savefig(output/'dma_common_observation_snr.png',dpi=180)
    fig.savefig(output/'dma_common_observation_snr.pdf');plt.close(fig)


if __name__=='__main__':
    main()
