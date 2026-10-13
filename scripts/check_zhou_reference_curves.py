"""Shared LMMSE/LASSO anchors on Zhou's public QuaDRiGa channels, without retraining."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

from _run_support import PROJECT_ROOT, create_run_archive, hash_file
if sys.platform == "win32":
    sys.path.insert(0,str(PROJECT_ROOT/'sources/code/zhou2025_python'))
import h5py
import numpy as np
import pandas as pd
import torch
from zhou2025_repro.data import angular
from zhou2025_repro.baselines import real_operator, lasso


def read_channels(path, start, stop):
    with h5py.File(path,'r') as file:
        if start<0 or stop<=start or stop>file['output_h'].shape[-1]:
            raise ValueError('channel range exceeds the public dataset')
        raw=file['output_h'][:,:,start:stop]
    h=(raw['real']+1j*raw['imag']).transpose(2,1,0)
    if h.shape[1:]!=(16,64) or not np.isfinite(h).all():
        raise ValueError('expected finite public 16x64 MIMO channels')
    return h.astype(np.complex64)


def normalize(h, mode, scale):
    if mode=='author_loader':
        h=h-h.mean(axis=(1,2),keepdims=True)
        scale=np.sqrt(np.mean(abs(h)**2,axis=(1,2),keepdims=True))
    if np.any(np.asarray(scale)<=0):
        raise ValueError('positive normalization scale required')
    return torch.as_tensor(h/scale,device='cuda:0')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--normalization',choices=['author_loader','paper_global'],default='author_loader')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if not rvalid(args.run_id) or not torch.cuda.is_available():
        raise ValueError('valid run ID and CUDA required')
    directory=PROJECT_ROOT/'sources/datasets/zhou2025'
    paths={s:directory/f'quadriga_Nt64_Nr16_ULA0.50_{s}.mat' for s in ('train','valid','test')}
    counts=dict(train=1024,valid=8,test=8) if args.smoke else dict(train=100000,valid=100,test=100)
    snrs=[-10.,20.] if args.smoke else list(map(float,range(-10,31,5)))
    setup=dict(reference='zhou2025generativechannels',seed=2026100801,Nt=64,Nr=16,
        pilot_counts=[64,38],snr_db=snrs,normalization=args.normalization,counts=counts,
        lasso_factors=[.01,.03,.1,.3,1.,3.],lasso_steps=200 if args.smoke else 1000,
        real_covariance_lmmse=True,smoke_only=args.smoke,
        author_commit='2b922440f72ff6943c1cbe5ae85cbd50c421ac83')
    archive=create_run_archive('zhou_reference',args.run_id,config_path=Path(__file__),config=setup,
        seeds={'seed':setup['seed']},script_path=__file__,environment={'torch':str(torch.__version__),'h5py':h5py.__version__},
        checkpoint_note='Shared reference baselines only, not exact reproduction or retrained DMA diffusion.\n')
    with zipfile.ZipFile(archive.directory/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for item in json.loads((archive.directory/'code_manifest.json').read_text()):
            z.write(PROJECT_ROOT/item['path'],item['path'])
    archive.write_json('dependencies.json',{s:{'path':str(p),'sha256':hash_file(p)} for s,p in paths.items()})
    try:
        if args.normalization=='paper_global':
            energy=0.
            for start in range(0,counts['train'],1024):
                h=read_channels(paths['train'],start,min(start+1024,counts['train']))
                energy+=np.sum(abs(h.astype(np.complex128))**2)
            scale=np.sqrt(energy/(counts['train']*1024))
        else:
            scale=1.
        covariance=torch.zeros((2048,2048),device='cuda:0',dtype=torch.float32)
        for start in range(0,counts['train'],1024):
            h=normalize(read_channels(paths['train'],start,min(start+1024,counts['train'])),args.normalization,scale)
            vectors=angular(h).transpose(-1,-2).flatten(1)
            vectors=torch.cat((vectors.real,vectors.imag),dim=1)
            covariance+=vectors.T@vectors/counts['train']
        covariance=(covariance+covariance.T)/2
        archive.write_json('normalization.json',{'training_global_scale':float(scale),
            'normalization':args.normalization,'mean_training_normalized_entry_power':float(2*covariance.diag().mean()),
            'paper_text_vs_author_loader_difference':True})
        channels={s:angular(normalize(read_channels(paths[s],0,counts[s]),args.normalization,scale)) for s in ('valid','test')}
        rng=np.random.default_rng(setup['seed'])
        tt=torch.tensor(np.fft.fft(np.eye(64),axis=0)/8,device='cuda:0',dtype=torch.complex64)
        rows=[];tuning=[]
        for pilots in setup['pilot_counts']:
            p=torch.tensor((2*rng.integers(0,2,(64,pilots))-1+1j*(2*rng.integers(0,2,(64,pilots))-1))/np.sqrt(2),device='cuda:0',dtype=torch.complex64)
            b=tt.H@p
            # Unit receive DFT preserves white noise and avoids an unnecessary dense receiver transform.
            a=torch.kron(b.T.contiguous(),torch.eye(16,device='cuda:0',dtype=torch.complex64))
            a=real_operator(a).double()
            cross=covariance.double()@a.T
            observed=a@cross
            values,vectors=torch.linalg.eigh((observed+observed.T)/2)
            values=values.clamp_min(0)
            noise={s:torch.tensor((rng.normal(size=(counts[s],16,pilots))+1j*rng.normal(size=(counts[s],16,pilots)))/np.sqrt(2),device='cuda:0',dtype=torch.complex64) for s in ('valid','test')}
            for snr in snrs:
                variance=64/10**(snr/10)
                received={s:h@b+np.sqrt(variance)*noise[s] for s,h in channels.items()}
                def losses(estimate,split):
                    return ((estimate-channels[split]).abs().square().sum((-1,-2))/channels[split].abs().square().sum((-1,-2))).cpu().numpy()
                penalties=np.asarray(setup['lasso_factors'])*np.sqrt(variance)*float(torch.linalg.vector_norm(b,dim=1).max())*np.sqrt(np.log(1024))
                options=[]
                for penalty in penalties:
                    estimate,_,_=lasso(received['valid'],b,float(penalty),steps=setup['lasso_steps'])
                    options.append(float(losses(estimate,'valid').mean()))
                chosen=int(np.argmin(options))
                estimate,iterations,converged=lasso(received['test'],b,float(penalties[chosen]),steps=setup['lasso_steps'])
                results={'lasso':losses(estimate,'test')}
                y=received['test'].transpose(-1,-2).flatten(1)
                y=torch.cat((y.real,y.imag),dim=1).double()
                dual=(y@vectors)/(values+variance/2)
                estimate=dual@vectors.T@cross.T
                estimate=torch.complex(estimate[:,:1024],estimate[:,1024:]).reshape(counts['test'],64,16).transpose(-1,-2)
                results['lmmse']=losses(estimate,'test')
                for method,errors in results.items():
                    for number,error in enumerate(errors):
                        rows.append(dict(scene_id=number,pilots=pilots,alpha=pilots/64,snr_db=snr,estimator=method,nmse_linear=float(error),
                            lasso_iterations=iterations if method=='lasso' else None,lasso_converged=converged if method=='lasso' else None))
                tuning.append(dict(pilots=pilots,snr_db=snr,selected_penalty=float(penalties[chosen]),
                    factors=setup['lasso_factors'],validation_nmse=options))
                archive.write_metrics(rows)
                archive.write_json('validation_tuning.json',tuning)
                print(pilots,snr,{m:round(10*np.log10(e.mean()),3) for m,e in results.items()},flush=True)
        frame=pd.DataFrame(rows)
        assert len(frame)==counts['test']*len(snrs)*2*2 and not frame.duplicated(['scene_id','pilots','snr_db','estimator']).any()
        summary=frame.groupby(['pilots','snr_db','estimator']).nmse_linear.mean().reset_index()
        summary['nmse_db']=10*np.log10(summary.nmse_linear)
        summary.to_csv(archive.directory/'comparison.csv',index=False)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True,constrained_layout=True)
        for ax,(pilots,g) in zip(axes,summary.groupby('pilots',sort=False)):
            for method,curve in g.groupby('estimator'):ax.plot(curve.snr_db,curve.nmse_db,'o-',label=method.upper())
            ax.set(title=f'Zhou public data, Np={pilots}',xlabel='Paper nominal pilot SNR (dB)',xlim=(-10,30));ax.grid(alpha=.3);ax.legend()
        axes[0].set_ylabel('NMSE (dB; mean sample ratio)')
        fig.savefig(archive.directory/'comparison.png',dpi=160);fig.savefig(archive.directory/'comparison.pdf');plt.close(fig)
        archive.write_status('complete',run_id=args.run_id,smoke_only=args.smoke)
    except BaseException as error:
        archive.write_status('failed',error=str(error));raise


def rvalid(identifier):
    import re
    return re.fullmatch(r'[A-Za-z0-9_-]+',identifier)


if __name__=='__main__':
    main()
