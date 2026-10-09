"""D-033: analytic Gaussian controls and paired physical error decomposition."""
import argparse
import json
from pathlib import Path
import zipfile

import run_direction1_diffusion as r
import numpy as np
import pandas as pd
import torch
from thz_dma.estimators.diffusion import heun_flow, unpack_channel


def gaussian_controls():
    rng = np.random.default_rng(2026100403)
    n, m = 16, 4
    a = rng.normal(size=(m,n))/np.sqrt(n)
    y = rng.normal(size=m)
    eye, noise = np.eye(n), .1*np.eye(m)
    rotation = np.linalg.qr(rng.normal(size=(n,n)))[0]
    results = []
    for kind, prior in (("isotropic",eye), ("correlated",(rotation*np.geomspace(.01,4,n))@rotation.T)):
        covariance = np.linalg.inv(np.linalg.inv(prior)+a.T@np.linalg.solve(noise,a))
        mean = covariance@a.T@np.linalg.solve(noise,y)
        def linear_score(sigma, variant):
            inv = np.linalg.inv(prior+sigma*sigma*eye)
            if variant == "exact":
                j = -np.linalg.inv(covariance+sigma*sigma*eye)
                return j, -j@mean
            d = prior@inv
            if variant == "old":
                c = noise+sigma*sigma*a@a.T
                v = a.T@np.linalg.inv(c)
                return -inv-v@a, v@y
            tau = sigma*sigma/(1+sigma*sigma)*eye if variant == "iso_moment" else prior-d@prior
            v = d.T@a.T@np.linalg.inv(noise+a@tau@a.T)
            return -inv-v@a@d, v@y
        for sigma in (.01,.3,1.,3.):
            jt,bt=linear_score(sigma,"exact")
            for variant in ("old","iso_moment","full_moment"):
                j,b=linear_score(sigma,variant)
                error=np.sqrt(np.sum((j-jt)**2)+np.sum((b-bt)**2))/np.sqrt(np.sum(jt**2)+np.sum(bt**2))
                results.append(dict(prior=kind,experiment="score",method=variant,sigma=sigma,relative_error=error))
                if variant=="full_moment" or (kind=="isotropic" and variant=="iso_moment"):
                    assert error < 1e-9
        # Exact moment recursion removes Monte Carlo noise from the old ALD control.
        mu, cov = np.zeros(n),9*eye
        for sigma in np.geomspace(3,.01,32):
            j,b=linear_score(sigma,"old");eta=.05*sigma*sigma;transition=eye+eta*j
            for _ in range(5):
                mu=transition@mu+eta*b;cov=transition@cov@transition.T+2*eta*eye
        d=prior@np.linalg.inv(prior+.01**2*eye);mu=d@mu;cov=d@cov@d.T
        results.append(dict(prior=kind,experiment="moments",method="old_ald",initial="sigma_gaussian",levels=32,
            mean_relative_error=np.linalg.norm(mu-mean)/np.linalg.norm(mean),
            covariance_relative_error=np.linalg.norm(cov-covariance)/np.linalg.norm(covariance)))
        points=np.concatenate((eye,-eye))*np.sqrt(n)
        for variant,initial,levels in (("exact","exact_marginal",32),("exact","exact_marginal",128),
                ("exact","sigma_gaussian",128),("iso_moment","sigma_gaussian",128)):
            start=mean+points@np.linalg.cholesky(covariance+9*eye).T if initial=="exact_marginal" else 3*points
            def score(x,s):
                j,b=linear_score(s,variant)
                return x@torch.tensor(j.T)+torch.tensor(b)
            samples,_=heun_flow(torch.tensor(start),np.geomspace(3,.01,levels),score)
            samples=samples.numpy();mu=samples.mean(0);diff=samples-mu;cov=diff.T@diff/len(diff)
            err=np.linalg.norm(cov-covariance)/np.linalg.norm(covariance)
            if variant=="exact" and initial=="exact_marginal" and levels==128:
                assert np.linalg.norm(mu-mean)<1e-10 and err<.002
            results.append(dict(prior=kind,experiment="moments",method=variant+"_heun",initial=initial,levels=levels,
                mean_relative_error=np.linalg.norm(mu-mean)/np.linalg.norm(mean),covariance_relative_error=err))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",required=True)
    args=parser.parse_args()
    if not r.re.fullmatch(r"[A-Za-z0-9_-]+",args.run_id):
        raise ValueError("invalid run id")
    settings=r.read_toml(r.PROJECT_ROOT/"configs/direction1_diffusion_guided_probe.toml")
    config=r.read_toml(r.PROJECT_ROOT/settings["base_config"])
    config["diffusion"]=settings
    config["diagnostic"]={"scenes":2,"snr_db":55,"gaussian_seed":2026100403,
        "data_run":"diffusion_data_20261002_v1","train_run":"diffusion_train_20261002_v1"}
    archive=r.create_run_archive("diffusion_diagnostic",args.run_id,
        config_path=r.PROJECT_ROOT/"configs/direction1_diffusion_guided_probe.toml",config=config,
        seeds=config["diagnostic"],script_path=__file__,environment={"torch":str(torch.__version__)},
        checkpoint_note="D-033 diagnostic only; immutable Gaussian/physical controls.\n",
        extra_manifest_files=[r.PROJECT_ROOT/"scripts/run_direction1_diffusion.py"])
    manifest=json.loads((archive.directory/"code_manifest.json").read_text())
    with zipfile.ZipFile(archive.directory/"source_snapshot.zip","w",zipfile.ZIP_DEFLATED) as bundle:
        for row in manifest:bundle.write(r.PROJECT_ROOT/row["path"],row["path"])
    try:
        pd.DataFrame(gaussian_controls()).to_csv(archive.directory/"gaussian_controls.csv",index=False)
        print("Gaussian controls passed",flush=True)
        data_dir=r.PROJECT_ROOT/"runs"/config["diagnostic"]["data_run"]
        data=json.loads((data_dir/"dataset.json").read_text())
        path=r.PROJECT_ROOT/"runs"/config["diagnostic"]["train_run"]/"checkpoint/training.pt"
        checkpoint=torch.load(path,map_location=settings["device"],weights_only=True)
        assert checkpoint["dataset_sha256"]==r.hash_file(data_dir/"dataset.json")
        model=r.ConditionalScoreNet(width=settings["network"]["width"]).to(settings["device"])
        model.load_state_dict(checkpoint["best_model"]);model.eval()
        archive.write_json("dependencies.json",{"model_sha256":r.hash_file(path),"dataset_sha256":r.hash_file(data_dir/"dataset.json")})
        platform,weights,operators=r.build_stage2a_operators(config,r.PROJECT_ROOT)
        rows=[]
        for name in data["index"]["validation"][:2]:
            assert r.hash_file(data_dir/"data"/name)==data["file_sha256"][name]
            _,record=r.read_example(data_dir,name);seed,sample=map(int,record["scene_id"].split(":"))
            truth=r.record_channel(record,platform,config);scale=data["real_component_scale"]
            condition=torch.tensor(r.center_condition(record["centers_m_rad_rad"],config),device=settings["device"])[None]
            for si,schedule in enumerate(platform.schedules):
                operator=operators[schedule.name];op=r.TorchDmaOperator(operator,schedule.pilot_indices,device=settings["device"])
                variance=r.array_input_noise_variance(truth,55);nu=variance/scale**2
                obs=r.observe_multistrip_schedule(weights[schedule.name],truth[schedule.pilot_indices],pilot_energy_per_re=1.,
                    element_noise_variance=variance,rf_noise_variance=variance*config["noise"]["rf_noise_variance_fraction_of_element_noise"],
                    rng=np.random.default_rng(np.random.SeedSequence([seed,sample,si,2026090921])))
                y=torch.tensor((operator.whiten(obs.observed)/scale).reshape(operator.observation_shape),device=settings["device"],dtype=torch.complex64)[None]
                matrix=op.matrix.cpu().numpy().astype(np.complex128);inverse=np.linalg.pinv(matrix)
                ps=int(np.random.SeedSequence([settings["posterior"]["seed"],seed,sample,si,2]).generate_state(1)[0])
                for label,levels in (("ald",32),("guided_heun",64)):
                    posterior={**settings["posterior"],"sampler":label,"levels":levels}
                    samples,trace=r.posterior_samples(model,op,y,condition,nu,posterior,seed=ps)
                    estimate=samples.mean(0).cpu().numpy()*scale;error=estimate-truth
                    measured=np.einsum('ksje,kse->ksj',matrix,error[schedule.pilot_indices])
                    projected=np.einsum('ksej,ksj->kse',inverse,measured)
                    total=np.sum(abs(error)**2);visible=np.sum(abs(projected)**2)
                    row=dict(scene_id=record["scene_id"],schedule=schedule.name,sampler=label,
                        **r.channel_domain_metrics(truth,estimate,schedule.pilot_indices,np.arange(128)),
                        nullspace_error_fraction=float(1-visible/total),
                        residual_over_noise=float((op.forward(r.pack_channel(samples.mean(0)[None]))-y).abs().square().mean())/nu,
                        estimate_energy_over_truth=float(np.sum(abs(estimate)**2)/np.sum(abs(truth)**2)),
                        correlation=float(abs(np.vdot(estimate,truth))/np.linalg.norm(estimate)/np.linalg.norm(truth)))
                    rows.append(row);archive.write_metrics(rows)
                    r.atomic_json(archive.directory/"checkpoint"/f"{sample}_{si}_{label}.json",trace)
                    print(record["scene_id"],schedule.name,label,round(row["fullband_128_nmse_db"],4),flush=True)
        archive.write_status("complete",run_id=args.run_id,diagnostic_only=True)
    except BaseException as error:
        archive.write_status("failed",error=str(error));raise


if __name__=="__main__":
    main()
