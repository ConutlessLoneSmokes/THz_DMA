"""D-035: frozen denoiser-value/Jacobian controls on identical DMA observations."""
import argparse
import json
import time
import zipfile

import run_direction1_diffusion as r
import numpy as np
import pandas as pd
import torch
from thz_dma.estimators.diffusion import SubspaceScoreModel, denoised_likelihood_score, unpack_channel


class MixedDenoiser(torch.nn.Module):
    """Diagnostic only: retain one denoiser's value and the other's Jacobian."""
    def __init__(self, core, prior, *, learned_value):
        super().__init__()
        self.core, self.prior, self.learned_value = core, prior, learned_value

    def forward(self, x, sigma, centers):
        s = sigma[:, None, None, None]
        learned = self.prior.project(x-s*self.core(x, sigma, centers))
        gaussian = self.prior.denoise(x, sigma)
        clean = (gaussian+(learned-gaussian).detach() if self.learned_value
                 else learned+(gaussian-learned).detach())
        return (x-clean)/s


def channel_errors(estimate, truth):
    error = float((estimate-truth).abs().square().sum()/truth.abs().square().sum())
    return dict(nmse_linear=error, nmse_db=float(10*np.log10(max(error, 1e-30))),
                energy_ratio=float(estimate.abs().square().sum()/truth.abs().square().sum()),
                correlation=float(abs(torch.vdot(estimate.flatten(), truth.flatten()))/
                                  (torch.linalg.vector_norm(estimate)*torch.linalg.vector_norm(truth)).clamp_min(1e-30)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--moment-check', action='store_true')
    args = parser.parse_args()
    if not r.re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        raise ValueError('invalid run ID')
    source = r.PROJECT_ROOT/'runs/diffusion_observation_snr_20261008_v1'
    config = json.loads((source/'config_resolved.json').read_text())['config']
    settings = config['diffusion']
    setup = dict(data_run='diffusion_data_20261002_v1', train_run='diffusion_v4_train_20261004_v1',
                 reference_run=source.name, cases={'validation': [0, 7], 'train': [0]},
                 snr_db=[0., 15., 30.], denoising_sigmas=[.01, .1, .3, 1., 3., 10., 80.],
                 seed=2026100901, frozen_weights=True, moment_check=args.moment_check)
    if args.moment_check:
        settings['posterior']['guidance'] = 'conditional_moment_mean'
    archive = r.create_run_archive('diffusion_components', args.run_id,
        config_path=r.PROJECT_ROOT/'configs/direction1_diffusion_v4.toml',
        config={**config, 'diagnostic': setup}, seeds=setup, script_path=__file__,
        environment={'torch': str(torch.__version__)},
        checkpoint_note='Frozen validation/training controls only; mixed Jacobians are not posterior samplers.\n',
        extra_manifest_files=[r.PROJECT_ROOT/'scripts/run_direction1_diffusion.py'])
    with zipfile.ZipFile(archive.directory/'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        for row in json.loads((archive.directory/'code_manifest.json').read_text()):
            bundle.write(r.PROJECT_ROOT/row['path'], row['path'])
    try:
        data_dir, data = r.require_dataset(setup['data_run'], config)
        model_path = r.PROJECT_ROOT/'runs'/setup['train_run']/'checkpoint/training.pt'
        ck = torch.load(model_path, map_location=settings['device'], weights_only=True)
        assert ck['dataset_sha256'] == r.hash_file(data_dir/'dataset.json')
        model = r.build_score_model(settings['network']).to(settings['device']).eval()
        model.load_state_dict(ck['best_model'])
        archive.write_json('dependencies.json', {'model_sha256': r.hash_file(model_path),
            'dataset_sha256': r.hash_file(data_dir/'dataset.json'),
            'reference_metrics_sha256': r.hash_file(source/'metrics_raw.csv'),
            'calibration_sha256': r.hash_file(source/'observation_snr_calibration.json')})
        calibration = json.loads((source/'observation_snr_calibration.json').read_text())
        assert calibration['dataset_sha256'] == r.hash_file(data_dir/'dataset.json')
        reference = calibration['training_reference_power']
        old = pd.read_csv(source/'metrics_raw.csv')
        component_reference = r.PROJECT_ROOT/'runs/diffusion_components_20261009_v3/metrics_raw.csv'
        if args.moment_check:
            component_rows = pd.read_csv(component_reference)
            archive.write_json('component_dependency.json', {'sha256': r.hash_file(component_reference)})
        platform, weights, operators = r.build_stage2a_operators(config, r.PROJECT_ROOT)
        scale = data['real_component_scale']
        recovery, denoising, scores, checks = [], [], [], []
        generator = torch.Generator(device=settings['device']).manual_seed(setup['seed'])
        for split, indices in setup['cases'].items():
            for index in indices:
                name = data['index'][split][index]
                _, record = r.read_example(data_dir, name)
                truth = r.record_channel(record, platform, config)
                seed, sample = map(int, record['scene_id'].split(':'))
                clean, centers = r.load_batch(data_dir, [name], scale, config, settings['device'])
                basis = r.conditional_planar_covariance_basis(platform, config, record,
                    draws=settings['posterior']['prior_draws'], rng=np.random.default_rng(
                    np.random.SeedSequence([settings['posterior']['prior_seed'], seed, sample])))
                prior = r.LowRankChannelPrior(basis/scale, device=settings['device'],
                    relative_cutoff=settings['posterior']['covariance_relative_cutoff'])
                gaussian = SubspaceScoreModel(None, prior)
                learned = SubspaceScoreModel(model, prior)
                variants = {'gaussian': None, 'learned': model,
                    'learned_value_gaussian_jacobian': MixedDenoiser(model, prior, learned_value=True),
                    'gaussian_value_learned_jacobian': MixedDenoiser(model, prior, learned_value=False)}
                if args.moment_check:
                    variants = {'gaussian': None, 'learned_moment_conditioning': model}
                with torch.no_grad():
                    projection_floor = channel_errors(unpack_channel(prior.project(clean)), unpack_channel(clean))['nmse_linear']
                    for sigma in setup['denoising_sigmas']:
                        x = clean+sigma*torch.randn(clean.shape, device=clean.device, generator=generator)
                        level = clean.new_full((1,), sigma)
                        estimates = {'gaussian': x-sigma*gaussian(x, level, centers),
                                     'learned_raw': x-sigma*model(x, level, centers),
                                     'learned_projected': x-sigma*learned(x, level, centers)}
                        for method, estimate in estimates.items():
                            denoising.append(dict(split=split, scene_id=record['scene_id'], sigma=sigma,
                                method=method, projection_floor=projection_floor,
                                **channel_errors(unpack_channel(estimate), unpack_channel(clean))))
                pd.DataFrame(denoising).to_csv(archive.directory/'denoising.csv', index=False)
                for si, schedule in enumerate(platform.schedules):
                    operator = operators[schedule.name]
                    op = r.TorchDmaOperator(operator, schedule.pilot_indices, device=settings['device'])
                    for snr in setup['snr_db']:
                        variance = reference[schedule.name]/10**(snr/10)
                        nu = variance/scale**2
                        obs = r.observe_multistrip_schedule(weights[schedule.name], truth[schedule.pilot_indices],
                            pilot_energy_per_re=1., element_noise_variance=variance,
                            rf_noise_variance=variance*config['noise']['rf_noise_variance_fraction_of_element_noise'],
                            rng=np.random.default_rng(np.random.SeedSequence([seed, sample, si, 2026090921])))
                        y = torch.as_tensor((operator.whiten(obs.observed)/scale).reshape(operator.observation_shape),
                            dtype=torch.complex64, device=settings['device'])[None]
                        mean, _ = prior.posterior_moments(op, y, nu)
                        start_mean = prior.synthesize(mean[None].to(prior.u.dtype))
                        recovery.append(dict(split=split, scene_id=record['scene_id'], schedule=schedule.name,
                            snr_db=snr, method='truncated_lmmse', seconds=0.,
                            **channel_errors(unpack_channel(start_mean), unpack_channel(clean)),
                            residual_over_noise=float((op.forward(start_mean)-y).abs().square().mean()/nu)))
                        solve = prior.observation_solver(op, nu)
                        for sigma in (.1, 1., 10.):
                            x = start_mean+sigma*torch.randn(clean.shape, device=clean.device, generator=generator)
                            parts = {}
                            for method, denoiser in [('gaussian', gaussian), ('learned', learned)]:
                                total = denoised_likelihood_score(denoiser, op, x, y, centers, nu, sigma, solve)
                                with torch.no_grad():
                                    d = x-sigma*denoiser(x, x.new_full((1,), sigma), centers)
                                    prior_score = (d-x)/sigma**2
                                parts[method] = (prior_score, total-prior_score)
                            for component, position in [('prior', 0), ('guidance', 1)]:
                                g, l = parts['gaussian'][position], parts['learned'][position]
                                scores.append(dict(split=split, scene_id=record['scene_id'], schedule=schedule.name,
                                    snr_db=snr, sigma=sigma, component=component,
                                    learned_over_gaussian_norm=float(torch.linalg.vector_norm(l)/torch.linalg.vector_norm(g).clamp_min(1e-30)),
                                    cosine=float((g*l).sum()/(torch.linalg.vector_norm(g)*torch.linalg.vector_norm(l)).clamp_min(1e-30))))
                        posterior_seed = int(np.random.SeedSequence([settings['posterior']['seed'], seed,
                            sample, si, config['job']['observation_snr_db'].index(snr)]).generate_state(1)[0])
                        for method, core in variants.items():
                            started = time.perf_counter()
                            samples, trace = r.posterior_samples(core, op, y, centers, nu,
                                settings['posterior'], seed=posterior_seed, gaussian_prior=prior)
                            estimate = samples.mean(0)[None]
                            metrics = channel_errors(estimate, unpack_channel(clean))
                            residual = float((op.forward(r.pack_channel(estimate))-y).abs().square().mean()/nu)
                            recovery.append(dict(split=split, scene_id=record['scene_id'], schedule=schedule.name,
                                snr_db=snr, method=method, seconds=time.perf_counter()-started,
                                **metrics, residual_over_noise=residual))
                            if split == 'validation' and method in ('gaussian', 'learned'):
                                estimator = 'gaussian_diffusion_mean' if method == 'gaussian' else 'diffusion_mean'
                                previous = old[(old.scene_id == record['scene_id']) & (old.schedule == schedule.name)
                                    & (old.observation_snr_target_db == snr) & (old.estimator == estimator)]
                                assert len(previous) == 1
                                physical = estimate[0].cpu().numpy()*scale
                                current_db = r.channel_domain_metrics(truth, physical,
                                    schedule.pilot_indices, np.arange(128))['fullband_128_nmse_db']
                                delta = current_db-float(previous.iloc[0].fullband_128_nmse_db)
                                assert abs(delta) < .002, (record['scene_id'], schedule.name, snr, method, delta)
                                checks.append(dict(scene_id=record['scene_id'], schedule=schedule.name,
                                    snr_db=snr, method=method, previous_delta_db=delta))
                            if method == 'learned_moment_conditioning':
                                previous = component_rows[(component_rows.scene_id == record['scene_id'])
                                    & (component_rows.schedule == schedule.name) & (component_rows.snr_db == snr)
                                    & (component_rows.method == 'learned_value_gaussian_jacobian')]
                                assert len(previous) == 1
                                delta = metrics['nmse_db']-float(previous.iloc[0].nmse_db)
                                assert abs(delta) < .05, (record['scene_id'], schedule.name, snr, delta)
                                checks.append(dict(scene_id=record['scene_id'], schedule=schedule.name,
                                    snr_db=snr, method=method, previous_delta_db=delta))
                            archive.write_metrics(recovery)
                            pd.DataFrame(scores).to_csv(archive.directory/'scores.csv', index=False)
                            archive.write_json('reference_checks.json', checks)
                            print(split, sample, schedule.name, snr, method, round(metrics['nmse_db'], 3), flush=True)
                del prior, basis
        frame = pd.DataFrame(recovery)
        summary = frame.groupby(['split', 'schedule', 'snr_db', 'method']).nmse_linear.mean().reset_index()
        summary['nmse_db'] = 10*np.log10(summary.nmse_linear)
        summary.to_csv(archive.directory/'comparison.csv', index=False)
        assert len(recovery) == 3*2*3*(1+len(variants))
        assert len(checks) == (2*2*3+3*2*3 if args.moment_check else 2*2*3*2)
        archive.write_status('complete', run_id=args.run_id, diagnostic_only=True)
    except BaseException as error:
        archive.write_status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
