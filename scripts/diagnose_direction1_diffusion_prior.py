"""D-036: frozen prior quality, center dependence, and local error transmission."""
import argparse
import json
import zipfile

import numpy as np
import pandas as pd
import torch
import run_direction1_diffusion as r
from thz_dma.estimators.diffusion import unpack_channel


def energy(x):
    return float(x.square().sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not r.re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        raise ValueError('invalid run ID')
    source = r.PROJECT_ROOT/'runs/diffusion_v5_observation_snr_20261009_v1'
    assert json.loads((source/'status.json').read_text())['status'] == 'complete'
    config = json.loads((source/'config_resolved.json').read_text())['config']
    settings = config['diffusion']
    setup = dict(cases={'train': [0], 'validation': [0, 7, 14]},
        sigmas=[.1, .3, 1., 3., 10., 80.], snr_db=[0., 30.], seed=2026100903,
        frozen_weights=True, natural_noisy_csi_only=True,
        wrong_center_rule='next case in the same split; keep x and projection fixed',
        local_fusion_is_not_an_estimator_benchmark=True)
    archive = r.create_run_archive('diffusion_prior', args.run_id,
        config_path=r.PROJECT_ROOT/'configs/direction1_diffusion_v5.toml',
        config={**config, 'diagnostic': setup}, seeds=setup, script_path=__file__,
        environment={'torch': str(torch.__version__)},
        checkpoint_note='Frozen prior probes; full noisy CSI is diagnostic input, not a DMA observation.\n',
        extra_manifest_files=[r.PROJECT_ROOT/'scripts/run_direction1_diffusion.py'])
    with zipfile.ZipFile(archive.directory/'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        for row in json.loads((archive.directory/'code_manifest.json').read_text()):
            bundle.write(r.PROJECT_ROOT/row['path'], row['path'])
    try:
        directory, data = r.require_dataset(config['job']['data_run_id'], config)
        model_path = r.PROJECT_ROOT/'runs'/config['job']['train_run_id']/'checkpoint/training.pt'
        original_hash = r.hash_file(model_path)
        checkpoint = torch.load(model_path, map_location=settings['device'], weights_only=True)
        assert checkpoint['dataset_sha256'] == r.hash_file(directory/'dataset.json')
        dependencies = json.loads((source/'dependencies.json').read_text())
        assert dependencies['model_sha256'] == original_hash
        model = r.build_score_model(settings['network']).to(settings['device']).eval()
        model.load_state_dict(checkpoint['best_model'])
        calibration_path = source/'observation_snr_calibration.json'
        calibration = json.loads(calibration_path.read_text())
        assert calibration['dataset_sha256'] == r.hash_file(directory/'dataset.json')
        archive.write_json('dependencies.json', {**dependencies,
            'reference_metrics_sha256': r.hash_file(source/'metrics_raw.csv'),
            'calibration_sha256': r.hash_file(calibration_path)})
        platform, weights, operators = r.build_stage2a_operators(config, r.PROJECT_ROOT)
        generator = torch.Generator(device=settings['device']).manual_seed(setup['seed'])
        scale = data['real_component_scale']
        denoising, transmission, checks = [], [], []
        with torch.no_grad():
            for split, indices in setup['cases'].items():
                for index in indices:
                    name = data['index'][split][index]
                    _, record = r.read_example(directory, name)
                    clean, centers = r.load_batch(directory, [name], scale, config, settings['device'])
                    _, other = r.read_example(directory, data['index'][split][index+1])
                    wrong = torch.as_tensor(r.center_condition(other['centers_m_rad_rad'], config),
                        device=clean.device)[None]
                    seed, sample = map(int, record['scene_id'].split(':'))
                    basis = r.conditional_planar_covariance_basis(platform, config, record,
                        draws=settings['posterior']['prior_draws'], rng=np.random.default_rng(
                        np.random.SeedSequence([settings['posterior']['prior_seed'], seed, sample])))
                    prior = r.LowRankChannelPrior(basis/scale, device=settings['device'],
                        relative_cutoff=settings['posterior']['covariance_relative_cutoff'])
                    truth = r.record_channel(record, platform, config)
                    contexts = []
                    for si, schedule in enumerate(platform.schedules):
                        op = r.TorchDmaOperator(operators[schedule.name], schedule.pilot_indices,
                            device=settings['device'])
                        for snr in setup['snr_db']:
                            variance = calibration['training_reference_power'][schedule.name]/10**(snr/10)
                            nu = variance/scale**2
                            observed = r.observe_multistrip_schedule(weights[schedule.name], truth[schedule.pilot_indices],
                                pilot_energy_per_re=1., element_noise_variance=variance,
                                rf_noise_variance=variance*config['noise']['rf_noise_variance_fraction_of_element_noise'],
                                rng=np.random.default_rng(np.random.SeedSequence([seed, sample, si, 2026090921])))
                            y = torch.as_tensor((operators[schedule.name].whiten(observed.observed)/scale).reshape(
                                operators[schedule.name].observation_shape), dtype=torch.complex64, device=clean.device)[None]
                            contexts.append((schedule.name, snr, op, nu, y, prior.observation_solver(op, nu)))
                    for sigma in setup['sigmas']:
                        noise = torch.randn(clean.shape, device=clean.device, generator=generator)
                        x = clean+sigma*noise
                        level = clean.new_full((1,), sigma)
                        epsilon = model(x, level, centers)
                        raw = x-sigma*epsilon
                        projected = prior.project(raw)
                        gaussian = prior.denoise(x, level)
                        wrong_raw = x-sigma*model(x, level, wrong)
                        estimates = {'gaussian': gaussian,
                            'isotropic_projected': prior.project(x/(1+sigma*sigma)),
                            'learned_raw': raw, 'learned_projected': projected,
                            'wrong_center_raw': wrong_raw, 'wrong_center_projected': prior.project(wrong_raw)}
                        base = dict(split=split, scene_id=record['scene_id'], sigma=sigma,
                            rank=prior.rank, real_dimension=clean.numel(), retained_energy=prior.retained_energy)
                        for method, estimate in estimates.items():
                            error = estimate-clean
                            supported = prior.project(error)
                            total = energy(error)
                            discarded = energy(error-supported)
                            identity_error = abs(total-energy(supported)-discarded)/max(total, 1e-30)
                            assert identity_error < 2e-5
                            checks.append(identity_error)
                            denoising.append({**base, 'method': method, 'nmse_linear': total/energy(clean),
                                'discarded_error_fraction': discarded/max(total, 1e-30),
                                'weighted_epsilon_mse': (1+sigma*sigma)*total/(sigma*sigma*clean.numel()),
                                'center_change_relative_rms': np.sqrt(energy(estimate-projected)/max(energy(projected), 1e-30))
                                    if method == 'wrong_center_projected' else None})
                        direct_loss = float((1+sigma*sigma)*(epsilon-noise).square().mean())
                        reconstructed_loss = denoising[-4]['weighted_epsilon_mse']
                        assert np.isclose(direct_loss, reconstructed_loss, rtol=2e-4, atol=1e-8)
                        delta = projected-gaussian
                        for schedule, snr, op, nu, y, solve in contexts:
                            def correction(residual):
                                return 2*sigma*sigma*prior.denoise(op.adjoint(solve(residual, sigma)), level)
                            g_fused = gaussian+correction(y-op.forward(gaussian))
                            l_fused = projected+correction(y-op.forward(projected))
                            transmitted = delta-correction(op.forward(delta))
                            difference = l_fused-g_fused
                            identity_error = np.sqrt(energy(difference-transmitted)/max(energy(delta), 1e-30))
                            assert identity_error < 2e-3
                            checks.append(identity_error)
                            transmission.append({**base, 'schedule': schedule, 'snr_db': snr,
                                'denoiser_difference_nmse': energy(delta)/energy(clean),
                                'transmitted_difference_nmse': energy(transmitted)/energy(clean),
                                'surviving_difference_energy_fraction': energy(transmitted)/max(energy(delta), 1e-30),
                                'local_gaussian_fused_nmse': energy(g_fused-clean)/energy(clean),
                                'local_learned_fused_nmse': energy(l_fused-clean)/energy(clean),
                                'identity_relative_error': identity_error})
                    pd.DataFrame(denoising).to_csv(archive.directory/'denoising.csv', index=False)
                    pd.DataFrame(transmission).to_csv(archive.directory/'transmission.csv', index=False)
                    print(split, record['scene_id'], 'rank', prior.rank, 'complete', flush=True)
                    del prior, basis, contexts
        assert len(denoising) == 4*6*6 and len(transmission) == 4*6*2*2
        assert r.hash_file(model_path) == original_hash
        archive.write_json('selfchecks.json', dict(denoising_rows=len(denoising),
            transmission_rows=len(transmission), algebra_checks=len(checks),
            max_identity_relative_error=max(checks), training_model_unchanged=True,
            dataset_caches_verified=len(data['file_sha256'])))
        archive.write_status('complete', run_id=args.run_id, diagnostic_only=True)
    except BaseException as error:
        archive.write_status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
