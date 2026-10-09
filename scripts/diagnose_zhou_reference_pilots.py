"""D-035: predefined fixed-QPSK controls for the public Zhou LMMSE reference."""
import argparse
import json
import zipfile

import check_zhou_reference_curves as z
from _run_support import PROJECT_ROOT, create_run_archive, hash_file
import numpy as np
import pandas as pd
import torch


def author_pilots(seed, pilots, *, training_count=10000, channel_count=100):
    """Reproduce the public test script's NumPy state after training-loader pilots."""
    rng = np.random.RandomState(seed)
    for _ in range(2):
        for start in range(0, training_count, 1000):
            rng.binomial(1, .5, (min(1000, training_count-start), 64, 64))
    real = rng.binomial(1, .5, (channel_count, 64, pilots))
    imag = rng.binomial(1, .5, (channel_count, 64, pilots))
    return ((2*real[0]-1)+1j*(2*imag[0]-1))/np.sqrt(2)


def reference_lmmse(covariance, channels, p, noise, snrs):
    transmit = torch.tensor(np.fft.fft(np.eye(64), axis=0)/8, device=channels.device, dtype=channels.dtype)
    b = transmit.H@p
    a = z.real_operator(torch.kron(b.T.contiguous(), torch.eye(16, device=p.device, dtype=p.dtype))).double()
    cross = covariance.double()@a.T
    observed = a@cross
    values, vectors = torch.linalg.eigh((observed+observed.T)/2)
    values = values.clamp_min(0)
    power = float(observed.trace()/(16*p.shape[1]))
    rows = []
    for snr in snrs:
        variance = 64/10**(snr/10)
        y = (channels@b+np.sqrt(variance)*noise).transpose(-1, -2).flatten(1)
        y = torch.cat((y.real, y.imag), dim=1).double()
        estimate = ((y@vectors)/(values+variance/2))@vectors.T@cross.T
        estimate = torch.complex(estimate[:, :1024], estimate[:, 1024:]).reshape(-1, 64, 16).transpose(-1, -2)
        errors = ((estimate-channels).abs().square().sum((-1, -2))/channels.abs().square().sum((-1, -2))).cpu().numpy()
        for index, error in enumerate(errors):
            rows.append(dict(scene_id=index, snr_db=snr, nmse_linear=float(error),
                training_receive_snr_db=float(snr+10*np.log10(power/64))))
    return rows, power


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not z.rvalid(args.run_id) or not torch.cuda.is_available():
        raise ValueError('valid run ID and CUDA required')
    setup = dict(author_commit='2b922440f72ff6943c1cbe5ae85cbd50c421ac83',
        seeds=[1, 2, 3], pilot_counts=[64, 38], snr_db=list(map(float, range(-10, 31, 5))),
        training_count=100000, test_count=100, author_test_training_loader_count=10000,
        current_seed=2026100801, no_seed_selection=True)
    archive = create_run_archive('zhou_pilot_diagnostic', args.run_id, config_path=z.Path(__file__),
        config=setup, seeds=setup, script_path=__file__, environment={'torch': str(torch.__version__)},
        checkpoint_note='Predefined pilot sensitivity only; no selection based on evaluation NMSE.\n',
        extra_manifest_files=[PROJECT_ROOT/'scripts/check_zhou_reference_curves.py'])
    with zipfile.ZipFile(archive.directory/'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        for row in json.loads((archive.directory/'code_manifest.json').read_text()):
            bundle.write(PROJECT_ROOT/row['path'], row['path'])
    try:
        paths = {s: PROJECT_ROOT/'sources/datasets/zhou2025'/f'quadriga_Nt64_Nr16_ULA0.50_{s}.mat' for s in ('train', 'test')}
        reference_runs = {'author_loader': 'zhou_reference_loader_20261008_v1',
                          'paper_global': 'zhou_reference_global_20261008_v1'}
        scale_path = PROJECT_ROOT/'runs'/reference_runs['paper_global']/'normalization.json'
        scale = json.loads(scale_path.read_text())['training_global_scale']
        archive.write_json('dependencies.json', {**{s: hash_file(p) for s, p in paths.items()},
            'scale_sha256': hash_file(scale_path),
            'author_loader_sha256': hash_file(PROJECT_ROOT/'sources/code/zhou2025_diffusion_reference/loaders.py'),
            'author_test_sha256': hash_file(PROJECT_ROOT/'sources/code/zhou2025_diffusion_reference/test_diffusion_cnn.py')})
        rng = np.random.default_rng(setup['current_seed'])
        pilots, noises = {}, {}
        for count in setup['pilot_counts']:
            pilots[count] = {'current': (2*rng.integers(0, 2, (64, count))-1
                +1j*(2*rng.integers(0, 2, (64, count))-1))/np.sqrt(2)}
            for split in ('valid', 'test'):
                noise = (rng.normal(size=(100, 16, count))+1j*rng.normal(size=(100, 16, count)))/np.sqrt(2)
                if split == 'test':
                    noises[count] = torch.tensor(noise, device='cuda:0', dtype=torch.complex64)
            for seed in setup['seeds']:
                pilots[count][f'author_seed_{seed}'] = author_pilots(seed, count)
        np.savez_compressed(archive.directory/'fixed_pilots.npz',
            **{f'{count}_{label}': p for count, variants in pilots.items() for label, p in variants.items()})
        covariance = {mode: torch.zeros((2048, 2048), device='cuda:0') for mode in reference_runs}
        for start in range(0, setup['training_count'], 1024):
            h = z.read_channels(paths['train'], start, min(start+1024, setup['training_count']))
            for mode in reference_runs:
                vectors = z.angular(z.normalize(h, mode, scale)).transpose(-1, -2).flatten(1)
                vectors = torch.cat((vectors.real, vectors.imag), dim=1)
                covariance[mode] += vectors.T@vectors/setup['training_count']
        raw = z.read_channels(paths['test'], 0, 100)
        rows, spectra, checks = [], [], []
        for mode, run in reference_runs.items():
            channels = z.angular(z.normalize(raw, mode, scale))
            covariance[mode] = (covariance[mode]+covariance[mode].T)/2
            np.save(archive.directory/f'covariance_{mode}.npy', covariance[mode].cpu().numpy())
            old = pd.read_csv(PROJECT_ROOT/'runs'/run/'comparison.csv')
            for count, variants in pilots.items():
                for label, matrix in variants.items():
                    p = torch.tensor(matrix, device='cuda:0', dtype=torch.complex64)
                    metrics, power = reference_lmmse(covariance[mode], channels, p, noises[count], setup['snr_db'])
                    rows.extend(dict(normalization=mode, pilots=count, pilot_rule=label, **row) for row in metrics)
                    singular = np.linalg.svd(matrix, compute_uv=False)
                    spectra.append(dict(normalization=mode, pilots=count, pilot_rule=label,
                        min_singular_value=float(singular[-1]), max_singular_value=float(singular[0]),
                        condition_number=float(singular[0]/singular[-1]), training_receive_power=power))
                    summary = pd.DataFrame(metrics).groupby('snr_db').nmse_linear.mean()
                    if label == 'current':
                        for snr, error in summary.items():
                            previous = old[(old.pilots == count) & (old.snr_db == snr) & (old.estimator == 'lmmse')]
                            delta = 10*np.log10(error)-float(previous.iloc[0].nmse_db)
                            assert len(previous) == 1 and abs(delta) < .001, (mode, count, snr, delta)
                            checks.append(dict(normalization=mode, pilots=count, snr_db=snr, delta_db=delta))
                    archive.write_metrics(rows)
                    print(mode, count, label, {s: round(10*np.log10(summary.loc[s]), 3) for s in (0., 20., 30.)}, flush=True)
        frame = pd.DataFrame(rows)
        summary = frame.groupby(['normalization', 'pilots', 'pilot_rule', 'snr_db']).nmse_linear.mean().reset_index()
        summary['nmse_db'] = 10*np.log10(summary.nmse_linear)
        summary.to_csv(archive.directory/'comparison.csv', index=False)
        pd.DataFrame(spectra).to_csv(archive.directory/'pilot_spectra.csv', index=False)
        archive.write_json('reference_checks.json', checks)
        assert len(rows) == 2*2*4*9*100 and len(checks) == 2*2*9
        archive.write_status('complete', run_id=args.run_id, diagnostic_only=True)
    except BaseException as error:
        archive.write_status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
