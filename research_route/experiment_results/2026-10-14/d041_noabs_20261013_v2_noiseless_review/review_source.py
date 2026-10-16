"""Read-only replay of the completed D-041 paired noiseless experiment.

Writes a separate review folder; leaves all runs and their original audit intact.
No training, tuning, or driver changes are performed.
"""
import ctypes
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import warnings
import zipfile

ROOT = Path('/home/lsy/THz_DMA')
PREFIX = 'd041_noabs_20261013_v2'
OUT = ROOT / 'research_route/experiment_results/2026-10-14' / (PREFIX + '_noiseless_review')
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import pandas as pd
import torch
from _run_support import hash_file, write_json
import analyze_d041_diagnostic as analyzer
from run_d041_diagnostic import operator_groups, posterior_seed, sample_observation, training_reference
from run_zhou2025_repro import load_data, load_model, read_json
from zhou2025_repro.d041 import paired_array, paired_channel, diagnostics
from zhou2025_repro.data import decode
from zhou2025_repro.ddpm import recover

assert not (OUT / 'verification.json').exists(), 'completed review outputs are immutable'
torch.set_num_threads(4)
run_dirs = sorted((ROOT / 'runs').glob(PREFIX + '_*'))
assert len(run_dirs) == 14
before = {}
source_entries = 0
for directory in run_dirs:
    assert read_json(directory / 'status.json')['status'] == 'complete'
    config = read_json(directory / 'config_resolved.json')['config']
    assert config['device'] == 'cuda:0' and not config['job']['smoke']
    entries = read_json(directory / 'code_manifest.json')
    with zipfile.ZipFile(directory / 'source_snapshot.zip') as archive:
        assert set(archive.namelist()) == {r['path'] for r in entries}
        for record in entries:
            content = archive.read(record['path'])
            assert len(content) == record['size_bytes']
            assert hashlib.sha256(content).hexdigest() == record['sha256']
            assert hash_file(ROOT / record['path']) == record['sha256']
            source_entries += 1
    for file in directory.rglob('*'):
        if file.is_file():
            before[str(file.relative_to(ROOT))] = {'size_bytes': file.stat().st_size, 'sha256': hash_file(file)}
original_audit = ROOT / 'results' / (PREFIX + '_noiseless')
for file in original_audit.iterdir():
    if file.is_file():
        before[str(file.relative_to(ROOT))] = {'size_bytes': file.stat().st_size, 'sha256': hash_file(file)}

# Only the analyzer's output location changes. Its loaders still read the real runs.
OUT.mkdir(exist_ok=True)
analyzer.PROJECT_ROOT = OUT / 'reanalyzed'
repeat = OUT / 'reanalyzed/results' / (PREFIX + '_noiseless')
if not repeat.exists():
    analyzer.audit(PREFIX, 'noiseless')
rechecked = read_json(repeat / 'audit.json')
assert rechecked == read_json(original_audit / 'audit.json')
for name in ['README.md', 'audit.json', 'noiseless_controls.csv', 'group_summary.csv', 'run_inventory.csv', 'analysis_source.py']:
    assert hash_file(repeat / name) == hash_file(original_audit / name), name

data = load_data(PREFIX + '_wb_data')
config = read_json(data.directory / 'config_resolved.json')['config']
array = paired_array(config)
physics_replay = []
for split in ['train', 'validation', 'test']:
    for index in [0, data.counts[split]-1]:
        channel, record = paired_channel(config, array, data.metadata['split_seeds'][split], index)
        assert record == data.record(split, index)
        saved = data.physical(split, [index])[0]
        np.testing.assert_array_equal(saved, channel.astype(saved.dtype))
        physics_replay.append({'split': split, 'index': index, 'storage_dtype': str(saved.dtype),
                               'exact_path_and_stored_channel_match': True})

lib = ctypes.CDLL('libnvidia-ml.so.1')
rc = lib.nvmlInit_v2()
lib.nvmlErrorString.restype = ctypes.c_char_p
gpu = {'python_executable': sys.executable, 'torch': str(torch.__version__),
       'cuda_build': torch.version.cuda, 'nvml_return': rc, 'nvml_error': lib.nvmlErrorString(rc).decode(),
       'nvml_library': str(Path('/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1').resolve()),
       'kernel_driver': Path('/proc/driver/nvidia/version').read_text().splitlines()[0]}
with warnings.catch_warnings(record=True) as observed_warnings:
    warnings.simplefilter('default')
    gpu['cuda_available'] = torch.cuda.is_available()
    assert gpu['cuda_available'], 'this probe must run outside the GPU-invisible sandbox'
    x = torch.arange(1024, device='cuda:0', dtype=torch.float32)
    gpu['tensor_device'] = str(x.device)
    gpu['computed_sum'] = float((x*x).sum().item())
    assert gpu['computed_sum'] == float(np.sum(np.arange(1024, dtype=np.float32)**2, dtype=np.float32))
    gpu['gpu'] = torch.cuda.get_device_name(0)
    gpu['device_count'] = torch.cuda.device_count()
    model, schedule, checkpoint, _ = load_model(PREFIX + '_wb_clean', 'cuda:0')
    assert checkpoint['step'] == 256000
    assert checkpoint['best_step'] == 252000
    assert all(torch.isfinite(value).all() for value in checkpoint['best_model'].values())
    assert len(checkpoint['history']) == 1024
    replays, references = [], []
    for stage, tag in [('full-noiseless', 'full'), ('reduced-noiseless', 'reduced')]:
        directory = ROOT / 'runs' / (PREFIX + '_wb_' + tag + '_test')
        settings = read_json(directory / 'config_resolved.json')['config']
        frame = pd.read_csv(directory / 'metrics_raw.csv')
        tuning = read_json(directory / 'tuning.json')
        reference = read_json(directory / 'observation_reference.json')['reference']
        for label, operator, _ in operator_groups(settings, stage):
            measured_reference = training_reference(data, operator, settings)
            np.testing.assert_allclose(measured_reference, reference[label]['normalized_whitened_power'], rtol=1e-12)
            references.append({'operator': label, 'training_scenes': data.counts['train'],
                               'recomputed_whitened_power': measured_reference, 'matches_saved': True})
            # Fixed scene endpoints were chosen before inspecting their errors.
            for index in [0, 99]:
                for variance in settings['diagnostic']['noiseless_likelihood_variances']:
                    condition = f'noiseless_v{variance:.0e}'
                    guidance = tuning[f'{label}:{condition}']['selected']
                    truth, clean, observed = sample_observation(data, operator, settings, label, index, 'test', variance, True)
                    with torch.no_grad():
                        image, _ = recover(model, schedule, operator, observed, variance, scale=guidance,
                            seed=posterior_seed(settings, label, index, 'test'), refinement=settings['evaluation_run']['refinement'])
                    computed = diagnostics(operator, truth, decode(image, 'dma'), observed, clean)
                    row = frame.loc[(frame.operator == label) & (frame.condition == condition)
                                    & (frame.scene_index == index) & (frame.estimator == 'clean_ddpm')].iloc[0]
                    gaps = {key: abs(float(row[key])-value) for key, value in computed.items()}
                    assert gaps['nmse_db'] <= 0.02, (label, condition, index, gaps)
                    np.testing.assert_allclose([float(row[k]) for k in computed], list(computed.values()), rtol=1e-4, atol=1e-6)
                    replays.append({'operator': label, 'condition': condition, 'scene_index': index,
                                    'frozen_guidance': guidance, 'nmse_db': computed['nmse_db'],
                                    'nmse_db_gap_from_saved': gaps['nmse_db'], 'all_metrics_match': True})
    torch.cuda.synchronize()
    gpu['probe_warning_messages'] = sorted({str(w.message) for w in observed_warnings})
for cmd, key in [(['nvidia-smi'], 'nvidia_smi'), (['modinfo', '-F', 'version', 'nvidia'], 'installed_module')]:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    gpu[key] = {'returncode': result.returncode, 'output': (result.stdout+result.stderr).strip()}

frames = [pd.read_csv(ROOT / 'runs' / (PREFIX + '_wb_' + tag + '_test') / 'metrics_raw.csv') for tag in ['full', 'reduced']]
frame = pd.concat(frames, ignore_index=True)
summary = []
sensitivity = []
for label, block in frame.loc[frame.estimator == 'clean_ddpm'].groupby('operator'):
    pivot = block.pivot(index='scene_id', columns='condition', values='nmse_db')
    spread = pivot.max(axis=1)-pivot.min(axis=1)
    absolute = block.pivot(index='scene_id', columns='condition', values='nmse_linear')
    sensitivity.append({'operator': label, 'max_spread_db': float(spread.max()),
        'scenes_above_predeclared_0p1db': int((spread > .1).sum()),
        'maximum_absolute_nmse_difference': float((absolute.max(axis=1)-absolute.min(axis=1)).max()),
        'same_recovery_pass_fail_for_every_scene': bool(((pivot <= -10).all(axis=1) | (pivot > -10).all(axis=1)).all())})
    pinv = frame.loc[(frame.operator == label) & (frame.estimator == 'pinv_ls')]
    summary.append({'operator': label, 'ddpm_nmse_db': 10*math.log10(float(block.nmse_linear.mean())),
        'pinv_nmse_db': 10*math.log10(float(pinv.nmse_linear.mean())),
        'observation_residual_relative': float(block.observation_residual_relative.mean()),
        'prediction_energy_ratio': float(block.prediction_energy_ratio.mean()),
        'normalized_correlation': float(block.normalized_correlation.mean()),
        'row_space_nmse_linear': float(block.row_space_nmse_linear.mean()),
        'null_space_nmse_linear': float(block.null_space_nmse_linear.mean()),
        'ddpm_improvement_over_pinv_db': 10*math.log10(float(pinv.nmse_linear.mean())/float(block.nmse_linear.mean()))})
pd.DataFrame(summary).to_csv(OUT / 'test_diagnostics.csv', index=False)
pd.DataFrame(sensitivity).to_csv(OUT / 'variance_sensitivity.csv', index=False)
write_json(OUT / 'gpu_probe.json', gpu)
write_json(OUT / 'replay.json', {'physics': physics_replay, 'frozen_gpu_endpoints': replays, 'snr_training_references': references})

# Compact complete run records; large channel arrays and model weights stay in runs.
excluded = {'.npy', '.pt'}
members = []
with zipfile.ZipFile(OUT / 'run_records.zip', 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
    for directory in [p for p in run_dirs if '_wb_' in p.name]:
        for file in sorted(directory.rglob('*')):
            if file.is_file() and file.suffix not in excluded:
                name = str(file.relative_to(ROOT))
                bundle.write(file, name)
                members.append(name)
write_json(OUT / 'source_inputs.json', before)
for name, record in before.items():
    file = ROOT / name
    assert file.stat().st_size == record['size_bytes'] and hash_file(file) == record['sha256'], name
write_json(OUT / 'verification.json', {
    'integrity_passed': True, 'training_updates': 256000, 'training_history_records': 1024,
    'run_count': 14, 'source_snapshot_members_verified': source_entries,
    'reanalyzed_report_byte_identical': True, 'metric_rows': rechecked['rows'],
    'trajectory_rows': rechecked['trajectory_rows'], 'physics_replays': len(physics_replay),
    'frozen_gpu_replays': len(replays), 'maximum_replay_nmse_db_gap': max(r['nmse_db_gap_from_saved'] for r in replays),
    'all_training_snr_references_recomputed': True, 'all_original_inputs_byte_unchanged': True,
    'original_files_verified': len(before), 'run_record_zip_members': len(members),
    'paired_predeclared_numerical_variance_gate_passed': False, 'wide_predeclared_numerical_variance_gate_passed': True, 'unique_causal_attribution': False,
    'new_training_or_tuning_performed': False, 'driver_or_ssh_configuration_changed_by_reviewer': False})
print(json.dumps(read_json(OUT / 'verification.json'), indent=2))
print(pd.DataFrame(summary).to_string(index=False))
print(pd.DataFrame(sensitivity).to_string(index=False))
