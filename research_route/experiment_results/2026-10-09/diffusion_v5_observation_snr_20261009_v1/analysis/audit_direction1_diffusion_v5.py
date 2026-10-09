"""Audit the complete D-035 v5 run against frozen v4 observations and baselines."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd
import run_direction1_diffusion as r


def read_run(folder, *, current_source=False):
    assert json.loads((folder/'status.json').read_text())['status'] == 'complete'
    resolved = json.loads((folder/'config_resolved.json').read_text())
    config = resolved['config']
    manifest = json.loads((folder/'code_manifest.json').read_text())
    with zipfile.ZipFile(folder/'source_snapshot.zip') as bundle:
        for item in manifest:
            assert hashlib.sha256(bundle.read(item['path'].replace('\\', '/'))).hexdigest() == item['sha256']
            if current_source:
                assert r.hash_file(r.PROJECT_ROOT/item['path']) == item['sha256'], item['path']
    frame = pd.read_csv(folder/'metrics_raw.csv')
    keys = ['scene_id', 'schedule', 'observation_snr_target_db', 'estimator']
    methods = ['oracle_geometry_ls', 'grid_omp_3d', 'sbl_jointfreq_3d', 'lmmse',
               'diffusion_mean', 'gaussian_diffusion_mean']
    expected = {(s['name'], snr, method) for s in config['acquisition']['schedules']
                for snr in config['job']['observation_snr_db'] for method in methods}
    assert len(frame) == 1728 and frame.scene_id.nunique() == 16
    assert not frame.duplicated(keys).any()
    assert all(set(group[keys[1:]].itertuples(index=False, name=None)) == expected
               for _, group in frame.groupby('scene_id'))
    for column in ('fullband_128_nmse_linear', 'pilot_nmse_linear', 'pilot_net_rate'):
        assert np.isfinite(frame[column]).all()
    assert frame.fullband_128_nmse_subcarrier_count.eq(128).all()
    observed = frame.heldout_112_nmse_subcarrier_count.gt(0)
    assert frame.loc[observed, 'heldout_112_nmse_subcarrier_count'].eq(112).all()
    assert np.isfinite(frame.loc[observed, 'heldout_112_nmse_linear']).all()
    summary = pd.read_csv(folder/'comparison.csv')
    means = frame.groupby(keys[1:]).fullband_128_nmse_linear.mean()
    for _, row in summary.iterrows():
        assert abs(10*np.log10(means.loc[(row.schedule, row.snr_db, row.estimator)])
                   -row.mean_linear_nmse_db) < 1e-10
    files = list((folder/'checkpoint').glob('*.json'))
    saved = pd.DataFrame([row for path in files for row in json.loads(path.read_text())])
    assert len(files) == 16
    pd.testing.assert_frame_equal(frame.sort_values(keys).reset_index(drop=True),
        saved.sort_values(keys).reset_index(drop=True).replace('', np.nan),
        check_dtype=False, atol=1e-12, rtol=1e-12)
    return config, frame, len(manifest)


def main():
    new = r.PROJECT_ROOT/'runs/diffusion_v5_observation_snr_20261009_v1'
    old = r.PROJECT_ROOT/'runs/diffusion_observation_snr_20261008_v1'
    output = new/'analysis'
    output.mkdir(exist_ok=True)
    config, frame, source_count = read_run(new, current_source=True)
    previous_config, previous, old_source_count = read_run(old)
    assert config['diffusion']['posterior']['guidance'] == 'conditional_moment_mean'
    prior_settings = dict(config['diffusion']['posterior'])
    prior_settings.pop('guidance')
    assert prior_settings == previous_config['diffusion']['posterior']
    assert all(config[key] == previous_config[key] for key in config if key != 'diffusion')
    assert all(config['diffusion'][key] == previous_config['diffusion'][key]
               for key in config['diffusion'] if key != 'posterior')
    data_dir, data = r.require_dataset(config['job']['data_run_id'], config)
    dependencies = json.loads((new/'dependencies.json').read_text())
    assert dependencies == json.loads((old/'dependencies.json').read_text())
    assert dependencies['dataset_sha256'] == r.hash_file(data_dir/'dataset.json')
    model = r.PROJECT_ROOT/'runs'/config['job']['train_run_id']/'checkpoint/training.pt'
    assert dependencies['model_sha256'] == r.hash_file(model)
    calibration = json.loads((new/'observation_snr_calibration.json').read_text())
    assert calibration == json.loads((old/'observation_snr_calibration.json').read_text())
    trace_count = 0
    for _, row in frame[frame.estimator.isin(['diffusion_mean', 'gaussian_diffusion_mean'])].iterrows():
        trace = json.loads(row.sampler_trace)
        assert len(trace) == config['diffusion']['posterior']['levels']
        assert trace[-1]['sigma'] == 0 and trace[-1]['guidance'] == 'conditional_moment_mean'
        assert trace[-1]['initialization'] == 'conditional_gaussian' and trace[-1]['antithetic']
        assert not row.posterior_variance_calibrated
        trace_count += 1
    keys = ['scene_id', 'schedule', 'observation_snr_target_db', 'estimator']
    matched = frame.merge(previous, on=keys, suffixes=('_v5', '_v4'), validate='one_to_one')
    assert len(matched) == 1728
    for field in ['array_reference_snr_db', 'whitened_observation_snr_db', 'effective_output_snr_db',
                  'posterior_seed', 'covariance_rank', 'covariance_retained_energy', 'prior_draws']:
        np.testing.assert_allclose(matched[field+'_v5'], matched[field+'_v4'],
                                   rtol=0, atol=0, equal_nan=True)
    base = matched[matched.estimator.isin(['oracle_geometry_ls', 'grid_omp_3d', 'sbl_jointfreq_3d', 'lmmse'])]
    assert len(base) == 1152
    for field in ['fullband_128_nmse_linear', 'pilot_nmse_linear', 'heldout_112_nmse_linear',
                  'pilot_beam_loss_db', 'pilot_net_rate']:
        np.testing.assert_allclose(base[field+'_v5'], base[field+'_v4'], rtol=0, atol=0, equal_nan=True)
    gaussian = matched[matched.estimator == 'gaussian_diffusion_mean']
    gaussian_gap = float(abs(gaussian.fullband_128_nmse_db_v5-gaussian.fullband_128_nmse_db_v4).max())
    assert gaussian_gap < .001
    assert frame[frame.estimator == 'sbl_jointfreq_3d'].algorithm_converged.eq(1).all()
    checks = dict(rows=1728, scenes=16, matched_baseline_rows=1152, sbl_converged=288,
        new_snapshot_files=source_count, old_snapshot_files=old_source_count,
        dataset_files_verified=len(data['file_sha256']), traces_verified=trace_count,
        same_model_data_calibration_and_seeds=True, gaussian_max_difference_db=gaussian_gap,
        dependencies=dependencies, source_metrics_sha256=r.hash_file(new/'metrics_raw.csv'),
        previous_metrics_sha256=r.hash_file(old/'metrics_raw.csv'))
    r.atomic_json(output/'integrity.json', checks)
    paired = matched[matched.estimator == 'diffusion_mean'].copy()
    paired['v5_minus_v4_db'] = paired.fullband_128_nmse_db_v5-paired.fullband_128_nmse_db_v4
    paired.to_csv(output/'paired_v5_v4.csv', index=False)
    comparisons = []
    rng = np.random.default_rng(2026100913)
    for (schedule, snr), group in frame.groupby(['schedule', 'observation_snr_target_db']):
        pivot = group.pivot(index='scene_id', columns='estimator', values='fullband_128_nmse_db')
        old_nn = previous[(previous.schedule == schedule) & (previous.observation_snr_target_db == snr)
                          & (previous.estimator == 'diffusion_mean')].set_index('scene_id').fullband_128_nmse_db
        for reference, values in [('v4', old_nn.reindex(pivot.index)),
                                  ('lmmse', pivot.lmmse), ('gaussian', pivot.gaussian_diffusion_mean)]:
            delta = (pivot.diffusion_mean-values).to_numpy()
            ci = r.bootstrap_interval(delta, np.mean, rng=rng, resamples=10000)
            comparisons.append(dict(schedule=schedule, snr_db=snr, reference=reference,
                paired_mean_delta_db=float(delta.mean()), ci95_low=ci[0], ci95_high=ci[1],
                improved_cases=int((delta < 0).sum()), worse_cases=int((delta > 0).sum())))
    pd.DataFrame(comparisons).to_csv(output/'paired_comparisons.csv', index=False)
    nn = frame[frame.estimator == 'diffusion_mean']
    tails = []
    for (scene, schedule), group in nn.groupby(['scene_id', 'schedule']):
        group = group.sort_values('observation_snr_target_db')
        for first, second in zip(group.iloc[:-1].itertuples(), group.iloc[1:].itertuples()):
            tails.append(dict(scene_id=scene, schedule=schedule,
                from_snr=first.observation_snr_target_db, to_snr=second.observation_snr_target_db,
                from_nmse_db=first.fullband_128_nmse_db, to_nmse_db=second.fullband_128_nmse_db,
                change_db=second.fullband_128_nmse_db-first.fullband_128_nmse_db))
    pd.DataFrame(tails).to_csv(output/'per_scene_snr_steps.csv', index=False)
    curves = frame.groupby(['schedule', 'observation_snr_target_db', 'estimator']).fullband_128_nmse_linear.mean().reset_index()
    curves['nmse_db'] = 10*np.log10(curves.fullband_128_nmse_linear)
    curves.to_csv(output/'curves.csv', index=False)
    make_report(output, frame, previous, curves, pd.DataFrame(comparisons), pd.DataFrame(tails), checks)
    shutil.copyfile(__file__, output/Path(__file__).name)
    print(json.dumps(checks, ensure_ascii=False), flush=True)
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)


def make_report(output, frame, previous, curves, comparisons, tails, checks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, constrained_layout=True)
    old_curve = previous[previous.estimator == 'diffusion_mean'].groupby(
        ['schedule', 'observation_snr_target_db']).fullband_128_nmse_linear.mean()
    names = {'diffusion_mean': 'Diffusion v5', 'gaussian_diffusion_mean': 'Gaussian control',
             'lmmse': 'LMMSE', 'sbl_jointfreq_3d': 'SBL', 'grid_omp_3d': 'OMP',
             'oracle_geometry_ls': 'Oracle geometry + LS'}
    lines = ['# D-035 v5 完整验证核验', '',
        '1728 行、16 验证场景、F1/F2、−10:5:30 dB、6 方法完整；无缺组或重复。'
        '模型、数据、SNR 标定、观测与后验种子一致；1152 条传统基线逐项相同。', '',
        '| 协议 | SNR/dB | v4 | v5 | LMMSE | 高斯控制 |', '|---|---:|---:|---:|---:|---:|']
    for ax, (schedule, group) in zip(axes, curves.groupby('schedule')):
        label = 'F1' if 'single' in schedule else 'F2'
        baseline = old_curve.loc[schedule]
        ax.plot(baseline.index, 10*np.log10(baseline), 'o--', label='Diffusion v4')
        for method, title in names.items():
            curve = group[group.estimator == method]
            ax.plot(curve.observation_snr_target_db, curve.nmse_db, 'o-', label=title)
        ax.set(title=label, xlabel='Training-reference whitened received SNR (dB)', xlim=(-11, 31))
        ax.grid(alpha=.25)
        pivot = group.pivot(index='observation_snr_target_db', columns='estimator', values='nmse_db')
        assert np.all(np.diff(pivot.diffusion_mean.to_numpy()) < 0)
        for snr in (0., 15., 20., 30.):
            row = pivot.loc[snr]
            lines.append(f"| {label} | {snr:g} | {10*np.log10(baseline.loc[snr]):.3f} | {row.diffusion_mean:.3f} | {row.lmmse:.3f} | {row.gaussian_diffusion_mean:.3f} |")
    axes[0].set_ylabel('Fullband NMSE (dB; mean sample ratio)')
    axes[1].legend(fontsize=8)
    fig.suptitle('16 validation scenes; fixed v4 weights; matched data, observations and seeds', fontsize=11)
    fig.savefig(output/'v4_v5_baselines.png', dpi=180)
    fig.savefig(output/'v4_v5_baselines.pdf')
    plt.close(fig)
    old_delta = comparisons[comparisons.reference == 'v4']
    lmmse_delta = comparisons[comparisons.reference == 'lmmse']
    regression = tails[(tails.from_snr >= 15) & (tails.change_db > 0)]
    lines.extend(['', f"v5 相对 v4 的配对 95% CI 在 {int(old_delta.ci95_high.lt(0).sum())}/18 组完全小于零；"
        f"相对 LMMSE 的配对 CI 在 {int(lmmse_delta.ci95_low.gt(0).sum())}/18 组完全大于零。区间重采样 16 个验证场景，不能解释为独立测试证据。", '',
        '两个协议的组均值均随 SNR 单调下降，但个别案例仍会回升：'
        f"15→20、20→25、25→30 dB 的 96 个相邻步骤中，{len(regression)} 个回升，最大 {regression.change_db.max():.3f} dB。", '',
        f"288 次 SBL 全部达到停止线。高斯控制与 v4 最大差 {checks['gaussian_max_difference_db']:.6f} dB。", '',
        'v5 的引导修正在完整验证中有效；去噪先验与条件矩近似仍未达到条件 LMMSE。'
        '论文参考块本轮未重跑，原高 SNR 数值差距保持待核对状态。独立测试未使用。', '',
        '配对差值为逐场景 dB 差的均值，表中 NMSE 为逐场景线性比值均值转 dB；两者不可混用。', ''])
    (output/'audit_report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
