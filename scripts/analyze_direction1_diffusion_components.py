"""Summarize D-035 frozen controls and the v5 moment-guidance check."""
import json
import hashlib
import shutil
import zipfile
from pathlib import Path

from _run_support import PROJECT_ROOT, hash_file
import numpy as np
import pandas as pd


def main():
    runs = {label: PROJECT_ROOT/'runs'/name for label, name in {
        'components': 'diffusion_components_20261009_v3',
        'moments': 'diffusion_moment_check_20261009_v1',
        'pilots': 'zhou_pilot_diagnostic_20261009_v1'}.items()}
    for path in runs.values():
        assert json.loads((path/'status.json').read_text())['status'] == 'complete'
    output = runs['moments']/'analysis'
    output.mkdir(exist_ok=True)
    controls = pd.read_csv(runs['components']/'metrics_raw.csv')
    moments = pd.read_csv(runs['moments']/'metrics_raw.csv')
    pilots = pd.read_csv(runs['pilots']/'comparison.csv')
    assert len(controls) == 90 and len(moments) == 54 and len(pilots) == 144
    for frame in (controls, moments):
        assert not frame.duplicated(['split', 'scene_id', 'schedule', 'snr_db', 'method']).any()
        assert np.isfinite(frame.nmse_linear).all()
    learned = controls[controls.method == 'learned']
    candidate = moments[moments.method == 'learned_moment_conditioning']
    paired = learned.merge(candidate, on=['split', 'scene_id', 'schedule', 'snr_db'], suffixes=('_v4', '_v5'))
    assert len(paired) == 18
    paired['v5_minus_v4_db'] = paired.nmse_db_v5-paired.nmse_db_v4
    paired.to_csv(output/'paired_recovery.csv', index=False)
    summary = pd.concat([controls, candidate]).groupby(['split', 'schedule', 'snr_db', 'method']).nmse_linear.mean().reset_index()
    summary['nmse_db'] = 10*np.log10(summary.nmse_linear)
    summary.to_csv(output/'recovery_summary.csv', index=False)
    denoising = pd.read_csv(runs['components']/'denoising.csv')
    dsummary = denoising.groupby(['split', 'sigma', 'method']).nmse_linear.mean().reset_index()
    dsummary['nmse_db'] = 10*np.log10(dsummary.nmse_linear)
    dsummary.to_csv(output/'denoising_summary.csv', index=False)
    check_rows = json.loads((runs['moments']/'reference_checks.json').read_text())
    component_checks = json.loads((runs['components']/'reference_checks.json').read_text())
    pilot_checks = json.loads((runs['pilots']/'reference_checks.json').read_text())
    assert len(check_rows) == 30 and len(component_checks) == 24 and len(pilot_checks) == 36
    model_hashes = {json.loads((path/'dependencies.json').read_text())['model_sha256'] for path in
                    (runs['components'], runs['moments'])}
    assert len(model_hashes) == 1
    checks = dict(component_rows=90, moment_rows=54, pilot_summary_rows=144,
        denoising_rows=len(denoising), score_rows=len(pd.read_csv(runs['components']/'scores.csv')),
        v4_reference_max_difference_db=max(abs(row['previous_delta_db']) for row in component_checks),
        moment_equivalence_max_difference_db=max(abs(row['previous_delta_db']) for row in check_rows
            if row['method'] == 'learned_moment_conditioning'),
        pilot_reference_max_difference_db=max(abs(row['delta_db']) for row in pilot_checks),
        model_sha256=next(iter(model_hashes)),
        inputs={label: hash_file(path/'metrics_raw.csv') for label, path in runs.items()})
    integration = PROJECT_ROOT/'runs/diffusion_v5_integration_20261009_v1'
    assert json.loads((integration/'status.json').read_text())['status'] == 'complete'
    integrated = pd.read_csv(integration/'metrics_raw.csv')
    assert len(integrated) == 24 and np.isfinite(integrated.fullband_128_nmse_linear).all()
    assert integrated[integrated.estimator == 'sbl_jointfreq_3d'].algorithm_converged.eq(1).all()
    original = pd.read_csv(PROJECT_ROOT/'runs/diffusion_observation_snr_20261008_v1/metrics_raw.csv')
    baselines = integrated[integrated.estimator != 'diffusion_mean'].merge(original,
        on=['scene_id', 'schedule', 'observation_snr_target_db', 'estimator'], suffixes=('_new', '_old'))
    assert len(baselines) == 20
    delta = baselines.fullband_128_nmse_db_new-baselines.fullband_128_nmse_db_old
    assert abs(delta).max() < .001
    integration_model = json.loads((integration/'dependencies.json').read_text())['model_sha256']
    assert integration_model == checks['model_sha256']
    checks.update(integration_rows=24, integration_sbl_converged=4,
        integration_baseline_max_difference_db=float(abs(delta).max()),
        integration_metrics_sha256=hash_file(integration/'metrics_raw.csv'))
    snapshots = {}
    for label, path in {**runs, 'integration': integration}.items():
        manifest = json.loads((path/'code_manifest.json').read_text())
        with zipfile.ZipFile(path/'source_snapshot.zip') as bundle:
            for row in manifest:
                assert hashlib.sha256(bundle.read(row['path'].replace('\\', '/'))).hexdigest() == row['sha256']
        snapshots[label] = len(manifest)
    checks['verified_snapshot_files'] = snapshots
    (output/'integrity.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True, constrained_layout=True)
    labels = {'learned': 'v4: network value + network Jacobian',
              'learned_moment_conditioning': 'v5: conditional moment mean',
              'gaussian_value_learned_jacobian': 'Gaussian value + network Jacobian',
              'gaussian': 'Gaussian control'}
    for ax, (schedule, group) in zip(axes, summary[summary.split == 'validation'].groupby('schedule')):
        for method, label in labels.items():
            curve = group[group.method == method]
            ax.plot(curve.snr_db, curve.nmse_db, 'o-', label=label)
        ax.set(title='F1' if 'single' in schedule else 'F2', xlabel='Training-reference received SNR (dB)',
               xticks=[0, 15, 30], xlim=(-1, 31))
        ax.grid(alpha=.25)
    axes[0].set_ylabel('Fullband NMSE (dB; mean sample ratio)')
    axes[1].legend(fontsize=8)
    fig.suptitle('Frozen weights; selected validation cases 0 and 7; diagnostic only', fontsize=11)
    fig.savefig(output/'diffusion_guidance_controls.png', dpi=180)
    fig.savefig(output/'diffusion_guidance_controls.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    for ax, (count, group) in zip(axes, pilots[pilots.normalization == 'author_loader'].groupby('pilots', sort=False)):
        for rule, curve in group.groupby('pilot_rule'):
            ax.plot(curve.snr_db, curve.nmse_db, 'o-', label=rule)
        if count == 64:
            ax.scatter([20, 30], [-19.5, -26.], marker='x', color='black', s=70,
                       label='Paper Fig.4 visual anchors (approx.)', zorder=10)
        ax.set(title=f'Public loader, Np={count}', xlabel='Paper nominal SNR (dB)', xlim=(-10, 30))
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    axes[0].set_ylabel('LMMSE NMSE (dB; mean sample ratio)')
    fig.savefig(output/'zhou_pilot_sensitivity.png', dpi=180)
    fig.savefig(output/'zhou_pilot_sensitivity.pdf')
    plt.close(fig)
    validation = summary[(summary.split == 'validation') & summary.method.isin(['learned', 'learned_moment_conditioning', 'gaussian'])]
    table = validation.pivot(index=['schedule', 'snr_db'], columns='method', values='nmse_db')
    lines = ['# D-035 固定权重诊断与条件均值引导核验', '',
        '两个验证案例为已知尾部问题的 0、7；训练案例 0 单列。统计是线性 NMSE 均值转 dB，不代表完整验证集。', '',
        '| 协议 | SNR/dB | v4 | v5 候选 | 高斯对照 |', '|---|---:|---:|---:|---:|']
    for (schedule, snr), row in table.iterrows():
        label = 'F1' if 'single' in schedule else 'F2'
        lines.append(f"| {label} | {snr:g} | {row['learned']:.3f} | {row['learned_moment_conditioning']:.3f} | {row['gaussian']:.3f} |")
    lines.extend(['', '24 个 v4 对照项与 D-034 原始指标一致。v5 的 18 个学习项与保留网络值、替换高斯雅可比的交叉控制最大差 '
                  f"{checks['moment_equivalence_max_difference_db']:.6f} dB；这只核验实现，不证明其近似后验校准。", '',
        '预设 QPSK 种子控制不足以解释论文高 SNR 残余差距。64 导频 20 dB 公开加载器 LMMSE 为约 −17.5 至 −17.7 dB，'
        '论文 Fig.4 约 −19.5 dB；30 dB 为约 −21.7 至 −22.0，论文约 −26。图上读数仅作数量级参照。', '',
        '完整公式、边界和运行命令见 [19d 号文档](../../../research_route/19d_方向一扩散引导消融与SNR一致性核查.md)。', ''])
    lines.extend(['完整运行入口的单场景 −10/30 dB 检查完成 24 行，4 次 SBL 均达到停止线；20 条共同参照与 D-034 的最大差 '
                  f"{checks['integration_baseline_max_difference_db']:.6f} dB。", ''])
    (output/'analysis_report.md').write_text('\n'.join(lines), encoding='utf-8')
    shutil.copyfile(__file__, output/Path(__file__).name)
    print(json.dumps(checks, ensure_ascii=False), flush=True)
    print(dsummary[dsummary.sigma.isin([1., 3., 10.])].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
