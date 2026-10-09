"""Save compact, hash-verified experiment evidence for a Git progress snapshot."""
import argparse
import json
from pathlib import Path
import re
import shutil
from _run_support import hash_file as sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', args.snapshot):
        raise ValueError('snapshot must be YYYY-MM-DD')
    root = Path(__file__).resolve().parents[1]
    output = root/'research_route/experiment_results'/args.snapshot
    if (output/'manifest.json').exists():
        raise FileExistsError('completed snapshot exists; choose a new snapshot date')
    output.mkdir(parents=True, exist_ok=True)
    copied, omitted, runs = [], [], []
    for directory in sorted((root/'runs').iterdir()):
        date = re.search(r'2026\d{4}', directory.name)
        if not directory.is_dir() or not date or date[0] < '20260926':
            continue
        status = directory/'status.json'
        if not status.is_file():
            continue
        state = json.loads(status.read_text(encoding='utf-8'))['status']
        selected = [path for path in directory.iterdir() if path.is_file()
            and (path.suffix in ('.json', '.csv')
                 or path.name in ('environment.txt', 'source_snapshot.zip'))]
        analysis = directory/'analysis'
        if analysis.exists():
            selected += [path for path in analysis.iterdir() if path.is_file()
                and path.suffix in ('.csv', '.json', '.md', '.png', '.pdf', '.py')]
        count = 0
        for path in sorted(selected):
            relative = path.relative_to(root/'runs')
            if path.stat().st_size > 5*1024*1024:
                omitted.append(dict(source=str(path.relative_to(root)).replace('\\', '/'),
                    bytes=path.stat().st_size, sha256=sha256(path), reason='over 5 MiB; local original retained'))
                continue
            target = output/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = sha256(path)
            if target.exists():
                assert digest == sha256(target), ('partial snapshot differs', relative)
            else:
                shutil.copy2(path, target)
            assert digest == sha256(target), relative
            copied.append(dict(source=str(path.relative_to(root)).replace('\\', '/'),
                path=str(relative).replace('\\', '/'), bytes=path.stat().st_size, sha256=digest))
            count += 1
        assert count > 0
        runs.append(dict(run_id=directory.name, status=state, files=count))
    assert any(item['run_id'] == 'diffusion_v5_observation_snr_20261009_v1' for item in runs)
    assert any(item['run_id'] == 'diffusion_prior_diagnostic_20261009_v1' for item in runs)
    manifest = dict(snapshot=args.snapshot, scope='runs dated 2026-09-26 onward; statuses preserved',
        excludes='channel caches, model weights, paper datasets/full text and per-scene checkpoints',
        copied_files=copied, oversized_local_files=omitted, runs=runs)
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    table = '\n'.join(f"| [{item['run_id']}]({item['run_id']}/status.json) | {item['status']} | {item['files']} |" for item in runs)
    (output/'README.md').write_text(f'''# 实验与测试进度快照：{args.snapshot}

覆盖上次提交后 2026-09-26 起的本地运行。保存原始状态、配置、来源哈希、可用原始 CSV、汇总、核验和图表；失败、中断及仅作连通性检查的运行均保留其状态，不能当作性能通过。逐文件 SHA256 与未复制大文件信息见 [manifest.json](manifest.json)。

数据缓存、模型权重、第三方数据/论文全文及逐场景检查点仍保留在本地。单文件超过 5 MiB 时只保存其大小和来源哈希。此快照用于审阅进度与数值证据；完整历史运行须按配置重新生成或迁移本地数据和权重，不能声称仅克隆仓库即可恢复全部历史运行。

## 当前结果

- [v5 完整核验](diffusion_v5_observation_snr_20261009_v1/analysis/audit_report.md)：1728 行、16 验证场景、18 个协议/SNR 组。v5 全部组改善，但仍落后条件 LMMSE；30 dB 的 F1/F2 为 −15.285/−27.023 dB，LMMSE 为 −29.246/−40.289 dB。
- [新先验诊断](diffusion_prior_diagnostic_20261009_v1/selfchecks.json)：144 条去噪、96 条局部更新、240 次代数检查；选定案例仅作诊断。[原因及下一方案](../../19e_方向一学习先验差距与下一改进方案.md)。
- [测试记录](tests.txt)：本次完整轻量检查 19/19 通过。新诊断自检独立列在对应运行中。
- 论文参考高 SNR 仍未逐点对齐；DMA 独立测试集未使用。学习收益与新残差方案尚未成立。

## 运行目录

| 运行 | 原始状态 | 复制文件数 |
|---|---|---:|
{table}
''', encoding='utf-8')
    assert all(sha256(output/item['path']) == item['sha256'] for item in copied)
    print(json.dumps(dict(snapshot=str(output), runs=len(runs), files=len(copied),
        bytes=sum(item['bytes'] for item in copied), oversized_omitted=len(omitted)), ensure_ascii=False))


if __name__ == '__main__':
    main()
