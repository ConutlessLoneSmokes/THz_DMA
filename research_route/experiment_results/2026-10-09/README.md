# 实验与测试进度快照：2026-10-09

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
| [diffusion_components_20261009_v1](diffusion_components_20261009_v1/status.json) | failed | 9 |
| [diffusion_components_20261009_v2](diffusion_components_20261009_v2/status.json) | failed | 11 |
| [diffusion_components_20261009_v3](diffusion_components_20261009_v3/status.json) | complete | 12 |
| [diffusion_data_20261002_v1](diffusion_data_20261002_v1/status.json) | complete | 7 |
| [diffusion_data_smoke_20261002_v1](diffusion_data_smoke_20261002_v1/status.json) | complete | 7 |
| [diffusion_diagnosis_20261004_v1](diffusion_diagnosis_20261004_v1/status.json) | complete | 10 |
| [diffusion_eval_smoke_20261002_v1](diffusion_eval_smoke_20261002_v1/status.json) | complete | 10 |
| [diffusion_initialization_20261004_v1](diffusion_initialization_20261004_v1/status.json) | complete | 10 |
| [diffusion_moment_check_20261009_v1](diffusion_moment_check_20261009_v1/status.json) | complete | 23 |
| [diffusion_observation_snr_20261008_v1](diffusion_observation_snr_20261008_v1/status.json) | complete | 25 |
| [diffusion_observation_snr_endpoint_20261008_v1](diffusion_observation_snr_endpoint_20261008_v1/status.json) | complete | 14 |
| [diffusion_observation_snr_probe_20261008_v1](diffusion_observation_snr_probe_20261008_v1/status.json) | complete | 13 |
| [diffusion_prior_diagnostic_20261009_v1](diffusion_prior_diagnostic_20261009_v1/status.json) | complete | 10 |
| [diffusion_train_20261002_v1](diffusion_train_20261002_v1/status.json) | complete | 8 |
| [diffusion_train_smoke_20261002_v1](diffusion_train_smoke_20261002_v1/status.json) | complete | 8 |
| [diffusion_v2_train_probe_20261004_v1](diffusion_v2_train_probe_20261004_v1/status.json) | complete | 8 |
| [diffusion_v2_val_probe_20261004_v1](diffusion_v2_val_probe_20261004_v1/status.json) | complete | 10 |
| [diffusion_v3_val_probe_20261004_v1](diffusion_v3_val_probe_20261004_v1/status.json) | complete | 10 |
| [diffusion_v4_train_20261004_v1](diffusion_v4_train_20261004_v1/status.json) | complete | 8 |
| [diffusion_v4_val_20261004_v1](diffusion_v4_val_20261004_v1/status.json) | complete | 21 |
| [diffusion_v4_val_probe_20261004_v1](diffusion_v4_val_probe_20261004_v1/status.json) | complete | 10 |
| [diffusion_v5_integration_20261009_v1](diffusion_v5_integration_20261009_v1/status.json) | complete | 12 |
| [diffusion_v5_observation_snr_20261009_v1](diffusion_v5_observation_snr_20261009_v1/status.json) | complete | 21 |
| [diffusion_val_20261002_v1](diffusion_val_20261002_v1/status.json) | complete | 16 |
| [stage2a_linear_20260929_v1](stage2a_linear_20260929_v1/status.json) | complete | 11 |
| [stage2a_linear_smoke_20260929_v1](stage2a_linear_smoke_20260929_v1/status.json) | complete | 11 |
| [stage2a_p016_config_curve_20260926_v1](stage2a_p016_config_curve_20260926_v1/status.json) | complete | 15 |
| [stage2a_sbl_checkpoint_20261002_v1](stage2a_sbl_checkpoint_20261002_v1/status.json) | complete | 16 |
| [stage2a_sbl_smoke_20261001_v1](stage2a_sbl_smoke_20261001_v1/status.json) | complete | 16 |
| [stage2a_sbl_smoke_20261001_v2](stage2a_sbl_smoke_20261001_v2/status.json) | complete | 16 |
| [stage2a_sbl_snr_20261001_v1](stage2a_sbl_snr_20261001_v1/status.json) | interrupted | 5 |
| [stage2a_sbl_snr_20261002_v1](stage2a_sbl_snr_20261002_v1/status.json) | complete | 20 |
| [stage2a_somp_snr_20260929_v1](stage2a_somp_snr_20260929_v1/status.json) | complete | 15 |
| [zhou_pilot_diagnostic_20261009_v1](zhou_pilot_diagnostic_20261009_v1/status.json) | complete | 11 |
| [zhou_reference_global_20261008_v1](zhou_reference_global_20261008_v1/status.json) | complete | 11 |
| [zhou_reference_global_smoke_20261008_v1](zhou_reference_global_smoke_20261008_v1/status.json) | complete | 11 |
| [zhou_reference_loader_20261008_v1](zhou_reference_loader_20261008_v1/status.json) | complete | 11 |
| [zhou_reference_real_smoke_20261008_v1](zhou_reference_real_smoke_20261008_v1/status.json) | complete | 11 |
| [zhou_reference_smoke_20261008_v1](zhou_reference_smoke_20261008_v1/status.json) | failed | 1 |
| [zhou_reference_smoke_20261008_v2](zhou_reference_smoke_20261008_v2/status.json) | complete | 11 |
