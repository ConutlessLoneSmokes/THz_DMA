# 冻结先验与标量融合正式核验

2026-10-16。三个正式阶段、5376端点/426624状态完整，20固定GPU端点和6条离线去噪复算通过。无噪声8192 RE由−13.105至−14.824 dB；20 dB却由−12.716退至−9.785 dB。4096与128 RE仍未通过−10 dB门槛。本轮为64场景冻结验证筛选，不是新独立测试或创新性证明。

- [完整报告](audit_report.md)：结果、引导、SNR、行/零空间和在线方差适用性分析。
- [完整性](verification.json)、[复算](gpu_replays.json)、[旧硬件与标定](legacy_metadata.json)、[保留核验](preservation.json)。
- [汇总](screen_summary.csv)、[配对差](paired_effects.csv)、[方差误差对比](variance_shift.csv)、[引导网格](guidance_grid.csv)。
- [主图](screen_comparison.png)、[方差对比图](variance_alignment.png)，对应PDF同名。

原始三个运行、模型、数据和自动报告保留于`runs/d041_fusion_probe_20261016_v1_*`。本目录保存派生证据、脚本和原输入SHA256；没有改原恢复算法或选择新的参数。
