# D-041正式配置扫描核验

2026-10-15：[完整报告](audit_report.md)。6运行、105264条评价、3543888条轨迹、35088个案例和510个源码成员核验通过；20个固定GPU端点复算一致。原4088个历史文件保持。

验证选定−10 dB工作点为窄带J67/宽带J56，测试分别−12.552/−11.231 dB、均98%通过。接近J192的J161/J163满足性能标准，但原0.1 dB数值敏感性超限。固定128 RE的13组均约0 dB，F3没有实现完整CSI恢复。

阈值未在测试集重选；相邻更低测试配置也通过，不能把验证工作点称为系统下限。原始运行、配置、冻结选择和自动报告保留。本次只读核验，没有新训练、选参、带噪评价或多种子。

- `verification.json`：完整性、来源、依赖和冻结检查。
- `source_inputs.json`：35681个原文件的SHA256及字节数。
- `replay.json`：20个固定GPU端点复算。
- `selected_guidance_check.json`：834项选中引导的验证目标与原始验证指标交叉核对。
- `endpoint_summary.csv`：独立按最坏数值方差汇总全部方法。
- `validation_thresholds.csv`：只从验证集计算最小J及相对J192资源变化。
- `variance_sensitivity.csv`：逐组数值敏感性与门槛翻转。
- `trajectory_summary.csv`：100场景的选定VP时刻描述，只有t=0为最终CSI。
- `paired_test_contrasts.csv`：配对效应量与10000次bootstrap区间。
- `configuration_overview.png/.pdf`：验证曲线和冻结测试明确区分，固定RE图不放大微小差异。
- `run_records.zip`：原轻量记录，不打包大案例/轨迹/频谱副本。
- `manifest.json`：本目录证据的字节校验。
