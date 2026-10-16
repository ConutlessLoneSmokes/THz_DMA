# D-041 四组无噪声核验

用户完成两带宽14阶段，7920条评价/266640条轨迹完整；来源、601批配对路径、冻结选参及20个固定宽带GPU端点复算通过。

宽带数字/J192约−70.52 dB，J32/J8/J1约−5.19/−0.20/−0.015 dB。充分配置两带宽均通过，少配置主要失败在零空间；宽带J32有部分补全，J8/J1仍未恢复完整CSI。宽带数值敏感性通过，原窄带超限保留，不作唯一因果归因。当前驱动/NVML均595.99.02，GPU和NVML正常。

- [完整检查报告](audit_report.md)
- [完整性](verification.json)、[固定端点复算](replay.json)、[GPU检查](gpu_probe.json)
- [独立四组核验](reanalyzed/results/d041_noabs_20261013_v2_noiseless/README.md)、[配对控制](paired_test_summary.csv)、[宽带诊断](test_diagnostics.csv)、[数值敏感性](variance_sensitivity.csv)、[恢复轨迹](trajectory_summary.csv)
- [原输入清单](source_inputs.json)、[新增宽带2037个记录](run_records.zip)、[核验源码](review_source.py)

原数据、模型、窄带档案和四组报告保留；没有重训、调参或启动带噪/等资源/多种子评价。
