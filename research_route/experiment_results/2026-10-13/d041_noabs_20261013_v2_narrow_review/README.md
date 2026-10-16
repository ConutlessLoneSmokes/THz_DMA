# D-041 窄带正式结果核验

用户完成 `d041_noabs_20261013_v2` 窄带无噪声7阶段：256000次训练更新，3960条评价、133320条轨迹。来源、冻结选参、指标、数据及20个GPU恢复端点复核通过。

数字/J192约−66.87 dB，J32/J8/J1为−4.05/−0.13/+0.09 dB，少配置的误差主要在零空间。充分配置部分场景未通过原数值方差0.1 dB门槛，判定未改变；保留限制，不作唯一归因。NVML返回18，确认为595.91.07加载驱动与595.99.02库不匹配，CUDA计算仍通过。

- [完整检查报告](audit_report.md)
- [完整性核验](verification.json)、[冻结端点复算](replay.json)、[GPU/NVML诊断](gpu_probe.json)
- [独立自动报告](reanalyzed/results/d041_noabs_20261013_v2_narrow_noiseless/README.md)、[最终指标](test_diagnostics.csv)、[数值敏感性](variance_sensitivity.csv)、[恢复轨迹汇总](trajectory_summary.csv)
- [原输入大小/哈希](source_inputs.json)、[2037个原记录压缩包](run_records.zip)、[核验源码](review_source.py)

原运行和原报告字节保持；没有重训、调参、修改驱动或SSH/代理配置。宽带与带噪尚未运行。
