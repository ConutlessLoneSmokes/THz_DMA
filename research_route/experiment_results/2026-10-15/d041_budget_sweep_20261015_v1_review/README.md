# D-041预算扩展正式核验档案

日期2026-10-15；[完整核验与分析](audit_report.md)。原运行和自动汇总保留，新材料只写此目录。

- `verification.json`、`source_inputs.json`：14阶段原始完整性、冻结和来源哈希，保留前次35681个文件；本次21607个输入文件另存哈希。
- `replay.json`：场景0/99的22个固定GPU端点复算，没有选参。
- `*_endpoints.csv`：14个已核验原始最终指标副本；`endpoint_summary.csv`独立重算三种估计器的480组汇总。
- `worst_variance_scene_endpoints.csv`：每场景两种数值方差中较差的DDPM端点；`variance_sensitivity.csv`单列数值稳定性。
- `paired_test_contrasts.csv`：10000次配对场景bootstrap，种子2026101546；`8192_frequency_coverage_scene_pairs.csv`保留全部100场景的覆盖配对。
- `per_budget_validation_best_mean.csv`仅描述每预算验证平均NMSE最低的主设置，不替代原“最少RE再最少符号”冻结工作点。
- `previous_full_frequency_anchor_checks.csv`对照旧全频J扫描；`trajectory_summary.csv`保留两方差的指定VP时刻均值。
- `budget_review.png`、`budget_review.pdf`及`figure_a_source.csv`、`figure_b_source.csv`：全部24主设置资源曲线和8192 RE逐场景控制，灰点是场景、菱形是线性均值转dB。
- `review_source.py`、`analysis_source.py`及日志：独立核验、固定复算和后处理代码。原始权威记录在`runs/d041_budget_sweep_20261015_v1_*`，自动汇总在`results/d041_budget_sweep/`。

统计单位是生成的三径场景，测试沿用原100场景，数值方差不是物理噪声。图为一般研究核验用途，未核对特定期刊要求；图表未添加平滑、删点或测试选参。最终文件哈希另存`manifest.json`，复现读取核验源码中的固定前缀与目录；已完成的核验和复算会拒绝覆盖其JSON结论。
