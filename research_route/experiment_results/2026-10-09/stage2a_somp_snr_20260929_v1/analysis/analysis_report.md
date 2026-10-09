# Stage 2-A 三维传统参照自动报告

- 数据完整：`True`
- 实现门槛：`diagnostic_only`
- 主 SNR 方法筛查：`not_interpreted_diagnostic`
- Oracle 使用真实路径几何，只作 genie-aided 结构参照，不参与可部署方法排名，也不宣称是所有估计器的严格统计上界。
- F1 与 F2 分开报告；二者共享 128 导频 RE，但训练符号和切换开销不同。

## 实现检查

- `raw_data_complete_and_paired`：`True`
- `same_observation_across_estimators`：`True`
- `physical_truth_only`：`True`
- `equal_128_pilot_re_across_schedules`：`True`
- `oracle_geometry_matching_exact`：`True`

## 固定扇区历史诊断

本次为波束中心自适应局部网格运行，不套用旧五场景 55→65 dB 固定扇区判据。

## 自适应局部网格诊断

每条路径的随机粗中心在观测前生成；连续真值在对应中心的半功率邻域内随机偏移，没有人为半格放置。该信息条件只用于条件化实现诊断。
候选数范围为 6860–21666，中位数 14653.0；全部真值位于关联局部窗：`True`。
局部相关宽度在全局搜索边界截断的半宽数为 22；涉及场景数为 11。

- `frequency_single_shot_C2_J1_K128` CUDA—NumPy 模板等价性：`pass`，相对误差 1.252e-05，最小 $\rho^2$=1.000000000。
- `tensor_structured_J8_K16_RE128` CUDA—NumPy 模板等价性：`pass`，相对误差 1.331e-05，最小 $\rho^2$=1.000000000。
- frequency_single_shot_C2_J1_K128 / 45 dB / `grid_somp_3d`：全带线性均值转 dB 1.057，逐样本中位数 0.682，≤−10 dB 比例 0.0%，三径成功率 0.0%。
- frequency_single_shot_C2_J1_K128 / 45 dB / `oracle_geometry_ls`：全带线性均值转 dB -16.362，逐样本中位数 -17.426，≤−10 dB 比例 99.0%，三径成功率 100.0%。
- frequency_single_shot_C2_J1_K128 / 45 dB / `grid_omp_3d`：全带线性均值转 dB -4.098，逐样本中位数 -5.658，≤−10 dB 比例 8.0%，三径成功率 33.0%。
- frequency_single_shot_C2_J1_K128 / 50 dB / `grid_somp_3d`：全带线性均值转 dB 0.934，逐样本中位数 0.735，≤−10 dB 比例 0.0%，三径成功率 0.0%。
- frequency_single_shot_C2_J1_K128 / 50 dB / `oracle_geometry_ls`：全带线性均值转 dB -21.362，逐样本中位数 -22.426，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- frequency_single_shot_C2_J1_K128 / 50 dB / `grid_omp_3d`：全带线性均值转 dB -6.826，逐样本中位数 -7.800，≤−10 dB 比例 37.0%，三径成功率 65.0%。
- frequency_single_shot_C2_J1_K128 / 55 dB / `grid_omp_3d`：全带线性均值转 dB -8.566，逐样本中位数 -10.545，≤−10 dB 比例 52.0%，三径成功率 80.0%。
- frequency_single_shot_C2_J1_K128 / 55 dB / `grid_somp_3d`：全带线性均值转 dB 0.905，逐样本中位数 0.533，≤−10 dB 比例 0.0%，三径成功率 0.0%。
- frequency_single_shot_C2_J1_K128 / 55 dB / `oracle_geometry_ls`：全带线性均值转 dB -26.362，逐样本中位数 -27.426，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- frequency_single_shot_C2_J1_K128 / 60 dB / `grid_omp_3d`：全带线性均值转 dB -9.705，逐样本中位数 -12.338，≤−10 dB 比例 63.0%，三径成功率 82.0%。
- frequency_single_shot_C2_J1_K128 / 60 dB / `oracle_geometry_ls`：全带线性均值转 dB -31.362，逐样本中位数 -32.426，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- frequency_single_shot_C2_J1_K128 / 60 dB / `grid_somp_3d`：全带线性均值转 dB 0.904，逐样本中位数 0.505，≤−10 dB 比例 0.0%，三径成功率 0.0%。
- frequency_single_shot_C2_J1_K128 / 65 dB / `grid_omp_3d`：全带线性均值转 dB -9.746，逐样本中位数 -12.784，≤−10 dB 比例 63.0%，三径成功率 83.0%。
- frequency_single_shot_C2_J1_K128 / 65 dB / `oracle_geometry_ls`：全带线性均值转 dB -36.362，逐样本中位数 -37.426，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- frequency_single_shot_C2_J1_K128 / 65 dB / `grid_somp_3d`：全带线性均值转 dB 0.944，逐样本中位数 0.561，≤−10 dB 比例 0.0%，三径成功率 0.0%。
- frequency_single_shot_C2_J1_K128 / 70 dB / `grid_omp_3d`：全带线性均值转 dB -10.043，逐样本中位数 -12.795，≤−10 dB 比例 65.0%，三径成功率 82.0%。
- frequency_single_shot_C2_J1_K128 / 70 dB / `grid_somp_3d`：全带线性均值转 dB 0.902，逐样本中位数 0.533，≤−10 dB 比例 0.0%，三径成功率 0.0%。
- frequency_single_shot_C2_J1_K128 / 70 dB / `oracle_geometry_ls`：全带线性均值转 dB -41.362，逐样本中位数 -42.426，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- tensor_structured_J8_K16_RE128 / 45 dB / `grid_omp_3d`：全带线性均值转 dB -8.698，逐样本中位数 -10.037，≤−10 dB 比例 51.0%，三径成功率 51.0%。
- tensor_structured_J8_K16_RE128 / 45 dB / `grid_somp_3d`：全带线性均值转 dB 0.245，逐样本中位数 0.379，≤−10 dB 比例 0.0%，三径成功率 0.0%。
- tensor_structured_J8_K16_RE128 / 45 dB / `oracle_geometry_ls`：全带线性均值转 dB -17.596，逐样本中位数 -18.080，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- tensor_structured_J8_K16_RE128 / 50 dB / `oracle_geometry_ls`：全带线性均值转 dB -22.596，逐样本中位数 -23.080，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- tensor_structured_J8_K16_RE128 / 50 dB / `grid_omp_3d`：全带线性均值转 dB -13.420，逐样本中位数 -13.706，≤−10 dB 比例 97.0%，三径成功率 87.0%。
- tensor_structured_J8_K16_RE128 / 50 dB / `grid_somp_3d`：全带线性均值转 dB 0.190，逐样本中位数 0.309，≤−10 dB 比例 0.0%，三径成功率 14.0%。
- tensor_structured_J8_K16_RE128 / 55 dB / `grid_omp_3d`：全带线性均值转 dB -15.136，逐样本中位数 -15.115，≤−10 dB 比例 100.0%，三径成功率 99.0%。
- tensor_structured_J8_K16_RE128 / 55 dB / `oracle_geometry_ls`：全带线性均值转 dB -27.596，逐样本中位数 -28.080，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- tensor_structured_J8_K16_RE128 / 55 dB / `grid_somp_3d`：全带线性均值转 dB 0.186，逐样本中位数 0.339，≤−10 dB 比例 0.0%，三径成功率 31.0%。
- tensor_structured_J8_K16_RE128 / 60 dB / `grid_omp_3d`：全带线性均值转 dB -15.648，逐样本中位数 -15.863，≤−10 dB 比例 100.0%，三径成功率 99.0%。
- tensor_structured_J8_K16_RE128 / 60 dB / `oracle_geometry_ls`：全带线性均值转 dB -32.596，逐样本中位数 -33.080，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- tensor_structured_J8_K16_RE128 / 60 dB / `grid_somp_3d`：全带线性均值转 dB 0.221，逐样本中位数 0.304，≤−10 dB 比例 0.0%，三径成功率 34.0%。
- tensor_structured_J8_K16_RE128 / 65 dB / `grid_omp_3d`：全带线性均值转 dB -15.818，逐样本中位数 -16.084，≤−10 dB 比例 100.0%，三径成功率 99.0%。
- tensor_structured_J8_K16_RE128 / 65 dB / `grid_somp_3d`：全带线性均值转 dB 0.208，逐样本中位数 0.318，≤−10 dB 比例 0.0%，三径成功率 35.0%。
- tensor_structured_J8_K16_RE128 / 65 dB / `oracle_geometry_ls`：全带线性均值转 dB -37.596，逐样本中位数 -38.080，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- tensor_structured_J8_K16_RE128 / 70 dB / `oracle_geometry_ls`：全带线性均值转 dB -42.596，逐样本中位数 -43.080，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- tensor_structured_J8_K16_RE128 / 70 dB / `grid_somp_3d`：全带线性均值转 dB 0.259，逐样本中位数 0.379，≤−10 dB 比例 0.0%，三径成功率 35.0%。
- tensor_structured_J8_K16_RE128 / 70 dB / `grid_omp_3d`：全带线性均值转 dB -15.846，逐样本中位数 -16.103，≤−10 dB 比例 100.0%，三径成功率 99.0%。

## OMP—SOMP 多信噪比对照

SOMP 用各频点独立系数选共同几何支持，随后用共享复增益重构全带 CSI；两方法复用相同白化观测和字典。下表 NMSE 为逐场景线性值取均值后转 dB；差值及区间为逐场景 dB 配对结果。

| 协议 | 输入参考 SNR/dB | OMP NMSE/dB | SOMP NMSE/dB | SOMP−OMP/dB，95% CI |
|---|---:|---:|---:|---:|
| frequency_single_shot_C2_J1_K128 | 45 | -4.098 | 1.057 | +6.612 [+5.929, +7.287] |
| frequency_single_shot_C2_J1_K128 | 50 | -6.826 | 0.934 | +9.441 [+8.721, +10.213] |
| frequency_single_shot_C2_J1_K128 | 55 | -8.566 | 0.905 | +11.459 [+10.590, +12.317] |
| frequency_single_shot_C2_J1_K128 | 60 | -9.705 | 0.904 | +12.568 [+11.715, +13.475] |
| frequency_single_shot_C2_J1_K128 | 65 | -9.746 | 0.944 | +12.753 [+11.892, +13.584] |
| frequency_single_shot_C2_J1_K128 | 70 | -10.043 | 0.902 | +12.951 [+12.123, +13.815] |
| tensor_structured_J8_K16_RE128 | 45 | -8.698 | 0.245 | +10.069 [+9.546, +10.583] |
| tensor_structured_J8_K16_RE128 | 50 | -13.420 | 0.190 | +14.175 [+13.739, +14.674] |
| tensor_structured_J8_K16_RE128 | 55 | -15.136 | 0.186 | +15.738 [+15.202, +16.222] |
| tensor_structured_J8_K16_RE128 | 60 | -15.648 | 0.221 | +16.457 [+15.920, +16.990] |
| tensor_structured_J8_K16_RE128 | 65 | -15.818 | 0.208 | +16.611 [+16.136, +17.131] |
| tensor_structured_J8_K16_RE128 | 70 | -15.846 | 0.259 | +16.733 [+16.208, +17.274] |


## 主 SNR 兼容字段汇总（旧 16 点口径，不用于宽带恢复结论）

### frequency_single_shot_C2_J1_K128

1. `grid_omp_3d`：NMSE -8.557 dB，全路径成功率 80.0%，波束损失 0.199 dB，净谱效率 11.318 bit/s/Hz。
2. `grid_somp_3d`：NMSE 0.907 dB，全路径成功率 0.0%，波束损失 12.856 dB，净谱效率 7.214 bit/s/Hz。
- Oracle 参考：NMSE -26.365 dB，波束损失 0.002 dB。

### tensor_structured_J8_K16_RE128

1. `grid_omp_3d`：NMSE -15.091 dB，全路径成功率 99.0%，波束损失 0.055 dB，净谱效率 10.471 bit/s/Hz。
2. `grid_somp_3d`：NMSE 0.037 dB，全路径成功率 31.0%，波束损失 2.594 dB，净谱效率 9.328 bit/s/Hz。
- Oracle 参考：NMSE -27.617 dB，波束损失 0.002 dB。

## 判读边界

本轮只检验已知逐径粗中心后的局部候选采样与捕获域。即使条件诊断通过，也不能据此启动正式比较；未知全局几何下的粗获取仍需单独验证。
