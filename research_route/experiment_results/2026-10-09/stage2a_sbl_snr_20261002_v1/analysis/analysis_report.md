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
- frequency_single_shot_C2_J1_K128 / 45 dB / `sbl_jointfreq_3d`：全带线性均值转 dB -2.529，逐样本中位数 -3.391，≤−10 dB 比例 0.0%，三径成功率 12.0%。
- frequency_single_shot_C2_J1_K128 / 45 dB / `oracle_geometry_ls`：全带线性均值转 dB -16.362，逐样本中位数 -17.426，≤−10 dB 比例 99.0%，三径成功率 100.0%。
- frequency_single_shot_C2_J1_K128 / 45 dB / `grid_omp_3d`：全带线性均值转 dB -4.098，逐样本中位数 -5.658，≤−10 dB 比例 8.0%，三径成功率 33.0%。
- frequency_single_shot_C2_J1_K128 / 50 dB / `sbl_jointfreq_3d`：全带线性均值转 dB -5.824，逐样本中位数 -6.901，≤−10 dB 比例 9.0%，三径成功率 35.0%。
- frequency_single_shot_C2_J1_K128 / 50 dB / `oracle_geometry_ls`：全带线性均值转 dB -21.362，逐样本中位数 -22.426，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- frequency_single_shot_C2_J1_K128 / 50 dB / `grid_omp_3d`：全带线性均值转 dB -6.826，逐样本中位数 -7.800，≤−10 dB 比例 37.0%，三径成功率 65.0%。
- frequency_single_shot_C2_J1_K128 / 55 dB / `grid_omp_3d`：全带线性均值转 dB -8.566，逐样本中位数 -10.545，≤−10 dB 比例 52.0%，三径成功率 80.0%。
- frequency_single_shot_C2_J1_K128 / 55 dB / `sbl_jointfreq_3d`：全带线性均值转 dB -9.410，逐样本中位数 -10.692，≤−10 dB 比例 56.0%，三径成功率 42.0%。
- frequency_single_shot_C2_J1_K128 / 55 dB / `oracle_geometry_ls`：全带线性均值转 dB -26.362，逐样本中位数 -27.426，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- frequency_single_shot_C2_J1_K128 / 60 dB / `grid_omp_3d`：全带线性均值转 dB -9.705，逐样本中位数 -12.338，≤−10 dB 比例 63.0%，三径成功率 82.0%。
- frequency_single_shot_C2_J1_K128 / 60 dB / `oracle_geometry_ls`：全带线性均值转 dB -31.362，逐样本中位数 -32.426，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- frequency_single_shot_C2_J1_K128 / 60 dB / `sbl_jointfreq_3d`：全带线性均值转 dB -13.971，逐样本中位数 -15.860，≤−10 dB 比例 93.0%，三径成功率 48.0%。
- frequency_single_shot_C2_J1_K128 / 65 dB / `grid_omp_3d`：全带线性均值转 dB -9.746，逐样本中位数 -12.784，≤−10 dB 比例 63.0%，三径成功率 83.0%。
- frequency_single_shot_C2_J1_K128 / 65 dB / `oracle_geometry_ls`：全带线性均值转 dB -36.362，逐样本中位数 -37.426，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- frequency_single_shot_C2_J1_K128 / 65 dB / `sbl_jointfreq_3d`：全带线性均值转 dB -17.577，逐样本中位数 -19.636，≤−10 dB 比例 98.0%，三径成功率 44.0%。
- frequency_single_shot_C2_J1_K128 / 70 dB / `grid_omp_3d`：全带线性均值转 dB -10.043，逐样本中位数 -12.795，≤−10 dB 比例 65.0%，三径成功率 82.0%。
- frequency_single_shot_C2_J1_K128 / 70 dB / `sbl_jointfreq_3d`：全带线性均值转 dB -17.965，逐样本中位数 -23.229，≤−10 dB 比例 94.0%，三径成功率 43.0%。
- frequency_single_shot_C2_J1_K128 / 70 dB / `oracle_geometry_ls`：全带线性均值转 dB -41.362，逐样本中位数 -42.426，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- tensor_structured_J8_K16_RE128 / 45 dB / `grid_omp_3d`：全带线性均值转 dB -8.698，逐样本中位数 -10.037，≤−10 dB 比例 51.0%，三径成功率 51.0%。
- tensor_structured_J8_K16_RE128 / 45 dB / `sbl_jointfreq_3d`：全带线性均值转 dB -2.903，逐样本中位数 -2.923，≤−10 dB 比例 0.0%，三径成功率 18.0%。
- tensor_structured_J8_K16_RE128 / 45 dB / `oracle_geometry_ls`：全带线性均值转 dB -17.596，逐样本中位数 -18.080，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- tensor_structured_J8_K16_RE128 / 50 dB / `oracle_geometry_ls`：全带线性均值转 dB -22.596，逐样本中位数 -23.080，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- tensor_structured_J8_K16_RE128 / 50 dB / `grid_omp_3d`：全带线性均值转 dB -13.420，逐样本中位数 -13.706，≤−10 dB 比例 97.0%，三径成功率 87.0%。
- tensor_structured_J8_K16_RE128 / 50 dB / `sbl_jointfreq_3d`：全带线性均值转 dB -7.726，逐样本中位数 -7.848，≤−10 dB 比例 12.0%，三径成功率 38.0%。
- tensor_structured_J8_K16_RE128 / 55 dB / `grid_omp_3d`：全带线性均值转 dB -15.136，逐样本中位数 -15.115，≤−10 dB 比例 100.0%，三径成功率 99.0%。
- tensor_structured_J8_K16_RE128 / 55 dB / `oracle_geometry_ls`：全带线性均值转 dB -27.596，逐样本中位数 -28.080，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- tensor_structured_J8_K16_RE128 / 55 dB / `sbl_jointfreq_3d`：全带线性均值转 dB -12.564，逐样本中位数 -12.689，≤−10 dB 比例 96.0%，三径成功率 51.0%。
- tensor_structured_J8_K16_RE128 / 60 dB / `grid_omp_3d`：全带线性均值转 dB -15.648，逐样本中位数 -15.863，≤−10 dB 比例 100.0%，三径成功率 99.0%。
- tensor_structured_J8_K16_RE128 / 60 dB / `oracle_geometry_ls`：全带线性均值转 dB -32.596，逐样本中位数 -33.080，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- tensor_structured_J8_K16_RE128 / 60 dB / `sbl_jointfreq_3d`：全带线性均值转 dB -17.282，逐样本中位数 -17.357，≤−10 dB 比例 100.0%，三径成功率 48.0%。
- tensor_structured_J8_K16_RE128 / 65 dB / `grid_omp_3d`：全带线性均值转 dB -15.818，逐样本中位数 -16.084，≤−10 dB 比例 100.0%，三径成功率 99.0%。
- tensor_structured_J8_K16_RE128 / 65 dB / `sbl_jointfreq_3d`：全带线性均值转 dB -22.012，逐样本中位数 -22.388，≤−10 dB 比例 100.0%，三径成功率 52.0%。
- tensor_structured_J8_K16_RE128 / 65 dB / `oracle_geometry_ls`：全带线性均值转 dB -37.596，逐样本中位数 -38.080，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- tensor_structured_J8_K16_RE128 / 70 dB / `oracle_geometry_ls`：全带线性均值转 dB -42.596，逐样本中位数 -43.080，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- tensor_structured_J8_K16_RE128 / 70 dB / `sbl_jointfreq_3d`：全带线性均值转 dB -26.486，逐样本中位数 -26.895，≤−10 dB 比例 100.0%，三径成功率 46.0%。
- tensor_structured_J8_K16_RE128 / 70 dB / `grid_omp_3d`：全带线性均值转 dB -15.846，逐样本中位数 -16.103，≤−10 dB 比例 100.0%，三径成功率 99.0%。


## 主 SNR 兼容字段汇总（旧 16 点口径，不用于宽带恢复结论）

### frequency_single_shot_C2_J1_K128

1. `sbl_jointfreq_3d`：NMSE -9.396 dB，全路径成功率 42.0%，波束损失 0.183 dB，净谱效率 11.342 bit/s/Hz。
2. `grid_omp_3d`：NMSE -8.557 dB，全路径成功率 80.0%，波束损失 0.199 dB，净谱效率 11.318 bit/s/Hz。
- Oracle 参考：NMSE -26.365 dB，波束损失 0.002 dB。

### tensor_structured_J8_K16_RE128

1. `grid_omp_3d`：NMSE -15.091 dB，全路径成功率 99.0%，波束损失 0.055 dB，净谱效率 10.471 bit/s/Hz。
2. `sbl_jointfreq_3d`：NMSE -12.725 dB，全路径成功率 51.0%，波束损失 0.118 dB，净谱效率 10.427 bit/s/Hz。
- Oracle 参考：NMSE -27.617 dB，波束损失 0.002 dB。

## 判读边界

本轮只检验已知逐径粗中心后的局部候选采样与捕获域。即使条件诊断通过，也不能据此启动正式比较；未知全局几何下的粗获取仍需单独验证。
