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
候选数范围为 19216–19216，中位数 19216.0；全部真值位于关联局部窗：`True`。
局部相关宽度在全局搜索边界截断的半宽数为 2；涉及场景数为 1。

- `frequency_single_shot_C2_J1_K128` CUDA—NumPy 模板等价性：`pass`，相对误差 1.252e-05，最小 $\rho^2$=1.000000000。
- `tensor_structured_J8_K16_RE128` CUDA—NumPy 模板等价性：`pass`，相对误差 1.331e-05，最小 $\rho^2$=1.000000000。
- frequency_single_shot_C2_J1_K128 / 45 dB / `sbl_jointfreq_3d`：全带线性均值转 dB -4.531，逐样本中位数 -4.531，≤−10 dB 比例 0.0%，三径成功率 0.0%。
- frequency_single_shot_C2_J1_K128 / 45 dB / `oracle_geometry_ls`：全带线性均值转 dB -22.884，逐样本中位数 -22.884，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- frequency_single_shot_C2_J1_K128 / 45 dB / `grid_omp_3d`：全带线性均值转 dB -5.452，逐样本中位数 -5.452，≤−10 dB 比例 0.0%，三径成功率 0.0%。
- frequency_single_shot_C2_J1_K128 / 50 dB / `sbl_jointfreq_3d`：全带线性均值转 dB -5.146，逐样本中位数 -5.146，≤−10 dB 比例 0.0%，三径成功率 100.0%。
- frequency_single_shot_C2_J1_K128 / 50 dB / `oracle_geometry_ls`：全带线性均值转 dB -27.884，逐样本中位数 -27.884，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- frequency_single_shot_C2_J1_K128 / 50 dB / `grid_omp_3d`：全带线性均值转 dB -5.583，逐样本中位数 -5.583，≤−10 dB 比例 0.0%，三径成功率 0.0%。
- frequency_single_shot_C2_J1_K128 / 55 dB / `grid_omp_3d`：全带线性均值转 dB -15.538，逐样本中位数 -15.538，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- frequency_single_shot_C2_J1_K128 / 55 dB / `sbl_jointfreq_3d`：全带线性均值转 dB -9.875，逐样本中位数 -9.875，≤−10 dB 比例 0.0%，三径成功率 100.0%。
- frequency_single_shot_C2_J1_K128 / 55 dB / `oracle_geometry_ls`：全带线性均值转 dB -32.884，逐样本中位数 -32.884，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- frequency_single_shot_C2_J1_K128 / 60 dB / `grid_omp_3d`：全带线性均值转 dB -17.553，逐样本中位数 -17.553，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- frequency_single_shot_C2_J1_K128 / 60 dB / `oracle_geometry_ls`：全带线性均值转 dB -37.884，逐样本中位数 -37.884，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- frequency_single_shot_C2_J1_K128 / 60 dB / `sbl_jointfreq_3d`：全带线性均值转 dB -13.261，逐样本中位数 -13.261，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- frequency_single_shot_C2_J1_K128 / 65 dB / `grid_omp_3d`：全带线性均值转 dB -20.079，逐样本中位数 -20.079，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- frequency_single_shot_C2_J1_K128 / 65 dB / `oracle_geometry_ls`：全带线性均值转 dB -42.884，逐样本中位数 -42.884，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- frequency_single_shot_C2_J1_K128 / 65 dB / `sbl_jointfreq_3d`：全带线性均值转 dB -19.368，逐样本中位数 -19.368，≤−10 dB 比例 100.0%，三径成功率 0.0%。
- frequency_single_shot_C2_J1_K128 / 70 dB / `grid_omp_3d`：全带线性均值转 dB -20.088，逐样本中位数 -20.088，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- frequency_single_shot_C2_J1_K128 / 70 dB / `sbl_jointfreq_3d`：全带线性均值转 dB -26.878，逐样本中位数 -26.878，≤−10 dB 比例 100.0%，三径成功率 0.0%。
- frequency_single_shot_C2_J1_K128 / 70 dB / `oracle_geometry_ls`：全带线性均值转 dB -47.884，逐样本中位数 -47.884，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- tensor_structured_J8_K16_RE128 / 45 dB / `grid_omp_3d`：全带线性均值转 dB -11.348，逐样本中位数 -11.348，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- tensor_structured_J8_K16_RE128 / 45 dB / `sbl_jointfreq_3d`：全带线性均值转 dB -3.933，逐样本中位数 -3.933，≤−10 dB 比例 0.0%，三径成功率 0.0%。
- tensor_structured_J8_K16_RE128 / 45 dB / `oracle_geometry_ls`：全带线性均值转 dB -16.508，逐样本中位数 -16.508，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- tensor_structured_J8_K16_RE128 / 50 dB / `oracle_geometry_ls`：全带线性均值转 dB -21.508，逐样本中位数 -21.508，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- tensor_structured_J8_K16_RE128 / 50 dB / `grid_omp_3d`：全带线性均值转 dB -17.870，逐样本中位数 -17.870，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- tensor_structured_J8_K16_RE128 / 50 dB / `sbl_jointfreq_3d`：全带线性均值转 dB -8.425，逐样本中位数 -8.425，≤−10 dB 比例 0.0%，三径成功率 0.0%。
- tensor_structured_J8_K16_RE128 / 55 dB / `grid_omp_3d`：全带线性均值转 dB -19.663，逐样本中位数 -19.663，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- tensor_structured_J8_K16_RE128 / 55 dB / `oracle_geometry_ls`：全带线性均值转 dB -26.508，逐样本中位数 -26.508，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- tensor_structured_J8_K16_RE128 / 55 dB / `sbl_jointfreq_3d`：全带线性均值转 dB -12.712，逐样本中位数 -12.712，≤−10 dB 比例 100.0%，三径成功率 0.0%。
- tensor_structured_J8_K16_RE128 / 60 dB / `grid_omp_3d`：全带线性均值转 dB -20.599，逐样本中位数 -20.599，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- tensor_structured_J8_K16_RE128 / 60 dB / `oracle_geometry_ls`：全带线性均值转 dB -31.508，逐样本中位数 -31.508，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- tensor_structured_J8_K16_RE128 / 60 dB / `sbl_jointfreq_3d`：全带线性均值转 dB -16.902，逐样本中位数 -16.902，≤−10 dB 比例 100.0%，三径成功率 0.0%。
- tensor_structured_J8_K16_RE128 / 65 dB / `grid_omp_3d`：全带线性均值转 dB -21.051，逐样本中位数 -21.051，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- tensor_structured_J8_K16_RE128 / 65 dB / `sbl_jointfreq_3d`：全带线性均值转 dB -22.338，逐样本中位数 -22.338，≤−10 dB 比例 100.0%，三径成功率 0.0%。
- tensor_structured_J8_K16_RE128 / 65 dB / `oracle_geometry_ls`：全带线性均值转 dB -36.508，逐样本中位数 -36.508，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- tensor_structured_J8_K16_RE128 / 70 dB / `oracle_geometry_ls`：全带线性均值转 dB -41.508，逐样本中位数 -41.508，≤−10 dB 比例 100.0%，三径成功率 100.0%。
- tensor_structured_J8_K16_RE128 / 70 dB / `sbl_jointfreq_3d`：全带线性均值转 dB -26.516，逐样本中位数 -26.516，≤−10 dB 比例 100.0%，三径成功率 0.0%。
- tensor_structured_J8_K16_RE128 / 70 dB / `grid_omp_3d`：全带线性均值转 dB -21.268，逐样本中位数 -21.268，≤−10 dB 比例 100.0%，三径成功率 100.0%。


## 主 SNR 兼容字段汇总（旧 16 点口径，不用于宽带恢复结论）

### frequency_single_shot_C2_J1_K128

1. `grid_omp_3d`：NMSE -15.533 dB，全路径成功率 100.0%，波束损失 0.036 dB，净谱效率 11.295 bit/s/Hz。
2. `sbl_jointfreq_3d`：NMSE -9.873 dB，全路径成功率 100.0%，波束损失 0.178 dB，净谱效率 11.249 bit/s/Hz。
- Oracle 参考：NMSE -32.882 dB，波束损失 0.001 dB。

### tensor_structured_J8_K16_RE128

1. `grid_omp_3d`：NMSE -19.645 dB，全路径成功率 100.0%，波束损失 0.016 dB，净谱效率 10.303 bit/s/Hz。
2. `sbl_jointfreq_3d`：NMSE -12.708 dB，全路径成功率 0.0%，波束损失 0.099 dB，净谱效率 10.278 bit/s/Hz。
- Oracle 参考：NMSE -26.507 dB，波束损失 0.002 dB。

## 判读边界

本轮只检验已知逐径粗中心后的局部候选采样与捕获域。即使条件诊断通过，也不能据此启动正式比较；未知全局几何下的粗获取仍需单独验证。
