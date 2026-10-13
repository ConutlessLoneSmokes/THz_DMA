# 方向一：MIMO 冻结选参后的独立测试

初稿：2026-10-10；更新：2026-10-11。状态：**用户已完成独立测试，21600 行核验通过，DDPM/SURE 的主要趋势在测试集保持。** 第 1–5 节保留交接计划和命令，第 6 节记录实际核验。

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：plan
- Origin Date：2026-10-10
- Verification Status：ANALYZED
- Version Label：zhou_mimo_independent_test_v1

## 1. 本轮问题与范围

检验 [19g 第 7 节](19g_方向一Zhou_DDPM与SURE_DM实现与最小验证.md) 的 DDPM/SURE 收益是否保留在未参与选参的独立场景上。当前共同验证同时用于引导/惩罚选择和性能报告；独立测试将参数固定后只评价误差。

复用 4096/256/100 的既有 MIMO 数据，不重新生成场景、不重训模型。测试为原 `test` split 的全部 100 个场景，种子 2026100903；训练和验证种子分别为 2026100901、2026100902。模型训练仍只有一个种子。

该实验是本地 300 GHz 三径参数化信道上的独立验收。原文为 40 GHz QuaDRiGa，主曲线采用增强迭代。独立测试通过也不代表这些条件已与原文对齐，数值比较继续标记为原图近似读数。[Zhou@2025，TWC](https://arxiv.org/html/2408.10501v2)

## 2. 冻结对象

选参来源：`zhou_mimo_formal_20261009_val_v1`。新运行名：`zhou_mimo_independent_test_20261010_v1`。

| 对象 | 冻结内容 |
|---|---|
| 真值与尺度 | `zhou_mimo_formal_20261009_data_v1`；训练集全局复 RMS；只使用既有测试 split |
| 七个模型 | Clean、Naive(0.1/1)、SURE-DM paper/corrected(0.1/1)，验证时的最佳权重 |
| 线性/稀疏基线 | 干净及两档含噪训练样本协方差 LMMSE、LS、LASSO |
| 观测 | 同一 16×64 系统；Np=64/38；原 QPSK 导频矩阵与原 SNR 定义 |
| 反向更新 | 原 T=100 日程、基础更新、每场景 100 次网络调用 |
| 选参 | 原 126 个网络引导项与 18 个 LASSO 惩罚项，共 144 项；测试不搜索候选 |
| 求解预算 | LASSO 原 1000 次预算；是否收敛逐项保留 |
| 随机量 | 观测噪声和反向初始化使用入口既有 test split 的独立种子分支 |

完整配置仍为 [zhou2025_parametric_mimo.toml](../configs/zhou2025_parametric_mimo.toml)。配置、模型和依赖必须与来源验证一致，七个模型全部保留，不依据测试误差选择 SURE 版本。预期为 100×2×9×12=21600 行，SNR 为 −10:5:30 dB。

`tuning.json` 的冻结 SHA-256 为 `51b09c7d4e66ba8c0020ce9b8a538a9469b9b9a8abd9beb33759eafb0d76bbe0`。入口保存选参来源哈希；中断续跑若来源被修改，拒绝继续，防止不同块混用不同选参。算法方程和原训练/验证配置保持原值。

## 3. 2026-10-10 已完成的交接检查

- 七个 DDPM、三个协方差来源及 300 个数据文件的依赖与原验证一致；144 个选参项齐全，选中值与原候选最小验证误差一致。
- 与原源码清单的差异仅为测试入口的选参来源续跑保护及对应检查。两个原始验证指标文件、配置、模型和选参未改写。
- 四项已有验证场景的恢复复算最大差为 $3.55\times10^{-15}$ dB，覆盖 Np64/38、20 dB 的 Clean 和 paper SURE-DM(1)。观测批量与原运行一致，避免改变 GPU 运算批量带来的舍入差异。
- 27 项检查通过。合成独立测试确认每场景只用一次已选引导/惩罚，不重新调参；选参来源变化会拒绝续跑。
- 交接时真实测试 CSI 未用于恢复、模型选择或参数调整，正式测试运行目录尚未创建；2026-10-11 的运行核验见第 6 节。

可复核记录：[independent_test_preflight.json](experiment_results/2026-10-13/zhou_mimo_formal_20261009_val_v1/analysis/independent_test_preflight.json)。轻量检查保存于来源验证的 analysis，不是独立测试结果。

## 4. 用户运行的 PowerShell

本节保留交接命令。该运行已完成，不重复执行同一运行号。

该块自动读取原验证保存的七个模型列表和数据运行号。使用本机已核验的 Python 环境；无需执行 prepare 或 train。

```powershell
Set-Location -LiteralPath 'C:/Users/Milo/Documents/Obsidian_data/THz_phase_array/research_ai_edge_dma_rhs'
$py = 'D:/Program Files/Miniconda/envs/ris_td3/python.exe'
$cfg = 'configs/zhou2025_parametric_mimo.toml'
$valId = 'zhou_mimo_formal_20261009_val_v1'
$testId = 'zhou_mimo_independent_test_20261010_v1'
$frozen = Get-Content -Raw -Encoding UTF8 -LiteralPath "runs/$valId/config_resolved.json" | ConvertFrom-Json

$runArgs = @(
    'scripts/run_zhou2025_repro.py', 'evaluate',
    '--config', $cfg,
    '--run-id', $testId,
    '--data-run-id', $frozen.config.job.data_run_id,
    '--train-run-id'
) + @($frozen.config.job.train_run_id) + @(
    '--split', 'test',
    '--tuning-run-id', $valId
)

& $py @runArgs
if ($LASTEXITCODE -ne 0) { throw 'Independent test failed; inspect the saved status and error.' }
```

中断后，重新设置同一组变量，把运行行改为 `& $py @runArgs --resume`。配置、源码、模型和选参来源必须保持相同；完整运行不可覆盖。现有已完成的验证仍使用其原始源码快照，本轮的新测试会保存当前源码。

结果写入 `runs/zhou_mimo_independent_test_20261010_v1/`，包括 `metrics_raw.csv`、`metrics_summary.json`、`comparison.csv`、`comparison.png`、`tuning_source.json` 与状态/检查点。以 `status.json` 为 complete 且 summary 为 test、21600 行判断执行完成；性能结论另行核验。

## 5. 运行后固定核验内容

核对 100 个测试 ID 与训练/验证分离、21600 行完整、模型/数据/尺度/选参哈希不变；每组方法共享同一观测，144 个测试选参项与来源一致。报告两种导频、九 SNR 的全部曲线，不因个别场景或方法排序回落筛除记录。

按既有验证核验的八种方法配对关系报告 144 组场景 dB 差：Clean 对干净 LMMSE/LASSO；四个 SURE-DM 对各自 Naive；两档 corrected 对 paper。追加 10000 次场景重采样、固定分析种子 2026101001，并保存全部比较。区间仅描述当前单训练种子，未作多重比较校正，不代替多种子结论。曲线采用线性 NMSE 均值转 dB，配对采用逐场景 dB 差。

单列 15→20、20→25、25→30 dB 的逐场景变化，检查高 SNR 是否回升。如 LASSO 未收敛，固定原惩罚/观测/停止阈值补充至 10000 次并报告敏感性；不得在测试集重选惩罚，不改写原始指标。独立测试收益与原文数值相符程度分别判定。

本轮没有运行多训练种子、增强迭代、原文数据分布、低位 ADC、BER、硬件时延比较或 DMA 长训练。

## 6. 2026-10-11 独立测试实际核验

用户已完成 `zhou_mimo_independent_test_20261010_v1`，全部 100 个 test 场景、21600 行完整。训练/验证/测试 ID 无重叠，归一化和协方差仍来自原训练集；七个模型、四个 SURE 数据生成依赖、300 个数据文件及 71 个源码成员核对通过。144 个测试选参项及其文件哈希与原验证相同，未重新调参。24 项固定恢复复算最大差为 $3.55\times10^{-15}$ dB，原始数据、模型、选参及指标没有改写。

以下为场景线性 NMSE 均值转 dB，LASSO 为原 1000 次预算：

| 方法 | Np64，20 dB | Np38，20 dB | Np64，30 dB | Np38，30 dB |
|---|---:|---:|---:|---:|
| 干净训练 LMMSE | −16.961 | −4.238 | −24.887 | −4.213 |
| LASSO | −17.993 | −13.174 | −23.002 | −14.079 |
| Clean DDPM | −25.473 | −19.589 | −31.406 | −21.639 |
| Naive DM，$\sigma_w^2=1$ | −12.204 | −7.750 | −17.351 | −8.101 |
| SURE-DM paper，$\sigma_w^2=1$ | −14.010 | −9.527 | −18.973 | −10.033 |
| SURE-DM corrected，$\sigma_w^2=1$ | −16.492 | −11.964 | −21.400 | −12.715 |

Clean 相对 LMMSE/LASSO、四种 SURE 相对各自 Naive 的配对 95% 区间均各为 18/18 组低于零。20 dB 的 Clean 对 LASSO：Np64/38 的逐场景 dB 差均值分别为 −7.319/−5.987，区间为 [−7.535,−7.097]/[−6.426,−5.537]，100/100 和 98/100 场景改善。原文系数版 SURE 也保持收益；低训练噪声下 corrected 对 paper 的区间只有 15/18 组低于零，细小差异保留。144 个区间未作多重校正，结果只描述当前固定训练种子和导频实现。

227/1800 条原预算 LASSO 未收敛。保持原惩罚和观测，补充至 10000 次预算后全部在 1007–2364 次收敛，组均值最大变化为 0.000842 dB，方法排序保持。七个学习模型的组均值随 SNR 提高下降；4200 个高 SNR 逐场景相邻步骤均无回升。

七个模型的各组测试曲线减原验证曲线范围为 −0.191 至 +0.284 dB。两集合为独立场景，不作同序号场景配对或单因素归因。64 导频的 DDPM 数值量级与原文主曲线接近；38 导频、30 dB 仍高于原文增强版约 6 dB、高于原文 T100 基础版约 3 dB。原信道分布、训练规模和增强迭代的影响尚未分别验证，基线数值也有差异。

**本地 MIMO 核心算法与主要性能趋势已完成独立验收。** 这支持 DDPM 先验恢复和 SURE 两阶段训练在当前场景有效，不将结论扩大为原论文全部数值/公式或宽带 DMA 已复现。

完整报告含 11 项统计解释核查及全部曲线，见 [audit_report.md](experiment_results/2026-10-13/zhou_mimo_independent_test_20261010_v1/analysis/audit_report.md)。配对数据为 [paired_comparisons.csv](experiment_results/2026-10-13/zhou_mimo_independent_test_20261010_v1/analysis/paired_comparisons.csv)，原验证对照为 [validation_test_curves.csv](experiment_results/2026-10-13/zhou_mimo_independent_test_20261010_v1/analysis/validation_test_curves.csv)，曲线为 [mimo_independent_test_curves.png](experiment_results/2026-10-13/zhou_mimo_independent_test_20261010_v1/analysis/mimo_independent_test_curves.png)。核验入口为 [audit_zhou2025_mimo_test.py](../scripts/audit_zhou2025_mimo_test.py)。多种子及 DMA 等后续实验尚未运行。
