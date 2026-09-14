# 方向一 Stage 1-C 最近邻方法迁移与同场景比较

更新日期：2026-09-08  
状态：正式 Monte Carlo 与自动分析已完成；已据结果进入 Stage 1-D

## 1. 本阶段问题

Stage 1-C 不再把物理频响 J1 与时间扫描 J2 当作两个互斥算法。J1 是同一配置下跨子载波的固有频率编码，J2 是增加时间配置，两者属于可组合的采集维度。J2 保留为资源参考，主要问题改为：

> 在完全相同的太赫兹近场 DMA 观测、导频、噪声和双径信道上，文献中的代表性传统估计思想相对基础 Grid OMP 能否改善通信所需的 CSI、波束和净谱效率，代价是多少？

Stage 1-B 的 `stop_or_redesign` 标签不追溯修改。它表示 Grid OMP 未达到当时固定的精确双径门槛，不代表频率响应方向停止。

## 2. 比较设计

正式配置固定 Stage 1-B 的 300 GHz、30 GHz 带宽、320 阵元、128 个导频 RE 和 `nominal` LoS+单反射场景，不扫描频段、阵列、路径功率或硬件失配。

| 实验轴 | 水平 | 解释 |
|---|---|---|
| 估计器 | Grid OMP、Yang-style OG-OLS 适配、联合频率 SBL 适配 | 主比较轴；同一设计内共享完全相同的逐样本观测 |
| 采集设计 | 物理频响 J1、物理时间扫描 J2 | J1 是主块；J2 只给资源参考，不与估计器效应混合 |
| 噪声 | 匹配输出 SNR 0 dB、无噪声 | 0 dB 比较实际性能；无噪声定位算法地板 |
| 独立重复 | 5 个种子，每个 20 个场景 | 每个场景是统计独立单位；所有方法使用配对样本 |

正式运行共生成 $1\times100\times2\times2\times3=1200$ 行逐样本结果。执行顺序在每个场景—噪声块内随机化，随机种子固定并归档。

## 3. 方法及迁移边界

### 3.1 Grid OMP

沿用 Stage 1-B 的双原子 OMP：先后按归一化相关选择两个网格原子，再联合最小二乘估计复增益。它是低复杂度内部基线，不对应某篇论文的完整复现。

### 3.2 Yang-style OG-OLS 适配

来源边界为 [Yang@2024，预印本](https://arxiv.org/abs/2407.04954) <!--ref:yang2024xldma--><!--anchor:algorithm:1--> 的 OG-DOLS 思路。当前实现保留两项核心机制：

1. 使用 OLS 的投影后残差下降准则选择网格原子；
2. 使用精确球面波—分子吸收—真实 DMA 频响模板的一阶导数，对距离和角度进行迭代离网格修正，并在每次更新后重估复增益。

原论文是窄带 UPA、多 RF 链、EL-AZ 解耦和分布式观测；本项目是单 RF 输出的一维接收 DMA 与联合频率观测。因此当前方法是 OG-DOLS 的单联合观测特化，不包含原论文的 UPA 解耦和测量矩阵优化，不标为论文原始数值复现。

### 3.3 联合频率 SBL 适配

来源边界为 [Gao@2024，FITEE](https://arxiv.org/abs/2308.13381) <!--ref:gao2024thzunfolding--><!--anchor:section:3.1--> 使用的 SBL/AMP-SBL 母算法。当前实现采用经典复高斯 SBL 的 EM 更新，将整个多子载波观测写成一个联合频率字典，并用确定性的匹配滤波预筛选将候选数限制为 64。

Gao 原文针对每个子载波具有多维混合阵列观测的 MMV 模型，并进一步展开 AMP-SBL。当前接收结构每个子载波只有一个 RF 输出，观测维度不相同，因此本实现既不是其 MMV AMP-SBL，也不是展开网络。它用于判断贝叶斯稀疏恢复思想在当前算子上的迁移效果，完整 Gao 复现留到观测维度和学习阶段明确后再做。

## 4. 评价与自动判读

主块固定为 `nominal`、匹配输出 SNR 0 dB、物理频响 J1。评价顺序为：

1. 宽带阵元域 CSI NMSE；
2. 可行 DMA 数据波束损失与扣除训练后的净谱效率；
3. 批量 1 CPU 在线时间；
4. 精确双径成功率和残差，只作诊断。

每个外部适配方法与 Grid OMP 使用同一场景的配对差值和场景级 bootstrap 95% 区间。自动状态只有方法筛查含义：

- `clear_external_method_gain`：至少一种适配方法的配对 NMSE 区间完全低于 0；
- `inconclusive_external_method_gain`：点估计改善但区间跨 0；
- `grid_omp_not_beaten`：当前适配方法没有降低平均配对 NMSE；

这些状态不作为论文创新性、方向继续或停止的单一门槛。J1—J2 结果单独写入资源参考对比，不能归因于估计器。

## 5. 运行与输出

正式运行：

```powershell
conda run -n ris_td3 python scripts\run_direction1_stage1c.py --config configs\direction1_stage1c_method_transfer.toml
```

只重新分析已有运行：

```powershell
conda run -n ris_td3 python scripts\analyze_direction1_stage1c.py --run-dir runs\<run_id>
```

每次运行自动保存逐样本 CSV、环境、哈希、随机顺序和方法来源说明，并在 `analysis/` 下生成：

- `analysis_compact.json`：完整性、主排名、配对方法效应和资源参考；
- `analysis_report.md`：供快速阅读的中文报告；
- `summary_table.csv`：全部场景—设计—估计器汇总；
- `paired_method_effects.csv`：外部适配方法相对 OMP 的配对差值；
- `figures/01_nmse_by_estimator.png`；
- `figures/02_primary_communication_metrics.png`；
- `figures/03_nmse_runtime_tradeoff.png`。

2026-09-10 已清理早期临时接口检查配置、结果与专用测试文件；正式运行和分析产物保留。

## 6. 正式结果后的最小决策

- 若离网格或 SBL 适配明确优于 OMP：保留获胜传统方法作为后续强基线，再迁移 Ruoyu Zhang 的张量方法；只有仍存在稳定精度—在线时间缺口时才加入展开学习。
- 若传统适配均不能改善 J1，但在 J2 上能够改善：优先判断 J1 观测信息限制，不通过调参把采集问题伪装成算法问题。
- 若 OMP 已在通信指标上与复杂方法相当：不增加网络，将贡献重心放在少配置采集和配置设计。
- Deshpande 单次波束训练属于直接波束输出，应在后续以可行波束、获取时间和净吞吐比较，不与 CSI NMSE 强行排序。

## 7. 正式结果与执行决定

正式运行 `stage1c_20260908T122650Z` 完成 1200/1200 行，无重复键或必需指标非有限值。主场景 J1、匹配输出 SNR 0 dB 下：

- Yang-style OG-OLS 相对 Grid OMP 的逐场景 NMSE 差为 $-5.64$ dB，95% CI 为 $[-6.29,-4.98]$ dB，100 个场景中 96 个改善；净谱效率差为 $0.0578$ bit/s/Hz；批量 1 CPU 时间约为 OMP 的 8.35 倍。
- 联合频率 SBL 的 NMSE 差为 $-0.052$ dB，95% CI 跨 0；噪声块全部运行至 40 次上限，时间约为 OMP 的 59.7 倍。本结果只否定当前单联合观测 SBL 适配，不否定 SBL 家族。
- 三种方法的精确双径成功率均为 36%，说明改善发生在 CSI 重构和通信波束层，不构成精确路径参数恢复结论。
- J2 复现了相同方法排序，继续只作为采集资源参考。

按本节预先规定的第一种分支，保留 Yang-style OG-OLS 作为强传统基线并进入 [Stage 1-D 张量结构迁移](16_方向一Stage1D张量结构迁移.md)。
