# 方向一 Stage 1-D 张量结构迁移与低秩假设核查

更新日期：2026-09-08  
状态：正式 Monte Carlo 已完成；旧一维低秩迁移未超过 Yang-style OG-OLS，作为 Stage 2 二维统一平台的反例保留

## 1. 进入依据

Stage 1-C 正式运行 `stage1c_20260908T122650Z` 完成 1200/1200 行。相同 J1 观测下，Yang-style OG-OLS 相对 Grid OMP 的逐场景 NMSE 差为 $-5.64$ dB，95% CI 为 $[-6.29,-4.98]$ dB；100 个场景中 96 个改善。联合频率 SBL 没有形成稳定收益且在线时间明显更高。

因此本阶段保留 Yang-style OG-OLS 作为强传统参照，迁移张量方法。Grid OMP 只保留为低复杂度基线，不再把精确双径成功率作为单一门槛。

## 2. 本阶段问题

本阶段只回答两个有边界的问题：

1. 在完全相同的 J1 单配置观测上，跨频率低秩处理能否进一步改善 Yang-style OG-OLS？
2. 若张量方法需要额外配置维度，它在相同 128 个导频 RE 下能否通过更高 CSI 或波束精度抵消四个训练符号和三次切换的代价？

J1 与张量原生采集属于不同资源轴。第二个问题不用于把时间配置和频率响应写成互斥算法。

## 3. 来源方法与迁移边界

[Ruoyu Zhang@2025，TWC](https://doi.org/10.1109/TWC.2025.3551144) <!--ref:zhang2025tensorofdm--><!--anchor:section:II-III--> 针对发射 DMA 构造微带顺序训练，把观测写为 $M\times T_e\times T_s\times K$ 四阶 CP 张量，并利用 Vandermonde 因子完成两阶段参数提取。

当前项目是单 RF 输出的一维接收 DMA。J1 只有一个配置和一条联合频率观测，不具备原文的用户合并、微带和单元配置四个独立模式；真实 Lorentzian—波导频响还会在配置与频率之间产生耦合。因此不能把 J1 任意 reshape 后标为原论文复现，也不能直接套用其 CP 唯一性结论。

当前 `zhang_cpd_ogols_adapted` 只迁移低秩频率机制：

- J1：把 128 点频率观测构造成 block-Hankel 矩阵，作 rank-2 截断 SVD；
- 张量原生参考：四个物理配置分别在同一组 32 个子载波上采样，形成配置×Hankel 频率的三阶张量，作 rank-2 复数 CP-ALS；
- 两种情况下均把低秩重构结果送入精确球面波—分子吸收—真实 DMA 频响的 OG-OLS；最终复增益和残差重新在原始观测上拟合。

它不是原论文的四阶 CP、微带顺序训练或代数参数提取复现。该实现只筛查低秩张量机制能否迁移到当前观测算子。

## 4. 固定实验设计

正式实验继续固定 300 GHz、30 GHz 带宽、320 阵元、LoS+单反射和 128 个导频 RE，不扩展频段、阵列、路径分布或硬件失配。

| 实验轴 | 水平 | 解释 |
|---|---|---|
| 估计器 | Grid OMP、Yang-style OG-OLS、Zhang-style CPD→OG-OLS | 同一采集设计内共享完全相同的信道、噪声和观测 |
| J1 主块 | 1 配置、128 子载波、1 符号、0 切换 | 检验同观测张量预处理是否超过 Yang |
| 张量原生参考 | 4 配置、每配置 32 个相同导频子载波、4 符号、3 切换 | 总计仍为 128 RE；单独报告资源代价 |
| 噪声 | 匹配输出 SNR 0 dB、无噪声 | 0 dB 比较通信性能；无噪声核查 rank-2 结构失配 |
| 独立重复 | 5 个种子，每个 20 个场景 | 场景是统计独立单位；方法比较采用场景级配对 bootstrap |

正式运行预计生成 $1\times100\times2\times2\times3=1200$ 行。方法执行顺序在每个场景—噪声块内按固定种子随机化。张量原生参考中重复频率的不同训练符号使用独立噪声实现。

## 5. 指标与自动判读

主效应是 J1 上张量适配相对 Yang 的逐场景 CSI NMSE 差。次要指标包括可行 DMA 波束损失、扣除训练后的净谱效率、批量 1 CPU 时间、CP 收敛率，以及无噪声 block-Hankel 张量的 rank-2 重构残差。路径参数成功率继续只作诊断。

自动状态为：

- `clear_same_observation_tensor_gain`：J1 相同观测下，张量适配的配对 NMSE 95% 区间完全低于 0；
- `tensor_gain_requires_native_acquisition`：J1 未形成明确收益，但四配置张量原生块形成明确收益；
- `inconclusive_tensor_gain`：点估计改善但区间跨 0；
- `tensor_not_better_than_ogols`：两个采集块中均未超过 Yang；

无噪声 rank-2 中位残差超过 5% 只产生结构告警，不单独决定方法胜负。

## 6. 运行与输出

正式运行：

```powershell
conda run -n ris_td3 python scripts\run_direction1_stage1d.py --config configs\direction1_stage1d_tensor_transfer.toml
```

当前机器若终端没有初始化 Conda，可使用：

```powershell
& 'D:\Program Files\Miniconda\Scripts\conda.exe' run -n ris_td3 python scripts\run_direction1_stage1d.py --config configs\direction1_stage1d_tensor_transfer.toml
```

只重新分析已有运行：

```powershell
conda run -n ris_td3 python scripts\analyze_direction1_stage1d.py --run-dir runs\<run_id>
```

每次运行保存逐样本指标、完整配置、代码哈希、随机顺序和方法来源，并在 `analysis/` 生成紧凑 JSON、Markdown 报告、两个 CSV 汇总和四张固定图。2026-09-10 已清理早期临时接口检查产物，正式运行和分析结果保留。

## 7. 正式结果后的最小决策

正式运行 `stage1d_20260909T011834Z` 完成 1200/1200 行，自动状态为 `tensor_not_better_than_ogols`。匹配输出 SNR 0 dB 时，张量迁移相对 Yang-style OG-OLS 的 CSI NMSE 在 J1 同观测块恶化 6.18 dB，95% CI 为 $[5.23,7.14]$ dB；在四配置原生块恶化 3.28 dB，95% CI 为 $[2.03,4.52]$ dB。无噪声 rank-2 残差中位数分别为 52.5% 和 73.3%。

因此下列预设分支实际落在“两个块均无改善且结构残差高”。该结果否定的是把一维接收观测作 Hankel/低秩预处理的当前迁移方式，不足以否定 Zhang 原系统中的微带、配置和频率多维机制。后续不再继续调旧实现，改由 [Stage 2 统一二维基准](17_方向一Stage2统一二维基准.md) 在真实 $S\times J\times K$ 观测张量上重新核查。

- J1 上明确改善：张量低秩机制进入后续主基线，并继续检查其在线时间是否优于直接增加 OG-OLS 迭代。
- 只有四配置块改善：把收益表述为依赖额外采集维，不归因于 J1；是否保留由净谱效率和获取时间决定。
- 两个块均无改善且无噪声 rank-2 残差较高：判定原 CP 可分离结构不适合当前真实频响算子，保留 Yang，转向带物理算子的结构化拟合或直接波束基线。
- 传统方法比较后仍存在稳定精度—在线时间缺口时，才进入轻量展开学习；不使用网络挽救一个已被结构诊断否定的张量假设。
