# 方向一：Zhou DDPM / SURE-DM 的实现与最小验证

初稿：2026-10-09；更新：2026-10-11。状态：**MIMO 长训练、完整验证和冻结独立测试已核验，DDPM/SURE 的主要趋势在独立测试保持；多种子及 DMA 长训练未运行。** 用户已确认 [19f 号方案](19f_方向一Zhou_DDPM与SURE_DM复现计划.md)，决策为 D-037。第 1–6 节保留初稿的实现与最小验证，第 7 节记录完整验证。D-038 的独立测试结果与范围见 [19h 第 6 节](19h_方向一MIMO冻结独立测试.md)。

2026-10-13 后续：D-039 的宽带训练和数字/DMA冻结测试已完成，稀疏选频与DMA全局DDPM仍失败，详见 [19i 第11节](19i_方向一传统参照与分步场景迁移.md#11-2026-10-13-正式结果核验与下一建议)。上面的“DMA未运行”属于2026-10-11历史状态；本轮完整证据已归档到 [结果快照](experiment_results/2026-10-13/README.md)，Linux入口见 [运行指南](../docs/LINUX_RUN.md)。

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：run
- Origin Date：2026-10-09
- Verification Status：ANALYZED
- Version Label：zhou_parametric_implementation_v1
- 核验范围：初稿的 27 项检查、35 个小阶段运行；追加的 20 个长训练/验证阶段、24 项恢复复算、209 项 LASSO 预算核验。统计结论限于单训练种子及验证集。

## 1. 实际实现与复用

主算法依据 [Zhou@2025，TWC](https://arxiv.org/abs/2408.10501v2) 的正文 Algorithm 1、式(22)–(23)、SURE 两阶段训练和附录 I 的网络结构重新实现。参数化 MIMO 替代 QuaDRiGa，DMA 属于本地观测适配；本轮没有复现原论文 Fig.4/5/13 的数据分布或具体数值。

| 文件 | 实际职责与复用 |
|---|---|
| [mimo.py](../src/zhou2025_repro/mimo.py) | 复用单行 UPA、ULA、三径抽样、可分阵列响应、自由空间及吸收损耗，补齐发射端响应和数字 QPSK 导频算子 |
| [dma.py](../src/zhou2025_repro/dma.py) | 包装原 TorchDmaOperator、噪声白化和分块求解，保留原 F1/F2；额外信息参照复用原 OMP/SBL/条件 LMMSE |
| [data.py](../src/zhou2025_repro/data.py) | 单位空间 DFT、实虚布局、原单位 CSI 读写、训练尺度；DMA 直接索引现有缓存 |
| [baselines.py](../src/zhou2025_repro/baselines.py) | 提取既有 angular LASSO 和实值等效，增加全局样本统计 LMMSE，以较小 Gram 求解和分批信道合成避免全带稠密协方差 |
| [ddpm.py](../src/zhou2025_repro/ddpm.py) | 两层前置卷积、时间缩放/偏置、三层后置卷积；VP 加噪、噪声预测损失、当前状态似然引导及确定性反向更新 |
| [sure.py](../src/zhou2025_repro/sure.py) | 两种 SURE 散度系数、有限差分、单噪声去噪器和最近扩散时刻 |
| [运行入口](../scripts/run_zhou2025_repro.py) | prepare、corrupt、train、train-sure、denoise、evaluate；复用原归档、原子保存、指标与数据波束评价 |

新包为 `src/zhou2025_repro/`。原 [参考脚本](../scripts/check_zhou_reference_curves.py) 改为导入共享 angular、real_operator 和 lasso，原调用名称保留；三个有效参数控制的 LASSO 输出、迭代数及收敛标记与提取前完全一致。SciPy 已在现有环境中，补入 pyproject.toml 的依赖声明；本轮未安装新库。

MIMO 使用 300 GHz、16×64、三径可分双端模型，接收端单行 UPA 与 ULA 几何一致。DMA 保持 300 GHz、30 GHz、128 子载波、8×160 阵元、球面波真值及真实频响。DMA 的 DFT 仅作用于两个空间轴，保留全部频率；它不保证近场信道稀疏，DFT 索引不直接当作路径角度。

主 DDPM 不输入粗中心或在线导频，先与同训练数据的全局 LMMSE 比较。已有条件参照的逐径中心和 Oracle 真几何分别标记在 information_condition，不能用它们证明同信息条件下的学习优劣。离线训练位于工作站，在线恢复位于基站边缘处理器；恢复的原单位 CSI 经既有可行码本选择 DMA 数据状态。

## 2. 噪声、尺度与 SURE 目标

信道按训练复数 RMS $\rho$ 归一化；实部和虚部组成网络输入。MIMO 输入为 `(batch,2,Nr,Nt)`，DMA 为 `(batch,2S,E,K)`。观测中的复噪声方差为 $\nu$，对应每个实分量方差 $\nu/2$；标准实扩散噪声在复域对应方差 2。两种观测算子的似然得分与独立构造的稠密实值公式一致。

离线含噪数据为 $z=h+w$，$\sigma_w^2=0.1/1$ 表示每个实分量的方差；物理噪声方差为 $\rho^2\sigma_w^2$。两档使用同一标准噪声并改变尺度，导频观测噪声采用独立种子。含噪训练尺度仅由训练二阶矩减去已知复噪声功率计算：

$$
\widehat\rho^2=\frac{1}{M}\sum_{m=1}^{M}\lvert\widetilde H_m\rvert^2-2\sigma_{w,\mathrm{phys}}^2.
$$

SURE 的拟合与模型选择不读取干净标签；后处理报告才使用模拟真值检查实际去噪误差。去噪数据继承含噪数据估计的信号尺度，不将输出再次归一化。在线比较统一使用目标干净训练集的物理尺度，再按各模型自己的尺度换算输入和噪声，所有方法共用同一带噪观测。

令 $a=1/(1+\sigma_w^2)$、$q=\sqrt a\,z$。以下风险按实维数 $d$ 平均：

$$
R_{\mathrm{corrected}}
=\frac{\lVert f(q)-z\rVert_2^2}{d}
+\frac{2\sigma_w^2\sqrt a}{d}\operatorname{div}_{q}f(q)
-\sigma_w^2.
$$

`paper` 保留原文式(34)的散度系数，去掉上式的 $\sqrt a$；`corrected` 是本地按式(35)求导变量应用链式法则的修正。两者都加入相同的常数项 $-\sigma_w^2$，它不改变训练梯度。散度采用原文的高斯随机方向和 $10^{-5}$ 有限差分。SURE 使用 float64，避免小差分被 float32 舍入吞没。

线性控制在 $\sigma_w^2=1$、$g(z)=0.4z$ 下，修正风险与真实 MSE 的均值差小于 0.02；原文系数相对修正风险的偏差为 $0.8(\sqrt2-1)\approx0.33137$。这是对指定变量定义的数学与数值检查，不是两种训练目标的完整性能排名。

主 DDPM 保留 100 步、线性 $\beta=10^{-4}\rightarrow0.1$、标准高斯初始化和固定最终输出。默认 refinement=1；未混入旧 VE/EDM/Heun、条件高斯初始化或作者公开代码的去噪状态引导。引导强度和 LASSO 惩罚在验证集选取，独立测试必须引用完全相同模型、数据、评价设置的完成验证记录。

## 3. 已完成的运行与实际结果

35 个阶段运行均完成：MIMO 连通性 8 个、DMA 连通性 8 个、完整网络的 MIMO 短验证 19 个。训练数据和模型依赖通过核对；三个评价运行共 296 行，分别为 56、48、192 行。19 个运行有完整源码 ZIP，1349 个 ZIP 成员与启动时哈希一致；更早的 16 个连通性运行保存源码清单，没有 ZIP。逐运行列表见 [run_inventory.csv](experiment_results/2026-10-13/zhou_mimo_probe_eval_20261009_v1/analysis/run_inventory.csv)。

### 3.1 100 步、完整 CNN 的 MIMO 短验证

配置为 [zhou2025_parametric_mimo_probe.toml](../configs/zhou2025_parametric_mimo_probe.toml)：256/32/4 个训练/验证/测试场景，网络最大宽度 64、55025 个参数。DDPM 训练 16 epochs、128 次更新；四个 SURE 去噪器训练 8 epochs、64 次更新。在线评价只用了前 4 个验证场景、0/20 dB、Np=64/38；测试未参与选参或评价。

以下为 20 dB 的 NMSE，先取场景线性均值再转 dB。SURE 行中的方差是离线训练噪声，在线观测始终相同。

| 方法 | Np=64 | Np=38 |
|---|---:|---:|
| 干净训练协方差 LMMSE，256 样本 | −3.920 | +0.144 |
| LASSO | −17.278 | −12.993 |
| Clean DDPM | −11.183 | −1.846 |
| Naive DDPM，$\sigma_w^2=0.1$ | −11.055 | −1.433 |
| SURE-DM corrected，$\sigma_w^2=0.1$ | −11.046 | −1.445 |
| SURE-DM paper，$\sigma_w^2=0.1$ | −11.052 | −1.448 |
| Naive DDPM，$\sigma_w^2=1$ | −10.854 | +1.498 |
| SURE-DM corrected，$\sigma_w^2=1$ | −10.863 | +1.441 |
| SURE-DM paper，$\sigma_w^2=1$ | −10.854 | +1.473 |

Clean DDPM 的验证噪声预测 MSE 从第 32 次更新的 0.940 降至第 128 次的 0.391，但仍落后 LASSO。Np=38 时，全局 LMMSE 的组均值从 0 dB 的 −1.821 回升到 20 dB 的 +0.144；其样本协方差误差、先验失配与有限场景波动尚未分离，不将它解释为贝叶斯估计的普遍性质。

分时刻去噪以同一验证输入和标准人工噪声计算：

| t | $\bar\alpha_t$ | 噪声预测 MSE | Clean DDPM 去噪 NMSE / dB |
|---|---:|---:|---:|
| 1 | 0.999900 | 1.124 | −36.610 |
| 25 | 0.735074 | 0.433 | −5.188 |
| 50 | 0.282981 | 0.211 | +0.158 |
| 75 | 0.056081 | 0.149 | +6.861 |
| 100 | 0.005619 | 0.141 | +16.837 |

高噪声端的噪声预测误差较小，换算到干净信道仍被 $(1-\bar\alpha_t)/\bar\alpha_t$ 放大。当前去噪先验和在线欠定恢复均未通过性能验收。这里是完整 CSI 人工加噪诊断，不是导频恢复曲线。各训练分布的 35 行结果见 [denoising_profiles.csv](experiment_results/2026-10-13/zhou_mimo_probe_eval_20261009_v1/analysis/denoising_profiles.csv)；noisy/denoised 模型的表内目标是自己的训练数据分布，不能直接当作相同干净目标排名。

SURE 去噪输出在全部 32 个验证场景上的真值误差为：

| $\sigma_w^2$ | 原含噪数据 / dB | corrected 输出 / dB | paper 输出 / dB |
|---|---:|---:|---:|
| 0.1 | −7.095 | −7.169 | −7.147 |
| 1 | +2.905 | +2.825 | +2.845 |

改善仅约 0.05–0.08 dB。SURE-DM 与 Naive DDPM 的在线差异也很小，尚未复现预期收益。原始结果保留；不通过选择测试场景、反向中间步或调整 SNR 横轴改善曲线。

### 3.2 宽带 DMA 的连通性检查

配置使用 `--smoke`：8 个训练场景、4 个验证场景，评价前 2 个验证场景；网络宽度 16、9832 个参数，扩散 8 步，各训练只更新 4 次。F1/F2、0/20 dB 和三类 DM、干净/含噪全局 LMMSE、真几何 LS 共 48 行。

20 dB 的 Clean DDPM 全带 NMSE 为 F1 +9.673/F2 +9.666 dB，SURE-DM 为 +9.515/+9.509 dB。此结果只证明入口连通，恢复性能未通过。导频、完整 128 点、F2 未观测 112 点和可行数据波束均已输出；F1 的未观测域为空值。

后续代码已恢复 `channel_nmse` 的原 16 点口径，完整频带始终使用 `fullband_128_nmse` / `nmse_linear`。首批 smoke 记录的 `channel_nmse_subcarrier_count=128` 保留原值，不用该别名与历史 16 点结果混比。

## 4. 数值检查、归档与续跑

完整 27 项检查通过，其中新包相关 8 项覆盖单位 DFT/列向量化/ULA、MIMO 与 DMA 伴随和似然得分、VP 噪声、两种 SURE 风险与梯度、两种 Gram 大小的 LMMSE、训练续跑、评价续跑及冻结测试约束。正交导频的 LS 和共享 LASSO 沿用既有检查。

训练中断后恢复的模型、历史和分时刻去噪记录与连续训练一致。评价选参先保存、指标块后保存；模拟在模型指标块刚保存时中断，续跑仍保留完整的选参记录。较大场景批次的观测噪声保留小批次的同一前缀。

8 项 MIMO/DMA、两协议、两 SNR 的已保存恢复结果用当前代码复算，最大差为 $1.78\times10^{-15}$ dB。该检查确认后续去噪报告、保存顺序和指标别名修订未改变主恢复数值。原始 metrics、权重和数据没有改写。核验结果见 [integrity.json](experiment_results/2026-10-13/zhou_mimo_probe_eval_20261009_v1/analysis/integrity.json)，核验过程保存在 [verify_and_profile.py](experiment_results/2026-10-13/zhou_mimo_probe_eval_20261009_v1/analysis/verify_and_profile.py)。

每个阶段保存配置、随机种子、数据/模型依赖及源码哈希；新运行还保存 source_snapshot.zip。数据数组分批落盘，训练保存优化器和随机状态，评价按方法块保存。中断只能在源码、配置和依赖完全一致时加 `--resume`；完整运行拒绝覆盖。训练源码更新后不能续跑旧源码的检查点，应使用原源码快照或新的运行号。

训练后的 denoising_profile.json 使用最佳验证权重，在五个预定时刻报告验证输入分布的去噪误差；首批短运行的同类补充结果保存在 analysis，不追写训练摘要。

## 5. 正式运行入口，保留初稿交接命令

2026-10-10：用户已执行本节 MIMO 数据、全部训练和验证命令。DMA 长训练和独立测试命令未执行；最新结果见第 7 节。

完整配置是缩小数据规模的学习试验，不是原文 10 万样本复现。MIMO 为 4096/256/100 个场景、DDPM 500 epochs 即每模型 16000 次更新，SURE 100 epochs 即每模型 3200 次更新。先完成干净 DDPM 的验证，再运行两档 SURE 对比；验证不通过时保留结果并定位误差。

在项目目录的 PowerShell 执行。每条命令成功完成后再进入下一阶段；运行号均为新名称。

```powershell
$py = 'D:/Program Files/Miniconda/envs/ris_td3/python.exe'
$cfg = 'configs/zhou2025_parametric_mimo.toml'
$prefix = 'zhou_mimo_formal_20261009'
$dataId = "${prefix}_data_v1"
$cleanId = "${prefix}_clean_v1"

& $py scripts/run_zhou2025_repro.py prepare --config $cfg --run-id $dataId
& $py scripts/run_zhou2025_repro.py train --config $cfg --run-id $cleanId --data-run-id $dataId
& $py scripts/run_zhou2025_repro.py evaluate --config $cfg --run-id "${prefix}_clean_val_v1" --data-run-id $dataId --train-run-id $cleanId --split validation
```

两档噪声、两种 SURE 目标及共同验证：

```powershell
$models = @($cleanId)
foreach ($level in @(@{tag='01'; variance=0.1}, @{tag='1'; variance=1.0})) {
    $noisyId = "${prefix}_noisy$($level.tag)_v1"
    $naiveId = "${prefix}_naive$($level.tag)_v1"
    & $py scripts/run_zhou2025_repro.py corrupt --config $cfg --run-id $noisyId --data-run-id $dataId --noise-variance $level.variance
    if ($LASTEXITCODE -ne 0) { throw 'corrupt failed' }
    & $py scripts/run_zhou2025_repro.py train --config $cfg --run-id $naiveId --data-run-id $noisyId
    if ($LASTEXITCODE -ne 0) { throw 'naive training failed' }
    $models += $naiveId
    foreach ($variant in @('paper','corrected')) {
        $sureId = "${prefix}_sure$($level.tag)_${variant}_v1"
        $denoisedId = "${prefix}_denoised$($level.tag)_${variant}_v1"
        $dmId = "${prefix}_suredm$($level.tag)_${variant}_v1"
        & $py scripts/run_zhou2025_repro.py train-sure --config $cfg --run-id $sureId --data-run-id $noisyId --sure-variant $variant
        if ($LASTEXITCODE -ne 0) { throw 'SURE training failed' }
        & $py scripts/run_zhou2025_repro.py denoise --config $cfg --run-id $denoisedId --data-run-id $noisyId --train-run-id $sureId
        if ($LASTEXITCODE -ne 0) { throw 'denoising failed' }
        & $py scripts/run_zhou2025_repro.py train --config $cfg --run-id $dmId --data-run-id $denoisedId
        if ($LASTEXITCODE -ne 0) { throw 'SURE-DM training failed' }
        $models += $dmId
    }
}
& $py scripts/run_zhou2025_repro.py evaluate --config $cfg --run-id "${prefix}_val_v1" --data-run-id $dataId --train-run-id $models --split validation
```

共同验证覆盖 100 个验证场景、两种导频、九档 SNR、七个 DDPM 与五个线性/稀疏基线，共 21600 行。结果先检查 comparison.csv、denoising_profile.json、tuning.json 和逐场景 metrics_raw.csv。正式主结果在配置冻结后至少使用三个独立训练种子；`--seed` 修改训练种子，并为各阶段指定新运行号，不覆盖一号种子结果。

独立测试入口只在完成验证、决定冻结配置后使用；以下为初稿示例，未运行。2026-10-10 用户已授权下一步，采用 [19h 号文档](19h_方向一MIMO冻结独立测试.md) 的完整命令和新运行名：

```powershell
& $py scripts/run_zhou2025_repro.py evaluate --config $cfg --run-id "${prefix}_test_v1" --data-run-id $dataId --train-run-id $models --split test --tuning-run-id "${prefix}_val_v1"
```

DMA 使用 [zhou2025_parametric_dma.toml](../configs/zhou2025_parametric_dma.toml)。首个正式数据阶段只引用 `diffusion_data_20261002_v1` 的既有 2048/256/100 缓存，不再生成真值：

```powershell
$cfg = 'configs/zhou2025_parametric_dma.toml'
$prefix = 'zhou_dma_formal_20261009'
$dataId = "${prefix}_data_v1"
$cleanId = "${prefix}_clean_v1"
& $py scripts/run_zhou2025_repro.py prepare --config $cfg --run-id $dataId
& $py scripts/run_zhou2025_repro.py train --config $cfg --run-id $cleanId --data-run-id $dataId
& $py scripts/run_zhou2025_repro.py evaluate --config $cfg --run-id "${prefix}_clean_val_v1" --data-run-id $dataId --train-run-id $cleanId --split validation
```

两阶段 SURE 对比可复用上面的噪声/目标循环，保持新的 DMA prefix 和 cfg。DMA 当前 batch=4，500 epochs 对每个 DDPM 为 256000 次更新；100 epochs 对每个 SURE 去噪器为 51200 次更新。该预算远大于本轮短检查，应先在目标工作站测量训练吞吐和显存。现有本机为 RTX 4050 6 GB；本轮没有测量正式模型的显存峰值或承诺完成时间。

DMA 仍按训练平均白化观测功率设置接收 SNR，输出白化和实际 RF SNR。各协议分别匹配噪声的这一诊断不用于同输入噪声、同发射能量下的 F1/F2 物理收益结论。净谱效率已扣训练和切换占用，当前字段明确未扣在线求解时间。

## 6. 初稿未完成的验证，2026-10-09 历史状态

4096 样本学习试验、正式宽带训练、三个训练种子、独立测试、原文 10 万样本、完整 SURE 收益及作者代码引导消融均未运行。低位 ADC、外部深度基线、LDPC/BER 不在本次实现范围。当前最小验证不能支持方法优越性或可发表性判断。

下一项已准备的检查是干净 MIMO 正式验证：同时观察先验分时刻去噪、Np=38 的恢复、全局样本协方差与 LASSO。若干净去噪仍弱，定位训练尺度、日程和网络能力；若去噪改善而在线恢复失败，再隔离正文近似似然及反向更新误差。SURE 的阶段性效果以保存的去噪误差和共同在线观测比较判定。

## 7. 2026-10-10 MIMO 长训练与完整验证核验

20 个阶段运行全部完成：七个 DDPM 各训练 500 epochs、16000 次更新，四个 SURE 去噪器各训练 100 epochs、3200 次更新。独立数据种子生成 4096/256/100 个训练/验证/测试场景；评价使用前 100 个验证场景、Np=64/38、−10:5:30 dB。共同验证 21600 行与先前单独 Clean 验证 7200 行均完整。

300 个数据文件、11 个模型依赖、20 个源码 ZIP 的 1420 个成员与当前源码核对通过。7200 条共同记录的 NMSE、SNR、选参和方法状态完全一致，耗时分别保留；24 项固定恢复复算最大差为 $3.55\times10^{-15}$ dB。原始数据、权重与指标没有改写。

表中为逐场景线性 NMSE 均值转 dB，单位均为 dB：

| 方法 | Np64，20 dB | Np38，20 dB | Np64，30 dB | Np38，30 dB |
|---|---:|---:|---:|---:|
| Clean-data LMMSE | −17.168 | −4.295 | −25.055 | −4.261 |
| LASSO，原预算 | −18.158 | −13.670 | −23.047 | −14.729 |
| Clean DDPM | −25.578 | −19.743 | −31.545 | −21.923 |
| Naive DM，$\sigma_w^2=0.1$ | −16.336 | −12.264 | −21.091 | −13.179 |
| SURE-DM paper，$\sigma_w^2=0.1$ | −20.180 | −15.853 | −25.245 | −17.181 |
| SURE-DM corrected，$\sigma_w^2=0.1$ | −20.219 | −15.927 | −25.238 | −17.252 |
| Naive DM，$\sigma_w^2=1$ | −12.263 | −7.870 | −17.370 | −8.227 |
| SURE-DM paper，$\sigma_w^2=1$ | −14.104 | −9.664 | −19.049 | −10.184 |
| SURE-DM corrected，$\sigma_w^2=1$ | −16.608 | −12.089 | −21.498 | −12.866 |

Clean DDPM 相对全局 LMMSE 和 LASSO 的配对 95% 区间各为 18/18 组完全小于零；四种 SURE-DM 相对各自 Naive DM 也各为 18/18 组小于零。20 dB 下 corrected SURE-DM 相对 Naive DM 的组均值改善约 3.7–4.3 dB，但仍落后 Clean DDPM。高训练噪声下 corrected 相对 paper 的在线配对改善约 2.5 dB；低噪声下两种目标的差异通常不足 0.1 dB。两种目标继续保留，本轮未更改算法或冻结新的方法选择。

SURE 在全部 256 个验证 CSI 上的实际去噪误差如下，干净标签只用于核验：

| $\sigma_w^2$ | 原含噪 CSI | paper 输出 | corrected 输出 |
|---|---:|---:|---:|
| 0.1 | −6.970 | −14.050 | −14.009 |
| 1 | +3.030 | −2.482 | −5.393 |

低噪声下 paper 的去噪 MSE 略低，corrected 的多数在线配对误差略低，两种指标的细小排序差异保留。高噪声下 corrected 在去噪和在线恢复均更低。两种训练损失包含不同散度系数，效果比较采用实际 CSI 误差。

原 1800 条 LASSO 中有 209 条达到 1000 次预算：Np38 的 25/30 dB 各 72/84 条，Np64 的 30 dB 为 53 条。固定原惩罚与观测，将预算提高至 10000 后全部在 1005–2494 次收敛；逐例 NMSE 变化为 −0.0581 至 +0.0504 dB，三个受影响组的曲线变化绝对值都小于 0.002 dB，当前排序保持。该检查未重新选择惩罚，不作为重新调参后的 LASSO 排名。

七个 DDPM 的组均值随 SNR 下降，15→20、20→25、25→30 dB 的 4200 个逐场景相邻步骤全部下降。Clean DDPM 的五个自然去噪时刻仍有高噪声端误差，t=100 的 NMSE 为 +3.181 dB；在线收益不代表所有时刻的先验去噪都已准确。

本轮同一 100 个验证场景用于选参和评价，144 个配对区间未作多重校正，只描述当前模型及选参下的效果。独立测试、多种子、宽带 DMA 长训练、QuaDRiGa 原分布及原图逐点一致仍未验证。下一阶段建议先冻结当前配置核验独立测试；该建议未代替用户的运行选择。

完整报告、11 项统计解释核查、逐场景差值和图表分别见 [audit_report.md](experiment_results/2026-10-13/zhou_mimo_formal_20261009_val_v1/analysis/audit_report.md)、[paired_comparisons.csv](experiment_results/2026-10-13/zhou_mimo_formal_20261009_val_v1/analysis/paired_comparisons.csv) 与 [完整曲线](experiment_results/2026-10-13/zhou_mimo_formal_20261009_val_v1/analysis/mimo_validation_curves.png)。核验入口为 [audit_zhou2025_mimo.py](../scripts/audit_zhou2025_mimo.py)，图表和报告重建脚本保存在该运行的 analysis 目录。
