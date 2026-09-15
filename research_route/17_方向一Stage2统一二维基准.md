# 方向一 Stage 2 统一二维太赫兹 DMA 基准

更新日期：2026-09-15
状态：100 场景正式平台验证通过；Stage 2-A 的 F2 条件通过，F1 单配置角度弱区分性已定位，正式比较暂停

## 1. 阶段目标

Stage 2 不再逐篇搭建互不兼容的论文场景。先固定一个能够容纳频率单次训练、参数化估计、稀疏恢复和张量估计的二维接收 DMA 场景，再把文献机制作为适配器接到同一物理真值、硬件集合和评价端点上。

当前 Stage 2-0 只验收平台和观测算子，不产生方法排名。它回答：二维几何、OFDM 时序、太赫兹传播、DMA 频响、噪声与两类采集协议能否在一个闭环中一致运行；简化层能否恢复预期低秩结构；目标物理层能否保留导致迁移失配的真实耦合。

## 2. 公共系统

采用单用户上行 OFDM、单天线终端和基站侧二维接收 DMA。每条微带接一个 RF 输出。

| 项目 | 当前参考值 | 作用 |
|---|---:|---|
| 载频 / 带宽 | 300 GHz / 30 GHz | 固定 285.117–314.883 GHz 数据栅格 |
| 子载波 / FFT | 128 / 128 | $Delta f=234.375$ MHz，$T_u=4.267$ ns |
| CP | 32 样点 | 1.067 ns；相对 0.25 m 最大超额路程留 0.233 ns 裕量 |
| DMA | 8 条微带 × 每条 160 单元 | 1280 个单元、8 个 RF 输出 |
| 微带内间距 | $0.2\lambda_c=0.1999$ mm | 孔径主方向长度 31.778 mm |
| 微带间距 | $0.5\lambda_c=0.4997$ mm | 正交方向长度 3.498 mm |
| 孔径对角线 | 31.970 mm | Rayleigh 距离 2.0456 m |
| 用户距离 | 1.85–2.15 m | 覆盖 Rayleigh 边界附近的近/混合场区域 |
| 路径 | LoS + 2 条镜面反射 | 反射额外路程 0.08–0.25 m，复增益低 6–15 dB |
| DMA 品质因数 | $Q=100$ | 谐振半功率带宽量级约 $f_r/Q\approx3$ GHz |
| 馈线 | $n_{\rm eff}=2.5$，末端功率比例 0.2 | 保留传播相位、色散和沿线衰减 |

$0.2\lambda_c$ 的空间间距不会产生通信波形的码间干扰。ISI 由无线与前端的时域记忆相对 CP 决定；小间距主要带来互耦、相关性和制造约束。Stage 2 的 M2 首轮把互耦视为已校准或暂不建模的因素，后续只在方法主结果通过后作为 M3 失配加入。

## 3. 两级物理模型

同一组路径参数生成两个诊断层级：

- `compatibility`：载频决定平面波空间相位，子载波只承载公共传播时延和扩散损耗；DMA 响应冻结在载频。该层是张量接口的 M0 极限检查。
- `thz_physical`：每个阵元使用精确球面距离，逐频率加入自由空间扩散、用户给定吸收系数表、传播相位、Lorentzian 幅相耦合及有损色散馈线。该层是 Stage 2 的目标模型。

两个层级不是待比较的方法。`compatibility` 只验证算法在其成立假设下没有实现错误；最终方法结论只由 `thz_physical` 层及必要的受控消融给出。

## 4. 统一采集协议

固定每个场景 128 个导频 RE 和每 RE 单位能量，但允许方法使用其结构所需的采集协议。

| 协议 | 配置 | 导频 | 符号 / 切换 | 接入机制 |
|---|---|---:|---:|---|
| F1 `frequency_single_shot_J1_K128` | 1 个分组频率扇区状态 | $1\times128=128$ RE | 1 / 0 | Deshpande 类单次频率解码；Yang、OMP、SBL 的同观测比较 |
| F2 `tensor_structured_J8_K16_RE128` | 8 个循环频率铺排状态 | $8\times16=128$ RE | 8 / 7 | Zhang 类接收侧三阶张量；Yang、OMP、SBL 的同协议参照 |

F2 直接形成“微带输出 × 配置 × 频率”的 $8\times8\times16$ 观测张量，不再把一维频率序列任意 Hankel 化后称作论文张量复现。在 `compatibility` 层，同一微带内状态模式跨微带复用，使 $P$ 路信道的三种展开均满足 rank 不超过 $P$；在目标物理层不强迫该假设成立。

公平性分两层：

1. 同一协议内比较估计器时，所有方法复用完全相同的信道、噪声和观测。
2. 比较完整方法时，允许方法选择 F1 或 F2，但固定真值、阵列、数据频段、可行状态集合、128 导频 RE 和总导频能量，并分别计入训练符号、切换、保护与在线时间。

观测保留实际 DMA 权重能量。单元白噪声经每条微带合并后的方差为 $sigma_e^2\lVert\mathbf w_{s,j,k}\rVert_2^2$，不对每个方法或子载波单独归一化。当前名义输入参考 SNR 设为 55 dB，使初始未优化 DMA 状态的输出处于约 0 dB；这只是后续主 SNR 扫描的中心点。

## 5. 已实现接口

- `src/thz_dma/channels/planar.py`：二维坐标、Rayleigh 距离、精确球面和可分兼容信道。
- `src/thz_dma/surfaces/multistrip.py`：多微带馈线距离、F1/F2 状态及未归一化频变 DMA 权重。
- `src/thz_dma/observations/multistrip.py`：8 RF 输出的干净/带噪观测与输出噪声协方差对角项。
- `src/thz_dma/stage2.py`：公共场景、配对随机化、资源核对和 rank-$P$ 结构验收。
- `src/thz_dma/analysis/stage2.py`：自动完整性检查、紧凑 JSON、CSV、Markdown 和三张固定图。
- `configs/direction1_stage2_unified_2d.toml`：正式 100 场景配置。
- `src/thz_dma/estimators/planar_sparse.py`：精确三维球面字典、OMP、OLS、离网格变量投影和 Oracle 增益拟合。
- `src/thz_dma/evaluation/beamforming.py`：独立可行数据态、模拟态选择、逐微带数字合并和净谱效率。
- `src/thz_dma/stage2a.py`、`src/thz_dma/analysis/stage2a.py`：Stage 2-A 配对实验、自动统计和固定图。
- `configs/direction1_stage2a_classical.toml`：100 场景、三 SNR 的正式传统参照配置。
- `src/thz_dma/estimators/planar_sparse.py`：除固定笛卡尔字典外，支持候选并集字典和经数值等价检查的 CUDA 构建路径；旧配置默认继续使用 NumPy。
- `configs/direction1_stage2a_adaptive_300ghz.toml`：每径粗波束中心已知条件下的 100 场景、55/65 dB 局部网格诊断配置。

## 6. 平台正式结果与含义

正式运行 `stage2_20260909T032102Z` 完成 400/400 个配对条件和 100 个独立场景，状态为 `platform_ready_for_method_adapters`。7 项平台检查全部通过：两协议均为 128 RE、8 RF 输出与 8 条微带一致、CP 覆盖最大超额时延、用户距离跨越 Rayleigh 边界、兼容层 F2 满足 rank-3、绝对权重能量未被归一化、全部输出有限。

100 场景中，F2 的最大展开 rank-3 残差在兼容层的中位数为 $1.09\times10^{-14}$，在目标物理层为 0.512，物理层减兼容层的配对中位差为 0.512，100% 场景均增加。该结果证明旧 Stage 1-D 的高低秩残差不能只归因于 CP 代码：真实球面波和频变硬件确实会破坏简单可分结构。它仍不说明张量估计一定失败，后续应让物理感知张量适配器和强传统方法在相同 F2 观测上直接比较。

55 dB 理想全数字输入参考下，F1 和 F2 的物理输出 SNR 中位数分别为 −0.064 和 1.937 dB。该差异属于状态与协议的实际合并增益，不在方法比较前归一化抹去；Stage 2-A 因此以 45/55/65 dB 形成约低/中/高输出工作区间。

Stage 2-A 的实现和公共比较口径见 [18_方向一Stage2A三维传统参照.md](18_方向一Stage2A三维传统参照.md)；最新固定扇区诊断、失败定位和下一验证见 [19_方向一Stage2A固定扇区诊断与下一验证.md](19_方向一Stage2A固定扇区诊断与下一验证.md)。

## 7. 运行入口

正式平台验证：

```powershell
& 'D:\Program Files\Miniconda\Scripts\conda.exe' run -n ris_td3 python scripts\run_direction1_stage2.py --config configs\direction1_stage2_unified_2d.toml
```

正式 Stage 2-A 传统参照：

```powershell
& 'D:\Program Files\Miniconda\Scripts\conda.exe' run --no-capture-output -n ris_td3 python scripts\run_direction1_stage2a.py --config configs\direction1_stage2a_classical.toml
```

当前应先运行的 300 GHz 条件诊断：

```powershell
& 'D:\Program Files\Miniconda\Scripts\conda.exe' run --no-capture-output -n ris_td3 python scripts\run_direction1_stage2a.py --config configs\direction1_stage2a_adaptive_300ghz.toml --run-id stage2a_adaptive_300ghz_20260911_v1
```

只重新分析已有运行：

```powershell
& 'D:\Program Files\Miniconda\Scripts\conda.exe' run -n ris_td3 python scripts\analyze_direction1_stage2.py --run-dir runs\<run_id>
```

运行目录保存完整配置、环境、随机种子、代码哈希、逐场景指标、执行顺序、平台清单、紧凑分析和三张固定图。

## 8. 下一实现顺序

平台门槛通过后接入适配器，而不再改变场景：

1. 二维球面字典算子、Oracle 与 Yang-style OG-OLS/OMP 已实现；65 dB、网格内、离网格全因子和 oracle 最近格初始化诊断均已完成。读取 19 号第 7–11 节，不重复已有运行。
2. D-024 的 1200 行条件诊断已完成。F2 Yang 在 55/65 dB 的全带均值为 −23.821/−33.734 dB，达标率和三径成功率均为 100%；F1 Yang 的全带均值为 −6.691/−6.754 dB，达标率仅 63%/67%。读取 19 号第 13 节，不重复运行。
3. D-026 的 F1 分解已完成。无噪声单径 OMP/Yang 均为 260/300 达标，40 条失败与约 1.8–2.2°错误粗角峰逐条重合；Yang 三径失败 28/100，其中 23 个含单径失败、5 个是纯多径失败。读取 19 号第 14 节，不重复运行。
4. 暂不设计未知全局几何搜索。先在独立设计场景上固定 F1 状态规则，再用现有 100 个评价场景比较重新设计的单配置和等 128 RE 的少量互补配置；该建议尚未确认。F1 条件恢复稳定后，再确定全局获取和 Stage 2-A 正式配对，并接入 Deshpande 类单次频率解码和 Zhang 类接收侧物理感知张量分解。
5. 统一输出路径/CSI 指标、可行数据态波束损失、扣除训练与切换后的净谱效率和批量 1 时间。
6. 经典方法比较完成且仍存在稳定精度—在线时间缺口后，再把 DMA 扩展方法升级到轻量展开学习；当前不加入 AI/深度学习。
