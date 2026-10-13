# 方向一：Zhou DDPM 与 SURE-DM 的复现计划

日期：2026-10-09。状态：**用户已确认方案；D-037 实现与小规模验证完成，正式性能待验证**。原计划保留；实际范围、结果及正式命令见 [19g 号文档](19g_方向一Zhou_DDPM与SURE_DM实现与最小验证.md)。

后续状态：D-037/D-038 的正式 MIMO 验证与独立测试、D-039 的数字/载频/DMA迁移已完成核验，见 [19i](19i_方向一传统参照与分步场景迁移.md)。本文初稿状态与计划作为历史保留；Linux入口见 [运行指南](../docs/LINUX_RUN.md)。

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：plan
- Origin Date：2026-10-09
- Verification Status：ANALYZED；计划的实现和最小验证已完成，正式学习收益未验证。
- Version Label：zhou_parametric_reproduction_plan_v1

## 1. 目标与已经明确的边界

本轮任务是阅读本地实现、核对论文并提出复现方案。用户确认后才创建新包、配置、入口与测试。

用户已经明确：

- 优先复用本项目的场景、阵列、DMA、传播、观测及评价函数。
- 信道先使用本地参数化模型，不使用 QuaDRiGa 构建新实验数据。
- ULA 可以通过 UPA 的单行几何实现。
- 无法复用的功能放进新的独立 Python 包。

**建议目标：在本地参数化数据上复现论文的 DDPM 先验学习、Algorithm 1 后验恢复和 SURE 两阶段训练，再接入现有宽带 DMA。** 这验证算法实现及其在本地场景的效果。更换信道分布后，不以复现论文 Fig.4/5/13 的具体 NMSE 数值作为验收标准。

首轮复现集合为上述算法、LMMSE/LASSO 基线与 DMA 适配。低位 ADC、其他深度学习基线、LDPC/BER 链及论文原硬件时延表留作后续扩展，不计入首轮完成条件。

主依据是用户提供的正式发表全文：[Zhou@2025，TWC](https://arxiv.org/abs/2408.10501v2)，以及[作者公开代码的固定版本](https://github.com/le-liang/Diffusion-models-for-channel-estimation/tree/2b922440f72ff6943c1cbe5ae85cbd50c421ac83)。本地证据已登记为 `zhou2025generativechannels`，无需重复新增文献记录。arXiv v3 是另一版本，本计划不默认替换正式发表稿。

## 2. 本地代码阅读结论

本地已有一套完整的**单用户宽带接收 DMA**实现。现有完整信道形状为 `(K,S,E)`，分别代表子载波、微带和微带内阵元；它不包含多天线发射端。

当前参考配置来自 [direction1_stage2a_sbl_snr.toml](../configs/direction1_stage2a_sbl_snr.toml)：300 GHz 载频、30 GHz 带宽、128 子载波、8×160 接收 DMA、三径精确球面波及分子吸收。F1 为 1 配置×128 导频频点，F2 为 8 配置×16 导频频点，均占 128 RE。

| 功能 | 已有实现 | 本次使用方式 |
|---|---|---|
| UPA/ULA 几何 | [planar.py](../src/thz_dma/channels/planar.py) 的 planar_dma_positions；[geometry.py](../src/thz_dma/channels/geometry.py) 的 ula_positions | 直接复用。1×N 的平面阵列给出 ULA 几何，统一坐标与阵元顺序 |
| 路径与真实信道 | PlanarPath、sample_stage2_paths、spherical_planar_path_channel、multipath_planar_channel、channel_from_planar_estimate | DMA 阶段直接复用；MIMO 对照只补发射端响应与双端组合 |
| 传播参数 | [atmosphere.py](../src/thz_dma/channels/atmosphere.py) 的自由空间振幅及 AbsorptionTable | 复用损耗和插值规则，不重复计算吸收或路径衰减 |
| DMA 硬件 | [dma.py](../src/thz_dma/surfaces/dma.py)、[multistrip.py](../src/thz_dma/surfaces/multistrip.py) | 直接复用 Lorentzian 响应、馈线传播与未归一化权重 |
| 完整 DMA 平台 | [observations/planar.py](../src/thz_dma/observations/planar.py) 的 build_stage2_platform、stage2_schedule_weights | DMA 阶段直接复用。全数字 MIMO 对照复用其底层组件，不放宽 Stage 2 的协议校验 |
| 带噪观测与白化 | [observations/multistrip.py](../src/thz_dma/observations/multistrip.py)、PlanarDmaTemplateOperator.whiten | 直接复用，保留合并前阵元噪声及真实输出噪声增益 |
| GPU 观测算子 | [estimators/diffusion.py](../src/thz_dma/estimators/diffusion.py) 的 TorchDmaOperator | 复用 forward、adjoint、分块 Gram 与噪声求解，不复用旧采样器 |
| 实虚打包 | 同文件的 pack_channel、unpack_channel | DMA 阶段直接复用；MIMO 单独保持论文的两个实虚通道 |
| MIMO 角域/基线 | [check_zhou_reference_curves.py](../scripts/check_zhou_reference_curves.py) 的 angular、real_operator、lasso，以及实值协方差 LMMSE 求解段 | 提取纯计算部分并复用；新包不依赖该脚本的 HDF5 路径与固定 CUDA 设置 |
| DMA 传统参照 | [planar_baselines.py](../src/thz_dma/estimators/planar_baselines.py)、[linear_channel.py](../src/thz_dma/estimators/linear_channel.py) | 复用 OMP/SBL/Oracle 与 LMMSE 求解。逐径粗中心属于额外信息，单独标注 |
| 指标与数据波束 | [metrics/diagnostics.py](../src/thz_dma/metrics/diagnostics.py)、[metrics/statistics.py](../src/thz_dma/metrics/statistics.py)、[evaluation/beamforming.py](../src/thz_dma/evaluation/beamforming.py) | 复用分域 NMSE、配对置信区间、可行数据状态和净速率 |
| 运行归档 | [scripts/_run_support.py](../scripts/_run_support.py)、已有原子保存与续跑逻辑 | 复用不可覆盖运行目录、配置/源码/数据哈希、逐样本指标与检查点 |

有三个不能通过改形状解决的差异：

1. **接收 UPA 的两条空间轴不是 MIMO 的收发轴。** 不能把 `(S,E)` 改名为 `(Nr,Nt)`，也不能把频率轴当发射天线。论文的 $H\in\mathbb C^{N_r\times N_t}$ 需要真正的发射端阵列响应。
2. **ULA 几何不决定 RF 架构。** 1×N DMA 仍通常只有一个微带输出；论文的全数字接收机需要 Nr 个独立接收通道。数字对照与 DMA 合并采用不同观测接口。
3. **现有 v4/v5 不是论文 DDPM。** 本地网络使用粗中心条件、VE/EDM 参数化、频率—阵元 DFT、Heun 更新及条件协方差；论文使用 VP/DDPM、角域轻量 CNN 和确定性反向均值更新。旧权重不能直接作为新 DDPM 的预训练权重。

已有 D-035/D-036 诊断显示旧学习先验仍落后条件 LMMSE；该结论保留。固定中心残差学习仍是此前建议，本次不将其视为已经采纳的复现方法。

## 3. 论文与公开实现的口径差异

| 项目 | 正式发表稿 | 已检查的作者代码/本地旧实现 | 计划处理 |
|---|---|---|---|
| 归一化 | 训练集所有样本和信道元素的全局平均功率 | 作者 loaders.py 先做逐样本去均值/缩放；本地 DMA 缓存使用全局实分量 RMS | 主流程用训练集全局尺度，明确复元素/实分量的因子 2；不照搬逐样本处理 |
| 扩散噪声 | 实值式(8)中 $\epsilon\sim\mathcal N(0,I)$ | 作者 DiffusionModel 在 complex_data 下将每个实分量噪声乘 $1/\sqrt2$ | 首轮以正文实值方程为准；作者代码口径作为独立对照，不能混用两者的 score 系数 |
| likelihood 输入 | Algorithm 1 在当前 $h_t$ 上计算式(22)/(23) | 公开测试脚本的 use_x0=True 使用去噪估计，并带额外缩放 | 首轮实现正文更新，公开代码改动在验证通过后单独比较 |
| 迭代细化 | 正文另比较 T=100 后半段十次重复更新的增强方案 | 公开测试脚本默认包含后半段十次重复 | 基础 T=100 为第一项；增强版单列并报告真实网络调用次数 |
| 结果选择 | 超参数在验证集选择 | 公开测试脚本保存沿迭代轨迹的 best_nmse | 测试只报告预先冻结的终点/停止规则，完整轨迹用于诊断 |
| SURE 与 ADC | 正文给出 SURE-DM 和低位 ADC 扩展 | 所检查快照未找到对应的完整训练/量化恢复入口 | SURE 需新增；低位 ADC 留在主流程通过后的扩展范围 |

## 4. 拟复现的计算链

### 4.1 参数化数字 MIMO 对照

保留论文 $N_r=16,N_t=64$、$N_p=64/38$ 的观测规模，用两个单行 UPA 表示两端 ULA，半波长间距。**建议对照起步使用本地可分平面波兼容层的三径机制和 300 GHz 单频，作为 M0 算法控制。** 载频及路径分布与原文不同，会写入配置；40 GHz 原场景对照不作为首轮必要项。

新增的双端组合采用

$$
H(f_c)=\sum_{\ell=1}^{L} g_\ell(f_c)
a_R(f_c,\Omega_{R,\ell})a_T(f_c,\Omega_{T,\ell})^H.
$$

接收端路径分布复用本地配置。发射端角度作为显式新增随机量，建议首轮采用声明范围内独立均匀分布；总路径长度、两端角度与复增益随场景保存。$g_\ell$ 只应用一次公共传播损耗与路径相位，不将两个已有带衰减的信道向量直接相乘。本块是可分参数化 MIMO 控制，不声称已实现双端精确近场传播；目标 DMA 真值仍使用当前球面波模型。

采用固定随机种子的单位模 QPSK 导频。同一 $N_p$ 的所有估计器共享 P；全局训练归一化后记录名义 SNR 和实际平均接收功率。若采用论文名义横轴，则复噪声方差为 $\nu=N_t10^{-\rho_{\rm dB}/10}$，实分量方差为 $\nu/2$。

复用 angular 的等价表示：

$$
H_{\rm ad}=F_R^H H F_T,\qquad
Y_{\rm ad}=F_R^H Y=H_{\rm ad}(F_T^H P)+N_{\rm ad}.
$$

按列向量化后，测量矩阵为 $(F_T^H P)^T\otimes I_{N_r}$。实数向量按“全部实部、全部虚部”排列；CNN 使用形状 `(batch,2,Nr,Nt)`。所有展平顺序通过同一无噪声观测检查。

### 4.2 干净数据 DDPM

复现式(8)/(12)的 VP 前向加噪与噪声预测 MSE；训练只读取信道和 t，不输入在线观测、SNR 或导频矩阵。

网络按 Appendix I/Fig.14 的轻量结构：两层 3×3 卷积、时间正弦嵌入及缩放/偏置注入、三层输出卷积，最大特征通道 64。第一版不使用旧 U-Net、EDM 预条件化、物理子空间投影或条件高斯初始化。

反向阶段以 Algorithm 1 为准：

$$
\widehat h_{t-1}
=\frac{1}{\sqrt{\alpha_t}}
\left[
h_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(h_t,t)
+\;s(1-\alpha_t)\ell_t
\right].
$$

$\ell_t$ 按式(22)/(23)的噪声扰动似然计算，使用当前 $h_t$；初始化为标准实高斯，每一步采用确定性更新。该似然仍是原文的无信息先验近似，不称为精确后验。

首轮 T=100，线性噪声计划参考已核对的作者设置：$\beta_1\approx10^{-4}$、$\beta_T=0.1$。Adam、学习率 $10^{-4}$、batch 128 为起点，完整训练参考 500 epochs。$s$ 的候选预算和选择规则在验证前写入配置。T=1000、后半段细化及作者代码引导作为后续消融。

### 4.3 SURE-DM

第一项验证严格使用式(31)的**完整含噪信道**：

$$
z=h+w,\qquad w\sim\mathcal N(0,\sigma_w^2 I),\qquad
\sigma_w^2\in\{0.1,1\}.
$$

这里 $\sigma_w^2$ 是每个实分量的方差，不能与复噪声方差混淆。合成完整信道噪声与在线导频观测噪声独立。

顺序训练两个独立参数的同结构网络：

1. 固定噪声等级下训练 SURE 去噪器 $\theta_1$，不使用干净标签。
2. 冻结 $\theta_1$，输出并保存去噪训练样本。
3. 使用去噪样本重新训练 DDPM $\theta_2$，不将 $\theta_1$ 直接当作完整 DDPM。
4. 在共同在线观测上比较 Clean/Oracle DM、Naive DM、SURE-DM，以及干净/含噪协方差 LMMSE。

原文训练预算为 SURE 100 epochs、其后 DM 500 epochs。缩小试验使用较小数据与训练预算并明确标记；SURE 阶段只能用可访问的含噪数据和已知噪声参数计算训练目标。若归一化也要求仅依赖含噪样本，采用训练样本二阶矩减已知噪声功率计算尺度，保存有限样本误差；不能隐式读取干净训练集功率。共同基线使用同一已登记尺度。

**SURE 换元需要先核验。** 正式稿式(34)使用 $q=\sqrt a\,z$，$a=1/(1+\sigma_w^2)$，而式(35)将散度定义为对 q 求导。令 $g(z)=f(\sqrt a\,z)$，链式法则给出

$$
\operatorname{div}_{z}g(z)=\sqrt a\,\operatorname{div}_{q}f(q).
$$

因此按这些变量定义，严格的风险估计为

$$
\operatorname{SURE}
=\left\lVert f(q)-\frac{q}{\sqrt a}\right\rVert_2^2
+2\sigma_w^2\sqrt a\,\operatorname{div}_{q}f(q)
-d\sigma_w^2,
$$

其中 d 是实值信道维数；最后一项可在优化时删除，但风险无偏性自检必须保留。正文式(34)未显示 $\sqrt a$，本轮推导发现这处尺度不一致。计划先用线性去噪器检查“真实 MSE 与平均 SURE”，再分别登记**原文式(34)版**和**严格换元版**，不无说明地改写论文方法。

散度采用原文式(36)的 Monte Carlo 有限差分，$\varepsilon=10^{-5}$ 为参考。先用双精度和已知 Jacobian 检查，再核验网络的梯度和差分精度；SURE 差分不默认使用混合精度。连续目标 a 与离散日程 $\bar\alpha_{t_w}$ 的差额另记录，不将最近步匹配写成精确相等。

### 4.4 接回真实宽带 DMA

复用现有平台、三径球面波真值、F1/F2、相同观测和数据波束。完整输出仍为 `(K,S,E)`。

DMA 使用已有 pack/unpack 保持 `(batch,2S,E,K)` 布局。拟对每个子载波的两个空间轴 S/E 做单位 DFT，再按该布局打包，保留完整频率轴；这与旧 fourier_channel 的频率—微带内阵元 DFT不同，需新建少量变换适配。该单位变换保留全部 CSI；在近场及非半波长间距下，不保证角域稀疏，也不将 DFT 索引直接解释为真实路径角度。CNN 输入通道变为 2S，权重须重新训练，不能沿用 2 通道 MIMO 权重。

在角域调用物理算子时先逆变换回阵元域；伴随按相反次序变换。复用 TorchDmaOperator 的分块求解，不构造全带稠密 A 或完整信道协方差。单位变换不改变观测 Gram。正文实扩散噪声方差为 1 时，对应复扩散噪声方差为 2，复域求解与实域式(22)必须保持一致。

主 DDPM 先不输入粗中心，与同一训练分布的全局样本统计 LMMSE 比较。现有 OMP/SBL/条件 MC-LMMSE 使用逐径粗中心，只作为信息增强的单列参照；若后续增加条件 DDPM，必须明确标为论文算法的条件扩展，并让相应基线共享条件。

SURE 接入这一块时，先用参数化完整 CSI 加独立 AWGN 构造离线训练样本，在线仍读取真实 DMA 导频。**不把压缩 DMA 的 LMMSE 输出直接当作式(31)的数据。** 其误差一般具有偏置、相关性和低秩结构；观测白化不能自动将信道估计误差变成独立白高斯。只靠 DMA 压缩观测训练的广义 SURE 需要另立模型，不在首轮假定成立。

## 5. 分步实施与验收

以下全部为建议，确认后才执行。

| 步骤 | 工作与建议规模 | 验收 |
|---|---|---|
| P0：代数与接口 | 小尺寸数字 MIMO、1×N ULA、少频点 DMA；无需网络训练 | DFT 逆变换/能量、列向量化、复实等效、物理 forward/adjoint、VP 方差、式(22)梯度与 SURE 风险自检通过 |
| P1：干净 DDPM | 参数化 16×64 数字 MIMO；入口检查 256/32 个训练/验证场景，学习试验建议 4096/256/100 个独立训练/验证/测试场景；Np=64/38 | LS 极限与 LMMSE/LASSO 正常；保存分时刻去噪误差和最终在线 NMSE，先在验证集冻结 s 与终点规则 |
| P2：SURE-DM | 同一场景生成两个预设含噪数据等级，按去噪器→新 DDPM 顺序训练 | SURE 风险尺度与可训练梯度通过；三类 DM 共享在线观测，比较收益及相对 Clean DM 的剩余差距 |
| P3：宽带 DMA | 先小规模适配，再使用原 8×160、128 频点、F1/F2；优先读取已有原始 CSI 缓存及其记录，不重新定义真值分布 | 物理观测与旧入口一致；输出原单位完整 CSI；报告全带/导频/未观测域、净速率与 batch=1 在线时间 |

算法曲线建议沿用 −10:5:30 dB 接收 SNR 诊断。论文名义 SNR、训练集参考接收 SNR和实测 RF 输出 SNR分别登记。跨协议物理收益继续采用共同输入噪声/发射能量的项目口径；不能用各协议独立匹配接收 SNR的图证明 F1/F2 资源优势。

入口和学习筛查先用一个训练种子。主结果冻结配置后至少三个训练种子，保存逐场景配对指标及 95% CI；独立测试不用于模型、引导强度或停止步选择。4096 样本结果是缩小规模验证，100000 样本/500 epochs 的原文训练规模在前述验证通过后再评估预算。

实施检查应包括：

- 单行 UPA 与 ULA 的坐标、距离和观测一致；DMA 与数字 RF 数目分别正确。
- 正交完整导频的 LS 误差符合已知噪声尺度；Np=38 时保留欠定问题。
- 直接求解、SVD/分块求解及实值/复值 likelihood 梯度等价。论文近似似然与真实高斯后验的偏差另报，不要求近似方法必然等于 LMMSE。
- SURE 的线性控制在蒙特卡洛标准误差范围内匹配真实 MSE；有限差分和梯度均为有限值。
- 数据划分、训练尺度及不同训练模式可访问信息正确；恢复原单位后与物理算子一致。
- 新入口具有检查点、随机状态、配置/数据指纹和中断续跑核验。

如果训练场景的去噪也失败，先检查尺度、日程、目标和网络能力，再扩大数据/阵列。如果先验去噪正常而在线恢复失败，比较当前状态引导、作者引导、重复更新及近似似然误差。对未形成学习收益的设置保存实际结果，不通过改 SNR 横轴或挑选测试步数修饰曲线。

## 6. 新包与受影响文件

建议新建独立包 **src/zhou2025_repro/**，由当前 setuptools 的 src 包发现机制一起安装。按必要功能划分：

| 拟建文件 | 职责 |
|---|---|
| `__init__.py` | 最小包入口 |
| `mimo.py` | 参数化双端 MIMO 组合、数字导频观测及测量适配；底层几何/传播导入 thz_dma |
| `data.py` | MIMO/UPA 角域、实虚布局、全局尺度和两种数据形状的最小适配 |
| `baselines.py` | 从现有参考脚本提取 LASSO、实等效/LMMSE 的纯计算部分，保持行为 |
| `ddpm.py` | 轻量时间条件 CNN、VP 日程、噪声预测损失与 Algorithm 1 |
| `sure.py` | 单噪声去噪器、风险目标、散度和去噪数据输出 |

建议新增一个分阶段入口 `scripts/run_zhou2025_repro.py`，两个独立场景配置及一个必要的数值测试文件。共享记录、保存和续跑能力复用已有实现，不复制整个旧运行器，也不建立通用平台框架。

若提取已有脚本函数，影响范围为：

- `scripts/check_zhou_reference_curves.py`：改为导入共享函数，保留原名称和运行行为。
- `scripts/diagnose_zhou_reference_pilots.py`、`scripts/audit_direction1_snr_curves.py` 及 `tests/test_zhou_reference.py`：核对现有导入链和结果一致性。
- 新入口、新配置和新数值检查，以及 `README.md` / 研究目录导航。
- 正式确认涉及新数据/模型口径时，先在 99 号登记，再定向同步公共模型、90 号及实验索引；已有文献证据与历史运行继续保留。

当前环境已确认安装 NumPy、PyTorch、SciPy、Pandas、Matplotlib，首轮不需 QuaDRiGa、MATLAB、HDF5 数据加载器或新增训练框架。PyTorch 当前为 2.11.0+cu128；每次运行仍需归档实际环境。现有 pyproject.toml 尚未声明已被共享估计器使用的 SciPy，实施涉及该导入链时补齐依赖声明，不为本轮分析安装或升级库。

单个 16×64 MIMO 样本有 1024 个复数，complex64 占 8 KiB；10 万完整样本约 0.82 GB。原 128×8×160 DMA 样本约 1.31 MB，2048 样本约 2.68 GB，因此复用缓存/分块读取。运行时间和显存峰值尚未实测，不承诺训练完成时间。

## 7. 原计划的待确认内容，历史保留

以下为最初规划阶段的请求。用户随后确认按该思路和方法复现，已登记 D-037；不再作为待批准事项。

建议确认以下两点后进入实现：

1. 是否采用“本地参数化数字 MIMO 控制 → 干净 DDPM → SURE-DM → 现有宽带 DMA”的顺序，并保留原平台为最终目标；数字控制首轮使用 300 GHz、可分三径模型。
2. 第一批是否只实现 P0/P1 的最小新包与可运行验证，P2/P3 按通过情况继续搭建；SURE 阶段同时核验原文式(34)与严格换元版。

以上尚未成为已确认研究决策。按用户要求，本轮止于源码/原文分析及本计划。已跳过实现代码生成、依赖安装、模型训练、性能实验与历史运行重算；更换数据分布后的算法收益、SURE 风险尺度及宽带适配均待实际验证。

## 8. D-037 执行状态

独立包 zhou2025_repro、六阶段入口、MIMO/DMA 适配、SURE 两种目标和两档噪声已实现。35 个阶段运行完成，三个共同评价共 296 行；27 项检查通过，8 项恢复复算与原始指标一致。100 步完整网络的 MIMO 短验证未超过 LASSO，SURE 去噪仅约 0.05–0.08 dB 改善；DMA 仅通过连通性，恢复误差仍为正 dB。正式训练、独立测试和性能复现尚未完成。详见 [19g 号实现与结果](19g_方向一Zhou_DDPM与SURE_DM实现与最小验证.md)。
