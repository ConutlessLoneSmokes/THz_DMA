# D-041 无吸收对照：预声明与 Linux 入口

2026-10-14后续：四组无噪声正式结果保留；用户授权J192至J1扫描及固定128 RE的1/2/4/8/16配置比较，新F3=16×8。新入口复用本页数据/模型，不重新训练；原运行源码保持，扫描命令、门槛和对照见[D041_SWEEP_RUN.md](D041_SWEEP_RUN.md)。带噪仍未运行。

本轮交付为实现与小规模检查；匹配先验的正式训练和评价由用户执行。下列预算与门槛在新运行前固定，结果只作失败原因诊断。既有含吸收缓存、模型和 D-038/D-039 记录保持原样。

2026-10-14正式宽带续记：用户完成两带宽无噪声14运行，7920条评价/266640条轨迹、路径配对和20个固定宽带GPU端点复核通过。宽带数字/J192约−70.52 dB，J32/J8/J1为−5.19/−0.20/−0.015 dB；宽带数值敏感性通过，原窄带超限保留。少配置补全问题仍未解决，频率—配置等资源分配尚未评价，带噪4阶段未运行。驱动/NVML现均595.99.02，实算正常。本次复用入口、未修改运行代码或原报告。见 [四组检查报告](../research_route/experiment_results/2026-10-14/d041_noabs_20261013_v2_noiseless_review/audit_report.md)；下文预算与预声明保持，已完成阶段不重跑。

2026-10-13正式续记：用户完成`d041_noabs_20261013_v2`窄带无噪声7阶段，256000次训练更新、3960条评价和133320条轨迹核验通过。数字/J192约−66.87 dB，J32/J8/J1为−4.05/−0.13/+0.09 dB；充分配置部分场景数值方差敏感性超原0.1 dB门槛，保留数值限制。20个固定GPU端点复算一致，NVML已定位为驱动/库不匹配，CUDA计算仍可用。检查没有修改运行代码、重训或重选参数；宽带及带噪尚未运行。报告见 [窄带正式核验](../research_route/experiment_results/2026-10-13/d041_noabs_20261013_v2_narrow_review/audit_report.md)。先核查数值限制与验证引导边界，不重跑已完成窄带，不覆盖已存在的自动报告；下面预声明保持。

## 1. 固定与改变

中心频率 300 GHz、单发射端、8×160 阵列固定。阵元间距 0.00019986163866666668 m，微带间距 0.0004996540966666667 m，等于原中心频率的 0.2/0.5 波长。各带宽的场景 ID、三径几何和复增益完全配对，均由 `sample_stage2_paths` 生成；不采用原含吸收缓存的自适应粗中心微扰。因此新数据是匹配的新分布，不能将新旧差值单独解释为吸收效应。

| 控制 | 固定 | 改变 | 用途 |
|---|---|---|---|
| A/B | 单频 CSI、训练数据与 Clean DDPM | 全阵元数字接收 / DMA 合并 | 隔离 DMA 恢复链路 |
| C/D | 30 GHz、128 点全频 CSI、训练数据与 Clean DDPM | 全阵元数字接收 / DMA 合并 | 隔离宽带 DMA 的恢复问题 |
| A/C、B/D | 米制阵列、中心频率、配对路径、状态库、网络和更新预算 | 单频 / 宽带；必须匹配各自先验 | 区分带宽作用及联合因素 |
| DMA 配置曲线 | CSI、状态库、所有频点、单位 RE 导频能量 | 状态库前缀 J=192→32→8→1 | 区分充分观测与少配置恢复 |
| 无噪声/带噪 | CSI、模型、状态和固定噪声种子 | 无噪声 / 训练均值标定的白化 SNR | 区分结构/恢复问题与噪声影响 |

窄带仅在 300 GHz 计算一次，形状 `(1,8,160)`；宽带沿用原中心对称的 128 点 OFDM 栅格，形状 `(128,8,160)`。原偶数栅格不包含正好 300 GHz，不能把其中一个点冒充中心频率。表示仍为二维空间酉 DFT 后打包的实张量，输入分别为 `(16,160,1)` 和 `(16,160,128)`。CNN 通道数、宽度、扩散日程与参数量相同；窄带频率轴长度为 1，卷积在该轴读取零填充，不能称为相同有效频率感受野。

固定种子 2026101344 在 285–315 GHz 可行调谐范围内生成 192 个配置；同微带内图案复制至 8 条微带，少配置只取同一库前缀。保留 Q=100、有效折射率 2.5、馈线末端功率比 0.2 和真实 Lorentzian 频响。没有按信道/评价结果挑状态。首轮两种带宽都观测所有频点，不加入 F2 的缺频效应。数字 1 次接收、DMA J 次接收不是等硬件或等成本比较：每次保存 RF 数、复观测数、秩、RE、能量、符号和切换。

## 2. 数据、模型与预算

- 正式训练/验证/测试场景数为 2048/256/100，分割种子分别为 2026101341/42/43。按所有训练 CSI 的复 RMS 归一化，不按样本归一化。窄/宽带的路径参数记录逐字节相同，各自 CSI 与尺度独立生成。
- 两个 Clean DDPM 各使用 width=64、time_dim=16、100 个 VP 步、beta=0.0001→0.1、batch=4、500 epochs，即各 256000 次更新，与旧宽带预算相同。只采用一个固定训练种子，参数量和人工去噪分时刻误差随训练保存；同一带宽全部前端只加载该带宽这一个模型。
- 在线引导候选 `[0.5,1,2,4]`，refinement=1，使用 32 个验证场景的最终 CSI 线性 NMSE 选参。每个前端/配置/噪声条件分别冻结；test 只读相应 validation 运行的冻结记录。保存轨迹但不按真值选中间步骤。
- 带噪白化 SNR 为 0/10/20/30 dB。每个算子遍历所有训练 CSI 标定平均白化功率 $P_A$，再令复数噪声方差为 $v_A=P_A/10^{s/10}$，实分量方差减半。DMA 同一配置前缀、场景跨 J/SNR 共用标准噪声；同带宽所有前端/配置/引导/方差使用相同后验初始化。保存目标、逐例期望及实现 SNR、物理阵元/RF 噪声和输入参考。不同前端匹配接收 SNR 是诊断干预。
- 无噪声观测严格不添加噪声；数值似然复方差分别取归一化单位的 `1e-8` 与 `1e-10`，逐例记录并检查两者 NMSE 差是否超过 0.1 dB 或跨越 −10 dB 描述性恢复门槛。该方差不是物理噪声。
- 同观测的伪逆 LS 检查任意 CSI 的线性可恢复性；全秩 DMA 若该参照通过而 DDPM 失败，信息缺失不能解释该控制的失败。真几何 LS 只读取几何拟合复增益，始终标为额外信息参照。首轮无全局 LMMSE、SURE、非规则选频及多种子。

正式新 CSI 数组约 2.96 GiB（另加模型、来源和轨迹）；不依赖尚未传输的旧大数据。两张 3090 可分别放置窄/宽带任务，每个进程只用一张卡。必须先通过 CUDA 和下节检查；当前环境的 CPU 小检查不代表 GPU 已准备好。

具体地，$A$ 包含真实合并与已知噪声白化，$m_A$ 是复观测数，$\rho$ 是该带宽训练集复RMS：

$$
P_A=\frac{1}{N_{\rm tr}m_A}\sum_{i\in\mathcal D_{\rm tr}}
\left\|A\left(\frac{\mathbf h_i}{\rho}\right)\right\|_2^2,
\qquad
\mathbf y=A\left(\frac{\mathbf h}{\rho}\right)+\sqrt{v_A}\,\boldsymbol\zeta,
\qquad \boldsymbol\zeta\sim\mathcal{CN}(\mathbf0,\mathbf I).
$$

物理阵元复噪声方差为 $\rho^2v_A$，RF额外噪声按原配置比例加在合并后；真实损耗仍在$A$及相对输出方差中。无噪声时观测中的噪声严格为零，两个正数方差仅用于似然求解。

## 3. 小检查与停止条件

预声明：每个频率充分配置秩为 160，最大条件数 ≤1e4；无噪声数字/充分配置伪逆 CSI NMSE ≤1e-9、观测相对残差 ≤1e-10；complex128 伴随相对差 ≤1e-10；逐 RF 白化噪声功率相对偏差 ≤0.06。各检查保存实际数值，失败时检查入口报错，先修复再开展正式训练。少配置伪逆残余主要在零空间不构成检查失败。

`--smoke` 保持 8×160 和宽带全部 128 点，仅将样本改为 8/4/4、网络宽度 16、扩散 8 步、训练 4 次更新、评价 2 场景、引导 2 候选、SNR 0/20 dB。它核对接口、数值与冻结链路，不能验证匹配先验训练充分或正式性能。

## 4. 指标与判别

`metrics_raw.csv` 保存最终 CSI NMSE、带噪及干净观测残差、预测能量比、归一化复相关及实部方向一致性、行空间/零空间 NMSE（共用总真值能量分母）、分解一致性和噪声/资源信息。`trajectories.csv` 保存初始化及每一步；状态除以当前 sqrt(alpha_bar) 后记录，显式标注 VP 噪声层级，只有 t=0 为最终 CSI。轨迹诊断计入记录的在线时长，不能把该时长作为不含评价开销的部署速度。

SVD精确计算原算法的线性高斯似然近似，不等于真实物理先验下的精确边缘似然，因此数值梯度检查通过仍不能排除引导近似与学习先验不匹配。CSV中的guidance/model_run_id是同观测组DDPM的冻结上下文，伪逆与真几何参照不使用这些参数。

| 结果 | 支持的判断与下一步 |
|---|---|
| A 通过；全秩 B 的 LS 通过但 DDPM 失败 | 核查 DMA 似然/尺度及先验—恢复耦合；不能归咎于该控制的信息缺失 |
| B 仅在减少配置后失败，行误差小/零误差大 | 完整 CSI 的补全失败；继续区分先验不够与合法信道歧义 |
| A/B 通过而 C 失败 | 检查宽带训练/表示和分时刻去噪能力；不直接归因于 DMA |
| B/C 通过而 D 失败 | 指向宽带与 DMA 的联合条件；后续再考虑冻结中心频率权重消融 |
| 无噪声通过而匹配接收 SNR 后失败 | 噪声敏感性、白化或引导强度成为重点 |
| 观测残差小、能量近零、相关低、零空间误差大 | 观测拟合未补回完整 CSI；结合训练去噪 profile 和轨迹定位发生阶段 |
| 只有真几何 LS 通过 | 只证明额外几何信息下的条件恢复，仍不能排除未知几何歧义 |
| 数值方差改变结论 | 先处理无噪声数值敏感性；暂不作物理归因 |

## 5. 完整 Linux 命令

在 `/home/lsy/THz_DMA` 使用本轮工作树。以下安装步骤仅在项目环境尚未建立时执行，保留其他项目的 EIT_test 环境。当前会话没有改动驱动、SSH 或代理配置。若下载需要代理，先从 Windows 执行 [LINUX_RUN.md 第5节](LINUX_RUN.md#5-现有服务器与-ssh-代理隧道) 的原 SSH 转发命令，再在服务器终端执行已有 `proxy_on 28090`；不重写配置文件。

```bash
cd /home/lsy/THz_DMA
conda create -n thz_dma python=3.10 pip -y
conda activate thz_dma
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-linux.txt -e .
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/d041_matplotlib
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count()); assert torch.cuda.is_available()'
python -m unittest discover -s tests -v
```

已建立环境时从 `conda activate thz_dma` 开始，省略创建与安装。CUDA 检查失败先处理已有驱动/库不匹配，不启动下面 GPU 阶段。CPU 最小链路可独立执行，运行号必须未使用：

```bash
D041_PREFIX=d041_linux_smoke_20261013_v1 D041_DEVICE=cpu D041_SMOKE=1 \
  bash scripts/run_d041_linux.sh all
```

用户重申排除计划后，入口改为先窄带7阶段，再宽带7阶段，先形成无噪声四组对照；带噪4阶段后置。每个带宽的无噪声次序为新数据→代数检查→匹配Clean训练→充分配置validation/test→少配置validation/test。无噪声也经过验证选参与冻结测试。以下没有设置 `D041_SMOKE=1`，会执行声明的较大预算，继续由用户启动。代码更新后使用新前缀，旧小运行和旧源码快照保留：

```bash
export D041_PREFIX=d041_noabs_20261013_v2
CUDA_VISIBLE_DEVICES=0 bash scripts/run_d041_linux.sh narrow-noiseless
```

第一条完成后停在窄带报告 `results/${D041_PREFIX}_narrow_noiseless/README.md`。先看数字与J192的DDPM和同观测LS，再看J32/8/1的行/零空间误差与恢复轨迹。**J192的线性参照通过不代表匹配DDPM已通过；原窄带MIMO不能代替这一组。** 窄带报告不依赖宽带或带噪任务存在。

继续宽带无噪声后，自动核验四组配对控制，带噪尚未启动：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/run_d041_linux.sh wide-noiseless
```

四组报告位于 `results/${D041_PREFIX}_noiseless/README.md`。`noiseless_controls.csv`只读取冻结test最终端点，保守报告两个预声明数值方差下的最坏NMSE、−10 dB门槛通过比例、预测能量、相关性、行/零空间误差；`audit.json`同时保存匹配模型的更新预算和各扩散时刻人工去噪profile。若数值方差敏感，先处理该问题；若只是smoke模型，报告不作正式学习失败归因。完整数据/预算仍是必要的判断条件。

两带宽无噪声任务均完整后，再执行匹配白化SNR的带噪阶段。入口会检查无噪声记录、配置和源码，缺项时在新评价前退出：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_d041_linux.sh noisy
```

也可将两个带噪任务分别放到两张卡，然后单独汇总：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_d041_linux.sh narrow-noisy
CUDA_VISIBLE_DEVICES=1 bash scripts/run_d041_linux.sh wide-noisy
python scripts/analyze_d041_diagnostic.py --prefix "$D041_PREFIX" --scope all
```

在已经存在的运行上只生成部分报告也可直接执行以下核验命令，不会重新训练或评价：

```bash
python scripts/analyze_d041_diagnostic.py --prefix "$D041_PREFIX" --scope narrow-noiseless
python scripts/analyze_d041_diagnostic.py --prefix "$D041_PREFIX" --scope noiseless
```

每个任务指定进程内 `cuda:0`；CUDA_VISIBLE_DEVICES选择物理卡，不把显存相加。`noiseless`可串行完成两种带宽的无噪声阶段；`all`按“窄带无噪声→宽带无噪声→无噪声核验→两种带宽带噪→完整核验”执行。`narrow`/`wide`现在分别是`narrow-noiseless`/`wide-noiseless`的别名，不再自动进入带噪。推荐分段执行以便先查看关键窄带结果；不要重复使用已完成运行号。各阶段保存独立来源、配置、状态、源码ZIP和原子检查点。

| 每个带宽的评价 | validation 行数 | test 行数 | 算子与条件 |
|---|---:|---:|---|
| full-noiseless | 384 | 1200 | 数字、J192；2 个数值方差；3 方法 |
| reduced-noiseless | 576 | 1800 | J32/J8/J1；2 个数值方差；3 方法 |
| noisy | 1920 | 6000 | 数字、4 个 DMA 配置数；4 档 SNR；3 方法 |
| 两种带宽合计 | 5760 | 18000 | **23760 行，另有 799920 条恢复轨迹** |

总计18个运行目录与1份完整汇总，分段执行另生成窄带和四组无噪声的部分报告；两个代数检查各10行参照。不重跑旧运行，不调用非规则选频、LMMSE扫描或SURE。完整汇总位于 `results/$D041_PREFIX/`，运行位于 `runs/${D041_PREFIX}_{nb,wb}_...`。

第一轮窄带无噪声为7个运行、3960条评价/133320条轨迹；补齐宽带无噪声后累计14个运行、7920条评价/266640条轨迹。部分核验只重读这些结果，不增加评价行数；随后带噪4个运行才达到上表的18个运行和23760条评价。

若中断且代码、配置、设备与依赖完全相同，按原 phase 设置 `D041_RESUME=1`。脚本校验已完成任务的源码后跳过它们，对未完成任务续跑；CPU/GPU、smoke/正式或修改代码不能共用同一前缀。独立入口也可按原完整参数追加 `--resume`。例如恢复宽带充分配置的 validation：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_d041_diagnostic.py evaluate \
  --band wide --stage full-noiseless --device cuda:0 \
  --run-id "${D041_PREFIX}_wb_full_val" \
  --data-run-id "${D041_PREFIX}_wb_data" \
  --train-run-id "${D041_PREFIX}_wb_clean" --split validation --resume
```

已完成运行不可加 `--resume` 重做；重新试验使用新前缀。汇总核验检查参数配对、全部行数、训练预算、模型共享、冻结来源、SNR 公式和最终轨迹一致性，并输出按场景的区间与窄/宽带配对差。区间不覆盖训练种子不确定度。

## 6. 已完成的小检查与证据边界

本轮 CPU 环境为 Python 3.11.15、PyTorch 2.11.0+cu128、NumPy 2.4.6、SciPy 1.17.1、pandas 2.3.3。51 项检查通过；环境缺少的 h5py 3.16.0 只安装到 `/tmp/thz_dma_d041_testdeps` 后通过 PYTHONPATH 供参考检查读取，没有修改现有 Conda 环境。正式运行仍按第5节 Python3.10 固定环境。

分段入口续记：增加6项阶段顺序/诊断解释检查，当前共57项通过。旧v3小记录的窄带部分核验为120行/360条轨迹，四组无噪声核验为240行/720条轨迹，均通过；这些是对既有结果的只读分析，不是新训练。原完整v3档案不改写，部分报告存到新目录。

更新后的`all`另在新前缀`d041_noabs_staged_smoke_20261013_v1`完成18个CPU小运行。14个无噪声运行及240行/720条轨迹的部分核验先完成，再进入4个带噪运行；完整480行/1440条轨迹核验通过。两个模型仍仅各4次更新，未执行正式训练。该流程续记与只读部分报告见 [分段核验档案](../research_route/experiment_results/2026-10-13/d041_staged_checks_20261013/README.md)。

充分配置的无噪声检查：窄带/宽带秩为 1280/163840，最大条件数约 62.8/108，complex128 伪逆 NMSE <1e-27；数值伴随、实际前向与白化噪声检查通过。充分观测的线性恢复链路已成立。少配置行空间/零空间分解通过，只说明对任意 CSI 存在缺失部分，不等于三径合法信道不可辨识。

最新 `d041_noabs_smoke_20261013_v3` 的18个运行全部完成：480条评价、1440条轨迹、20条代数参照。数值方差敏感性均通过0.1 dB门槛（最大约0.03335 dB）；错误带宽先验和错误冻结阶段均被拒绝，续跑入口校验并跳过18个完成运行。561个历史快照原文件的字节哈希全部保持。

CPU 打包非连续布局的宽带二维 FFT 在旧调用方式下曾原生崩溃。先将张量转成连续布局后，完整 `(128,8,160)` 的往返误差与酉能量检查通过，DDPM 数学表达未改变。中止运行和修复前检查保留，不把该 Linux 后端问题当作 D-039 约0 dB 的已证原因。

小训练的恢复结果只用来验收生成→训练→验证选参→测试冻结→轨迹核对流程；两个匹配先验仅各4次更新。正式先验/引导失败环节仍待用户长运行，不能从这些小指标作唯一归因。最新紧凑证据链接见 [19i 第14节](../research_route/19i_方向一传统参照与分步场景迁移.md#14-d-041最小实现小检查与linux交付)。
