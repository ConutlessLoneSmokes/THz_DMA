# Linux 实验运行

2026-10-14配置扫描入口见[D041_SWEEP_RUN.md](D041_SWEEP_RUN.md)：复用已完成D-041数据/模型，两卡分别运行窄/宽带J曲线，固定128 RE另行评价。66项检查和6阶段CPU小链路完成；不重新安装环境或修改驱动/SSH/代理配置，正式扫描由用户启动。

2026-10-14更新：用户补齐D-041宽带7阶段，两带宽无噪声14运行已核验。当前加载驱动/NVML均595.99.02，NVML和nvidia-smi返回0、CUDA及20个固定宽带恢复端点复算正常，初始化警告未再出现。宽带归档内核7.0.0-38，与窄带7.0.0-31的差异保留；Python3.10.21/PyTorch2.11.0+cu128及直接依赖一致。报告见 [四组核验](../research_route/experiment_results/2026-10-14/d041_noabs_20261013_v2_noiseless_review/audit_report.md)。本次检查没有修改驱动、环境或SSH/代理配置，下列旧版本不匹配说明属于此前状态。

2026-10-13正式运行核验：Python3.10.21 `thz_dma`环境已完成D-041窄带7阶段，训练/冻结评价完整，20个GPU恢复端点复算一致。NVML返回18，加载内核驱动595.91.07与库595.99.02不匹配；已安装模块也是595.99.02。沙盒外CUDA可识别两张RTX3090并完成实际运算，不能以沙盒内探测失败判断正式运行退回CPU。既有实验无需因警告重跑；建议空闲时重启加载已安装模块，再检查版本一致、nvidia-smi及CUDA实际运算。未执行重启、驱动/环境安装或SSH/代理改动。证据见 [窄带核验报告](../research_route/experiment_results/2026-10-13/d041_noabs_20261013_v2_narrow_review/audit_report.md#5-nvml警告)。以下同日较早的准备状态和历史传输记录保留，不表示当前正式窄带仍待安装或运行。

2026-10-13 D-041实施续记：新无吸收同阵列对照的完整命令见 [D041_RUN.md](D041_RUN.md)。51项CPU检查和18阶段小链路已完成，正式Clean DDPM训练/评价仍由用户启动；D-041新生成数据，不依赖旧3.47 GiB输入传输。当前会话CUDA不可用，先校验项目环境与驱动。下文D-040/D-042的准备和历史传输说明保留。

用户重申计划后的最新入口为先窄带无噪声、再宽带无噪声、最后带噪，新增部分核验与6项顺序/解释检查（当前57项）。请使用 [最新分段命令](D041_RUN.md#5-完整-linux-命令) 和新前缀；`narrow`/`wide`不再自动进入带噪。旧记录及源码快照保留。

本次迁移对应 D-040。研究代码保持现有目录与入口，只修复清单路径读取和运行配置的路径记录，并增加设备覆盖参数。正式算法、训练精度、数据划分和历史配置不变。

## 1. 获取代码与环境

以下使用已安装的 Conda 建立 Python 3.10 环境；也可在已有 Python 3.10 下使用 venv。Ubuntu 的默认 Python 不替代已核验版本。

    git clone https://github.com/ConutlessLoneSmokes/THz_DMA.git
    cd THz_DMA
    conda create -n thz_dma python=3.10 pip -y
    conda activate thz_dma
    python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
    python -m pip install -r requirements-linux.txt -e .

直接科学依赖固定为原核验环境中的版本，见 [requirements-linux.txt](../requirements-linux.txt)。CUDA 12.8 和 CPU 的 2.11.0 wheel 均由 [PyTorch 官方页面](https://pytorch.org/get-started/previous-versions/#v2110) 提供。本项目不需要 torchvision、torchaudio 或本地编译 CUDA 扩展。

HDF5 参考脚本使用 h5py，Linux 环境文件也已固定它的版本。Linux 不加载本地 sources/code/zhou2025_python 下的旧 Windows 二进制依赖；第三方公开 MAT 数据仍须另行取得，不属于本次输入快照。

    export MPLBACKEND=Agg
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export OPENBLAS_NUM_THREADS=4
    nvidia-smi
    python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count()); assert torch.cuda.is_available()'
    python -m unittest discover -s tests -v

CUDA 检查失败时先核对驱动与 wheel；本次没有连接服务器安装或修改驱动。固定直接依赖不表示所有操作系统、驱动和浮点结果完全相同，运行环境仍由每次实验归档。

## 2. 无历史数据的最小检查

以下四步使用 8/4/4 个场景、8 步扩散和 4 次训练更新，只检查数据、训练、选参冻结和测试链路。它们可以在没有旧数据和权重的全新 clone 中运行，不构成性能实验。

    python scripts/run_zhou2025_repro.py prepare --smoke --device cpu --run-id linux_mimo_smoke_data
    python scripts/run_zhou2025_repro.py train --smoke --device cpu --run-id linux_mimo_smoke_train --data-run-id linux_mimo_smoke_data
    python scripts/run_zhou2025_repro.py evaluate --smoke --device cpu --run-id linux_mimo_smoke_validation --data-run-id linux_mimo_smoke_data --train-run-id linux_mimo_smoke_train
    python scripts/run_zhou2025_repro.py evaluate --smoke --device cpu --run-id linux_mimo_smoke_test --data-run-id linux_mimo_smoke_data --train-run-id linux_mimo_smoke_train --split test --tuning-run-id linux_mimo_smoke_validation

所有运行 ID 不可覆盖，重复检查时换新前缀。[Linux checks](../.github/workflows/linux-checks.yml) 在 Ubuntu 26.04、Python 3.10、CPU PyTorch 上执行全部单元检查与这四步；具体结果以仓库 Actions 记录为准。

## 3. 两张 RTX 3090 的使用

运行器采用单进程单 GPU。用 CUDA_VISIBLE_DEVICES 将每个进程限制到一张物理卡，进程内仍指定 cuda:0。下列两个短任务共享第 2 节的数据，使用不同运行 ID：

    CUDA_VISIBLE_DEVICES=0 python scripts/run_zhou2025_repro.py train --smoke --device cuda:0 --run-id linux_gpu0_smoke_train --data-run-id linux_mimo_smoke_data
    CUDA_VISIBLE_DEVICES=1 python scripts/run_zhou2025_repro.py train --smoke --device cuda:0 --run-id linux_gpu1_smoke_train --data-run-id linux_mimo_smoke_data

正式任务使用相同放置方式；模型、配置及是否开展新种子实验仍需按研究任务确定。两张卡的显存分别使用，现有运行器没有 DDP。先保持原 batch size、精度、随机种子和更新次数，再单独评价缓存、批量推理或混合精度的数值与速度影响。本轮没有添加性能优化或验证服务器加速倍数。

## 4. 继续使用现有数据和训练权重

GitHub 包含代码、物理表、配置、运行清单、逐场景 CSV、选参、报告、图和历史源码 ZIP。完整 CSI 数组、NPZ 缓存和模型检查点由本地单独传输。

当前已完成实验的目录和字节哈希保存在 [结果快照](../research_route/experiment_results/2026-10-13/README.md)。先在空的 runs 中复制记录，再将该快照 large_artifacts.json 列出的输入按原相对路径补齐。保留运行 ID、文件名和文件字节；不要编辑 dataset.json、dependencies.json 或 tuning.json 来消除报错。

    mkdir -p runs
    for record_dir in research_route/experiment_results/2026-10-13/*/; do
        cp -an "$record_dir" runs/
    done

例如，使用已训练的宽带 Clean DDPM 至少需要补齐：

- runs/diffusion_data_20261002_v1/data/ 下的原始缓存；
- runs/zhou_wideband_formal_20261011_clean_v1/checkpoint/training.pt；
- 来源缓存、宽带数据包装与训练运行的原始清单。这些小记录已在快照内。

读取器把旧 source_directory 中的 Windows 前缀映射到当前 runs 下同名来源，并统一相对路径的反斜杠；它仍逐项校验原数据哈希、模型的数据清单哈希和吸收表哈希。路径迁移不会使缺失数据或被修改的权重通过检查。新生成的包装直接记录 source_run_id。

可在新 ID 下评价完成的旧模型。旧验证选参用于新 test 时，仍要求数据、模型、评价设置及 tuning.json 完全匹配。旧实验的源码已与本次维护不同，不能直接对它们使用 --resume；原始源码 ZIP 和审计报告用于历史复核。Linux 上产生的新任务只在相同代码、配置、设备和依赖下续跑。

历史 audit_zhou2025_* 脚本绑定既有运行、原平台路径和源码一致性条件；它们的已保存报告属于历史证据，不是新环境的通用自检入口。完整历史审计需归档版本及对应环境。Linux新增任务使用第2节与当前prepare/train/evaluate链路，不修改旧记录来绕过来源校验。

较大的验证集诊断、基线扫描及正式训练仍由用户启动。本次代码归档没有采纳新的研究方案或重新选择测试参数。

## 5. 现有服务器与 SSH 代理隧道

2026-10-13已按用户授权完成服务器克隆：

- 预留SSH：`lsy@10.112.129.125`。
- 项目路径：`/home/lsy/THz_DMA`，初次克隆为 `ae86129`（标签 `linux-ready-2026-10-13`）；D-042按用户要求随 `main` 快进更新。当前提交用 `git rev-parse HEAD` 核对。
- 用户提供的硬件：Xeon Gold 6248R、2×RTX 3090；系统为Ubuntu 26.04.1。同步前服务器实测Python3.14.4；`nvidia-smi` 报驱动/库版本不匹配，内核驱动595.91.07、NVML库595.99。
- GitHub Ubuntu 26.04 CI的43项检查及四阶段CPU最小运行已通过 [记录](https://github.com/ConutlessLoneSmokes/THz_DMA/actions/runs/38345102265)。这不是服务器GPU实验。

SSH反向代理将服务器 `127.0.0.1:28090` 转发到Windows已有代理 `127.0.0.1:7897`。当时HTTP/SOCKS连通性及TLS验证通过；新会话先检查本地代理、SSH进程和服务器监听，避免重复绑定端口。已确认的纯命令启动方式为：

```powershell
ssh -N -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -R 127.0.0.1:28090:127.0.0.1:7897 lsy@10.112.129.125
```

服务器新终端执行 `proxy_on 28090`，仅设置该终端的代理环境。`proxy_on` 已定义在 `~/.codex-proxy.sh`，由zsh配置引用。本地已有 `codex-proxy-dorm` 辅助命令，但其脚本会写日志和环境文件；本次为满足用户“只执行命令、不修改文件或配置”的约束，仅执行SSH转发。继续开启/重建隧道也遵循该约束。连接依赖本地代理与SSH会话持续运行。

服务器目前只有Git中的代码和紧凑证据；环境安装、原大型数据与权重传输、正式训练和GPU性能检查尚未进行。D-041关闭吸收的对照已由用户另行采纳，见 [19i第13节](../research_route/19i_方向一传统参照与分步场景迁移.md#13-d-041关闭吸收的最小对照与接管状态)。D-042将最新文档与 [CODEX_START_PROMPT.md](../CODEX_START_PROMPT.md) 纳入Git，并同步服务器。可以在该目录续接设计、代码和文档工作。正式实验使用Python3.10项目环境；服务器已有Miniconda和 `EIT_test` 环境，沿用项目环境创建命令，保留其他项目环境。GPU实验前处理驱动/库版本不匹配并验证CUDA，旧数据和权重按传输清单补齐。
