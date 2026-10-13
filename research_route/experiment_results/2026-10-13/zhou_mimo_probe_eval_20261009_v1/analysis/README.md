# D-037 最小流水线核验

日期：2026-10-09。35 个阶段运行完成；这是实现与短验证核验，正式性能复现未完成。

- [integrity.json](integrity.json)：13 个数据集、19 个模型、19 个源码 ZIP / 1349 个成员及三个评价运行的核验；8 项恢复复算最大差 1.78e-15 dB。
- [run_inventory.csv](run_inventory.csv)：35 个阶段运行、场景与连通性标记。
- [denoising_profiles.csv](denoising_profiles.csv)：7 个短训练 DDPM、5 个固定时刻，各自验证输入分布的去噪误差。
- [sure_denoising.csv](sure_denoising.csv)：两档噪声、两种目标，在 32 个验证场景上的实际 CSI 误差。
- [verify_and_profile.py](verify_and_profile.py)：上述核验的固定来源，参考提取前的 Git 版本 3a767ff368e121cd59be271706f33e2e5cc2db10。

本次完整检查命令为 `python -m unittest discover -s tests -v`，结果为 27 项通过、OK，最后一次耗时 2.284 s。另完成 10 个相关 Python 文件的语法检查、3 个配置、142 个文档链接及 50 条 CSV/BibTeX 对照；git diff --check 通过。检查未运行独立测试或正式训练。

实际效果、原单位及 SNR 定义、历史字段差异和正式运行命令见 [19g 号文档](../../../research_route/19g_方向一Zhou_DDPM与SURE_DM实现与最小验证.md)。原始指标、配置和训练权重保留未改写。
