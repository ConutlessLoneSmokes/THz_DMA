# 2026-10-13 DDPM/SURE-DM 与分步迁移结果快照

对应 D-040 的 GitHub 代码归档。包含 44 个运行记录：42 个正式/来源记录，以及 2 个早期验证分析记录。正式逐场景指标共 194400 行，其中 D-039 的 20 个阶段占 144000 行；D-037/D-038 的原共同验证、单独 Clean 验证和独立测试一并保留。旧结果中的宽带/DMA 恢复失败没有筛除。

561 个原文件按字节复制，合计约 91.79 MiB；逐文件大小、原路径及 SHA-256 见 [manifest.json](manifest.json)。3221 个历史源码 ZIP 成员与各自启动时清单一致。文件时间和旧平台绝对路径是历史来源信息，当前可运行版本位于仓库 src/ 和 scripts/，二者不能混同。

## 核验入口

- [运行索引](run_inventory.csv)
- [D-037 MIMO 验证审计](zhou_mimo_formal_20261009_val_v1/analysis/audit_report.md)
- [D-038 冻结独立测试审计](zhou_mimo_independent_test_20261010_v1/analysis/audit_report.md)
- [D-039 分步迁移审计](zhou_wideband_formal_20261011_dma_test_v1/analysis/audit_report.md)
- [D-039 六组曲线](zhou_wideband_formal_20261011_dma_test_v1/analysis/migration_test_curves.png)
- [复算与诊断记录](zhou_wideband_formal_20261011_dma_test_v1/analysis/replay_integrity.json)
- [本次 Linux 准备检查](linux_preparation_checks.json)

## 未放入 Git 的运行输入

[large_artifacts.json](large_artifacts.json) 列出 3667 个外部输入文件、原相对路径、字节数与原清单/检查点哈希，合计约 3.47 GiB。包括完整 CSI NPY、原 NPZ、原始参数 JSON 和 15 个正式模型检查点；参数 JSON 虽然较小，也属于读数据时必须校验的输入。两项早期分析记录仅归档报告，外部输入清单针对正式/来源运行。

这是传输清单，文件不包含在本快照。数据哈希沿用已核验原数据清单；检查点哈希在复制归档时重新读取。执行新评价需要先将记录和外部输入恢复到 runs 下同名目录。路径读取器兼容旧 Windows 清单，并继续验证哈希，不改写历史 JSON。

源码 ZIP 是原始运行版本。任何源代码维护后，旧审计中的“当前源码与启动时一致”条件不再自动成立；历史复核应使用对应原始版本，新实验使用当前代码和新运行 ID。详见 [Linux 指南](../../../docs/LINUX_RUN.md)。

由 [archive_results.py](archive_results.py) 生成，脚本不覆盖已有快照、不编辑原运行；复制字节、正式行数、运行状态及源码 ZIP 都有自检。此前 [2026-10-09 的 40 运行快照](../2026-10-09/README.md) 保持不变。
