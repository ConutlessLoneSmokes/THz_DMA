# D-041 分段顺序与部分核验

用户重申先补齐同阵列窄带DMA、第一轮去掉观测噪声。本次增加独立窄带无噪声入口、四组无噪声报告和带噪前置检查；物理数据、状态库、模型和正式预算未改。

57项检查通过。新前缀`d041_noabs_staged_smoke_20261013_v1`完成18个CPU小运行；先14个无噪声阶段和240行/720条轨迹核验，再4个带噪阶段，完整480行/1440条轨迹通过。两个模型各4次更新，8步恢复，正式学习性能和唯一原因仍未验证。

[verification.json](verification.json)保存顺序与字节核对，[staged_smoke.log](staged_smoke.log)保存实际阶段日志。[四组无噪声报告](staged_noiseless/README.md)和[完整报告](staged_complete/README.md)对应新版流程。完整数据、逐案例指标、源码ZIP和模型仍保存在项目runs下的18个新运行中；此紧凑档案仅保存本轮派生报告和路由证据，避免重复归档未改变的物理实现。

旧v3仅只读生成[窄带独立报告](narrow_noiseless/README.md)和[四组无噪声报告](paired_noiseless/README.md)，原v3的314个档案文件与561个历史原文件字节哈希全部不变。部分报告只读取冻结最终端点；smoke和数值敏感结果不作正式原因归属，不据测试结果重选参数。

正式命令见[最新指南](../../../../docs/D041_RUN.md#5-完整-linux-命令)。当前CUDA不可用，长训练与正式评价继续由用户启动；第一步为`narrow-noiseless`，而非原窄带MIMO或16点宽带采样。
