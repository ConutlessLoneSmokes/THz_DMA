# 既有分子吸收系数表

`data_freq_abscoe.txt` 是用户在 2026-09-07 提供的既有 HITRAN 系数数据。原始 MATLAB 调用保存在 `legacy_matlab/`，用于核对数值口径，不作为新的实现入口。

## 当前解释

- 第一列为频率，单位 Hz。
- 第二列按原 `getAbsLoss.m` 的计算式解释为功率吸收系数 $\kappa(f)$，单位推定为 $\mathrm{m}^{-1}$。
- 原 MATLAB 损耗为

$$
L_{\mathrm{abs,dB}}=10\log_{10}(e)\,\kappa(f)d.
$$

- 因而功率传输因子为 $e^{-\kappa d}$，复信道的振幅因子为 $e^{-\kappa d/2}$。代码只在一个位置应用该振幅因子，避免与 dB 路损重复计入。

## Python 迁移口径

- 默认在表内进行线性插值，并对越界频率报错。
- 提供 `legacy-nearest` 模式，等价复现原 MATLAB 的最近点和 $9.894\times10^8$ Hz 容差。
- 当前方向只使用 $0<f\leq1$ THz 的数据。该区间共有 1,311 个严格递增采样点，间隔约 0.76 GHz。
- 原文件覆盖至 50 THz；高频端文本有效位数造成重复频率，加载器会合并重复项，但这部分不进入当前通信实验。

## 尚缺元数据

现有文件没有记录生成时采用的 HITRAN 版本、温度、压力、湿度、气体组分和谱线展宽设置。因此它适合 Stage 0 机制筛查，正式结果前必须找回生成脚本或与可复现的权威模型交叉验证。代码不会把这些未知条件伪装成“标准大气”。

## 完整性哈希

- `data_freq_abscoe.txt`: `3AA52BF00E30831C82B98B06D48EA8FDD8EBE7CC2DA6E233AD2ED3238DC4DCA8`
- `legacy_matlab/getAbsLoss.m`: `5565651853DEE0428CB88DB98ADFFDC57423B72B849BDE5F5E7E42350C83A3A9`
- `legacy_matlab/getSpreadLoss.m`: `AC1825C91DBE0326C9BDDFCEC92E0038055076CBD1114A0493B98E318E4A4352`

