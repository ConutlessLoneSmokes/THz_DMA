"""Summarize audited D-039 results and separate recommendations from outcomes."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
import audit_zhou2025_migration as a
import run_zhou2025_repro as r

OUT=Path(__file__).parent
c=pd.read_csv(OUT/"curves.csv")
test=c[c.split=="test"]
integrity=json.loads((OUT/"integrity.json").read_text())
replay=json.loads((OUT/"replay_integrity.json").read_text())
control=json.loads((OUT/"digital_control_integrity.json").read_text())
ls=pd.read_csv(OUT/"digital_ls_curves.csv")
ls_rows=pd.read_csv(OUT/"digital_ls_per_scene.csv")
pairs=pd.read_csv(OUT/"paired_comparisons.csv")
digital=pd.read_csv(ROOT/"runs"/(a.FAMILIES[4]+"_test_v1")/"metrics_raw.csv")
model="zhou_wideband_formal_20261011_clean_v1"
ls_pairs=a.paired(digital[digital.estimator==model],ls_rows,a.FAMILIES[4],
                  "Clean DDPM minus supplementary direct LS",np.random.default_rng(2026101302))
pd.DataFrame(ls_pairs).to_csv(OUT/"digital_ddpm_vs_ls_paired.csv",index=False)
steps=[]
for family in a.FAMILIES:
    folder=ROOT/"runs"/(family+"_test_v1")
    config=r.read_json(folder/"config_resolved.json")["config"]
    raw=pd.read_csv(folder/"metrics_raw.csv")
    name=config["job"]["train_run_id"][0]
    for (scene,schedule),block in raw[(raw.estimator==name)&(raw.snr_db>=15)].groupby(["scene_id","schedule"]):
        block=block.sort_values("snr_db")
        for first,second in zip(block.iloc[:-1].itertuples(),block.iloc[1:].itertuples()):
            steps.append({"family":family,"scene_id":scene,"schedule":schedule,"from_snr":first.snr_db,
                          "to_snr":second.snr_db,"change_db":second.nmse_db-first.nmse_db})
pd.DataFrame(steps).to_csv(OUT/"high_snr_steps.csv",index=False)

def curve(family,schedule,method):
    return test[(test.family==family)&(test.schedule==schedule)&(test.estimator==method)].sort_values("snr_db")

def value(family,schedule,method,snr=30):
    return float(curve(family,schedule,method).query("snr_db==@snr").iloc[0].nmse_db)

def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"---|"*len(headers),
                      *["| "+" | ".join(str(v) for v in row)+" |" for row in rows]])

plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
figure,axes=plt.subplots(2,3,figsize=(17,10),layout="constrained")
def line(ax,block,label,color,style="-"):
    ax.plot(block.snr_db,block.nmse_db,"o",linestyle=style,ms=3.5,lw=1.6,label=label,color=color)

extended=a.FAMILIES[0]
old_model="zhou_mimo_formal_20261009_clean_v1"
for ax,npilots in zip(axes[0,:2],(64,38)):
    label=f"QPSK_Np{npilots}"
    for method,title,color,style in [(old_model,"Clean DDPM","#0072B2","-"),
        ("lmmse_zhou_mimo_formal_20261009_data_v1","Global LMMSE","#999999","-"),
        ("lasso","LASSO, original budget","#D55E00","-"),("angular_omp","DFT-bin OMP","#009E73","-"),
        ("oracle_geometry_ls","True-geometry LS","#CC79A7","--")]:
        line(ax,curve(extended,label,method),title,color,style)
    ax.set_title(f"Original MIMO: {npilots} pilots")
ax=axes[0,2]
for family,title,color in zip(a.FAMILIES[1:4],("40 GHz, no absorption","300 GHz, no absorption","300 GHz, absorption"),
                            ("#0072B2","#D55E00","#009E73")):
    name=family+"_clean_v1"
    line(ax,curve(family,"QPSK_Np38",name),title+" DDPM",color)
    line(ax,curve(family,"QPSK_Np38","lmmse_"+family+"_data_v1"),title+" LMMSE",color,"--")
ax.set_title("Fixed aperture/noise: 38 pilots")
ax=axes[1,0]
for schedule,label,color in [("digital_frequency_single_shot_C2_J1_K128","All 128 tones","#0072B2"),
    ("digital_tensor_structured_J8_K16_RE128","16 tones, 8 repeats","#D55E00")]:
    line(ax,curve(a.FAMILIES[4],schedule,model),label+" DDPM",color)
    line(ax,ls[ls.schedule==schedule],label+" direct LS",color,"--")
ax.set_title("Wideband full-digital reception")
for ax,schedule,label in [(axes[1,1],"frequency_single_shot_C2_J1_K128","F1"),
                         (axes[1,2],"tensor_structured_J8_K16_RE128","F2")]:
    for method,title,color,style in [(model,"Global DDPM","#0072B2","-"),
        ("lmmse_zhou_wideband_formal_20261011_data_v1","Global LMMSE (256)","#999999","-"),
        ("conditional_grid_omp_3d","OMP, coarse centers","#D55E00","-"),
        ("conditional_sbl_jointfreq_3d","SBL, coarse centers","#E69F00","-"),
        ("conditional_conditional_lmmse","LMMSE, coarse centers","#009E73","-"),
        ("conditional_oracle_geometry_ls","True-geometry LS","#CC79A7","--")]:
        line(ax,curve(a.FAMILIES[5],schedule,method),title,color,style)
    ax.set_title("Wideband DMA: "+label)
for ax in axes.flat:
    ax.set(xlabel="Declared SNR (dB)",ylabel="NMSE (mean sample ratio, dB)")
    ax.grid(alpha=.22)
    ax.legend(fontsize=7.6,loc="best")
figure.suptitle("D-039 frozen test results: 100 scenes, one training seed",fontsize=15)
figure.savefig(OUT/"migration_test_curves.png",dpi=180)
figure.savefig(OUT/"migration_test_curves.pdf")
plt.close(figure)

original_rows=[]
for method,title in [(old_model,"Clean DDPM"),("lmmse_zhou_mimo_formal_20261009_data_v1","全局 LMMSE"),
                     ("lasso","LASSO，原预算"),("angular_omp","角域 OMP"),("oracle_geometry_ls","真实几何 LS")]:
    original_rows.append([title,*[f"{value(extended,schedule,method):.3f}" for schedule in ("QPSK_Np64","QPSK_Np38")]])
carrier_rows=[]
for family,title in zip(a.FAMILIES[1:4],("40 GHz，无吸收","300 GHz，无吸收","300 GHz，含吸收")):
    carrier_rows.append([title,*[f"{value(family,schedule,method):.3f}" for schedule in ("QPSK_Np64","QPSK_Np38")
                  for method in (family+"_clean_v1","lmmse_"+family+"_data_v1")]])
dma_rows=[]
for method,title in [(model,"全局 DDPM"),("lmmse_zhou_wideband_formal_20261011_data_v1","全局 LMMSE，256 样本"),
    ("conditional_grid_omp_3d","已知粗中心 OMP"),("conditional_sbl_jointfreq_3d","已知粗中心 SBL"),
    ("conditional_conditional_lmmse","已知粗中心 LMMSE"),("conditional_oracle_geometry_ls","真实几何 LS")]:
    dma_rows.append([title,*[f"{value(a.FAMILIES[5],schedule,method):.3f}" for schedule in
                  ("frequency_single_shot_C2_J1_K128","tensor_structured_J8_K16_RE128")]])
report=f"""# D-039 分步迁移核验

核验日期：2026-10-13。用户已完成全部 20 个正式阶段，12 个验证/测试运行共 144000 行。**原参数化 MIMO 结果保持；宽带全数字全频点恢复有效，稀疏选频与 DMA 的全局 DDPM 尚未实现有效全带恢复。**

## 1. 完整性与复算

全部阶段完成，无缺行、重复或非有限误差。1520 个源码快照成员与当前源文件一致，11 个数据集的 3552 个清单成员核对通过；三个载频的 4452 组路径记录逐场景相同。六个测试运行与各自验证的依赖、评价设置和选参文件一致；没有测试重选参。原 12 方法在新增 MIMO 对照中的 43200 条共同记录数值一致。

64 项固定恢复复算最大差 {replay['max_difference_db']:.3g} dB。重算误差用 complex128，原存储误差分子使用 complex64，微小差异不改变曲线。三个新 MIMO DDPM 各 16000 次更新；宽带模型完成 256000 次更新，最佳验证检查点为 255250。源码、配置和依赖核验见 [integrity.json](integrity.json)，复算见 [replay_integrity.json](replay_integrity.json)。没有重训或修改原始指标、选参和权重。

配对区间采用同一场景的 dB 差、10000 次 bootstrap；固定单个训练种子，不含训练不确定性且未作多重比较校正。跨前端使用相同信道和噪声方差，噪声实现因观测维度不同而不同。验证用于选参，以下主表均为冻结 test 的逐样本线性 NMSE 均值转 dB。

## 2. 原 MIMO 与新参照

30 dB 的冻结测试：

{table(['方法','Np64','Np38'],original_rows)}

Clean DDPM 相对全局 LMMSE、LASSO 和角域 OMP，各 18/18 个协议×SNR 组的配对区间都低于零。原 SURE 结果和选参保持。OMP 在高 SNR 约 −9 dB，固定 DFT-bin 三项适配存在地板；没有据此判断所有 OMP 或几何估计都失效。真实几何 LS 使用额外真值信息，不与可部署方法混排。

## 3. 载频控制

30 dB、同一米制间距和物理噪声参照：

{table(['配置','Np64 DDPM','Np64 LMMSE','Np38 DDPM','Np38 LMMSE'],carrier_rows)}

40 GHz 的 DDPM 相对 LMMSE 在 18/18 组配对区间高于零；Np38 从 20 dB 的 −22.450 回升到 30 dB 的 −21.577。两个 300 GHz 配置的 DDPM 相对 LMMSE 各为 18/18 组低于零。学习优势依赖场景，不能从原 300 GHz 结果推广到所有载频。固定米制间距使电尺寸和接收功率都随载频变化，这一控制不是只改变吸收。

在当前约 2 m 的 300 GHz 设置，含/不含吸收对照的 DDPM 组均值差均不足 0.02 dB；少数配对区间不含零不等于具有明显工程影响。此结果不支持吸收驱动的大收益，也不覆盖其他距离/吸收窗口。载频间比较保存实际接收 SNR，不能把名义横轴直接当成每个场景的实际输出 SNR。

## 4. 宽带数字接收与基线补核

30 dB 下，全 128 点接收的 DDPM 为 −32.291 dB。只观测 16 个频点并重复 8 次时，全带为 −1.479 dB，导频域为 −38.935 dB，未观测 112 点为 −0.899 dB。观测频点已经拟合，主要失败位于未观测频率。

原全局 LMMSE 只使用 256 个训练场景，在全频点接收时约 −1.418 dB。以无噪声观测、$10^{{-8}}$ 正则噪声极限重算全部 100 个 test 场景，仍为 −1.418310 dB，定位到本次有限训练样本张成空间的误差地板。

补算同一带噪观测的直接数字 LS，不重选任何参数：全频点 20/30 dB 为 −19.901/−29.901 dB，DDPM 为 −26.060/−32.291 dB，改善约 6.16/2.39 dB；不是相对弱 LMMSE 表现出的约 25/31 dB 增益。16 点最小范数 LS 的 30 dB 全带值为 −0.579 dB，DDPM 改善约 0.90 dB，仍未恢复大部分未观测频点。

新增 1800 行 LS 只保存在 analysis，不追加到原实验指标。无噪声 LS 的全频点恒等和选频零填充均通过；详见 [digital_control_integrity.json](digital_control_integrity.json)、[digital_ls_curves.csv](digital_ls_curves.csv) 与 [covariance_span_projection.csv](covariance_span_projection.csv)。

## 5. DMA 迁移

30 dB 的冻结测试，横轴为训练平均单阵元输入参考：

{table(['方法与信息条件','F1 全带','F2 全带'],dma_rows)}

实际白化接收 SNR 均值为 F1 8.762 dB、F2 8.378 dB，区别于横轴 30 dB。真实几何 LS 与已知粗中心 LMMSE 随 SNR 改善；全局 DDPM 接近零估计的 0 dB 地板。与条件方法的信息条件不同，不据此形成同信息条件的优越性排名。

固定案例 0/99、30 dB 的 DDPM 预测能量仅占真值 0.0226%–0.0291%，预测与真值相关度平方很低；压缩观测的相对残差却约 0.7%–1.0%。因此不能用观测拟合替代完整 CSI 恢复。这两个案例的真值在观测行空间中的能量也很低，具体记录见 [fixed_weight_diagnostics.csv](fixed_weight_diagnostics.csv)；它们只作因果定位，不替代 100 场景统计。

宽带网络在人工噪声 t=100 的干净 CSI 去噪 NMSE 为约 +9.19 dB，t=50 为约 −10.97 dB。该先验在强人工噪声段仍弱；数字全频点接收能恢复，不足以说明稀疏观测下的先验也已通过。当前证据支持联合检查先验、未观测频率与压缩行空间，尚未隔离出唯一原因。

DMA 仍输出可行波束与净速率，但全局 DDPM 的 30 dB 净速率低于已知粗中心 LMMSE。速率扣导频/切换，尚未扣在线求解时间；数字前端没有相同的链路指标，不能直接比较端到端通信收益。

## 6. 收敛与适用范围

DMA SBL 的验证/测试各 1800 行全部达到停止线。新载频 LASSO 仍有 2606 行达到原 1000 次预算，其中 40 GHz 验证/测试各 874/861 行；300 GHz 无吸收各 208/227，含吸收各 209/227。原扩展 MIMO 的 209/227 条与历史相同，已有固定惩罚补充迭代报告可参考。新载频没有在本轮重新运行全部补充预算，所以涉及其 LASSO 的精确排名须保留迭代预算限定。

本轮均为单个训练种子、匹配训练分布、单发射端宽带平台。没有复现原 QuaDRiGa 分布，没有多种子、跨分布迁移或实测硬件结果；窄带 MIMO→宽带单发射端还改变了系统结构，不能称为纯带宽消融。

## 7. 下一步建议，尚未实施

1. **优先固定当前宽带权重，做稀疏观测的分解控制。** 用验证场景对比全频点、16 点规则/非规则选频、DMA，保存无噪声与带噪的导频/未观测域误差、行空间/零空间误差和引导轨迹。无噪声真实几何参照检查结构可辨识性；只在验证集选择下一方案，不用本轮 test 重新调参。
2. **补强传统基线，再评价学习增益。** 数字全频点保留直接 LS，LMMSE 扫描 256/512/1024/2048 训练样本；新载频 LASSO 固定现有惩罚补充迭代并核对目标函数与 NMSE 变化。协方差扫描需分块控制内存，较大运行交由用户执行。
3. **控制指向先验不足后，再改频率联合结构。** 比较物理参数域联合恢复、低维结构化先验或更明确的频率联合表示与当前完整 CSI CNN；新方法属于 DMA/宽带适配，保留原 Zhou 参数化复现记录。不能把添加网络名称本身写成创新。
4. **暂不追加新场景 SURE 长训练或全套多种子。** 先让稀疏频点/DMA 的完整 CSI 恢复达到可验证改善，再对入选方法开展多种子与独立冻结测试；40 GHz 的学习收益也需与已表现更强的 LMMSE 比较。

建议没有被写成用户已采纳的决策，本轮没有执行新训练、修改表面状态、改变导频或重新选择测试参数。
"""
(OUT/"audit_report.md").write_text(report,encoding="utf-8")
print("report and PNG/PDF written",flush=True)
