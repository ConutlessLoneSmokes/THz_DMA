from pathlib import Path
import numpy as np,pandas as pd
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
f=pd.read_csv(p.parent/"metrics_raw.csv")
labels={"grid_omp_3d":"OMP","sbl_jointfreq_3d":"SBL","lmmse":"LMMSE","oracle_geometry_ls":"Oracle geometry + LS","diffusion_mean":"Diffusion v4"}
records=[]
for (s,x,m),g in f[f.estimator.isin(labels)].groupby(["schedule","array_reference_snr_db","estimator"]):
 records.append(dict(schedule=s,array_reference_snr_db=x,estimator=m,mean_rf_output_snr_db=float(g.effective_output_snr_db.mean()),nmse_db=float(10*np.log10(g.fullband_128_nmse_linear.mean())),rf_snr_min=float(g.effective_output_snr_db.min()),rf_snr_max=float(g.effective_output_snr_db.max())))
t=pd.DataFrame(records);t.to_csv(p/"snr_bridge_existing_points.csv",index=False)
fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True,constrained_layout=True)
for ax,(s,g) in zip(axes,t.groupby("schedule")):
 for m,c in g.groupby("estimator"):
  c=c.sort_values("array_reference_snr_db");ax.plot(c.mean_rf_output_snr_db,c.nmse_db,"o-",label=labels[m])
 ax.set(title="F1: 1 config x 128 tones" if "J1" in s else "F2: 8 configs x 16 tones",xlabel="Mean measured RF-output SNR (dB)",xlim=(-10,30),ylim=(-45,2));ax.grid(alpha=.25);ax.legend(fontsize=8)
axes[0].set_ylabel("Fullband NMSE (dB; linear mean then dB)")
fig.suptitle("Existing 16-scene results: relabeled SNR only; no new simulations",fontsize=10)
fig.savefig(p/"snr_bridge_existing_points.png",dpi=180);fig.savefig(p/"snr_bridge_existing_points.pdf");plt.close(fig)
