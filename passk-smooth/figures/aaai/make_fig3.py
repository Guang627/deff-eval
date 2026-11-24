import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.serif":["Times","DejaVu Serif"],"font.size":8,
  "axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#52514e","xtick.color":"#52514e","ytick.color":"#52514e",
  "legend.frameon":False,"pdf.fonttype":42})
R="../../results/"
p=pd.read_csv(R+"swebench_verified/per_submission.csv"); p=p[p.scheme=="iid"]
w=p.pivot(index="submission",columns="tier",values="sd")
i=pd.read_csv(R+"deff_validation/swe_submission_icc.csv").set_index("submission")
fig,ax=plt.subplots(figsize=(3.3,2.3))
xs=np.linspace(0,0.1,50)
for t,V,c,pred in [("4x10",10,"#eb6834",1.126),("2x20",20,"#1baf7a",1.272)]:
    j=i.join((w[t]/w["natural"]).rename("r"))
    ax.scatter(j.icc_repo,j.r,s=9,color=c,alpha=0.75,edgecolor="none",label=f"{t.replace('x','×')}, each submission")
    ax.plot(xs,np.sqrt(1+(V-1)*xs),color=c,lw=0.9,ls="--")
    ax.axhline(pred,color=c,lw=0.8,ls=":")
ax.set_xlabel("submission's own repository ρ"); ax.set_ylabel("SD ratio vs. 40 random tasks")
ax.set_xlim(0,0.1); ax.set_ylim(0.8,1.8); print("max",max(i.join((w[t]/w["natural"]).rename("r")).r.max() for t in ["4x10","2x20"])); ax.legend(loc="upper left",fontsize=6.5,handletextpad=0.2)
ax.text(0.099,1.255,"pilot plan 2×20",fontsize=6,color="#52514e",ha="right",va="top")
ax.text(0.099,1.11,"pilot plan 4×10",fontsize=6,color="#52514e",ha="right",va="top")
fig.tight_layout(); fig.savefig("figs/fig_swe_submissions.pdf"); fig.savefig("figs/fig_swe_submissions.png",dpi=200)
