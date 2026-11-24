import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.serif":["Times","DejaVu Serif"],"font.size":8,
  "axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#52514e","xtick.color":"#52514e","ytick.color":"#52514e","legend.frameon":False,"pdf.fonttype":42})
R="../../results/deff_validation/"
fig,ax=plt.subplots(1,2,figsize=(3.3,2.1))
rng=np.random.default_rng(0)
for j,(m,lab) in enumerate([("level","by level"),("original","by question")]):
    d=pd.read_csv(R+f"gsm_{m}_20.csv")
    for a,k,c in [(ax[0],1,"#2a78d6"),(ax[1],16,"#eb6834")]:
        r=d[(d.k==k)&(d.tier=="10x10")].rho_pilot.to_numpy()
        a.scatter(j+rng.uniform(-.12,.12,r.size),r,s=10,color=c,alpha=0.8,edgecolor="none")
        a.plot([j-.25,j+.25],[np.median(r)]*2,color="#0b0b0b",lw=1.2)
for a,t in [(ax[0],"pass@1"),(ax[1],"pass@16")]:
    a.set_xticks([0,1]); a.set_xticklabels(["by\nlevel","by\nquestion"]); a.set_xlim(-.5,1.5); a.set_ylim(0,1); a.set_title(t,fontsize=8,loc="left")
ax[0].set_ylabel("pilot ρ (one dot per split)")
fig.tight_layout(); fig.savefig("figs/fig_leak_rho.pdf"); fig.savefig("figs/fig_leak_rho.png",dpi=200)
