import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.serif":["Times","DejaVu Serif"],"font.size":8,
  "axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#52514e","axes.labelcolor":"#0b0b0b",
  "xtick.color":"#52514e","ytick.color":"#52514e","legend.frameon":False,"pdf.fonttype":42})
BLUE,ORANGE,AQUA,INK2="#2a78d6","#eb6834","#1baf7a","#52514e"
d=pd.read_csv("../../results/deff_validation/deff_splits.csv")
def agg(b,k,tiers,col):
    s=d[(d.benchmark==b)&(d.k==k)].groupby("tier")[col]
    return np.array([[s.median()[t],s.quantile(.05)[t],s.quantile(.95)[t]] for t in tiers])
fig,ax=plt.subplots(1,2,figsize=(6.75,2.45))
G=["50x2","20x5","10x10"]; S=["8x5","4x10","2x20","1x40"]
def panel(a,b,k,tiers,color,lab,design=True,marker="o",lim=None,labels=True,offset=(4,-3)):
    m=agg(b,k,tiers,"measured")
    for col,filled,nm in ([("pred_design",True,"design")] if design else [])+[("pred_naive",False,"naive")]:
        p=agg(b,k,tiers,col)
        a.errorbar(p[:,0],m[:,0],xerr=[p[:,0]-p[:,1],p[:,2]-p[:,0]],yerr=[m[:,0]-m[:,1],m[:,2]-m[:,0]],
          fmt=marker,ms=5,mfc=color if filled else "white",mec=color,mew=1.2,ecolor=color,elinewidth=0.6,
          capsize=0,alpha=1,label=f"{lab}, {nm}",zorder=3)
        if labels and (filled or not design):
            for (x,_,_),(y,_,_),t in zip(p,m,tiers):
                a.annotate(t.replace("x","×"),(x,y),xytext=offset,textcoords="offset points",fontsize=6.5,color=INK2)
for a,lim,title in [(ax[0],(1,4),"GSM-Symbolic (templates), baseline 100×1"),(ax[1],(0.85,1.7),"SWE-bench Verified (repos), baseline random 40")]:
    a.plot(lim,lim,color="#b5b4ae",lw=0.8,ls="--",zorder=1); a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel("predicted SD ratio (from pilot)"); a.set_title(title,fontsize=8,color="#0b0b0b",loc="left")
ax[0].set_ylabel("measured SD ratio (held out)")
panel(ax[0],"GSM-Symbolic",1,G,BLUE,"pass@1")
panel(ax[0],"GSM-Symbolic",16,G,ORANGE,"pass@16",design=False,marker="s",offset=(4,2))
panel(ax[1],"SWE-bench Verified",1,S,AQUA,"pass@1",offset=(-22,5))
ax[0].legend(loc="upper left",fontsize=6.5,handletextpad=0.3); ax[1].legend(loc="upper left",fontsize=6.5,handletextpad=0.3)
fig.tight_layout(); fig.savefig("figs/fig_pred_vs_measured.pdf"); fig.savefig("figs/fig_pred_vs_measured.png",dpi=200)
fig,ax=plt.subplots(1,2,figsize=(6.75,2.2),gridspec_kw={"width_ratios":[1,1.15]})
def cov(a,b,k,tiers,color,shift,lab):
    x=np.arange(len(tiers))+shift
    for col,filled,nm in [("cov_iid",False,"iid"),("cov_deff",True,"iid × √deff (pilot)")]:
        cs=d[(d.benchmark==b)&(d.k==k)].groupby("tier")[col].mean(); v=np.array([cs[t] for t in tiers])
        a.plot(x,v,marker="o" if k==1 else "s",ms=5,mfc=color if filled else "white",mec=color,mew=1.2,
               color=color,lw=0.8 if filled else 0.6,ls="-" if filled else ":",label=f"{lab}, {nm}")
GT=["100x1","50x2","20x5","10x10"]; ST=["natural","8x5","4x10","2x20","1x40"]
cov(ax[0],"GSM-Symbolic",1,GT,BLUE,-0.06,"pass@1"); cov(ax[0],"GSM-Symbolic",16,GT,ORANGE,0.06,"pass@16")
cov(ax[1],"SWE-bench Verified",1,ST,AQUA,0,"pass@1")
for a,t,title in [(ax[0],GT,"GSM-Symbolic"),(ax[1],ST,"SWE-bench Verified")]:
    a.axhline(0.95,color="#b5b4ae",lw=0.8,ls="--"); a.set_ylim(0.45,1.01)
    a.set_xticks(range(len(t))); a.set_xticklabels([s.replace("x","×") for s in t])
    a.set_xlabel("sources × problems per source"); a.set_title(title,fontsize=8,loc="left")
ax[0].set_ylabel("coverage of nominal 95% interval")
ax[0].legend(loc="lower left",fontsize=6.5); ax[1].legend(loc="lower left",fontsize=6.5)
ax[0].annotate("0.85",(2.06,0.850),xytext=(6,-2),textcoords="offset points",fontsize=6.5,color=INK2)
ax[0].annotate("0.76",(3.06,0.763),xytext=(-4,-10),textcoords="offset points",fontsize=6.5,color=INK2)
fig.tight_layout(); fig.savefig("figs/fig_coverage.pdf"); fig.savefig("figs/fig_coverage.png",dpi=200)
