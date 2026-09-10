"""Information ladder with paired step sizes and closed-form-vs-RHJB contrasts (Fig. paired_ladder).
Reads the committed *_paired_contrasts.csv files and the toy terminal-PnL path cache (paired bootstrap for
the toy Delta-CE).  Level values are the penalized means of Tables 4-5.  Writes to paper/images_final/."""
import os, numpy as np, matplotlib, csv
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
R = str(REPO); OUT = REPO / "paper" / "images_final"
plt.rcParams.update({"font.size":13,"axes.titlesize":14,"axes.labelsize":13,"xtick.labelsize":11.5,"ytick.labelsize":11.5,"legend.fontsize":11.5,"axes.grid":True,"grid.alpha":.25})
def readc(p):
    rows=list(csv.DictReader(open(p))); return {r["contrast"]:(float(r["mean_diff"]),float(r["paired_se"])) for r in rows}
fa=readc(f"{R}/imagens_tex/alpha_studies_T10000_qmax60/alpha_reduced_paired_contrasts.csv")
rg=readc(f"{R}/imagens_tex/regime_studies_T10000_qmax60/regime_paired_contrasts.csv")
# levels (Tables 4-5, penalized)
lev_fa={"GLFT":-3527.26,"I-aware":473.57,"Alpha/I-aware":2199.32}
lev_rg={"GLFT":-4153.82,"I-aware":281.23,"Alpha/I-aware":2248.43,"Regime-aware":2423.51}
# toy paired CE / PnL
g=0.01; d=np.load(f"{R}/imagens_tex/toy_model_T10000_qmax60/cara_glft_terminal_pnl_paths.npz")
ref=d["I-aware_RHJB"]; rng=np.random.default_rng(20260910); B=2000; n=len(ref)
def ce(x): return -np.log(np.mean(np.exp(-g*x)))/g
toy={}
for key,name in [("cubic-skew_refined_ansatz","toy self-consistent cubic"),("I-aware_galerkin_ansatz","toy Galerkin-cubic")]:
    x=d[key]; dpnl=x-ref; dce=ce(x)-ce(ref)
    idx=rng.integers(0,n,size=(B,n)); boots=np.array([ce(x[j])-ce(ref[j]) for j in idx])
    w=np.exp(-g*(x-x.min())); ess=w.sum()**2/(w**2).sum()
    toy[name]=(dce, boots.std(ddof=1), np.percentile(boots,[2.5,97.5]), dpnl.mean(), dpnl.std(ddof=1)/np.sqrt(n), ess)
    print(f"{name}: dCE={dce:+.2f} boot-se={boots.std(ddof=1):.2f} CI95=[{np.percentile(boots,2.5):+.2f},{np.percentile(boots,97.5):+.2f}] | dPnL={dpnl.mean():+.2f} se={dpnl.std(ddof=1)/np.sqrt(n):.2f} | ESS={ess:.0f}")
# ---- figure
fig,ax=plt.subplots(1,2,figsize=(14,5.2),gridspec_kw={"width_ratios":[1.15,1]})
a=ax[0]
for lev,con,col,lab,steps in [(lev_fa,fa,"#E69F00","fast-alpha study",["I over GLFT","alpha over I"]),(lev_rg,rg,"#1B9E77","regime study",["I over GLFT","alpha over I","regime over alpha/I (Delta_R)"])]:
    names=list(lev); y=[lev[k] for k in names]; x=np.arange(len(names))+(0.08 if lab.startswith("fast") else -0.08)
    a.plot(x,y,marker="o",ms=8,lw=2.2,color=col,label=lab)
    for j,st in enumerate(steps):
        md,se=con[st]; a.annotate(f"{md:+,.0f}\n(s.e. {se:.1f})",xy=((x[j]+x[j+1])/2,(y[j]+y[j+1])/2),xytext=(12 if lab.startswith("fast") else -12,-16 if lab.startswith("fast") else 16),textcoords="offset points",ha="left" if lab.startswith("fast") else "right",va="center",fontsize=10.5,color=col)
a.set_xticks(range(4)); a.set_xticklabels(["signal-blind\nGLFT","$I$-aware\nRHJB","Alpha/$I$-aware\nRHJB","Regime-aware\nRHJB"]); a.set_ylabel("penalized PnL (path mean)"); a.set_title("Information ladder (paired step sizes)"); a.legend(loc="lower right"); a.axhline(0,color="k",lw=.7,alpha=.4); a.set_ylim(-4600,3500)
b=ax[1]
rows=[("toy self-consistent cubic\n($\\Delta$CE, paired bootstrap)",toy["toy self-consistent cubic"][0],toy["toy self-consistent cubic"][1],True,"#8c564b"),
      ("toy Galerkin-cubic\n($\\Delta$CE, paired bootstrap)",toy["toy Galerkin-cubic"][0],toy["toy Galerkin-cubic"][1],True,"#8c564b"),
      ("fast-alpha hybrid cubic\n($\\Delta$penalized)",fa["sc-cubic vs RHJB"][0],fa["sc-cubic vs RHJB"][1],True,"#E69F00"),
      ("fast-alpha Galerkin-cubic\n($\\Delta$penalized)",fa["galerkin vs RHJB"][0],fa["galerkin vs RHJB"][1],True,"#E69F00"),
      ("regime hybrid cubic\n($\\Delta$penalized)",rg["sc-cubic vs RHJB"][0],rg["sc-cubic vs RHJB"][1],True,"#1B9E77"),
      ("regime Galerkin-cubic\n($\\Delta$penalized)",rg["galerkin vs RHJB"][0],rg["galerkin vs RHJB"][1],True,"#1B9E77")]
yy=np.arange(len(rows))[::-1]
for (lab,md,se,paired,col),y in zip(rows,yy):
    if paired: b.errorbar(md,y,xerr=1.96*se,fmt="o",ms=8,color=col,capsize=4,lw=2)
    else: b.plot(md,y,marker="o",ms=9,mfc="white",mec=col,mew=2,ls="none")
    b.annotate(f"{md:+.1f}"+(f" ± {1.96*se:.1f}" if paired else ""),xy=(md,y),xytext=(0,10),textcoords="offset points",ha="center",fontsize=10.5,color=col)
b.set_yticks(yy); b.set_yticklabels([r[0] for r in rows]); b.axvline(0,color="k",lw=.9,alpha=.5); b.set_xlabel("closed form minus its RHJB reference (95% CI)"); b.set_title("Closed forms vs. numerical RHJB"); b.set_xlim(-165,30); b.set_ylim(-0.6,len(rows)-0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/paired_ladder.png",dpi=170); print("wrote paired_ladder.png")
