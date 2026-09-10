"""Regime-study depth figures (vs imbalance at q=10 for R=-1,0,+1; vs regime level z_r) and the z_r-slopes
quoted in the paper.  Re-solves the regime RHJB with regime_studies/regime_fast_alpha_ergodic_cole_hopf.py
(fast; no Monte Carlo).  Writes the *_v2.png files to paper/images_final/."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "regime_studies"))
import regime_fast_alpha_ergodic_cole_hopf as R
OUT = REPO / "paper" / "images_final"
plt.rcParams.update({"font.size":15,"axes.titlesize":16,"axes.labelsize":15,"xtick.labelsize":13,"ytick.labelsize":13,"legend.fontsize":13,"axes.grid":True,"grid.alpha":.25})
print("solving regime RHJB ..."); u,info=R.solve_regime_ergodic(); print("converged:",info.get("converged"))
Zc=R.Z_LEVELS; fin=lambda a: np.where(np.isfinite(a),a,np.nan)
rhjb=lambda q,y,r: R.regime_aware_depths(u,np.asarray(q,int),np.asarray(y,float),np.asarray(r,int))
MODELS=[("Regime-aware RHJB",rhjb,dict(ls="-",lw=2.8,color="#111111")),
        ("Regime hybrid cubic",lambda q,y,r:R.regime_sc_cubic_depths(q,y,r),dict(ls="--",lw=2.5,color="#1B9E77")),
        ("Regime Galerkin-cubic",lambda q,y,r:R.regime_galerkin_depths(q,y,r),dict(ls=":",lw=2.6,color="#B07AA1"))]
RL={0:"$R=-1$ ($z=-1.25$)",1:"$R=0$",2:"$R=+1$ ($z=+1.25$)"}
yy=np.linspace(-6,6,301); q0=10
fig,ax=plt.subplots(3,2,figsize=(12,11),sharex=True)
for r in range(3):
    for name,fn,st in MODELS:
        a,b=fn(np.full_like(yy,q0),yy,np.full_like(yy,r,dtype=int)); ax[r,0].plot(yy,fin(a),label=name,**st); ax[r,1].plot(yy,fin(b),label=name,**st)
    ax[r,0].set_title(f"Ask depth, $q={q0}$, {RL[r]}"); ax[r,1].set_title(f"Bid depth, $q={q0}$, {RL[r]}")
    for a in ax[r]: a.axhline(0,color="k",lw=.8,alpha=.3); a.axvline(0,color="k",lw=.6,alpha=.2); a.set_ylabel("signed depth")
for a in ax[-1]: a.set_xlabel("imbalance $i$")
ax[0,0].legend(loc="upper left",framealpha=.9); fig.tight_layout(); fig.savefig(f"{OUT}/regime_fast_alpha_depths_vs_i_v2.png",dpi=170); plt.close(fig); print("wrote vs_i v2")
combos=[(0,0.0),(10,2.0),(15,-2.0)]
fig,ax=plt.subplots(3,2,figsize=(11,10),sharex=True); slopes={}
for row,(qv,iv) in enumerate(combos):
    for name,fn,st in MODELS:
        av=[float(fn(np.array([qv]),np.array([iv]),np.array([r]))[0][0]) for r in range(3)]
        bv=[float(fn(np.array([qv]),np.array([iv]),np.array([r]))[1][0]) for r in range(3)]
        slopes[(name,qv,iv)]=((av[2]-av[0])/(Zc[2]-Zc[0]),(bv[2]-bv[0])/(Zc[2]-Zc[0]))
        ax[row,0].plot(Zc,av,marker="o",ms=7,label=name,**st); ax[row,1].plot(Zc,bv,marker="o",ms=7,label=name,**st)
    ax[row,0].set_title(f"Ask depth, $q={qv}$, $i={iv:g}$"); ax[row,1].set_title(f"Bid depth, $q={qv}$, $i={iv:g}$")
    for a in ax[row]: a.axhline(0,color="k",lw=.8,alpha=.3); a.set_ylabel("signed depth"); a.set_xticks(Zc)
for a in ax[-1]: a.set_xlabel("regime level $z_r$")
ax[0,0].legend(loc="upper left",framealpha=.9); fig.tight_layout(); fig.savefig(f"{OUT}/regime_fast_alpha_depths_vs_z_v2.png",dpi=170); plt.close(fig); print("wrote vs_z v2")
print("z-slopes (ask, bid) per (model,q,i):")
for k,v in slopes.items(): print(f"  {k}: ask {v[0]:+.4f}  bid {v[1]:+.4f}")
print("A_R =", getattr(R.P,"A_R",None) or "(see module)")
