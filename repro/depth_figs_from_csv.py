"""Depth figures redrawn from the committed depth CSVs (toy and fast-alpha, vs imbalance and vs inventory)
plus the introductory OU sample path (same seed/step as toy_intro_diagnostics.py).  Writes the *_v2.png files
used by paper/main.tex to paper/images_final/."""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "imagens_tex"; OUT = REPO / "paper" / "images_final"
plt.rcParams.update({"font.size":15,"axes.titlesize":16,"axes.labelsize":15,"xtick.labelsize":13,"ytick.labelsize":13,"legend.fontsize":13,"axes.grid":True,"grid.alpha":.25})
def load(p): return np.genfromtxt(p, delimiter=",", names=True)
def fin(a): return np.where(np.isfinite(a), a, np.nan)
def rows_fig(d, key, vals, xkey, xlabel, cols, labels, styles, title_fn, out, figsize):
    fig, ax = plt.subplots(len(vals), 2, figsize=figsize, sharex=True)
    ax = np.atleast_2d(ax)
    for r, v in enumerate(vals):
        m = np.isclose(d[key], v); x = d[xkey][m]
        for (ca, cb), lab, st in zip(cols, labels, styles):
            ax[r,0].plot(x, fin(d[ca][m]), label=lab, **st); ax[r,1].plot(x, fin(d[cb][m]), label=lab, **st)
        ax[r,0].set_title(title_fn("Ask", v)); ax[r,1].set_title(title_fn("Bid", v))
        for a in ax[r]: a.axhline(0, color="k", lw=.8, alpha=.3); a.axvline(0, color="k", lw=.6, alpha=.2); a.set_ylabel("signed depth")
    for a in ax[-1]: a.set_xlabel(xlabel)
    ax[0,0].legend(loc="best", framealpha=.9); fig.tight_layout(); fig.savefig(out, dpi=170); plt.close(fig); print("wrote", out)
S_RHJB=dict(ls="-",lw=2.8,color="#111111"); S_HYB=dict(ls="--",lw=2.5,color="#E69F00"); S_GAL=dict(ls=":",lw=2.6,color="#B07AA1")
S_TOY_RHJB=dict(ls="-",lw=2.8,color="#1f77b4"); S_TOY_SC=dict(ls="--",lw=2.5,color="#8c564b"); S_TOY_GAL=dict(ls=":",lw=2.6,color="#d62728")
# B toy depths vs imbalance (q = 0, 3, 5 -- only values stored)
d=load(f"{ROOT}/toy_model_T10000_qmax60/cara_glft_yaware_ansatz_depths_vs_imbalance.csv")
rows_fig(d,"q",[0,3,5],"y","imbalance $i$",[("ask_hjb","bid_hjb"),("ask_cubic","bid_cubic"),("ask_gal","bid_gal")],
         ["$I$-aware RHJB","toy self-consistent cubic","toy Galerkin-cubic"],[S_TOY_RHJB,S_TOY_SC,S_TOY_GAL],
         lambda s,v:f"{s} depth, $q={int(v)}$", f"{OUT}/cara_glft_yaware_ansatz_depths_vs_imbalance_v2.png",(12,11))
# C fast-alpha depths vs imbalance (q = 0, 10, 15)
d=load(f"{ROOT}/alpha_studies_T10000_qmax60/alpha_reduced_galerkin_depths_vs_imbalance.csv")
rows_fig(d,"q",[0,10,15],"y","imbalance $i$",[("ask_hjb","bid_hjb"),("ask_sc","bid_sc"),("ask_gal","bid_gal")],
         ["Alpha/$I$-aware RHJB","Alpha/$I$ hybrid cubic","Alpha/$I$ Galerkin-cubic"],[S_RHJB,S_HYB,S_GAL],
         lambda s,v:f"{s} depth, $q={int(v)}$", f"{OUT}/alpha_reduced_galerkin_depths_vs_imbalance_v2.png",(12,11))
# D fast-alpha depths vs inventory (i = -3, +3), per-axis y-limits
d=load(f"{ROOT}/alpha_studies_T10000_qmax60/alpha_reduced_galerkin_depths_vs_inventory.csv")
rows_fig(d,"y",[-3,3],"q","inventory $q$",[("ask_hjb","bid_hjb"),("ask_sc","bid_sc"),("ask_gal","bid_gal")],
         ["Alpha/$I$-aware RHJB","Alpha/$I$ hybrid cubic","Alpha/$I$ Galerkin-cubic"],[S_RHJB,S_HYB,S_GAL],
         lambda s,v:f"{s} depth, $i={int(v):+d}$", f"{OUT}/alpha_reduced_galerkin_depths_vs_inventory_v2.png",(12,7.6))
# A toy sample path: same OU path as toy_intro_diagnostics.py (seed 54321, T=2000, dt=0.5)
beta,eta,barL,seed=0.0125,0.32,0.9,54321
rng=np.random.default_rng(seed); T,dt=2000.0,0.5; n=int(round(T/dt)); t=np.linspace(0,T,n+1); i=np.zeros(n+1)
for k in range(n): i[k+1]=i[k]-beta*i[k]*dt+eta*np.sqrt(dt)*rng.standard_normal()
la,lb=barL*(1+np.tanh(i)),barL*(1-np.tanh(i)); share=0.5*(1+np.tanh(i))
fig,ax=plt.subplots(3,1,figsize=(11,8.2),sharex=True)
ax[0].plot(t,i,color="#4c72b0",lw=1.6); ax[0].axhline(0,color="k",lw=.9,alpha=.45); ax[0].set_ylabel("imbalance $I_t$")
ax[1].plot(t,la,color="#55a868",lw=1.5,label=r"$\Lambda^{a,0}(I_t)$"); ax[1].plot(t,lb,color="#c44e52",lw=1.5,label=r"$\Lambda^{b,0}(I_t)$")
ax[1].axhline(2*barL,color="#222222",lw=1.3,ls="--",label=r"$\Lambda^{a,0}+\Lambda^{b,0}=2\bar\Lambda$"); ax[1].set_ylabel("zero-depth intensity"); ax[1].set_ylim(-0.05,2.45); ax[1].legend(ncol=3,loc="upper center",framealpha=.9)
ax[2].plot(t,share,color="#dd8452",lw=1.5,label=r"buy share $\frac{1}{2}(1+\tanh I_t)$"); ax[2].axhline(0.88,color="k",lw=1,ls=":"); ax[2].axhline(0.12,color="k",lw=1,ls=":")
frac=np.mean(np.abs(np.tanh(i))>0.76); ax[2].set_ylabel("buy share"); ax[2].set_xlabel("time"); ax[2].set_ylim(-0.02,1.3)
ax[2].legend(loc="upper center",framealpha=.9)
fig.tight_layout(); fig.savefig(f"{OUT}/toy_intro_ou_lambda_sample_path_v2.png",dpi=170); print("wrote sample path v2; one-sided fraction on this path:", round(frac,3))
