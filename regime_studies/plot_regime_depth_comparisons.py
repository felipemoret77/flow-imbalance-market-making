#!/usr/bin/env python3
"""Paper companion script: deterministic depth-comparison figures for the regime
section (regime_fast_alpha_depths_vs_{i,q,alpha,z}.png) and the quoted depth-RMSE
numbers, reproducible from the repo alone: 3 models
(regime RHJB, Regime Galerkin-cubic, Regime self-consistent cubic) on depth
vs i, vs q, vs alpha, vs z_r.
Saves paper-named PNGs to the regime output dir."""
import os, sys
os.environ.setdefault("MPLCONFIGDIR", "/tmp/asou5_repro/.mpl")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import regime_fast_alpha_ergodic_cole_hopf as R
OUT = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "imagens_tex" / "regime_studies_T10000_qmax60")
P = R.P; Zc = R.Z_LEVELS; a_slope = P.alpha_slope

print("solving regime RHJB for depth figures...")
u, info = R.solve_regime_ergodic(); print("  converged:", info.get("converged"))

def rhjb(q, y, r): return R.regime_aware_depths(u, np.asarray(q,int), np.asarray(y,float), np.asarray(r,int))
def gal(q, y, r):  return R.regime_galerkin_depths(q, y, r)
def sc(q, y, r):   return R.regime_sc_cubic_depths(q, y, r)
def sc_gated(q, y, r): return R.regime_sc_cubic_depths(q, y, r, gate_regime=True)
MODELS = [("Regime-aware RHJB", rhjb, dict(ls="-",lw=2.6,color="#111111")),
          ("Regime Galerkin-cubic", gal, dict(ls=":",lw=2.3,color="#B07AA1")),
          ("Regime self-consistent cubic", sc, dict(ls="--",lw=2.4,color="#1B9E77"))]
RL = {0:"regime - (z=-1.25)", 1:"regime 0", 2:"regime + (z=+1.25)"}
fin = lambda a: np.where(np.isfinite(a), a, np.nan)
yy = np.linspace(-6,6,301); qq = R.Q[np.abs(R.Q)<=30].astype(float)

def rows_fig(xvals, get_xy, xlabel, title, fname, titles):
    fig, ax = plt.subplots(R.NR, 2, figsize=(13,11), sharex=True)
    for r in range(R.NR):
        for name, fn, st in MODELS:
            xa, aa, ba = get_xy(fn, r)
            ax[r,0].plot(xa, fin(aa), label=name, **st); ax[r,1].plot(xa, fin(ba), label=name, **st)
        ax[r,0].set_title(titles(r,"Ask")); ax[r,1].set_title(titles(r,"Bid"))
    for a in ax.ravel(): a.axhline(0,color="k",lw=.8,alpha=.3); a.axvline(0,color="k",lw=.6,alpha=.2); a.set_xlabel(xlabel); a.set_ylabel("signed depth"); a.grid(alpha=.25)
    ax[0,0].legend(fontsize=8); fig.suptitle(title, fontsize=12); fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=170); plt.close(fig); print("wrote", fname)

# vs i (q=10)
q0=10
rows_fig(yy, lambda fn,r:(yy,)+fn(np.full_like(yy,q0),yy,np.full_like(yy,r,dtype=int)),
         "OU state i", f"Regime: depth vs imbalance i (q={q0})", "regime_fast_alpha_depths_vs_i.png",
         lambda r,s:f"{s} depth, q={q0}, {RL[r]}")
# vs q (i=3)
i0=3.0
rows_fig(qq, lambda fn,r:(qq,)+fn(qq,np.full_like(qq,i0),np.full_like(qq,r,dtype=int)),
         "inventory q", f"Regime: depth vs inventory q (i={i0:g})", "regime_fast_alpha_depths_vs_q.png",
         lambda r,s:f"{s} depth, i={i0:g}, {RL[r]}")
# vs alpha (q=10, x = alphabar(i))
abar = a_slope*np.tanh(yy)
rows_fig(abar, lambda fn,r:(abar,)+fn(np.full_like(yy,q0),yy,np.full_like(yy,r,dtype=int)),
         r"conditional alpha $\bar\alpha(i)$", f"Regime: depth vs alpha (q={q0})", "regime_fast_alpha_depths_vs_alpha.png",
         lambda r,s:f"{s} depth, q={q0}, {RL[r]}")

# vs z_r (3 points) at a few (q,i)
combos = [(0,0.0),(10,2.0),(15,-2.0)]
fig, ax = plt.subplots(len(combos),2,figsize=(12,10),sharex=True)
for row,(qv,iv) in enumerate(combos):
    for name, fn, st in MODELS:
        av=[fn(np.array([qv]),np.array([iv]),np.array([r]))[0][0] for r in range(R.NR)]
        bv=[fn(np.array([qv]),np.array([iv]),np.array([r]))[1][0] for r in range(R.NR)]
        ax[row,0].plot(Zc,[x if np.isfinite(x) else np.nan for x in av],marker="o",label=name,**st)
        ax[row,1].plot(Zc,[x if np.isfinite(x) else np.nan for x in bv],marker="o",label=name,**st)
    ax[row,0].set_title(f"Ask depth vs z_r, q={qv}, i={iv:g}"); ax[row,1].set_title(f"Bid depth vs z_r, q={qv}, i={iv:g}")
for a in ax.ravel(): a.axhline(0,color="k",lw=.8,alpha=.3); a.set_xlabel("regime level z_r"); a.set_ylabel("signed depth"); a.grid(alpha=.25)
ax[0,0].legend(fontsize=8); fig.suptitle("Regime: depth vs regime level z_r",fontsize=12); fig.tight_layout()
fig.savefig(f"{OUT}/regime_fast_alpha_depths_vs_z.png",dpi=170); plt.close(fig); print("wrote regime_fast_alpha_depths_vs_z.png")

# RMSE table (vs RHJB) for the caption
def rmse(fn):
    e=[]
    for r in range(R.NR):
        for qv in (-5,0,10,15):
            qa=np.full_like(yy,qv);ra=np.full_like(yy,r,dtype=int)
            a,b=fn(qa,yy,ra);ar,br=rhjb(qa,yy,ra);est=np.concatenate([a,b]);ref=np.concatenate([ar,br])
            g=np.isfinite(est)&np.isfinite(ref);e.append(est[g]-ref[g])
    return np.sqrt(np.mean(np.concatenate(e)**2))
def aff(q, y, r): return R.local_ansatz_depths(np.asarray(q,int), np.asarray(y,float), np.asarray(r,int))
vals = {
    "sc_cubic": rmse(sc),
    "regime_gated_cubic_ablation": rmse(sc_gated),
    "galerkin": rmse(gal),
    "affine": rmse(aff),
}
print("\ndepth RMSE vs RHJB (pooled q in {-5,0,10,15} x 3 regimes):", {k: round(v,4) for k,v in vals.items()})
with open(f"{OUT}/regime_fast_alpha_depth_rmse.csv","w") as fh:
    fh.write("form,pooled_ask_bid_depth_rmse\n")
    for k,v in vals.items(): fh.write(f"{k},{v:.6f}\n")
print(f"wrote {OUT}/regime_fast_alpha_depth_rmse.csv")
