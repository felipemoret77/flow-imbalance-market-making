#!/usr/bin/env python3
"""Holdout v2: solve each parameter setting on a domain SCALED to its stationary
width sigma_I=eta/sqrt(2beta) (solve +-3.5 sigma_I, eval +-3 sigma_I), removing the
fixed-grid truncation confound. Isolates true closed-form generalization."""
import os, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parent.parent
LOG = open(ROOT/"rerun_campaign"/"logs"/"param_holdout_scaled.log", "w", buffering=1)
CSV = open(ROOT/"rerun_campaign"/"csv"/"param_holdout_scaled.csv", "w", buffering=1)
def log(m): print(m); LOG.write(m+"\n")
CSV.write("param,value,rel,sigmaI,sc_vs_srn_stat\n")
QS = (-5, 0, 10, 15)
BASE = {"ASOU5_ALPHA_BETA": 0.0125, "ASOU5_ALPHA_ETA": 0.32, "ASOU5_ALPHA_ZETA": 0.25,
        "ASOU5_ALPHA_EPSILON": 0.005, "ASOU5_ALPHA_K": 2.0}
SWEEP = {"ASOU5_ALPHA_BETA": [0.00625, 0.0125, 0.025], "ASOU5_ALPHA_ETA": [0.24, 0.32, 0.42],
         "ASOU5_ALPHA_ZETA": [0.15, 0.25, 0.40], "ASOU5_ALPHA_EPSILON": [0.0025, 0.005, 0.01],
         "ASOU5_ALPHA_K": [1.0, 2.0, 3.0]}

def load():
    for m in list(sys.modules):
        if "alpha_reduced" in m: del sys.modules[m]
    sys.path.insert(0, str(ROOT/"alpha_studies"))
    import alpha_reduced_cole_hopf as A
    return A

def sc(A, q, y):
    P = A.P; k = P.k; rG = 2*P.p_glft; D0 = P.delta0
    qf = np.asarray(q, float); y = np.asarray(y, float)
    tf, ap, g = A.toy_h_flow_alpha_components(y)
    th = y - k*tf; r = rG*np.sqrt(np.cosh(y)/np.cosh(th)); m = r/rG
    sh = m*tf + (1+g*(m-1))*ap; thf = y - k*sh; s = (r**2/6)*np.tanh(thf)
    a = D0+sh-(r/(2*k))*(2*qf-1)+(s/k)*(qf**2-qf+1/3)
    b = D0-sh+(r/(2*k))*(2*qf+1)-(s/k)*(qf**2+qf+1/3)
    return np.where(q > -P.qmax, a, np.inf), np.where(q < P.qmax, b, np.inf)

log("holdout v2 (sigma_I-scaled domains): sc-cubic stationary-weighted RMSE vs RHJB")
res = {}
for pname, vals in SWEEP.items():
    res[pname] = []
    for v in vals:
        for kk, bv in BASE.items(): os.environ[kk] = str(bv)
        os.environ[pname] = str(v); os.environ["ASOU5_ALPHA_GAMMA"] = "0.01"
        beta = float(os.environ["ASOU5_ALPHA_BETA"]); eta = float(os.environ["ASOU5_ALPHA_ETA"])
        sigI = eta/np.sqrt(2*beta); ymax = round(3.5*sigI, 2); ny = int(2*ymax/0.1)+1
        os.environ["ASOU5_ALPHA_Y_MAX"] = str(ymax); os.environ["ASOU5_ALPHA_NY"] = str(ny)
        A = load(); P = A.P
        u, _ = A.solve_ou_ergodic(A.ALPHA_BAR)
        yy = np.linspace(-3*sigI, 3*sigI, 241); w = np.exp(-yy**2/(2*sigI**2)); w /= w.sum()
        rh = lambda q, y: A.y_aware_depths(u, np.asarray(q, int), np.asarray(y, float))
        E, W = [], []
        for qv in QS:
            qa = np.full_like(yy, qv); x, y2 = sc(A, qa, yy); ar, br = rh(qa, yy)
            for a, r in ((x, ar), (y2, br)):
                gg = np.isfinite(a) & np.isfinite(r); E.append((a-r)[gg]); W.append(w[gg])
        e = np.concatenate(E); ww = np.concatenate(W)
        rs = float(np.sqrt(np.sum(ww*e**2)/np.sum(ww)))
        rel = v/BASE[pname]; res[pname].append((rel, rs))
        log(f"  {pname.split('_')[-1]:8s}={v:<8g}(x{rel:.2f}) sigmaI={sigI:.2f} domain=+-{ymax}: stat RMSE {rs:.4f}")
        CSV.write(f"{pname.split('_')[-1]},{v},{rel:.3f},{sigI:.3f},{rs:.5f}\n")

fig, ax = plt.subplots(figsize=(8, 5))
for pname, pts in res.items():
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=pname.split("_")[-1].lower())
ax.axhline(0.049, color="k", ls="--", lw=.8, label="baseline 0.049")
ax.set_xlabel("parameter / baseline"); ax.set_ylabel("stationary-weighted RMSE (sigma_I-scaled domain)")
ax.legend(fontsize=9); ax.grid(alpha=.3); ax.set_title("Closed-form generalization (domain-truncation removed)")
fig.tight_layout(); fig.savefig(ROOT/"rerun_campaign"/"plots"/"param_holdout_scaled.png", dpi=150)
log("wrote plots/param_holdout_scaled.png")
