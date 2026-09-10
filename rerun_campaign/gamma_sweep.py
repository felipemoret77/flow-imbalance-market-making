#!/usr/bin/env python3
"""gamma-sweep (curvature-scaled, relative): at the paper calibration with gamma
varied, measure relative depth RMSE of (a) sc-cubic vs SRN RHJB, (b) sc-cubic vs
exact-CARA RHJB, (c) CARA-aware sc-cubic vs exact-CARA RHJB.

Metric:  Q_gamma = round[(-1.2,0,2.4,3.4)/sqrt(r_G)]   (curvature-scaled grid)
         RelRMSE = ||d_cf - d_ref||_{mu_I,Q} / ||d_ref||_{mu_I,Q},
         mu_I = OU stationary weight on i in [-6,6], uniform over Q_gamma.
Writes logs/gamma_sweep.log, csv/gamma_sweep_scaled.csv, plots/gamma_sweep.png.
"""
import os, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parent.parent
LOG = open(ROOT/"rerun_campaign"/"logs"/"gamma_sweep.log", "w", buffering=1)
CSV = open(ROOT/"rerun_campaign"/"csv"/"gamma_sweep_scaled.csv", "w", buffering=1)
def log(m): print(m); LOG.write(m+"\n")
CSV.write("gamma,qgrid,sc_vs_srn_rel,sc_vs_cara_rel,caraaware_vs_cara_rel\n")
GAMMAS = [0.005, 0.01, 0.02, 0.03, 0.035, 0.04, 0.05, 0.1]
yy = np.linspace(-6, 6, 241)

def load(gm):
    os.environ["ASOU5_ALPHA_GAMMA"] = str(gm)
    os.environ["ASOU5_ALPHA_Y_MAX"] = "6.0"; os.environ["ASOU5_ALPHA_NY"] = "121"
    for m in list(sys.modules):
        if "alpha_reduced" in m: del sys.modules[m]
    sys.path.insert(0, str(ROOT/"alpha_studies"))
    import alpha_reduced_cole_hopf as A
    return A

def solve_cara(A, mt=3000.0, tol=2e-8, ce=20):
    P = A.P; inv_e = A.build_inventory_exponentials(A.ALPHA_BAR)
    u = A.normalize_common(np.zeros((A.NQ, P.ny))); prev = None
    def oh(u):
        sh = np.max(u, axis=1, keepdims=True)
        z = np.exp(np.clip(-P.gamma*(u-sh), -700, 700)) @ A.OU_E_HALF.T
        return sh - np.log(np.maximum(z, 1e-300))/P.gamma
    for s in range(1, int(mt/P.dt_hjb)+1):
        u = oh(u); u = A.alpha_inventory_step(u, inv_e); u = oh(u); u = A.normalize_common(u)
        if s % ce == 0:
            a, b = A.depth_grid_from_u(u)
            if prev is not None and max(np.nanmax(np.abs(a-prev[0])), np.nanmax(np.abs(b-prev[1]))) < tol: break
            prev = (a, b)
    return u

def scfam(A, q, y, cara):
    P = A.P; k = P.k; rG = 2*P.p_glft; D0 = P.delta0; SIG2 = P.sigma_price**2
    qf = np.asarray(q, float); y = np.asarray(y, float)
    tf, ap, g = A.toy_h_flow_alpha_components(y)
    th = y-k*tf; r = rG*np.sqrt(np.cosh(y)/np.cosh(th)); m = r/rG
    sh = m*tf + (1+g*(m-1))*ap
    if cara:
        e2 = 1e-4; tf2, ap2, g2 = A.toy_h_flow_alpha_components(y+e2); th2 = (y+e2)-k*tf2
        r2 = rG*np.sqrt(np.cosh(y+e2)/np.cosh(th2)); m2 = r2/rG
        sh2 = m2*tf2 + (1+g2*(m2-1))*ap2
        r = r*np.sqrt(1.0 + (P.eta**2*((sh2-sh)/e2)**2)/SIG2)
    thf = y-k*sh; s = (r**2/6)*np.tanh(thf)
    a = D0+sh-(r/(2*k))*(2*qf-1)+(s/k)*(qf**2-qf+1/3)
    b = D0-sh+(r/(2*k))*(2*qf+1)-(s/k)*(qf**2+qf+1/3)
    return np.where(q > -P.qmax, a, np.inf), np.where(q < P.qmax, b, np.inf)

def rel(A, u, fn, QS, w):
    rh = lambda q, y: A.y_aware_depths(u, np.asarray(q, int), np.asarray(y, float))
    E, D, W = [], [], []
    for qv in QS:
        qa = np.full_like(yy, qv); x, y2 = fn(qa, yy); ar, br = rh(qa, yy)
        for e, r in ((x, ar), (y2, br)):
            g = np.isfinite(e) & np.isfinite(r)
            E.append((e-r)[g]); D.append(r[g]); W.append(w[g])
    e = np.concatenate(E); d = np.concatenate(D); ww = np.concatenate(W)
    return float(np.sqrt(np.sum(ww*e**2)/np.sum(ww))/np.sqrt(np.sum(ww*d**2)/np.sum(ww)))

log("gamma-sweep, curvature-scaled grid, RELATIVE RMSE")
log(f"{'gamma':>6} {'qgrid':>18} {'sc/SRN':>7} {'sc/CARA':>8} {'aware/CARA':>10}")
rows = []
for gm in GAMMAS:
    A = load(gm); P = A.P; rG = 2*P.p_glft
    sq = 1/np.sqrt(rG); QS = [int(round(f*sq)) for f in (-1.2, 0, 2.4, 3.4)]
    sig = P.eta/np.sqrt(2*P.beta); w = np.exp(-yy**2/(2*sig**2)); w /= w.sum()
    u_s, _ = A.solve_ou_ergodic(A.ALPHA_BAR); u_c = solve_cara(A)
    a1 = rel(A, u_s, lambda q, y: scfam(A, q, y, False), QS, w)
    a2 = rel(A, u_c, lambda q, y: scfam(A, q, y, False), QS, w)
    a3 = rel(A, u_c, lambda q, y: scfam(A, q, y, True), QS, w)
    log(f"{gm:>6.3f} {str(QS):>18} {100*a1:>6.1f}% {100*a2:>7.1f}% {100*a3:>9.1f}%")
    CSV.write(f'{gm},"{QS}",{a1:.5f},{a2:.5f},{a3:.5f}\n')
    rows.append((gm, a1, a2, a3))
g = [r[0] for r in rows]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(g, [100*r[1] for r in rows], "o-", label="sc-cubic vs SRN RHJB")
ax.plot(g, [100*r[2] for r in rows], "s--", label="sc-cubic vs exact-CARA RHJB")
ax.plot(g, [100*r[3] for r in rows], "^--", label="CARA-aware vs exact-CARA RHJB")
ax.axvline(0.01, color="k", lw=.8, alpha=.4); ax.set_xscale("log")
ax.set_xlabel(r"$\gamma$"); ax.set_ylabel("relative depth RMSE (%)")
ax.legend(fontsize=9); ax.grid(alpha=.3)
ax.set_title("Curvature-scaled relative RMSE vs gamma (baseline marked)")
fig.tight_layout(); fig.savefig(ROOT/"rerun_campaign"/"plots"/"gamma_sweep.png", dpi=150)
log("wrote csv/gamma_sweep_scaled.csv and plots/gamma_sweep.png")
