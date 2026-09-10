#!/usr/bin/env python3
"""SRN closed-form robustness across central model parameters.
Toy (cubic-skew) : beta, eta, gamma.
Regime (sc-cubic): z_star, regime_tau, beta.
Fast-alpha       : eta_alpha (alpha noise; other params already swept).
Each point: re-solve the engine's RHJB with the parameter changed, recompute the
closed form FROM PRIMITIVES (parameter-free), measure stationary-weighted
RELATIVE depth RMSE on sigma_I-scaled domain and curvature-scaled q-grid.
Incremental log + CSV."""
import os, sys, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
LOG = open(ROOT/"rerun_campaign"/"logs"/"srn_robustness_sweep.log", "w", buffering=1)
CSV = open(ROOT/"rerun_campaign"/"csv"/"srn_robustness_sweep.csv", "w", buffering=1)
def log(m): print(m); LOG.write(m+"\n")
CSV.write("model,param,value,rel,relRMSE\n")

def fresh(mod_sub):
    for m in list(sys.modules):
        if mod_sub in m: del sys.modules[m]

def grid_env(prefix, beta, eta):
    sigI = eta/np.sqrt(2*beta); ymax = round(3.5*sigI, 2)
    os.environ[f"{prefix}_Y_MAX"] = str(ymax)
    os.environ[f"{prefix}_NY"] = str(int(2*ymax/0.1)+1)
    return sigI

def relrmse(depth_cf, depth_rh, yy, w, QS):
    E, D, W = [], [], []
    for qv in QS:
        qa = np.full_like(yy, qv)
        a, b = depth_cf(qa, yy); ar, br = depth_rh(qa, yy)
        for e, r in ((a, ar), (b, br)):
            g = np.isfinite(e) & np.isfinite(r)
            E.append((e-r)[g]); D.append(r[g]); W.append(w[g])
    e = np.concatenate(E); d = np.concatenate(D); ww = np.concatenate(W)
    return float(np.sqrt(np.sum(ww*e**2)/np.sum(ww)) / np.sqrt(np.sum(ww*d**2)/np.sum(ww)))

BASE_TOY = {"ASOU5_CARA_BETA": 0.0125, "ASOU5_CARA_ETA": 0.32, "ASOU5_CARA_GAMMA": 0.01}
SW_TOY = {"ASOU5_CARA_BETA": [0.00625, 0.025], "ASOU5_CARA_ETA": [0.24, 0.42], "ASOU5_CARA_GAMMA": [0.005, 0.05]}
log("== TOY cubic-skew vs toy RHJB (relative, scaled domains) ==")
for pn, vals in [("BASELINE", [None])] + list(SW_TOY.items()):
    for v in vals:
        for k2, b in BASE_TOY.items(): os.environ[k2] = str(b)
        if v is not None: os.environ[pn] = str(v)
        beta = float(os.environ["ASOU5_CARA_BETA"]); eta = float(os.environ["ASOU5_CARA_ETA"])
        sigI = grid_env("ASOU5_CARA", beta, eta)
        fresh("cara_glft"); sys.path.insert(0, str(ROOT))
        import cara_glft_yaware_study as C
        u, _ = C.solve_y_aware_hjb()
        rG = 2*C.P.p_glft; sq = 1/np.sqrt(rG)
        QS = [int(round(f*sq)) for f in (-1.2, 0, 2.4, 3.4)]
        yy = np.linspace(-3*sigI, 3*sigI, 241); w = np.exp(-yy**2/(2*sigI**2)); w /= w.sum()
        r = relrmse(lambda q, y: C.y_curvature_refined_depths(q.astype(int), y),
                    lambda q, y: C.y_aware_hjb_depths(u, q.astype(int), y), yy, w, QS)
        lbl = "baseline" if v is None else f"{pn.split('_')[-1]}={v}"
        rel = 1.0 if v is None else v/BASE_TOY[pn]
        log(f"  toy {lbl:16s} relRMSE={100*r:.1f}%")
        CSV.write(f"toy,{lbl},{v if v is not None else ''},{rel:.2f},{r:.5f}\n")

BASE_FA = {"ASOU5_ALPHA_ETA_ALPHA": 0.001}
log("== FAST-ALPHA sc-cubic: eta_alpha sweep ==")
for v in [0.0, 0.001, 0.003]:
    os.environ["ASOU5_ALPHA_ETA_ALPHA"] = str(v)
    os.environ["ASOU5_ALPHA_Y_MAX"] = "6.0"; os.environ["ASOU5_ALPHA_NY"] = "121"
    fresh("alpha_reduced"); sys.path.insert(0, str(ROOT/"alpha_studies"))
    import alpha_reduced_cole_hopf as A
    u, _ = A.solve_ou_ergodic(A.ALPHA_BAR)
    sigI = A.P.eta/np.sqrt(2*A.P.beta)
    yy = np.linspace(-6, 6, 241); w = np.exp(-yy**2/(2*sigI**2)); w /= w.sum()
    r = relrmse(lambda q, y: A.toy_h_sc_cubic_depths(q.astype(float), y),
                lambda q, y: A.y_aware_depths(u, q.astype(int), y), yy, w, (-5, 0, 10, 15))
    log(f"  fast-alpha eta_alpha={v}: relRMSE={100*r:.1f}%  (note: abar indep of eta_alpha)")
    CSV.write(f"fast-alpha,eta_alpha,{v},{v/0.001 if v else 0},{r:.5f}\n")
os.environ["ASOU5_ALPHA_ETA_ALPHA"] = "0.001"

BASE_RG = {"ASOU5_REGIME_Z_STAR": 1.25, "ASOU5_REGIME_TAU": 120.0, "ASOU5_REGIME_BETA": 0.0125}
SW_RG = {"ASOU5_REGIME_Z_STAR": [0.6, 2.0], "ASOU5_REGIME_TAU": [60.0, 240.0], "ASOU5_REGIME_BETA": [0.00625, 0.025]}
log("== REGIME sc-cubic vs regime RHJB (joint-weighted, scaled domain) ==")
for pn, vals in [("BASELINE", [None])] + list(SW_RG.items()):
    for v in vals:
        for k2, b in BASE_RG.items(): os.environ[k2] = str(b)
        if v is not None: os.environ[pn] = str(v)
        beta = float(os.environ["ASOU5_REGIME_BETA"]); eta = 0.32
        sigI = grid_env("ASOU5_REGIME", beta, eta)
        fresh("regime_fast_alpha"); sys.path.insert(0, str(ROOT/"regime_studies"))
        import regime_fast_alpha_ergodic_cole_hopf as R
        u, _ = R.solve_regime_ergodic()
        # per-regime shifted OU weight approx: N(c*z_r, eta^2/2beta), c = beta/(beta+lam)
        c = beta/(beta + R.P.regime_rate)
        yy = np.linspace(-3*sigI, 3*sigI, 201)
        E, D, W = [], [], []
        for rr in range(R.NR):
            wr = np.exp(-(yy - c*R.Z_LEVELS[rr])**2/(2*sigI**2)) * (0.25 if rr != 1 else 0.5)
            for qv in (-5, 0, 10, 15):
                qa = np.full_like(yy, qv); ra = np.full_like(yy, rr, dtype=int)
                a, b2 = R.regime_sc_cubic_depths(qa, yy, ra)
                ar, br = R.regime_aware_depths(u, qa.astype(int), yy, ra)
                for e, r2 in ((a, ar), (b2, br)):
                    g = np.isfinite(e) & np.isfinite(r2)
                    E.append((e-r2)[g]); D.append(r2[g]); W.append(wr[g])
        e = np.concatenate(E); d = np.concatenate(D); ww = np.concatenate(W)
        r = float(np.sqrt(np.sum(ww*e**2)/np.sum(ww))/np.sqrt(np.sum(ww*d**2)/np.sum(ww)))
        lbl = "baseline" if v is None else f"{pn.split('_')[-1]}={v}"
        rel = 1.0 if v is None else v/BASE_RG[pn]
        log(f"  regime {lbl:16s} relRMSE={100*r:.1f}%")
        CSV.write(f"regime,{lbl},{v if v is not None else ''},{rel:.2f},{r:.5f}\n")
log("done.")
