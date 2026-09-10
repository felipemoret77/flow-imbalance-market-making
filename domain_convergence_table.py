#!/usr/bin/env python3
"""Domain-convergence table: cubic closed-form RMSE vs RHJB, uniform and
stationary-weighted, at solve domains [-6,6] and [-9,9], evaluated on the common
interior band |i|<=6.  Toy uses the centred OU weight mu=N(0,eta^2/2beta); the
regime uses the JOINT invariant weight p_r(i) of (I,R) (per-regime shifted
mixture), estimated from a simulation of the modulated OU."""
import os, sys, pathlib
import numpy as np
trapz = getattr(np, "trapezoid", None) or np.trapz

ROOT = pathlib.Path(__file__).resolve().parent


def configured_output_dir(name, default, relative_root):
    """Resolve a study output override using that study's script directory."""
    out = pathlib.Path(os.environ.get(name, default))
    return out if out.is_absolute() else relative_root / out

# ---------- TOY ----------
def toy_row(ymax, ny):
    os.environ["ASOU5_CARA_Y_MAX"] = str(ymax); os.environ["ASOU5_CARA_NY"] = str(ny)
    for m in list(sys.modules):
        if "cara_glft" in m: del sys.modules[m]
    root_str = str(ROOT)
    if root_str not in sys.path: sys.path.insert(0, root_str)
    import cara_glft_yaware_study as C
    u, _ = C.solve_y_aware_hjb()
    yy = np.linspace(-6, 6, 241); QS = (-5, 0, 10, 15)
    b, e = C.P.beta, C.P.eta; w = np.exp(-yy**2/(2*(e/np.sqrt(2*b))**2)); w /= w.sum()
    def rmse(weights):
        E, W = [], []
        for q in QS:
            qa = np.full_like(yy, q)
            a, bd = C.y_curvature_refined_depths(qa.astype(int), yy)
            ar, br = C.y_aware_hjb_depths(u, qa.astype(int), yy)
            for x, r in ((a, ar), (bd, br)):
                g = np.isfinite(x) & np.isfinite(r); E.append((x-r)[g]); W.append((weights if weights is not None else np.ones_like(yy))[g])
        x = np.concatenate(E); ww = np.concatenate(W); return np.sqrt(np.sum(ww*x**2)/np.sum(ww))
    return rmse(None), rmse(w)

print("TOY cubic-skew RMSE vs RHJB (eval |i|<=6):")
for ym, ny in [(6.0, 121), (9.0, 181)]:
    ru, rs = toy_row(ym, ny)
    print(f"  solve [-{ym:g},{ym:g}]:  uniform {ru:.4f}   OU-weighted {rs:.4f}")

# ---------- FAST-ALPHA (centred OU weight) ----------
def fast_alpha_row(ymax, ny):
    os.environ["ASOU5_ALPHA_Y_MAX"] = str(ymax); os.environ["ASOU5_ALPHA_NY"] = str(ny)
    for m in list(sys.modules):
        if "alpha_reduced" in m: del sys.modules[m]
    sys.path.insert(0, str(ROOT / "alpha_studies"))
    import alpha_reduced_cole_hopf as A
    u, _ = A.solve_ou_ergodic(A.ALPHA_BAR)
    yy = np.linspace(-6, 6, 241); QS = (-5, 0, 10, 15)
    b, e = A.P.beta, A.P.eta; w = np.exp(-yy**2/(2*(e/np.sqrt(2*b))**2)); w /= w.sum()
    def rmse(weights):
        E, W = [], []
        for q in QS:
            qa = np.full_like(yy, q)
            a, bd = A.toy_h_sc_cubic_depths(qa.astype(float), yy)
            ar, br = A.y_aware_depths(u, qa.astype(int), yy)
            for x, r in ((a, ar), (bd, br)):
                g = np.isfinite(x) & np.isfinite(r); E.append((x-r)[g]); W.append((weights if weights is not None else np.ones_like(yy))[g])
        x = np.concatenate(E); ww = np.concatenate(W); return np.sqrt(np.sum(ww*x**2)/np.sum(ww))
    return rmse(None), rmse(w)

print("FAST-ALPHA self-consistent-cubic RMSE vs RHJB (eval |i|<=6):")
for ym, ny in [(6.0, 121), (9.0, 181)]:
    ru, rs = fast_alpha_row(ym, ny)
    print(f"  solve [-{ym:g},{ym:g}]:  uniform {ru:.4f}   OU-weighted {rs:.4f}")

# ---------- REGIME (joint p_r(i) weight) ----------
REGIME_ROOT = ROOT / "regime_studies"
sys.path.insert(0, str(REGIME_ROOT))
def regime_pr(nbin=241):
    import regime_fast_alpha_ergodic_cole_hopf as R
    from scipy.linalg import expm
    P = R.P; Z = R.Z_LEVELS; Q = R.Q_REGIME; beta = P.beta; eta = P.eta; NR = R.NR
    rng = np.random.default_rng(1)
    NP, NT, burn, dt = 20000, 2500, 700, 0.5
    Pstep = expm(Q*dt); Pc = np.cumsum(Pstep, axis=1)
    sd = eta*np.sqrt((1-np.exp(-2*beta*dt))/(2*beta)); dec = np.exp(-beta*dt)
    r = np.full(NP, 1); I = Z[r].copy()
    edges = np.linspace(-6, 6, nbin+1); ctr = 0.5*(edges[:-1]+edges[1:])
    H = np.zeros((NR, nbin)); pit = np.zeros(NR)
    for t in range(NT):
        rr = rng.random(NP); r = np.clip((rr[:, None] > Pc[r]).sum(1), 0, NR-1)
        I = Z[r] + (I-Z[r])*dec + sd*rng.standard_normal(NP)
        if t > burn:
            for k in range(NR):
                m = r == k; pit[k] += int(m.sum())
                H[k] += np.histogram(I[m], bins=edges)[0]
    pit /= pit.sum()
    H = H / H.sum()                                          # global joint density p(r,i) on the band
    return ctr, H, pit

def regime_row(ymax, ny, ctr, pcond, pi):
    os.environ["ASOU5_REGIME_Y_MAX"] = str(ymax); os.environ["ASOU5_REGIME_NY"] = str(ny)
    for m in list(sys.modules):
        if "regime_fast_alpha" in m: del sys.modules[m]
    import regime_fast_alpha_ergodic_cole_hopf as R
    u, _ = R.solve_regime_ergodic()
    QS = (-5, 0, 10, 15); NR = R.NR
    yy = ctr
    num_u = num_w = den_u = den_w = 0.0
    for rr in range(NR):
        for q in QS:
            qa = np.full_like(yy, q); ra = np.full_like(yy, rr, dtype=int)
            a, b = R.regime_sc_cubic_depths(qa, yy, ra)
            ar, br = R.regime_aware_depths(u, qa.astype(int), yy, ra)
            for x, rf in ((a, ar), (b, br)):
                g = np.isfinite(x) & np.isfinite(rf); d = (x-rf)[g]
                num_u += np.sum(d**2); den_u += g.sum()
                wt = pcond[rr][g]                    # global joint density p(r,i) (already includes pi_r)
                num_w += np.sum(wt*d**2); den_w += np.sum(wt)
    return np.sqrt(num_u/den_u), np.sqrt(num_w/den_w)

print("\nbuilding regime joint weight p_r(i)...")
ctr, pcond, pi = regime_pr()
print(f"  pi={np.round(pi,3)}, E[I|R]={[round(float(np.sum(ctr*pcond[k])/pcond[k].sum()),3) for k in range(len(pi))]}")
print("REGIME self-consistent-cubic RMSE vs RHJB (eval |i|<=6):")
for ym, ny in [(6.0, 121), (9.0, 181)]:
    ru, rw = regime_row(ym, ny, ctr, pcond, pi)
    print(f"  solve [-{ym:g},{ym:g}]:  uniform {ru:.4f}   joint p_r(i)-weighted {rw:.4f}")

# --- dump table to CSV for reproducibility ---
_out_dir = configured_output_dir(
    "ASOU5_REGIME_OUTPUT_DIR",
    "../imagens_tex/regime_studies_T10000_qmax60",
    REGIME_ROOT,
)
_out_dir.mkdir(parents=True, exist_ok=True)
_out = _out_dir / "domain_convergence_table.csv"
_rows = []
for ym, ny in [(6.0, 121), (9.0, 181)]:
    tu, ts = toy_row(ym, ny)
    fu, fs = fast_alpha_row(ym, ny)
    ru, rw = regime_row(ym, ny, ctr, pcond, pi)
    _rows.append((ym, tu, fu, ru, ts, fs, rw))
_out.write_text("solve_ymax,toy_uniform,fast_alpha_uniform,regime_uniform,toy_statwt,fast_alpha_statwt,regime_statwt_joint\n" +
    "\n".join(f"{a:g},{b:.4f},{c:.4f},{d:.4f},{e:.4f},{f:.4f},{g:.4f}" for a,b,c,d,e,f,g in _rows) + "\n")
print(f"wrote {_out}")
