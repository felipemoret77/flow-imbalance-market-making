"""Paired PnL contrast between the signal-risk-neutral RHJB and the exact-CARA RHJB (fast-alpha).

Remark 4.2 quantifies the signal-risk-neutral simplification only in posted depths.  This runs the
two RHJB policies on the SAME fast-alpha environment paths and reports the paired difference in the
penalized diagnostic AND in the CARA certainty equivalent, which is the objective the policies
actually derive from.  The exact-CARA solve is the psi = e^{-gamma u} Strang variant of
closed_form_tests/exact_cara_solver.py; the penalized diagnostic always uses the true gamma*sigma^2/2."""
import os, sys, json, numpy as np
sys.path.insert(0, "/Users/felipemoret/Desktop/AS_OU_5/alpha_studies")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import alpha_reduced_cole_hopf as A
P = A.P
NPATH = int(os.environ.get("NPATH", "4000"))

def ou_half_cara(u):
    sh = np.max(u, axis=1, keepdims=True)
    z = np.exp(np.clip(-P.gamma * (u - sh), -700, 700)) @ A.OU_E_HALF.T
    return sh - np.log(np.maximum(z, 1e-300)) / P.gamma

def solve(cara, max_tau=3000.0, tol=2e-8, ce=20):
    inv_e = A.build_inventory_exponentials(A.ALPHA_BAR)
    u = A.normalize_common(np.zeros((A.NQ, P.ny))); prev = None; tau = None
    for step in range(1, int(round(max_tau / P.dt_hjb)) + 1):
        u = ou_half_cara(u) if cara else u @ A.OU_E_HALF.T
        u = A.alpha_inventory_step(u, inv_e)
        u = ou_half_cara(u) if cara else u @ A.OU_E_HALF.T
        u = A.normalize_common(u)
        if step % ce == 0:
            a, b = A.depth_grid_from_u(u)
            if prev is not None and max(np.nanmax(np.abs(a - prev[0])), np.nanmax(np.abs(b - prev[1]))) < tol:
                tau = step * P.dt_hjb; break
            prev = (a, b)
    return u, tau

print("solving SRN and exact-CARA fast-alpha RHJBs ...", flush=True)
u_srn, tau_s = solve(False); print("  SRN  tau =", tau_s, flush=True)
u_cara, tau_c = solve(True);  print("  CARA tau =", tau_c, flush=True)

names = ["Alpha/I-aware RHJB (SRN)", "Alpha/I-aware RHJB (exact CARA)"]
U = {names[0]: u_srn, names[1]: u_cara}
rng = np.random.default_rng(P.seed)
steps = int(round(P.T / P.dt_sim)); sqrt_dt = np.sqrt(P.dt_sim); n = NPATH
y = np.zeros(n); alpha = np.zeros(n)
z = lambda: {k: np.zeros(n) for k in names}
q = {k: np.zeros(n, dtype=int) for k in names}
spread, adrift, mart, pen_run, absq, fills = z(), z(), z(), z(), z(), z()
for step in range(steps):
    lp = P.bar_lambda * (1.0 + np.tanh(y)); lm = P.bar_lambda * (1.0 - np.tanh(y))
    a0 = alpha.copy(); noise = P.sigma_price * sqrt_dt * rng.standard_normal(n)
    for name in names:
        qb = q[name].copy()
        ask, bid = A.y_aware_depths(U[name], qb, y)
        af = A.sample_capped_poisson(rng, A.fill_count_mean(lp, ask), qb + P.qmax)
        bf = A.sample_capped_poisson(rng, A.fill_count_mean(lm, bid), P.qmax - qb)
        spread[name] += np.where(np.isfinite(ask), ask, 0.0) * af + np.where(np.isfinite(bid), bid, 0.0) * bf
        adrift[name] += qb.astype(float) * a0 * P.dt_sim
        mart[name]   += qb.astype(float) * noise
        q[name] += -af + bf
        fills[name] += af.astype(float) + bf.astype(float)
        pen_run[name] += P.phi * qb.astype(float) ** 2 * P.dt_sim
        absq[name] += np.abs(qb) * P.dt_sim
    ab = rng.poisson(lp * P.dt_sim); as_ = rng.poisson(lm * P.dt_sim)
    alpha += -P.zeta*alpha*P.dt_sim + P.eta_alpha*sqrt_dt*rng.standard_normal(n) + P.epsilon*(ab-as_)
    y += -P.beta*y*P.dt_sim + P.eta*sqrt_dt*rng.standard_normal(n)
    if step % 20000 == 0: print(f"  step {step}/{steps}", flush=True)

g = P.gamma
CE = lambda x: float(-np.log(np.mean(np.exp(-g * x))) / g)
res, TOT, PEN = {}, {}, {}
print(f"\n{'policy':34s} {'penalized':>11s} {'se':>6s} {'total':>10s} {'CE':>9s} {'E|Q|':>7s} {'fills':>8s}")
for name in names:
    tot = spread[name] + adrift[name] + mart[name]
    pen = tot - pen_run[name] - P.alpha_l * q[name].astype(float) ** 2
    TOT[name], PEN[name] = tot, pen
    res[name] = dict(pen=float(pen.mean()), pen_se=float(pen.std(ddof=1)/np.sqrt(n)),
                     total=float(tot.mean()), ce=CE(tot), absq=float(absq[name].mean()/P.T),
                     fills=float(fills[name].mean()))
    r = res[name]
    print(f"{name:34s} {r['pen']:11.1f} {r['pen_se']:6.1f} {r['total']:10.1f} {r['ce']:9.1f} {r['absq']:7.2f} {r['fills']:8.0f}")
d = PEN[names[1]] - PEN[names[0]]
print(f"\npaired penalized (exact CARA minus SRN): {d.mean():+.2f}  paired se {d.std(ddof=1)/np.sqrt(n):.2f}"
      f"  CI95 [{d.mean()-1.96*d.std(ddof=1)/np.sqrt(n):+.2f}, {d.mean()+1.96*d.std(ddof=1)/np.sqrt(n):+.2f}]")
dp = TOT[names[1]] - TOT[names[0]]
print(f"paired total PnL:                       {dp.mean():+.2f}  paired se {dp.std(ddof=1)/np.sqrt(n):.2f}")
b = np.random.default_rng(20260910).integers(0, n, size=(2000, n))
bd = np.array([CE(TOT[names[1]][j]) - CE(TOT[names[0]][j]) for j in b])
print(f"paired Delta CE (2000 bootstrap):       {CE(TOT[names[1]])-CE(TOT[names[0]]):+.2f}  boot se {bd.std(ddof=1):.2f}"
      f"  CI95 [{np.percentile(bd,2.5):+.2f}, {np.percentile(bd,97.5):+.2f}]")
res["paired_penalized_mean"] = float(d.mean()); res["paired_penalized_se"] = float(d.std(ddof=1)/np.sqrt(n))
res["paired_delta_ce"] = CE(TOT[names[1]])-CE(TOT[names[0]]); res["paired_delta_ce_boot_se"] = float(bd.std(ddof=1))
res["n_paths"] = n
json.dump(res, open(os.environ.get("OUTJSON", "/tmp/exact_cara_paired.json"), "w"), indent=1)
