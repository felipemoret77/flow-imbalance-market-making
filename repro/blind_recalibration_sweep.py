"""Signal-blind GLFT ablation: sweep the maker's inventory-risk curvature only.
Mirrors the fast-alpha simulator loop of alpha_studies/alpha_reduced_cole_hopf.py exactly
(same primitives, same fill law, same capped Poisson draws). The penalized diagnostic always
uses the TRUE phi = gamma*sigma^2/2; only the GLFT quoting curvature p_glft is scaled by m."""
import os, sys, json, numpy as np
sys.path.insert(0, "/Users/felipemoret/Desktop/AS_OU_5/alpha_studies")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import alpha_reduced_cole_hopf as A
P = A.P
NPATH = int(os.environ.get("NPATH", "3000"))
MS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 20.0]

def blind_depths(q, p):
    qf = q.astype(float)
    ask = P.delta0 - (p / P.k) * (2.0 * qf - 1.0)
    bid = P.delta0 + (p / P.k) * (2.0 * qf + 1.0)
    return np.where(q > -P.qmax, ask, np.inf), np.where(q < P.qmax, bid, np.inf)

print("solving I-aware RHJB (reference) ...", flush=True)
y_u, info = A.solve_ou_ergodic(np.zeros_like(A.Y))
print("  converged:", info.get("converged"), "tau", info.get("tau"), flush=True)

names = [f"blind m={m:g}" for m in MS] + ["I-aware RHJB"]
rng = np.random.default_rng(P.seed)
steps = int(round(P.T / P.dt_sim)); sqrt_dt = np.sqrt(P.dt_sim)
n = NPATH
y = np.zeros(n); alpha = np.zeros(n)
z = lambda: {k: np.zeros(n) for k in names}
q = {k: np.zeros(n, dtype=int) for k in names}
spread, adrift, mart, pen_run, absq, fills = z(), z(), z(), z(), z(), z()
minask = {k: np.full(n, np.inf) for k in names}
for step in range(steps):
    lp = P.bar_lambda * (1.0 + np.tanh(y)); lm = P.bar_lambda * (1.0 - np.tanh(y))
    a0 = alpha.copy(); noise = P.sigma_price * sqrt_dt * rng.standard_normal(n)
    for j, name in enumerate(names):
        qb = q[name].copy()
        if name == "I-aware RHJB": ask, bid = A.y_aware_depths(y_u, qb, y)
        else:                      ask, bid = blind_depths(qb, MS[j] * P.p_glft)
        minask[name] = np.minimum(minask[name], np.where(np.isfinite(ask), ask, np.inf))
        af = A.sample_capped_poisson(rng, A.fill_count_mean(lp, ask), qb + P.qmax)
        bf = A.sample_capped_poisson(rng, A.fill_count_mean(lm, bid), P.qmax - qb)
        spread[name] += np.where(np.isfinite(ask), ask, 0.0) * af + np.where(np.isfinite(bid), bid, 0.0) * bf
        adrift[name] += qb.astype(float) * a0 * P.dt_sim
        mart[name]   += qb.astype(float) * noise
        q[name] += -af + bf
        fills[name] += af.astype(float) + bf.astype(float)
        pen_run[name] += P.phi * qb.astype(float) ** 2 * P.dt_sim   # TRUE phi, never rescaled
        absq[name] += np.abs(qb) * P.dt_sim
    ab = rng.poisson(lp * P.dt_sim); as_ = rng.poisson(lm * P.dt_sim)
    alpha += -P.zeta*alpha*P.dt_sim + P.eta_alpha*sqrt_dt*rng.standard_normal(n) + P.epsilon*(ab-as_)
    y += -P.beta*y*P.dt_sim + P.eta*sqrt_dt*rng.standard_normal(n)
    if step % 20000 == 0: print(f"  step {step}/{steps}", flush=True)

out = {}
ref = None
print(f"\n{'policy':16s} {'penalized':>11s} {'se':>6s} {'total':>10s} {'spread':>9s} {'alphadrift':>11s} {'E|Q|':>7s} {'fills':>8s} {'min ask':>8s}")
for name in names:
    tot = spread[name] + adrift[name] + mart[name]
    pen = tot - pen_run[name] - P.alpha_l * q[name].astype(float) ** 2
    if name == "I-aware RHJB": ref = pen
    out[name] = dict(pen=float(pen.mean()), pen_se=float(pen.std(ddof=1)/np.sqrt(n)),
                     total=float(tot.mean()), spread=float(spread[name].mean()/1.0),
                     adrift=float(adrift[name].mean()), absq=float(absq[name].mean()/P.T),
                     fills=float(fills[name].mean()), minask=float(np.mean(minask[name])))
    o = out[name]
    print(f"{name:16s} {o['pen']:11.1f} {o['pen_se']:6.1f} {o['total']:10.1f} {o['spread']:9.1f} {o['adrift']:11.1f} {o['absq']:7.2f} {o['fills']:8.0f} {o['minask']:8.2f}")
print("\npaired vs I-aware RHJB (same environment paths):")
for name in names[:-1]:
    tot = spread[name] + adrift[name] + mart[name]
    pen = tot - pen_run[name] - P.alpha_l * q[name].astype(float) ** 2
    d = pen - ref
    print(f"  {name:14s} d={d.mean():+9.1f}  paired se {d.std(ddof=1)/np.sqrt(n):5.1f}")
json.dump({"n_paths": n, "results": out}, open(os.environ.get("OUTJSON", "/tmp/blind_sweep.json"), "w"), indent=1)
