#!/usr/bin/env python3
"""Gap-checks for the audit:
#1  PDE-domain truncation: solve the I-aware RHJB on [-6,6] (NY=121) and on
    [-9,9] (NY=181, same spacing), compare depths on the common grid.
#2  Cross-validate the separate fig6/7/8 solver path
    (y_aware_depths_y0_for_params) against solve_y_aware_hjb at the baseline.
Run AFTER setting ASOU5_CARA_Y_MAX/NY via env in subprocess mode, or as parent
(default) to orchestrate both solves.
"""
import os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)

if len(sys.argv) > 1 and sys.argv[1] == "solve":
    # child: solve with current env, dump depths for q in a set, on its own grid
    sys.path.insert(0, PROJ)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJ, ".mplconfig"))
    import cara_glft_yaware_study as st
    u_y, info = st.solve_y_aware_hjb()
    Y = st.Y
    qs = [-15, -10, -5, 0, 5, 10, 15]
    out = {"y": Y, "info_steps": info.get("steps", -1)}
    for q in qs:
        qa = np.full(Y.shape, q, dtype=int)
        ask, bid = st.y_aware_hjb_depths(u_y, qa, Y)
        out[f"ask_{q}"] = ask
        out[f"bid_{q}"] = bid
    # gap#2: the fig6/7/8 path at baseline (only on the default grid run)
    if abs(Y[-1] - 6.0) < 1e-9:
        q_y, ask_y0, bid_y0 = st.y_aware_depths_y0_for_params(
            st.P.sigma, st.P.bar_lambda, st.P.k, st.P.gamma)
        out["fig_path_q"] = q_y
        out["fig_path_ask_y0"] = ask_y0
        out["fig_path_bid_y0"] = bid_y0
        # main-path depths at i=0 for same q
        i0 = np.zeros(1)
        ask0 = np.array([st.y_aware_hjb_depths(u_y, np.array([int(q)]), i0)[0][0]
                         for q in q_y])
        bid0 = np.array([st.y_aware_hjb_depths(u_y, np.array([int(q)]), i0)[1][0]
                         for q in q_y])
        out["main_path_ask_y0"] = ask0
        out["main_path_bid_y0"] = bid0
    np.savez(sys.argv[2], **out)
    print("child done:", sys.argv[2], "steps:", out["info_steps"])
    sys.exit(0)

# parent: run the two solves sequentially, then compare
env6 = dict(os.environ)
env9 = dict(os.environ, ASOU5_CARA_Y_MAX="9.0", ASOU5_CARA_NY="181")
f6, f9 = os.path.join(HERE, "trunc6.npz"), os.path.join(HERE, "trunc9.npz")
for env, f in [(env6, f6), (env9, f9)]:
    r = subprocess.run([sys.executable, __file__, "solve", f], env=env,
                       capture_output=True, text=True)
    print(r.stdout.strip()); print(r.stderr[-500:] if r.returncode else "", end="")
    r.check_returncode()

d6, d9 = np.load(f6), np.load(f9)
y6, y9 = d6["y"], d9["y"]
# common grid: y6 subset of y9 (both spacing 0.1)
idx9 = np.searchsorted(y9, y6)
assert np.allclose(y9[idx9], y6, atol=1e-9)

print("\n=== Gap#1: domain truncation [-6,6] vs [-9,9], depth differences ===")
print(f"{'q':>4} {'max|d ask|':>11} {'max|d bid|':>11} {'@|i|<=4':>9} {'@i=0':>9}")
rows = []
for q in [-15, -10, -5, 0, 5, 10, 15]:
    da = np.abs(d6[f"ask_{q}"] - d9[f"ask_{q}"][idx9])
    db = np.abs(d6[f"bid_{q}"] - d9[f"bid_{q}"][idx9])
    inner = np.abs(y6) <= 4.0
    at0 = np.abs(y6) < 1e-9
    m = max(da.max(), db.max())
    mi = max(da[inner].max(), db[inner].max())
    m0 = max(da[at0].max(), db[at0].max())
    rows.append((q, m, mi, m0))
    print(f"{q:>4} {da.max():>11.5f} {db.max():>11.5f} {mi:>9.5f} {m0:>9.2e}")

print("\n=== Gap#2: fig6/7/8 solver path vs main solver, depths at i=0 ===")
qf = d6["fig_path_q"]
da = np.abs(d6["fig_path_ask_y0"] - d6["main_path_ask_y0"])
db = np.abs(d6["fig_path_bid_y0"] - d6["main_path_bid_y0"])
fin = np.isfinite(da) & np.isfinite(db)
print(f"max|d ask| = {da[fin].max():.3e}   max|d bid| = {db[fin].max():.3e} over q in [{qf.min()},{qf.max()}]")
sp_q0 = d6["fig_path_ask_y0"][list(qf).index(0)] + d6["fig_path_bid_y0"][list(qf).index(0)]
print(f"fig-path spread at q=0, i=0, gamma=0.01: {sp_q0:.5f}  (sanity ~1.0285)")
