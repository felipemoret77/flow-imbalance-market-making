#!/usr/bin/env python3
"""Exact-CARA reduced-HJB solver and the SRN-vs-CARA depth-gap diagnostic.

The reported RHJBs drop the signal-risk term -gamma eta^2/2 (d_i u)^2.  That term
is absorbed EXACTLY by a Cole-Hopf transform in the signal variable: with
z = e^{-gamma u}, the OU signal sub-step becomes linear (d_tau z = L_I z), while
the inventory sub-step keeps its own Cole-Hopf form w = e^{k u}.  So the exact
CARA reduced equation is solved by the SAME Strang scheme (signal half-step /
full inventory step / signal half-step) with the signal half-step applied to z.

This script solves both the signal-risk-neutral (SRN) and exact-CARA reduced
HJBs for the toy and fast-alpha engines, and reports
    RMSE(delta_SRN, delta_CARA)   (uniform and OU-stationary-weighted)
on |i|<=6, q in {-5,0,10,15}.  Reproduces the numbers in Appendix
"An elementary CARA-aware curvature variant".
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
QS = (-5, 0, 10, 15)


def _weighted(engine, u_a, u_b, depth_fn):
    yy = np.linspace(-6.0, 6.0, 241)
    sig = engine.P.eta / np.sqrt(2.0 * engine.P.beta)
    w = np.exp(-yy ** 2 / (2.0 * sig ** 2)); w /= w.sum()
    E, W = [], []
    for qv in QS:
        qa = np.full_like(yy, qv)
        aA, bA = depth_fn(u_a, qa, yy)
        aB, bB = depth_fn(u_b, qa, yy)
        for dA, dB in ((aA, aB), (bA, bB)):
            g = np.isfinite(dA) & np.isfinite(dB)
            E.append((dA - dB)[g]); W.append(w[g])
    e = np.concatenate(E); ww = np.concatenate(W)
    return float(np.sqrt(np.mean(e ** 2))), float(np.sqrt(np.sum(ww * e ** 2) / np.sum(ww)))


def run_toy():
    sys.path.insert(0, str(ROOT))
    import cara_glft_yaware_study as C
    P = C.P
    ou_half = C.build_ou_step(0.5 * P.dt_hjb)
    inv_steps = C.precompute_y_inventory_steps()

    def ou_half_cara(u):
        sh = np.max(u, axis=1, keepdims=True)
        z = np.exp(np.clip(-P.gamma * (u - sh), -700, 700)) @ ou_half.T
        return sh - np.log(np.maximum(z, 1e-300)) / P.gamma

    def solve(cara):
        u = C.normalize_u(np.zeros((C.NQ, P.ny))); prev = None
        for step in range(1, int(round(P.T / P.dt_hjb)) + 1):
            u = ou_half_cara(u) if cara else u @ ou_half.T
            shift = np.max(u, axis=0)
            wv = np.exp(np.clip(P.k * (u - shift[None, :]), -700, 700))
            wn = np.empty_like(wv)
            for j in range(P.ny):
                wn[:, j] = inv_steps[j] @ wv[:, j]
            u = np.log(np.maximum(wn, 1e-300)) / P.k + shift[None, :]
            u = ou_half_cara(u) if cara else u @ ou_half.T
            u = C.normalize_u(u)
            if step % P.check_every == 0:
                a, b = C.depth_grid_from_u(u)
                if prev is not None and max(np.nanmax(np.abs(a - prev[0])), np.nanmax(np.abs(b - prev[1]))) < P.hjb_tol:
                    break
                prev = (a, b)
        return u

    dep = lambda u, q, y: C.y_aware_hjb_depths(u, np.asarray(q, int), np.asarray(y, float))
    return _weighted(C, solve(False), solve(True), dep)


def run_fast_alpha():
    sys.path.insert(0, str(ROOT / "alpha_studies"))
    import alpha_reduced_cole_hopf as A
    P = A.P

    def ou_half_cara(u):
        sh = np.max(u, axis=1, keepdims=True)
        z = np.exp(np.clip(-P.gamma * (u - sh), -700, 700)) @ A.OU_E_HALF.T
        return sh - np.log(np.maximum(z, 1e-300)) / P.gamma

    def solve(cara, max_tau=3000.0, tol=2e-8, ce=20):
        inv_e = A.build_inventory_exponentials(A.ALPHA_BAR)
        u = A.normalize_common(np.zeros((A.NQ, P.ny))); prev = None
        for step in range(1, int(round(max_tau / P.dt_hjb)) + 1):
            u = ou_half_cara(u) if cara else u @ A.OU_E_HALF.T
            u = A.alpha_inventory_step(u, inv_e)
            u = ou_half_cara(u) if cara else u @ A.OU_E_HALF.T
            u = A.normalize_common(u)
            if step % ce == 0:
                a, b = A.depth_grid_from_u(u)
                if prev is not None and max(np.nanmax(np.abs(a - prev[0])), np.nanmax(np.abs(b - prev[1]))) < tol:
                    break
                prev = (a, b)
        return u

    dep = lambda u, q, y: A.y_aware_depths(u, np.asarray(q, int), np.asarray(y, float))
    return _weighted(A, solve(False), solve(True), dep)


def main():
    tu, ts = run_toy()
    fu, fs = run_fast_alpha()
    print("RMSE(delta_SRN, delta_CARA) on |i|<=6, q in {-5,0,10,15}:")
    print(f"  toy        : uniform {tu:.5f}   stationary-weighted {ts:.5f}")
    print(f"  fast-alpha : uniform {fu:.5f}   stationary-weighted {fs:.5f}")
    out = ROOT / "imagens_tex" / "alpha_studies_T10000_qmax60" / "exact_cara_gap.csv"
    out.write_text("model,srn_cara_uniform,srn_cara_statwt\n"
                   f"toy,{tu:.6f},{ts:.6f}\nfast-alpha,{fu:.6f},{fs:.6f}\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
