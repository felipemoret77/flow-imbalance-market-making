#!/usr/bin/env python3
"""Robustness of the ergodic RHJB depths to the frozen-I alpha closure.

The body uses the frozen-I drift  abar(i) = (2 barLam eps / zeta) tanh(i).  The exact
stationary I-conditional mean is the bounded odd solution m of the resolvent equation

    (zeta - L_I) m(i) = 2 barLam eps tanh(i),   L_I = -beta i d/di + (eta^2/2) d^2/di^2.

This script solves both the resolvent (tridiagonal) and the two ergodic RHJBs (frozen
source vs exact m), then reports the OU-stationary-weighted L2 depth distance

    Delta_2 = [ (1/(2|Q|)) sum_{q in Q} sum_{l in {a,b}} int (d^l(q,i;abar)-d^l(q,i;m))^2 mu(di) ]^{1/2}

on Q_eval = {-5,0,10,15}, i in [-6,6], with mu = N(0, eta^2/(2 beta)); also the sup
difference and the ratio to the inventory-independent half-spread delta_gamma.
Writes alpha_reduced_robustness.csv into the fast-alpha output directory.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import alpha_reduced_cole_hopf as A  # noqa: E402

trapz = getattr(np, "trapezoid", None) or np.trapz
P = A.P
Y = A.Y
# Reuse the production study's resolved output directory.  This honors
# ASOU5_ALPHA_OUTPUT_DIR (including relative paths resolved from alpha_studies/)
# and preserves its legacy/default behavior.
OUT = A.OUT


def solve_resolvent() -> np.ndarray:
    """Exact stationary conditional mean m on the engine grid Y (tridiagonal solve)."""
    h = Y[1] - Y[0]
    n = Y.size
    d2 = (P.eta ** 2 / 2.0) / h ** 2
    plateau = 2.0 * P.bar_lambda * P.epsilon / P.zeta          # = alpha_slope; m(+-inf)
    mat = np.zeros((n, n))
    rhs = 2.0 * P.bar_lambda * P.epsilon * np.tanh(Y)
    for j in range(1, n - 1):
        drift = P.beta * Y[j] / (2.0 * h)
        mat[j, j - 1] = -drift - d2
        mat[j, j] = P.zeta + 2.0 * d2
        mat[j, j + 1] = drift - d2
    mat[0, 0] = 1.0
    mat[-1, -1] = 1.0
    rhs[0] = -plateau
    rhs[-1] = plateau
    return np.linalg.solve(mat, rhs)


def main() -> None:
    m = solve_resolvent()
    h = Y[1] - Y[0]
    j0 = np.argmin(np.abs(Y))
    m_slope0 = float((m[j0 + 1] - m[j0 - 1]) / (2.0 * h))
    frozen_slope0 = float(P.alpha_slope)                       # 2 barLam eps / zeta

    u_frozen, _ = A.solve_ou_ergodic(A.ALPHA_BAR)
    u_exact, _ = A.solve_ou_ergodic(m)

    def depths(u, q, y):
        return A.y_aware_depths(u, np.asarray(q, int), np.asarray(y, float))

    ii = np.linspace(-6.0, 6.0, 241)
    Q_eval = [-5, 0, 10, 15]
    sigma_I = P.eta / np.sqrt(2.0 * P.beta)
    mu = np.exp(-ii ** 2 / (2.0 * sigma_I ** 2))
    mu = mu / trapz(mu, ii)                                    # probability density on [-6,6]

    sq_acc = 0.0
    sup = 0.0
    for q in Q_eval:
        qa = np.full_like(ii, q)
        af, bf = depths(u_frozen, qa, ii)
        am, bm = depths(u_exact, qa, ii)
        for diff in (af - am, bf - bm):
            good = np.isfinite(diff)
            sq_acc += trapz(np.where(good, diff ** 2, 0.0) * mu, ii)
            sup = max(sup, float(np.max(np.abs(diff[good]))))
    delta2 = float(np.sqrt(sq_acc / (2.0 * len(Q_eval))))
    delta_gamma = (1.0 / P.gamma) * np.log(1.0 + P.gamma / P.k)

    rows = [
        ("m_slope_origin", m_slope0),
        ("frozen_slope_origin", frozen_slope0),
        ("slope_ratio", m_slope0 / frozen_slope0),
        ("Delta2_statwt_L2", delta2),
        ("sup_abs_diff", sup),
        ("delta_gamma_halfspread", float(delta_gamma)),
        ("Delta2_over_halfspread", delta2 / float(delta_gamma)),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "alpha_reduced_robustness.csv").write_text(
        "quantity,value\n" + "\n".join(f"{k},{v:.6f}" for k, v in rows) + "\n"
    )
    print(f"Q_eval = {Q_eval},  i in [-6,6],  mu = N(0, eta^2/2beta), sigma_I = {sigma_I:.3f}")
    for k, v in rows:
        print(f"  {k:26s} = {v:.4f}")
    print(f"wrote {OUT / 'alpha_reduced_robustness.csv'}")


if __name__ == "__main__":
    main()
