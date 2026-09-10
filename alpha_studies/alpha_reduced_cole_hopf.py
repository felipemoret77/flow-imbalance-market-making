#!/usr/bin/env python3
"""Fast-alpha reduced CARA/GLFT Cole-Hopf policy study.

The control equation uses the fast conditional mean

    alpha_bar(y) = 2 * bar_lambda * epsilon / zeta * tanh(y),

while simulation uses the full alpha jump-diffusion environment.  The script
generates the alpha-study figures directly into the configured output folder.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh, expm
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parent
out_env = os.environ.get(
    "ASOU5_ALPHA_OUTPUT_DIR",
    os.environ.get("ASOU4_ALPHA_OUTPUT_DIR", "../imagens_tex/alpha_studies_T10000_qmax60"),
)
OUT = Path(out_env)
if not OUT.is_absolute():
    OUT = ROOT / OUT
OUT.mkdir(parents=True, exist_ok=True)


def env_float(name: str, default: float) -> float:
    if name.startswith("ASOU5_ALPHA_"):
        legacy = name.replace("ASOU5_ALPHA_", "ASOU4_ALPHA_", 1)
        return float(os.environ.get(name, os.environ.get(legacy, default)))
    return float(os.environ.get(name, default))


def env_int(name: str, default: int) -> int:
    if name.startswith("ASOU5_ALPHA_"):
        legacy = name.replace("ASOU5_ALPHA_", "ASOU4_ALPHA_", 1)
        return int(os.environ.get(name, os.environ.get(legacy, default)))
    return int(os.environ.get(name, default))


def standard_error(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


@dataclass(frozen=True)
class Params:
    T: float = env_float("ASOU5_ALPHA_T", 10000.0)
    qmax: int = env_int("ASOU5_ALPHA_QMAX", 60)
    plot_qmax: int = env_int("ASOU5_ALPHA_PLOT_QMAX", 30)
    y_max: float = env_float("ASOU5_ALPHA_Y_MAX", 6.0)
    ny: int = env_int("ASOU5_ALPHA_NY", 121)
    dt_hjb: float = env_float("ASOU5_ALPHA_DT_HJB", 0.25)
    n_paths: int = env_int("ASOU5_ALPHA_N_PATHS", 20000)
    dt_sim: float = env_float("ASOU5_ALPHA_DT_SIM", 0.05)
    record_dt: float = env_float("ASOU5_ALPHA_RECORD_DT", 1.0)
    seed: int = env_int("ASOU5_ALPHA_SEED", 86420)
    gamma: float = env_float("ASOU5_ALPHA_GAMMA", 0.01)
    bar_lambda: float = env_float("ASOU5_ALPHA_BAR_LAMBDA", 0.9)
    k: float = env_float("ASOU5_ALPHA_K", 2.0)
    beta: float = env_float("ASOU5_ALPHA_BETA", 0.0125)
    eta: float = env_float("ASOU5_ALPHA_ETA", 0.32)
    phi_input: float = env_float("ASOU5_ALPHA_PHI", -1.0)
    alpha_l: float = env_float("ASOU5_ALPHA_ALPHA_L", 0.01)
    zeta: float = env_float("ASOU5_ALPHA_ZETA", 0.25)
    epsilon: float = env_float("ASOU5_ALPHA_EPSILON", 0.005)
    eta_alpha: float = env_float("ASOU5_ALPHA_ETA_ALPHA", 0.001)
    sigma_price: float = env_float("ASOU5_ALPHA_SIGMA_PRICE", 0.3)
    surface_nt: int = env_int("ASOU5_ALPHA_SURFACE_NT", 61)
    surface_T: float = env_float("ASOU5_ALPHA_SURFACE_T", 10000.0)
    surface_qmax: int = env_int("ASOU5_ALPHA_SURFACE_QMAX", 30)
    surface_q_pad: int = env_int("ASOU5_ALPHA_SURFACE_Q_PAD", 60)
    surface_dt_hjb: float = env_float(
        "ASOU5_ALPHA_SURFACE_DT_HJB",
        env_float("ASOU5_ALPHA_DT_HJB", 0.25),
    )

    @property
    def delta0(self) -> float:
        return np.log1p(self.gamma / self.k) / self.gamma

    @property
    def chi_gamma(self) -> float:
        return (1.0 + self.gamma / self.k) ** (-(1.0 + self.k / self.gamma))

    @property
    def phi(self) -> float:
        if self.phi_input >= 0.0:
            return self.phi_input
        return 0.5 * self.gamma * self.sigma_price**2

    @property
    def alpha_c(self) -> float:
        return self.k * self.phi

    @property
    def eta_c(self) -> float:
        return self.bar_lambda * self.chi_gamma

    @property
    def p_glft(self) -> float:
        return 0.5 * np.sqrt(self.alpha_c / self.eta_c)

    @property
    def rho(self) -> float:
        return self.alpha_c / self.p_glft

    @property
    def alpha_slope(self) -> float:
        return 2.0 * self.bar_lambda * self.epsilon / self.zeta


P = Params()
Q = np.arange(-P.qmax, P.qmax + 1, dtype=int)
QF = Q.astype(float)
NQ = len(Q)
Y = np.linspace(-P.y_max, P.y_max, P.ny)
DY = Y[1] - Y[0]
Y0_IDX = int(np.argmin(np.abs(Y)))
Q0_IDX = int(np.where(Q == 0)[0][0])
LP = P.bar_lambda * (1.0 + np.tanh(Y))
LM = P.bar_lambda * (1.0 - np.tanh(Y))
ALPHA_BAR = P.alpha_slope * np.tanh(Y)


COLORS = {
    "GLFT": "#4E79A7",
    "Y-aware": "#59A14F",
    "Alpha/Y-aware": "#111111",
    "Alpha/Y ansatz": "#8172b2",
    "Alpha/Y affine": "#F28E2B",
    "Alpha/Y toy-h target": "#0072B2",
    "Alpha/Y flow-gated": "#E15759",
    "Alpha/Y toy-h curvature": "#984ea3",
    "Alpha/Y toy-h cubic": "#a65628",
    "Alpha/Y galerkin": "#B07AA1",
    "Alpha/Y geom-weight": "#17BECF",
    "Alpha/Y sc-cubic": "#E6AB02",
}


POLICY_DISPLAY = {
    "GLFT": "GLFT",
    "Y-aware": "I-aware RHJB",
    "Alpha/Y-aware": "Alpha/I-aware RHJB",
    "Alpha/Y sc-cubic": "Alpha/I self-consistent cubic",
    "Alpha/Y galerkin": "Alpha/I Galerkin-cubic",
}


def policy_display(name: str) -> str:
    """Canonical paper-facing policy name for legends and tick labels."""
    return POLICY_DISPLAY.get(name, name.replace("Alpha/Y", "Alpha/I"))


def policy_tick(name: str) -> str:
    """Compact multiline variant used on dense bar-chart axes."""
    return {
        "Y-aware": "I-aware\nRHJB",
        "Alpha/Y-aware": "Alpha/I-aware\nRHJB",
        "Alpha/Y sc-cubic": "Alpha/I\nself-consistent cubic",
        "Alpha/Y galerkin": "Alpha/I\nGalerkin-cubic",
    }.get(name, policy_display(name))


def colors_for(policies: list[str]) -> list[str]:
    fallback = ["#8da0cb", "#a6d854", "#ffd92f", "#e5c494"]
    return [COLORS.get(name, fallback[i % len(fallback)]) for i, name in enumerate(policies)]


def q_plot_mask() -> np.ndarray:
    qmax_plot = min(P.plot_qmax, P.qmax)
    return (np.abs(Q) <= qmax_plot) & (Q > -P.qmax) & (Q < P.qmax)


def build_ou_operator(dt: float) -> np.ndarray:
    n = P.ny
    d1 = np.zeros((n, n))
    d2 = np.zeros((n, n))
    for j in range(1, n - 1):
        d1[j, j - 1] = -0.5 / DY
        d1[j, j + 1] = 0.5 / DY
        d2[j, j - 1] = 1.0 / DY**2
        d2[j, j] = -2.0 / DY**2
        d2[j, j + 1] = 1.0 / DY**2

    d1[0, 0] = -1.0 / DY
    d1[0, 1] = 1.0 / DY
    d1[-1, -2] = -1.0 / DY
    d1[-1, -1] = 1.0 / DY
    d2[0, 0] = -2.0 / DY**2
    d2[0, 1] = 2.0 / DY**2
    d2[-1, -2] = 2.0 / DY**2
    d2[-1, -1] = -2.0 / DY**2
    generator = np.diag(-P.beta * Y) @ d1 + 0.5 * P.eta**2 * d2
    return expm(dt * generator)


OU_E_HALF = build_ou_operator(0.5 * P.dt_hjb)


def inventory_matrix(lambda_plus: float, lambda_minus: float, alpha_value: float) -> np.ndarray:
    diagonal = P.k * QF * alpha_value - P.alpha_c * QF**2
    mat = np.diag(diagonal)
    for i in range(NQ):
        if i > 0:
            mat[i, i - 1] = P.chi_gamma * lambda_plus
        if i < NQ - 1:
            mat[i, i + 1] = P.chi_gamma * lambda_minus
    return mat


def build_inventory_exponentials(alpha_values: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            expm(P.dt_hjb * inventory_matrix(lp, lm, alpha_value))
            for lp, lm, alpha_value in zip(LP, LM, alpha_values)
        ]
    )


def normalize_common(u: np.ndarray) -> np.ndarray:
    return u - u[Q0_IDX, Y0_IDX]


def alpha_inventory_step(u: np.ndarray, inv_e: np.ndarray) -> np.ndarray:
    shift = np.max(u, axis=0)
    w = np.exp(np.clip(P.k * (u - shift[None, :]), -700.0, 700.0))
    w_next = np.empty_like(w)
    for j in range(P.ny):
        w_next[:, j] = inv_e[j] @ w[:, j]
    return np.log(np.maximum(w_next, 1e-300)) / P.k + shift[None, :]


def depth_grid_from_u(u_tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ask = np.full((NQ, P.ny), np.nan)
    bid = np.full((NQ, P.ny), np.nan)
    ask[1:] = P.delta0 - u_tau[:-1] + u_tau[1:]
    bid[:-1] = P.delta0 - u_tau[1:] + u_tau[:-1]
    return ask, bid


def solve_ou_ergodic(
    alpha_values: np.ndarray,
    max_tau: float = 3000.0,
    tol: float = 2e-8,
    check_every: int = 20,
) -> tuple[np.ndarray, dict[str, float]]:
    inv_e = build_inventory_exponentials(alpha_values)
    steps = int(round(max_tau / P.dt_hjb))
    u = normalize_common(np.zeros((NQ, P.ny)))
    prev_ask: np.ndarray | None = None
    prev_bid: np.ndarray | None = None
    last_diff = np.inf
    converged = False
    for step in range(1, steps + 1):
        u = u @ OU_E_HALF.T
        u = alpha_inventory_step(u, inv_e)
        u = u @ OU_E_HALF.T
        u = normalize_common(u)
        if step % check_every == 0:
            ask, bid = depth_grid_from_u(u)
            if prev_ask is not None and prev_bid is not None:
                last_diff = float(
                    max(np.nanmax(np.abs(ask - prev_ask)), np.nanmax(np.abs(bid - prev_bid)))
                )
                if last_diff < tol:
                    converged = True
                    break
            prev_ask = ask
            prev_bid = bid
    return u, {
        "tau": float(step * P.dt_hjb),
        "steps": float(step),
        "last_depth_diff": float(last_diff),
        "converged": float(converged),
        "tolerance": float(tol),
    }


def interp_y(values: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.interp(np.clip(y, Y[0], Y[-1]), Y, values)


def y_aware_depths(u_tau: np.ndarray, q: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ask = np.full_like(y, np.inf, dtype=float)
    bid = np.full_like(y, np.inf, dtype=float)
    for qval in np.unique(q):
        mask = q == qval
        i = int(qval + P.qmax)
        ui = interp_y(u_tau[i], y[mask])
        if i > 0:
            ask[mask] = P.delta0 - interp_y(u_tau[i - 1], y[mask]) + ui
        if i < NQ - 1:
            bid[mask] = P.delta0 - interp_y(u_tau[i + 1], y[mask]) + ui
    return ask, bid


def glft_depths(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qf = q.astype(float)
    ask = P.delta0 - (P.p_glft / P.k) * (2.0 * qf - 1.0)
    bid = P.delta0 + (P.p_glft / P.k) * (2.0 * qf + 1.0)
    ask = np.where(q > -P.qmax, ask, np.inf)
    bid = np.where(q < P.qmax, bid, np.inf)
    return ask, bid


def glft_depth_grid() -> tuple[np.ndarray, np.ndarray]:
    q_grid, _ = np.meshgrid(Q, Y, indexing="ij")
    return glft_depths(q_grid)


def local_ansatz_coefficients() -> dict[str, float]:
    flow_loading = (P.rho / P.k) / (P.beta + P.rho)
    alpha_loading = P.alpha_slope / (P.beta + P.rho)
    toy_flow_slope = (P.rho / P.k) / (P.rho + 2.0 * P.beta)
    toy_flow_scale = np.sqrt(P.beta) / P.eta
    toy_flow_amplitude = toy_flow_slope / toy_flow_scale
    return {
        "p": float(P.p_glft),
        "inventory_rate": float(P.rho),
        "flow_loading": float(flow_loading),
        "alpha_loading": float(alpha_loading),
        "linear_loading": float(flow_loading + alpha_loading),
        "toy_flow_slope": float(toy_flow_slope),
        "toy_flow_amplitude": float(toy_flow_amplitude),
        "toy_flow_scale": float(toy_flow_scale),
        "alpha_slope": float(P.alpha_slope),
    }


def local_ansatz_shift(y: np.ndarray) -> np.ndarray:
    coeff = local_ansatz_coefficients()
    return coeff["flow_loading"] * y + coeff["alpha_loading"] * np.tanh(y)


def local_ansatz_depths(q: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qf = q.astype(float)
    h = local_ansatz_shift(y)
    ask = P.delta0 - (P.p_glft / P.k) * (2.0 * qf - 1.0) + h
    bid = P.delta0 + (P.p_glft / P.k) * (2.0 * qf + 1.0) - h
    ask = np.where(q > -P.qmax, ask, np.inf)
    bid = np.where(q < P.qmax, bid, np.inf)
    return ask, bid


# --- Legacy/ablation closed forms (toy_h_* family below): retained for reference,
# NOT exercised by main(); the reported forms are toy_h_sc_cubic_depths and
# galerkin_cubic_depths. ---
def toy_h_alpha_shift(y: np.ndarray) -> np.ndarray:
    """Fast-alpha loading that keeps the toy-model saturating flow loading and
    adds the closed fast-alpha tanh drift loading."""
    coeff = local_ansatz_coefficients()
    y = np.asarray(y, dtype=float)
    toy_flow = coeff["toy_flow_amplitude"] * np.tanh(coeff["toy_flow_scale"] * y)
    alpha_part = coeff["alpha_loading"] * np.tanh(y)
    return toy_flow + alpha_part


def toy_h_flow_alpha_components(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coeff = local_ansatz_coefficients()
    y = np.asarray(y, dtype=float)
    toy_flow = coeff["toy_flow_amplitude"] * np.tanh(coeff["toy_flow_scale"] * y)
    alpha_part = coeff["alpha_loading"] * np.tanh(y)
    gate = np.tanh(coeff["toy_flow_scale"] * y) ** 2
    return toy_flow, alpha_part, gate


def toy_h_flow_curvature_profiles(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    toy_flow, alpha_part, gate = toy_h_flow_alpha_components(y)
    theta = y - P.k * toy_flow
    r_g = 2.0 * P.p_glft
    r = r_g * np.sqrt(np.cosh(y) / np.cosh(theta))
    return toy_flow, alpha_part, gate, r


def toy_h_target_depths(q: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qf = q.astype(float)
    toy_flow, alpha_part, _, r = toy_h_flow_curvature_profiles(y)
    r_g = 2.0 * P.p_glft
    shift = (r / r_g) * (toy_flow + alpha_part)
    ask = P.delta0 + shift - (r / (2.0 * P.k)) * (2.0 * qf - 1.0)
    bid = P.delta0 - shift + (r / (2.0 * P.k)) * (2.0 * qf + 1.0)
    ask = np.where(q > -P.qmax, ask, np.inf)
    bid = np.where(q < P.qmax, bid, np.inf)
    return ask, bid


def toy_h_flow_gated_depths(q: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qf = q.astype(float)
    toy_flow, alpha_part, gate, r = toy_h_flow_curvature_profiles(y)
    r_g = 2.0 * P.p_glft
    mult = r / r_g
    shift = mult * toy_flow + (1.0 + gate * (mult - 1.0)) * alpha_part
    ask = P.delta0 + shift - (r / (2.0 * P.k)) * (2.0 * qf - 1.0)
    bid = P.delta0 - shift + (r / (2.0 * P.k)) * (2.0 * qf + 1.0)
    ask = np.where(q > -P.qmax, ask, np.inf)
    bid = np.where(q < P.qmax, bid, np.inf)
    return ask, bid


def toy_h_alpha_profiles(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    h = toy_h_alpha_shift(y)
    theta = y - P.k * h
    r_g = 2.0 * P.p_glft
    r = r_g * np.sqrt(np.cosh(y) / np.cosh(theta))
    s = (r**2 / 6.0) * np.tanh(theta)
    return h, r, s


def toy_h_curvature_depths(q: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qf = q.astype(float)
    h, r, _ = toy_h_alpha_profiles(y)
    ask = P.delta0 + h - (r / (2.0 * P.k)) * (2.0 * qf - 1.0)
    bid = P.delta0 - h + (r / (2.0 * P.k)) * (2.0 * qf + 1.0)
    ask = np.where(q > -P.qmax, ask, np.inf)
    bid = np.where(q < P.qmax, bid, np.inf)
    return ask, bid


def toy_h_cubic_depths(q: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qf = q.astype(float)
    h, r, s = toy_h_alpha_profiles(y)
    ask = P.delta0 + h - (r / (2.0 * P.k)) * (2.0 * qf - 1.0) + (s / P.k) * (qf**2 - qf + 1.0 / 3.0)
    bid = P.delta0 - h + (r / (2.0 * P.k)) * (2.0 * qf + 1.0) - (s / P.k) * (qf**2 + qf + 1.0 / 3.0)
    ask = np.where(q > -P.qmax, ask, np.inf)
    bid = np.where(q < P.qmax, bid, np.inf)
    return ask, bid


# Fast-alpha curvature-refined (Galerkin) ansatz: cubic tilted-GLFT with state-dependent
# curvature r(i), cubic s(i), and a SINGLE saturating Galerkin loading
# h = A i / sqrt(1 + beta i^2/(c eta^2)), (A,c) fixed by the OU-Gaussian moment projection of
# the fast-alpha q^1 residual (with the extra a_alpha*tanh i drift source).  Derived/verified
# in closed_form_tests/alpha_rhjb_loading.py.  Non-elementary (offline moment solve); appendix only.
A_GAL_ALPHA, C_GAL_ALPHA = 0.881800, 4.666300


def galerkin_cubic_depths(q: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qf = q.astype(float)
    yc = np.clip(y, Y[0], Y[-1])          # clip y to grid: cosh r(i) blows up off-grid
    h = A_GAL_ALPHA * yc / np.sqrt(1.0 + P.beta * yc**2 / (C_GAL_ALPHA * P.eta**2))
    th = yc - P.k * h
    r_g = 2.0 * P.p_glft
    r = r_g * np.sqrt(np.cosh(yc) / np.cosh(th))
    s = (r**2 / 6.0) * np.tanh(th)
    ask = P.delta0 + h - (r / (2.0 * P.k)) * (2.0 * qf - 1.0) + (s / P.k) * (qf**2 - qf + 1.0 / 3.0)
    bid = P.delta0 - h + (r / (2.0 * P.k)) * (2.0 * qf + 1.0) - (s / P.k) * (qf**2 + qf + 1.0 / 3.0)
    ask = np.where(q > -P.qmax, ask, np.inf)
    bid = np.where(q < P.qmax, bid, np.inf)
    return ask, bid


def toy_h_geomweight_depths(q: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Candidate C: flow-gated skew with a GEOMETRIC alpha weight W = M^gate instead
    of the additive 1 + gate*(M-1).  M = r/r_G, gate = tanh^2(A2 y).  Both forms run
    1 -> M as the gate 0 -> 1, but the geometric interpolation bows below the additive
    line in the mid-y region, killing the skew overshoot the flow-gated leaves there.
    Parameter-free (no new constant); reduces to GLFT at y=0.  Offline depth RMSE vs
    fast-alpha RHJB (imbalance grid): 0.0922 vs flow-gated 0.0998."""
    qf = q.astype(float)
    toy_flow, alpha_part, gate, r = toy_h_flow_curvature_profiles(y)
    r_g = 2.0 * P.p_glft
    mult = r / r_g
    shift = mult * toy_flow + (mult ** gate) * alpha_part
    ask = P.delta0 + shift - (r / (2.0 * P.k)) * (2.0 * qf - 1.0)
    bid = P.delta0 - shift + (r / (2.0 * P.k)) * (2.0 * qf + 1.0)
    ask = np.where(q > -P.qmax, ask, np.inf)
    bid = np.where(q < P.qmax, bid, np.inf)
    return ask, bid


def toy_h_sc_cubic_depths(q: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Candidate D: flow-gated skew + curvature-consistent cubic evaluated at the
    SELF-CONSISTENT full quoted shift theta_full = y - k*shift (not flow-only).  Where
    shift > y/k the convexity term flips sign on its own, matching the opposite-sign
    q-convexity of the fast-alpha RHJB, so the cubic helps instead of blowing up.  The
    1/6 coefficient is the toy curvature-backreaction value; no new constant; reduces to
    GLFT at y=0.  Offline depth RMSE vs RHJB: imbalance 0.0963, inventory 0.120 (vs
    flow-gated 0.0998 / 0.184), combined 0.0998."""
    qf = q.astype(float)
    toy_flow, alpha_part, gate, r = toy_h_flow_curvature_profiles(y)
    r_g = 2.0 * P.p_glft
    mult = r / r_g
    shift = mult * toy_flow + (1.0 + gate * (mult - 1.0)) * alpha_part
    theta_full = np.asarray(y, dtype=float) - P.k * shift
    s = (r ** 2 / 6.0) * np.tanh(theta_full)
    ask = P.delta0 + shift - (r / (2.0 * P.k)) * (2.0 * qf - 1.0) + (s / P.k) * (qf**2 - qf + 1.0 / 3.0)
    bid = P.delta0 - shift + (r / (2.0 * P.k)) * (2.0 * qf + 1.0) - (s / P.k) * (qf**2 + qf + 1.0 / 3.0)
    ask = np.where(q > -P.qmax, ask, np.inf)
    bid = np.where(q < P.qmax, bid, np.inf)
    return ask, bid


def fill_count_mean(lambda0: np.ndarray, depth: np.ndarray) -> np.ndarray:
    mult = np.exp(np.clip(-P.k * depth, -60.0, 60.0))
    lam = np.where(np.isfinite(depth), lambda0 * mult, 0.0)
    return lam * P.dt_sim


def sample_capped_poisson(
    rng: np.random.Generator,
    mean: np.ndarray,
    cap: np.ndarray,
) -> np.ndarray:
    cap = np.maximum(cap, 0).astype(int)
    counts = np.zeros_like(cap, dtype=int)
    active = (cap > 0) & np.isfinite(mean) & (mean > 0.0)
    if not np.any(active):
        return counts
    threshold = cap + 10.0 * np.sqrt(np.maximum(mean, 1.0))
    saturated = active & (mean > threshold)
    sampled = active & ~saturated
    counts[saturated] = cap[saturated]
    if np.any(sampled):
        raw = rng.poisson(mean[sampled])
        counts[sampled] = np.minimum(raw, cap[sampled])
    return counts


def simulate_policies(
    y_aware_u: np.ndarray,
    alpha_y_aware_u: np.ndarray,
) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    rng = np.random.default_rng(P.seed)
    steps = int(round(P.T / P.dt_sim))
    times = np.linspace(0.0, P.T, steps + 1)
    sqrt_dt = np.sqrt(P.dt_sim)
    policies = [
        "GLFT",
        "Y-aware",
        "Alpha/Y-aware",
        "Alpha/Y sc-cubic",
        "Alpha/Y galerkin",
    ]

    y = np.zeros(P.n_paths)
    alpha = np.zeros(P.n_paths)
    price = np.zeros(P.n_paths)
    q = {name: np.zeros(P.n_paths, dtype=int) for name in policies}
    spread_cash = {name: np.zeros(P.n_paths) for name in policies}
    alpha_pnl = {name: np.zeros(P.n_paths) for name in policies}
    price_martingale_pnl = {name: np.zeros(P.n_paths) for name in policies}
    q_y_integral = {name: np.zeros(P.n_paths) for name in policies}
    fills = {name: np.zeros(P.n_paths) for name in policies}
    running_penalty = {name: np.zeros(P.n_paths) for name in policies}
    abs_q_integral = {name: np.zeros(P.n_paths) for name in policies}
    min_ask = {name: np.full(P.n_paths, np.inf) for name in policies}
    min_bid = {name: np.full(P.n_paths, np.inf) for name in policies}

    record_every = max(1, int(round(P.record_dt / P.dt_sim)))
    rec_times: list[float] = []
    rec_total = {name: [] for name in policies}
    rec_decomposed = {name: [] for name in policies}
    rec_pen = {name: [] for name in policies}
    rec_absq = {name: [] for name in policies}
    lifetime_q = {name: [] for name in policies}
    signal_lifetime_q = {name: [] for name in policies}
    y_samples: list[np.ndarray] = []
    alpha_samples: list[np.ndarray] = []
    sample_price: list[float] = []
    sample_y: list[float] = []
    sample_alpha: list[float] = []
    sample_inventory = {name: [] for name in policies}
    y_clip_count = 0
    y_clip_total = 0

    for step, t in enumerate(times[:-1]):
        do_record = step % record_every == 0
        y_clip_count += int(np.sum((y < Y[0]) | (y > Y[-1])))
        y_clip_total += P.n_paths
        lambda_plus0 = P.bar_lambda * (1.0 + np.tanh(y))
        lambda_minus0 = P.bar_lambda * (1.0 - np.tanh(y))
        alpha_before = alpha.copy()
        price_noise = P.sigma_price * sqrt_dt * rng.standard_normal(P.n_paths)

        for name in policies:
            q_before = q[name].copy()
            if name == "GLFT":
                ask, bid = glft_depths(q_before)
            elif name == "Y-aware":
                ask, bid = y_aware_depths(y_aware_u, q_before, y)
            elif name == "Alpha/Y-aware":
                ask, bid = y_aware_depths(alpha_y_aware_u, q_before, y)
            elif name == "Alpha/Y sc-cubic":
                ask, bid = toy_h_sc_cubic_depths(q_before, y)
            elif name == "Alpha/Y galerkin":
                ask, bid = galerkin_cubic_depths(q_before, y)
            else:
                raise ValueError(name)

            min_ask[name] = np.minimum(min_ask[name], ask)
            min_bid[name] = np.minimum(min_bid[name], bid)
            ask_fill = sample_capped_poisson(
                rng,
                fill_count_mean(lambda_plus0, ask),
                q_before + P.qmax,
            )
            bid_fill = sample_capped_poisson(
                rng,
                fill_count_mean(lambda_minus0, bid),
                P.qmax - q_before,
            )
            ask_exec = np.where(np.isfinite(ask), ask, 0.0)
            bid_exec = np.where(np.isfinite(bid), bid, 0.0)
            spread_cash[name] += ask_exec * ask_fill + bid_exec * bid_fill
            alpha_pnl[name] += q_before.astype(float) * alpha_before * P.dt_sim
            price_martingale_pnl[name] += q_before.astype(float) * price_noise
            q_y_integral[name] += q_before.astype(float) * y * P.dt_sim
            if do_record:
                signal_lifetime_q[name].append(q_before.copy())
                sample_inventory[name].append(int(q_before[0]))
            q[name] += -ask_fill + bid_fill
            fills[name] += ask_fill.astype(float) + bid_fill.astype(float)
            running_penalty[name] += P.phi * q_before.astype(float) ** 2 * P.dt_sim
            abs_q_integral[name] += np.abs(q_before) * P.dt_sim

        if do_record:
            rec_times.append(t)
            y_samples.append(y.copy())
            alpha_samples.append(alpha_before.copy())
            sample_price.append(float(price[0]))
            sample_y.append(float(y[0]))
            sample_alpha.append(float(alpha_before[0]))
            for name in policies:
                decomposed = spread_cash[name] + alpha_pnl[name]
                total = decomposed + price_martingale_pnl[name]
                pen = total - running_penalty[name] - P.alpha_l * q[name].astype(float) ** 2
                rec_total[name].append(float(np.mean(total)))
                rec_decomposed[name].append(float(np.mean(decomposed)))
                rec_pen[name].append(float(np.mean(pen)))
                rec_absq[name].append(float(np.mean(np.abs(q[name]))))
                lifetime_q[name].append(q[name].copy())

        aggregate_buy = rng.poisson(lambda_plus0 * P.dt_sim)
        aggregate_sell = rng.poisson(lambda_minus0 * P.dt_sim)
        price += alpha_before * P.dt_sim + price_noise
        alpha += (
            -P.zeta * alpha * P.dt_sim
            + P.eta_alpha * sqrt_dt * rng.standard_normal(P.n_paths)
            + P.epsilon * aggregate_buy
            - P.epsilon * aggregate_sell
        )
        y += -P.beta * y * P.dt_sim + P.eta * sqrt_dt * rng.standard_normal(P.n_paths)

    metrics: dict[str, dict[str, float]] = {}
    y_clip_share = float(y_clip_count / max(1, y_clip_total))
    terminal = {}
    terminal_decomposed = {}
    pen_paths = {}
    for name in policies:
        decomposed = spread_cash[name] + alpha_pnl[name]
        total = decomposed + price_martingale_pnl[name]
        pen = total - running_penalty[name] - P.alpha_l * q[name].astype(float) ** 2
        pen_paths[name] = pen.copy()
        terminal[name] = total
        terminal_decomposed[name] = decomposed
        lifetime = np.concatenate(lifetime_q[name]) if lifetime_q[name] else q[name]
        metrics[name] = {
            "total_pnl_mean": float(np.mean(total)),
            "total_pnl_std": float(np.std(total)),
            "total_pnl_stderr": standard_error(total),
            "decomposed_pnl_mean": float(np.mean(decomposed)),
            "decomposed_pnl_std": float(np.std(decomposed)),
            "decomposed_pnl_stderr": standard_error(decomposed),
            "penalized_pnl_mean": float(np.mean(pen)),
            "penalized_pnl_std": float(np.std(pen)),
            "penalized_pnl_stderr": standard_error(pen),
            "spread_capture_mean": float(np.mean(spread_cash[name])),
            "spread_capture_stderr": standard_error(spread_cash[name]),
            "alpha_drift_pnl_mean": float(np.mean(alpha_pnl[name])),
            "alpha_drift_pnl_stderr": standard_error(alpha_pnl[name]),
            "price_martingale_pnl_mean": float(np.mean(price_martingale_pnl[name])),
            "price_martingale_pnl_std": float(np.std(price_martingale_pnl[name])),
            "price_martingale_pnl_stderr": standard_error(price_martingale_pnl[name]),
            "avg_q_alpha": float(np.mean(alpha_pnl[name] / P.T)),
            "avg_q_y": float(np.mean(q_y_integral[name] / P.T)),
            "running_penalty_mean": float(np.mean(running_penalty[name])),
            "terminal_penalty_mean": float(np.mean(P.alpha_l * q[name].astype(float) ** 2)),
            "fills_mean": float(np.mean(fills[name])),
            "avg_abs_inventory": float(np.mean(abs_q_integral[name] / P.T)),
            "terminal_abs_inventory": float(np.mean(np.abs(q[name]))),
            "lifetime_q01": float(np.quantile(lifetime, 0.01)),
            "lifetime_q99": float(np.quantile(lifetime, 0.99)),
            "mean_min_ask_depth": float(np.mean(min_ask[name])),
            "mean_min_bid_depth": float(np.mean(min_bid[name])),
            "share_negative_min_ask": float(np.mean(min_ask[name] < 0.0)),
            "share_negative_min_bid": float(np.mean(min_bid[name] < 0.0)),
            "y_clip_share": y_clip_share,
        }

    y_lifetime = np.concatenate(y_samples) if y_samples else y
    alpha_lifetime = np.concatenate(alpha_samples) if alpha_samples else alpha
    lifetime_inventory = {
        name: np.concatenate(lifetime_q[name]) if lifetime_q[name] else q[name]
        for name in policies
    }
    lifetime_inventory_histograms = make_lifetime_inventory_histograms(lifetime_inventory)
    signal_inventory = {
        name: np.concatenate(signal_lifetime_q[name]) if signal_lifetime_q[name] else q[name]
        for name in policies
    }
    paths: dict[str, object] = {
        "times": np.array(rec_times),
        "total": {name: np.array(rec_total[name]) for name in policies},
        "decomposed": {name: np.array(rec_decomposed[name]) for name in policies},
        "penalized": {name: np.array(rec_pen[name]) for name in policies},
        "absq": {name: np.array(rec_absq[name]) for name in policies},
        "terminal_total": terminal,
        "pen_paths": pen_paths,
        "terminal_decomposed": terminal_decomposed,
        # Keep only the aggregate histograms needed by the paper figure.  The raw
        # lifetime arrays can contain hundreds of millions of entries and are not
        # needed after the conditional summaries above have been computed.
        "lifetime_inventory_histograms": lifetime_inventory_histograms,
        "sample": {
            "times": np.array(rec_times),
            "price": np.array(sample_price),
            "y": np.array(sample_y),
            "alpha": np.array(sample_alpha),
            "inventory": {name: np.array(sample_inventory[name]) for name in policies},
        },
    }
    paths.update(make_conditional_inventory(y_lifetime, signal_inventory))
    paths.update(make_conditional_inventory_by_alpha(alpha_lifetime, signal_inventory))
    # Depth-vs-alpha is a DEPTH plot: only the fast-alpha RHJB and the two retained
    # closed forms (sc-cubic, galerkin). Baselines GLFT/Y-aware are kept in the
    # conditional-INVENTORY plots, not here.
    depth_fns = {
        "Alpha/Y-aware": lambda q, y: y_aware_depths(alpha_y_aware_u, q, y),
        "Alpha/Y sc-cubic": lambda q, y: toy_h_sc_cubic_depths(q, y),
        "Alpha/Y galerkin": lambda q, y: galerkin_cubic_depths(q, y),
    }
    paths.update(make_conditional_depth_by_alpha(alpha_lifetime, y_lifetime, signal_inventory, depth_fns))
    return metrics, paths


def make_conditional_inventory(
    y_state: np.ndarray,
    lifetime_inventory: dict[str, np.ndarray],
) -> dict[str, object]:
    bins = np.linspace(Y[0], Y[-1], 25)
    centers = 0.5 * (bins[:-1] + bins[1:])
    means = {}
    counts = {}
    for name, inv in lifetime_inventory.items():
        vals = np.full(len(centers), np.nan)
        cnt = np.zeros(len(centers), dtype=int)
        for j in range(len(centers)):
            if j == len(centers) - 1:
                mask = (y_state >= bins[j]) & (y_state <= bins[j + 1])
            else:
                mask = (y_state >= bins[j]) & (y_state < bins[j + 1])
            cnt[j] = int(np.sum(mask))
            if cnt[j] > 0:
                vals[j] = float(np.mean(inv[mask]))
        means[name] = vals
        counts[name] = cnt
    return {"y_centers": centers, "inventory_by_y": means, "inventory_by_y_counts": counts}


def make_conditional_inventory_by_alpha(
    alpha_state: np.ndarray,
    lifetime_inventory: dict[str, np.ndarray],
) -> dict[str, object]:
    lo, hi = np.quantile(alpha_state, [0.01, 0.99])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = -0.05, 0.05
    bins = np.linspace(float(lo), float(hi), 25)
    centers = 0.5 * (bins[:-1] + bins[1:])
    means = {}
    counts = {}
    for name, inv in lifetime_inventory.items():
        vals = np.full(len(centers), np.nan)
        cnt = np.zeros(len(centers), dtype=int)
        for j in range(len(centers)):
            if j == len(centers) - 1:
                mask = (alpha_state >= bins[j]) & (alpha_state <= bins[j + 1])
            else:
                mask = (alpha_state >= bins[j]) & (alpha_state < bins[j + 1])
            cnt[j] = int(np.sum(mask))
            if cnt[j] > 0:
                vals[j] = float(np.mean(inv[mask]))
        means[name] = vals
        counts[name] = cnt
    return {
        "alpha_centers": centers,
        "inventory_by_alpha": means,
        "inventory_by_alpha_counts": counts,
    }


def make_conditional_depth_by_alpha(
    alpha_state: np.ndarray,
    y_state: np.ndarray,
    lifetime_inventory: dict[str, np.ndarray],
    depth_fns: dict,
    max_samples: int = 3_000_000,
) -> dict[str, object]:
    """Realized posted depth conditioned on realized alpha: for each policy recompute the
    posted ask/bid depth delta(q_t, I_t) from the recorded (q,y) samples and bin by alpha_t.
    The reduced policy is alpha-blind (a function of (q,I)); this diagnostic shows how its
    realized quotes still track alpha through the I-alpha correlation."""
    n = len(alpha_state)
    step = max(1, n // max_samples)
    a = alpha_state[::step]
    ys = y_state[::step]
    lo, hi = np.quantile(a, [0.01, 0.99])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = -0.05, 0.05
    bins = np.linspace(float(lo), float(hi), 25)
    centers = 0.5 * (bins[:-1] + bins[1:])
    ask_means: dict[str, np.ndarray] = {}
    bid_means: dict[str, np.ndarray] = {}
    for name, inv in lifetime_inventory.items():
        if name not in depth_fns:
            continue
        q = inv[::step]
        ask, bid = depth_fns[name](q, ys)
        ask = np.where(np.isfinite(ask), ask, np.nan)
        bid = np.where(np.isfinite(bid), bid, np.nan)
        av = np.full(len(centers), np.nan)
        bv = np.full(len(centers), np.nan)
        for j in range(len(centers)):
            if j == len(centers) - 1:
                mask = (a >= bins[j]) & (a <= bins[j + 1])
            else:
                mask = (a >= bins[j]) & (a < bins[j + 1])
            if np.any(mask):
                av[j] = float(np.nanmean(ask[mask]))
                bv[j] = float(np.nanmean(bid[mask]))
        ask_means[name] = av
        bid_means[name] = bv
    return {
        "depth_alpha_centers": centers,
        "ask_by_alpha": ask_means,
        "bid_by_alpha": bid_means,
    }


def plot_quote_depths(y_aware_u: np.ndarray, alpha_y_aware_u: np.ndarray) -> None:
    yy = np.linspace(Y[0], Y[-1], 301)
    q0 = np.zeros_like(yy, dtype=int)
    g_ask, g_bid = glft_depths(q0)
    y_ask, y_bid = y_aware_depths(y_aware_u, q0, yy)
    ay_ask, ay_bid = y_aware_depths(alpha_y_aware_u, q0, yy)
    ansatz_ask, ansatz_bid = local_ansatz_depths(q0, yy)
    target_ask, target_bid = toy_h_target_depths(q0, yy)
    gated_ask, gated_bid = toy_h_flow_gated_depths(q0, yy)
    gal_ask, gal_bid = galerkin_cubic_depths(q0, yy)
    geom_ask, geom_bid = toy_h_geomweight_depths(q0, yy)
    sc_ask, sc_bid = toy_h_sc_cubic_depths(q0, yy)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    ax[0].plot(yy, g_ask, lw=2.1, color=COLORS["GLFT"], label="GLFT")
    ax[0].plot(yy, y_ask, lw=2.3, color=COLORS["Y-aware"], label="I-aware RHJB")
    ax[0].plot(yy, ay_ask, lw=2.6, color=COLORS["Alpha/Y-aware"], label="Alpha/I-aware RHJB")
    ax[0].plot(yy, ansatz_ask, "-.", lw=2.1, color=COLORS["Alpha/Y affine"], label="Alpha/I affine")
    ax[0].plot(yy, target_ask, "-", lw=2.3, color=COLORS["Alpha/Y toy-h target"], label="toy-h target")
    ax[0].plot(yy, gated_ask, "--", lw=2.3, color=COLORS["Alpha/Y flow-gated"], label="flow-gated")
    ax[0].plot(yy, gal_ask, ":", lw=2.5, color=COLORS["Alpha/Y galerkin"], label="Galerkin-cubic")
    ax[0].plot(yy, geom_ask, linestyle=(0, (3, 1, 1, 1)), lw=2.2, color=COLORS["Alpha/Y geom-weight"], label="geom-weight (C)")
    ax[0].plot(yy, sc_ask, linestyle=(0, (5, 1)), lw=2.2, color=COLORS["Alpha/Y sc-cubic"], label="self-consistent cubic")
    ax[0].set_title("Ask depth, q=0")
    ax[1].plot(yy, g_bid, lw=2.1, color=COLORS["GLFT"], label="GLFT")
    ax[1].plot(yy, y_bid, lw=2.3, color=COLORS["Y-aware"], label="I-aware RHJB")
    ax[1].plot(yy, ay_bid, lw=2.6, color=COLORS["Alpha/Y-aware"], label="Alpha/I-aware RHJB")
    ax[1].plot(yy, ansatz_bid, "-.", lw=2.1, color=COLORS["Alpha/Y affine"], label="Alpha/I affine")
    ax[1].plot(yy, target_bid, "-", lw=2.3, color=COLORS["Alpha/Y toy-h target"], label="toy-h target")
    ax[1].plot(yy, gated_bid, "--", lw=2.3, color=COLORS["Alpha/Y flow-gated"], label="flow-gated")
    ax[1].plot(yy, gal_bid, ":", lw=2.5, color=COLORS["Alpha/Y galerkin"], label="Galerkin-cubic")
    ax[1].plot(yy, geom_bid, linestyle=(0, (3, 1, 1, 1)), lw=2.2, color=COLORS["Alpha/Y geom-weight"], label="geom-weight (C)")
    ax[1].plot(yy, sc_bid, linestyle=(0, (5, 1)), lw=2.2, color=COLORS["Alpha/Y sc-cubic"], label="self-consistent cubic")
    ax[1].set_title("Bid depth, q=0")
    for a in ax:
        a.axhline(0.0, color="black", lw=1, alpha=0.45)
        a.set_xlabel("OU state i")
        a.set_ylabel("Raw unconstrained depth")
        a.grid(True, alpha=0.25)
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "alpha_reduced_quote_depths_q0.png", dpi=180)
    plt.close(fig)


def plot_alpha_closed_ansatz_depths_for_q(q_value: int, alpha_y_aware_u: np.ndarray) -> None:
    yy = np.linspace(Y[0], Y[-1], 301)
    q_arr = np.full_like(yy, int(q_value), dtype=int)
    ay_ask, ay_bid = y_aware_depths(alpha_y_aware_u, q_arr, yy)
    ansatz_ask, ansatz_bid = local_ansatz_depths(q_arr, yy)
    target_ask, target_bid = toy_h_target_depths(q_arr, yy)
    gated_ask, gated_bid = toy_h_flow_gated_depths(q_arr, yy)
    gal_ask, gal_bid = galerkin_cubic_depths(q_arr, yy)
    geom_ask, geom_bid = toy_h_geomweight_depths(q_arr, yy)
    sc_ask, sc_bid = toy_h_sc_cubic_depths(q_arr, yy)
    q_label = f"qm{abs(q_value)}" if q_value < 0 else f"q{q_value}"
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    ax[0].plot(yy, ay_ask, lw=2.6, color=COLORS["Alpha/Y-aware"], label="Alpha/I-aware RHJB")
    ax[0].plot(yy, ansatz_ask, "-.", lw=2.1, color=COLORS["Alpha/Y affine"], label="affine old")
    ax[0].plot(yy, target_ask, "-", lw=2.3, color=COLORS["Alpha/Y toy-h target"], label="toy-h target")
    ax[0].plot(yy, gated_ask, "--", lw=2.3, color=COLORS["Alpha/Y flow-gated"], label="flow-gated")
    ax[0].plot(yy, gal_ask, ":", lw=2.5, color=COLORS["Alpha/Y galerkin"], label="Galerkin-cubic")
    ax[0].plot(yy, geom_ask, linestyle=(0, (3, 1, 1, 1)), lw=2.2, color=COLORS["Alpha/Y geom-weight"], label="geom-weight (C)")
    ax[0].plot(yy, sc_ask, linestyle=(0, (5, 1)), lw=2.2, color=COLORS["Alpha/Y sc-cubic"], label="self-consistent cubic")
    ax[0].set_title(f"Fast-alpha ask depth, q={q_value}")
    ax[1].plot(yy, ay_bid, lw=2.6, color=COLORS["Alpha/Y-aware"], label="Alpha/I-aware RHJB")
    ax[1].plot(yy, ansatz_bid, "-.", lw=2.1, color=COLORS["Alpha/Y affine"], label="affine old")
    ax[1].plot(yy, target_bid, "-", lw=2.3, color=COLORS["Alpha/Y toy-h target"], label="toy-h target")
    ax[1].plot(yy, gated_bid, "--", lw=2.3, color=COLORS["Alpha/Y flow-gated"], label="flow-gated")
    ax[1].plot(yy, gal_bid, ":", lw=2.5, color=COLORS["Alpha/Y galerkin"], label="Galerkin-cubic")
    ax[1].plot(yy, geom_bid, linestyle=(0, (3, 1, 1, 1)), lw=2.2, color=COLORS["Alpha/Y geom-weight"], label="geom-weight (C)")
    ax[1].plot(yy, sc_bid, linestyle=(0, (5, 1)), lw=2.2, color=COLORS["Alpha/Y sc-cubic"], label="self-consistent cubic")
    ax[1].set_title(f"Fast-alpha bid depth, q={q_value}")
    for a in ax:
        a.axhline(0.0, color="black", lw=1, alpha=0.45)
        a.axvline(0.0, color="black", lw=0.8, alpha=0.25)
        a.set_xlabel("OU state i")
        a.set_ylabel("Raw unconstrained depth")
        a.grid(True, alpha=0.25)
        a.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"alpha_reduced_closed_ansatz_depths_{q_label}.png", dpi=180)
    plt.close(fig)


def plot_alpha_closed_ansatz_depths_selected_q(alpha_y_aware_u: np.ndarray) -> None:
    selected_q = [q for q in [0, -5, 10, 15] if -P.qmax <= q <= P.qmax]
    for q_value in selected_q:
        plot_alpha_closed_ansatz_depths_for_q(q_value, alpha_y_aware_u)


def plot_alpha_closed_ansatz_depths_vs_inventory(alpha_y_aware_u: np.ndarray) -> None:
    y_values = [-3.0, 0.0, 3.0]
    q_plot = Q[q_plot_mask()]
    fig, ax = plt.subplots(len(y_values), 2, figsize=(12, 9), sharex=True, sharey=True)
    rows = [
        "y,q,ask_hjb,bid_hjb,ask_aff,bid_aff,ask_target,bid_target,"
        "ask_gated,bid_gated,ask_gal,bid_gal,ask_geom,bid_geom,ask_sc,bid_sc"
    ]

    for row_idx, y_value in enumerate(y_values):
        yy = np.full(q_plot.shape, y_value, dtype=float)
        ask_hjb, bid_hjb = y_aware_depths(alpha_y_aware_u, q_plot, yy)
        ask_ansatz, bid_ansatz = local_ansatz_depths(q_plot, yy)
        ask_target, bid_target = toy_h_target_depths(q_plot, yy)
        ask_gated, bid_gated = toy_h_flow_gated_depths(q_plot, yy)
        ask_geom, bid_geom = toy_h_geomweight_depths(q_plot, yy)
        ask_sc, bid_sc = toy_h_sc_cubic_depths(q_plot, yy)
        ask_gal, bid_gal = galerkin_cubic_depths(q_plot, yy)

        ax[row_idx, 0].plot(q_plot, ask_hjb, lw=2.5, color=COLORS["Alpha/Y-aware"], label="Alpha/I-aware RHJB")
        ax[row_idx, 0].plot(q_plot, ask_ansatz, "-.", lw=2.1, color=COLORS["Alpha/Y affine"], label="affine old")
        ax[row_idx, 0].plot(q_plot, ask_target, "-", lw=2.3, color=COLORS["Alpha/Y toy-h target"], label="toy-h target")
        ax[row_idx, 0].plot(q_plot, ask_gated, "--", lw=2.3, color=COLORS["Alpha/Y flow-gated"], label="flow-gated")
        ax[row_idx, 0].plot(q_plot, ask_gal, ":", lw=2.3, color=COLORS["Alpha/Y galerkin"], label="Galerkin-cubic")
        ax[row_idx, 0].plot(q_plot, ask_geom, linestyle=(0, (3, 1, 1, 1)), lw=2.2, color=COLORS["Alpha/Y geom-weight"], label="geom-weight (C)")
        ax[row_idx, 0].plot(q_plot, ask_sc, linestyle=(0, (5, 1)), lw=2.2, color=COLORS["Alpha/Y sc-cubic"], label="self-consistent cubic")
        ax[row_idx, 0].set_title(f"Ask depth, i={y_value:g}")

        ax[row_idx, 1].plot(q_plot, bid_hjb, lw=2.5, color=COLORS["Alpha/Y-aware"], label="Alpha/I-aware RHJB")
        ax[row_idx, 1].plot(q_plot, bid_ansatz, "-.", lw=2.1, color=COLORS["Alpha/Y affine"], label="affine old")
        ax[row_idx, 1].plot(q_plot, bid_target, "-", lw=2.3, color=COLORS["Alpha/Y toy-h target"], label="toy-h target")
        ax[row_idx, 1].plot(q_plot, bid_gated, "--", lw=2.3, color=COLORS["Alpha/Y flow-gated"], label="flow-gated")
        ax[row_idx, 1].plot(q_plot, bid_gal, ":", lw=2.3, color=COLORS["Alpha/Y galerkin"], label="Galerkin-cubic")
        ax[row_idx, 1].plot(q_plot, bid_geom, linestyle=(0, (3, 1, 1, 1)), lw=2.2, color=COLORS["Alpha/Y geom-weight"], label="geom-weight (C)")
        ax[row_idx, 1].plot(q_plot, bid_sc, linestyle=(0, (5, 1)), lw=2.2, color=COLORS["Alpha/Y sc-cubic"], label="self-consistent cubic")
        ax[row_idx, 1].set_title(f"Bid depth, i={y_value:g}")

        for values in zip(
            q_plot,
            ask_hjb,
            bid_hjb,
            ask_ansatz,
            bid_ansatz,
            ask_target,
            bid_target,
            ask_gated,
            bid_gated,
            ask_gal,
            bid_gal,
            ask_geom,
            bid_geom,
            ask_sc,
            bid_sc,
        ):
            rows.append(f"{y_value:.10g}," + ",".join(f"{x:.10g}" for x in values))

    for a in ax.ravel():
        a.axhline(0.0, color="black", lw=1, alpha=0.45)
        a.axvline(0.0, color="black", lw=0.8, alpha=0.25)
        a.set_xlabel("inventory q")
        a.set_ylabel("Raw unconstrained depth")
        a.grid(True, alpha=0.25)
    ax[0, 0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "alpha_reduced_closed_ansatz_depths_vs_inventory.png", dpi=180)
    plt.close(fig)
    (OUT / "alpha_reduced_closed_ansatz_depths_vs_inventory.csv").write_text("\n".join(rows) + "\n")


def plot_alpha_galerkin_depths_vs_inventory(alpha_y_aware_u: np.ndarray) -> None:
    """Depths vs inventory at selected imbalance values: fast-alpha RHJB vs the two
    retained closed forms --- sc-cubic (body ansatz) and Galerkin-cubic (appendix)."""
    y_values = [-3.0, 0.0, 3.0]
    q_plot = Q[q_plot_mask()]
    fig, ax = plt.subplots(len(y_values), 2, figsize=(12, 9), sharex=True, sharey=True)
    rows = ["y,q,ask_hjb,bid_hjb,ask_sc,bid_sc,ask_gal,bid_gal"]
    for row_idx, y_value in enumerate(y_values):
        yy = np.full(q_plot.shape, y_value, dtype=float)
        ask_hjb, bid_hjb = y_aware_depths(alpha_y_aware_u, q_plot, yy)
        ask_sc, bid_sc = toy_h_sc_cubic_depths(q_plot, yy)
        ask_gal, bid_gal = galerkin_cubic_depths(q_plot, yy)
        ax[row_idx, 0].plot(q_plot, ask_hjb, "-", lw=2.6, color=COLORS["Alpha/Y-aware"], label="Alpha/I-aware RHJB")
        ax[row_idx, 0].plot(q_plot, ask_sc, "--", lw=2.5, color=COLORS["Alpha/Y sc-cubic"], label="self-consistent cubic")
        ax[row_idx, 0].plot(q_plot, ask_gal, ":", lw=2.3, color=COLORS["Alpha/Y galerkin"], label="Galerkin-cubic")
        ax[row_idx, 0].set_title(f"Ask depth, i={y_value:g}")
        ax[row_idx, 1].plot(q_plot, bid_hjb, "-", lw=2.6, color=COLORS["Alpha/Y-aware"], label="Alpha/I-aware RHJB")
        ax[row_idx, 1].plot(q_plot, bid_sc, "--", lw=2.5, color=COLORS["Alpha/Y sc-cubic"], label="self-consistent cubic")
        ax[row_idx, 1].plot(q_plot, bid_gal, ":", lw=2.3, color=COLORS["Alpha/Y galerkin"], label="Galerkin-cubic")
        ax[row_idx, 1].set_title(f"Bid depth, i={y_value:g}")
        for values in zip(q_plot, ask_hjb, bid_hjb, ask_sc, bid_sc, ask_gal, bid_gal):
            rows.append(f"{y_value:.10g}," + ",".join(f"{x:.10g}" for x in values))
    for a in ax.ravel():
        a.axhline(0.0, color="black", lw=1, alpha=0.45)
        a.axvline(0.0, color="black", lw=0.8, alpha=0.25)
        a.set_xlabel("inventory q")
        a.set_ylabel("Raw unconstrained depth")
        a.grid(True, alpha=0.25)
    ax[0, 0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "alpha_reduced_galerkin_depths_vs_inventory.png", dpi=180)
    plt.close(fig)
    (OUT / "alpha_reduced_galerkin_depths_vs_inventory.csv").write_text("\n".join(rows) + "\n")


def plot_alpha_galerkin_depths_vs_imbalance(alpha_y_aware_u: np.ndarray) -> None:
    """Depths vs imbalance at selected inventory values: fast-alpha RHJB vs the two
    retained closed forms --- sc-cubic (body ansatz) and Galerkin-cubic (appendix)."""
    q_values = [q for q in [-5, 0, 10, 15] if -P.qmax < q < P.qmax]
    yy = np.linspace(Y[0], Y[-1], 301)
    fig, ax = plt.subplots(len(q_values), 2, figsize=(12, 11), sharex=True)
    rows = ["y,q,ask_hjb,bid_hjb,ask_sc,bid_sc,ask_gal,bid_gal"]
    for row_idx, q_value in enumerate(q_values):
        q_arr = np.full_like(yy, int(q_value), dtype=int)
        ask_hjb, bid_hjb = y_aware_depths(alpha_y_aware_u, q_arr, yy)
        ask_sc, bid_sc = toy_h_sc_cubic_depths(q_arr, yy)
        ask_gal, bid_gal = galerkin_cubic_depths(q_arr, yy)
        ax[row_idx, 0].plot(yy, ask_hjb, "-", lw=2.6, color=COLORS["Alpha/Y-aware"], label="Alpha/I-aware RHJB")
        ax[row_idx, 0].plot(yy, ask_sc, "--", lw=2.5, color=COLORS["Alpha/Y sc-cubic"], label="self-consistent cubic")
        ax[row_idx, 0].plot(yy, ask_gal, ":", lw=2.3, color=COLORS["Alpha/Y galerkin"], label="Galerkin-cubic")
        ax[row_idx, 0].set_title(f"Ask depth, q={q_value}")
        ax[row_idx, 1].plot(yy, bid_hjb, "-", lw=2.6, color=COLORS["Alpha/Y-aware"], label="Alpha/I-aware RHJB")
        ax[row_idx, 1].plot(yy, bid_sc, "--", lw=2.5, color=COLORS["Alpha/Y sc-cubic"], label="self-consistent cubic")
        ax[row_idx, 1].plot(yy, bid_gal, ":", lw=2.3, color=COLORS["Alpha/Y galerkin"], label="Galerkin-cubic")
        ax[row_idx, 1].set_title(f"Bid depth, q={q_value}")
        for values in zip(yy, q_arr, ask_hjb, bid_hjb, ask_sc, bid_sc, ask_gal, bid_gal):
            rows.append(f"{values[0]:.10g}," + ",".join(f"{x:.10g}" for x in values[1:]))
    for a in ax.ravel():
        a.axhline(0.0, color="black", lw=1, alpha=0.45)
        a.axvline(0.0, color="black", lw=0.8, alpha=0.25)
        a.set_xlabel("OU state i")
        a.set_ylabel("Raw unconstrained depth")
        a.grid(True, alpha=0.25)
    ax[0, 0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "alpha_reduced_galerkin_depths_vs_imbalance.png", dpi=180)
    plt.close(fig)
    (OUT / "alpha_reduced_galerkin_depths_vs_imbalance.csv").write_text("\n".join(rows) + "\n")


def plot_depth_heatmaps(y_aware_u: np.ndarray, alpha_y_aware_u: np.ndarray) -> None:
    glft_ask, glft_bid = glft_depth_grid()
    y_ask, y_bid = depth_grid_from_u(y_aware_u)
    ay_ask, ay_bid = depth_grid_from_u(alpha_y_aware_u)
    panels = [glft_ask, y_ask, ay_ask, glft_bid, y_bid, ay_bid]
    abs_values = np.concatenate([np.abs(panel[np.isfinite(panel)]).ravel() for panel in panels])
    vmax = float(np.nanpercentile(abs_values, 98.0))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    titles = [
        "GLFT ask",
        "I-aware RHJB ask",
        "Alpha/I-aware RHJB ask",
        "GLFT bid",
        "I-aware RHJB bid",
        "Alpha/I-aware RHJB bid",
    ]
    fig, ax = plt.subplots(2, 3, figsize=(14.5, 7.8), sharex=True, sharey=True)
    for a, data, title in zip(ax.ravel(), panels, titles):
        plot_data = np.where(np.isfinite(data), data, np.nan)
        im = a.imshow(
            plot_data,
            extent=[Y[0], Y[-1], Q[0], Q[-1]],
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
        )
        a.set_title(title)
        a.set_xlabel("OU state i")
        a.set_ylabel("Inventory q")
    fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.85)
    fig.savefig(OUT / "alpha_reduced_depth_heatmaps.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_paths(paths: dict[str, object]) -> None:
    policies = list(paths["total"])
    fig, ax = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for name in policies:
        color = COLORS[name]
        ax[0, 0].plot(paths["times"], paths["total"][name], lw=2.0, color=color, label=policy_display(name))
        ax[0, 1].plot(paths["times"], paths["decomposed"][name], lw=2.0, color=color, label=policy_display(name))
        ax[1, 0].plot(paths["times"], paths["penalized"][name], lw=2.0, color=color, label=policy_display(name))
        ax[1, 1].plot(paths["times"], paths["absq"][name], lw=2.0, color=color, label=policy_display(name))
    titles = [
        "Mean realized PnL",
        "Mean spread + alpha-drift PnL",
        "Mean penalized PnL",
        "Mean absolute inventory",
    ]
    for a, title in zip(ax.ravel(), titles):
        a.set_title(title)
        a.set_xlabel("time")
        a.grid(True, alpha=0.25)
    ax[0, 0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "alpha_reduced_simulation_paths.png", dpi=180)
    plt.close(fig)


def plot_inventory_by_y(paths: dict[str, object]) -> None:
    policies = list(paths["inventory_by_y"])
    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    for name in policies:
        ax.plot(paths["y_centers"], paths["inventory_by_y"][name], lw=2.4, color=COLORS[name], label=policy_display(name))
    ax.axhline(0.0, color="black", lw=1, alpha=0.45)
    ax.axvline(0.0, color="black", lw=1, alpha=0.25)
    ax.set_xlabel("OU state i")
    ax.set_ylabel(r"$E[Q_t\mid I_t=i]$")
    ax.set_title("Conditional inventory by imbalance")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "alpha_reduced_expected_inventory_vs_y.png", dpi=180)
    plt.close(fig)


def plot_inventory_by_alpha(paths: dict[str, object]) -> None:
    policies = list(paths["inventory_by_alpha"])
    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    for name in policies:
        ax.plot(
            paths["alpha_centers"],
            paths["inventory_by_alpha"][name],
            lw=2.4,
            color=COLORS[name],
            label=policy_display(name),
        )
    ax.axhline(0.0, color="black", lw=1, alpha=0.45)
    ax.axvline(0.0, color="black", lw=1, alpha=0.25)
    ax.set_xlabel("realized alpha")
    ax.set_ylabel(r"$E[Q_t\mid \alpha_t]$")
    ax.set_title("Conditional inventory by realized alpha")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "alpha_reduced_expected_inventory_vs_alpha.png", dpi=180)
    plt.close(fig)


def plot_depth_by_alpha(paths: dict[str, object]) -> None:
    policies = list(paths["ask_by_alpha"])
    centers = paths["depth_alpha_centers"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for name in policies:
        ax[0].plot(centers, paths["ask_by_alpha"][name], lw=2.2, color=COLORS[name], label=policy_display(name))
        ax[1].plot(centers, paths["bid_by_alpha"][name], lw=2.2, color=COLORS[name], label=policy_display(name))
    ax[0].set_title(r"Realized ask depth $E[\delta^a\mid\alpha_t]$")
    ax[1].set_title(r"Realized bid depth $E[\delta^b\mid\alpha_t]$")
    for a in ax:
        a.axhline(0.0, color="black", lw=0.8, alpha=0.35)
        a.axvline(0.0, color="black", lw=0.6, alpha=0.25)
        a.set_xlabel("realized alpha")
        a.set_ylabel("signed depth")
        a.grid(True, alpha=0.25)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "alpha_reduced_depth_vs_alpha.png", dpi=180)
    plt.close(fig)


def make_lifetime_inventory_histograms(
    lifetime_inventory: dict[str, np.ndarray],
) -> dict[str, object]:
    """Reduce raw lifetime inventories to the exact curves used by the plot.

    Keeping counts rather than the raw recorded inventories makes the replot
    cache compact while preserving every ordinate in the four-panel figure.
    """
    bins = np.arange(-P.qmax - 0.5, P.qmax + 1.5, 1.0)
    centers = np.arange(-P.qmax, P.qmax + 1)
    abs_bins = np.arange(-0.5, P.qmax + 1.5, 1.0)
    abs_centers = np.arange(0, P.qmax + 1)
    q_counts: dict[str, np.ndarray] = {}
    abs_counts: dict[str, np.ndarray] = {}
    n_observations: dict[str, int] = {}
    for name, inventory in lifetime_inventory.items():
        qvals = np.asarray(inventory, dtype=float).ravel()
        if not np.all(np.isfinite(qvals)):
            raise ValueError(f"non-finite lifetime inventory for {name}")
        counts, _ = np.histogram(qvals, bins=bins)
        absolute_counts, _ = np.histogram(np.abs(qvals), bins=abs_bins)
        if int(np.sum(counts)) != qvals.size or int(np.sum(absolute_counts)) != qvals.size:
            raise ValueError(f"lifetime inventory outside [-qmax,qmax] for {name}")
        q_counts[name] = counts.astype(np.int64, copy=False)
        abs_counts[name] = absolute_counts.astype(np.int64, copy=False)
        n_observations[name] = int(qvals.size)
    return {
        "q_centers": centers,
        "abs_centers": abs_centers,
        "q_counts": q_counts,
        "abs_counts": abs_counts,
        "n_observations": n_observations,
    }


def _resolve_lifetime_inventory_histograms(
    paths: dict[str, object],
) -> dict[str, object]:
    """Return current histogram data, accepting the legacy raw-path format."""
    histograms = paths.get("lifetime_inventory_histograms")
    if histograms is not None:
        return histograms
    lifetime_inventory = paths.get("lifetime_inventory")
    if lifetime_inventory is None:
        raise KeyError(
            "paths must contain either 'lifetime_inventory_histograms' "
            "or legacy 'lifetime_inventory' data"
        )
    return make_lifetime_inventory_histograms(lifetime_inventory)


def plot_lifetime_inventory(paths: dict[str, object]) -> None:
    histograms = _resolve_lifetime_inventory_histograms(paths)
    policies = list(histograms["q_counts"])
    centers = np.asarray(histograms["q_centers"])
    abs_centers = np.asarray(histograms["abs_centers"])
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for name in policies:
        n_observations = max(1, int(histograms["n_observations"][name]))
        counts = np.asarray(histograms["q_counts"][name], dtype=np.int64)
        prob = counts / n_observations
        ax[0, 0].plot(centers, prob, drawstyle="steps-mid", lw=2.0, color=COLORS[name], label=policy_display(name))
        zoom = (centers >= -10) & (centers <= 10)
        ax[0, 1].plot(centers[zoom], prob[zoom], drawstyle="steps-mid", lw=2.0, color=COLORS[name], label=policy_display(name))
        absolute_counts = np.asarray(histograms["abs_counts"][name], dtype=np.int64)
        abs_prob = absolute_counts / n_observations
        ax[1, 0].plot(abs_centers, abs_prob, drawstyle="steps-mid", lw=2.0, color=COLORS[name], label=policy_display(name))
        tail = np.array([
            np.sum(absolute_counts[abs_centers > x]) / n_observations
            for x in abs_centers
        ])
        ax[1, 1].plot(abs_centers, tail, lw=2.0, color=COLORS[name], label=policy_display(name))
    titles = [
        "Lifetime inventory probability",
        "Lifetime inventory central zoom",
        "Absolute inventory probability",
        r"Tail probability $P(|Q|>x)$",
    ]
    for a, title in zip(ax.ravel(), titles):
        a.set_title(title)
        a.grid(True, alpha=0.25)
        a.legend()
    fig.tight_layout()
    fig.savefig(OUT / "alpha_reduced_lifetime_inventory_hist.png", dpi=180)
    plt.close(fig)


def plot_pnl_hist(paths: dict[str, object]) -> None:
    policies = list(paths["terminal_total"])
    all_total = np.concatenate([np.asarray(paths["terminal_total"][name]) for name in policies])
    all_decomp = np.concatenate([np.asarray(paths["terminal_decomposed"][name]) for name in policies])
    bins_total = np.linspace(*np.quantile(all_total, [0.005, 0.995]), 80)
    bins_decomp = np.linspace(*np.quantile(all_decomp, [0.005, 0.995]), 80)
    centers_total = 0.5 * (bins_total[:-1] + bins_total[1:])
    centers_decomp = 0.5 * (bins_decomp[:-1] + bins_decomp[1:])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for name in policies:
        counts, _ = np.histogram(paths["terminal_total"][name], bins=bins_total)
        ax[0].plot(centers_total, counts / max(1, len(paths["terminal_total"][name])), drawstyle="steps-mid", lw=2, color=COLORS[name], label=policy_display(name))
        counts, _ = np.histogram(paths["terminal_decomposed"][name], bins=bins_decomp)
        ax[1].plot(centers_decomp, counts / max(1, len(paths["terminal_decomposed"][name])), drawstyle="steps-mid", lw=2, color=COLORS[name], label=policy_display(name))
    ax[0].set_title("Realized terminal PnL")
    ax[1].set_title("Spread + alpha terminal PnL")
    for a in ax:
        a.grid(True, alpha=0.25)
        a.legend()
    fig.tight_layout()
    fig.savefig(OUT / "alpha_reduced_pnl_hist.png", dpi=180)
    plt.close(fig)


def plot_pnl_kde(paths: dict[str, object]) -> None:
    policies = list(paths["terminal_total"])
    all_total = np.concatenate([np.asarray(paths["terminal_total"][name]) for name in policies])
    all_decomp = np.concatenate([np.asarray(paths["terminal_decomposed"][name]) for name in policies])
    total_grid = np.linspace(*np.quantile(all_total, [0.0025, 0.9975]), 500)
    decomp_grid = np.linspace(*np.quantile(all_decomp, [0.0025, 0.9975]), 500)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for name in policies:
        total = np.asarray(paths["terminal_total"][name], dtype=float)
        decomp = np.asarray(paths["terminal_decomposed"][name], dtype=float)
        if np.std(total) > 0.0:
            ax[0].plot(total_grid, gaussian_kde(total)(total_grid), lw=2.2, color=COLORS[name], label=policy_display(name))
        if np.std(decomp) > 0.0:
            ax[1].plot(decomp_grid, gaussian_kde(decomp)(decomp_grid), lw=2.2, color=COLORS[name], label=policy_display(name))
    ax[0].set_title("Terminal PnL kernel density")
    ax[1].set_title("Spread + alpha PnL kernel density")
    for a in ax:
        a.grid(True, alpha=0.25)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "alpha_reduced_pnl_kde.png", dpi=180)
    plt.close(fig)


def plot_summary(metrics: dict[str, dict[str, float]]) -> None:
    policies = list(metrics)
    fields = [
        ("total_pnl_mean", "Total"),
        ("penalized_pnl_mean", "Penalized"),
        ("spread_capture_mean", "Spread"),
        ("alpha_drift_pnl_mean", "Alpha drift"),
        ("price_martingale_pnl_std", "Martingale std"),
        ("avg_abs_inventory", "Avg |Q|"),
    ]
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.5))
    colors = colors_for(policies)
    for a, (field, title) in zip(ax.ravel(), fields):
        a.bar([policy_tick(name) for name in policies], [metrics[name][field] for name in policies], color=colors)
        a.set_title(title)
        # anchor the rotated multi-line policy names at their right edge, else the
        # long canonical names ("Alpha/I self-consistent cubic") overprint each other
        plt.setp(a.get_xticklabels(), rotation=40, ha="right",
                 rotation_mode="anchor", fontsize=7.5)
        a.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "alpha_reduced_summary_bars.png", dpi=180)
    plt.close(fig)


def plot_signal_alignment(metrics: dict[str, dict[str, float]]) -> None:
    policies = list(metrics)
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.8))
    fields = [
        ("avg_q_y", r"$T^{-1}\int Q_tI_t\,dt$"),
        ("avg_q_alpha", r"$T^{-1}\int Q_t\alpha_tdt$"),
        ("alpha_drift_pnl_mean", r"$\int Q_t\alpha_tdt$"),
    ]
    colors = colors_for(policies)
    for a, (field, title) in zip(ax, fields):
        a.bar([policy_tick(name) for name in policies], [metrics[name][field] for name in policies], color=colors)
        a.set_title(title)
        # anchor the rotated multi-line policy names at their right edge, else the
        # long canonical names ("Alpha/I self-consistent cubic") overprint each other
        plt.setp(a.get_xticklabels(), rotation=40, ha="right",
                 rotation_mode="anchor", fontsize=7.5)
        a.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "alpha_reduced_signal_alignment.png", dpi=180)
    plt.close(fig)


def plot_sample_path(paths: dict[str, object]) -> None:
    sample = paths["sample"]
    fig, ax = plt.subplots(5, 1, figsize=(12, 10.5), sharex=True)
    ax[0].plot(sample["times"], sample["price"], color="black", lw=1.6)
    ax[0].set_ylabel("price")
    ax[1].plot(sample["times"], sample["y"], color="#55a868", lw=1.6, label="I")
    ax[1].legend()
    ax[1].set_ylabel("I")
    ax[2].plot(sample["times"], sample["alpha"], color="#c44e52", lw=1.5, label="alpha")
    ax[2].axhline(0.0, color="black", lw=0.8, alpha=0.35)
    ax[2].legend()
    ax[2].set_ylabel("alpha")
    for name, inv in sample["inventory"].items():
        ax[3].plot(sample["times"], inv, lw=1.7, label=policy_display(name))
    ax[3].legend()
    ax[3].set_ylabel("inventory")
    for name, inv in sample["inventory"].items():
        ax[4].plot(sample["times"], np.abs(inv), lw=1.7, label=policy_display(name))
    ax[4].set_ylabel("|inventory|")
    ax[4].set_xlabel("time")
    for a in ax:
        a.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "alpha_reduced_sample_path.png", dpi=180)
    plt.close(fig)


def finite_horizon_state_surfaces(
    alpha_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tau_grid = np.linspace(0.0, P.surface_T, max(2, P.surface_nt))
    target_steps = np.unique(np.rint(tau_grid / P.surface_dt_hjb).astype(int))
    step_to_pos = {int(step): int(pos) for pos, step in enumerate(target_steps)}
    tau_recorded = target_steps.astype(float) * P.surface_dt_hjb

    surface_qmax = min(P.surface_qmax, P.qmax)
    q_pad = max(0, P.surface_q_pad)
    q_hidden = np.arange(-(surface_qmax + q_pad), surface_qmax + q_pad + 1, dtype=int)
    q_hidden_f = q_hidden.astype(float)
    n_hidden = len(q_hidden)
    q0_hidden_idx = int(np.where(q_hidden == 0)[0][0])
    visible = (q_hidden >= -surface_qmax) & (q_hidden <= surface_qmax)
    q_visible = q_hidden[visible]
    ask = np.full((len(target_steps), len(q_visible)), np.nan)
    bid = np.full((len(target_steps), len(q_visible)), np.nan)

    inv_e = []
    for lp, lm, alpha_value in zip(LP, LM, alpha_values):
        diagonal = P.k * q_hidden_f * alpha_value - P.alpha_c * q_hidden_f**2
        mat = np.diag(diagonal)
        idx = np.arange(n_hidden - 1)
        mat[idx + 1, idx] = P.chi_gamma * lp
        mat[idx, idx + 1] = P.chi_gamma * lm
        inv_e.append(expm(P.surface_dt_hjb * mat))
    inv_e = np.stack(inv_e)
    ou_e_half = build_ou_operator(0.5 * P.surface_dt_hjb)

    steps = int(target_steps[-1])
    u = np.zeros((n_hidden, P.ny))

    def normalize_hidden(u_now: np.ndarray) -> np.ndarray:
        return u_now - u_now[q0_hidden_idx, Y0_IDX]

    def inventory_step_hidden(u_now: np.ndarray) -> np.ndarray:
        shift = np.max(u_now, axis=0)
        w = np.exp(np.clip(P.k * (u_now - shift[None, :]), -700.0, 700.0))
        w_next = np.empty_like(w)
        for j in range(P.ny):
            w_next[:, j] = inv_e[j] @ w[:, j]
        return np.log(np.maximum(w_next, 1e-300)) / P.k + shift[None, :]

    def depth_grid_hidden(u_now: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ask_hidden = np.full((n_hidden, P.ny), np.nan)
        bid_hidden = np.full((n_hidden, P.ny), np.nan)
        ask_hidden[1:] = P.delta0 - u_now[:-1] + u_now[1:]
        bid_hidden[:-1] = P.delta0 - u_now[1:] + u_now[:-1]
        return ask_hidden, bid_hidden

    u = normalize_hidden(u)

    def record(step: int, u_now: np.ndarray) -> None:
        pos = step_to_pos.get(step)
        if pos is None:
            return
        ask_grid, bid_grid = depth_grid_hidden(u_now)
        ask[pos] = ask_grid[visible, Y0_IDX]
        bid[pos] = bid_grid[visible, Y0_IDX]

    record(0, u)
    for step in range(1, steps + 1):
        u = u @ ou_e_half.T
        u = inventory_step_hidden(u)
        u = u @ ou_e_half.T
        u = normalize_hidden(u)
        record(step, u)
    return tau_recorded, q_visible, ask, bid, ask + bid


def finite_horizon_glft_surfaces() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tau_grid = np.linspace(0.0, P.surface_T, max(2, P.surface_nt))
    surface_qmax = min(P.surface_qmax, P.qmax)
    q_pad = max(0, P.surface_q_pad)
    q_hidden = np.arange(-(surface_qmax + q_pad), surface_qmax + q_pad + 1, dtype=int)
    q_hidden_f = q_hidden.astype(float)
    n_hidden = len(q_hidden)
    mat = np.diag(-P.alpha_c * q_hidden_f**2)
    idx = np.arange(n_hidden - 1)
    mat[idx, idx + 1] = P.eta_c
    mat[idx + 1, idx] = P.eta_c
    evals, evecs = eigh(mat)
    lambda_max = float(np.max(evals))
    coeff = evecs.T @ np.ones(n_hidden)
    ask_hidden = np.full((len(tau_grid), n_hidden), np.nan)
    bid_hidden = np.full((len(tau_grid), n_hidden), np.nan)
    q0_hidden_idx = int(np.where(q_hidden == 0)[0][0])
    for it, tau in enumerate(tau_grid):
        theta = evecs @ (np.exp((evals - lambda_max) * tau) * coeff)
        if theta[q0_hidden_idx] < 0.0:
            theta *= -1.0
        theta = np.maximum(theta, 1e-300)
        theta /= theta[q0_hidden_idx]
        ask_hidden[it, 1:] = P.delta0 + np.log(theta[1:] / theta[:-1]) / P.k
        bid_hidden[it, :-1] = P.delta0 + np.log(theta[:-1] / theta[1:]) / P.k
    visible = (q_hidden >= -surface_qmax) & (q_hidden <= surface_qmax)
    q_visible = q_hidden[visible]
    ask = ask_hidden[:, visible]
    bid = bid_hidden[:, visible]
    return tau_grid, q_visible, ask, bid, ask + bid


def plot_surface(
    tau_grid: np.ndarray,
    q_values: np.ndarray,
    values: np.ndarray,
    title: str,
    zlabel: str,
    filename: str,
    cmap: str = "viridis",
    azim: float = -60.0,
    zlim: tuple[float, float] | None = None,
    zticks: np.ndarray | None = None,
) -> None:
    qq, tt = np.meshgrid(q_values, tau_grid)
    finite = values[np.isfinite(values)]
    if zlim is None and finite.size:
        lo, hi = np.nanpercentile(finite, [1.0, 99.0])
        pad = 0.05 * max(1e-8, hi - lo)
        zlim = (lo - pad, hi + pad)
    fig = plt.figure(figsize=(7.6, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        qq,
        tt,
        values,
        cmap=cmap,
        linewidth=0.12,
        edgecolor=(0, 0, 0, 0.22),
        antialiased=True,
    )
    ax.set_xlabel("Inventory")
    ax.set_ylabel("Time [Sec]")
    ax.set_zlabel(zlabel)
    ax.set_xlim(float(q_values.min()), float(q_values.max()))
    ax.set_ylim(float(tau_grid.min()), float(tau_grid.max()))
    if zlim is not None:
        ax.set_zlim(*zlim)
    if zticks is not None:
        ax.set_zticks(zticks)
    ax.view_init(elev=24, azim=azim)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def write_surface_csv(
    tau_grid: np.ndarray,
    q_values: np.ndarray,
    values: np.ndarray,
    filename: str,
    value_name: str,
) -> None:
    qq, tt = np.meshgrid(q_values, tau_grid)
    data = np.column_stack([tt.ravel(), qq.ravel(), values.ravel()])
    np.savetxt(
        OUT / filename,
        data,
        delimiter=",",
        header=f"tau,q,{value_name}",
        comments="",
    )


def plot_glft_style_surfaces() -> None:
    tau_g, q_g, ask_g, bid_g, spread_g = finite_horizon_glft_surfaces()
    # The GLFT paper-style z-axis limits are appropriate for the paper
    # parameterization, where k=0.3.  In the alpha experiments we sometimes
    # change k to study tradability of alpha; then depths live on a different
    # scale and fixed paper limits visually flatten the surface.
    use_paper_axis = abs(P.k - 0.3) < 1e-12 and abs(P.bar_lambda - 0.9) < 1e-12
    quote_zlim = (1.0, 5.5) if use_paper_axis else None
    quote_zticks = np.arange(1.0, 5.51, 0.5) if use_paper_axis else None
    spread_zlim = (6.55, 6.64) if use_paper_axis else None
    spread_zticks = np.arange(6.55, 6.641, 0.01) if use_paper_axis else None
    plot_surface(tau_g, q_g, bid_g, "GLFT bid depth", r"$s-s^b$ [Tick]", "alpha_reduced_glft_bid_surface.png", azim=-135.0, zlim=quote_zlim, zticks=quote_zticks)
    plot_surface(tau_g, q_g, ask_g, "GLFT ask depth", r"$s^a-s$ [Tick]", "alpha_reduced_glft_ask_surface.png", zlim=quote_zlim, zticks=quote_zticks)
    plot_surface(tau_g, q_g, spread_g, "GLFT bid-ask spread", r"$\psi$ [Tick]", "alpha_reduced_glft_spread_surface.png", zlim=spread_zlim, zticks=spread_zticks)
    write_surface_csv(tau_g, q_g, bid_g, "alpha_reduced_glft_bid_surface.csv", "bid_depth")
    write_surface_csv(tau_g, q_g, ask_g, "alpha_reduced_glft_ask_surface.csv", "ask_depth")
    write_surface_csv(tau_g, q_g, spread_g, "alpha_reduced_glft_spread_surface.csv", "spread")

    tau_y, q_y, ask_y, bid_y, spread_y = finite_horizon_state_surfaces(np.zeros_like(Y))
    plot_surface(tau_y, q_y, bid_y, "I-aware RHJB bid depth at i=0", r"$s-s^b$ [Tick]", "alpha_reduced_yaware_bid_surface_y0.png", cmap="YlOrRd", azim=-135.0)
    plot_surface(tau_y, q_y, ask_y, "I-aware RHJB ask depth at i=0", r"$s^a-s$ [Tick]", "alpha_reduced_yaware_ask_surface_y0.png", cmap="YlOrRd")
    plot_surface(tau_y, q_y, spread_y, "I-aware RHJB bid-ask spread at i=0", r"$\psi$ [Tick]", "alpha_reduced_yaware_spread_surface_y0.png", cmap="YlOrRd")
    write_surface_csv(tau_y, q_y, bid_y, "alpha_reduced_yaware_bid_surface_y0.csv", "bid_depth")
    write_surface_csv(tau_y, q_y, ask_y, "alpha_reduced_yaware_ask_surface_y0.csv", "ask_depth")
    write_surface_csv(tau_y, q_y, spread_y, "alpha_reduced_yaware_spread_surface_y0.csv", "spread")

    tau_a, q_a, ask_a, bid_a, spread_a = finite_horizon_state_surfaces(ALPHA_BAR)
    plot_surface(tau_a, q_a, bid_a, "Alpha/I-aware RHJB bid depth at i=0", r"$s-s^b$ [Tick]", "alpha_reduced_alpha_yaware_bid_surface_y0.png", cmap="coolwarm", azim=-135.0)
    plot_surface(tau_a, q_a, ask_a, "Alpha/I-aware RHJB ask depth at i=0", r"$s^a-s$ [Tick]", "alpha_reduced_alpha_yaware_ask_surface_y0.png", cmap="coolwarm")
    plot_surface(tau_a, q_a, spread_a, "Alpha/I-aware RHJB bid-ask spread at i=0", r"$\psi$ [Tick]", "alpha_reduced_alpha_yaware_spread_surface_y0.png", cmap="coolwarm")
    write_surface_csv(tau_a, q_a, bid_a, "alpha_reduced_alpha_yaware_bid_surface_y0.csv", "bid_depth")
    write_surface_csv(tau_a, q_a, ask_a, "alpha_reduced_alpha_yaware_ask_surface_y0.csv", "ask_depth")
    write_surface_csv(tau_a, q_a, spread_a, "alpha_reduced_alpha_yaware_spread_surface_y0.csv", "spread")


REPLOT_CACHE_SCHEMA_VERSION = 1


def _stack_policy_arrays(
    mapping: dict[str, object],
    policy_names: list[str],
    field_name: str,
    dtype: type | None = None,
) -> np.ndarray:
    """Stack same-shaped policy arrays in an explicit, stable policy order."""
    missing = [name for name in policy_names if name not in mapping]
    if missing:
        raise KeyError(f"{field_name} is missing policies: {missing}")
    arrays = [np.asarray(mapping[name], dtype=dtype) for name in policy_names]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"{field_name} policy arrays have inconsistent shapes: {shapes}")
    return np.stack(arrays, axis=0)


def save_replot_data(
    paths: dict[str, object],
    cache_path: Path | str | None = None,
) -> Path:
    """Persist the minimal lossless data needed for five Monte Carlo figures.

    Policy-indexed quantities are stored as numeric matrices rather than as NPZ
    keys derived from policy labels.  This keeps every key deterministic and
    permits loading with ``allow_pickle=False`` even though labels contain '/' or
    non-ASCII text in their paper-facing forms.
    """
    policy_names = list(paths["terminal_total"])
    if not policy_names or len(set(policy_names)) != len(policy_names):
        raise ValueError("policy names must be non-empty and unique")

    histograms = _resolve_lifetime_inventory_histograms(paths)
    sample = paths["sample"]
    max_name_length = max(len(name) for name in policy_names)
    unicode_policy_names = np.asarray(policy_names, dtype=f"<U{max_name_length}")

    payload = {
        "schema_version": np.asarray(REPLOT_CACHE_SCHEMA_VERSION, dtype=np.int64),
        "policy_names": unicode_policy_names,
        "conditional_y_centers": np.asarray(paths["y_centers"], dtype=float),
        "conditional_y_mean": _stack_policy_arrays(
            paths["inventory_by_y"], policy_names, "inventory_by_y", float
        ),
        "conditional_y_counts": _stack_policy_arrays(
            paths["inventory_by_y_counts"], policy_names, "inventory_by_y_counts", np.int64
        ),
        "conditional_alpha_centers": np.asarray(paths["alpha_centers"], dtype=float),
        "conditional_alpha_mean": _stack_policy_arrays(
            paths["inventory_by_alpha"], policy_names, "inventory_by_alpha", float
        ),
        "conditional_alpha_counts": _stack_policy_arrays(
            paths["inventory_by_alpha_counts"], policy_names, "inventory_by_alpha_counts", np.int64
        ),
        "lifetime_q_centers": np.asarray(histograms["q_centers"], dtype=np.int64),
        "lifetime_q_counts": _stack_policy_arrays(
            histograms["q_counts"], policy_names, "lifetime q counts", np.int64
        ),
        "lifetime_abs_centers": np.asarray(histograms["abs_centers"], dtype=np.int64),
        "lifetime_abs_counts": _stack_policy_arrays(
            histograms["abs_counts"], policy_names, "lifetime absolute-q counts", np.int64
        ),
        "lifetime_n_observations": np.asarray(
            [histograms["n_observations"][name] for name in policy_names], dtype=np.int64
        ),
        # Terminal pathwise arrays are retained because the published histogram
        # chooses common bins from cross-policy empirical quantiles.
        "terminal_total": _stack_policy_arrays(
            paths["terminal_total"], policy_names, "terminal_total", float
        ),
        "terminal_decomposed": _stack_policy_arrays(
            paths["terminal_decomposed"], policy_names, "terminal_decomposed", float
        ),
        "sample_times": np.asarray(sample["times"], dtype=float),
        "sample_price": np.asarray(sample["price"], dtype=float),
        "sample_i": np.asarray(sample["y"], dtype=float),
        "sample_alpha": np.asarray(sample["alpha"], dtype=float),
        "sample_inventory": _stack_policy_arrays(
            sample["inventory"], policy_names, "sample inventory", float
        ),
    }

    target = OUT / "alpha_reduced_replot_data.npz" if cache_path is None else Path(cache_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **payload)
    return target


def load_replot_data(cache_path: Path | str | None = None) -> dict[str, object]:
    """Load :func:`save_replot_data` output into the plotting API's dictionary."""
    source = OUT / "alpha_reduced_replot_data.npz" if cache_path is None else Path(cache_path)
    with np.load(source, allow_pickle=False) as cache:
        version = int(np.asarray(cache["schema_version"]).item())
        if version != REPLOT_CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported replot-cache schema {version}; expected {REPLOT_CACHE_SCHEMA_VERSION}"
            )
        policy_names = [str(name) for name in cache["policy_names"]]

        def policy_dict(key: str) -> dict[str, np.ndarray]:
            matrix = np.asarray(cache[key]).copy()
            if matrix.ndim < 1 or matrix.shape[0] != len(policy_names):
                raise ValueError(f"{key} has incompatible policy dimension {matrix.shape}")
            return {name: matrix[j] for j, name in enumerate(policy_names)}

        n_observations = np.asarray(cache["lifetime_n_observations"], dtype=np.int64)
        if n_observations.shape != (len(policy_names),):
            raise ValueError("lifetime_n_observations has an incompatible policy dimension")
        return {
            "y_centers": np.asarray(cache["conditional_y_centers"], dtype=float).copy(),
            "inventory_by_y": policy_dict("conditional_y_mean"),
            "inventory_by_y_counts": policy_dict("conditional_y_counts"),
            "alpha_centers": np.asarray(cache["conditional_alpha_centers"], dtype=float).copy(),
            "inventory_by_alpha": policy_dict("conditional_alpha_mean"),
            "inventory_by_alpha_counts": policy_dict("conditional_alpha_counts"),
            "lifetime_inventory_histograms": {
                "q_centers": np.asarray(cache["lifetime_q_centers"], dtype=np.int64).copy(),
                "abs_centers": np.asarray(cache["lifetime_abs_centers"], dtype=np.int64).copy(),
                "q_counts": policy_dict("lifetime_q_counts"),
                "abs_counts": policy_dict("lifetime_abs_counts"),
                "n_observations": {
                    name: int(n_observations[j]) for j, name in enumerate(policy_names)
                },
            },
            "terminal_total": policy_dict("terminal_total"),
            "terminal_decomposed": policy_dict("terminal_decomposed"),
            "sample": {
                "times": np.asarray(cache["sample_times"], dtype=float).copy(),
                "price": np.asarray(cache["sample_price"], dtype=float).copy(),
                "y": np.asarray(cache["sample_i"], dtype=float).copy(),
                "alpha": np.asarray(cache["sample_alpha"], dtype=float).copy(),
                "inventory": policy_dict("sample_inventory"),
            },
        }


def replot_cached_mc_figures(cache_path: Path | str | None = None) -> None:
    """Regenerate the five legacy Monte Carlo figures without rerunning simulation."""
    cached_paths = load_replot_data(cache_path)
    plot_inventory_by_y(cached_paths)
    plot_inventory_by_alpha(cached_paths)
    plot_lifetime_inventory(cached_paths)
    plot_pnl_hist(cached_paths)
    plot_sample_path(cached_paths)


def write_outputs(
    metrics: dict[str, dict[str, float]],
    paths: dict[str, object],
    y_aware_info: dict[str, float],
    alpha_y_aware_info: dict[str, float],
) -> None:
    payload = {
        "params": asdict(P),
        "derived_cara_glft": {
            "delta_gamma": P.delta0,
            "chi_gamma": P.chi_gamma,
            "phi_eff": P.phi,
            "alpha_c": P.alpha_c,
            "eta_c": P.eta_c,
            "p_glft": P.p_glft,
            "rho": P.rho,
            "alpha_bar_slope": P.alpha_slope,
        },
        "ergodic_solvers": {
            "y_aware": y_aware_info,
            "alpha_y_aware": alpha_y_aware_info,
        },
        "natural_ansatz": local_ansatz_coefficients(),
        "metrics": metrics,
    }
    (OUT / "alpha_reduced_summary.json").write_text(json.dumps(payload, indent=2))
    fields = list(next(iter(metrics.values())).keys())
    lines = ["policy," + ",".join(fields)]
    for name, vals in metrics.items():
        lines.append(name + "," + ",".join(f"{vals[f]:.8g}" for f in fields))
    (OUT / "alpha_reduced_summary.csv").write_text("\n".join(lines) + "\n")
    save_replot_data(paths)


def main() -> None:
    print("Solving Y-aware flow-composition ergodic approximation...")
    y_aware_u, y_aware_info = solve_ou_ergodic(np.zeros_like(Y))
    print("Solving Alpha/Y-aware fast-alpha ergodic approximation...")
    alpha_y_aware_u, alpha_y_aware_info = solve_ou_ergodic(ALPHA_BAR)
    print("Simulating policies in the full alpha jump-diffusion environment...")
    metrics, paths = simulate_policies(y_aware_u, alpha_y_aware_u)

    # Paired penalized-PnL contrasts on shared environment paths.
    pp = paths["pen_paths"]
    ladder = [("I over GLFT", "Y-aware", "GLFT"),
              ("alpha over I", "Alpha/Y-aware", "Y-aware"),
              ("sc-cubic vs RHJB", "Alpha/Y sc-cubic", "Alpha/Y-aware"),
              ("galerkin vs RHJB", "Alpha/Y galerkin", "Alpha/Y-aware")]
    lines = ["contrast,mean_diff,paired_se,ci95_low,ci95_high,paired_corr"]
    print("\n=== Paired penalized-PnL contrasts (shared paths) ===")
    for label, hi, lo in ladder:
        if hi not in pp or lo not in pp:
            continue
        d = np.asarray(pp[hi]) - np.asarray(pp[lo])
        se = float(np.std(d, ddof=1) / np.sqrt(d.size))
        corr = float(np.corrcoef(pp[hi], pp[lo])[0, 1])
        md = float(np.mean(d))
        lines.append(f"{label},{md:.4f},{se:.4f},{md-1.96*se:.4f},{md+1.96*se:.4f},{corr:.4f}")
        print(f"  {label:24s} = {md:+9.2f}  paired se {se:6.2f}  CI95 [{md-1.96*se:+.2f}, {md+1.96*se:+.2f}]  corr {corr:.3f}")
    (OUT / "alpha_reduced_paired_contrasts.csv").write_text("\n".join(lines) + "\n")
    np.savez_compressed(
        OUT / "alpha_reduced_penalized_pnl_paths.npz",
        **{k.replace(" ", "_").replace("/", "_"): np.asarray(v) for k, v in pp.items()},
    )
    print(f"wrote {OUT / 'alpha_reduced_paired_contrasts.csv'}")

    # Auxiliary multi-form depth diagnostics disabled: the paper depth figures are the
    # galerkin_* pair below, restricted to {fast-alpha RHJB, sc-cubic, Galerkin-cubic}.
    # plot_quote_depths(y_aware_u, alpha_y_aware_u)
    # plot_alpha_closed_ansatz_depths_selected_q(alpha_y_aware_u)
    # plot_alpha_closed_ansatz_depths_vs_inventory(alpha_y_aware_u)
    plot_alpha_galerkin_depths_vs_inventory(alpha_y_aware_u)
    plot_alpha_galerkin_depths_vs_imbalance(alpha_y_aware_u)
    plot_depth_heatmaps(y_aware_u, alpha_y_aware_u)
    plot_paths(paths)
    plot_inventory_by_y(paths)
    plot_inventory_by_alpha(paths)
    plot_depth_by_alpha(paths)
    plot_lifetime_inventory(paths)
    plot_pnl_hist(paths)
    plot_pnl_kde(paths)
    plot_summary(metrics)
    plot_signal_alignment(metrics)
    plot_sample_path(paths)
    plot_glft_style_surfaces()
    write_outputs(metrics, paths, y_aware_info, alpha_y_aware_info)

    print(json.dumps(metrics, indent=2))
    print(f"Y-aware ergodic info: {json.dumps(y_aware_info)}")
    print(f"Alpha/Y-aware ergodic info: {json.dumps(alpha_y_aware_info)}")
    print(f"Fast-alpha closed ansatz: {json.dumps(local_ansatz_coefficients())}")
    print(f"Wrote outputs to {OUT}")


if __name__ == "__main__":
    main()
