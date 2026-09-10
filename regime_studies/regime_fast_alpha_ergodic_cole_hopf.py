#!/usr/bin/env python3
"""Regime-aware CARA/GLFT fast-alpha Cole-Hopf study.

The simulation environment contains slow regimes, OU imbalance, and the full
alpha jump-diffusion.  The compared policies are:

1. GLFT: inventory-only, signal blind (closed-form baseline).
2. Regime-aware: numerical RHJB observing both Y and the slow regime R.
3. Regime Galerkin-cubic: offline-calibrated closed form (appendix).
4. Regime self-consistent cubic: the body closed form (ungated
   regime channel A_R z_r).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parent
out_env = os.environ.get(
    "ASOU5_REGIME_OUTPUT_DIR",
    os.environ.get("ASOU4_REGIME_OUTPUT_DIR", "../imagens_tex/regime_studies_T10000_qmax60"),
)
OUT = Path(out_env)
if not OUT.is_absolute():
    OUT = ROOT / OUT
OUT.mkdir(parents=True, exist_ok=True)


def env_float(name: str, default: float) -> float:
    if name.startswith("ASOU5_REGIME_"):
        legacy = name.replace("ASOU5_REGIME_", "ASOU4_REGIME_", 1)
        return float(os.environ.get(name, os.environ.get(legacy, default)))
    return float(os.environ.get(name, default))


def env_int(name: str, default: int) -> int:
    if name.startswith("ASOU5_REGIME_"):
        legacy = name.replace("ASOU5_REGIME_", "ASOU4_REGIME_", 1)
        return int(os.environ.get(name, os.environ.get(legacy, default)))
    return int(os.environ.get(name, default))


def standard_error(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


@dataclass(frozen=True)
class Params:
    T: float = env_float("ASOU5_REGIME_T", 10000.0)
    qmax: int = env_int("ASOU5_REGIME_QMAX", 60)
    surface_qmax: int = env_int("ASOU5_REGIME_SURFACE_QMAX", 30)
    y_max: float = env_float("ASOU5_REGIME_Y_MAX", 6.0)
    ny: int = env_int("ASOU5_REGIME_NY", 121)
    dt_hjb: float = env_float("ASOU5_REGIME_DT_HJB", 0.25)
    n_paths: int = env_int("ASOU5_REGIME_N_PATHS", 20000)
    dt_sim: float = env_float("ASOU5_REGIME_DT_SIM", 0.05)
    record_dt: float = env_float("ASOU5_REGIME_RECORD_DT", 1.0)
    seed: int = env_int("ASOU5_REGIME_SEED", 24680)
    gamma: float = env_float("ASOU5_REGIME_GAMMA", 0.01)
    bar_lambda: float = env_float("ASOU5_REGIME_BAR_LAMBDA", 0.9)
    k: float = env_float("ASOU5_REGIME_K", 2.0)
    beta: float = env_float("ASOU5_REGIME_BETA", 0.0125)
    eta: float = env_float("ASOU5_REGIME_ETA", 0.32)
    phi_input: float = env_float("ASOU5_REGIME_PHI", -1.0)
    alpha_l: float = env_float("ASOU5_REGIME_ALPHA_L", 0.01)
    zeta: float = env_float("ASOU5_REGIME_ZETA", 0.25)
    epsilon: float = env_float("ASOU5_REGIME_EPSILON", 0.005)
    eta_alpha: float = env_float("ASOU5_REGIME_ETA_ALPHA", 0.001)
    sigma_price: float = env_float("ASOU5_REGIME_SIGMA_PRICE", 0.3)
    z_star: float = env_float("ASOU5_REGIME_Z_STAR", 1.25)
    regime_tau: float = env_float("ASOU5_REGIME_TAU", 120.0)
    ergodic_max_tau: float = env_float("ASOU5_REGIME_ERGODIC_MAX_TAU", 2500.0)
    ergodic_tol: float = env_float("ASOU5_REGIME_ERGODIC_TOL", 2e-8)

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

    @property
    def regime_rate(self) -> float:
        return 1.0 / self.regime_tau


P = Params()
Q = np.arange(-P.qmax, P.qmax + 1)
NQ = len(Q)
Y = np.linspace(-P.y_max, P.y_max, P.ny)
DY = Y[1] - Y[0]
Y0_IDX = int(np.argmin(np.abs(Y)))
REGIME_LABELS = np.array([-1, 0, 1], dtype=int)
Z_LEVELS = P.z_star * REGIME_LABELS.astype(float)
NR = len(REGIME_LABELS)
NEUTRAL_IDX = int(np.where(REGIME_LABELS == 0)[0][0])
LP = P.bar_lambda * (1.0 + np.tanh(Y))
LM = P.bar_lambda * (1.0 - np.tanh(Y))
ALPHA_BAR = P.alpha_slope * np.tanh(Y)


def regime_generator() -> np.ndarray:
    """Three-state persistent regime generator on (-, 0, +)."""
    rate = P.regime_rate
    qmat = np.zeros((NR, NR))
    qmat[0, 1] = rate
    qmat[1, 0] = 0.5 * rate
    qmat[1, 2] = 0.5 * rate
    qmat[2, 1] = rate
    qmat[np.diag_indices(NR)] = -np.sum(qmat, axis=1)
    return qmat


Q_REGIME = regime_generator()


def colors_for(policies: list[str]) -> list[str]:
    palette = {
        "GLFT": "#4c72b0",
        "Y-aware": "#55a868",
        "Alpha/Y-aware": "#c44e52",
        "Regime-aware": "#111111",
        "Regime Galerkin": "#B07AA1",
        "Regime sc-cubic": "#1B9E77",
        "Regime ansatz": "#e78ac3",
    }
    fallback = ["#a6d854", "#ffd92f", "#e5c494"]
    return [palette.get(name, fallback[i % len(fallback)]) for i, name in enumerate(policies)]


POLICY_DISPLAY = {
    "GLFT": "GLFT",
    "Y-aware": "I-aware RHJB",
    "Alpha/Y-aware": "Alpha/I-aware RHJB",
    "Regime-aware": "Regime-aware RHJB",
    "Regime Galerkin": "Regime Galerkin-cubic",
    "Regime sc-cubic": "Regime self-consistent cubic",
}


def policy_display(name: str) -> str:
    """Canonical paper-facing policy name for legends and tick labels."""
    return POLICY_DISPLAY.get(name, name.replace("Alpha/Y", "Alpha/I"))


def policy_tick(name: str) -> str:
    """Compact multiline variant used on dense bar-chart axes."""
    return {
        "Y-aware": "I-aware\nRHJB",
        "Alpha/Y-aware": "Alpha/I-aware\nRHJB",
        "Regime-aware": "Regime-aware\nRHJB",
        "Regime Galerkin": "Regime\nGalerkin-cubic",
        "Regime sc-cubic": "Regime self-\nconsistent cubic",
    }.get(name, policy_display(name))


def alpha_bar(y: np.ndarray) -> np.ndarray:
    return P.alpha_slope * np.tanh(y)


def build_yr_operator(dt: float) -> np.ndarray:
    """Matrix exponential for the joint (regime, y) generator."""
    n = NR * P.ny
    generator = np.zeros((n, n))

    for r_idx, z_level in enumerate(Z_LEVELS):
        drift = -P.beta * (Y - z_level)
        offset = r_idx * P.ny
        for j in range(P.ny):
            row = offset + j

            if j == 0:
                generator[row, offset + j] += drift[j] * (-1.0 / DY)
                generator[row, offset + j + 1] += drift[j] * (1.0 / DY)
                generator[row, offset + j] += 0.5 * P.eta**2 * (-2.0 / DY**2)
                generator[row, offset + j + 1] += 0.5 * P.eta**2 * (2.0 / DY**2)
            elif j == P.ny - 1:
                generator[row, offset + j - 1] += drift[j] * (-1.0 / DY)
                generator[row, offset + j] += drift[j] * (1.0 / DY)
                generator[row, offset + j - 1] += 0.5 * P.eta**2 * (2.0 / DY**2)
                generator[row, offset + j] += 0.5 * P.eta**2 * (-2.0 / DY**2)
            else:
                generator[row, offset + j - 1] += drift[j] * (-0.5 / DY)
                generator[row, offset + j + 1] += drift[j] * (0.5 / DY)
                generator[row, offset + j - 1] += 0.5 * P.eta**2 * (1.0 / DY**2)
                generator[row, offset + j] += 0.5 * P.eta**2 * (-2.0 / DY**2)
                generator[row, offset + j + 1] += 0.5 * P.eta**2 * (1.0 / DY**2)

    for r_idx in range(NR):
        for rp_idx in range(NR):
            rate = Q_REGIME[r_idx, rp_idx]
            if rate == 0.0:
                continue
            for j in range(P.ny):
                generator[r_idx * P.ny + j, rp_idx * P.ny + j] += rate

    return expm(dt * generator)


YR_E_HALF = build_yr_operator(0.5 * P.dt_hjb)


def build_y_operator(dt: float, z_level: float = 0.0) -> np.ndarray:
    """Matrix exponential for the OU generator with a fixed long-run mean."""
    generator = np.zeros((P.ny, P.ny))
    drift = -P.beta * (Y - z_level)

    for j in range(P.ny):
        if j == 0:
            generator[j, j] += drift[j] * (-1.0 / DY)
            generator[j, j + 1] += drift[j] * (1.0 / DY)
            generator[j, j] += 0.5 * P.eta**2 * (-2.0 / DY**2)
            generator[j, j + 1] += 0.5 * P.eta**2 * (2.0 / DY**2)
        elif j == P.ny - 1:
            generator[j, j - 1] += drift[j] * (-1.0 / DY)
            generator[j, j] += drift[j] * (1.0 / DY)
            generator[j, j - 1] += 0.5 * P.eta**2 * (2.0 / DY**2)
            generator[j, j] += 0.5 * P.eta**2 * (-2.0 / DY**2)
        else:
            generator[j, j - 1] += drift[j] * (-0.5 / DY)
            generator[j, j + 1] += drift[j] * (0.5 / DY)
            generator[j, j - 1] += 0.5 * P.eta**2 * (1.0 / DY**2)
            generator[j, j] += 0.5 * P.eta**2 * (-2.0 / DY**2)
            generator[j, j + 1] += 0.5 * P.eta**2 * (1.0 / DY**2)

    return expm(dt * generator)


Y_E_HALF = build_y_operator(0.5 * P.dt_hjb, 0.0)


def inventory_matrix(lambda_plus: float, lambda_minus: float, alpha_value: float) -> np.ndarray:
    diagonal = P.k * Q.astype(float) * alpha_value - P.alpha_c * Q.astype(float) ** 2
    mat = np.diag(diagonal)
    for i in range(NQ):
        if i > 0:
            mat[i, i - 1] = P.chi_gamma * lambda_plus
        if i < NQ - 1:
            mat[i, i + 1] = P.chi_gamma * lambda_minus
    return mat


INV_E_Y_ALPHA = np.stack(
    [expm(P.dt_hjb * inventory_matrix(lp, lm, a)) for lp, lm, a in zip(LP, LM, ALPHA_BAR)]
)
INV_E_Y_FLOW = np.stack(
    [expm(P.dt_hjb * inventory_matrix(lp, lm, 0.0)) for lp, lm in zip(LP, LM)]
)


def apply_yr_exp(u: np.ndarray, matrix: np.ndarray = YR_E_HALF) -> np.ndarray:
    flat = u.reshape(NQ, NR * P.ny)
    return (flat @ matrix.T).reshape(NQ, NR, P.ny)


def apply_y_exp(u: np.ndarray, matrix: np.ndarray = Y_E_HALF) -> np.ndarray:
    return u @ matrix.T


def normalize_common(u: np.ndarray) -> np.ndarray:
    return u - u[P.qmax, NEUTRAL_IDX, Y0_IDX]


def normalize_y(u: np.ndarray) -> np.ndarray:
    return u - u[P.qmax, Y0_IDX]


def alpha_inventory_step(u: np.ndarray) -> np.ndarray:
    shift = np.max(u, axis=0)
    w = np.exp(np.clip(P.k * (u - shift[None, :, :]), -700.0, 700.0))
    w_next = np.empty_like(w)
    for j in range(P.ny):
        w_next[:, :, j] = INV_E_Y_ALPHA[j] @ w[:, :, j]
    return np.log(np.maximum(w_next, 1e-300)) / P.k + shift[None, :, :]


def alpha_inventory_step_y(u: np.ndarray) -> np.ndarray:
    shift = np.max(u, axis=0)
    w = np.exp(np.clip(P.k * (u - shift[None, :]), -700.0, 700.0))
    w_next = np.empty_like(w)
    for j in range(P.ny):
        w_next[:, j] = INV_E_Y_ALPHA[j] @ w[:, j]
    return np.log(np.maximum(w_next, 1e-300)) / P.k + shift[None, :]


def flow_inventory_step_y(u: np.ndarray) -> np.ndarray:
    shift = np.max(u, axis=0)
    w = np.exp(np.clip(P.k * (u - shift[None, :]), -700.0, 700.0))
    w_next = np.empty_like(w)
    for j in range(P.ny):
        w_next[:, j] = INV_E_Y_FLOW[j] @ w[:, j]
    return np.log(np.maximum(w_next, 1e-300)) / P.k + shift[None, :]


def depth_grid_from_u(u_tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ask = np.full((NQ, NR, P.ny), np.nan)
    bid = np.full((NQ, NR, P.ny), np.nan)
    ask[1:] = P.delta0 - u_tau[:-1] + u_tau[1:]
    bid[:-1] = P.delta0 - u_tau[1:] + u_tau[:-1]
    return ask, bid


def depth_y_grid_from_u(u_tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ask = np.full((NQ, P.ny), np.nan)
    bid = np.full((NQ, P.ny), np.nan)
    ask[1:] = P.delta0 - u_tau[:-1] + u_tau[1:]
    bid[:-1] = P.delta0 - u_tau[1:] + u_tau[:-1]
    return ask, bid


def solve_regime_ergodic(check_every: int = 20) -> tuple[np.ndarray, dict[str, float]]:
    steps = int(round(P.ergodic_max_tau / P.dt_hjb))
    u = normalize_common(np.zeros((NQ, NR, P.ny)))
    prev_ask: np.ndarray | None = None
    prev_bid: np.ndarray | None = None
    last_diff = np.inf
    converged = False

    for step in range(1, steps + 1):
        u = apply_yr_exp(u, YR_E_HALF)
        u = alpha_inventory_step(u)
        u = apply_yr_exp(u, YR_E_HALF)
        u = normalize_common(u)

        if step % check_every == 0:
            ask, bid = depth_grid_from_u(u)
            if prev_ask is not None and prev_bid is not None:
                last_diff = float(
                    max(np.nanmax(np.abs(ask - prev_ask)), np.nanmax(np.abs(bid - prev_bid)))
                )
                if last_diff < P.ergodic_tol:
                    converged = True
                    break
            prev_ask = ask
            prev_bid = bid

    info = {
        "tau": float(step * P.dt_hjb),
        "steps": float(step),
        "last_depth_diff": float(last_diff),
        "converged": float(converged),
        "tolerance": float(P.ergodic_tol),
    }
    return u, info


def solve_y_aware_ergodic(check_every: int = 20) -> tuple[np.ndarray, dict[str, float]]:
    """Long-horizon fast-alpha profile that observes Y but not the slow regime."""
    steps = int(round(P.ergodic_max_tau / P.dt_hjb))
    u = normalize_y(np.zeros((NQ, P.ny)))
    prev_ask: np.ndarray | None = None
    prev_bid: np.ndarray | None = None
    last_diff = np.inf
    converged = False

    for step in range(1, steps + 1):
        u = apply_y_exp(u, Y_E_HALF)
        u = alpha_inventory_step_y(u)
        u = apply_y_exp(u, Y_E_HALF)
        u = normalize_y(u)

        if step % check_every == 0:
            ask, bid = depth_y_grid_from_u(u)
            if prev_ask is not None and prev_bid is not None:
                last_diff = float(
                    max(np.nanmax(np.abs(ask - prev_ask)), np.nanmax(np.abs(bid - prev_bid)))
                )
                if last_diff < P.ergodic_tol:
                    converged = True
                    break
            prev_ask = ask
            prev_bid = bid

    info = {
        "tau": float(step * P.dt_hjb),
        "steps": float(step),
        "last_depth_diff": float(last_diff),
        "converged": float(converged),
        "tolerance": float(P.ergodic_tol),
    }
    return u, info


def solve_y_flow_ergodic(check_every: int = 20) -> tuple[np.ndarray, dict[str, float]]:
    """Long-horizon flow-composition profile that observes Y but has no alpha term."""
    steps = int(round(P.ergodic_max_tau / P.dt_hjb))
    u = normalize_y(np.zeros((NQ, P.ny)))
    prev_ask: np.ndarray | None = None
    prev_bid: np.ndarray | None = None
    last_diff = np.inf
    converged = False

    for step in range(1, steps + 1):
        u = apply_y_exp(u, Y_E_HALF)
        u = flow_inventory_step_y(u)
        u = apply_y_exp(u, Y_E_HALF)
        u = normalize_y(u)

        if step % check_every == 0:
            ask, bid = depth_y_grid_from_u(u)
            if prev_ask is not None and prev_bid is not None:
                last_diff = float(
                    max(np.nanmax(np.abs(ask - prev_ask)), np.nanmax(np.abs(bid - prev_bid)))
                )
                if last_diff < P.ergodic_tol:
                    converged = True
                    break
            prev_ask = ask
            prev_bid = bid

    info = {
        "tau": float(step * P.dt_hjb),
        "steps": float(step),
        "last_depth_diff": float(last_diff),
        "converged": float(converged),
        "tolerance": float(P.ergodic_tol),
    }
    return u, info


def interp_y(values: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.interp(np.clip(y, Y[0], Y[-1]), Y, values)


def regime_aware_depths(
    u_tau: np.ndarray,
    q: np.ndarray,
    y: np.ndarray,
    regime_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ask = np.full_like(y, np.inf, dtype=float)
    bid = np.full_like(y, np.inf, dtype=float)
    for qval in np.unique(q):
        q_mask = q == qval
        i = int(qval + P.qmax)
        for r_idx in np.unique(regime_idx[q_mask]):
            mask = q_mask & (regime_idx == r_idx)
            ui = interp_y(u_tau[i, r_idx], y[mask])
            if i > 0:
                ask[mask] = P.delta0 - interp_y(u_tau[i - 1, r_idx], y[mask]) + ui
            if i < NQ - 1:
                bid[mask] = P.delta0 - interp_y(u_tau[i + 1, r_idx], y[mask]) + ui
    return ask, bid


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


def local_ansatz_coefficients() -> dict[str, float]:
    p = P.p_glft
    inventory_rate = P.rho
    flow_loading = (inventory_rate / P.k) / (P.beta + inventory_rate)
    alpha_loading = P.alpha_slope / (P.beta + inventory_rate)
    linear_loading = flow_loading + alpha_loading
    regime_loading = P.beta * linear_loading / (P.regime_rate + inventory_rate)
    return {
        "p": float(p),
        "inventory_rate": float(inventory_rate),
        "regime_rate": float(P.regime_rate),
        "flow_loading": float(flow_loading),
        "alpha_loading": float(alpha_loading),
        "linear_loading": float(linear_loading),
        "regime_loading": float(regime_loading),
        "alpha_slope": float(P.alpha_slope),
        "signal_shift": "flow_loading*y + alpha_loading*tanh(y) + regime_loading*z_r",
    }


LOCAL_ANSATZ = local_ansatz_coefficients()


def local_ansatz_shift(y: np.ndarray, regime_idx: np.ndarray) -> np.ndarray:
    return (
        LOCAL_ANSATZ["flow_loading"] * y
        + LOCAL_ANSATZ["alpha_loading"] * np.tanh(y)
        + LOCAL_ANSATZ["regime_loading"] * Z_LEVELS[regime_idx]
    )


def local_ansatz_depths(
    q: np.ndarray,
    y: np.ndarray,
    regime_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q_float = q.astype(float)
    p = LOCAL_ANSATZ["p"]
    shift = local_ansatz_shift(y, regime_idx)
    ask = P.delta0 - (p / P.k) * (2.0 * q_float - 1.0) + shift
    bid = P.delta0 + (p / P.k) * (2.0 * q_float + 1.0) - shift
    ask = np.where(q > -P.qmax, ask, np.inf)
    bid = np.where(q < P.qmax, bid, np.inf)
    return ask, bid


def local_ansatz_depth_grid() -> tuple[np.ndarray, np.ndarray]:
    q_grid, r_grid, y_grid = np.meshgrid(Q, np.arange(NR), Y, indexing="ij")
    return local_ansatz_depths(q_grid, y_grid, r_grid)


# ---------------------------------------------------------------------------
# Closed-form regime quotes: self-consistent cubic (body) and Galerkin (appendix).
# Both add the third directional channel A_R z_r to the fast-alpha machinery.
# ---------------------------------------------------------------------------
_A2 = np.sqrt(P.beta) / P.eta                                   # toy flow scale = sqrt(beta)/eta
_A1 = ((P.rho / P.k) / (P.rho + 2.0 * P.beta)) / _A2            # toy flow amplitude (saturating)
A_ALPHA_LOAD = LOCAL_ANSATZ["alpha_loading"]                    # A_alpha
A_REGIME_LOAD = LOCAL_ANSATZ["regime_loading"]                 # A_R = beta*linear/(lambda_R+2kappa)
A_GAL, C_GAL = 0.881800, 4.666300                              # fast-alpha Galerkin pair (q^1 -> carries over)


def regime_sc_cubic_depths(q, y, regime_idx, *, gate_regime: bool = False):
    """Regime self-consistent cubic (body closed form).  Flow-gated skew for the
    flow (h_I) and fast-alpha (h_alpha) channels, an UNGATED slow regime channel
    A_R z_r, curvature from the flow-only shift, and the cubic evaluated at the
    self-consistent full quoted shift.  ``gate_regime=True`` exposes the
    historical gated-regime ablation used for the paper's RMSE comparison;
    the published policy and the default remain ungated.  Elementary, with no
    fitted or tuning constants."""
    q = np.asarray(q, dtype=float)
    y = np.asarray(y, dtype=float)
    z = Z_LEVELS[np.asarray(regime_idx, dtype=int)]
    hI = _A1 * np.tanh(_A2 * y)
    theta_I = y - P.k * hI
    r = 2.0 * P.p_glft * np.sqrt(np.cosh(y) / np.cosh(theta_I))
    M = r / (2.0 * P.p_glft)
    ha = A_ALPHA_LOAD * np.tanh(y)
    hR = A_REGIME_LOAD * z
    g = np.tanh(_A2 * y) ** 2
    gate = 1.0 + g * (M - 1.0)
    shift = M * hI + gate * ha + (gate * hR if gate_regime else hR)
    theta_full = y - P.k * shift
    s = (r ** 2 / 6.0) * np.tanh(theta_full)
    ask = P.delta0 + shift - (r / (2.0 * P.k)) * (2.0 * q - 1.0) + (s / P.k) * (q * q - q + 1.0 / 3.0)
    bid = P.delta0 - shift + (r / (2.0 * P.k)) * (2.0 * q + 1.0) - (s / P.k) * (q * q + q + 1.0 / 3.0)
    ask = np.where(q > -P.qmax, ask, np.inf)
    bid = np.where(q < P.qmax, bid, np.inf)
    return ask, bid


def regime_galerkin_depths(q, y, regime_idx):
    """Regime Galerkin-cubic (appendix companion).  Single global saturating loading
    (offline (A,c) from the fast-alpha moment projection) plus the regime channel
    A_R z_r; curvature/cubic on the Galerkin flow shift."""
    q = np.asarray(q, dtype=float)
    yc = np.clip(np.asarray(y, dtype=float), Y[0], Y[-1])
    z = Z_LEVELS[np.asarray(regime_idx, dtype=int)]
    hg = A_GAL * yc / np.sqrt(1.0 + P.beta * yc ** 2 / (C_GAL * P.eta ** 2))
    theta = yc - P.k * hg
    r = 2.0 * P.p_glft * np.sqrt(np.cosh(yc) / np.cosh(theta))
    s = (r ** 2 / 6.0) * np.tanh(theta)
    shift = hg + A_REGIME_LOAD * z
    ask = P.delta0 + shift - (r / (2.0 * P.k)) * (2.0 * q - 1.0) + (s / P.k) * (q * q - q + 1.0 / 3.0)
    bid = P.delta0 - shift + (r / (2.0 * P.k)) * (2.0 * q + 1.0) - (s / P.k) * (q * q + q + 1.0 / 3.0)
    ask = np.where(q > -P.qmax, ask, np.inf)
    bid = np.where(q < P.qmax, bid, np.inf)
    return ask, bid


def as_naive_depths(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = P.p_glft
    q_float = q.astype(float)
    ask = P.delta0 - (p / P.k) * (2.0 * q_float - 1.0)
    bid = P.delta0 + (p / P.k) * (2.0 * q_float + 1.0)
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


def initialize_regime_and_y(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    regime_idx = rng.choice(np.arange(NR), size=P.n_paths, p=np.array([0.25, 0.5, 0.25]))
    y_std = P.eta / np.sqrt(2.0 * P.beta)
    y = Z_LEVELS[regime_idx] + y_std * rng.standard_normal(P.n_paths)
    return regime_idx.astype(int), y


def step_regime(regime_idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    rate = P.regime_rate
    event = rng.random(P.n_paths) < (1.0 - np.exp(-rate * P.dt_sim))
    next_regime = regime_idx.copy()

    neg = event & (regime_idx == 0)
    neu = event & (regime_idx == 1)
    pos = event & (regime_idx == 2)
    next_regime[neg] = 1
    next_regime[pos] = 1
    if np.any(neu):
        next_regime[neu] = np.where(rng.random(np.sum(neu)) < 0.5, 0, 2)
    return next_regime


def simulate_policies(
    ergodic_u: np.ndarray,
    y_flow_u: np.ndarray,
    alpha_y_u: np.ndarray,
) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    rng = np.random.default_rng(P.seed)
    steps = int(round(P.T / P.dt_sim))
    times = np.linspace(0.0, P.T, steps + 1)
    sqrt_dt = np.sqrt(P.dt_sim)
    policies = ["GLFT", "Y-aware", "Alpha/Y-aware", "Regime-aware", "Regime Galerkin", "Regime sc-cubic"]

    regime_idx, y = initialize_regime_and_y(rng)
    alpha = np.zeros(P.n_paths)
    price = np.zeros(P.n_paths)
    q = {name: np.zeros(P.n_paths, dtype=int) for name in policies}
    spread_cash = {name: np.zeros(P.n_paths) for name in policies}
    alpha_pnl = {name: np.zeros(P.n_paths) for name in policies}
    price_martingale_pnl = {name: np.zeros(P.n_paths) for name in policies}
    q_y_integral = {name: np.zeros(P.n_paths) for name in policies}
    q_z_integral = {name: np.zeros(P.n_paths) for name in policies}
    q_alpha_integral = {name: np.zeros(P.n_paths) for name in policies}
    fills = {name: np.zeros(P.n_paths) for name in policies}
    running_penalty = {name: np.zeros(P.n_paths) for name in policies}
    abs_q_integral = {name: np.zeros(P.n_paths) for name in policies}
    min_ask = {name: np.full(P.n_paths, np.inf) for name in policies}
    min_bid = {name: np.full(P.n_paths, np.inf) for name in policies}
    regime_time = np.zeros(NR)
    y_clip_count = 0
    y_clip_total = 0

    record_every = max(1, int(round(P.record_dt / P.dt_sim)))
    rec_times = []
    rec_total = {name: [] for name in policies}
    rec_decomposed = {name: [] for name in policies}
    rec_pen = {name: [] for name in policies}
    rec_absq = {name: [] for name in policies}
    lifetime_q = {name: [] for name in policies}
    signal_lifetime_q = {name: [] for name in policies}
    y_samples = []
    alpha_samples = []
    regime_samples = []
    sample_price = []
    sample_y = []
    sample_alpha = []
    sample_regime = []
    sample_inventory = {name: [] for name in policies}

    for step, t in enumerate(times[:-1]):
        do_record = step % record_every == 0
        z_current = Z_LEVELS[regime_idx]
        y_clip_count += int(np.sum((y < Y[0]) | (y > Y[-1])))
        y_clip_total += P.n_paths
        lambda_plus0 = P.bar_lambda * (1.0 + np.tanh(y))
        lambda_minus0 = P.bar_lambda * (1.0 - np.tanh(y))
        alpha_before = alpha.copy()
        price_noise = P.sigma_price * sqrt_dt * rng.standard_normal(P.n_paths)
        regime_time += np.bincount(regime_idx, minlength=NR) * P.dt_sim

        for name in policies:
            q_before = q[name].copy()
            if name == "Regime-aware":
                ask, bid = regime_aware_depths(ergodic_u, q_before, y, regime_idx)
            elif name == "Regime Galerkin":
                ask, bid = regime_galerkin_depths(q_before, y, regime_idx)
            elif name == "Regime sc-cubic":
                ask, bid = regime_sc_cubic_depths(q_before, y, regime_idx)
            elif name == "Y-aware":
                ask, bid = y_aware_depths(y_flow_u, q_before, y)
            elif name == "Alpha/Y-aware":
                ask, bid = y_aware_depths(alpha_y_u, q_before, y)
            elif name == "GLFT":
                ask, bid = as_naive_depths(q_before)
            else:
                raise ValueError(f"Unknown policy: {name}")

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
            q_z_integral[name] += q_before.astype(float) * z_current * P.dt_sim
            q_alpha_integral[name] += q_before.astype(float) * alpha_before * P.dt_sim
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
            regime_samples.append(regime_idx.copy())
            sample_price.append(float(price[0]))
            sample_y.append(float(y[0]))
            sample_alpha.append(float(alpha_before[0]))
            sample_regime.append(int(regime_idx[0]))
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
        y += -P.beta * (y - z_current) * P.dt_sim + P.eta * sqrt_dt * rng.standard_normal(P.n_paths)
        regime_idx = step_regime(regime_idx, rng)

    metrics: dict[str, dict[str, float]] = {}
    y_clip_share = float(y_clip_count / max(1, y_clip_total))
    terminal_total = {}
    terminal_decomposed = {}
    pen_paths = {}
    for name in policies:
        decomposed = spread_cash[name] + alpha_pnl[name]
        total = decomposed + price_martingale_pnl[name]
        pen = total - running_penalty[name] - P.alpha_l * q[name].astype(float) ** 2
        pen_paths[name] = pen.copy()
        lifetime = np.concatenate(lifetime_q[name]) if lifetime_q[name] else q[name]
        terminal_total[name] = total.copy()
        terminal_decomposed[name] = decomposed.copy()
        metrics[name] = {
            "total_pnl_mean": float(np.mean(total)),
            "total_pnl_std": float(np.std(total)),
            "total_pnl_stderr": standard_error(total),
            "decomposed_pnl_mean": float(np.mean(decomposed)),
            "decomposed_pnl_std": float(np.std(decomposed)),
            "decomposed_pnl_stderr": standard_error(decomposed),
            "price_martingale_pnl_mean": float(np.mean(price_martingale_pnl[name])),
            "price_martingale_pnl_std": float(np.std(price_martingale_pnl[name])),
            "price_martingale_pnl_stderr": standard_error(price_martingale_pnl[name]),
            "penalized_pnl_mean": float(np.mean(pen)),
            "penalized_pnl_std": float(np.std(pen)),
            "penalized_pnl_stderr": standard_error(pen),
            "spread_capture_mean": float(np.mean(spread_cash[name])),
            "spread_capture_stderr": standard_error(spread_cash[name]),
            "alpha_drift_pnl_mean": float(np.mean(alpha_pnl[name])),
            "alpha_drift_pnl_stderr": standard_error(alpha_pnl[name]),
            "avg_q_alpha": float(np.mean(q_alpha_integral[name] / P.T)),
            "avg_q_y": float(np.mean(q_y_integral[name] / P.T)),
            "avg_q_z": float(np.mean(q_z_integral[name] / P.T)),
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
    regime_lifetime = np.concatenate(regime_samples) if regime_samples else regime_idx
    lifetime_inventory = {
        name: np.concatenate(lifetime_q[name]) if lifetime_q[name] else q[name]
        for name in policies
    }
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
        "terminal_total": terminal_total,
        "pen_paths": pen_paths,
        "terminal_decomposed": terminal_decomposed,
        "lifetime_inventory": lifetime_inventory,
        "regime_time_fraction": regime_time / np.sum(regime_time),
        "y_clip_share": y_clip_share,
        "sample": {
            "times": np.array(rec_times),
            "price": np.array(sample_price),
            "y": np.array(sample_y),
            "alpha": np.array(sample_alpha),
            "regime": np.array(sample_regime),
            "inventory": {name: np.array(sample_inventory[name]) for name in policies},
        },
    }
    paths.update(make_conditional_inventory_by_y(y_lifetime, signal_inventory))
    paths.update(make_conditional_inventory_by_alpha(alpha_lifetime, signal_inventory))
    paths.update(make_conditional_inventory_by_regime(regime_lifetime, signal_inventory))
    paths.update(
        make_regime_signal_alignment_by_regime(
            regime_lifetime,
            y_lifetime,
            alpha_lifetime,
            signal_inventory,
        )
    )
    return metrics, paths


def make_conditional_inventory_by_y(
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


def make_conditional_inventory_by_regime(
    regime_state: np.ndarray,
    lifetime_inventory: dict[str, np.ndarray],
) -> dict[str, object]:
    means = {}
    counts = {}
    for name, inv in lifetime_inventory.items():
        vals = np.full(NR, np.nan)
        cnt = np.zeros(NR, dtype=int)
        for r_idx in range(NR):
            mask = regime_state == r_idx
            cnt[r_idx] = int(np.sum(mask))
            if cnt[r_idx] > 0:
                vals[r_idx] = float(np.mean(inv[mask]))
        means[name] = vals
        counts[name] = cnt
    return {
        "regime_labels": REGIME_LABELS.copy(),
        "inventory_by_regime": means,
        "inventory_by_regime_counts": counts,
    }


def make_regime_signal_alignment_by_regime(
    regime_state: np.ndarray,
    y_state: np.ndarray,
    alpha_state: np.ndarray,
    lifetime_inventory: dict[str, np.ndarray],
) -> dict[str, object]:
    qy = {}
    qalpha = {}
    absq = {}
    counts = {}
    for name, inv in lifetime_inventory.items():
        vals_qy = np.full(NR, np.nan)
        vals_qa = np.full(NR, np.nan)
        vals_absq = np.full(NR, np.nan)
        cnt = np.zeros(NR, dtype=int)
        inv_float = inv.astype(float)
        for r_idx in range(NR):
            mask = regime_state == r_idx
            cnt[r_idx] = int(np.sum(mask))
            if cnt[r_idx] > 0:
                vals_qy[r_idx] = float(np.mean(inv_float[mask] * y_state[mask]))
                vals_qa[r_idx] = float(np.mean(inv_float[mask] * alpha_state[mask]))
                vals_absq[r_idx] = float(np.mean(np.abs(inv_float[mask])))
        qy[name] = vals_qy
        qalpha[name] = vals_qa
        absq[name] = vals_absq
        counts[name] = cnt
    return {
        "regime_qy": qy,
        "regime_qalpha": qalpha,
        "regime_absq": absq,
        "regime_signal_counts": counts,
    }


def plot_quote_depths(regime_u: np.ndarray) -> None:
    yy = np.linspace(Y[0], Y[-1], 301)
    q0 = np.zeros_like(yy, dtype=int)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    colors = ["#4c72b0", "#55a868", "#c44e52"]
    for r_idx, color in enumerate(colors):
        r_vec = np.full_like(q0, r_idx)
        e_ask, e_bid = regime_aware_depths(regime_u, q0, yy, r_vec)
        a_ask, a_bid = local_ansatz_depths(q0, yy, r_vec)
        label = f"R={REGIME_LABELS[r_idx]}"
        ax[0].plot(yy, e_ask, color=color, lw=2.4, label=f"{label} ergodic")
        ax[0].plot(yy, a_ask, color=color, lw=1.9, ls="--", label=f"{label} ansatz")
        ax[1].plot(yy, e_bid, color=color, lw=2.4, label=f"{label} ergodic")
        ax[1].plot(yy, a_bid, color=color, lw=1.9, ls="--", label=f"{label} ansatz")

    ax[0].set_title("Regime HJB vs ansatz ask depth, q=0")
    ax[1].set_title("Regime HJB vs ansatz bid depth, q=0")
    for a in ax:
        a.axhline(0.0, color="black", lw=1, alpha=0.45)
        a.set_xlabel("OU state i")
        a.set_ylabel("Raw unconstrained depth")
        a.grid(True, alpha=0.25)
    ax[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "regime_fast_alpha_quote_depths_q0.png", dpi=180)
    plt.close(fig)

    lines = [
        "y,regime,regime_ergodic_ask,regime_ergodic_bid,ansatz_ask,ansatz_bid"
    ]
    for r_idx in range(NR):
        r_vec = np.full_like(q0, r_idx)
        e_ask, e_bid = regime_aware_depths(regime_u, q0, yy, r_vec)
        a_ask, a_bid = local_ansatz_depths(q0, yy, r_vec)
        for row in zip(yy, e_ask, e_bid, a_ask, a_bid):
            lines.append(
                f"{row[0]:.10g},{int(REGIME_LABELS[r_idx])},"
                + ",".join(f"{x:.10g}" for x in row[1:])
            )
    (OUT / "regime_fast_alpha_quote_depths_q0.csv").write_text("\n".join(lines) + "\n")


def plot_depth_heatmaps(ergodic_u: np.ndarray) -> None:
    ergodic_ask, ergodic_bid = depth_grid_from_u(ergodic_u)
    ansatz_ask, ansatz_bid = local_ansatz_depth_grid()
    panels = []
    titles = []
    for data, side, model in [
        (ergodic_ask, "ask", "ergodic"),
        (ansatz_ask, "ask", "ansatz"),
        (ergodic_bid, "bid", "ergodic"),
        (ansatz_bid, "bid", "ansatz"),
    ]:
        for r_idx in range(NR):
            panels.append(data[:, r_idx, :])
            titles.append(f"R={REGIME_LABELS[r_idx]} {model} {side}")

    abs_values = np.concatenate([np.abs(panel[np.isfinite(panel)]).ravel() for panel in panels])
    vmax = float(np.nanpercentile(abs_values, 98.0))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    fig, ax = plt.subplots(4, NR, figsize=(4.4 * NR, 11), sharex=True, sharey=True)
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
    fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.88)
    fig.savefig(OUT / "regime_fast_alpha_depth_heatmaps.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_surface_3d(
    values: np.ndarray,
    title: str,
    zlabel: str,
    filename: str,
    cmap: str = "viridis",
) -> None:
    q_window = min(P.surface_qmax, P.qmax)
    q_mask = np.abs(Q) <= q_window
    q_values = Q[q_mask]
    q_grid, y_grid = np.meshgrid(q_values, Y, indexing="ij")
    z = np.where(np.isfinite(values[q_mask]), values[q_mask], np.nan)
    finite = z[np.isfinite(z)]
    fig = plt.figure(figsize=(8.2, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    is_flat = finite.size and float(np.nanmax(finite) - np.nanmin(finite)) < 1e-10
    if is_flat:
        ax.plot_surface(
            q_grid,
            y_grid,
            z,
            color="#4c72b0",
            edgecolor="#2f5f8f",
            linewidth=0.25,
            antialiased=True,
            alpha=0.9,
            shade=False,
        )
        ax.text2D(0.03, 0.94, f"constant = {float(finite[0]):.4f}", transform=ax.transAxes)
    else:
        ax.plot_surface(q_grid, y_grid, z, cmap=cmap, linewidth=0, antialiased=True, alpha=0.92)
    ax.set_title(title)
    ax.set_xlabel("Inventory q")
    ax.set_ylabel("OU state i")
    ax.set_zlabel(zlabel)
    if finite.size:
        lo, hi = np.nanpercentile(finite, [1.0, 99.0])
        if is_flat:
            center = float(finite[0])
            pad = max(0.02, 0.03 * abs(center))
            ax.set_zlim(center - pad, center + pad)
        elif np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            pad = 0.08 * (hi - lo)
            ax.set_zlim(lo - pad, hi + pad)
    ax.view_init(elev=27, azim=-135)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def plot_regime_surface_triptych(
    values: np.ndarray,
    title_prefix: str,
    zlabel: str,
    filename: str,
    cmap: str = "viridis",
) -> None:
    q_window = min(P.surface_qmax, P.qmax)
    q_mask = np.abs(Q) <= q_window
    q_values = Q[q_mask]
    q_grid, y_grid = np.meshgrid(q_values, Y, indexing="ij")
    cropped_values = values[q_mask]
    finite = cropped_values[np.isfinite(cropped_values)]
    fig = plt.figure(figsize=(15.0, 5.2))
    zlim = None
    if finite.size:
        lo, hi = np.nanpercentile(finite, [1.0, 99.0])
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            pad = 0.08 * (hi - lo)
            zlim = (lo - pad, hi + pad)
    for r_idx, regime_label in enumerate(REGIME_LABELS):
        ax = fig.add_subplot(1, NR, r_idx + 1, projection="3d")
        z = np.where(
            np.isfinite(cropped_values[:, r_idx, :]),
            cropped_values[:, r_idx, :],
            np.nan,
        )
        ax.plot_surface(q_grid, y_grid, z, cmap=cmap, linewidth=0, antialiased=True, alpha=0.92)
        ax.set_title(f"{title_prefix}: R={regime_label}")
        ax.set_xlabel("Inventory q")
        ax.set_ylabel("OU state i")
        ax.set_zlabel(zlabel)
        if zlim is not None:
            ax.set_zlim(*zlim)
        ax.view_init(elev=27, azim=-135)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def plot_depth_surfaces(
    regime_u: np.ndarray,
    y_flow_u: np.ndarray,
    alpha_y_u: np.ndarray,
) -> None:
    glft_ask_vec, glft_bid_vec = as_naive_depths(Q)
    glft_ask = np.repeat(glft_ask_vec[:, None], P.ny, axis=1)
    glft_bid = np.repeat(glft_bid_vec[:, None], P.ny, axis=1)
    y_ask, y_bid = depth_y_grid_from_u(y_flow_u)
    alpha_ask, alpha_bid = depth_y_grid_from_u(alpha_y_u)
    regime_ask, regime_bid = depth_grid_from_u(regime_u)

    surface_specs = [
        ("glft", "GLFT", glft_ask, glft_bid, "Blues"),
        ("yaware", "I-aware RHJB", y_ask, y_bid, "Greens"),
        ("alpha_yaware", "Alpha/I-aware RHJB", alpha_ask, alpha_bid, "Reds"),
    ]
    for slug, label, ask, bid, cmap in surface_specs:
        plot_surface_3d(
            ask,
            f"{label} ask depth surface",
            "Ask depth",
            f"regime_fast_alpha_{slug}_ask_surface.png",
            cmap,
        )
        plot_surface_3d(
            bid,
            f"{label} bid depth surface",
            "Bid depth",
            f"regime_fast_alpha_{slug}_bid_surface.png",
            cmap,
        )
        plot_surface_3d(
            ask + bid,
            f"{label} spread surface",
            "Ask + bid depth",
            f"regime_fast_alpha_{slug}_spread_surface.png",
            cmap,
        )

    plot_regime_surface_triptych(
        regime_ask,
        "Regime-aware RHJB ask depth",
        "Ask depth",
        "regime_fast_alpha_regime_ask_surfaces.png",
        "Purples",
    )
    plot_regime_surface_triptych(
        regime_bid,
        "Regime-aware RHJB bid depth",
        "Bid depth",
        "regime_fast_alpha_regime_bid_surfaces.png",
        "Purples",
    )
    plot_regime_surface_triptych(
        regime_ask + regime_bid,
        "Regime-aware RHJB spread",
        "Ask + bid depth",
        "regime_fast_alpha_regime_spread_surfaces.png",
        "Purples",
    )


def plot_paths(paths: dict[str, object]) -> None:
    policies = list(paths["total"])
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for name in policies:
        ax[0].plot(paths["times"], paths["total"][name], lw=2, label=policy_display(name))
        ax[1].plot(paths["times"], paths["penalized"][name], lw=2, label=policy_display(name))
        ax[2].plot(paths["times"], paths["absq"][name], lw=2, label=policy_display(name))
    ax[0].set_title("Mean total profit and loss")
    ax[1].set_title("Mean penalized profit and loss")
    ax[2].set_title("Mean |Q|")
    for a in ax:
        a.set_xlabel("time")
        a.grid(True, alpha=0.25)
    ax[0].set_ylabel("Profit and loss")
    ax[2].set_ylabel("Inventory")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "regime_fast_alpha_simulation_paths.png", dpi=180)
    plt.close(fig)


def plot_conditional_inventory(paths: dict[str, object], key: str, x_key: str, xlabel: str, filename: str) -> None:
    policies = list(paths[key])
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for name in policies:
        ax.plot(paths[x_key], paths[key][name], marker="o", ms=3.2, lw=2, label=policy_display(name))
    ax.axhline(0.0, color="black", lw=1, alpha=0.5)
    ax.axvline(0.0, color="black", lw=1, alpha=0.25)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Conditional mean inventory")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def plot_inventory_by_regime(paths: dict[str, object]) -> None:
    policies = list(paths["inventory_by_regime"])
    x = np.arange(NR)
    width = min(0.18, 0.75 / max(1, len(policies)))
    offsets = (np.arange(len(policies)) - 0.5 * (len(policies) - 1)) * width
    fig, ax = plt.subplots(2, 1, figsize=(9.2, 7.2), sharex=True)
    palette = colors_for(policies)

    panel_specs = [
        (
            ax[0],
            paths["inventory_by_regime"],
            r"Signed inventory: $\mathbb{E}[Q\mid Z]$",
            "Mean inventory",
            "{:+.2f}",
        ),
        (
            ax[1],
            paths["regime_absq"],
            r"Inventory activity: $\mathbb{E}[|Q|\mid Z]$",
            "Mean absolute inventory",
            "{:.2f}",
        ),
    ]

    for panel, value_by_policy, title, ylabel, fmt in panel_specs:
        for j, name in enumerate(policies):
            vals = np.asarray(value_by_policy[name], dtype=float)
            bars = panel.bar(x + offsets[j], vals, width=width, label=policy_display(name), color=palette[j])
            span = np.nanmax(np.abs(vals)) if np.any(np.isfinite(vals)) else 1.0
            pad = 0.08 * max(1.0, span)
            for bar, val in zip(bars, vals):
                if not np.isfinite(val):
                    continue
                y_text = val + pad if val >= 0 else val - pad
                va = "bottom" if val >= 0 else "top"
                panel.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_text,
                    fmt.format(val),
                    ha="center",
                    va=va,
                    fontsize=7,
                    rotation=90,
                )
        panel.axhline(0.0, color="black", lw=1, alpha=0.5)
        panel.set_ylabel(ylabel)
        panel.set_title(title)
        panel.grid(True, axis="y", alpha=0.25)

    ax[1].set_xticks(x)
    ax[1].set_xticklabels([str(x) for x in REGIME_LABELS])
    ax[1].set_xlabel("Regime R")
    ax[0].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "regime_fast_alpha_inventory_by_regime.png", dpi=180)
    plt.close(fig)


def plot_regime_signal_diagnostics(paths: dict[str, object]) -> None:
    policies = list(paths["regime_qy"])
    colors = colors_for(policies)
    x = np.arange(NR)
    width = min(0.18, 0.75 / max(1, len(policies)))
    offsets = (np.arange(len(policies)) - 0.5 * (len(policies) - 1)) * width
    panels = [
        ("regime_qy", r"$\mathbb{E}[QI\mid R]$"),
        ("regime_qalpha", r"$\mathbb{E}[Q\alpha\mid R]$"),
        ("regime_absq", r"$\mathbb{E}[|Q|\mid R]$"),
    ]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    for a, (key, title) in zip(ax, panels):
        for j, name in enumerate(policies):
            vals = np.asarray(paths[key][name], dtype=float)
            a.bar(x + offsets[j], vals, width=width, label=policy_display(name), color=colors[j])
        a.axhline(0.0, color="black", lw=1, alpha=0.55)
        a.set_xticks(x)
        a.set_xticklabels([str(x) for x in REGIME_LABELS])
        a.set_xlabel("Regime R")
        a.set_title(title)
        a.grid(True, axis="y", alpha=0.25)
    ax[0].set_ylabel("Conditional mean")
    ax[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "regime_fast_alpha_regime_signal_diagnostics.png", dpi=180)
    plt.close(fig)


def plot_regime_direction_alignment(paths: dict[str, object]) -> None:
    policies = list(paths["inventory_by_regime"])
    colors = colors_for(policies)
    x = np.arange(NR)
    width = min(0.18, 0.75 / max(1, len(policies)))
    offsets = (np.arange(len(policies)) - 0.5 * (len(policies) - 1)) * width

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8), sharex=True)
    for j, name in enumerate(policies):
        mean_q = np.asarray(paths["inventory_by_regime"][name], dtype=float)
        aligned = mean_q * Z_LEVELS
        ax[0].bar(x + offsets[j], mean_q, width=width, label=policy_display(name), color=colors[j])
        ax[1].bar(x + offsets[j], aligned, width=width, label=policy_display(name), color=colors[j])

    ax[0].set_title(r"Conditional inventory $\mathbb{E}[Q\mid R]$")
    ax[1].set_title(r"Regime-aligned exposure $\mathbb{E}[QZ_R\mid R]$")
    for a in ax:
        a.axhline(0.0, color="black", lw=1, alpha=0.55)
        a.set_xticks(x)
        a.set_xticklabels([str(x) for x in REGIME_LABELS])
        a.set_xlabel("Regime R")
        a.grid(True, axis="y", alpha=0.25)
    ax[0].set_ylabel("Inventory")
    ax[1].set_ylabel("Positive means aligned with regime")
    ax[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "regime_fast_alpha_inventory_direction_alignment.png", dpi=180)
    plt.close(fig)


def plot_lifetime_inventory(paths: dict[str, object]) -> None:
    policies = list(paths["lifetime_inventory"])
    bins = np.arange(-P.qmax - 0.5, P.qmax + 1.5, 1.0)
    fig, ax = plt.subplots(1, len(policies), figsize=(4.8 * len(policies), 4), sharex=True, sharey=True)
    ax = np.atleast_1d(ax)
    colors = colors_for(policies)
    for j, name in enumerate(policies):
        ax[j].hist(
            paths["lifetime_inventory"][name],
            bins=bins,
            density=True,
            color=colors[j],
            edgecolor="white",
            linewidth=0.5,
        )
        ax[j].axvline(0.0, color="black", lw=1, alpha=0.45)
        ax[j].set_title(policy_display(name))
        ax[j].set_xlabel("Inventory q")
        ax[j].grid(True, axis="y", alpha=0.25)
    ax[0].set_ylabel("Lifetime frequency")
    fig.tight_layout()
    fig.savefig(OUT / "regime_fast_alpha_lifetime_inventory_hist.png", dpi=180)
    plt.close(fig)


def plot_terminal_pnl_hist(paths: dict[str, object]) -> None:
    policies = list(paths["terminal_total"])
    colors = colors_for(policies)
    all_values = np.concatenate([np.asarray(paths["terminal_total"][name]) for name in policies])
    lo, hi = np.nanpercentile(all_values, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.nanmin(all_values)), float(np.nanmax(all_values))
    bins = np.linspace(lo, hi, 55)
    fig, ax = plt.subplots(1, 2, figsize=(13.2, 4.6))
    for j, name in enumerate(policies):
        ax[0].hist(
            paths["terminal_total"][name],
            bins=bins,
            density=True,
            histtype="step",
            lw=2.0,
            color=colors[j],
            label=policy_display(name),
        )
        ax[1].hist(
            paths["terminal_decomposed"][name],
            bins=bins,
            density=True,
            histtype="step",
            lw=2.0,
            color=colors[j],
            label=policy_display(name),
        )
    ax[0].set_title("Terminal PnL including price martingale")
    ax[1].set_title("Terminal spread + alpha-drift PnL")
    for a in ax:
        a.axvline(0.0, color="black", lw=1, alpha=0.45)
        a.set_xlabel("Terminal PnL")
        a.set_ylabel("Density")
        a.grid(True, alpha=0.25)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "regime_fast_alpha_terminal_pnl_hist.png", dpi=180)
    plt.close(fig)


def plot_terminal_pnl_kde(paths: dict[str, object]) -> None:
    """Kernel-density estimate of the terminal PnL distribution per policy."""
    policies = list(paths["terminal_total"])
    colors = colors_for(policies)
    all_values = np.concatenate([np.asarray(paths["terminal_total"][name]) for name in policies])
    lo, hi = np.nanpercentile(all_values, [0.5, 99.5])
    pad = 0.08 * (hi - lo)
    xg = np.linspace(lo - pad, hi + pad, 500)
    fig, ax = plt.subplots(1, 2, figsize=(13.2, 4.6))
    for j, name in enumerate(policies):
        for col, key in ((0, "terminal_total"), (1, "terminal_decomposed")):
            v = np.asarray(paths[key][name], dtype=float)
            v = v[np.isfinite(v)]
            ax[col].plot(xg, gaussian_kde(v)(xg), lw=2.3, color=colors[j], label=policy_display(name))
    ax[0].set_title("Terminal PnL density (incl. price martingale)")
    ax[1].set_title("Spread + alpha-drift PnL density")
    for a in ax:
        a.axvline(0.0, color="black", lw=1, alpha=0.45)
        a.set_xlabel("Terminal PnL")
        a.set_ylabel("Density")
        a.grid(True, alpha=0.25)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "regime_fast_alpha_terminal_pnl_kde.png", dpi=180)
    plt.close(fig)


def plot_signal_alignment(metrics: dict[str, dict[str, float]]) -> None:
    policies = list(metrics)
    colors = colors_for(policies)
    fields = [
        ("avg_q_y", r"Time average $\mathbb{E}[Q_tI_t]$"),
        ("avg_q_z", r"Time average $\mathbb{E}[Q_tZ_t]$"),
        ("avg_q_alpha", r"Time average $\mathbb{E}[Q_t\alpha_t]$"),
        ("alpha_drift_pnl_mean", r"$\mathbb{E}\int_0^T Q_t\alpha_t\,dt$"),
    ]
    fig, ax = plt.subplots(1, 4, figsize=(15.5, 4))
    for a, (field, title) in zip(ax, fields):
        vals = [metrics[name][field] for name in policies]
        a.bar([policy_tick(name) for name in policies], vals, color=colors)
        a.axhline(0.0, color="black", lw=1, alpha=0.55)
        a.set_title(title)
        # anchor the rotated multi-line policy names at their right edge, else the
        # long canonical names ("Regime self-consistent cubic") overprint each other
        plt.setp(a.get_xticklabels(), rotation=40, ha="right",
                 rotation_mode="anchor", fontsize=7.5)
        a.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "regime_fast_alpha_signal_alignment.png", dpi=180)
    plt.close(fig)


def plot_pnl_decomposition(metrics: dict[str, dict[str, float]]) -> None:
    policies = list(metrics)
    components = [
        ("spread_capture_mean", "Spread capture", "#4daf4a"),
        ("alpha_drift_pnl_mean", r"$\int Q_t\alpha_tdt$", "#377eb8"),
        ("running_penalty_mean", r"$-\phi\int Q_t^2dt$", "#984ea3"),
        ("terminal_penalty_mean", r"$-c_T Q_T^2$", "#e41a1c"),
    ]
    x = np.arange(len(policies))
    width = 0.18
    offsets = (np.arange(len(components)) - 1.5) * width
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for j, (key, label, color) in enumerate(components):
        vals = [metrics[name][key] for name in policies]
        if "penalty" in key:
            vals = [-v for v in vals]
        ax.bar(x + offsets[j], vals, width=width, label=label, color=color)
    ax.axhline(0.0, color="black", lw=1, alpha=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels([policy_tick(name) for name in policies], rotation=10, ha="right")
    ax.set_ylabel("Mean contribution")
    ax.set_title("Profit-and-loss decomposition")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "regime_fast_alpha_pnl_decomposition.png", dpi=180)
    plt.close(fig)


def plot_sample_path(paths: dict[str, object]) -> None:
    sample = paths["sample"]
    policies = list(sample["inventory"])
    fig, ax = plt.subplots(4, 1, figsize=(12.5, 9.5), sharex=True)

    ax[0].plot(sample["times"], sample["price"], color="#4c72b0", lw=1.8)
    ax[0].set_title("Sample reference price")
    ax[0].set_ylabel("S")
    ax[0].grid(True, alpha=0.25)

    ax[1].step(sample["times"], Z_LEVELS[sample["regime"]], where="post", color="#8172b2", lw=1.6, label="Z")
    ax[1].plot(sample["times"], sample["y"], color="#55a868", lw=1.2, alpha=0.9, label="I")
    ax[1].axhline(0.0, color="black", lw=0.8, alpha=0.35)
    ax[1].set_title("Slow regime and OU imbalance")
    ax[1].set_ylabel("state")
    ax[1].grid(True, alpha=0.25)
    ax[1].legend(loc="upper left")

    ax[2].plot(sample["times"], sample["alpha"], color="#c44e52", lw=1.35)
    ax[2].axhline(0.0, color="black", lw=0.8, alpha=0.35)
    ax[2].set_title("Generated alpha")
    ax[2].set_ylabel(r"$\alpha$")
    ax[2].grid(True, alpha=0.25)

    for name in policies:
        ax[3].plot(sample["times"], sample["inventory"][name], lw=1.45, label=policy_display(name))
    ax[3].axhline(0.0, color="black", lw=0.8, alpha=0.35)
    ax[3].set_title("Inventory on the same sample path")
    ax[3].set_xlabel("time")
    ax[3].set_ylabel("q")
    ax[3].grid(True, alpha=0.25)
    ax[3].legend(ncol=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "regime_fast_alpha_sample_path.png", dpi=180)
    plt.close(fig)


def plot_summary(metrics: dict[str, dict[str, float]]) -> None:
    policies = list(metrics)
    fields = [
        ("total_pnl_mean", "Total profit and loss"),
        ("penalized_pnl_mean", "Penalized profit and loss"),
        ("spread_capture_mean", "Spread capture"),
        ("alpha_drift_pnl_mean", "Alpha drift profit and loss"),
        ("fills_mean", "Fills"),
        ("avg_abs_inventory", "Average |Q|"),
    ]
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.5))
    colors = colors_for(policies)
    for a, (field, title) in zip(ax.ravel(), fields):
        vals = [metrics[name][field] for name in policies]
        a.bar([policy_tick(name) for name in policies], vals, color=colors)
        a.set_title(title)
        # anchor the rotated multi-line policy names at their right edge, else the
        # long canonical names ("Regime self-consistent cubic") overprint each other
        plt.setp(a.get_xticklabels(), rotation=40, ha="right",
                 rotation_mode="anchor", fontsize=7.5)
        a.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "regime_fast_alpha_summary_bars.png", dpi=180)
    plt.close(fig)


def save_terminal_pnl_replot_data(paths: dict[str, object]) -> Path:
    """Persist the path-level samples needed by the two terminal-PnL plots.

    ``policy_names`` and ``policy_keys`` have matching order.  The latter are
    ASCII-only, collision-free archive-key suffixes; the samples for entry
    ``j`` live under ``terminal_total__{policy_keys[j]}`` and
    ``terminal_decomposed__{policy_keys[j]}``.
    """
    terminal_total = paths["terminal_total"]
    terminal_decomposed = paths["terminal_decomposed"]
    policies = list(terminal_total)
    if set(policies) != set(terminal_decomposed):
        raise ValueError("terminal total/decomposed policy sets do not match")

    policy_keys: list[str] = []
    archive: dict[str, np.ndarray] = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "policy_names": np.asarray(policies, dtype=np.str_),
    }
    for index, name in enumerate(policies):
        slug = "_".join(
            part
            for part in "".join(
                char if ("a" <= char <= "z" or "0" <= char <= "9") else "_"
                for char in name.lower()
            ).split("_")
            if part
        )
        policy_key = f"p{index:02d}_{slug or 'unnamed'}"
        policy_keys.append(policy_key)
        archive[f"terminal_total__{policy_key}"] = np.asarray(
            terminal_total[name], dtype=float
        )
        archive[f"terminal_decomposed__{policy_key}"] = np.asarray(
            terminal_decomposed[name], dtype=float
        )
    archive["policy_keys"] = np.asarray(policy_keys, dtype=np.str_)

    destination = OUT / "regime_fast_alpha_replot_data.npz"
    np.savez_compressed(destination, **archive)
    return destination


def load_terminal_pnl_replot_data(
    cache_path: Path | str | None = None,
) -> dict[str, object]:
    """Load the compact terminal-PnL cache into the plotting API's format."""
    source = (
        OUT / "regime_fast_alpha_replot_data.npz"
        if cache_path is None
        else Path(cache_path)
    )
    with np.load(source, allow_pickle=False) as cache:
        version = int(np.asarray(cache["schema_version"]).item())
        if version != 1:
            raise ValueError(f"unsupported regime replot-cache schema {version}")
        policy_names = [str(name) for name in cache["policy_names"]]
        policy_keys = [str(key) for key in cache["policy_keys"]]
        if (
            not policy_names
            or len(policy_names) != len(policy_keys)
            or len(set(policy_names)) != len(policy_names)
            or len(set(policy_keys)) != len(policy_keys)
        ):
            raise ValueError("invalid policy metadata in regime replot cache")
        terminal_total = {}
        terminal_decomposed = {}
        for name, key in zip(policy_names, policy_keys):
            terminal_total[name] = np.asarray(
                cache[f"terminal_total__{key}"], dtype=float
            ).copy()
            terminal_decomposed[name] = np.asarray(
                cache[f"terminal_decomposed__{key}"], dtype=float
            ).copy()
        return {
            "terminal_total": terminal_total,
            "terminal_decomposed": terminal_decomposed,
        }


def replot_cached_terminal_pnl_figures(
    cache_path: Path | str | None = None,
) -> None:
    """Regenerate both terminal-PnL figures without rerunning simulation."""
    cached_paths = load_terminal_pnl_replot_data(cache_path)
    plot_terminal_pnl_hist(cached_paths)
    plot_terminal_pnl_kde(cached_paths)


def write_outputs(
    metrics: dict[str, dict[str, float]],
    paths: dict[str, object],
    regime_info: dict[str, float],
    y_flow_info: dict[str, float],
    alpha_y_info: dict[str, float],
) -> None:
    payload = {
        "params": asdict(P),
        "regimes": {
            "labels": REGIME_LABELS.tolist(),
            "z_levels": Z_LEVELS.tolist(),
            "generator": Q_REGIME.tolist(),
        },
        "local_ansatz": LOCAL_ANSATZ,
        "regime_ergodic_solver": regime_info,
        "y_flow_ergodic_solver": y_flow_info,
        "alpha_y_ergodic_solver": alpha_y_info,
        "metrics": metrics,
    }
    (OUT / "regime_fast_alpha_summary.json").write_text(json.dumps(payload, indent=2))
    fields = list(next(iter(metrics.values())).keys())
    lines = ["policy," + ",".join(fields)]
    for name, vals in metrics.items():
        lines.append(name + "," + ",".join(f"{vals[f]:.8g}" for f in fields))
    (OUT / "regime_fast_alpha_summary.csv").write_text("\n".join(lines) + "\n")
    save_terminal_pnl_replot_data(paths)

    for state_name, x_key, y_key, count_key in [
        ("y", "y_centers", "inventory_by_y", "inventory_by_y_counts"),
        ("alpha", "alpha_centers", "inventory_by_alpha", "inventory_by_alpha_counts"),
    ]:
        policies = list(paths[y_key])
        lines = [
            f"{state_name}_center,"
            + ",".join(f"{name}_mean_q" for name in policies)
            + ","
            + ",".join(f"{name}_count" for name in policies)
        ]
        for j, center in enumerate(paths[x_key]):
            means = [paths[y_key][name][j] for name in policies]
            counts = [paths[count_key][name][j] for name in policies]
            lines.append(
                f"{center:.8g},"
                + ",".join("" if np.isnan(x) else f"{x:.8g}" for x in means)
                + ","
                + ",".join(str(int(x)) for x in counts)
            )
        (OUT / f"regime_fast_alpha_expected_inventory_vs_{state_name}.csv").write_text(
            "\n".join(lines) + "\n"
        )

    policies = list(paths["inventory_by_regime"])
    lines = [
        "regime,z_level,"
        + ",".join(f"{name}_mean_q" for name in policies)
        + ","
        + ",".join(f"{name}_count" for name in policies)
    ]
    for r_idx, label in enumerate(REGIME_LABELS):
        means = [paths["inventory_by_regime"][name][r_idx] for name in policies]
        counts = [paths["inventory_by_regime_counts"][name][r_idx] for name in policies]
        lines.append(
            f"{int(label)},{Z_LEVELS[r_idx]:.8g},"
            + ",".join("" if np.isnan(x) else f"{x:.8g}" for x in means)
            + ","
            + ",".join(str(int(x)) for x in counts)
        )
    (OUT / "regime_fast_alpha_inventory_by_regime.csv").write_text("\n".join(lines) + "\n")

    lines = [
        "regime,z_level,"
        + ",".join(f"{name}_qy" for name in policies)
        + ","
        + ",".join(f"{name}_qalpha" for name in policies)
        + ","
        + ",".join(f"{name}_absq" for name in policies)
        + ","
        + ",".join(f"{name}_count" for name in policies)
    ]
    for r_idx, label in enumerate(REGIME_LABELS):
        qy_vals = [paths["regime_qy"][name][r_idx] for name in policies]
        qa_vals = [paths["regime_qalpha"][name][r_idx] for name in policies]
        absq_vals = [paths["regime_absq"][name][r_idx] for name in policies]
        counts = [paths["regime_signal_counts"][name][r_idx] for name in policies]
        lines.append(
            f"{int(label)},{Z_LEVELS[r_idx]:.8g},"
            + ",".join("" if np.isnan(x) else f"{x:.8g}" for x in qy_vals)
            + ","
            + ",".join("" if np.isnan(x) else f"{x:.8g}" for x in qa_vals)
            + ","
            + ",".join("" if np.isnan(x) else f"{x:.8g}" for x in absq_vals)
            + ","
            + ",".join(str(int(x)) for x in counts)
        )
    (OUT / "regime_fast_alpha_regime_signal_diagnostics.csv").write_text(
        "\n".join(lines) + "\n"
    )

    bins = np.arange(-P.qmax - 0.5, P.qmax + 1.5, 1.0)
    q_values = np.arange(-P.qmax, P.qmax + 1)
    policies = list(paths["lifetime_inventory"])
    counts = {
        name: np.histogram(paths["lifetime_inventory"][name], bins=bins)[0]
        for name in policies
    }
    totals = {name: max(1, int(np.sum(counts[name]))) for name in policies}
    lines = ["q," + ",".join(f"{name}_freq" for name in policies)]
    for j, q_value in enumerate(q_values):
        lines.append(
            str(int(q_value))
            + ","
            + ",".join(f"{counts[name][j] / totals[name]:.10g}" for name in policies)
        )
    (OUT / "regime_fast_alpha_lifetime_inventory_hist.csv").write_text(
        "\n".join(lines) + "\n"
    )

    sample = paths["sample"]
    sample_policies = list(sample["inventory"])
    clean = {name: name.lower().replace(" ", "_").replace("-", "_") for name in sample_policies}
    lines = [
        "time,price,y,alpha,regime,z,"
        + ",".join(f"{clean[name]}_inventory" for name in sample_policies)
    ]
    for j, t in enumerate(sample["times"]):
        regime = int(sample["regime"][j])
        row = [
            f"{t:.8g}",
            f"{sample['price'][j]:.8g}",
            f"{sample['y'][j]:.8g}",
            f"{sample['alpha'][j]:.8g}",
            str(int(REGIME_LABELS[regime])),
            f"{Z_LEVELS[regime]:.8g}",
        ]
        row.extend(str(int(sample["inventory"][name][j])) for name in sample_policies)
        lines.append(",".join(row))
    (OUT / "regime_fast_alpha_sample_path.csv").write_text("\n".join(lines) + "\n")


def main() -> None:
    print("Regime fast-alpha split local ansatz coefficients:")
    print(json.dumps(LOCAL_ANSATZ, indent=2))
    print("Solving regime-aware fast-alpha ergodic Cole-Hopf profile...")
    regime_u, regime_info = solve_regime_ergodic()
    print("Solving Y-aware flow-composition ergodic Cole-Hopf profile...")
    y_flow_u, y_flow_info = solve_y_flow_ergodic()
    print("Solving Alpha/Y-aware fast-alpha ergodic Cole-Hopf profile...")
    alpha_y_u, alpha_y_info = solve_y_aware_ergodic()
    print("Simulating all policies in the same regime/full-alpha-jump environment...")
    metrics, paths = simulate_policies(regime_u, y_flow_u, alpha_y_u)

    # Paired marginal-value contrasts on shared environment paths (per-path penalized PnL).
    pp = paths["pen_paths"]
    ladder = [("I over GLFT", "Y-aware", "GLFT"),
              ("alpha over I", "Alpha/Y-aware", "Y-aware"),
              ("regime over alpha/I (Delta_R)", "Regime-aware", "Alpha/Y-aware")]
    lines = ["contrast,mean_diff,paired_se,ci95_low,ci95_high,paired_corr"]
    print("\n=== Paired penalized-PnL contrasts (shared paths) ===")
    for label, hi, lo in ladder:
        d = np.asarray(pp[hi]) - np.asarray(pp[lo])
        se = float(np.std(d, ddof=1) / np.sqrt(d.size))
        corr = float(np.corrcoef(pp[hi], pp[lo])[0, 1])
        md = float(np.mean(d))
        lines.append(f"{label},{md:.4f},{se:.4f},{md-1.96*se:.4f},{md+1.96*se:.4f},{corr:.4f}")
        print(f"  {label:32s} = {md:+8.2f}  paired se {se:5.2f}  CI95 [{md-1.96*se:+.2f}, {md+1.96*se:+.2f}]  corr {corr:.3f}")
    (OUT / "regime_paired_contrasts.csv").write_text("\n".join(lines) + "\n")
    np.savez_compressed(OUT / "regime_penalized_pnl_paths.npz", **{k.replace(" ", "_").replace("/", "_"): np.asarray(v) for k, v in pp.items()})
    print(f"wrote {OUT / 'regime_paired_contrasts.csv'}")

    plot_quote_depths(regime_u)
    plot_depth_heatmaps(regime_u)
    plot_depth_surfaces(regime_u, y_flow_u, alpha_y_u)
    plot_paths(paths)
    plot_conditional_inventory(
        paths,
        "inventory_by_y",
        "y_centers",
        "OU state i bin center",
        "regime_fast_alpha_expected_inventory_vs_y.png",
    )
    plot_conditional_inventory(
        paths,
        "inventory_by_alpha",
        "alpha_centers",
        "alpha bin center",
        "regime_fast_alpha_expected_inventory_vs_alpha.png",
    )
    plot_inventory_by_regime(paths)
    plot_regime_signal_diagnostics(paths)
    plot_regime_direction_alignment(paths)
    plot_lifetime_inventory(paths)
    plot_terminal_pnl_hist(paths)
    plot_terminal_pnl_kde(paths)
    plot_summary(metrics)
    plot_signal_alignment(metrics)
    plot_pnl_decomposition(metrics)
    plot_sample_path(paths)
    write_outputs(metrics, paths, regime_info, y_flow_info, alpha_y_info)

    print(json.dumps(metrics, indent=2))
    print(f"Regime-aware ergodic info: {json.dumps(regime_info)}")
    print(f"Y-flow ergodic info: {json.dumps(y_flow_info)}")
    print(f"Alpha/Y ergodic info: {json.dumps(alpha_y_info)}")
    print(f"Wrote outputs to {OUT}")


if __name__ == "__main__":
    main()
