#!/usr/bin/env python
"""CARA/GLFT signal-blind and I-aware RHJB study.

This script is intentionally self-contained.  It computes:

1. The finite-horizon signal-blind GLFT inventory HJB on a finite inventory grid.
2. The GLFT Gaussian asymptotic approximation for the signal-blind depths.
3. The I-aware GLFT-reduced HJB by Strang splitting:
   half OU step in u, full Cole-Hopf inventory step in exp(k u), half OU step in u.
4. The closed-form I-aware quotes: dynamic-curvature scaffold, the body's
   self-consistent cubic form, and the appendix Galerkin-cubic loading.
5. Monte Carlo PnL and lifetime-inventory diagnostics under I-driven side flow
   for four policies: signal-blind Gaussian GLFT, I-aware RHJB,
   self-consistent cubic, and Galerkin-cubic.

Outputs are written to imagens_tex/toy_model_T10000_qmax60/ by default
(override with ASOU5_CARA_OUTPUT_DIR).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh, expm
from scipy.stats import gaussian_kde

OUT = ROOT / os.environ.get("ASOU5_CARA_OUTPUT_DIR", "imagens_tex/toy_model_T10000_qmax60")
OUT.mkdir(parents=True, exist_ok=True)


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def standard_error(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


@dataclass(frozen=True)
class Params:
    T: float = env_float("ASOU5_CARA_T", 10000.0)
    gamma: float = env_float("ASOU5_CARA_GAMMA", 0.01)
    sigma: float = env_float("ASOU5_CARA_SIGMA", 0.3)
    bar_lambda: float = env_float("ASOU5_CARA_BAR_LAMBDA", 0.9)
    k: float = env_float("ASOU5_CARA_K", 2.0)
    qmax: int = env_int("ASOU5_CARA_QMAX", 60)

    beta: float = env_float("ASOU5_CARA_BETA", 0.0125)
    eta: float = env_float("ASOU5_CARA_ETA", 0.32)
    y_max: float = env_float("ASOU5_CARA_Y_MAX", 6.0)
    ny: int = env_int("ASOU5_CARA_NY", 121)

    dt_hjb: float = env_float("ASOU5_CARA_DT_HJB", 0.5)
    hjb_tol: float = env_float("ASOU5_CARA_HJB_TOL", 2e-8)
    check_every: int = env_int("ASOU5_CARA_CHECK_EVERY", 20)

    n_paths: int = env_int("ASOU5_CARA_N_PATHS", 20000)
    dt_sim: float = env_float("ASOU5_CARA_DT_SIM", 0.05)
    seed: int = env_int("ASOU5_CARA_SEED", 20260519)
    record_dt: float = env_float("ASOU5_CARA_RECORD_DT", 1.0)
    surface_nt: int = env_int("ASOU5_CARA_SURFACE_NT", 61)
    surface_q_pad: int = env_int("ASOU5_CARA_SURFACE_Q_PAD", 60)
    plot_qmax: int = env_int("ASOU5_CARA_PLOT_QMAX", 30)

    @property
    def delta_gamma(self) -> float:
        return np.log1p(self.gamma / self.k) / self.gamma

    @property
    def chi_gamma(self) -> float:
        return (1.0 + self.gamma / self.k) ** (-(1.0 + self.k / self.gamma))

    @property
    def alpha_c(self) -> float:
        return 0.5 * self.k * self.gamma * self.sigma**2

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
    def A_I(self) -> float:
        return (self.rho / self.k) / (self.rho + 2.0 * self.beta)


P = Params()
Q = np.arange(-P.qmax, P.qmax + 1, dtype=int)
QF = Q.astype(float)
NQ = len(Q)
Y = np.linspace(-P.y_max, P.y_max, P.ny)
DY = Y[1] - Y[0]
Y0_IDX = int(np.argmin(np.abs(Y)))
Q0_IDX = int(np.where(Q == 0)[0][0])
LP0_Y = P.bar_lambda * (1.0 + np.tanh(Y))
LM0_Y = P.bar_lambda * (1.0 - np.tanh(Y))

COLORS = {
    "Signal-blind finite-grid GLFT": "#1b9e77",
    "Signal-blind Gaussian GLFT": "#66a61e",
    "I-aware RHJB": "#377eb8",
    "I-aware affine ansatz": "#ff7f00",
    "I-aware self-consistent cubic": "#a65628",
    # Legacy aliases retained for diagnostic scripts that import this palette.
    "cubic-skew refined ansatz": "#a65628",
    "curvature-refined ansatz": "#984ea3",
    "I-aware tilted-GLFT ansatz": "#984ea3",
    "I-aware Galerkin-cubic": "#e41a1c",
    "I-aware galerkin ansatz": "#e41a1c",
}


def q_plot_bounds() -> tuple[int, int]:
    qmax_plot = min(P.plot_qmax, P.qmax)
    return -qmax_plot, qmax_plot


def q_plot_mask(q_values: np.ndarray) -> np.ndarray:
    q_min, q_max = q_plot_bounds()
    return (q_values >= q_min) & (q_values <= q_max) & (q_values > -P.qmax) & (q_values < P.qmax)


def build_ou_step(dt: float) -> np.ndarray:
    """Matrix exponential of the OU generator on the truncated y-grid."""
    n = P.ny
    d1 = np.zeros((n, n))
    d2 = np.zeros((n, n))

    for j in range(1, n - 1):
        d1[j, j - 1] = -0.5 / DY
        d1[j, j + 1] = 0.5 / DY
        d2[j, j - 1] = 1.0 / DY**2
        d2[j, j] = -2.0 / DY**2
        d2[j, j + 1] = 1.0 / DY**2

    # Stable one-sided closure on the truncated y-domain.
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


def blind_inventory_matrix(lambda_plus_coeff: float, lambda_minus_coeff: float) -> np.ndarray:
    mat = np.diag(-P.alpha_c * QF**2)
    for i in range(NQ):
        if i > 0:
            mat[i, i - 1] = lambda_plus_coeff
        if i < NQ - 1:
            mat[i, i + 1] = lambda_minus_coeff
    return mat


def solve_blind_hjb() -> tuple[np.ndarray, dict[str, float]]:
    """Finite-horizon signal-blind GLFT theta, stabilized by spectral shifting."""
    mat = blind_inventory_matrix(P.eta_c, P.eta_c)
    evals, evecs = eigh(mat)
    principal_idx = int(np.argmax(evals))
    lambda_max = float(evals[principal_idx])
    coeff = evecs.T @ np.ones(NQ)
    theta_scaled = evecs @ (np.exp((evals - lambda_max) * P.T) * coeff)

    # Eigenvectors are defined up to sign.  Use the positive orientation.
    if theta_scaled[Q0_IDX] < 0.0:
        theta_scaled *= -1.0
    theta_scaled = np.maximum(theta_scaled, 1e-300)
    theta_scaled /= theta_scaled[Q0_IDX]

    info = {
        "lambda_max": lambda_max,
        "theta_min": float(np.min(theta_scaled)),
        "theta_max": float(np.max(theta_scaled)),
    }
    return theta_scaled, info


def principal_blind_eigenvector() -> np.ndarray:
    mat = blind_inventory_matrix(P.eta_c, P.eta_c)
    evals, evecs = eigh(mat)
    f = evecs[:, int(np.argmax(evals))].copy()
    if f[Q0_IDX] < 0.0:
        f *= -1.0
    f = np.maximum(f, 1e-300)
    return f / f[Q0_IDX]


def depths_from_theta(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # +inf marks the withdrawn side at the inventory boundary (same convention
    # as the other three policy depth functions; fills treat inf as rate 0).
    ask = np.full(NQ, np.inf)
    bid = np.full(NQ, np.inf)
    ask[1:] = P.delta_gamma + np.log(theta[1:] / theta[:-1]) / P.k
    bid[:-1] = P.delta_gamma + np.log(theta[:-1] / theta[1:]) / P.k
    return ask, bid


def surface_tau_grid() -> np.ndarray:
    return np.linspace(0.0, P.T, max(2, P.surface_nt))


def surface_tau_grid_for_params(T: float, surface_nt: int) -> np.ndarray:
    return np.linspace(0.0, T, max(2, surface_nt))


def cara_glft_coefficients(
    sigma: float,
    bar_lambda: float,
    k: float,
    gamma: float,
) -> tuple[float, float, float, float]:
    delta_gamma = np.log1p(gamma / k) / gamma
    chi_gamma = (1.0 + gamma / k) ** (-(1.0 + k / gamma))
    alpha_c = 0.5 * k * gamma * sigma**2
    eta_c = bar_lambda * chi_gamma
    return delta_gamma, chi_gamma, alpha_c, eta_c


def build_ou_step_for_grid(y_grid: np.ndarray, beta: float, eta: float, dt: float) -> np.ndarray:
    """Matrix exponential of the OU generator on a caller-supplied y-grid."""
    n = len(y_grid)
    dy = float(y_grid[1] - y_grid[0])
    d1 = np.zeros((n, n))
    d2 = np.zeros((n, n))

    for j in range(1, n - 1):
        d1[j, j - 1] = -0.5 / dy
        d1[j, j + 1] = 0.5 / dy
        d2[j, j - 1] = 1.0 / dy**2
        d2[j, j] = -2.0 / dy**2
        d2[j, j + 1] = 1.0 / dy**2

    d1[0, 0] = -1.0 / dy
    d1[0, 1] = 1.0 / dy
    d1[-1, -2] = -1.0 / dy
    d1[-1, -1] = 1.0 / dy
    d2[0, 0] = -2.0 / dy**2
    d2[0, 1] = 2.0 / dy**2
    d2[-1, -2] = 2.0 / dy**2
    d2[-1, -1] = -2.0 / dy**2

    generator = np.diag(-beta * y_grid) @ d1 + 0.5 * eta**2 * d2
    return expm(dt * generator)


def blind_depth_surfaces_for_params(
    T: float,
    sigma: float,
    bar_lambda: float,
    k: float,
    gamma: float,
    qmax: int,
    surface_nt: int,
    surface_q_pad: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Signal-blind finite-horizon HJB depths for a self-contained parameter set."""
    delta_gamma, _, alpha_c, eta_c = cara_glft_coefficients(sigma, bar_lambda, k, gamma)
    tau_grid = surface_tau_grid_for_params(T, surface_nt)
    qmax_hidden = qmax + max(0, surface_q_pad)
    q_hidden = np.arange(-qmax_hidden, qmax_hidden + 1, dtype=int)
    q_hidden_f = q_hidden.astype(float)
    n_hidden = len(q_hidden)
    q0_hidden_idx = int(np.where(q_hidden == 0)[0][0])

    mat = np.diag(-alpha_c * q_hidden_f**2)
    idx = np.arange(n_hidden - 1)
    mat[idx, idx + 1] = eta_c
    mat[idx + 1, idx] = eta_c
    evals, evecs = eigh(mat)
    lambda_max = float(np.max(evals))
    coeff = evecs.T @ np.ones(n_hidden)
    ask_hidden = np.full((len(tau_grid), n_hidden), np.nan)
    bid_hidden = np.full((len(tau_grid), n_hidden), np.nan)

    for it, tau in enumerate(tau_grid):
        theta = evecs @ (np.exp((evals - lambda_max) * tau) * coeff)
        if theta[q0_hidden_idx] < 0.0:
            theta *= -1.0
        theta = np.maximum(theta, 1e-300)
        theta /= theta[q0_hidden_idx]
        ask_hidden[it, 1:] = delta_gamma + np.log(theta[1:] / theta[:-1]) / k
        bid_hidden[it, :-1] = delta_gamma + np.log(theta[:-1] / theta[1:]) / k

    visible = (q_hidden >= -qmax) & (q_hidden <= qmax)
    q_visible = q_hidden[visible]
    ask = ask_hidden[:, visible]
    bid = bid_hidden[:, visible]
    spread = ask + bid
    return tau_grid, q_visible, ask, bid, spread


def y_aware_depth_surfaces_y0_for_params(
    T: float,
    sigma: float,
    bar_lambda: float,
    k: float,
    gamma: float,
    qmax: int,
    surface_nt: int,
    surface_q_pad: int,
    beta: float,
    eta: float,
    y_max: float,
    ny: int,
    dt_hjb: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """I-aware RHJB depths at y=0 for a self-contained parameter set."""
    delta_gamma, chi_gamma, alpha_c, _ = cara_glft_coefficients(sigma, bar_lambda, k, gamma)
    tau_grid = surface_tau_grid_for_params(T, surface_nt)
    target_steps = np.unique(np.rint(tau_grid / dt_hjb).astype(int))
    step_to_pos = {int(step): int(pos) for pos, step in enumerate(target_steps)}
    tau_recorded = target_steps.astype(float) * dt_hjb

    qmax_hidden = qmax + max(0, surface_q_pad)
    q_hidden = np.arange(-qmax_hidden, qmax_hidden + 1, dtype=int)
    q_hidden_f = q_hidden.astype(float)
    n_hidden = len(q_hidden)
    q0_hidden_idx = int(np.where(q_hidden == 0)[0][0])
    visible = (q_hidden >= -qmax) & (q_hidden <= qmax)
    q_visible = q_hidden[visible]

    y_grid = np.linspace(-y_max, y_max, ny)
    y0_idx = int(np.argmin(np.abs(y_grid)))
    lp0_y = bar_lambda * (1.0 + np.tanh(y_grid))
    lm0_y = bar_lambda * (1.0 - np.tanh(y_grid))

    ask = np.full((len(target_steps), len(q_visible)), np.nan)
    bid = np.full((len(target_steps), len(q_visible)), np.nan)
    ou_half = build_ou_step_for_grid(y_grid, beta, eta, 0.5 * dt_hjb)

    def hidden_y_inventory_matrix(lp0: float, lm0: float) -> np.ndarray:
        mat = np.diag(-alpha_c * q_hidden_f**2)
        for i in range(n_hidden):
            if i > 0:
                mat[i, i - 1] = chi_gamma * lp0
            if i < n_hidden - 1:
                mat[i, i + 1] = chi_gamma * lm0
        return mat

    inv_steps = np.stack([expm(dt_hjb * hidden_y_inventory_matrix(lp, lm)) for lp, lm in zip(lp0_y, lm0_y)])
    steps = int(round(T / dt_hjb))
    u = np.zeros((n_hidden, ny))

    def normalize_hidden(u_now: np.ndarray) -> np.ndarray:
        return u_now - u_now[q0_hidden_idx, y0_idx]

    def hidden_depth_grid_from_u(u_now: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ask_now = np.full((n_hidden, ny), np.nan)
        bid_now = np.full((n_hidden, ny), np.nan)
        ask_now[1:, :] = delta_gamma - u_now[:-1, :] + u_now[1:, :]
        bid_now[:-1, :] = delta_gamma - u_now[1:, :] + u_now[:-1, :]
        return ask_now, bid_now

    u = normalize_hidden(u)

    def record(step: int, u_now: np.ndarray) -> None:
        pos = step_to_pos.get(step)
        if pos is None:
            return
        ask_grid, bid_grid = hidden_depth_grid_from_u(u_now)
        ask[pos] = ask_grid[visible, y0_idx]
        bid[pos] = bid_grid[visible, y0_idx]

    record(0, u)
    for step in range(1, steps + 1):
        u = u @ ou_half.T
        shift = np.max(u, axis=0)
        w = np.exp(np.clip(k * (u - shift[None, :]), -700.0, 700.0))
        w_next = np.empty_like(w)
        for j in range(ny):
            w_next[:, j] = inv_steps[j] @ w[:, j]
        u = np.log(np.maximum(w_next, 1e-300)) / k + shift[None, :]
        u = u @ ou_half.T
        u = normalize_hidden(u)
        record(step, u)

    spread = ask + bid
    return tau_recorded, q_visible, ask, bid, spread


def blind_depth_surfaces() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Signal-blind finite-horizon HJB depths over time-to-maturity and inventory."""
    tau_grid = surface_tau_grid()
    qmax_hidden = P.qmax + max(0, P.surface_q_pad)
    q_hidden = np.arange(-qmax_hidden, qmax_hidden + 1, dtype=int)
    q_hidden_f = q_hidden.astype(float)
    n_hidden = len(q_hidden)
    q0_hidden_idx = int(np.where(q_hidden == 0)[0][0])

    mat = np.diag(-P.alpha_c * q_hidden_f**2)
    idx = np.arange(n_hidden - 1)
    mat[idx, idx + 1] = P.eta_c
    mat[idx + 1, idx] = P.eta_c
    evals, evecs = eigh(mat)
    lambda_max = float(np.max(evals))
    coeff = evecs.T @ np.ones(n_hidden)
    ask_hidden = np.full((len(tau_grid), n_hidden), np.nan)
    bid_hidden = np.full((len(tau_grid), n_hidden), np.nan)

    for it, tau in enumerate(tau_grid):
        theta = evecs @ (np.exp((evals - lambda_max) * tau) * coeff)
        if theta[q0_hidden_idx] < 0.0:
            theta *= -1.0
        theta = np.maximum(theta, 1e-300)
        theta /= theta[q0_hidden_idx]
        ask_hidden[it, 1:] = P.delta_gamma + np.log(theta[1:] / theta[:-1]) / P.k
        bid_hidden[it, :-1] = P.delta_gamma + np.log(theta[:-1] / theta[1:]) / P.k

    visible = (q_hidden >= -P.qmax) & (q_hidden <= P.qmax)
    ask = ask_hidden[:, visible]
    bid = bid_hidden[:, visible]
    spread = ask + bid
    return tau_grid, ask, bid, spread


def y_aware_depth_surfaces_y0() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """I-aware RHJB depths over time-to-maturity and inventory at balanced imbalance y=0."""
    tau_grid = surface_tau_grid()
    target_steps = np.unique(np.rint(tau_grid / P.dt_hjb).astype(int))
    step_to_pos = {int(step): int(pos) for pos, step in enumerate(target_steps)}
    tau_recorded = target_steps.astype(float) * P.dt_hjb

    qmax_hidden = P.qmax + max(0, P.surface_q_pad)
    q_hidden = np.arange(-qmax_hidden, qmax_hidden + 1, dtype=int)
    q_hidden_f = q_hidden.astype(float)
    n_hidden = len(q_hidden)
    q0_hidden_idx = int(np.where(q_hidden == 0)[0][0])
    visible = (q_hidden >= -P.qmax) & (q_hidden <= P.qmax)

    ask = np.full((len(target_steps), NQ), np.nan)
    bid = np.full((len(target_steps), NQ), np.nan)

    ou_half = build_ou_step(0.5 * P.dt_hjb)

    def hidden_y_inventory_matrix(lp0: float, lm0: float) -> np.ndarray:
        mat = np.diag(-P.alpha_c * q_hidden_f**2)
        for i in range(n_hidden):
            if i > 0:
                mat[i, i - 1] = P.chi_gamma * lp0
            if i < n_hidden - 1:
                mat[i, i + 1] = P.chi_gamma * lm0
        return mat

    inv_steps = np.stack(
        [expm(P.dt_hjb * hidden_y_inventory_matrix(lp, lm)) for lp, lm in zip(LP0_Y, LM0_Y)]
    )
    steps = int(round(P.T / P.dt_hjb))
    u = np.zeros((n_hidden, P.ny))

    def normalize_hidden(u_now: np.ndarray) -> np.ndarray:
        return u_now - u_now[q0_hidden_idx, Y0_IDX]

    def hidden_depth_grid_from_u(u_now: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ask_now = np.full((n_hidden, P.ny), np.nan)
        bid_now = np.full((n_hidden, P.ny), np.nan)
        ask_now[1:, :] = P.delta_gamma - u_now[:-1, :] + u_now[1:, :]
        bid_now[:-1, :] = P.delta_gamma - u_now[1:, :] + u_now[:-1, :]
        return ask_now, bid_now

    u = normalize_hidden(u)

    def record(step: int, u_now: np.ndarray) -> None:
        pos = step_to_pos.get(step)
        if pos is None:
            return
        ask_grid, bid_grid = hidden_depth_grid_from_u(u_now)
        ask[pos] = ask_grid[visible, Y0_IDX]
        bid[pos] = bid_grid[visible, Y0_IDX]

    record(0, u)
    for step in range(1, steps + 1):
        u = u @ ou_half.T
        shift = np.max(u, axis=0)
        w = np.exp(np.clip(P.k * (u - shift[None, :]), -700.0, 700.0))
        w_next = np.empty_like(w)
        for j in range(P.ny):
            w_next[:, j] = inv_steps[j] @ w[:, j]
        u = np.log(np.maximum(w_next, 1e-300)) / P.k + shift[None, :]
        u = u @ ou_half.T
        u = normalize_hidden(u)
        record(step, u)

    spread = ask + bid
    return tau_recorded, ask, bid, spread


def plot_surface_depth(
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

    fig = plt.figure(figsize=(7.6, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        qq,
        tt,
        values,
        cmap=cmap,
        linewidth=0.12,
        edgecolor=(0, 0, 0, 0.25),
        antialiased=True,
    )
    ax.set_xlabel("Inventory")
    ax.set_ylabel("Time [Sec]")
    ax.set_zlabel(zlabel.replace("tick", "Tick"))
    if zlim is not None:
        ax.set_zlim(*zlim)
    if zticks is not None:
        ax.set_zticks(zticks)
    ax.view_init(elev=24, azim=azim)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def glft_stationary_depths(
    sigma: float,
    bar_lambda: float,
    k: float,
    gamma: float,
    qmax: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_grid = np.arange(-qmax, qmax + 1, dtype=int)
    q_float = q_grid.astype(float)
    delta_gamma = np.log1p(gamma / k) / gamma
    chi_gamma = (1.0 + gamma / k) ** (-(1.0 + k / gamma))
    alpha_c = 0.5 * k * gamma * sigma**2
    eta_c = bar_lambda * chi_gamma

    n = len(q_grid)
    mat = np.zeros((n, n))
    np.fill_diagonal(mat, -alpha_c * q_float**2)
    idx = np.arange(n - 1)
    mat[idx, idx + 1] = eta_c
    mat[idx + 1, idx] = eta_c

    vals, vecs = eigh(mat)
    f = vecs[:, -1]
    if np.sum(f) < 0.0:
        f = -f
    f = np.maximum(f, 1e-300)

    ask = np.full(n, np.nan)
    bid = np.full(n, np.nan)
    ask[1:] = delta_gamma + np.log(f[1:] / f[:-1]) / k
    bid[:-1] = delta_gamma + np.log(f[:-1] / f[1:]) / k
    return q_grid, ask, bid


def glft_gaussian_approx_depths_for_params(
    q_grid: np.ndarray,
    sigma: float,
    bar_lambda: float,
    k: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    q_float = q_grid.astype(float)
    delta_gamma = np.log1p(gamma / k) / gamma
    chi_gamma = (1.0 + gamma / k) ** (-(1.0 + k / gamma))
    alpha_c = 0.5 * k * gamma * sigma**2
    eta_c = bar_lambda * chi_gamma
    p_glft = 0.5 * np.sqrt(alpha_c / eta_c)
    ask = delta_gamma - (p_glft / k) * (2.0 * q_float - 1.0)
    bid = delta_gamma + (p_glft / k) * (2.0 * q_float + 1.0)
    return ask, bid


def plot_glft_fig4_fig5_exact_vs_approx() -> None:
    """Generate GLFT exact eigenvector quote diagnostics versus the Gaussian approximation."""
    # Keep the bid and ask exact-vs-Gaussian diagnostics on the same baseline
    # used by the current GLFT surfaces and toy/fast-alpha experiments.
    baseline_specs = [
        (P.sigma, P.bar_lambda, P.k, P.gamma, None, None),
    ]
    bid_specs = baseline_specs
    ask_specs = baseline_specs
    qmax_plot = env_int("ASOU5_CARA_GLFT_FIG45_QMAX", P.plot_qmax)
    qmax_solve = env_int("ASOU5_CARA_GLFT_FIG45_Q_SOLVE", P.qmax)
    qmax_solve = max(qmax_solve, qmax_plot)

    fig_specs = [
        ("bid", r"$s-s^b$ [Tick]", "glft_fig4_bid_exact_vs_gaussian_approx.png", bid_specs, (5.2, 3.25)),
        ("ask", r"$s^a-s$ [Tick]", "glft_fig5_ask_exact_vs_gaussian_approx.png", ask_specs, (5.2, 3.25)),
    ]

    for side, ylabel, filename, specs, figsize in fig_specs:
        fig, axes = plt.subplots(1, len(specs), figsize=figsize, sharey=False, squeeze=False)
        for ax, (sigma, bar_lambda, k, gamma, ylim, yticks) in zip(axes.ravel(), specs):
            q_grid, ask_exact, bid_exact = glft_stationary_depths(
                sigma, bar_lambda, k, gamma, qmax_solve
            )
            ask_approx, bid_approx = glft_gaussian_approx_depths_for_params(
                q_grid, sigma, bar_lambda, k, gamma
            )
            visible = (q_grid >= -qmax_plot) & (q_grid <= qmax_plot)
            if side == "bid":
                mask = visible & (q_grid < qmax_solve)
                exact = bid_exact
                approx = bid_approx
            else:
                mask = visible & (q_grid > -qmax_solve)
                exact = ask_exact
                approx = ask_approx

            ax.plot(q_grid[mask], exact[mask], color="black", linewidth=2.1, label="Eigenvector")
            ax.plot(q_grid[mask], approx[mask], color="black", linewidth=1.8, linestyle=":", label="Gaussian approx")
            ax.set_xlim(-qmax_plot, qmax_plot)
            ax.set_xticks(np.arange(-30, 31, 10))
            if ylim is None:
                vals = np.concatenate([exact[mask], approx[mask]])
                pad = 0.08 * (np.nanmax(vals) - np.nanmin(vals))
                ax.set_ylim(np.nanmin(vals) - pad, np.nanmax(vals) + pad)
            else:
                ax.set_ylim(*ylim)
            if yticks is not None:
                ax.set_yticks(yticks)
            ax.set_xlabel("Inventory")
            ax.set_ylabel(ylabel)
            ax.grid(False)

        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=220)
        plt.close(fig)


def y_aware_depths_y0_for_params(
    sigma: float,
    bar_lambda: float,
    k: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finite-horizon Y-aware depths at y=0 for sensitivity surfaces."""
    q_min, q_max = q_plot_bounds()
    qmax_hidden = env_int("ASOU5_CARA_YAWARE_SENSITIVITY_Q_SOLVE", P.qmax)
    qmax_hidden = max(qmax_hidden, abs(q_min), abs(q_max))
    q_hidden = np.arange(-qmax_hidden, qmax_hidden + 1, dtype=int)
    q_hidden_f = q_hidden.astype(float)
    n_hidden = len(q_hidden)
    q0_hidden_idx = int(np.where(q_hidden == 0)[0][0])
    visible = (q_hidden >= q_min) & (q_hidden <= q_max)

    delta_gamma = np.log1p(gamma / k) / gamma
    chi_gamma = (1.0 + gamma / k) ** (-(1.0 + k / gamma))
    alpha_c = 0.5 * k * gamma * sigma**2
    lp0_y = bar_lambda * (1.0 + np.tanh(Y))
    lm0_y = bar_lambda * (1.0 - np.tanh(Y))

    def hidden_inventory_matrix(lp0: float, lm0: float) -> np.ndarray:
        mat = np.diag(-alpha_c * q_hidden_f**2)
        for i in range(n_hidden):
            if i > 0:
                mat[i, i - 1] = chi_gamma * lp0
            if i < n_hidden - 1:
                mat[i, i + 1] = chi_gamma * lm0
        return mat

    ou_half = build_ou_step(0.5 * P.dt_hjb)
    inv_steps = np.stack(
        [expm(P.dt_hjb * hidden_inventory_matrix(lp, lm)) for lp, lm in zip(lp0_y, lm0_y)]
    )
    steps = int(round(P.T / P.dt_hjb))
    u = np.zeros((n_hidden, P.ny))
    u -= u[q0_hidden_idx, Y0_IDX]

    for _ in range(steps):
        u = u @ ou_half.T
        shift = np.max(u, axis=0)
        w = np.exp(np.clip(k * (u - shift[None, :]), -700.0, 700.0))
        w_next = np.empty_like(w)
        for j in range(P.ny):
            w_next[:, j] = inv_steps[j] @ w[:, j]
        u = np.log(np.maximum(w_next, 1e-300)) / k + shift[None, :]
        u = u @ ou_half.T
        u -= u[q0_hidden_idx, Y0_IDX]

    ask_hidden = np.full((n_hidden, P.ny), np.nan)
    bid_hidden = np.full((n_hidden, P.ny), np.nan)
    ask_hidden[1:, :] = delta_gamma - u[:-1, :] + u[1:, :]
    bid_hidden[:-1, :] = delta_gamma - u[1:, :] + u[:-1, :]
    return q_hidden[visible], ask_hidden[visible, Y0_IDX], bid_hidden[visible, Y0_IDX]


def write_parameter_surface_csv(
    x_name: str,
    x_values: np.ndarray,
    q_values: np.ndarray,
    values: np.ndarray,
    filename: str,
    value_name: str,
) -> None:
    rows = [f"{x_name},q,{value_name}"]
    for ix, x_value in enumerate(x_values):
        for iq, q in enumerate(q_values):
            rows.append(f"{x_value:.10g},{q},{values[ix, iq]:.10g}")
    (OUT / filename).write_text("\n".join(rows) + "\n")


def plot_parameter_surface(
    x_values: np.ndarray,
    q_values: np.ndarray,
    values: np.ndarray,
    xlabel: str,
    zlabel: str,
    filename: str,
    cmap: str,
    azim: float = -60.0,
    zlim: tuple[float, float] | None = None,
    zticks: np.ndarray | None = None,
) -> None:
    finite_columns = np.any(np.isfinite(values), axis=0)
    q_values = q_values[finite_columns]
    values = values[:, finite_columns]
    xx, qq = np.meshgrid(x_values, q_values)
    finite_values = values[np.isfinite(values)]
    if finite_values.size:
        data_lo = float(np.nanmin(finite_values))
        data_hi = float(np.nanmax(finite_values))
        if zlim is None or data_lo < zlim[0] or data_hi > zlim[1]:
            pad = 0.08 * max(data_hi - data_lo, 1e-12)
            zlim = (data_lo - pad, data_hi + pad)
            zticks = None
    fig = plt.figure(figsize=(7.6, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        xx,
        qq,
        values.T,
        cmap=cmap,
        linewidth=0.12,
        edgecolor=(0, 0, 0, 0.25),
        antialiased=True,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Inventory")
    ax.set_zlabel(zlabel)
    if zlim is not None:
        ax.set_zlim(*zlim)
    if zticks is not None:
        ax.set_zticks(zticks)
    ax.view_init(elev=24, azim=azim)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def plot_glft_sensitivity_surfaces() -> None:
    sensitivity_nt = env_int("ASOU5_CARA_SENSITIVITY_NT", min(P.surface_nt, 31))
    gamma_grid = np.linspace(0.001, 0.5, sensitivity_nt)
    q_min, q_max = q_plot_bounds()
    qmax_hidden = env_int("ASOU5_CARA_SENSITIVITY_Q_SOLVE", P.qmax)
    qmax_hidden = max(qmax_hidden, abs(q_min), abs(q_max))
    q_spread = np.arange(q_min, q_max + 1, dtype=int)

    spread_specs = [
        ("fig6", P.sigma, P.bar_lambda, P.k, (4.0, 7.0), np.arange(4.0, 7.01, 0.5)),
        (
            "fig7",
            env_float("ASOU5_CARA_FIG7_SIGMA", 0.6),
            env_float("ASOU5_CARA_FIG7_BAR_LAMBDA", P.bar_lambda),
            env_float("ASOU5_CARA_FIG7_K", 0.9),
            (1.8, 2.5),
            np.arange(1.8, 2.501, 0.1),
        ),
    ]

    for tag, sigma, bar_lambda, k_value, zlim, zticks in spread_specs:
        spread_surface = np.empty((len(gamma_grid), len(q_spread)))
        y_spread_surface = np.empty_like(spread_surface)
        for i, gamma in enumerate(gamma_grid):
            q_hidden, ask, bid = glft_stationary_depths(sigma, bar_lambda, k_value, gamma, qmax_hidden)
            q_to_idx = {int(q): idx for idx, q in enumerate(q_hidden)}
            spread_surface[i] = np.array([ask[q_to_idx[int(q)]] + bid[q_to_idx[int(q)]] for q in q_spread])

            q_y, ask_y, bid_y = y_aware_depths_y0_for_params(sigma, bar_lambda, k_value, gamma)
            q_y_to_idx = {int(q): idx for idx, q in enumerate(q_y)}
            y_spread_surface[i] = np.array(
                [ask_y[q_y_to_idx[int(q)]] + bid_y[q_y_to_idx[int(q)]] for q in q_spread]
            )

        plot_parameter_surface(
            gamma_grid,
            q_spread,
            spread_surface,
            r"$\gamma$",
            "Spread [Tick]",
            f"glft_asymptotic_spread_gamma_inventory_surface_{tag}.png",
            cmap="viridis",
            zlim=zlim,
            zticks=zticks,
        )
        write_parameter_surface_csv(
            "gamma",
            gamma_grid,
            q_spread,
            spread_surface,
            f"glft_asymptotic_spread_gamma_inventory_surface_{tag}.csv",
            "spread",
        )
        plot_parameter_surface(
            gamma_grid,
            q_spread,
            y_spread_surface,
            r"$\gamma$",
            "Spread [Tick]",
            f"yaware_hjb_spread_gamma_inventory_surface_{tag}_y0.png",
            cmap="YlOrRd",
            zlim=zlim,
            zticks=zticks,
        )
        write_parameter_surface_csv(
            "gamma",
            gamma_grid,
            q_spread,
            y_spread_surface,
            f"yaware_hjb_spread_gamma_inventory_surface_{tag}_y0.csv",
            "spread",
        )

    k_grid = np.linspace(
        env_float("ASOU5_CARA_FIG8_K_MIN", 0.15),
        env_float("ASOU5_CARA_FIG8_K_MAX", 2.0),
        sensitivity_nt,
    )
    q_bid = np.arange(q_min, q_max + 1, dtype=int)
    bid_surface = np.empty((len(k_grid), len(q_bid)))
    y_bid_surface = np.empty_like(bid_surface)
    for i, k_value in enumerate(k_grid):
        q_hidden, _, bid = glft_stationary_depths(P.sigma, P.bar_lambda, k_value, P.gamma, qmax_hidden)
        q_to_idx = {int(q): idx for idx, q in enumerate(q_hidden)}
        bid_surface[i] = np.array([bid[q_to_idx[int(q)]] for q in q_bid])

        q_y, _, bid_y = y_aware_depths_y0_for_params(P.sigma, P.bar_lambda, k_value, P.gamma)
        q_y_to_idx = {int(q): idx for idx, q in enumerate(q_y)}
        y_bid_surface[i] = np.array([bid_y[q_y_to_idx[int(q)]] for q in q_bid])

    plot_parameter_surface(
        k_grid,
        q_bid,
        bid_surface,
        r"$k$ [Tick$^{-1}$]",
        r"$s-s^b$ [Tick]",
        "glft_asymptotic_bid_k_inventory_surface_fig8.png",
        cmap="viridis",
        zlim=(-2.0, 10.0),
        zticks=np.arange(-2.0, 10.01, 2.0),
    )
    write_parameter_surface_csv(
        "k",
        k_grid,
        q_bid,
        bid_surface,
        "glft_asymptotic_bid_k_inventory_surface_fig8.csv",
        "bid_depth",
    )
    plot_parameter_surface(
        k_grid,
        q_bid,
        y_bid_surface,
        r"$k$ [Tick$^{-1}$]",
        r"$s-s^b$ [Tick]",
        "yaware_hjb_bid_k_inventory_surface_fig8_y0.png",
        cmap="YlOrRd",
        zlim=(-2.0, 10.0),
        zticks=np.arange(-2.0, 10.01, 2.0),
    )
    write_parameter_surface_csv(
        "k",
        k_grid,
        q_bid,
        y_bid_surface,
        "yaware_hjb_bid_k_inventory_surface_fig8_y0.csv",
        "bid_depth",
    )


def write_surface_csv(
    tau_grid: np.ndarray,
    q_values: np.ndarray,
    values: np.ndarray,
    filename: str,
    value_name: str,
) -> None:
    rows = [f"tau,q,{value_name}"]
    for it, tau in enumerate(tau_grid):
        for iq, q in enumerate(q_values):
            val = values[it, iq]
            rows.append(f"{tau:.10g},{q},{val:.10g}")
    (OUT / filename).write_text("\n".join(rows) + "\n")


def plot_glft_style_surfaces() -> None:
    """Replicate the GLFT 3D diagnostic surfaces for blind and I-aware RHJBs."""
    surface_T = env_float("ASOU5_GLFT_SURFACE_T", 600.0)
    surface_gamma = env_float("ASOU5_GLFT_SURFACE_GAMMA", 0.01)
    surface_sigma = env_float("ASOU5_GLFT_SURFACE_SIGMA", 0.3)
    surface_bar_lambda = env_float("ASOU5_GLFT_SURFACE_BAR_LAMBDA", 0.9)
    surface_k = env_float("ASOU5_GLFT_SURFACE_K", P.k)
    # The GLFT paper figures use q in [-30, 30].  Keep that default separate
    # from P.plot_qmax, which may be enlarged for lifetime-inventory histograms.
    surface_qmax = env_int("ASOU5_GLFT_SURFACE_QMAX", 30)
    surface_nt = env_int("ASOU5_GLFT_SURFACE_NT", 61)
    surface_q_pad = env_int("ASOU5_GLFT_SURFACE_Q_PAD", 60)

    surface_beta = env_float("ASOU5_GLFT_SURFACE_BETA", P.beta)
    surface_eta = env_float("ASOU5_GLFT_SURFACE_ETA", P.eta)
    surface_y_max = env_float("ASOU5_GLFT_SURFACE_Y_MAX", P.y_max)
    surface_ny = env_int("ASOU5_GLFT_SURFACE_NY", P.ny)
    surface_dt_hjb = env_float("ASOU5_GLFT_SURFACE_DT_HJB", P.dt_hjb)

    tau_b, q_surface, ask_b, bid_b, spread_b = blind_depth_surfaces_for_params(
        surface_T,
        surface_sigma,
        surface_bar_lambda,
        surface_k,
        surface_gamma,
        surface_qmax,
        surface_nt,
        surface_q_pad,
    )
    # The paper-style z-axis limits are calibrated to the GLFT figure
    # parameterization. If we change k for the experiment, quote depths move to
    # a different numerical scale, so the surfaces should auto-scale.
    use_paper_axis = (
        abs(surface_k - 0.3) < 1e-12
        and abs(surface_bar_lambda - 0.9) < 1e-12
        and abs(surface_sigma - 0.3) < 1e-12
        and abs(surface_gamma - 0.01) < 1e-12
    )
    quote_zlim = (1.0, 5.5) if use_paper_axis else None
    quote_zticks = np.arange(1.0, 5.51, 0.5) if use_paper_axis else None
    spread_zlim = (6.55, 6.64) if use_paper_axis else None
    spread_zticks = np.arange(6.55, 6.641, 0.01) if use_paper_axis else None
    plot_surface_depth(tau_b, q_surface, bid_b, "Signal-blind HJB bid depth", r"$s-s^b$ [tick]", "cara_glft_blind_hjb_bid_surface.png", azim=-135.0, zlim=quote_zlim, zticks=quote_zticks)
    plot_surface_depth(tau_b, q_surface, ask_b, "Signal-blind HJB ask depth", r"$s^a-s$ [tick]", "cara_glft_blind_hjb_ask_surface.png", zlim=quote_zlim, zticks=quote_zticks)
    plot_surface_depth(tau_b, q_surface, spread_b, "Signal-blind HJB bid-ask spread", r"$\psi$ [tick]", "cara_glft_blind_hjb_spread_surface.png", zlim=spread_zlim, zticks=spread_zticks)
    write_surface_csv(tau_b, q_surface, bid_b, "cara_glft_blind_hjb_bid_surface.csv", "bid_depth")
    write_surface_csv(tau_b, q_surface, ask_b, "cara_glft_blind_hjb_ask_surface.csv", "ask_depth")
    write_surface_csv(tau_b, q_surface, spread_b, "cara_glft_blind_hjb_spread_surface.csv", "spread")

    tau_y, q_y_surface, ask_y, bid_y, spread_y = y_aware_depth_surfaces_y0_for_params(
        surface_T,
        surface_sigma,
        surface_bar_lambda,
        surface_k,
        surface_gamma,
        surface_qmax,
        surface_nt,
        surface_q_pad,
        surface_beta,
        surface_eta,
        surface_y_max,
        surface_ny,
        surface_dt_hjb,
    )
    plot_surface_depth(tau_y, q_y_surface, bid_y, "I-aware RHJB bid depth at i=0", r"$s-s^b$ [tick]", "cara_glft_yaware_hjb_bid_surface_y0.png", cmap="YlOrRd", azim=-135.0, zlim=quote_zlim, zticks=quote_zticks)
    plot_surface_depth(tau_y, q_y_surface, ask_y, "I-aware RHJB ask depth at i=0", r"$s^a-s$ [tick]", "cara_glft_yaware_hjb_ask_surface_y0.png", cmap="YlOrRd", zlim=quote_zlim, zticks=quote_zticks)
    plot_surface_depth(tau_y, q_y_surface, spread_y, "I-aware RHJB bid-ask spread at i=0", r"$\psi$ [tick]", "cara_glft_yaware_hjb_spread_surface_y0.png", cmap="YlOrRd", zlim=spread_zlim, zticks=spread_zticks)
    write_surface_csv(tau_y, q_y_surface, bid_y, "cara_glft_yaware_hjb_bid_surface_y0.csv", "bid_depth")
    write_surface_csv(tau_y, q_y_surface, ask_y, "cara_glft_yaware_hjb_ask_surface_y0.csv", "ask_depth")
    write_surface_csv(tau_y, q_y_surface, spread_y, "cara_glft_yaware_hjb_spread_surface_y0.csv", "spread")


def blind_glft_approx_depths(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qf = q.astype(float)
    ask = P.delta_gamma - (P.p_glft / P.k) * (2.0 * qf - 1.0)
    bid = P.delta_gamma + (P.p_glft / P.k) * (2.0 * qf + 1.0)
    ask = np.where(q > -P.qmax, ask, np.inf)
    bid = np.where(q < P.qmax, bid, np.inf)
    return ask, bid


def y_inventory_matrix(lp0: float, lm0: float) -> np.ndarray:
    return blind_inventory_matrix(P.chi_gamma * lp0, P.chi_gamma * lm0)


def precompute_y_inventory_steps() -> np.ndarray:
    return np.stack([expm(P.dt_hjb * y_inventory_matrix(lp, lm)) for lp, lm in zip(LP0_Y, LM0_Y)])


def normalize_u(u: np.ndarray) -> np.ndarray:
    return u - u[Q0_IDX, Y0_IDX]


def depth_grid_from_u(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ask = np.full((NQ, P.ny), np.nan)
    bid = np.full((NQ, P.ny), np.nan)
    ask[1:, :] = P.delta_gamma - u[:-1, :] + u[1:, :]
    bid[:-1, :] = P.delta_gamma - u[1:, :] + u[:-1, :]
    return ask, bid


def solve_y_aware_hjb() -> tuple[np.ndarray, dict[str, float]]:
    """Finite-horizon/large-horizon I-aware GLFT-reduced HJB by splitting."""
    ou_half = build_ou_step(0.5 * P.dt_hjb)
    inv_steps = precompute_y_inventory_steps()
    steps = int(round(P.T / P.dt_hjb))
    u = np.zeros((NQ, P.ny))
    u = normalize_u(u)
    prev_ask: np.ndarray | None = None
    prev_bid: np.ndarray | None = None
    last_diff = np.inf
    converged = False

    for step in range(1, steps + 1):
        u = u @ ou_half.T

        shift = np.max(u, axis=0)
        w = np.exp(np.clip(P.k * (u - shift[None, :]), -700.0, 700.0))
        w_next = np.empty_like(w)
        for j in range(P.ny):
            w_next[:, j] = inv_steps[j] @ w[:, j]
        u = np.log(np.maximum(w_next, 1e-300)) / P.k + shift[None, :]

        u = u @ ou_half.T
        u = normalize_u(u)

        if step % P.check_every == 0:
            ask, bid = depth_grid_from_u(u)
            if prev_ask is not None and prev_bid is not None:
                last_diff = float(max(np.nanmax(np.abs(ask - prev_ask)), np.nanmax(np.abs(bid - prev_bid))))
                if last_diff < P.hjb_tol:
                    converged = True
                    break
            prev_ask = ask
            prev_bid = bid

    info = {
        "tau": float(step * P.dt_hjb),
        "steps": float(step),
        "last_depth_diff": float(last_diff),
        "tolerance": float(P.hjb_tol),
        "converged": float(converged),
    }
    return u, info


def interp_y(values: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.interp(np.clip(y, Y[0], Y[-1]), Y, values)


def y_aware_hjb_depths(u: np.ndarray, q: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ask = np.full_like(y, np.inf, dtype=float)
    bid = np.full_like(y, np.inf, dtype=float)
    for qval in np.unique(q):
        mask = q == qval
        idx = int(qval + P.qmax)
        ui = interp_y(u[idx], y[mask])
        if idx > 0:
            ask[mask] = P.delta_gamma - interp_y(u[idx - 1], y[mask]) + ui
        if idx < NQ - 1:
            bid[mask] = P.delta_gamma - interp_y(u[idx + 1], y[mask]) + ui
    return ask, bid


def y_loading_profile(y_values: np.ndarray) -> np.ndarray:
    """Saturating toy loading h_I(i) = A_1 tanh(A_2 i) (tex eq:toy_loading_ansatz),
    with A_1 = A_I eta/sqrt(beta), A_2 = sqrt(beta)/eta and origin slope
    A_I = (2/k) kappa / (2 kappa + 2 beta) fixed by the i^1 local balance.
    All constants elementary in the model parameters."""
    y_values = np.asarray(y_values, dtype=float)
    return P.A_I * (P.eta / np.sqrt(P.beta)) * np.tanh((np.sqrt(P.beta) / P.eta) * y_values)


def y_curvature_profiles(y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loading h, refined Gaussian rate r(i), and inventory-curvature s(i)
    of the curvature-refined tilted-GLFT ansatz (Prop. curvature-refined quotes).
    r_G = 2*p_glft; theta = i - k h_I."""
    y = np.asarray(y_values, dtype=float)
    h = y_loading_profile(y)
    theta = y - P.k * h
    r_g = 2.0 * P.p_glft
    r = r_g * np.sqrt(np.cosh(y) / np.cosh(theta))
    s = (r**2 / 6.0) * np.tanh(theta)
    return h, r, s


def y_curvature_main_depths(q: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dynamic-curvature quotes (tex eq:toy_affine_quotes, prop:toy_affine): dynamic r(i),
    NO cubic s(i) -- the leading-order scaffold, not separately reported in figures."""
    qf = q.astype(float)
    h, r, s = y_curvature_profiles(y)
    ask = P.delta_gamma + h - (r / (2.0 * P.k)) * (2.0 * qf - 1.0)
    bid = P.delta_gamma - h + (r / (2.0 * P.k)) * (2.0 * qf + 1.0)
    ask = np.where(q > -P.qmax, ask, np.inf)
    bid = np.where(q < P.qmax, bid, np.inf)
    return ask, bid


def y_curvature_refined_depths(q: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Body self-consistent cubic quotes (tex eq:toy_curvature_quotes):
    dynamic r(i) AND cubic s(i) -- the working closed form of the paper."""
    qf = q.astype(float)
    h, r, s = y_curvature_profiles(y)
    ask = P.delta_gamma + h - (r / (2.0 * P.k)) * (2.0 * qf - 1.0) + (s / P.k) * (qf**2 - qf + 1.0 / 3.0)
    bid = P.delta_gamma - h + (r / (2.0 * P.k)) * (2.0 * qf + 1.0) - (s / P.k) * (qf**2 + qf + 1.0 / 3.0)
    ask = np.where(q > -P.qmax, ask, np.inf)
    bid = np.where(q < P.qmax, bid, np.inf)
    return ask, bid
def y_affine_depths(q: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frozen-curvature scaffold: constant GLFT curvature r_G and the loading h_I,
    NO i-dependent curvature, NO cubic.  NOTE: this is NOT eq:toy_affine_quotes
    (which carries the dynamic r(i)); kept for reference/ablation only."""
    qf = q.astype(float)
    h = y_loading_profile(y)
    r_g = 2.0 * P.p_glft
    ask = P.delta_gamma + h - (r_g / (2.0 * P.k)) * (2.0 * qf - 1.0)
    bid = P.delta_gamma - h + (r_g / (2.0 * P.k)) * (2.0 * qf + 1.0)
    ask = np.where(q > -P.qmax, ask, np.inf)
    bid = np.where(q < P.qmax, bid, np.inf)
    return ask, bid


# Galerkin Tier-1 loading: same sqrt-with-scale shape as the body (algebraic) form, but with
# (A, c) fixed by the OU-Gaussian Galerkin projection of the curvature-consistent loading
# residual (global yet parameter-free; see closed_form_tests/nlo_closed_form_tier1.py).  It is
# the most shape-faithful variant; we display it alongside the elementary algebraic loading.
A_GAL, C_GAL = 0.280126, 2.205139


def y_galerkin_loading(y_values: np.ndarray) -> np.ndarray:
    y = np.asarray(y_values, dtype=float)
    return A_GAL * y / np.sqrt(1.0 + P.beta * y**2 / (C_GAL * P.eta**2))


def y_galerkin_profiles(y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_values, dtype=float)
    h = y_galerkin_loading(y)
    theta = y - P.k * h
    r_g = 2.0 * P.p_glft
    r = r_g * np.sqrt(np.cosh(y) / np.cosh(theta))
    s = (r**2 / 6.0) * np.tanh(theta)
    return h, r, s


def y_galerkin_refined_depths(q: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qf = q.astype(float)
    h, r, s = y_galerkin_profiles(y)
    ask = P.delta_gamma + h - (r / (2.0 * P.k)) * (2.0 * qf - 1.0) + (s / P.k) * (qf**2 - qf + 1.0 / 3.0)
    bid = P.delta_gamma - h + (r / (2.0 * P.k)) * (2.0 * qf + 1.0) - (s / P.k) * (qf**2 + qf + 1.0 / 3.0)
    ask = np.where(q > -P.qmax, ask, np.inf)
    bid = np.where(q < P.qmax, bid, np.inf)
    return ask, bid


def fill_count_mean(lambda0: np.ndarray, depth: np.ndarray) -> np.ndarray:
    mult = np.exp(np.clip(-P.k * depth, -60.0, 60.0))
    intensity = np.where(np.isfinite(depth), lambda0 * mult, 0.0)
    return intensity * P.dt_sim


def sample_capped_poisson(
    rng: np.random.Generator,
    mean: np.ndarray,
    cap: np.ndarray,
) -> np.ndarray:
    cap = np.maximum(cap, 0).astype(int, copy=False)
    mean = np.where(np.isfinite(mean) & (mean > 0.0), mean, 0.0)
    threshold = cap + 10.0 * np.sqrt(np.maximum(mean, 1.0))
    saturated = mean > threshold
    sampled_mean = np.where(saturated, 0.0, mean)
    counts = np.minimum(rng.poisson(sampled_mean), cap).astype(int, copy=False)
    counts[saturated] = cap[saturated]
    return counts


def quantile_from_counts(centers: np.ndarray, counts: np.ndarray, prob: float) -> float:
    """Works for raw counts or probability weights (total need not be integer)."""
    total = float(np.sum(counts))
    if total <= 0:
        return float("nan")
    idx = int(np.searchsorted(np.cumsum(counts), prob * total, side="left"))
    idx = min(max(idx, 0), len(centers) - 1)
    return float(centers[idx])


def simulate_policies(theta_blind: np.ndarray, u_y: np.ndarray) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    rng = np.random.default_rng(P.seed)            # policy fills (per-policy, vectorized)
    rng_env = np.random.default_rng(P.seed + 777)  # exogenous (y, s) path: a single common
    # realization independent of the policy set.  Note: only the environment path is
    # policy-set-invariant; the fill draws share one stream, so per-policy metrics can
    # shift slightly if the policy list changes.
    steps = int(round(P.T / P.dt_sim))
    record_every = max(1, int(round(P.record_dt / P.dt_sim)))
    sqrt_dt = np.sqrt(P.dt_sim)

    blind_approx_ask, blind_approx_bid = blind_glft_approx_depths(Q)

    policies = [
        "Signal-blind Gaussian GLFT",
        "I-aware RHJB",
        "I-aware self-consistent cubic",
        "I-aware Galerkin-cubic",
    ]
    n_policies = len(policies)
    q = np.zeros((n_policies, P.n_paths), dtype=int)
    cash = np.zeros((n_policies, P.n_paths), dtype=float)
    fills = np.zeros((n_policies, P.n_paths), dtype=float)
    min_ask = np.full((n_policies, P.n_paths), np.inf)
    min_bid = np.full((n_policies, P.n_paths), np.inf)
    abs_q_integral = np.zeros((n_policies, P.n_paths), dtype=float)
    penalty_integral = np.zeros((n_policies, P.n_paths), dtype=float)
    phi_eff = 0.5 * P.gamma * P.sigma**2

    y = np.zeros(P.n_paths)
    s = np.zeros(P.n_paths)

    rec_t: list[float] = []
    rec_mean_pnl = {name: [] for name in policies}
    rec_mean_penalty = {name: [] for name in policies}
    rec_mean_absq = {name: [] for name in policies}
    lifetime_full_counts = np.zeros((n_policies, 2 * P.qmax + 1), dtype=np.int64)
    lifetime_abs_counts = np.zeros((n_policies, P.qmax + 1), dtype=np.int64)
    lifetime_total = 0
    y_bins = np.linspace(-P.y_max, P.y_max, 33)
    y_centers = 0.5 * (y_bins[:-1] + y_bins[1:])
    conditional_y_counts = np.zeros(len(y_centers), dtype=np.int64)
    conditional_q_sums = np.zeros((n_policies, len(y_centers)), dtype=float)
    y_clip_count = 0
    y_clip_total = 0

    for step in range(steps):
        t = step * P.dt_sim
        y_clip_count += int(np.sum((y < Y[0]) | (y > Y[-1])))
        y_clip_total += P.n_paths
        lambda_plus0 = P.bar_lambda * (1.0 + np.tanh(y))
        lambda_minus0 = P.bar_lambda * (1.0 - np.tanh(y))

        q_before = q.copy()
        ask = np.empty((n_policies, P.n_paths), dtype=float)
        bid = np.empty((n_policies, P.n_paths), dtype=float)
        idx0 = q_before[0] + P.qmax
        ask[0], bid[0] = blind_approx_ask[idx0], blind_approx_bid[idx0]   # Signal-blind Gaussian GLFT
        ask[1], bid[1] = y_aware_hjb_depths(u_y, q_before[1], y)          # I-aware RHJB
        ask[2], bid[2] = y_curvature_refined_depths(q_before[2], y)       # self-consistent cubic (body)
        ask[3], bid[3] = y_galerkin_refined_depths(q_before[3], y)        # Galerkin-cubic (appendix)

        min_ask = np.fmin(min_ask, ask)
        min_bid = np.fmin(min_bid, bid)
        ask_fill = sample_capped_poisson(rng, fill_count_mean(lambda_plus0, ask), q_before + P.qmax)
        bid_fill = sample_capped_poisson(rng, fill_count_mean(lambda_minus0, bid), P.qmax - q_before)

        ask_exec = np.where(np.isfinite(ask), ask, 0.0)
        bid_exec = np.where(np.isfinite(bid), bid, 0.0)
        cash += (s[None, :] + ask_exec) * ask_fill - (s[None, :] - bid_exec) * bid_fill
        q += -ask_fill + bid_fill
        fills += ask_fill.astype(float) + bid_fill.astype(float)
        abs_q_integral += np.abs(q_before) * P.dt_sim
        penalty_integral += phi_eff * q_before.astype(float) ** 2 * P.dt_sim

        # Exogenous state update after quoting/fills at S_t and Y_t.  Drawn from
        # the dedicated environment stream so the common path is policy-set-independent.
        s += P.sigma * sqrt_dt * rng_env.standard_normal(P.n_paths)
        y += -P.beta * y * P.dt_sim + P.eta * sqrt_dt * rng_env.standard_normal(P.n_paths)

        if step % record_every == 0 or step == steps - 1:
            rec_t.append(t + P.dt_sim)
            lifetime_total += P.n_paths
            y_bin_idx = np.digitize(y, y_bins) - 1
            valid_y = (y_bin_idx >= 0) & (y_bin_idx < len(y_centers))
            conditional_y_counts += np.bincount(y_bin_idx[valid_y], minlength=len(y_centers))
            for p_idx, name in enumerate(policies):
                pnl_now = cash[p_idx] + q[p_idx].astype(float) * s
                rec_mean_pnl[name].append(float(np.mean(pnl_now)))
                rec_mean_penalty[name].append(float(np.mean(penalty_integral[p_idx])))
                rec_mean_absq[name].append(float(np.mean(np.abs(q[p_idx]))))
                lifetime_full_counts[p_idx] += np.bincount(q[p_idx] + P.qmax, minlength=2 * P.qmax + 1)
                lifetime_abs_counts[p_idx] += np.bincount(np.abs(q[p_idx]), minlength=P.qmax + 1)
                conditional_q_sums[p_idx] += np.bincount(
                    y_bin_idx[valid_y],
                    weights=q[p_idx, valid_y],
                    minlength=len(y_centers),
                )

    metrics: dict[str, dict[str, float]] = {}
    terminal_pnl = {}
    _pnl_dump = {}
    y_clip_share = float(y_clip_count / max(1, y_clip_total))
    q_centers = np.arange(-P.qmax, P.qmax + 1, dtype=float)
    for p_idx, name in enumerate(policies):
        pnl = cash[p_idx] + q[p_idx].astype(float) * s
        terminal_pnl[name] = pnl
        _pnl_dump[name.replace(" ", "_").replace("/", "_")] = np.asarray(pnl)
        log_loss = -P.gamma * pnl
        m_loss = float(np.max(log_loss))
        certainty_equivalent = -(m_loss + np.log(np.mean(np.exp(log_loss - m_loss)))) / P.gamma
        boot_rng = np.random.default_rng(P.seed + 424242)
        n_boot = 200
        boot_ce = np.empty(n_boot)
        for b in range(n_boot):
            idx_b = boot_rng.integers(0, len(pnl), len(pnl))
            ll = log_loss[idx_b]
            mb = float(np.max(ll))
            boot_ce[b] = -(mb + np.log(np.mean(np.exp(ll - mb)))) / P.gamma
        certainty_equivalent_stderr = float(np.std(boot_ce, ddof=1))
        logw = -P.gamma * pnl; shp = float(np.max(logw))
        wgt = np.exp(logw - shp)
        ess = float(np.sum(wgt))**2 / float(np.sum(wgt**2))
        max_share = float(np.max(wgt) / np.sum(wgt))
        metrics[name] = {
            "terminal_pnl_mean": float(np.mean(pnl)),
            "ce_ess": ess,
            "ce_max_weight_share": max_share,
            "terminal_pnl_std": float(np.std(pnl)),
            "terminal_pnl_stderr": standard_error(pnl),
            "terminal_pnl_q05": float(np.quantile(pnl, 0.05)),
            "terminal_pnl_q50": float(np.quantile(pnl, 0.50)),
            "terminal_pnl_q95": float(np.quantile(pnl, 0.95)),
            "certainty_equivalent": float(certainty_equivalent),
            "certainty_equivalent_stderr": certainty_equivalent_stderr,
            "fills_mean": float(np.mean(fills[p_idx])),
            "fills_stderr": standard_error(fills[p_idx]),
            "running_penalty_mean": float(np.mean(penalty_integral[p_idx])),
            "running_penalty_stderr": standard_error(penalty_integral[p_idx]),
            "avg_abs_inventory": float(np.mean(abs_q_integral[p_idx] / P.T)),
            "terminal_abs_inventory": float(np.mean(np.abs(q[p_idx]))),
            "lifetime_q01": quantile_from_counts(q_centers, lifetime_full_counts[p_idx], 0.01),
            "lifetime_q99": quantile_from_counts(q_centers, lifetime_full_counts[p_idx], 0.99),
            "mean_min_ask_depth": float(np.mean(min_ask[p_idx])),
            "mean_min_bid_depth": float(np.mean(min_bid[p_idx])),
            "y_clip_share": y_clip_share,
        }

    paths: dict[str, object] = {
        "time": np.array(rec_t),
        "mean_pnl": {name: np.array(rec_mean_pnl[name]) for name in policies},
        "mean_penalty": {name: np.array(rec_mean_penalty[name]) for name in policies},
        "mean_absq": {name: np.array(rec_mean_absq[name]) for name in policies},
        "terminal_pnl": terminal_pnl,
        "lifetime_full_counts": {name: lifetime_full_counts[p_idx].copy() for p_idx, name in enumerate(policies)},
        "lifetime_abs_counts": {name: lifetime_abs_counts[p_idx].copy() for p_idx, name in enumerate(policies)},
        "lifetime_total": lifetime_total,
        "conditional_y_centers": y_centers,
        "conditional_y_counts": conditional_y_counts,
        "conditional_q_sums": {name: conditional_q_sums[p_idx].copy() for p_idx, name in enumerate(policies)},
    }
    np.savez_compressed(OUT / "cara_glft_terminal_pnl_paths.npz", **_pnl_dump)
    return metrics, paths


def simulate_glft_fair_environment() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(P.seed + 7919)
    steps = int(round(P.T / P.dt_sim))
    record_every = max(1, int(round(P.record_dt / P.dt_sim)))
    sqrt_dt = np.sqrt(P.dt_sim)

    ask_depths, bid_depths = blind_glft_approx_depths(Q)
    q = np.zeros(P.n_paths, dtype=int)
    cash = np.zeros(P.n_paths)
    s = np.zeros(P.n_paths)
    lifetime_inventory: list[np.ndarray] = []

    lambda0 = np.full(P.n_paths, P.bar_lambda, dtype=float)
    for step in range(steps):
        q_before = q.copy()
        idx = q_before + P.qmax
        ask = ask_depths[idx]
        bid = bid_depths[idx]

        ask_fill = sample_capped_poisson(
            rng,
            fill_count_mean(lambda0, ask),
            q_before + P.qmax,
        )
        bid_fill = sample_capped_poisson(
            rng,
            fill_count_mean(lambda0, bid),
            P.qmax - q_before,
        )

        ask_exec = np.where(np.isfinite(ask), ask, 0.0)
        bid_exec = np.where(np.isfinite(bid), bid, 0.0)
        cash += (s + ask_exec) * ask_fill - (s - bid_exec) * bid_fill
        q += -ask_fill + bid_fill
        s += P.sigma * sqrt_dt * rng.standard_normal(P.n_paths)

        if step % record_every == 0 or step == steps - 1:
            lifetime_inventory.append(q.copy())

    return {
        "terminal_pnl": cash + q.astype(float) * s,
        "lifetime_inventory": np.concatenate(lifetime_inventory) if lifetime_inventory else q,
    }


def plot_blind_depths(theta_blind: np.ndarray) -> None:
    ask_hjb, bid_hjb = depths_from_theta(theta_blind)
    ask_approx, bid_approx = blind_glft_approx_depths(Q)
    visible = q_plot_mask(Q)
    ask_mask = visible & np.isfinite(ask_hjb) & np.isfinite(ask_approx)
    bid_mask = visible & np.isfinite(bid_hjb) & np.isfinite(bid_approx)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    ax[0].plot(Q[ask_mask], ask_hjb[ask_mask], lw=2.5, color=COLORS["Signal-blind finite-grid GLFT"], label="finite-horizon HJB")
    ax[0].plot(Q[ask_mask], ask_approx[ask_mask], "--", lw=2.2, color=COLORS["Signal-blind Gaussian GLFT"], label="GLFT Gaussian approx")
    ax[0].set_title("Signal-blind ask depth")
    ax[1].plot(Q[bid_mask], bid_hjb[bid_mask], lw=2.5, color=COLORS["Signal-blind finite-grid GLFT"], label="finite-horizon HJB")
    ax[1].plot(Q[bid_mask], bid_approx[bid_mask], "--", lw=2.2, color=COLORS["Signal-blind Gaussian GLFT"], label="GLFT Gaussian approx")
    ax[1].set_title("Signal-blind bid depth")
    for a in ax:
        a.axhline(0.0, color="black", lw=1, alpha=0.35)
        a.set_xlim(*q_plot_bounds())
        a.set_xlabel("inventory q")
        a.set_ylabel("signed depth")
        a.grid(True, alpha=0.25)
        a.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cara_glft_blind_depths.png", dpi=180)
    plt.close(fig)

    rows = ["q,ask_hjb,bid_hjb,ask_glft_approx,bid_glft_approx"]
    for row in zip(Q, ask_hjb, bid_hjb, ask_approx, bid_approx):
        rows.append(",".join(str(x) if np.isfinite(x) else "nan" for x in row))
    (OUT / "cara_glft_blind_depths.csv").write_text("\n".join(rows) + "\n")


def plot_y_ansatz_depths(u_y: np.ndarray | None = None) -> None:
    plot_y_ansatz_depths_for_q(0, u_y, "cara_glft_yaware_ansatz_depths_q0")


def plot_y_ansatz_depths_for_q(
    q_value: int,
    u_y: np.ndarray | None = None,
    output_stem: str | None = None,
) -> None:
    yy = np.linspace(Y[0], Y[-1], 301)
    q_arr = np.full_like(yy, int(q_value), dtype=int)
    if u_y is not None:
        ask_hjb, bid_hjb = y_aware_hjb_depths(u_y, q_arr, yy)
    else:
        ask_hjb = None
        bid_hjb = None
    ask_cr, bid_cr = y_curvature_refined_depths(q_arr, yy)
    if output_stem is None:
        q_label = f"qm{abs(q_value)}" if q_value < 0 else f"q{q_value}"
        output_stem = f"cara_glft_yaware_ansatz_depths_{q_label}"

    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    if ask_hjb is not None:
        ax[0].plot(yy, ask_hjb, lw=2.4, color=COLORS["I-aware RHJB"], label="I-aware RHJB")
    ax[0].plot(yy, ask_cr, "--", lw=2.3, color=COLORS["curvature-refined ansatz"], label="tilted-GLFT ansatz")
    ax[0].set_title(f"I-aware ask depth, q={q_value}")

    if bid_hjb is not None:
        ax[1].plot(yy, bid_hjb, lw=2.4, color=COLORS["I-aware RHJB"], label="I-aware RHJB")
    ax[1].plot(yy, bid_cr, "--", lw=2.3, color=COLORS["curvature-refined ansatz"], label="tilted-GLFT ansatz")
    ax[1].set_title(f"I-aware bid depth, q={q_value}")

    for a in ax:
        a.axhline(0.0, color="black", lw=1, alpha=0.35)
        a.axvline(0.0, color="black", lw=0.8, alpha=0.25)
        a.set_xlabel("imbalance state i")
        a.set_ylabel("signed depth")
        a.grid(True, alpha=0.25)
        a.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"{output_stem}.png", dpi=180)
    plt.close(fig)

    rows = ["y,ask_hjb,bid_hjb,ask_cr,bid_cr"]
    if ask_hjb is None or bid_hjb is None:
        ask_hjb = np.full_like(yy, np.nan)
        bid_hjb = np.full_like(yy, np.nan)
    for row in zip(yy, ask_hjb, bid_hjb, ask_cr, bid_cr):
        rows.append(",".join(f"{x:.10g}" for x in row))
    (OUT / f"{output_stem}.csv").write_text("\n".join(rows) + "\n")


def plot_y_ansatz_depths_selected_q(u_y: np.ndarray | None = None) -> None:
    selected_q = [q for q in [0, -5, 10, 15] if -P.qmax <= q <= P.qmax]
    for q_value in selected_q:
        q_label = f"qm{abs(q_value)}" if q_value < 0 else f"q{q_value}"
        plot_y_ansatz_depths_for_q(
            q_value,
            u_y,
            f"cara_glft_yaware_ansatz_depths_{q_label}",
        )


def plot_y_ansatz_depths_vs_inventory(u_y: np.ndarray) -> None:
    y_values = [-3.0, 0.0, 3.0]
    visible = q_plot_mask(Q)
    q_plot = Q[visible]
    fig, ax = plt.subplots(len(y_values), 2, figsize=(12, 9), sharex=True, sharey=True)
    rows = ["y,q,ask_hjb,bid_hjb,ask_cubic,bid_cubic,ask_gal,bid_gal"]

    for row_idx, y_value in enumerate(y_values):
        yy = np.full(q_plot.shape, y_value, dtype=float)
        ask_hjb, bid_hjb = y_aware_hjb_depths(u_y, q_plot, yy)
        ask_cubic, bid_cubic = y_curvature_refined_depths(q_plot, yy)
        ask_gal, bid_gal = y_galerkin_refined_depths(q_plot, yy)

        ax[row_idx, 0].plot(q_plot, ask_hjb, "-", lw=2.6, color=COLORS["I-aware RHJB"], label="I-aware RHJB")
        ax[row_idx, 0].plot(q_plot, ask_cubic, "--", lw=2.5, color=COLORS["I-aware self-consistent cubic"], label="self-consistent cubic")
        ax[row_idx, 0].plot(q_plot, ask_gal, ":", lw=2.3, color=COLORS["I-aware Galerkin-cubic"], label="Galerkin-cubic")
        ax[row_idx, 0].set_title(f"Ask depth, i={y_value:g}")

        ax[row_idx, 1].plot(q_plot, bid_hjb, "-", lw=2.6, color=COLORS["I-aware RHJB"], label="I-aware RHJB")
        ax[row_idx, 1].plot(q_plot, bid_cubic, "--", lw=2.5, color=COLORS["I-aware self-consistent cubic"], label="self-consistent cubic")
        ax[row_idx, 1].plot(q_plot, bid_gal, ":", lw=2.3, color=COLORS["I-aware Galerkin-cubic"], label="Galerkin-cubic")
        ax[row_idx, 1].set_title(f"Bid depth, i={y_value:g}")

        for values in zip(q_plot, ask_hjb, bid_hjb, ask_cubic, bid_cubic, ask_gal, bid_gal):
            rows.append(f"{y_value:.10g}," + ",".join(f"{x:.10g}" for x in values))

    for a in ax.ravel():
        a.axhline(0.0, color="black", lw=1, alpha=0.35)
        a.axvline(0.0, color="black", lw=0.8, alpha=0.25)
        a.set_xlabel("inventory q")
        a.set_ylabel("signed depth")
        a.grid(True, alpha=0.25)
    ax[0, 0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "cara_glft_yaware_ansatz_depths_vs_inventory.png", dpi=180)
    plt.close(fig)
    (OUT / "cara_glft_yaware_ansatz_depths_vs_inventory.csv").write_text("\n".join(rows) + "\n")


def plot_y_ansatz_depths_vs_imbalance(u_y: np.ndarray) -> None:
    q_values = [0, 3, 5]
    fig, ax = plt.subplots(len(q_values), 2, figsize=(12, 9), sharex=True, sharey=True)
    yy = np.linspace(Y[0], Y[-1], 301)
    rows = ["y,q,ask_hjb,bid_hjb,ask_cubic,bid_cubic,ask_gal,bid_gal"]

    for row_idx, q_value in enumerate(q_values):
        q_arr = np.full_like(yy, q_value, dtype=int)
        ask_hjb, bid_hjb = y_aware_hjb_depths(u_y, q_arr, yy)
        ask_cubic, bid_cubic = y_curvature_refined_depths(q_arr, yy)
        ask_gal, bid_gal = y_galerkin_refined_depths(q_arr, yy)

        ax[row_idx, 0].plot(yy, ask_hjb, "-", lw=2.6, color=COLORS["I-aware RHJB"], label="I-aware RHJB")
        ax[row_idx, 0].plot(yy, ask_cubic, "--", lw=2.5, color=COLORS["I-aware self-consistent cubic"], label="self-consistent cubic")
        ax[row_idx, 0].plot(yy, ask_gal, ":", lw=2.3, color=COLORS["I-aware Galerkin-cubic"], label="Galerkin-cubic")
        ax[row_idx, 0].set_title(f"Ask depth, q={q_value}")

        ax[row_idx, 1].plot(yy, bid_hjb, "-", lw=2.6, color=COLORS["I-aware RHJB"], label="I-aware RHJB")
        ax[row_idx, 1].plot(yy, bid_cubic, "--", lw=2.5, color=COLORS["I-aware self-consistent cubic"], label="self-consistent cubic")
        ax[row_idx, 1].plot(yy, bid_gal, ":", lw=2.3, color=COLORS["I-aware Galerkin-cubic"], label="Galerkin-cubic")
        ax[row_idx, 1].set_title(f"Bid depth, q={q_value}")

        for values in zip(yy, q_arr, ask_hjb, bid_hjb, ask_cubic, bid_cubic, ask_gal, bid_gal):
            rows.append(f"{values[0]:.10g}," + ",".join(f"{x:.10g}" for x in values[1:]))

    for a in ax.ravel():
        a.axhline(0.0, color="black", lw=1, alpha=0.35)
        a.axvline(0.0, color="black", lw=0.8, alpha=0.25)
        a.set_xlabel("imbalance state i")
        a.set_ylabel("signed depth")
        a.grid(True, alpha=0.25)
    ax[0, 0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "cara_glft_yaware_ansatz_depths_vs_imbalance.png", dpi=180)
    plt.close(fig)
    (OUT / "cara_glft_yaware_ansatz_depths_vs_imbalance.csv").write_text("\n".join(rows) + "\n")


def plot_affine_body_depths_vs_inventory(u_y: np.ndarray) -> None:
    """Body figure: affine closed-form quotes (only) vs numerical RHJB, depths vs
    inventory at selected imbalance values."""
    y_values = [-3.0, 0.0, 3.0]
    visible = q_plot_mask(Q)
    q_plot = Q[visible]
    fig, ax = plt.subplots(len(y_values), 2, figsize=(12, 9), sharex=True, sharey=True)
    for row_idx, y_value in enumerate(y_values):
        yy = np.full(q_plot.shape, y_value, dtype=float)
        ask_hjb, bid_hjb = y_aware_hjb_depths(u_y, q_plot, yy)
        ask_aff, bid_aff = y_affine_depths(q_plot, yy)
        ax[row_idx, 0].plot(q_plot, ask_hjb, lw=2.4, color=COLORS["I-aware RHJB"], label="I-aware RHJB")
        ax[row_idx, 0].plot(q_plot, ask_aff, "-.", lw=2.2, color=COLORS["I-aware affine ansatz"], label="affine ansatz")
        ax[row_idx, 0].set_title(f"Ask depth, i={y_value:g}")
        ax[row_idx, 1].plot(q_plot, bid_hjb, lw=2.4, color=COLORS["I-aware RHJB"], label="I-aware RHJB")
        ax[row_idx, 1].plot(q_plot, bid_aff, "-.", lw=2.2, color=COLORS["I-aware affine ansatz"], label="affine ansatz")
        ax[row_idx, 1].set_title(f"Bid depth, i={y_value:g}")
    for a in ax.ravel():
        a.axhline(0.0, color="black", lw=1, alpha=0.35)
        a.axvline(0.0, color="black", lw=0.8, alpha=0.25)
        a.set_xlabel("inventory q")
        a.set_ylabel("signed depth")
        a.grid(True, alpha=0.25)
    ax[0, 0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "cara_glft_yaware_affine_depths_vs_inventory.png", dpi=180)
    plt.close(fig)


def plot_affine_body_depths_vs_imbalance(u_y: np.ndarray) -> None:
    """Body figure: affine closed-form quotes (only) vs numerical RHJB, depths vs
    imbalance at selected inventory values."""
    q_values = [q for q in [-5, 0, 10, 15] if -P.qmax < q < P.qmax]
    yy = np.linspace(-4.0, 4.0, 241)
    fig, ax = plt.subplots(len(q_values), 2, figsize=(12, 11), sharex=True)
    for row_idx, q_value in enumerate(q_values):
        q_arr = np.full_like(yy, int(q_value), dtype=int)
        ask_hjb, bid_hjb = y_aware_hjb_depths(u_y, q_arr, yy)
        ask_aff, bid_aff = y_affine_depths(q_arr, yy)
        ax[row_idx, 0].plot(yy, ask_hjb, lw=2.4, color=COLORS["I-aware RHJB"], label="I-aware RHJB")
        ax[row_idx, 0].plot(yy, ask_aff, "-.", lw=2.2, color=COLORS["I-aware affine ansatz"], label="affine ansatz")
        ax[row_idx, 0].set_title(f"Ask depth, q={q_value}")
        ax[row_idx, 1].plot(yy, bid_hjb, lw=2.4, color=COLORS["I-aware RHJB"], label="I-aware RHJB")
        ax[row_idx, 1].plot(yy, bid_aff, "-.", lw=2.2, color=COLORS["I-aware affine ansatz"], label="affine ansatz")
        ax[row_idx, 1].set_title(f"Bid depth, q={q_value}")
    for a in ax.ravel():
        a.axhline(0.0, color="black", lw=1, alpha=0.35)
        a.axvline(0.0, color="black", lw=0.8, alpha=0.25)
        a.set_xlabel("imbalance state i")
        a.set_ylabel("signed depth")
        a.grid(True, alpha=0.25)
    ax[0, 0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "cara_glft_yaware_affine_depths_vs_imbalance.png", dpi=180)
    plt.close(fig)


def plot_y_depth_heatmaps(u_y: np.ndarray) -> None:
    ask, bid = depth_grid_from_u(u_y)
    visible = q_plot_mask(Q)
    q_plot = Q[visible]
    ask = ask[visible, :]
    bid = bid[visible, :]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    extent = [Y[0], Y[-1], q_plot[0], q_plot[-1]]
    vmax = np.nanpercentile(np.abs(np.concatenate([ask[np.isfinite(ask)], bid[np.isfinite(bid)]])), 98)
    im0 = ax[0].imshow(ask, origin="lower", aspect="auto", extent=extent, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    im1 = ax[1].imshow(bid, origin="lower", aspect="auto", extent=extent, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax[0].set_title("I-aware RHJB ask depth")
    ax[1].set_title("I-aware RHJB bid depth")
    for a in ax:
        a.set_xlabel("imbalance state i")
        a.set_ylabel("inventory q")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "cara_glft_yaware_depth_heatmaps.png", dpi=180)
    plt.close(fig)


def lifetime_distributions(
    paths: dict[str, object],
    policies: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    full_centers = np.arange(-P.qmax, P.qmax + 1, dtype=float)
    abs_centers = np.arange(0, P.qmax + 1, dtype=float)
    tail_x = np.arange(0, P.qmax + 1, dtype=float)

    if "lifetime_inventory" in paths:
        lifetime_inventory = paths["lifetime_inventory"]
        full_bins = np.arange(-P.qmax - 0.5, P.qmax + 1.5, 1.0)
        abs_bins = np.arange(-0.5, P.qmax + 1.5, 1.0)
        full_prob: dict[str, np.ndarray] = {}
        abs_prob: dict[str, np.ndarray] = {}
        tail_prob: dict[str, np.ndarray] = {}
        for name in policies:
            values = np.asarray(lifetime_inventory[name], dtype=float)
            denom = max(1, values.size)
            counts, _ = np.histogram(values, bins=full_bins)
            full_prob[name] = counts.astype(float) / denom
            abs_values = np.abs(values)
            abs_counts, _ = np.histogram(abs_values, bins=abs_bins)
            abs_prob[name] = abs_counts.astype(float) / denom
            tail_prob[name] = np.array([np.mean(abs_values > x) for x in tail_x], dtype=float)
        return full_centers, abs_centers, tail_x, full_prob, abs_prob, tail_prob

    lifetime_full_counts = paths["lifetime_full_counts"]
    lifetime_abs_counts = paths["lifetime_abs_counts"]
    full_prob = {}
    abs_prob = {}
    tail_prob = {}
    for name in policies:
        full_counts = np.asarray(lifetime_full_counts[name], dtype=float)
        abs_counts = np.asarray(lifetime_abs_counts[name], dtype=float)
        full_denom = max(1.0, float(np.sum(full_counts)))
        abs_denom = max(1.0, float(np.sum(abs_counts)))
        full_prob[name] = full_counts / full_denom
        abs_prob[name] = abs_counts / abs_denom
        tail_counts = np.array([np.sum(abs_counts[int(x) + 1:]) for x in tail_x], dtype=float)
        tail_prob[name] = tail_counts / abs_denom
    return full_centers, abs_centers, tail_x, full_prob, abs_prob, tail_prob


def plot_pnl_outputs(paths: dict[str, object]) -> None:
    time = paths["time"]
    mean_pnl = paths["mean_pnl"]
    mean_absq = paths["mean_absq"]
    terminal_pnl = paths["terminal_pnl"]
    policies = list(mean_pnl.keys())

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for name in policies:
        ax[0].plot(time, mean_pnl[name], lw=2.2, color=COLORS[name], label=name)
        ax[1].plot(time, mean_absq[name], lw=2.2, color=COLORS[name], label=name)
    ax[0].set_title("Mean accumulated mark-to-market PnL")
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("PnL")
    ax[1].set_title("Mean absolute inventory")
    ax[1].set_xlabel("time")
    ax[1].set_ylabel(r"$E|Q_t|$")
    for a in ax:
        a.grid(True, alpha=0.25)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "cara_glft_cumulative_pnl.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    all_pnl = np.concatenate([np.asarray(terminal_pnl[name], dtype=float) for name in policies])
    lo, hi = np.quantile(all_pnl, [0.005, 0.995])
    pad = 0.08 * (hi - lo)
    x_grid = np.linspace(lo - pad, hi + pad, 500)
    for name in policies:
        values = np.asarray(terminal_pnl[name], dtype=float)
        kde = gaussian_kde(values)
        ax.plot(x_grid, kde(x_grid), lw=2.4, color=COLORS[name], label=name)
    ax.set_title("Terminal PnL density")
    ax.set_xlabel("terminal mark-to-market PnL")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "cara_glft_pnl_hist.png", dpi=180)
    plt.close(fig)

    full_centers, abs_centers, tail_x, full_prob, abs_prob, tail_prob = lifetime_distributions(paths, policies)
    q_min, q_max = q_plot_bounds()
    central_mask = (full_centers >= q_min) & (full_centers <= q_max)

    fig, ax = plt.subplots(2, 2, figsize=(13, 8.2))
    for name in policies:
        ax[0, 0].plot(full_centers, full_prob[name], lw=2.0, color=COLORS[name], label=name)
        ax[0, 1].plot(full_centers[central_mask], full_prob[name][central_mask], lw=2.0, color=COLORS[name], label=name)
        ax[1, 0].plot(abs_centers, abs_prob[name], lw=2.0, color=COLORS[name], label=name)
        ax[1, 1].plot(tail_x, tail_prob[name], lw=2.0, color=COLORS[name], label=name)

    ax[0, 0].set_title("Lifetime inventory probability, full range")
    ax[0, 0].set_xlabel("inventory q")
    ax[0, 0].set_ylabel("probability per inventory")
    ax[0, 0].set_xlim(-P.qmax, P.qmax)

    ax[0, 1].set_title("Central zoom, no renormalization")
    ax[0, 1].set_xlabel("inventory q")
    ax[0, 1].set_ylabel("probability per inventory")
    ax[0, 1].set_xlim(q_min, q_max)

    ax[1, 0].set_title("Absolute inventory probability")
    ax[1, 0].set_xlabel(r"$|Q|$")
    ax[1, 0].set_ylabel("probability")
    ax[1, 0].set_xlim(0, P.qmax)

    ax[1, 1].set_title("Inventory tail probability")
    ax[1, 1].set_xlabel(r"$x$")
    ax[1, 1].set_ylabel(r"$P(|Q| > x)$")
    ax[1, 1].set_xlim(0, P.qmax)
    ax[1, 1].set_ylim(bottom=0.0)

    for a in ax.ravel():
        a.grid(True, alpha=0.25)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "cara_glft_lifetime_inventory_hist.png", dpi=180)
    plt.close(fig)

    lines = ["inventory_q," + ",".join(policies)]
    for i, q_value in enumerate(full_centers):
        lines.append(f"{q_value:.0f}," + ",".join(f"{full_prob[name][i]:.10g}" for name in policies))
    (OUT / "cara_glft_lifetime_inventory_full_prob.csv").write_text("\n".join(lines) + "\n")

    lines = ["abs_inventory," + ",".join(policies)]
    for i, q_value in enumerate(abs_centers):
        lines.append(f"{q_value:.0f}," + ",".join(f"{abs_prob[name][i]:.10g}" for name in policies))
    (OUT / "cara_glft_abs_inventory_prob.csv").write_text("\n".join(lines) + "\n")

    lines = ["threshold_x," + ",".join(policies)]
    for i, x_value in enumerate(tail_x):
        lines.append(f"{x_value:.0f}," + ",".join(f"{tail_prob[name][i]:.10g}" for name in policies))
    (OUT / "cara_glft_inventory_tail_prob.csv").write_text("\n".join(lines) + "\n")


def plot_glft_vs_yaware_comparison(paths: dict[str, object]) -> None:
    """Dedicated comparison requested for the paper: GLFT blind vs I-aware RHJB."""
    pair = [
        ("Signal-blind Gaussian GLFT", "Signal-blind Gaussian GLFT"),
        ("I-aware RHJB", "I-aware RHJB"),
    ]
    color = {
        "Signal-blind Gaussian GLFT": COLORS["Signal-blind Gaussian GLFT"],
        "I-aware RHJB": COLORS["I-aware RHJB"],
    }

    time = np.asarray(paths["time"], dtype=float)
    mean_pnl = paths["mean_pnl"]
    mean_penalty = paths["mean_penalty"]
    terminal_pnl = paths["terminal_pnl"]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for key, label in pair:
        ax[0].plot(time, mean_pnl[key], lw=2.5, color=color[key], label=label)
        ax[1].plot(time, mean_penalty[key], lw=2.5, color=color[key], label=label)
    ax[0].set_title("Accumulated mark-to-market PnL")
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("mean PnL")
    ax[1].set_title(r"Accumulated running inventory penalty")
    ax[1].set_xlabel("time")
    ax[1].set_ylabel(r"mean $\int_0^t \frac{\gamma\sigma^2}{2}Q_s^2\,ds$")
    for a in ax:
        a.grid(True, alpha=0.25)
        a.legend()
    fig.tight_layout()
    fig.savefig(OUT / "toy_glft_vs_yaware_cumulative_pnl_penalty.png", dpi=180)
    plt.close(fig)

    lines = ["time,glft_blind_mean_pnl,yaware_hjb_mean_pnl,glft_blind_mean_penalty,yaware_hjb_mean_penalty"]
    for i, t in enumerate(time):
        lines.append(
            f"{t:.10g},"
            f"{mean_pnl['Signal-blind Gaussian GLFT'][i]:.10g},"
            f"{mean_pnl['I-aware RHJB'][i]:.10g},"
            f"{mean_penalty['Signal-blind Gaussian GLFT'][i]:.10g},"
            f"{mean_penalty['I-aware RHJB'][i]:.10g}"
        )
    (OUT / "toy_glft_vs_yaware_cumulative_pnl_penalty.csv").write_text("\n".join(lines) + "\n")

    all_pnl = np.concatenate([np.asarray(terminal_pnl[key], dtype=float) for key, _ in pair])
    lo, hi = np.quantile(all_pnl, [0.005, 0.995])
    pad = 0.08 * max(1.0, hi - lo)
    bins = np.linspace(lo - pad, hi + pad, 90)
    centers = 0.5 * (bins[:-1] + bins[1:])

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    hist_rows: dict[str, np.ndarray] = {}
    for key, label in pair:
        values = np.asarray(terminal_pnl[key], dtype=float)
        counts, _ = np.histogram(values, bins=bins)
        prob = counts.astype(float) / max(1, values.size)
        hist_rows[key] = prob
        ax.plot(centers, prob, lw=2.4, drawstyle="steps-mid", color=color[key], label=label)
    ax.set_title("Terminal PnL histogram")
    ax.set_xlabel("terminal mark-to-market PnL")
    ax.set_ylabel("probability per bin")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "toy_glft_vs_yaware_pnl_hist.png", dpi=180)
    plt.close(fig)

    lines = ["pnl_center,glft_blind,yaware_hjb"]
    for i, center in enumerate(centers):
        lines.append(f"{center:.10g},{hist_rows['Signal-blind Gaussian GLFT'][i]:.10g},{hist_rows['I-aware RHJB'][i]:.10g}")
    (OUT / "toy_glft_vs_yaware_pnl_hist.csv").write_text("\n".join(lines) + "\n")

    inv_centers, abs_centers, _, signed_prob, abs_prob, _ = lifetime_distributions(paths, [key for key, _ in pair])

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for key, label in pair:
        ax[0].plot(inv_centers, signed_prob[key], lw=2.2, drawstyle="steps-mid", color=color[key], label=label)
        ax[1].plot(abs_centers, abs_prob[key], lw=2.2, drawstyle="steps-mid", color=color[key], label=label)
    ax[0].set_title("Lifetime inventory histogram")
    ax[0].set_xlabel("inventory q")
    ax[0].set_ylabel("probability per inventory")
    ax[0].set_xlim(-P.qmax, P.qmax)
    ax[1].set_title("Lifetime absolute inventory histogram")
    ax[1].set_xlabel(r"$|Q|$")
    ax[1].set_ylabel("probability")
    ax[1].set_xlim(0, P.qmax)
    for a in ax:
        a.grid(True, alpha=0.25)
        a.legend()
    fig.tight_layout()
    fig.savefig(OUT / "toy_glft_vs_yaware_inventory_hist.png", dpi=180)
    plt.close(fig)

    lines = ["inventory_q,glft_blind,yaware_hjb"]
    for i, q_value in enumerate(inv_centers):
        lines.append(f"{q_value:.0f},{signed_prob['Signal-blind Gaussian GLFT'][i]:.10g},{signed_prob['I-aware RHJB'][i]:.10g}")
    (OUT / "toy_glft_vs_yaware_inventory_hist.csv").write_text("\n".join(lines) + "\n")

    lines = ["abs_inventory,glft_blind,yaware_hjb"]
    for i, q_value in enumerate(abs_centers):
        lines.append(f"{q_value:.0f},{abs_prob['Signal-blind Gaussian GLFT'][i]:.10g},{abs_prob['I-aware RHJB'][i]:.10g}")
    (OUT / "toy_glft_vs_yaware_abs_inventory_hist.csv").write_text("\n".join(lines) + "\n")


def plot_glft_fair_vs_imbalance(paths: dict[str, object]) -> None:
    fair = simulate_glft_fair_environment()
    inv_centers_stream, _, _, full_prob_stream, abs_prob_stream, _ = lifetime_distributions(paths, ["Signal-blind Gaussian GLFT"])
    imbalanced = {
        "terminal_pnl": np.asarray(paths["terminal_pnl"]["Signal-blind Gaussian GLFT"], dtype=float),
        "lifetime_full_prob": full_prob_stream["Signal-blind Gaussian GLFT"],
        "lifetime_abs_prob": abs_prob_stream["Signal-blind Gaussian GLFT"],
    }
    scenarios = {
        "GLFT, fair environment": fair,
        "GLFT, imbalanced I-environment": imbalanced,
    }
    colors = {
        "GLFT, fair environment": "#4d4d4d",
        "GLFT, imbalanced I-environment": COLORS["Signal-blind Gaussian GLFT"],
    }

    inv_bins = np.arange(-P.qmax - 0.5, P.qmax + 1.5, 1.0)
    inv_centers = 0.5 * (inv_bins[:-1] + inv_bins[1:])
    all_pnl = np.concatenate([scenarios[name]["terminal_pnl"] for name in scenarios])
    pnl_lo, pnl_hi = float(np.min(all_pnl)), float(np.max(all_pnl))
    pnl_pad = 0.05 * max(1.0, pnl_hi - pnl_lo)
    pnl_bins = np.linspace(pnl_lo - pnl_pad, pnl_hi + pnl_pad, 80)
    pnl_centers = 0.5 * (pnl_bins[:-1] + pnl_bins[1:])

    inv_prob: dict[str, np.ndarray] = {}
    pnl_prob: dict[str, np.ndarray] = {}
    for name, data in scenarios.items():
        pnl = data["terminal_pnl"]
        pnl_counts, _ = np.histogram(pnl, bins=pnl_bins)
        if "lifetime_inventory" in data:
            inv = data["lifetime_inventory"]
            inv_counts, _ = np.histogram(inv, bins=inv_bins)
            inv_prob[name] = inv_counts.astype(float) / max(1, inv.size)
        else:
            inv_prob[name] = np.interp(inv_centers, inv_centers_stream, data["lifetime_full_prob"], left=0.0, right=0.0)
        pnl_prob[name] = pnl_counts.astype(float) / max(1, pnl.size)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for name in scenarios:
        ax[0].plot(inv_centers, inv_prob[name], lw=2.2, drawstyle="steps-mid", color=colors[name], label=name)
        ax[1].plot(pnl_centers, pnl_prob[name], lw=2.2, drawstyle="steps-mid", color=colors[name], label=name)

    ax[0].set_title("GLFT lifetime inventory")
    ax[0].set_xlabel("inventory q")
    ax[0].set_ylabel("probability per inventory")
    ax[0].set_xlim(-P.qmax, P.qmax)

    ax[1].set_title("GLFT terminal PnL")
    ax[1].set_xlabel("terminal mark-to-market PnL")
    ax[1].set_ylabel("probability per bin")

    for a in ax:
        a.grid(True, alpha=0.25)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "cara_glft_fair_vs_imbalance_histograms.png", dpi=180)
    plt.close(fig)

    import csv as _csv
    with open(OUT / "cara_glft_fair_vs_imbalance_inventory.csv", "w", newline="") as _fh:
        _w = _csv.writer(_fh)
        _w.writerow(["inventory_q", *scenarios.keys()])
        for i, q_value in enumerate(inv_centers):
            _w.writerow([f"{q_value:.0f}", *(f"{inv_prob[name][i]:.10g}" for name in scenarios)])

    lines = ["pnl_center," + ",".join(scenarios.keys())]
    for i, pnl_value in enumerate(pnl_centers):
        lines.append(f"{pnl_value:.10g}," + ",".join(f"{pnl_prob[name][i]:.10g}" for name in scenarios))
    (OUT / "cara_glft_fair_vs_imbalance_pnl.csv").write_text("\n".join(lines) + "\n")

    summary_lines = ["scenario,terminal_pnl_mean,terminal_pnl_std,avg_abs_lifetime_inventory,lifetime_q01,lifetime_q99"]
    for name, data in scenarios.items():
        pnl = data["terminal_pnl"]
        if "lifetime_inventory" in data:
            inv = data["lifetime_inventory"]
            avg_abs_inv = float(np.mean(np.abs(inv)))
            q01 = float(np.quantile(inv, 0.01))
            q99 = float(np.quantile(inv, 0.99))
        else:
            full_prob = data["lifetime_full_prob"]
            avg_abs_inv = float(np.sum(np.abs(inv_centers_stream) * full_prob))
            q01 = quantile_from_counts(inv_centers_stream, full_prob, 0.01)
            q99 = quantile_from_counts(inv_centers_stream, full_prob, 0.99)
        summary_lines.append(
            f"{name},{np.mean(pnl):.10g},{np.std(pnl):.10g},"
            f"{avg_abs_inv:.10g},{q01:.10g},{q99:.10g}"
        )
    (OUT / "cara_glft_fair_vs_imbalance_summary.csv").write_text("\n".join(summary_lines) + "\n")


def plot_expected_inventory_vs_y(paths: dict[str, object]) -> None:
    policies = [
        "Signal-blind Gaussian GLFT",
        "I-aware RHJB",
        "I-aware self-consistent cubic",
        "I-aware Galerkin-cubic",
    ]
    labels = {name: name for name in policies}

    conditional_means: dict[str, np.ndarray] = {}
    if "conditional_y_counts" in paths:
        centers = np.asarray(paths["conditional_y_centers"], dtype=float)
        counts = np.asarray(paths["conditional_y_counts"], dtype=float)
        conditional_q_sums = paths["conditional_q_sums"]
        for name in policies:
            weighted = np.asarray(conditional_q_sums[name], dtype=float)
            conditional_means[name] = np.divide(
                weighted,
                counts,
                out=np.full_like(centers, np.nan, dtype=float),
                where=counts > 0,
            )
    else:
        y_values = np.asarray(paths["lifetime_y"], dtype=float)
        lifetime_inventory = paths["lifetime_inventory"]
        bins = np.linspace(-P.y_max, P.y_max, 33)
        centers = 0.5 * (bins[:-1] + bins[1:])
        bin_idx = np.digitize(y_values, bins) - 1
        valid = (bin_idx >= 0) & (bin_idx < len(centers))
        counts = np.bincount(bin_idx[valid], minlength=len(centers)).astype(float)
        for name in policies:
            q_values = np.asarray(lifetime_inventory[name], dtype=float)
            weighted = np.bincount(bin_idx[valid], weights=q_values[valid], minlength=len(centers))
            conditional_means[name] = np.divide(
                weighted,
                counts,
                out=np.full_like(centers, np.nan, dtype=float),
                where=counts > 0,
            )

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for name in policies:
        ax.plot(
            centers,
            conditional_means[name],
            marker="o",
            markersize=4.0,
            lw=2.3,
            color=COLORS[name],
            label=labels[name],
        )
    ax.axhline(0.0, color="black", lw=1.0, alpha=0.35)
    ax.axvline(0.0, color="black", lw=1.0, alpha=0.25)
    ax.set_title("Conditional mean inventory by imbalance")
    ax.set_xlabel(r"imbalance state $I_t$")
    ax.set_ylabel(r"$\mathbb{E}[Q_t\mid I_t]$")
    ax.set_xlim(-P.y_max, P.y_max)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cara_glft_expected_inventory_vs_y.png", dpi=180)
    plt.close(fig)

    lines = ["y_center,count," + ",".join(name.replace(" ", "_") for name in policies)]
    for i, center in enumerate(centers):
        lines.append(
            f"{center:.10g},{int(counts[i])},"
            + ",".join(f"{conditional_means[name][i]:.10g}" for name in policies)
        )
    (OUT / "cara_glft_expected_inventory_vs_y.csv").write_text("\n".join(lines) + "\n")


def write_outputs(metrics: dict[str, dict[str, float]], paths: dict[str, object], blind_info: dict[str, float], y_info: dict[str, float]) -> None:
    payload = {
        "params": asdict(P),
        "derived": {
            "delta_gamma": P.delta_gamma,
            "chi_gamma": P.chi_gamma,
            "alpha_c": P.alpha_c,
            "eta_c": P.eta_c,
            "p_glft": P.p_glft,
            "rho": P.rho,
            "loading_amplitude_LO": P.A_I,
            "loading_scale_B_LO": P.beta / P.eta**2,
            "loading_galerkin_A": A_GAL,
            "loading_galerkin_c": C_GAL,
        },
        "blind_hjb": blind_info,
        "y_aware_hjb": y_info,
        "metrics": metrics,
    }
    (OUT / "cara_glft_summary.json").write_text(json.dumps(payload, indent=2))

    policies = list(metrics.keys())
    metric_keys = list(next(iter(metrics.values())).keys())
    lines = ["policy," + ",".join(metric_keys)]
    for name in policies:
        lines.append(name + "," + ",".join(f"{metrics[name][key]:.10g}" for key in metric_keys))
    (OUT / "cara_glft_summary.csv").write_text("\n".join(lines) + "\n")

    time = paths["time"]
    mean_pnl = paths["mean_pnl"]
    mean_penalty = paths["mean_penalty"]
    lines = ["time," + ",".join(policies)]
    for i, t in enumerate(time):
        lines.append(f"{t:.10g}," + ",".join(f"{mean_pnl[name][i]:.10g}" for name in policies))
    (OUT / "cara_glft_cumulative_pnl.csv").write_text("\n".join(lines) + "\n")

    lines = ["time," + ",".join(policies)]
    for i, t in enumerate(time):
        lines.append(f"{t:.10g}," + ",".join(f"{mean_penalty[name][i]:.10g}" for name in policies))
    (OUT / "cara_glft_cumulative_penalty.csv").write_text("\n".join(lines) + "\n")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-surfaces", action="store_true",
                    help="Skip the loading-independent 3D GLFT sensitivity surfaces "
                         "(the ~10h bottleneck); regenerate only loading-dependent figures + table.")
    args = ap.parse_args()
    print("Parameters:")
    print(json.dumps(asdict(P), indent=2))
    print("Derived CARA/GLFT coefficients:")
    print(json.dumps({
        "delta_gamma": P.delta_gamma,
        "chi_gamma": P.chi_gamma,
        "alpha_c": P.alpha_c,
        "eta_c": P.eta_c,
        "p_glft": P.p_glft,
        "rho": P.rho,
        "loading_amplitude_LO": P.A_I,
        "loading_scale_B_LO": P.beta / P.eta**2,
        "loading_galerkin_A": A_GAL,
        "loading_galerkin_c": C_GAL,
    }, indent=2))

    print("Solving signal-blind finite-horizon GLFT HJB...")
    theta_blind, blind_info = solve_blind_hjb()
    print("Solving I-aware GLFT-reduced HJB...")
    u_y, y_info = solve_y_aware_hjb()
    print("Running Monte Carlo simulation...")
    metrics, paths = simulate_policies(theta_blind, u_y)

    print("Writing plots and tables...")
    plot_blind_depths(theta_blind)
    plot_y_ansatz_depths_vs_imbalance(u_y)
    plot_y_ansatz_depths_vs_inventory(u_y)
    plot_y_depth_heatmaps(u_y)
    if args.skip_surfaces:
        print("Skipping loading-independent 3D GLFT sensitivity surfaces (--skip-surfaces).")
    else:
        plot_glft_fig4_fig5_exact_vs_approx()
        plot_glft_style_surfaces()
        plot_glft_sensitivity_surfaces()
    plot_pnl_outputs(paths)
    plot_glft_vs_yaware_comparison(paths)
    plot_glft_fair_vs_imbalance(paths)
    plot_expected_inventory_vs_y(paths)
    write_outputs(metrics, paths, blind_info, y_info)

    print(f"Outputs written to: {OUT}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
