#!/usr/bin/env python3
"""Frozen out-of-sample validation of the regime asinh closure.

No parameter is selected inside this campaign.  The candidate is fixed as

    X = 1 + asinh(M - 1),
    H = X h_I + X**g h_alpha + A_R z_r,

with the regime channel ungated, the original curvature r = r_G M, and the
original self-consistent cubic.  Each holdout is compared with the numerical
regime RHJB on a +/-3 sigma_I evaluation band after solving on +/-3.5 sigma_I.

Two state measures are reported:
  * exact_discrete: invariant mass from the adjoint of the discretized joint
    (I,R) generator used by the RHJB;
  * gaussian_proxy: shifted-Gaussian proxy used by the earlier campaign.

Inventory slices are scaled by the GLFT curvature through
round((-1.2, 0, 2.4, 3.4) / sqrt(r_G)).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigs


ROOT = Path(__file__).resolve().parent.parent
ENGINE_DIR = ROOT / "regime_studies"
OUT_CSV = ROOT / "rerun_campaign" / "csv" / "asinh_holdout_regime.csv"
OUT_LOG = ROOT / "rerun_campaign" / "logs" / "asinh_holdout_regime.log"

BASE = {
    "beta": 0.0125,
    "eta": 0.32,
    "z_star": 1.25,
    "tau": 120.0,
    "zeta": 0.25,
    "bar_lambda": 0.9,
    "gamma": 0.01,
}

CASES = [
    ("baseline", {}),
    ("eta_0.16", {"eta": 0.16}),
    ("eta_0.64", {"eta": 0.64}),
    ("zstar_0.625", {"z_star": 0.625}),
    ("zstar_2.5", {"z_star": 2.5}),
    ("zeta_0.125", {"zeta": 0.125}),
    ("zeta_0.5", {"zeta": 0.5}),
    ("barlambda_0.45", {"bar_lambda": 0.45}),
    ("barlambda_1.8", {"bar_lambda": 1.8}),
    ("gamma_0.005", {"gamma": 0.005}),
    ("gamma_0.02", {"gamma": 0.02}),
    ("corner_betaHalf_zstar2.5", {"beta": 0.00625, "z_star": 2.5}),
    ("corner_betaHalf_tau240", {"beta": 0.00625, "tau": 240.0}),
    ("corner_eta0.64_zstar2.5", {"eta": 0.64, "z_star": 2.5}),
    ("corner_zeta0.125_barlambda1.8", {"zeta": 0.125, "bar_lambda": 1.8}),
    ("corner_gamma0.02_tau60", {"gamma": 0.02, "tau": 60.0}),
]

Q_FACTORS = (-1.2, 0.0, 2.4, 3.4)


def set_engine_environment(params: dict[str, float]) -> tuple[float, int, float]:
    sigma_i = params["eta"] / np.sqrt(2.0 * params["beta"])
    y_max = round(3.5 * sigma_i, 2)
    # Keep approximately dy=0.1 and force an odd grid so that i=0 is a node.
    ny = int(np.ceil(2.0 * y_max / 0.1)) + 1
    if ny % 2 == 0:
        ny += 1

    env = {
        "ASOU5_REGIME_QMAX": "60",
        "ASOU5_REGIME_Y_MAX": str(y_max),
        "ASOU5_REGIME_NY": str(ny),
        "ASOU5_REGIME_DT_HJB": "0.25",
        "ASOU5_REGIME_GAMMA": str(params["gamma"]),
        "ASOU5_REGIME_BAR_LAMBDA": str(params["bar_lambda"]),
        "ASOU5_REGIME_K": "2.0",
        "ASOU5_REGIME_BETA": str(params["beta"]),
        "ASOU5_REGIME_ETA": str(params["eta"]),
        "ASOU5_REGIME_PHI": "-1.0",
        "ASOU5_REGIME_ZETA": str(params["zeta"]),
        "ASOU5_REGIME_EPSILON": "0.005",
        "ASOU5_REGIME_ETA_ALPHA": "0.001",
        "ASOU5_REGIME_SIGMA_PRICE": "0.3",
        "ASOU5_REGIME_Z_STAR": str(params["z_star"]),
        "ASOU5_REGIME_TAU": str(params["tau"]),
        "ASOU5_REGIME_ERGODIC_MAX_TAU": "2500.0",
        "ASOU5_REGIME_ERGODIC_TOL": "2e-8",
    }
    os.environ.update(env)
    return sigma_i, ny, y_max


def fresh_engine():
    sys.modules.pop("regime_fast_alpha_ergodic_cole_hopf", None)
    if str(ENGINE_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_DIR))
    import regime_fast_alpha_ergodic_cole_hopf as engine

    return engine


def joint_generator(engine):
    """Sparse copy of the joint finite-difference generator used by the engine."""
    n = engine.NR * engine.P.ny
    generator = lil_matrix((n, n), dtype=float)

    for r_idx, z_level in enumerate(engine.Z_LEVELS):
        drift = -engine.P.beta * (engine.Y - z_level)
        offset = r_idx * engine.P.ny
        for j in range(engine.P.ny):
            row = offset + j
            if j == 0:
                generator[row, offset + j] += -drift[j] / engine.DY - engine.P.eta**2 / engine.DY**2
                generator[row, offset + j + 1] += drift[j] / engine.DY + engine.P.eta**2 / engine.DY**2
            elif j == engine.P.ny - 1:
                generator[row, offset + j - 1] += -drift[j] / engine.DY + engine.P.eta**2 / engine.DY**2
                generator[row, offset + j] += drift[j] / engine.DY - engine.P.eta**2 / engine.DY**2
            else:
                generator[row, offset + j - 1] += -0.5 * drift[j] / engine.DY + 0.5 * engine.P.eta**2 / engine.DY**2
                generator[row, offset + j] += -engine.P.eta**2 / engine.DY**2
                generator[row, offset + j + 1] += 0.5 * drift[j] / engine.DY + 0.5 * engine.P.eta**2 / engine.DY**2

    for r_idx in range(engine.NR):
        for rp_idx in range(engine.NR):
            rate = engine.Q_REGIME[r_idx, rp_idx]
            if rate == 0.0:
                continue
            for j in range(engine.P.ny):
                generator[r_idx * engine.P.ny + j, rp_idx * engine.P.ny + j] += rate

    return generator.tocsc()


def exact_joint_mass(engine) -> tuple[np.ndarray, float]:
    generator = joint_generator(engine)
    eigenvalues, eigenvectors = eigs(
        generator.T,
        k=1,
        sigma=1e-10,
        tol=1e-11,
        maxiter=100_000,
    )
    mass = np.real(eigenvectors[:, 0])
    if np.sum(mass) < 0.0:
        mass = -mass
    min_mass = float(np.min(mass))
    if min_mass < -1e-8:
        raise RuntimeError(f"Adjoint invariant vector has material negative mass: {min_mass}")
    mass = np.maximum(mass, 0.0)
    mass /= np.sum(mass)
    residual = float(np.max(np.abs(generator.T @ mass)))
    if abs(float(np.real(eigenvalues[0]))) > 1e-7:
        raise RuntimeError(f"Stationary eigenvalue is not zero: {eigenvalues[0]}")
    return mass.reshape(engine.NR, engine.P.ny), residual


def gaussian_proxy_mass(engine, sigma_i: float) -> np.ndarray:
    stationary_regime = np.array([0.25, 0.5, 0.25])
    c = engine.P.beta / (engine.P.beta + engine.P.regime_rate)
    weights = np.empty((engine.NR, engine.P.ny))
    for r_idx in range(engine.NR):
        center = c * engine.Z_LEVELS[r_idx]
        weights[r_idx] = stationary_regime[r_idx] * np.exp(
            -0.5 * ((engine.Y - center) / sigma_i) ** 2
        )
    weights /= np.sum(weights)
    return weights


def asinh_depths(engine, q, y, regime_idx):
    q = np.asarray(q, dtype=float)
    y = np.asarray(y, dtype=float)
    regime_idx = np.asarray(regime_idx, dtype=int)
    z = engine.Z_LEVELS[regime_idx]

    h_i = engine._A1 * np.tanh(engine._A2 * y)
    theta_i = y - engine.P.k * h_i
    m_raw = np.sqrt(np.cosh(y) / np.cosh(theta_i))
    x = 1.0 + np.arcsinh(m_raw - 1.0)
    g = np.tanh(engine._A2 * y) ** 2
    h_alpha = engine.A_ALPHA_LOAD * np.tanh(y)
    shift = x * h_i + np.power(x, g) * h_alpha + engine.A_REGIME_LOAD * z

    r = 2.0 * engine.P.p_glft * m_raw
    theta_full = y - engine.P.k * shift
    s = (r**2 / 6.0) * np.tanh(theta_full)
    ask = (
        engine.P.delta0
        + shift
        - (r / (2.0 * engine.P.k)) * (2.0 * q - 1.0)
        + (s / engine.P.k) * (q**2 - q + 1.0 / 3.0)
    )
    bid = (
        engine.P.delta0
        - shift
        + (r / (2.0 * engine.P.k)) * (2.0 * q + 1.0)
        - (s / engine.P.k) * (q**2 + q + 1.0 / 3.0)
    )
    return ask, bid


def metric(engine, u, depths, q_grid, state_mass, eval_mask):
    errors: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    y = engine.Y[eval_mask]

    for r_idx in range(engine.NR):
        w = state_mass[r_idx, eval_mask]
        regime_idx = np.full_like(y, r_idx, dtype=int)
        for q_value in q_grid:
            q_float = np.full_like(y, q_value, dtype=float)
            q_int = np.full_like(y, q_value, dtype=int)
            ask, bid = depths(q_float, y, regime_idx)
            ask_rh, bid_rh = engine.regime_aware_depths(u, q_int, y, regime_idx)
            for estimate, target in ((ask, ask_rh), (bid, bid_rh)):
                valid = np.isfinite(estimate) & np.isfinite(target)
                errors.append((estimate - target)[valid])
                targets.append(target[valid])
                weights.append(w[valid])

    error = np.concatenate(errors)
    target = np.concatenate(targets)
    weight = np.concatenate(weights)
    abs_rmse = float(np.sqrt(np.sum(weight * error**2) / np.sum(weight)))
    target_rms = float(np.sqrt(np.sum(weight * target**2) / np.sum(weight)))
    return abs_rmse, target_rms, abs_rmse / target_rms


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "changes",
        "variant",
        "beta",
        "eta",
        "z_star",
        "tau",
        "zeta",
        "bar_lambda",
        "gamma",
        "sigma_i",
        "solve_ymax",
        "ny",
        "dy",
        "q_grid",
        "abs_rmse_exact",
        "target_rms_exact",
        "rel_rmse_exact",
        "abs_rmse_proxy",
        "target_rms_proxy",
        "rel_rmse_proxy",
        "eval_mass_exact",
        "stationary_residual",
        "solve_tau",
        "solve_steps",
        "last_depth_diff",
        "converged",
    ]

    with OUT_CSV.open("w", newline="") as csv_file, OUT_LOG.open("w", buffering=1) as log_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for case_name, changes in CASES:
            params = BASE | changes
            sigma_i, ny, y_max = set_engine_environment(params)
            engine = fresh_engine()
            u, solve_info = engine.solve_regime_ergodic()
            exact_mass, stationary_residual = exact_joint_mass(engine)
            proxy_mass = gaussian_proxy_mass(engine, sigma_i)
            eval_mask = np.abs(engine.Y) <= 3.0 * sigma_i + 1e-12

            r_g = 2.0 * engine.P.p_glft
            q_grid = tuple(dict.fromkeys(int(round(f / np.sqrt(r_g))) for f in Q_FACTORS))
            if any(abs(q) >= engine.P.qmax for q in q_grid):
                raise RuntimeError(f"Evaluation q grid reaches inventory boundary: {q_grid}")

            variants = {
                "current": engine.regime_sc_cubic_depths,
                "asinh": lambda q, y, r: asinh_depths(engine, q, y, r),
            }
            log_file.write(
                f"{case_name}: params={params}, sigmaI={sigma_i:.9g}, "
                f"domain=[{-y_max:.6g},{y_max:.6g}], ny={ny}, q={q_grid}, "
                f"solve={solve_info}, stationary_residual={stationary_residual:.3e}\n"
            )

            for variant_name, depth_function in variants.items():
                exact = metric(engine, u, depth_function, q_grid, exact_mass, eval_mask)
                proxy = metric(engine, u, depth_function, q_grid, proxy_mass, eval_mask)
                row = {
                    "case": case_name,
                    "changes": ";".join(f"{key}={value}" for key, value in changes.items()),
                    "variant": variant_name,
                    **params,
                    "sigma_i": sigma_i,
                    "solve_ymax": y_max,
                    "ny": ny,
                    "dy": engine.DY,
                    "q_grid": " ".join(map(str, q_grid)),
                    "abs_rmse_exact": exact[0],
                    "target_rms_exact": exact[1],
                    "rel_rmse_exact": exact[2],
                    "abs_rmse_proxy": proxy[0],
                    "target_rms_proxy": proxy[1],
                    "rel_rmse_proxy": proxy[2],
                    "eval_mass_exact": float(np.sum(exact_mass[:, eval_mask])),
                    "stationary_residual": stationary_residual,
                    "solve_tau": solve_info["tau"],
                    "solve_steps": solve_info["steps"],
                    "last_depth_diff": solve_info["last_depth_diff"],
                    "converged": int(bool(solve_info["converged"])),
                }
                writer.writerow(row)
                csv_file.flush()
                log_file.write(
                    f"  {variant_name:7s} exact abs/target/rel="
                    f"{exact[0]:.6f}/{exact[1]:.6f}/{100.0 * exact[2]:.3f}% "
                    f"proxy={proxy[0]:.6f}/{proxy[1]:.6f}/{100.0 * proxy[2]:.3f}%\n"
                )


if __name__ == "__main__":
    main()
