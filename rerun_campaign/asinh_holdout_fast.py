#!/usr/bin/env python3
"""Frozen out-of-sample validation of the fast-alpha asinh transport.

The candidate is fixed before looking at the holdouts:

    x = M - 1,
    X = 1 + asinh(x),
    H = X h_I + X**g h_alpha,
    r = r_G M,
    s = r**2 tanh(i - k H) / 6.

Each numerical SRN target is solved on ``[-3.5 sigma_I, 3.5 sigma_I]``
with an approximately 0.1 grid spacing and evaluated on
``[-3 sigma_I, 3 sigma_I]``.  Inventory states are fixed multiples of the
curvature scale ``1/sqrt(r_G)`` and imbalance states receive stationary OU
weights.  No candidate constants are tuned in this script.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "rerun_campaign" / "csv" / "asinh_holdout_fast.csv"
ALPHA_DIR = ROOT / "alpha_studies"
if str(ALPHA_DIR) not in sys.path:
    sys.path.insert(0, str(ALPHA_DIR))


BASE = {
    "beta": 0.0125,
    "eta": 0.32,
    "zeta": 0.25,
    "epsilon": 0.005,
    "bar_lambda": 0.9,
    "k": 2.0,
    "gamma": 0.01,
    "sigma_price": 0.3,
}

CASES = [
    ("baseline", {}),
    ("eta_0.16", {"eta": 0.16}),
    ("eta_0.64", {"eta": 0.64}),
    ("zeta_0.125", {"zeta": 0.125}),
    ("zeta_0.5", {"zeta": 0.5}),
    ("bar_lambda_0.45", {"bar_lambda": 0.45}),
    ("bar_lambda_1.8", {"bar_lambda": 1.8}),
    ("sigma_price_0.15", {"sigma_price": 0.15}),
    ("sigma_price_0.6", {"sigma_price": 0.6}),
    ("corner_beta_0.00625_eta_0.64", {"beta": 0.00625, "eta": 0.64}),
    ("corner_beta_0.025_eta_0.16", {"beta": 0.025, "eta": 0.16}),
    ("corner_eta_0.64_zeta_0.125", {"eta": 0.64, "zeta": 0.125}),
    (
        "corner_bar_lambda_1.8_epsilon_0.01",
        {"bar_lambda": 1.8, "epsilon": 0.01},
    ),
    ("corner_gamma_0.02_sigma_price_0.6", {"gamma": 0.02, "sigma_price": 0.6}),
]

ENV_NAMES = {
    "beta": "ASOU5_ALPHA_BETA",
    "eta": "ASOU5_ALPHA_ETA",
    "zeta": "ASOU5_ALPHA_ZETA",
    "epsilon": "ASOU5_ALPHA_EPSILON",
    "bar_lambda": "ASOU5_ALPHA_BAR_LAMBDA",
    "k": "ASOU5_ALPHA_K",
    "gamma": "ASOU5_ALPHA_GAMMA",
    "sigma_price": "ASOU5_ALPHA_SIGMA_PRICE",
}


def load_engine(params: dict[str, float]):
    for key, env_name in ENV_NAMES.items():
        os.environ[env_name] = str(params[key])
    os.environ["ASOU5_ALPHA_PHI"] = "-1"
    os.environ["ASOU5_ALPHA_QMAX"] = "60"
    os.environ["ASOU5_ALPHA_DT_HJB"] = "0.25"

    sigma_i = params["eta"] / np.sqrt(2.0 * params["beta"])
    y_max = 3.5 * sigma_i
    n_half = max(1, int(np.ceil(y_max / 0.1)))
    ny = 2 * n_half + 1
    os.environ["ASOU5_ALPHA_Y_MAX"] = repr(float(y_max))
    os.environ["ASOU5_ALPHA_NY"] = str(ny)

    sys.modules.pop("alpha_reduced_cole_hopf", None)
    import alpha_reduced_cole_hopf as engine

    return engine, float(sigma_i), float(y_max), ny


def asinh_depths(engine, q: np.ndarray, i: np.ndarray):
    params = engine.P
    qf = np.asarray(q, dtype=float)
    i = np.asarray(i, dtype=float)

    h_i, h_alpha, gate = engine.toy_h_flow_alpha_components(i)
    theta_i = i - params.k * h_i
    r_g = 2.0 * params.p_glft
    m = np.sqrt(np.cosh(i) / np.cosh(theta_i))
    x = m - 1.0
    transport = 1.0 + np.arcsinh(x)
    shift = transport * h_i + transport**gate * h_alpha

    r = r_g * m
    theta_full = i - params.k * shift
    s = (r**2 / 6.0) * np.tanh(theta_full)
    ask = (
        params.delta0
        + shift
        - (r / (2.0 * params.k)) * (2.0 * qf - 1.0)
        + (s / params.k) * (qf**2 - qf + 1.0 / 3.0)
    )
    bid = (
        params.delta0
        - shift
        + (r / (2.0 * params.k)) * (2.0 * qf + 1.0)
        - (s / params.k) * (qf**2 + qf + 1.0 / 3.0)
    )
    ask = np.where(q > -params.qmax, ask, np.inf)
    bid = np.where(q < params.qmax, bid, np.inf)
    return ask, bid


def metrics(engine, value: np.ndarray, sigma_i: float, depth_fn):
    i_eval = np.linspace(-3.0 * sigma_i, 3.0 * sigma_i, 241)
    weights = np.exp(-0.5 * (i_eval / sigma_i) ** 2)
    weights /= weights.sum()

    r_g = 2.0 * engine.P.p_glft
    q_scale = 1.0 / np.sqrt(r_g)
    q_grid = tuple(int(round(factor * q_scale)) for factor in (-1.2, 0.0, 2.4, 3.4))

    errors = []
    targets = []
    metric_weights = []
    for q_value in q_grid:
        q = np.full_like(i_eval, q_value)
        ask, bid = depth_fn(q, i_eval)
        ask_target, bid_target = engine.y_aware_depths(value, q.astype(int), i_eval)
        for candidate, target in ((ask, ask_target), (bid, bid_target)):
            good = np.isfinite(candidate) & np.isfinite(target)
            errors.append((candidate - target)[good])
            targets.append(target[good])
            metric_weights.append(weights[good])

    error = np.concatenate(errors)
    target = np.concatenate(targets)
    weight = np.concatenate(metric_weights)
    abs_rmse = float(np.sqrt(np.sum(weight * error**2) / np.sum(weight)))
    target_rms = float(np.sqrt(np.sum(weight * target**2) / np.sum(weight)))
    return abs_rmse, target_rms, abs_rmse / target_rms, q_grid


def run_case(name: str, overrides: dict[str, float]):
    params = BASE | overrides
    engine, sigma_i, y_max, ny = load_engine(params)
    value, info = engine.solve_ou_ergodic(engine.ALPHA_BAR)

    current_abs, target_rms, current_rel, q_grid = metrics(
        engine, value, sigma_i, engine.toy_h_sc_cubic_depths
    )
    asinh_abs, target_rms_check, asinh_rel, q_grid_check = metrics(
        engine, value, sigma_i, lambda q, i: asinh_depths(engine, q, i)
    )
    if not np.isclose(target_rms, target_rms_check, rtol=0.0, atol=1e-13):
        raise RuntimeError("target RMS changed between formula evaluations")
    if q_grid != q_grid_check:
        raise RuntimeError("inventory evaluation grid changed between formulas")

    return {
        "case": name,
        **params,
        "sigma_i": sigma_i,
        "solve_y_max": y_max,
        "solve_ny": ny,
        "solve_dy": 2.0 * y_max / (ny - 1),
        "q_grid": " ".join(str(q) for q in q_grid),
        "target_rms": target_rms,
        "current_abs_rmse": current_abs,
        "current_rel_rmse": current_rel,
        "asinh_abs_rmse": asinh_abs,
        "asinh_rel_rmse": asinh_rel,
        "asinh_minus_current_rel": asinh_rel - current_rel,
        "converged": int(info["converged"]),
        "tau": info["tau"],
        "last_depth_diff": info["last_depth_diff"],
        "tolerance": info["tolerance"],
    }


def main() -> None:
    rows = []
    for name, overrides in CASES:
        row = run_case(name, overrides)
        rows.append(row)
        print(
            f"{name:40s} current={100.0 * row['current_rel_rmse']:7.2f}% "
            f"asinh={100.0 * row['asinh_rel_rmse']:7.2f}% "
            f"abs=({row['current_abs_rmse']:.5f},{row['asinh_abs_rmse']:.5f}) "
            f"target={row['target_rms']:.5f} tau={row['tau']:.1f} "
            f"conv={row['converged']}"
        )

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
