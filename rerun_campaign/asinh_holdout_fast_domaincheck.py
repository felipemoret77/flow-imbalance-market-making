#!/usr/bin/env python3
"""Anti-artifact checks for the two wide-imbalance asinh holdout failures.

This companion to :mod:`asinh_holdout_fast` keeps the frozen candidate and the
``[-3 sigma_I, 3 sigma_I]`` OU-weighted evaluation band, but expands the RHJB
solve to ``[-4.5 sigma_I, 4.5 sigma_I]`` and refines the target grid spacing to
at most 0.075.  It also repeats the metric on the primary, symmetric-wide, and
symmetric-central curvature-scaled inventory grids.

No formula or numerical constant is selected from these checks; they test
whether the two recorded failures are domain, resolution, or q-grid artifacts.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "rerun_campaign" / "csv" / "asinh_holdout_fast_domaincheck.csv"
SCRIPT_DIR = ROOT / "rerun_campaign"
ALPHA_DIR = ROOT / "alpha_studies"
for path in (SCRIPT_DIR, ALPHA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import asinh_holdout_fast as frozen


CASES = [
    ("eta_0.64", {"eta": 0.64}),
    ("corner_beta_0.00625_eta_0.64", {"beta": 0.00625, "eta": 0.64}),
]

Q_SCHEMES = {
    "primary": (-1.2, 0.0, 2.4, 3.4),
    "symmetric_wide": (-3.4, -2.4, -1.2, 0.0, 1.2, 2.4, 3.4),
    "symmetric_central": (-2.4, -1.2, 0.0, 1.2, 2.4),
}


def load_engine(params: dict[str, float]):
    for key, env_name in frozen.ENV_NAMES.items():
        os.environ[env_name] = str(params[key])
    os.environ["ASOU5_ALPHA_PHI"] = "-1"
    os.environ["ASOU5_ALPHA_QMAX"] = "60"
    os.environ["ASOU5_ALPHA_DT_HJB"] = "0.25"

    sigma_i = params["eta"] / np.sqrt(2.0 * params["beta"])
    y_max = 4.5 * sigma_i
    n_half = max(1, int(np.ceil(y_max / 0.075)))
    ny = 2 * n_half + 1
    os.environ["ASOU5_ALPHA_Y_MAX"] = repr(float(y_max))
    os.environ["ASOU5_ALPHA_NY"] = str(ny)

    sys.modules.pop("alpha_reduced_cole_hopf", None)
    import alpha_reduced_cole_hopf as engine

    return engine, float(sigma_i), float(y_max), ny


def metrics(engine, value, sigma_i: float, depth_fn, q_factors):
    i_eval = np.linspace(-3.0 * sigma_i, 3.0 * sigma_i, 241)
    weights = np.exp(-0.5 * (i_eval / sigma_i) ** 2)
    weights /= weights.sum()

    r_g = 2.0 * engine.P.p_glft
    q_grid = tuple(int(round(factor / np.sqrt(r_g))) for factor in q_factors)
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
    params = frozen.BASE | overrides
    engine, sigma_i, y_max, ny = load_engine(params)
    value, info = engine.solve_ou_ergodic(engine.ALPHA_BAR, max_tau=5000.0)

    rows = []
    for scheme, factors in Q_SCHEMES.items():
        current_abs, target_rms, current_rel, q_grid = metrics(
            engine, value, sigma_i, engine.toy_h_sc_cubic_depths, factors
        )
        asinh_abs, target_check, asinh_rel, q_check = metrics(
            engine,
            value,
            sigma_i,
            lambda q, i: frozen.asinh_depths(engine, q, i),
            factors,
        )
        if not np.isclose(target_rms, target_check, rtol=0.0, atol=1e-13):
            raise RuntimeError("target RMS changed between formula evaluations")
        if q_grid != q_check:
            raise RuntimeError("inventory grid changed between formula evaluations")
        rows.append(
            {
                "case": name,
                **params,
                "sigma_i": sigma_i,
                "solve_sigma_multiple": 4.5,
                "eval_sigma_multiple": 3.0,
                "solve_y_max": y_max,
                "solve_ny": ny,
                "solve_dy": 2.0 * y_max / (ny - 1),
                "q_scheme": scheme,
                "q_grid": " ".join(str(q) for q in q_grid),
                "target_rms": target_rms,
                "current_abs_rmse": current_abs,
                "current_rel_rmse": current_rel,
                "asinh_abs_rmse": asinh_abs,
                "asinh_rel_rmse": asinh_rel,
                "converged": int(info["converged"]),
                "tau": info["tau"],
                "steps": info["steps"],
                "last_depth_diff": info["last_depth_diff"],
                "tolerance": info["tolerance"],
            }
        )
    return rows


def main() -> None:
    rows = []
    for name, overrides in CASES:
        for row in run_case(name, overrides):
            rows.append(row)
            print(
                f"{row['case']:34s} {row['q_scheme']:18s} "
                f"current={100.0 * row['current_rel_rmse']:7.2f}% "
                f"asinh={100.0 * row['asinh_rel_rmse']:7.2f}% "
                f"tau={row['tau']:.1f} conv={row['converged']}"
            )

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
