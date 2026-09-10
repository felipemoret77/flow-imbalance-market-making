#!/usr/bin/env python3
"""Wide-domain checks for the frozen regime asinh holdout campaign."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

from asinh_holdout_regime import (
    BASE,
    Q_FACTORS,
    asinh_depths,
    exact_joint_mass,
    fresh_engine,
    metric,
    set_engine_environment,
)


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "rerun_campaign" / "csv" / "asinh_holdout_regime_domaincheck.csv"
LOG = ROOT / "rerun_campaign" / "logs" / "asinh_holdout_regime_domaincheck.log"

CASES = [
    ("zstar_2.5", {"z_star": 2.5}),
    ("corner_eta0.64_zstar2.5", {"eta": 0.64, "z_star": 2.5}),
]


def main() -> None:
    fields = [
        "case",
        "variant",
        "sigma_i",
        "solve_ymax",
        "eval_ymax",
        "ny",
        "dy",
        "q_grid",
        "abs_rmse_exact",
        "target_rms_exact",
        "rel_rmse_exact",
        "eval_mass_exact",
        "stationary_residual",
        "solve_tau",
        "solve_steps",
        "last_depth_diff",
        "converged",
    ]

    with OUT.open("w", newline="") as csv_file, LOG.open("w", buffering=1) as log_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()

        for case_name, changes in CASES:
            params = BASE | changes
            sigma_i, _, _ = set_engine_environment(params)
            solve_ymax = round(params["z_star"] + 4.0 * sigma_i, 2)
            eval_ymax = params["z_star"] + 3.0 * sigma_i
            ny = int(np.ceil(2.0 * solve_ymax / 0.1)) + 1
            if ny % 2 == 0:
                ny += 1
            os.environ["ASOU5_REGIME_Y_MAX"] = str(solve_ymax)
            os.environ["ASOU5_REGIME_NY"] = str(ny)

            engine = fresh_engine()
            u, info = engine.solve_regime_ergodic()
            exact_mass, residual = exact_joint_mass(engine)
            eval_mask = np.abs(engine.Y) <= eval_ymax + 1e-12
            r_g = 2.0 * engine.P.p_glft
            q_grid = tuple(dict.fromkeys(int(round(f / np.sqrt(r_g))) for f in Q_FACTORS))

            variants = {
                "current": engine.regime_sc_cubic_depths,
                "asinh": lambda q, y, r: asinh_depths(engine, q, y, r),
            }
            log_file.write(
                f"{case_name}: solve +/-{solve_ymax}, eval +/-{eval_ymax:.9g}, "
                f"ny={ny}, q={q_grid}, solve={info}, residual={residual:.3e}\n"
            )
            for name, depth_function in variants.items():
                result = metric(engine, u, depth_function, q_grid, exact_mass, eval_mask)
                writer.writerow(
                    {
                        "case": case_name,
                        "variant": name,
                        "sigma_i": sigma_i,
                        "solve_ymax": solve_ymax,
                        "eval_ymax": eval_ymax,
                        "ny": ny,
                        "dy": engine.DY,
                        "q_grid": " ".join(map(str, q_grid)),
                        "abs_rmse_exact": result[0],
                        "target_rms_exact": result[1],
                        "rel_rmse_exact": result[2],
                        "eval_mass_exact": float(np.sum(exact_mass[:, eval_mask])),
                        "stationary_residual": residual,
                        "solve_tau": info["tau"],
                        "solve_steps": info["steps"],
                        "last_depth_diff": info["last_depth_diff"],
                        "converged": int(bool(info["converged"])),
                    }
                )
                csv_file.flush()
                log_file.write(
                    f"  {name:7s} abs/target/rel="
                    f"{result[0]:.6f}/{result[1]:.6f}/{100.0 * result[2]:.3f}%\n"
                )


if __name__ == "__main__":
    main()
