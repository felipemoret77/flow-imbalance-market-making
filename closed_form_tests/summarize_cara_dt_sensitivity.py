#!/usr/bin/env python3
"""Collect paired policy differences across Monte Carlo time steps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=ROOT / "imagens_tex" / "cara_aware_closed_form_comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root_dir.resolve()
    candidates = (
        root / "paired_mc_dt0025_2k",
        root / "paired_mc",
        root / "paired_mc_20k",
    )
    rows = []
    for directory in candidates:
        for model in ("toy", "fast_alpha"):
            path = directory / f"{model}_paired_summary.json"
            if not path.exists():
                continue
            summary = json.loads(path.read_text())
            config = summary["config"]
            display_model = "fast-alpha" if model == "fast_alpha" else model
            for metric in (
                "terminal_pnl",
                "penalized_pnl",
                "fills",
                "avg_abs_inventory",
                "avg_q2",
            ):
                values = summary["paired"][metric]
                rows.append(
                    {
                        "model": display_model,
                        "paths": config["paths"],
                        "horizon": config["horizon"],
                        "dt": config["dt"],
                        "metric": metric,
                        "difference": values["paired_difference_mean"],
                        "se": values["paired_difference_se"],
                        "ci95_low": values["paired_ci95_low"],
                        "ci95_high": values["paired_ci95_high"],
                    }
                )
            ce = summary["paired_certainty_equivalent"]
            rows.append(
                {
                    "model": display_model,
                    "paths": config["paths"],
                    "horizon": config["horizon"],
                    "dt": config["dt"],
                    "metric": "certainty_equivalent",
                    "difference": ce["paired_difference"],
                    "se": ce["bootstrap_se"],
                    "ci95_low": ce["bootstrap_ci95_low"],
                    "ci95_high": ce["bootstrap_ci95_high"],
                }
            )
    rows.sort(key=lambda row: (row["model"], row["metric"], row["dt"], row["paths"]))
    fields = [
        "model",
        "paths",
        "horizon",
        "dt",
        "metric",
        "difference",
        "se",
        "ci95_low",
        "ci95_high",
    ]
    output = root / "dt_sensitivity_summary.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
