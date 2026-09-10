#!/usr/bin/env python3
"""Plot outputs from ``paired_cara_closed_form_mc.py``."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np


CURRENT = "Current elementary (SRN closure)"
LEADING = "Leading elementary CARA-aware"
COLORS = ("#0072B2", "#D55E00")


def save(fig: plt.Figure, out: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_model(model: str, data_dir: Path, out: Path) -> dict[str, object]:
    stem = model.replace("-", "_")
    data = np.load(data_dir / f"{stem}_paired_paths.npz")
    summary = json.loads((data_dir / f"{stem}_paired_summary.json").read_text())
    modes = summary["policy_modes"]
    labels = (CURRENT, LEADING if modes[1] == "leading_cara" else "SRN identity copy")
    t = data["record_time"]
    pnl = data["record_mean_pnl"]
    penalized = data["record_mean_penalized"]
    terminal = data["terminal_pnl"]

    fig, ax = plt.subplots(2, 2, figsize=(12.0, 8.0))
    for p in range(2):
        ax[0, 0].plot(t, pnl[p], lw=2.0, color=COLORS[p], label=labels[p])
        ax[0, 1].plot(t, penalized[p], lw=2.0, color=COLORS[p], label=labels[p])
    ax[0, 0].set_title("Mean accumulated terminal-PnL proxy")
    ax[0, 1].set_title("Mean penalized PnL")
    for axis in ax[0]:
        axis.set_xlabel("time")
        axis.set_ylabel("cross-path mean")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8)

    pooled = np.concatenate((terminal[0], terminal[1]))
    lo, hi = np.quantile(pooled, [0.005, 0.995])
    bins = np.linspace(lo, hi, 70)
    for p in range(2):
        ax[1, 0].hist(
            terminal[p], bins=bins, density=True, histtype="step", lw=2.0,
            color=COLORS[p], label=labels[p],
        )
    ax[1, 0].set_title("Terminal PnL distribution (0.5–99.5% window)")
    ax[1, 0].set_xlabel("terminal PnL")
    ax[1, 0].set_ylabel("density")
    ax[1, 0].legend(frameon=False, fontsize=8)
    ax[1, 0].grid(alpha=0.22)

    difference = terminal[1] - terminal[0]
    ax[1, 1].hist(difference, bins=60, density=True, color="#7A5195", alpha=0.75)
    ax[1, 1].axvline(np.mean(difference), color="black", lw=2.0, label=f"mean = {np.mean(difference):.3g}")
    ax[1, 1].axvline(0.0, color="black", lw=1.0, alpha=0.35)
    ax[1, 1].set_title("Pathwise paired PnL difference")
    ax[1, 1].set_xlabel("leading CARA-aware minus current")
    ax[1, 1].set_ylabel("density")
    ax[1, 1].legend(frameon=False, fontsize=8)
    ax[1, 1].grid(alpha=0.22)

    cfg = summary["config"]
    fig.suptitle(
        f"{model}: paired policy comparison "
        f"(N={cfg['paths']}, T={cfg['horizon']:g}, dt={cfg['dt']:g})",
        fontsize=14,
    )
    save(fig, out, f"{stem}_paired_pnl")

    if model == "fast-alpha":
        centers = data["alpha_bin_centers"]
        counts = data["alpha_bin_counts"]
        ask = data["conditional_ask_by_alpha"]
        bid = data["conditional_bid_by_alpha"]
        valid = counts > 0
        fig, axes = plt.subplots(
            3, 1, figsize=(9.0, 9.0), sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.0, 0.42]},
        )
        for p in range(2):
            axes[0].plot(
                centers[valid], ask[p, valid], color=COLORS[p], lw=2.1,
                ls="-" if p == 0 else "--", label=labels[p],
            )
            axes[1].plot(
                centers[valid], bid[p, valid], color=COLORS[p], lw=2.1,
                ls="-" if p == 0 else "--", label=labels[p],
            )
        axes[0].set_title(r"Realised conditional mean ask depth $\mathbb{E}[\delta^a_t\mid\alpha_t]$")
        axes[1].set_title(r"Realised conditional mean bid depth $\mathbb{E}[\delta^b_t\mid\alpha_t]$")
        for axis in axes[:2]:
            axis.set_ylabel("conditional mean depth")
            axis.axvline(0.0, color="black", alpha=0.25, lw=1.0)
            axis.grid(alpha=0.22)
            axis.legend(frameon=False, fontsize=8)
        bin_width = float(np.diff(centers).mean()) if len(centers) > 1 else 0.01
        axes[2].bar(centers[valid], counts[valid], width=bin_width * 0.8, color="#777777")
        axes[2].set_yscale("log")
        axes[2].set_ylabel("count")
        axes[2].set_xlabel(r"realised alpha $\alpha_t$")
        axes[2].grid(alpha=0.18, axis="y")
        fig.suptitle(
            "Fast-alpha: indirect depth dependence through visited $(q,I,alpha)$ states",
            fontsize=13,
        )
        save(fig, out, "fast_alpha_conditional_depths_vs_alpha")

    p = summary["paired"]["terminal_pnl"]
    ce = summary["paired_certainty_equivalent"]
    return {
        "model": model,
        "paths": cfg["paths"],
        "horizon": cfg["horizon"],
        "dt": cfg["dt"],
        "pnl_difference": p["paired_difference_mean"],
        "pnl_difference_se": p["paired_difference_se"],
        "pnl_ci95_low": p["paired_ci95_low"],
        "pnl_ci95_high": p["paired_ci95_high"],
        "ce_difference": ce["paired_difference"],
        "ce_bootstrap_se": ce["bootstrap_se"],
        "ce_ci95_low": ce["bootstrap_ci95_low"],
        "ce_ci95_high": ce["bootstrap_ci95_high"],
        "fills_difference": summary["paired"]["fills"]["paired_difference_mean"],
        "avg_abs_inventory_difference": summary["paired"]["avg_abs_inventory"]["paired_difference_mean"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "imagens_tex" / "cara_aware_closed_form_comparison" / "paired_mc",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "imagens_tex" / "cara_aware_closed_form_comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in ("toy", "fast-alpha"):
        stem = model.replace("-", "_")
        if (data_dir / f"{stem}_paired_paths.npz").exists():
            rows.append(plot_model(model, data_dir, out))
    if not rows:
        raise FileNotFoundError(f"no paired MC outputs found under {data_dir}")
    robustness_path = data_dir / "ce_tail_robustness.json"
    if robustness_path.exists():
        robust = {
            item["model"]: item for item in json.loads(robustness_path.read_text())
        }
        for row in rows:
            if row["model"] in robust:
                item = robust[row["model"]]
                row["ce_difference"] = item["ce_difference"]
                row["ce_bootstrap_se"] = item["bootstrap_se"]
                row["ce_ci95_low"] = item["bootstrap_ci95_low"]
                row["ce_ci95_high"] = item["bootstrap_ci95_high"]
    fields = list(rows[0].keys())
    with (out / "paired_mc_overview.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote paired MC plots to {out}")


if __name__ == "__main__":
    main()
