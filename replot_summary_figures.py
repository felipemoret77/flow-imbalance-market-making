#!/usr/bin/env python3
"""Redraw paper figures from committed summaries and optional MC caches.

These four figures are pure functions of the per-policy aggregate metrics that the
full Monte Carlo already wrote to ``*_summary.csv``; nothing about them depends on the
path-level arrays.  Redrawing from the CSV therefore reproduces the published figures
from the same numerical inputs, with the current plotting code, without repeating the
20000-path campaign.

Regenerates:
  imagens_tex/alpha_studies_T10000_qmax60/alpha_reduced_summary_bars.png
  imagens_tex/alpha_studies_T10000_qmax60/alpha_reduced_signal_alignment.png
  imagens_tex/regime_studies_T10000_qmax60/regime_fast_alpha_summary_bars.png
  imagens_tex/regime_studies_T10000_qmax60/regime_fast_alpha_pnl_decomposition.png

When the compact ``*_replot_data.npz`` caches produced by a current full run
are present, this command also redraws the five fast-alpha legacy MC figures
and the two regime terminal-PnL figures.  Missing optional caches are reported
and skipped; the four CSV-only figures are always regenerated.

Set ASOU5_ALPHA_OUTPUT_DIR / ASOU5_REGIME_OUTPUT_DIR to select non-default
study directories for both the input summary CSVs and the output figures.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "alpha_studies"))
sys.path.insert(0, str(ROOT / "regime_studies"))


def load_metrics(path: Path) -> dict[str, dict[str, float]]:
    """policy -> {field: float}, preserving the CSV's policy order."""
    metrics: dict[str, dict[str, float]] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            name = row.pop("policy")
            out: dict[str, float] = {}
            for key, value in row.items():
                try:
                    out[key] = float(value)
                except (TypeError, ValueError):
                    continue
            metrics[name] = out
    return metrics


def main() -> int:
    import alpha_reduced_cole_hopf as fa
    import regime_fast_alpha_ergodic_cole_hopf as rg

    alpha_csv = fa.OUT / "alpha_reduced_summary.csv"
    regime_csv = rg.OUT / "regime_fast_alpha_summary.csv"
    for path in (alpha_csv, regime_csv):
        if not path.is_file():
            print(f"error: missing {path}", file=sys.stderr)
            return 1

    fa_metrics = load_metrics(alpha_csv)
    rg_metrics = load_metrics(regime_csv)

    fa.plot_summary(fa_metrics)
    print(f"wrote {fa.OUT / 'alpha_reduced_summary_bars.png'}")

    fa.plot_signal_alignment(fa_metrics)
    print(f"wrote {fa.OUT / 'alpha_reduced_signal_alignment.png'}")

    rg.plot_summary(rg_metrics)
    print(f"wrote {rg.OUT / 'regime_fast_alpha_summary_bars.png'}")

    rg.plot_pnl_decomposition(rg_metrics)
    print(f"wrote {rg.OUT / 'regime_fast_alpha_pnl_decomposition.png'}")

    alpha_cache = fa.OUT / "alpha_reduced_replot_data.npz"
    if alpha_cache.is_file():
        fa.replot_cached_mc_figures(alpha_cache)
        print(f"redrew five cached fast-alpha MC figures from {alpha_cache}")
    else:
        print(f"optional cache not found; skipped five fast-alpha MC figures: {alpha_cache}")

    regime_cache = rg.OUT / "regime_fast_alpha_replot_data.npz"
    if regime_cache.is_file():
        rg.replot_cached_terminal_pnl_figures(regime_cache)
        print(f"redrew two cached regime terminal-PnL figures from {regime_cache}")
    else:
        print(f"optional cache not found; skipped two regime terminal-PnL figures: {regime_cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
