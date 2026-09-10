#!/usr/bin/env python3
"""Inventory-slice depth RMSE (RHJB vs closed forms) for the accuracy remarks.

Reads the official depth-vs-inventory CSV (y in {-3,0,3}; columns
ask_hjb,bid_hjb,ask_cubic,bid_cubic,ask_gal,bid_gal), and additionally builds the
'affine' baseline (curvature frozen at r_G, s=0, leading-order loading h_LO) to
quote the curvature-refinement gain.  Pools ask+bid over the three slices.

cr  = leading-order tilted-GLFT loading + curvature-refined quotes r(i),s(i)
gal = global Galerkin loading + the same curvature-refined quotes
"""
from __future__ import annotations
import csv
import math
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cara_glft_yaware_study as m  # constants + LO loading

OUT = Path(__file__).resolve().parent.parent / "imagens_tex" / "toy_model_T10000_qmax60"
CSV = OUT / "cara_glft_yaware_ansatz_depths_vs_inventory.csv"

P = m.P
DG, R_G, K = P.delta_gamma, 2.0 * P.p_glft, P.k


def affine_depths(q, y):
    """Frozen-r_G, s=0 quotes with the leading-order loading h_LO(i)."""
    h = m.y_loading_profile(np.asarray(y, float))
    q = np.asarray(q, float)
    ask = DG + h - (R_G / (2 * K)) * (2 * q - 1)
    bid = DG - h + (R_G / (2 * K)) * (2 * q + 1)
    return ask, bid


def main():
    rows = list(csv.DictReader(CSV.open()))
    y = np.array([float(r["y"]) for r in rows])
    q = np.array([float(r["q"]) for r in rows])
    cols = {k: np.array([float(r[k]) for r in rows]) for k in
            ("ask_hjb", "bid_hjb", "ask_cubic", "bid_cubic", "ask_gal", "bid_gal")}
    ask_aff, bid_aff = affine_depths(q, y)

    def rmse(maskname, qcap):
        sel = np.abs(q) <= qcap
        ref = np.concatenate([cols["ask_hjb"][sel], cols["bid_hjb"][sel]])
        out = {}
        for name, a, b in [
            ("affine    (frozen r_G, s=0)", ask_aff[sel], bid_aff[sel]),
            ("tilted-GLFT (LO + r(i),s(i))", cols["ask_cubic"][sel], cols["bid_cubic"][sel]),
            ("Galerkin   (gal + r(i),s(i))", cols["ask_gal"][sel], cols["bid_gal"][sel]),
        ]:
            est = np.concatenate([a, b])
            out[name] = math.sqrt(np.mean((est - ref) ** 2))
        return out

    for qcap in (14, 30):
        print(f"\n=== inventory-slice depth RMSE vs RHJB, |q|<={qcap} "
              f"(pooled ask+bid over i in {{-3,0,3}}) ===")
        for name, v in rmse(qcap, qcap).items():
            print(f"  {name:32} {v:.4f}")

    # galerkin gain over LO at |q|<=14
    r14 = rmse(14, 14)
    lo = r14["tilted-GLFT (LO + r(i),s(i))"]
    ga = r14["Galerkin   (gal + r(i),s(i))"]
    print(f"\nGalerkin changes the cubic-skew inventory-slice RMSE (|q|<=14) by "
          f"{100*(lo-ga)/lo:.0f}%  ({lo:.3f} -> {ga:.3f})")


if __name__ == "__main__":
    main()
