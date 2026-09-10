"""Toy-model conditional inventory E[Q | I] (Fig. cara_glft_expected_inventory_vs_y).
Redrawn from the committed cara_glft_expected_inventory_vs_y.csv with the paper's
per-policy palette, shared with the inventory-by-regime figure.  Writes to paper/images_final/."""
import csv, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "paper" / "images_final"; DATA = REPO / "imagens_tex" / "toy_model_T10000_qmax60"
plt.rcParams.update({"font.size":14,"axes.titlesize":15,"axes.labelsize":14,"xtick.labelsize":12,
                     "ytick.labelsize":12,"legend.fontsize":12,"axes.grid":True,"grid.alpha":.25})
rows = list(csv.DictReader(open(DATA / "cara_glft_expected_inventory_vs_y.csv")))
y = np.array([float(r["y_center"]) for r in rows])
SER = [("Signal-blind_Gaussian_GLFT", "Signal-blind GLFT",       "#7f7f7f", "-",  2.4),
       ("I-aware_RHJB",               "$I$-aware RHJB",          "#1f77b4", "-",  2.8),
       ("cubic-skew_refined_ansatz",  "Toy self-consistent cubic","#E69F00", "--", 2.5),
       ("I-aware_galerkin_ansatz",    "Toy Galerkin-cubic",      "#B07AA1", ":",  2.6)]
fig, ax = plt.subplots(figsize=(8.4, 5.0))
for key, lab, col, ls, lw in SER:
    ax.plot(y, np.array([float(r[key]) for r in rows]), label=lab, color=col, ls=ls, lw=lw)
ax.axhline(0, color="k", lw=.9, alpha=.45); ax.axvline(0, color="k", lw=.7, alpha=.3)
ax.set_xlabel("imbalance $i$"); ax.set_ylabel(r"$\mathbb{E}[Q_t\mid I_t=i]$")
ax.legend(loc="upper right", framealpha=.9)
fig.tight_layout(); fig.savefig(OUT / "cara_glft_expected_inventory_vs_y.png", dpi=170)
print("wrote", OUT / "cara_glft_expected_inventory_vs_y.png")
