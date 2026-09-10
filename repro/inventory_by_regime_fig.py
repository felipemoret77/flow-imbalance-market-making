"""Mean signed inventory by slow regime, E[Q | R] (Fig. regime_fast_alpha_inventory_by_regime).
Single panel redrawn from the committed regime_fast_alpha_inventory_by_regime.csv (path-and-time pooled means
written by regime_studies/regime_fast_alpha_ergodic_cole_hopf.py).  In the neutral regime all means vanish by
symmetry (|E[Q|R=0]| < 0.02), which is annotated instead of drawing invisible bars.  Writes to paper/images_final/."""
import csv, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "paper" / "images_final"; DATA = REPO / "imagens_tex" / "regime_studies_T10000_qmax60"
plt.rcParams.update({"font.size":13,"axes.titlesize":14,"axes.labelsize":13,"xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":11,"axes.grid":True,"grid.alpha":.25})
rows = list(csv.DictReader(open(DATA / "regime_fast_alpha_inventory_by_regime.csv")))
rows.sort(key=lambda r: int(r["regime"]))
POL = [("GLFT_mean_q","Signal-blind GLFT","#4c72b0"),("Y-aware_mean_q","$I$-aware RHJB","#55a868"),
       ("Alpha/Y-aware_mean_q","Alpha/$I$-aware RHJB","#c44e52"),("Regime-aware_mean_q","Regime-aware RHJB","#111111"),
       ("Regime sc-cubic_mean_q","Regime hybrid cubic","#1B9E77"),("Regime Galerkin_mean_q","Regime Galerkin-cubic","#B07AA1")]
reg = np.array([int(r["regime"]) for r in rows]); x = np.arange(len(reg)); n = len(POL)
width = 0.8 / n; offs = (np.arange(n) - 0.5 * (n - 1)) * width
fig, ax = plt.subplots(figsize=(9.2, 4.8))
for j, (key, lab, col) in enumerate(POL):
    v = np.array([float(r[key]) for r in rows])
    bars = ax.bar(x + offs[j], v, width=width, label=lab, color=col)
    for b, val, rg in zip(bars, v, reg):
        if rg == 0: continue
        ax.text(b.get_x() + b.get_width()/2, val + (0.35 if val >= 0 else -0.35), f"{val:+.1f}", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=8.5, color=col)
neutral = max(abs(float(r[k])) for r in rows if int(r["regime"]) == 0 for k, _, _ in POL)
ax.annotate(f"all $|\\mathbb{{E}}[Q\\mid R{{=}}0]| < {np.ceil(neutral*100)/100:.2f}$", xy=(1, 0), xytext=(1, 4.2),
            ha="center", va="bottom", fontsize=11, arrowprops=dict(arrowstyle="-", lw=.8, color="gray"))
ax.axhline(0, color="k", lw=1, alpha=.5); ax.set_xticks(x); ax.set_xticklabels([f"$R={r:+d}$ ($z={z:+.2f}$)" if r else "$R=0$" for r, z in zip(reg, [float(r['z_level']) for r in rows])])
ax.set_ylabel(r"mean signed inventory $\mathbb{E}[Q_t\mid R_t]$"); ax.set_xlabel("slow regime"); ax.set_ylim(-14.5, 14.5)
ax.legend(ncol=2, loc="upper left", framealpha=.9)
fig.tight_layout(); fig.savefig(OUT / "regime_fast_alpha_inventory_by_regime.png", dpi=170); print("wrote", OUT / "regime_fast_alpha_inventory_by_regime.png", "| max |E[Q|R=0]| =", round(neutral, 4))
