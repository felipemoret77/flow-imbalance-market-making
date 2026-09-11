"""Terminal-PnL and lifetime-inventory distributions, each policy against the signal-blind benchmark.

Two figures, three panels each (toy / fast-alpha / regime), drawn from the committed per-path caches:
  toy       cara_glft_terminal_pnl_paths.npz + cara_glft_lifetime_inventory_full_prob.csv
  fast-alpha alpha_reduced_replot_data.npz  (terminal_total, lifetime_q_centers/counts)
  regime    regime_fast_alpha_replot_data.npz + regime_fast_alpha_lifetime_inventory_hist.csv
A study whose cache is absent is skipped and reported.  Writes to paper/images_final/."""
import csv, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "paper" / "images_final"; D = REPO / "imagens_tex"
plt.rcParams.update({"font.size":16,"axes.titlesize":17,"axes.labelsize":16,"xtick.labelsize":14,
                     "ytick.labelsize":14,"legend.fontsize":15,"axes.grid":True,"grid.alpha":.25})
GLFT, RHJB_I, RHJB_A, RHJB_R = "#7f7f7f", "#1f77b4", "#c44e52", "#111111"
BODY, GAL = "#E69F00", "#B07AA1"
GAMMA = 0.01
CE = lambda x: float(-np.log(np.mean(np.exp(-GAMMA * x))) / GAMMA)

def stacked(z, base):
    """Return {name: array} from either a stacked `base` array or per-policy `base__pNN_name` keys."""
    names = [str(n) for n in z["policy_names"]]
    if base in z.files:
        return {n: z[base][j] for j, n in enumerate(names)}
    keys = sorted(k for k in z.files if k.startswith(base + "__"))
    return {names[j]: z[k] for j, k in enumerate(keys)}

# ---- per study: (title, [(label, colour, linestyle)], pnl dict, inventory (q, {label: prob}))
def toy():
    z = np.load(D / "toy_model_T10000_qmax60" / "cara_glft_terminal_pnl_paths.npz")
    pnl = {"Signal-blind GLFT": z["Signal-blind_Gaussian_GLFT"], "$I$-aware RHJB": z["I-aware_RHJB"],
           "Toy self-consistent cubic": z["cubic-skew_refined_ansatz"], "Toy Galerkin-cubic": z["I-aware_galerkin_ansatz"]}
    rows = list(csv.DictReader(open(D / "toy_model_T10000_qmax60" / "cara_glft_lifetime_inventory_full_prob.csv")))
    q = np.array([int(r["inventory_q"]) for r in rows])
    inv = {"Signal-blind GLFT": "Signal-blind Gaussian GLFT", "$I$-aware RHJB": "I-aware RHJB",
           "Toy self-consistent cubic": "cubic-skew refined ansatz", "Toy Galerkin-cubic": "I-aware galerkin ansatz"}
    inv = {k: np.array([float(r[v]) for r in rows]) for k, v in inv.items()}
    sty = {"Signal-blind GLFT": (GLFT, "-"), "$I$-aware RHJB": (RHJB_I, "-"),
           "Toy self-consistent cubic": (BODY, "--"), "Toy Galerkin-cubic": (GAL, ":")}
    return "Toy model (martingale price)", sty, pnl, (q, inv)

def fast_alpha():
    p = D / "alpha_studies_T10000_qmax60" / "alpha_reduced_replot_data.npz"
    if not p.exists(): return None
    z = np.load(p)
    raw = stacked(z, "terminal_total")
    ren = {"GLFT": "Signal-blind GLFT", "Y-aware": "$I$-aware RHJB", "Alpha/Y-aware": "Alpha/$I$-aware RHJB",
           "Alpha/Y sc-cubic": "Fast-alpha hybrid cubic", "Alpha/Y galerkin": "Fast-alpha Galerkin-cubic"}
    pnl = {ren.get(k, k): v for k, v in raw.items()}
    ctr = np.asarray(z["lifetime_q_centers"]); cnt = stacked(z, "lifetime_q_counts")
    inv = {ren.get(k, k): np.asarray(v, float) / max(1.0, np.sum(v)) for k, v in cnt.items()}
    sty = {"Signal-blind GLFT": (GLFT, "-"), "$I$-aware RHJB": (RHJB_I, "-"), "Alpha/$I$-aware RHJB": (RHJB_A, "-"),
           "Fast-alpha hybrid cubic": (BODY, "--"), "Fast-alpha Galerkin-cubic": (GAL, ":")}
    return "Fast-alpha model", sty, pnl, (ctr, inv)

def regime():
    z = np.load(D / "regime_studies_T10000_qmax60" / "regime_fast_alpha_replot_data.npz")
    raw = stacked(z, "terminal_total")
    ren = {"GLFT": "Signal-blind GLFT", "Y-aware": "$I$-aware RHJB", "Alpha/Y-aware": "Alpha/$I$-aware RHJB",
           "Regime-aware": "Regime-aware RHJB", "Regime sc-cubic": "Regime hybrid cubic", "Regime Galerkin": "Regime Galerkin-cubic"}
    pnl = {ren.get(k, k): v for k, v in raw.items()}
    rows = list(csv.DictReader(open(D / "regime_studies_T10000_qmax60" / "regime_fast_alpha_lifetime_inventory_hist.csv")))
    q = np.array([int(r["q"]) for r in rows])
    inv = {ren[k]: np.array([float(r[f"{k}_freq"]) for r in rows]) for k in ren}
    sty = {"Signal-blind GLFT": (GLFT, "-"), "$I$-aware RHJB": (RHJB_I, "-"), "Alpha/$I$-aware RHJB": (RHJB_A, "-"),
           "Regime-aware RHJB": (RHJB_R, "-"), "Regime hybrid cubic": (BODY, "--"), "Regime Galerkin-cubic": (GAL, ":")}
    return "Slow-regime model", sty, pnl, (q, inv)

from matplotlib.lines import Line2D
ROLES = [("Signal-blind GLFT", GLFT, "-"), ("$I$-aware RHJB", RHJB_I, "-"),
         ("Alpha/$I$-aware RHJB", RHJB_A, "-"), ("Regime-aware RHJB", RHJB_R, "-"),
         ("body closed form (cubic)", BODY, "--"), ("appendix Galerkin-cubic", GAL, ":")]
def shared_legend(fig):
    h = [Line2D([], [], color=c, ls=ls, lw=2.6) for _, c, ls in ROLES]
    fig.legend(h, [n for n, _, _ in ROLES], loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.005))

STUDIES = [s for s in (toy(), fast_alpha(), regime()) if s is not None]
if len(STUDIES) < 3: print("AVISO: rodando com", len(STUDIES), "de 3 estudos (cache do fast-alpha ausente)")

# ============ Figure A: terminal PnL distribution ============
fig, ax = plt.subplots(1, len(STUDIES), figsize=(5.6 * len(STUDIES), 4.9))
ax = np.atleast_1d(ax)
for a, (title, sty, pnl, _) in zip(ax, STUDIES):
    lo = min(np.percentile(v, 0.2) for v in pnl.values()); hi = max(np.percentile(v, 99.8) for v in pnl.values())
    bins = np.linspace(lo, hi, 90)
    ces = [CE(v) for v in pnl.values()]                     # keep every CE marker on scale
    xlo, xhi = min(lo, min(ces)), max(hi, max(ces)); pad = 0.04 * (xhi - xlo)
    peak = 0.0
    for lab, v in pnl.items():
        col, ls = sty[lab]
        n, _, _ = a.hist(v, bins=bins, density=True, histtype="step", lw=2.0, ls=ls, color=col, label=lab)
        peak = max(peak, n.max())
        a.axvline(CE(v), color=col, ls=(0, (1, 2)), lw=1.6, alpha=.85)
    ylo = 1e-6
    a.set_yscale("log"); a.set_ylim(ylo, peak * 6)
    a.set_xlim(xlo - pad, xhi + pad)
    a.set_title(title); a.set_xlabel("terminal PnL")
    print(f"  [{title}]")
    for lab, v in pnl.items():
        print(f"    {lab:28s} mean {v.mean():8.0f}  sd {v.std():7.0f}  p01 {np.percentile(v,1):8.0f}  CE {CE(v):8.0f}")
ax[0].set_ylabel("density (log scale)")
shared_legend(fig); fig.tight_layout(rect=(0, 0.11, 1, 1)); fig.savefig(OUT / "terminal_pnl_distributions.png", dpi=170)
print("wrote", OUT / "terminal_pnl_distributions.png")

# ============ Figure B: lifetime inventory distribution ============
fig, ax = plt.subplots(1, len(STUDIES), figsize=(5.6 * len(STUDIES), 4.9))
ax = np.atleast_1d(ax)
for a, (title, sty, _, (q, inv)) in zip(ax, STUDIES):
    for lab, p in inv.items():
        col, ls = sty[lab]
        a.step(q, np.where(p > 0, p, np.nan), where="mid", lw=2.0, ls=ls, color=col, label=lab)
    a.set_yscale("log"); a.set_ylim(1e-6, 0.6); a.set_xlim(-45, 45)
    a.set_yticks([1e-6, 1e-4, 1e-2])
    a.set_title(title); a.set_xlabel("inventory $q$")
    print(f"  [{title}] percentis do inventario")
    for lab, p in inv.items():
        c = np.cumsum(p) / p.sum()
        print(f"    {lab:28s} p01 {q[np.searchsorted(c,0.01)]:4d}  p99 {q[np.searchsorted(c,0.99)]:4d}  E|q| {np.sum(np.abs(q)*p)/p.sum():6.2f}")
ax[0].set_ylabel("time-in-state frequency (log scale)")
shared_legend(fig); fig.tight_layout(rect=(0, 0.11, 1, 1)); fig.savefig(OUT / "lifetime_inventory_distributions.png", dpi=170)
print("wrote", OUT / "lifetime_inventory_distributions.png")
