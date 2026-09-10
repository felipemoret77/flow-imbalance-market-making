# Market Making with Persistent Flow Imbalance and Alpha

Code, data, and paper source for *Market Making with Persistent Flow Imbalance
and Alpha* (Felipe Moret & Fabrizio Lillo, Scuola Normale Superiore, Pisa).

Classical market-making models in the Avellaneda–Stoikov and
Guéant–Lehalle–Fernandez-Tapia (GLFT) tradition assume that buy and sell market
orders arrive at constant, symmetric rates. Empirically, trade signs are
persistent and order-flow imbalance is informative about short-horizon price
changes. This work adapts the constant-absolute-risk-aversion (CARA)
market-making benchmark to persistent order flow through three increasingly
realistic models, each of which preserves the tractable inventory structure of
GLFT and yields interpretable closed-form quoting approximations that skew the
benchmark quotes in response to observed flow. Common-path Monte Carlo
simulations compare the closed forms with the numerical reduced HJB (RHJB)
policies solved by Cole–Hopf / Strang splitting.

| Model | Description | Engine |
|-------|-------------|--------|
| **Toy model** (§3) | Persistent OU imbalance shifting the buy/sell composition, martingale reference price | [`cara_glft_yaware_study.py`](cara_glft_yaware_study.py) |
| **Fast alpha** (§4) | Imbalance also generates a short-lived price drift (adverse selection) | [`alpha_studies/alpha_reduced_cole_hopf.py`](alpha_studies/alpha_reduced_cole_hopf.py) |
| **Slow regime** (§5) | Slowly switching regime that separates transient imbalance from persistent flow pressure | [`regime_studies/regime_fast_alpha_ergodic_cole_hopf.py`](regime_studies/regime_fast_alpha_ergodic_cole_hopf.py) |

Each engine solves its RHJB, evaluates the closed-form policies (self-consistent
"hybrid" cubic and Galerkin-cubic), runs the paired common-path Monte Carlo, and
writes the CSV/JSON/NPZ summaries in `imagens_tex/<study>/`. Signed quote
depths are intentional: the analytical benchmark uses the unconstrained
interior CARA/GLFT branch and does not apply a positive-part projection.

## Repository layout

```
flow-imbalance-market-making/
├── cara_glft_yaware_study.py      # toy-model engine (RHJB, closed forms, paired MC)
├── toy_intro_diagnostics.py       # introductory OU-imbalance sample path
├── domain_convergence_table.py    # y-domain truncation check across the three engines
├── replot_summary_figures.py      # summary figures of the engines from *_summary.csv
├── regen_all.sh                   # full baseline regeneration driver
├── alpha_studies/                 # fast-alpha engine + frozen-I vs resolvent robustness check
├── regime_studies/                # slow-regime engine + deterministic depth overlays
├── closed_form_tests/             # closed-form policies, exact CARA solver, paired-MC harness, unit tests
├── rerun_campaign/                # parameter holdouts, gamma sweep, SRN robustness sweep (scripts + CSV)
├── repro/                         # scripts that redraw the manuscript figures from the committed data
├── imagens_tex/                   # committed CSV/JSON/NPZ outputs of the engines (inputs of the paper)
├── paper/                         # main.tex and images_final/ (the figures used in the manuscript)
├── requirements.txt
└── Makefile
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Only NumPy, SciPy, and Matplotlib are needed. The engines are imported by
module name from their own directories (`import cara_glft_yaware_study`,
`from alpha_studies import alpha_reduced_cole_hopf`, …), and each writes its
outputs relative to its own location, so run the scripts from the repository
root (or from the engine directory, as `regen_all.sh` does).

## Usage

```bash
make paper     # compile paper/main.tex -> paper/main.pdf (tectonic or latexmk)
make test      # closed-form / cache regression tests
make figures   # redraw the CSV-driven manuscript figures into paper/images_final/
make regen     # rerun the three engines and the Monte Carlo campaign (hours)
```

## Reproducibility

- The committed `imagens_tex/<study>/*.csv|json|npz` files are the numerical
  inputs of the tables and figures: per-policy summaries, paired contrasts with
  paired standard errors, depth slices of the RHJB and closed-form policies,
  and the toy terminal-PnL path cache used by the paired bootstrap.
  `imagens_tex/cara_aware_closed_form_comparison/` holds the exact-CARA vs.
  closed-form comparison (including the 20k-path paired Monte Carlo).
- `paper/images_final/` contains the figures used in the manuscript. The
  scripts in `repro/` redraw the data-driven ones from the committed CSV/NPZ
  files without repeating the Monte Carlo campaign; the remaining figures
  (expected inventory vs. imbalance, inventory by regime) are produced by the
  engines during a full run.
- `rerun_campaign/` collects the robustness checks referenced in the paper
  (parameter holdouts, scaled holdouts, gamma sweep, SRN robustness sweep,
  domain-truncation checks) with their CSV outputs and the decision notes.
- Baseline calibration (Table 1): `γ = 0.01`, `σ = 0.30`, `Λ̄ = 0.9`,
  `k = 2.0`, `β = 0.0125`, `η = 0.32`, `ζ = 0.25`, `ε = 0.005`,
  `τ_R = 120`; horizon `T = 10000`, `q_max = 60`. Derived constants:
  `χ_γ = 0.3670`, `c_GLFT = 0.3303`, `r_G = 0.0522`, `κ = 0.01724`,
  `δ_γ = 0.4988`, `A_I = 0.2898`, `A_α = 0.766`, `A_R = 0.331`.
- All policy comparisons use common-path (paired) simulations under fixed
  seeds; paired standard errors are reported alongside the mean differences.

## Citation

If you use this code, please cite the paper; see [`CITATION.cff`](CITATION.cff).

## License

Released under the MIT License; see [`LICENSE`](LICENSE).
