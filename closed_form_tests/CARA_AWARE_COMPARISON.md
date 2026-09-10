# Current versus leading-CARA-aware elementary closures

This diagnostic compares the paper's current elementary SRN closures with the
single-factor signal-risk extension

\[
r(i)\longmapsto r(i)
\sqrt{1+\eta^2\{\partial_i h_{\rm shift}(i)\}^2/\sigma^2}.
\]

The extension is a **leading elementary CARA-aware closure**, not the numerical
solution of the exact CARA HJB.  It keeps the current loading, gates and angles,
changes the inventory curvature, and recomputes the cubic coefficient.

## Files

- `cara_aware_closed_forms.py`: audited toy and fast-alpha APIs with
  `risk_mode="srn"|"leading_cara"`.
- `plot_cara_aware_closed_forms.py`: profiles, depth slices, heatmaps, direct
  realised-alpha slice, and frozen-I alpha-proxy slice.
- `paired_cara_closed_form_mc.py`: paired PnL/CE Monte Carlo with independent
  environment streams and common-component Poisson coupling.
- `plot_paired_cara_closed_form_mc.py`: PnL and realised conditional-alpha
  figures.
- `test_cara_aware_closed_forms.py` and
  `test_paired_cara_closed_form_mc.py`: formula and simulation regressions.

No script in this diagnostic edits `main.tex` or the production study outputs.

## Reproduce

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest \
  closed_form_tests.test_cara_aware_closed_forms \
  closed_form_tests.test_paired_cara_closed_form_mc -v

python closed_form_tests/plot_cara_aware_closed_forms.py

python -u closed_form_tests/paired_cara_closed_form_mc.py \
  --model toy --paths 20000 --horizon 10000 --dt 0.05 \
  --record-dt 10 --bootstrap 500 \
  --output-dir imagens_tex/cara_aware_closed_form_comparison/paired_mc_20k

python -u closed_form_tests/paired_cara_closed_form_mc.py \
  --model fast-alpha --paths 20000 --horizon 10000 --dt 0.05 \
  --record-dt 10 --bootstrap 500 \
  --output-dir imagens_tex/cara_aware_closed_form_comparison/paired_mc_20k

python closed_form_tests/plot_paired_cara_closed_form_mc.py \
  --data-dir imagens_tex/cara_aware_closed_form_comparison/paired_mc_20k
```

Use `--right-mode srn` for the identity check: both arms must then be identical
path by path.

## Interpretation of alpha plots

The reduced fast-alpha policies are functions of `(q,I)`, not of the realised
alpha state.  Therefore the direct partial slice versus realised alpha is flat.
Two other plots answer different questions:

1. depths versus the frozen-I alpha proxy, which is just a reparameterisation
   of the imbalance slice;
2. realised conditional mean depth given alpha in the simulation, which may
   vary because alpha is correlated with visited `(q,I)` states.

## Statistical caution

Mean PnL, penalized PnL and inventory differences use pathwise paired standard
errors.  CE uses a joint bootstrap of path indices.  CE rankings should not be
reported unless the exponential-weight ESS, maximum weight share and tail
sensitivity are acceptable; bootstrap resampling cannot discover tail events
that are absent from the simulated support.
