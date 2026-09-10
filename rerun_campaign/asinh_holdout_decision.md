# Decision: asinh tail-robust closure

Date: 2026-07-27.

The candidate and promotion screen were frozen in
`asinh_holdout_protocol.md` before the new CSV outputs were produced.  The
candidate was not changed after inspecting the holdouts.

## Frozen candidate

\[
X(i)=1+\operatorname{asinh}(M(i)-1),\qquad
H_X=Xh_I+X^g h_\alpha+A_Rz_r.
\]

The regime term is omitted in fast-alpha.  The raw curvature \(r=r_GM\), the
self-consistent cubic, and the quote map remain unchanged.

## Deterministic result

All 14 fast-alpha and all 16 regime RHJB solves converged to the frozen depth
tolerance.  The regime metrics use the invariant mass of the discretized joint
\((I,R)\) generator.  Expanded-domain checks confirm the conclusions in the
critical wide-imbalance cases.

### Fast-alpha

- Baseline relative RMSE: current \(3.46\%\), asinh \(6.85\%\).
- Across the 13 fresh holdouts, the asinh candidate wins 9, and the median
  changes from \(13.65\%\) to \(12.31\%\).
- Material failures remain:
  - \(\eta=0.64\): \(90.34\%\to39.76\%\);
  - \((\beta,\eta)=(0.00625,0.64)\):
    \(356.72\%\to62.47\%\);
  - \((\eta,\zeta)=(0.64,0.125)\):
    \(71.40\%\to31.64\%\);
  - \((\bar\Lambda,\epsilon)=(1.8,0.01)\):
    \(42.76\%\to34.63\%\).
- The two wide-imbalance failures are unchanged under a larger
  \(\pm4.5\sigma_I\) solve domain, finer imbalance resolution, and alternative
  inventory grids.

### Regime

- Baseline relative RMSE: current \(5.11\%\), asinh \(7.46\%\).
- Across baseline plus 15 holdouts, the asinh candidate wins 11; mean relative
  RMSE changes from \(25.97\%\) to \(15.50\%\), median from \(11.91\%\) to
  \(10.86\%\), and worst case from \(89.70\%\) to \(42.81\%\).
- Material failures remain:
  - \(\eta=0.64\): \(89.70\%\to39.53\%\);
  - \((\eta,z_\star)=(0.64,2.5)\):
    \(88.34\%\to39.06\%\);
  - \((\zeta,\bar\Lambda)=(0.125,1.8)\):
    \(51.16\%\to42.81\%\).
- An expanded domain covering the shifted regimes gives \(38.67\%\), rather
  than \(39.06\%\), for the wide-imbalance/regime corner.  The failure is not a
  truncation artifact.

## Frozen screen

1. Baseline deterioration at most 3 percentage points: **fail** in fast-alpha
   (\(+3.386\) points); pass in regime (\(+2.35\) points).
2. No fresh holdout worsens by more than 5 points: **pass**.
3. Candidate relative RMSE at most 15% at every holdout/corner: **fail** in
   both fast-alpha and regime.
4. Materially lower worst-case error: **pass**, but insufficient to override
   criteria 1 and 3.

## Decision

Do **not** promote the asinh closure to the body.  Keep the current sc-cubic
formula as the paper's primary baseline formula.  The asinh form is a genuine
tail regularizer and may be documented as an exploratory appendix alternative,
but it does not provide uniform robustness over the frozen parameter set.

The paired Monte Carlo stage is not run: the frozen protocol triggers it only
after all deterministic promotion conditions pass.  This is a gate decision,
not a computational omission.

## Reproducibility

- `asinh_holdout_fast.py` and `csv/asinh_holdout_fast.csv`;
- `asinh_holdout_regime.py`, `csv/asinh_holdout_regime.csv`, and its log;
- `asinh_holdout_regime_domaincheck.py`,
  `csv/asinh_holdout_regime_domaincheck.csv`, and its log;
- `asinh_holdout_fast_domaincheck.py` and
  `csv/asinh_holdout_fast_domaincheck.csv`.
