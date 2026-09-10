# Rerun campaign (2026-07)

All studies use the paper baseline calibration; each study varies only its target.
Baseline: gamma=0.01, sigma=0.3, k=2, barLambda=0.9, beta=0.0125, eta=0.32,
          zeta=0.25, eps=0.005, eta_alpha=0.001, T=10000, dt=0.05, 20000 paths.

Outputs: logs/*.log (live progress), csv/*.csv (numbers), plots/*.png (figures).

Studies:
  1. gamma_sweep        - robustness + signal-risk vs gamma (deterministic)
  2. param_holdout      - closed-form RMSE out of calibration (deterministic)
  3. domain_resolve     - RHJB [-6,6]/[-9,9] convergence (deterministic)
  4. regime_mc_rerun    - engine + pathwise penalized PnL -> PAIRED CI on Delta_R
                          (single shared RNG stream retained; no burn-in change;
                          same seed => tables identical to the paper run)
  5. fast_alpha_mc_rerun- same pathwise addition -> paired CIs (sc-cubic vs RHJB)
  6. hjb3d_qialpha      - 3D SRN RHJB in (q,i,alpha), diffusion approx of alpha
                          jumps: EXPLORATORY quote-RMSE diagnostic (not value-of-
                          information, not exact CARA)
  7. ce_tail            - CE tail diagnostics (ESS, batch stability) for the
                          blind benchmark; no importance sampling implemented
  8. asinh_holdout_*     - frozen internal holdout of the elementary
                          X=1+asinh(M-1) tail-robust transport against the
                          current sc-cubic formulas (fast-alpha and regime),
                          including exact joint stationary weighting and
                          expanded-domain checks.  See
                          asinh_holdout_protocol.md and
                          asinh_holdout_decision.md.  The candidate failed the
                          frozen promotion screen and was not promoted.
NOTE: gamma_sweep.py stores the fixed-grid absolute-RMSE sweep; the
curvature-scaled relative-RMSE numbers quoted in the paper are in
csv/gamma_sweep_scaled.csv (grid: inventories at (-1.2,0,2.4,3.4)/sqrt(r_G),
OU weight in i, RelRMSE = ||d_cf - d_CARA|| / ||d_CARA||).
