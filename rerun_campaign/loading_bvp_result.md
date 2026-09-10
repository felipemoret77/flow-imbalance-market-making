# Residual-solved loading (BVP) — first test, 2026-07-28

Solve eq. (fa_loading_residual) of the paper (q^1 balance, main.tex app.):
  L_I h + abar(i) + (rho/k) e^{-r/2} sinh(theta)/sqrt(cosh i cosh theta) = 0,
  theta = i - k h,  r = r_G sqrt(cosh i/cosh theta),  2 kappa = rho.
Pseudo-time relaxation, odd grid [0, 3.5 sigma_I], h(0)=0, h'(L)=0,
residual < 5e-9.  Quotes: h_BVP + FLOW-ONLY curvature r (structural, see
self-consistent failures) + s = r^2/6 tanh(i - k h_BVP).
Metric: |q|<=15, stationary weight in i, eval +-3 sigma_I, relative RMSE.

| case              | current | h_BVP quotes | oracle h-only |
|-------------------|---------|--------------|---------------|
| baseline          | 3.39%   | 1.92%        | --            |
| eta = 2 eta_0     | 91.57%  | 5.58%        | 4.97%         |

- h_BVP + self-consistent r[h_BVP] EXPLODES (1155% at corner): third
  independent confirmation that flow-only curvature is structural.
- The oracle->BVP gap feared (5%->20%) did not materialize: 5.58% vs 4.97%.
- BVP cost: seconds (1D relaxation).  Semi-elementary (offline solve per
  parameter set), one rung above Galerkin (A,c) in the cost hierarchy.
- Reproduce: script embedded in session log; equation and constants are
  paper primitives only (rho, k, r_G, alpha_slope, beta, eta).
Status: NOT in the paper (frozen).  Candidate opening result for follow-up.
Next gate before any promotion: the frozen holdout battery of
asinh_holdout_protocol.md (baseline, eta x2, beta/2+eta x2, alpha forte,
zeta/2, regime with 3 coupled BVPs), all |q|<=15.
