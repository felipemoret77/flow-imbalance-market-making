# Frozen validation protocol: asinh tail-robust closure

Date frozen: 2026-07-27, before running the holdouts below.

## Candidate (no retuning after holdout inspection)

Let

\[
M(i)=\sqrt{\frac{\cosh i}{\cosh(i-kh_I(i))}},\qquad
g(i)=\tanh^2(A_2 i),\qquad
X(i)=1+\operatorname{asinh}(M(i)-1).
\]

The candidate changes only the transported directional shift:

\[
H_{\rm asinh}(i,r)=X(i)h_I(i)+X(i)^{g(i)}h_\alpha(i)+A_Rz_r,
\]

where the last term is omitted in fast-alpha.  The raw curvature
\(r=r_GM\), the self-consistent cubic
\(s=r^2\tanh(i-kH_{\rm asinh})/6\), and the quote map are unchanged.

## Frozen deterministic metric

- Solve the matching SRN RHJB on \([-3.5\sigma_I,3.5\sigma_I]\).
- Evaluate on \([-3\sigma_I,3\sigma_I]\).
- Use the curvature-scaled inventory grid from the existing campaign.
- Weight the imbalance coordinate by its stationary law.  For the regime
  model, prefer the invariant law of the discretized joint \((I,R)\)
  generator and report explicitly if a shifted-Gaussian proxy is used.
- Report absolute depth RMSE, target RMS, relative depth RMSE, solver
  convergence, terminal pseudo-time, domain, and inventory grid.
- Compare the frozen asinh candidate with the current sc-cubic formula at
  every point.  Do not select or modify the transform after inspecting these
  results.

## Fresh fast-alpha holdouts

- One at a time: \(\eta\in\{0.16,0.64\}\),
  \(\zeta\in\{0.125,0.5\}\),
  \(\bar\Lambda\in\{0.45,1.8\}\), and
  \(\sigma\in\{0.15,0.6\}\).
- Joint corners:
  \((\beta,\eta)=(0.00625,0.64)\),
  \((0.025,0.16)\),
  \((\eta,\zeta)=(0.64,0.125)\),
  \((\bar\Lambda,\epsilon)=(1.8,0.01)\), and
  \((\gamma,\sigma)=(0.02,0.6)\).

## Fresh regime holdouts

- One at a time: \(\eta\in\{0.16,0.64\}\),
  \(z_\star\in\{0.625,2.5\}\),
  \(\zeta\in\{0.125,0.5\}\),
  \(\bar\Lambda\in\{0.45,1.8\}\), and
  \(\gamma\in\{0.005,0.02\}\).
- Joint corners:
  \((\beta,z_\star)=(0.00625,2.5)\),
  \((\beta,\tau_R)=(0.00625,240)\),
  \((\eta,z_\star)=(0.64,2.5)\),
  \((\zeta,\bar\Lambda)=(0.125,1.8)\), and
  \((\gamma,\tau_R)=(0.02,60)\).

## Frozen promotion screen

Promotion to the body is considered only if all deterministic conditions hold:

1. Baseline relative RMSE deteriorates by no more than 3 percentage points.
2. The candidate is no worse than the current formula by more than 5
   percentage points at any fresh holdout.
3. The candidate's relative RMSE is at most 15% at every fresh one-at-a-time
   holdout and joint corner.
4. The worst-case error is materially lower than for the current formula.

Passing this screen triggers a paired Monte Carlo comparison at the baseline
and at least one predeclared stress point.  Promotion additionally requires no
economically material deterioration in baseline penalized PnL/CE and a stable
improvement in the stress case.  Failure of any deterministic condition keeps
the current formula in the body; the asinh form may remain an explicitly
exploratory appendix closure.
