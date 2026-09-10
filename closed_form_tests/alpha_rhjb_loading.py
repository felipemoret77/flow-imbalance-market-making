#!/usr/bin/env python3
"""Solve the fast-alpha ergodic RHJB and test loading proposals (writes ONLY here).

Fast-alpha inventory operator = toy operator + diagonal k q alphabar(i)  (eq:fast_alpha_w),
with alphabar(i) = (2 barL eps/zeta) tanh i.  We solve the ergodic RHJB by the SAME Strang
split as the toy (reusing the module's OU step), extract the CLEAN loading
   h_RHJB(i) = (u[q0+1,i] - u[q0-1,i]) / 2          (curvature & cubic cancel at q=0)
and local curvature  r_RHJB(i) = -k (u[q0+1]-2u[q0]+u[q0-1]) , then compare three loadings:
  (old)       paper affine:  A_flow i + A_alpha tanh i,  denom beta+2kappa, LINEAR flow
  (proposal)  external-AI :  A_flow i/sqrt(1+(b/eta^2)i^2) [denom 2k+2.5b] + A_alpha tanh i
                             [denom 2k+beta+eta^2]                (the eta^2 'diffusion penalty')
  (L2-opt)    same sqrt+tanh SHAPE, amplitudes that best fit h_RHJB under the OU weight
              -> the CEILING of the elementary 2-channel form (is the shape even capable?)
No Monte Carlo, no official output touched."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cara_glft_yaware_study as m

P = m.P
Y = m.Y
Q = m.Q
QF = m.QF if hasattr(m, "QF") else Q.astype(float)
NQ = m.NQ
Q0 = m.Q0_IDX
k = P.k
dg = P.delta_gamma
r_G = 2.0 * P.p_glft
kappa = 0.5 * P.rho
beta, eta = P.beta, P.eta
barL = P.bar_lambda
zeta, eps = 0.25, 0.005
a_alpha = 2.0 * barL * eps / zeta
alphabar = a_alpha * np.tanh(Y)


def alpha_inventory_matrix(j):
    """toy inventory matrix at Y[j] + diagonal k q alphabar(Y[j])."""
    lp = P.chi_gamma * P.bar_lambda * (1.0 + np.tanh(Y[j]))
    lm = P.chi_gamma * P.bar_lambda * (1.0 - np.tanh(Y[j]))
    mat = np.diag(-P.alpha_c * QF**2 + k * QF * alphabar[j])
    for i in range(NQ):
        if i > 0:
            mat[i, i - 1] = lp
        if i < NQ - 1:
            mat[i, i + 1] = lm
    return mat


def solve_alpha_rhjb(T=6000.0):
    ou_half = m.build_ou_step(0.5 * P.dt_hjb)
    inv_steps = np.stack([expm(P.dt_hjb * alpha_inventory_matrix(j)) for j in range(P.ny)])
    steps = int(round(T / P.dt_hjb))
    u = np.zeros((NQ, P.ny))
    u = u - u[Q0, m.Y0_IDX]
    prev = None
    for step in range(1, steps + 1):
        u = u @ ou_half.T
        shift = np.max(u, axis=0)
        w = np.exp(np.clip(k * (u - shift[None, :]), -700.0, 700.0))
        for j in range(P.ny):
            w[:, j] = inv_steps[j] @ w[:, j]
        u = np.log(np.maximum(w, 1e-300)) / k + shift[None, :]
        u = u @ ou_half.T
        u = u - u[Q0, m.Y0_IDX]
        if step % P.check_every == 0:
            cur = u[Q0 + 1] - u[Q0 - 1]
            if prev is not None and np.max(np.abs(cur - prev)) < 1e-9:
                break
            prev = cur.copy()
    return u, step


print(f"r_G={r_G:.5f} kappa={kappa:.5f} a_alpha={a_alpha:.4f} eta^2={eta**2:.4f} beta={beta}")
u, nsteps = solve_alpha_rhjb()
print(f"RHJB solved in {nsteps} steps")

h_rhjb = 0.5 * (u[Q0 + 1] - u[Q0 - 1])              # clean loading
r_rhjb = -k * (u[Q0 + 1] - 2 * u[Q0] + u[Q0 - 1])   # local curvature

# --- three loadings ---
Af_old = (2 / k) * kappa / (beta + 2 * kappa)
Aa_old = a_alpha / (beta + 2 * kappa)
Af_new = (2 / k) * kappa / (2 * kappa + 2.5 * beta)
Aa_new = a_alpha / (2 * kappa + beta + eta**2)
flow_shape = Y / np.sqrt(1.0 + (beta / eta**2) * Y**2)
tanh_shape = np.tanh(Y)
h_old = Af_old * Y + Aa_old * tanh_shape
h_new = Af_new * flow_shape + Aa_new * tanh_shape

# L2-optimal amplitudes of the sqrt+tanh shape under OU weight (ceiling of the form)
w_ou = np.exp(-beta * Y**2 / eta**2)
G = np.array([[np.sum(w_ou * flow_shape**2), np.sum(w_ou * flow_shape * tanh_shape)],
              [np.sum(w_ou * flow_shape * tanh_shape), np.sum(w_ou * tanh_shape**2)]])
b = np.array([np.sum(w_ou * flow_shape * h_rhjb), np.sum(w_ou * tanh_shape * h_rhjb)])
Af_opt, Aa_opt = np.linalg.solve(G, b)
h_opt = Af_opt * flow_shape + Aa_opt * tanh_shape

print(f"\namplitudes:  old (Af,Aa)=({Af_old:.4f},{Aa_old:.4f})  "
      f"proposal=({Af_new:.4f},{Aa_new:.4f})  L2opt=({Af_opt:.4f},{Aa_opt:.4f})")

print(f"\n{'i':>4} {'h_RHJB':>9} {'h_old':>8} {'h_proposal':>11} {'h_L2opt':>9}   {'r_RHJB':>8} {'r_G':>7}")
for yv in [0, 1, 2, 3, 4, 5]:
    j = int(np.argmin(np.abs(Y - yv)))
    print(f"{yv:>4} {h_rhjb[j]:9.4f} {h_old[j]:8.4f} {h_new[j]:11.4f} {h_opt[j]:9.4f}   {r_rhjb[j]:8.4f} {r_G:7.4f}")

band = np.abs(Y) <= 4.0

def wrmse(hh):
    return np.sqrt(np.sum(w_ou[band] * (hh - h_rhjb)[band] ** 2) / np.sum(w_ou[band]))

print(f"\nloading wRMSE vs RHJB over |i|<=4 (OU-weighted):")
print(f"  old(affine)={wrmse(h_old):.4f}   proposal={wrmse(h_new):.4f}   L2opt(ceiling)={wrmse(h_opt):.4f}")
print(f"  [L2opt small => the elementary sqrt+tanh SHAPE can capture h_RHJB (Galerkin viable)]")
print(f"  [proposal >> old => the eta^2 'diffusion penalty' loading fits WORSE than the current affine]")

# ---------------------------------------------------------------------------
# Can a SINGLE saturating channel (toy 3.6 form, bigger amplitude) capture h_RHJB?
# This is the cleanest elementary candidate: alpha just enlarges the source.
# ---------------------------------------------------------------------------
from scipy.optimize import least_squares, fsolve
from scipy.integrate import quad

def single(i, A, rho):
    return A * i / np.sqrt(1.0 + (beta / (rho * eta**2)) * i**2)

# L2 (band) fit of the single channel -> ceiling for the clean form
def resid_l2(p):
    A, rho = p
    return np.sqrt(w_ou[band]) * (single(Y[band], A, rho) - h_rhjb[band])
sol = least_squares(resid_l2, [0.3, 2.0], bounds=([0, 0.05], [5, 200]))
A1, rho1 = sol.x
h_single = single(Y, A1, rho1)

# Galerkin moment solve (parameter-free): impose R1 averaged against i, i^3 = 0
def Hc(i, A, rho): return A * i / np.sqrt(1.0 + (beta / (rho * eta**2)) * i**2)
def Hp(i, A, rho):
    d = 1.0 + (beta / (rho * eta**2)) * i**2
    return A / np.sqrt(d) - A * i * ((beta / (rho * eta**2)) * i) / d**1.5
def Hpp(i, A, rho, e=1e-5): return (Hp(i + e, A, rho) - Hp(i - e, A, rho)) / (2 * e)
def R1(i, A, rho):
    h = Hc(i, A, rho); th = i - k * h
    r = r_G * np.sqrt(np.cosh(i) / np.cosh(th))
    src = a_alpha * np.tanh(i) + (2 * kappa / k) * np.sinh(th) / np.sqrt(np.cosh(i) * np.cosh(th)) * np.exp(-r / 2)
    return 0.5 * eta**2 * Hpp(i, A, rho) - beta * i * Hp(i, A, rho) + src
def proj(A, rho, n): return quad(lambda i: R1(i, A, rho) * i**n * np.exp(-beta * i**2 / eta**2), 0, 30, limit=200)[0]
gal = fsolve(lambda v: [proj(v[0], v[1], 1), proj(v[0], v[1], 3)], [0.6, 5.0], full_output=True)
Ag, rhog = float(gal[0][0]), float(gal[0][1])
h_gal = single(Y, Ag, rhog)

print(f"\nsingle saturating channel:  L2-fit (A,rho)=({A1:.4f},{rho1:.3f})  "
      f"Galerkin (A,rho)=({Ag:.4f},{rhog:.3f})")
print(f"  loading wRMSE |i|<=4:  single-L2={wrmse(h_single):.4f}   single-Galerkin={wrmse(h_gal):.4f}")

# ---- bottom line: QUOTE RMSE vs RHJB over the populated band ----
ask_rhjb = np.full((NQ, P.ny), np.nan); bid_rhjb = np.full((NQ, P.ny), np.nan)
ask_rhjb[1:] = dg - u[:-1] + u[1:]; bid_rhjb[:-1] = dg - u[1:] + u[:-1]
qcol = QF[:, None]

def quotes(h_vec, cubic):
    th = Y - k * h_vec
    if cubic:
        r = r_G * np.sqrt(np.cosh(Y) / np.cosh(th)); s = (r**2 / 6) * np.tanh(th)
    else:
        r = np.full_like(Y, r_G); s = np.zeros_like(Y)
    a = dg + h_vec[None, :] - (r[None, :] / (2 * k)) * (2 * qcol - 1) + (s[None, :] / k) * (qcol**2 - qcol + 1/3)
    b = dg - h_vec[None, :] + (r[None, :] / (2 * k)) * (2 * qcol + 1) - (s[None, :] / k) * (qcol**2 + qcol + 1/3)
    return a, b

qband = (np.abs(QF) <= 16)[:, None] & (np.abs(Y) <= 4.0)[None, :]
def qrmse(h_vec, cubic):
    a, b = quotes(h_vec, cubic)
    e = np.concatenate([(a - ask_rhjb)[qband & np.isfinite(ask_rhjb)],
                        (b - bid_rhjb)[qband & np.isfinite(bid_rhjb)]])
    return np.sqrt(np.mean(e**2))

print(f"\nQUOTE RMSE vs RHJB, |q|<=16 & |i|<=4 (ask+bid pooled):")
print(f"  old affine (r_G, h_old)              : {qrmse(h_old, False):.4f}   <- current paper")
print(f"  proposal (cubic, h_proposal)         : {qrmse(h_new, True):.4f}")
print(f"  cubic + single-Galerkin loading      : {qrmse(h_gal, True):.4f}   <- candidate")
print(f"  cubic + L2-opt single loading (ceil) : {qrmse(h_single, True):.4f}")

# ===========================================================================
# Can we stay in the SPIRIT OF 3.5 -- a purely ALGEBRAIC loading, no offline solve?
# All amplitudes below are closed-form expressions in the model constants.
# ===========================================================================
# (i) single saturating channel, FIXED toy scale rho=1, amplitude = i^1 match with
#     alpha folded into the source:  A = (2k/k + a_alpha)/(2kappa + 5beta/2)
A_alg = (2 * kappa / k + a_alpha) / (2 * kappa + 2.5 * beta)
h_alg1 = single(Y, A_alg, 1.0)
# (ii) pure linear loading, i^1-matched slope:  s = (2kappa/k + a_alpha)/(beta + 2kappa)
s_lin = (2 * kappa / k + a_alpha) / (beta + 2 * kappa)
h_lin = s_lin * Y
print(f"\nALGEBRAIC candidates:  single-rho1 A={A_alg:.4f}   linear slope={s_lin:.4f}")
print(f"  loading wRMSE |i|<=4:  single-rho1={wrmse(h_alg1):.4f}   linear={wrmse(h_lin):.4f}")
print(f"  (recall: old affine={wrmse(h_old):.4f}  Galerkin(offline)={wrmse(h_gal):.4f})")

print(f"\nQUOTE RMSE vs RHJB (|q|<=16,|i|<=4), all with cubic r(i),s(i) ON unless noted:")
print(f"  CURRENT paper (affine, no cubic)        : {qrmse(h_old, False):.4f}")
print(f"  ALGEBRAIC single-channel rho=1 (3.5-style): {qrmse(h_alg1, True):.4f}")
print(f"  ALGEBRAIC linear i^1-slope               : {qrmse(h_lin, True):.4f}")
print(f"  ALGEBRAIC proposal (sqrt+tanh decoupled) : {qrmse(h_new, True):.4f}")
print(f"  OFFLINE Galerkin single channel          : {qrmse(h_gal, True):.4f}")
# how much of the gap is loading vs curvature? show single-rho1 with cubic OFF too
print(f"\n  [isolating]: ALGEBRAIC single rho=1, cubic OFF (affine) : {qrmse(h_alg1, False):.4f}")

# ===========================================================================
# ISOLATE the cubic: same (good) Galerkin loading, affine r_G vs full r(i),s(i).
# ===========================================================================
print("\n=== does the CUBIC ansatz help, holding the loading fixed? ===")
print(f"  Galerkin loading, AFFINE r_G (no cubic) : {qrmse(h_gal, False):.4f}")
print(f"  Galerkin loading, CUBIC r(i),s(i)       : {qrmse(h_gal, True):.4f}")
print(f"  old affine loading, AFFINE r_G          : {qrmse(h_old, False):.4f}")
print(f"  old affine loading, CUBIC r(i),s(i)     : {qrmse(h_old, True):.4f}")

# per-i-slice (ask side), Galerkin loading, to show WHERE the cubic matters
print(f"\n  per-slice ask RMSE (Galerkin loading):   {'i':>3} {'affine r_G':>11} {'cubic r(i)':>11}  r_RHJB")
for yv in [0, 1, 2, 3, 4]:
    j = int(np.argmin(np.abs(Y - yv)))
    qm = (np.abs(QF) <= 16)
    a_aff, _ = quotes(h_gal, False); a_cub, _ = quotes(h_gal, True)
    e_aff = np.sqrt(np.nanmean(((a_aff[:, j] - ask_rhjb[:, j])[qm]) ** 2))
    e_cub = np.sqrt(np.nanmean(((a_cub[:, j] - ask_rhjb[:, j])[qm]) ** 2))
    print(f"  {'':>40} {yv:>3} {e_aff:11.4f} {e_cub:11.4f}  {r_rhjb[j]:.4f}")

# ===========================================================================
# Intermediate ansatz: r(i) curvature but NO cubic (quote still affine in q).
# ===========================================================================
def quotes_mode(h_vec, mode):  # mode: 'rG' | 'ri' | 'cubic'
    th = Y - k * h_vec
    if mode == 'rG':
        r = np.full_like(Y, r_G); s = np.zeros_like(Y)
    elif mode == 'ri':
        r = r_G * np.sqrt(np.cosh(Y) / np.cosh(th)); s = np.zeros_like(Y)
    else:
        r = r_G * np.sqrt(np.cosh(Y) / np.cosh(th)); s = (r ** 2 / 6) * np.tanh(th)
    a = dg + h_vec[None, :] - (r[None, :] / (2 * k)) * (2 * qcol - 1) + (s[None, :] / k) * (qcol**2 - qcol + 1/3)
    b = dg - h_vec[None, :] + (r[None, :] / (2 * k)) * (2 * qcol + 1) - (s[None, :] / k) * (qcol**2 + qcol + 1/3)
    return a, b
def qrmse_mode(h_vec, mode):
    a, b = quotes_mode(h_vec, mode)
    e = np.concatenate([(a - ask_rhjb)[qband & np.isfinite(ask_rhjb)], (b - bid_rhjb)[qband & np.isfinite(bid_rhjb)]])
    return np.sqrt(np.mean(e**2))
print("\n=== FAST-ALPHA: r_G vs r(i)-no-cubic vs cubic (Galerkin loading, |q|<=16,|i|<=4) ===")
for mode, lab in [('rG','affine r_G'), ('ri','r(i), NO cubic'), ('cubic','r(i)+cubic s(i)')]:
    print(f"  {lab:18s}: {qrmse_mode(h_gal, mode):.4f}")

# ===========================================================================
# ELEMENTARY-ONLY table (no Galerkin): each algebraic loading x {rG, r(i), r(i)+cubic}
# ===========================================================================
print("\n=== FAST-ALPHA elementary loadings x curvature (|q|<=16,|i|<=4) ===")
elem = [("paper (linear flow + tanh)", h_old),
        ("single-channel sqrt rho=1", h_alg1),
        ("pure linear i^1-slope", h_lin)]
print(f"  {'loading':28s} {'affine r_G':>11} {'r(i) no cubic':>14} {'r(i)+cubic':>11}")
for lab, hv in elem:
    print(f"  {lab:28s} {qrmse_mode(hv,'rG'):11.4f} {qrmse_mode(hv,'ri'):14.4f} {qrmse_mode(hv,'cubic'):11.4f}")
