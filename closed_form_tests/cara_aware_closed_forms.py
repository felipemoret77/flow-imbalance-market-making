"""Elementary SRN and leading-CARA-aware closed-form quote APIs.

This module is deliberately isolated from the production study scripts.  In
``srn`` mode it reproduces the current toy cubic-skew and fast-alpha SC-cubic
policies.  In ``leading_cara`` mode it keeps their loadings, gates, and
expansion angles fixed and applies only the leading signal-risk curvature
correction

    r -> r * sqrt(1 + eta**2 * (d h / d i)**2 / sigma**2),

then recomputes the cubic coefficient from the corrected curvature.  This is a
leading CARA-aware closure, not the solution of the exact CARA HJB.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cara_glft_yaware_study as _toy_legacy
from alpha_studies import alpha_reduced_cole_hopf as _fast_legacy


RiskMode = Literal["srn", "leading_cara"]


@dataclass(frozen=True)
class ClosedFormProfiles:
    """State profiles used by either closed-form quote map.

    All fields have the broadcast shape of the input imbalance state.  The
    ``r_srn`` field is always the uncorrected curvature, while ``r`` and ``s``
    correspond to the selected risk mode.
    """

    shift: np.ndarray
    theta_flow: np.ndarray
    theta_full: np.ndarray
    m_flow: np.ndarray
    dshift_di: np.ndarray
    sigma_eff: np.ndarray
    risk_factor: np.ndarray
    r_srn: np.ndarray
    r: np.ndarray
    s: np.ndarray


def _validate_risk_mode(risk_mode: RiskMode) -> None:
    if risk_mode not in ("srn", "leading_cara"):
        raise ValueError(
            f"unknown risk_mode {risk_mode!r}; expected 'srn' or 'leading_cara'"
        )


def _sech2(x: np.ndarray) -> np.ndarray:
    return 1.0 / np.cosh(x) ** 2


def _risk_adjustment(
    dshift_di: np.ndarray,
    *,
    eta: float,
    sigma: float,
    risk_mode: RiskMode,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_risk_mode(risk_mode)
    dshift_di = np.asarray(dshift_di, dtype=float)
    if risk_mode == "srn":
        return np.full_like(dshift_di, sigma), np.ones_like(dshift_di)
    sigma_eff = np.sqrt(sigma**2 + eta**2 * dshift_di**2)
    return sigma_eff, sigma_eff / sigma


def toy_profiles(
    i: np.ndarray | float,
    *,
    params: Any = _toy_legacy.P,
    risk_mode: RiskMode = "srn",
) -> ClosedFormProfiles:
    """Profiles for the paper's toy cubic-skew closed form.

    The leading-CARA derivative is the derivative of the toy value loading
    ``h_I``.  The loading itself and the flow angle are identical in both risk
    modes.
    """

    state = np.asarray(i, dtype=float)
    a2 = np.sqrt(params.beta) / params.eta
    amplitude = params.A_I / a2
    shift = amplitude * np.tanh(a2 * state)
    dshift_di = params.A_I * _sech2(a2 * state)

    theta = state - params.k * shift
    m_flow = np.sqrt(np.cosh(state) / np.cosh(theta))
    r_srn = (2.0 * params.p_glft) * m_flow
    sigma_eff, risk_factor = _risk_adjustment(
        dshift_di,
        eta=params.eta,
        sigma=params.sigma,
        risk_mode=risk_mode,
    )
    r = r_srn * risk_factor
    s = (r**2 / 6.0) * np.tanh(theta)

    return ClosedFormProfiles(
        shift=shift,
        theta_flow=theta,
        theta_full=theta,
        m_flow=m_flow,
        dshift_di=dshift_di,
        sigma_eff=sigma_eff,
        risk_factor=risk_factor,
        r_srn=r_srn,
        r=r,
        s=s,
    )


def toy_depths(
    q: np.ndarray | float,
    i: np.ndarray | float,
    *,
    params: Any = _toy_legacy.P,
    risk_mode: RiskMode = "srn",
) -> tuple[np.ndarray, np.ndarray]:
    """Toy ask/bid signed depths, broadcast over ``q`` and ``i``."""

    inventory, state = np.broadcast_arrays(
        np.asarray(q), np.asarray(i, dtype=float)
    )
    qf = inventory.astype(float)
    profile = toy_profiles(state, params=params, risk_mode=risk_mode)
    ask = (
        params.delta_gamma
        + profile.shift
        - (profile.r / (2.0 * params.k)) * (2.0 * qf - 1.0)
        + (profile.s / params.k) * (qf**2 - qf + 1.0 / 3.0)
    )
    bid = (
        params.delta_gamma
        - profile.shift
        + (profile.r / (2.0 * params.k)) * (2.0 * qf + 1.0)
        - (profile.s / params.k) * (qf**2 + qf + 1.0 / 3.0)
    )
    ask = np.where(inventory > -params.qmax, ask, np.inf)
    bid = np.where(inventory < params.qmax, bid, np.inf)
    return ask, bid


def fast_alpha_profiles(
    i: np.ndarray | float,
    *,
    params: Any = _fast_legacy.P,
    risk_mode: RiskMode = "srn",
) -> ClosedFormProfiles:
    """Profiles for the paper's fast-alpha self-consistent cubic.

    The signal-risk derivative is the analytic derivative of the full quoted
    shift, including the flow multiplier and alpha gate.  The leading-CARA
    factor is *not* fed back into that multiplier.
    """

    state = np.asarray(i, dtype=float)
    a2 = np.sqrt(params.beta) / params.eta
    toy_flow_slope = (params.rho / params.k) / (
        params.rho + 2.0 * params.beta
    )
    a1 = toy_flow_slope / a2
    alpha_loading = params.alpha_slope / (params.beta + params.rho)

    tanh_flow = np.tanh(a2 * state)
    sech2_flow = _sech2(a2 * state)
    h_flow = a1 * tanh_flow
    dh_flow = a1 * a2 * sech2_flow

    h_alpha = alpha_loading * np.tanh(state)
    dh_alpha = alpha_loading * _sech2(state)

    theta_flow = state - params.k * h_flow
    dtheta_flow = 1.0 - params.k * dh_flow
    m_flow = np.sqrt(np.cosh(state) / np.cosh(theta_flow))
    dm_flow = 0.5 * m_flow * (
        np.tanh(state) - dtheta_flow * np.tanh(theta_flow)
    )

    gate = tanh_flow**2
    dgate = 2.0 * a2 * tanh_flow * sech2_flow
    alpha_weight = 1.0 + gate * (m_flow - 1.0)
    shift = m_flow * h_flow + alpha_weight * h_alpha
    dshift_di = (
        dm_flow * (h_flow + gate * h_alpha)
        + m_flow * dh_flow
        + alpha_weight * dh_alpha
        + (m_flow - 1.0) * dgate * h_alpha
    )

    theta_full = state - params.k * shift
    r_srn = (2.0 * params.p_glft) * m_flow
    sigma_eff, risk_factor = _risk_adjustment(
        dshift_di,
        eta=params.eta,
        sigma=params.sigma_price,
        risk_mode=risk_mode,
    )
    r = r_srn * risk_factor
    s = (r**2 / 6.0) * np.tanh(theta_full)

    return ClosedFormProfiles(
        shift=shift,
        theta_flow=theta_flow,
        theta_full=theta_full,
        m_flow=m_flow,
        dshift_di=dshift_di,
        sigma_eff=sigma_eff,
        risk_factor=risk_factor,
        r_srn=r_srn,
        r=r,
        s=s,
    )


def fast_alpha_depths(
    q: np.ndarray | float,
    i: np.ndarray | float,
    *,
    params: Any = _fast_legacy.P,
    risk_mode: RiskMode = "srn",
) -> tuple[np.ndarray, np.ndarray]:
    """Fast-alpha SC-cubic ask/bid signed depths, with broadcasting."""

    inventory, state = np.broadcast_arrays(
        np.asarray(q), np.asarray(i, dtype=float)
    )
    qf = inventory.astype(float)
    profile = fast_alpha_profiles(state, params=params, risk_mode=risk_mode)
    ask = (
        params.delta0
        + profile.shift
        - (profile.r / (2.0 * params.k)) * (2.0 * qf - 1.0)
        + (profile.s / params.k) * (qf**2 - qf + 1.0 / 3.0)
    )
    bid = (
        params.delta0
        - profile.shift
        + (profile.r / (2.0 * params.k)) * (2.0 * qf + 1.0)
        - (profile.s / params.k) * (qf**2 + qf + 1.0 / 3.0)
    )
    ask = np.where(inventory > -params.qmax, ask, np.inf)
    bid = np.where(inventory < params.qmax, bid, np.inf)
    return ask, bid


__all__ = [
    "ClosedFormProfiles",
    "RiskMode",
    "fast_alpha_depths",
    "fast_alpha_profiles",
    "toy_depths",
    "toy_profiles",
]
