#!/usr/bin/env python3
"""Paired Monte Carlo for current versus leading-CARA-aware closed forms.

The two policies see identical exogenous paths.  Their policy-dependent fills
are strongly coupled with a common Poisson component plus an excess component
for the policy with the larger intensity.  The script is deliberately separate
from the production engines so adding a policy cannot perturb the environment.

Examples
--------
Smoke test::

    python closed_form_tests/paired_cara_closed_form_mc.py \
        --model toy --paths 100 --horizon 20 --dt 0.05

Paper-horizon run::

    python -u closed_form_tests/paired_cara_closed_form_mc.py \
        --model fast-alpha --paths 2000 --horizon 10000 --dt 0.05
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import cara_glft_yaware_study as toy_legacy
from alpha_studies import alpha_reduced_cole_hopf as fast_legacy
from closed_form_tests.cara_aware_closed_forms import (
    RiskMode,
    fast_alpha_depths,
    toy_depths,
)


POLICY_NAMES = ("srn", "leading_cara")


@dataclass(frozen=True)
class RunConfig:
    model: Literal["toy", "fast-alpha"]
    paths: int
    horizon: float
    dt: float
    record_dt: float
    seed: int
    right_mode: RiskMode
    bootstrap: int
    policy_i_max: float
    policy_i_step: float
    alpha_min: float
    alpha_max: float
    alpha_bins: int


@dataclass
class CouplingDiagnostics:
    quoted: int = 0
    negative_depth: int = 0
    raw_multiple_fills: int = 0
    capped_fills: int = 0
    saturated_shortcut: int = 0

    def update(self, other: "CouplingDiagnostics") -> None:
        self.quoted += other.quoted
        self.negative_depth += other.negative_depth
        self.raw_multiple_fills += other.raw_multiple_fills
        self.capped_fills += other.capped_fills
        self.saturated_shortcut += other.saturated_shortcut


def standard_error(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def stable_ce(values: np.ndarray, gamma: float) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    logw = -gamma * values
    shift = float(np.max(logw))
    weights = np.exp(logw - shift)
    mean_weight = float(np.mean(weights))
    ce = -(shift + np.log(mean_weight)) / gamma
    sumw = float(np.sum(weights))
    ess = sumw**2 / float(np.sum(weights**2))
    max_share = float(np.max(weights) / sumw)
    return float(ce), float(ess), max_share


def paired_summary(right: np.ndarray, left: np.ndarray) -> dict[str, float]:
    right = np.asarray(right, dtype=float)
    left = np.asarray(left, dtype=float)
    delta = right - left
    se = standard_error(delta)
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        corr = 1.0 if np.array_equal(left, right) else float("nan")
    else:
        corr = float(np.corrcoef(left, right)[0, 1])
    return {
        "left_mean": float(np.mean(left)),
        "right_mean": float(np.mean(right)),
        "paired_difference_mean": float(np.mean(delta)),
        "paired_difference_se": se,
        "paired_ci95_low": float(np.mean(delta) - 1.96 * se),
        "paired_ci95_high": float(np.mean(delta) + 1.96 * se),
        "paired_correlation": corr,
        "unpaired_difference_se": float(
            np.sqrt(np.var(left, ddof=1) / len(left) + np.var(right, ddof=1) / len(right))
        )
        if len(left) > 1
        else 0.0,
    }


def make_rngs(seed: int) -> dict[str, np.random.Generator]:
    names = (
        "price",
        "imbalance",
        "alpha_noise",
        "aggregate_buy",
        "aggregate_sell",
        "ask_common",
        "ask_left",
        "ask_right",
        "bid_common",
        "bid_left",
        "bid_right",
        "bootstrap",
    )
    children = np.random.SeedSequence(seed).spawn(len(names))
    return {name: np.random.default_rng(child) for name, child in zip(names, children)}


def _single_capped_poisson(
    rng: np.random.Generator,
    mean: np.ndarray,
    cap: np.ndarray,
) -> tuple[np.ndarray, CouplingDiagnostics]:
    mean = np.where(np.isfinite(mean) & (mean > 0.0), mean, 0.0)
    cap = np.maximum(np.asarray(cap, dtype=int), 0)
    result = np.zeros_like(cap)
    diag = CouplingDiagnostics(quoted=int(mean.size))
    active = cap > 0
    saturated = active & (mean > cap + 10.0 * np.sqrt(np.maximum(mean, 1.0)))
    result[saturated] = cap[saturated]
    diag.saturated_shortcut = int(np.count_nonzero(saturated))
    sampled = active & ~saturated & (mean > 0.0)
    if np.any(sampled):
        raw = rng.poisson(mean[sampled])
        diag.raw_multiple_fills = int(np.count_nonzero(raw > 1))
        diag.capped_fills = int(np.count_nonzero(raw > cap[sampled]))
        result[sampled] = np.minimum(raw, cap[sampled])
    return result, diag


def coupled_capped_poisson(
    mean_left: np.ndarray,
    mean_right: np.ndarray,
    cap_left: np.ndarray,
    cap_right: np.ndarray,
    rng_common: np.random.Generator,
    rng_left: np.random.Generator,
    rng_right: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, CouplingDiagnostics]:
    """Common-component Poisson coupling with legacy tail protection."""

    mean_left = np.where(np.isfinite(mean_left) & (mean_left > 0.0), mean_left, 0.0)
    mean_right = np.where(np.isfinite(mean_right) & (mean_right > 0.0), mean_right, 0.0)
    cap_left = np.maximum(np.asarray(cap_left, dtype=int), 0)
    cap_right = np.maximum(np.asarray(cap_right, dtype=int), 0)
    left = np.zeros_like(cap_left)
    right = np.zeros_like(cap_right)
    diag = CouplingDiagnostics(quoted=int(2 * mean_left.size))

    sat_left = (cap_left > 0) & (
        mean_left > cap_left + 10.0 * np.sqrt(np.maximum(mean_left, 1.0))
    )
    sat_right = (cap_right > 0) & (
        mean_right > cap_right + 10.0 * np.sqrt(np.maximum(mean_right, 1.0))
    )
    left[sat_left] = cap_left[sat_left]
    right[sat_right] = cap_right[sat_right]
    diag.saturated_shortcut = int(np.count_nonzero(sat_left) + np.count_nonzero(sat_right))

    pair = ~sat_left & ~sat_right & ((cap_left > 0) | (cap_right > 0))
    if np.any(pair):
        ml = np.where(cap_left[pair] > 0, mean_left[pair], 0.0)
        mr = np.where(cap_right[pair] > 0, mean_right[pair], 0.0)
        common_mean = np.minimum(ml, mr)
        common = rng_common.poisson(common_mean)
        extra_left = np.zeros_like(common)
        extra_right = np.zeros_like(common)
        dl = ml - common_mean
        dr = mr - common_mean
        has_left = dl > 0.0
        has_right = dr > 0.0
        if np.any(has_left):
            extra_left[has_left] = rng_left.poisson(dl[has_left])
        if np.any(has_right):
            extra_right[has_right] = rng_right.poisson(dr[has_right])
        raw_left = common + extra_left
        raw_right = common + extra_right
        cl = cap_left[pair]
        cr = cap_right[pair]
        diag.raw_multiple_fills += int(
            np.count_nonzero(raw_left > 1) + np.count_nonzero(raw_right > 1)
        )
        diag.capped_fills += int(
            np.count_nonzero(raw_left > cl) + np.count_nonzero(raw_right > cr)
        )
        left[pair] = np.minimum(raw_left, cl)
        right[pair] = np.minimum(raw_right, cr)

    only_left_sat = sat_left & ~sat_right
    if np.any(only_left_sat):
        sampled, extra_diag = _single_capped_poisson(
            rng_right, mean_right[only_left_sat], cap_right[only_left_sat]
        )
        right[only_left_sat] = sampled
        diag.raw_multiple_fills += extra_diag.raw_multiple_fills
        diag.capped_fills += extra_diag.capped_fills
    only_right_sat = sat_right & ~sat_left
    if np.any(only_right_sat):
        sampled, extra_diag = _single_capped_poisson(
            rng_left, mean_left[only_right_sat], cap_left[only_right_sat]
        )
        left[only_right_sat] = sampled
        diag.raw_multiple_fills += extra_diag.raw_multiple_fills
        diag.capped_fills += extra_diag.capped_fills
    return left, right, diag


class DepthLookup:
    def __init__(
        self,
        depth_fn: Callable[..., tuple[np.ndarray, np.ndarray]],
        qmax: int,
        i_max: float,
        i_step: float,
        right_mode: RiskMode,
    ) -> None:
        self.qmax = qmax
        self.i_max = float(i_max)
        n_i = int(round(2.0 * i_max / i_step)) + 1
        self.i_grid = np.linspace(-i_max, i_max, n_i)
        self.i_step = float(self.i_grid[1] - self.i_grid[0])
        q_grid = np.arange(-qmax, qmax + 1, dtype=int)[:, None]
        i_grid = self.i_grid[None, :]
        self.ask = np.empty((2, 2 * qmax + 1, n_i), dtype=float)
        self.bid = np.empty_like(self.ask)
        for p_idx, mode in enumerate(("srn", right_mode)):
            self.ask[p_idx], self.bid[p_idx] = depth_fn(q_grid, i_grid, risk_mode=mode)

    def depths(
        self,
        policy: int,
        q: np.ndarray,
        i: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        clipped = np.clip(i, -self.i_max, self.i_max)
        off_grid = int(np.count_nonzero(clipped != i))
        position = (clipped + self.i_max) / self.i_step
        lower = np.floor(position).astype(int)
        lower = np.clip(lower, 0, len(self.i_grid) - 2)
        frac = position - lower
        q_idx = np.asarray(q, dtype=int) + self.qmax
        ask0 = self.ask[policy, q_idx, lower]
        ask1 = self.ask[policy, q_idx, lower + 1]
        bid0 = self.bid[policy, q_idx, lower]
        bid1 = self.bid[policy, q_idx, lower + 1]
        ask_forbidden = np.isposinf(ask0) & np.isposinf(ask1)
        bid_forbidden = np.isposinf(bid0) & np.isposinf(bid1)
        with np.errstate(invalid="ignore"):
            ask = ask0 + frac * (ask1 - ask0)
            bid = bid0 + frac * (bid1 - bid0)
        ask = np.where(ask_forbidden, np.inf, ask)
        bid = np.where(bid_forbidden, np.inf, bid)
        return ask, bid, off_grid


def fill_mean(lambda0: np.ndarray, depth: np.ndarray, k: float, dt: float) -> np.ndarray:
    if np.any(np.isnan(depth)) or np.any(np.isneginf(depth)):
        raise FloatingPointError("depth contains NaN or -inf")
    multiplier = np.exp(np.clip(-k * depth, -60.0, 60.0))
    return np.where(np.isfinite(depth), lambda0 * multiplier * dt, 0.0)


def init_path_arrays(n: int) -> dict[str, np.ndarray]:
    return {
        "spread_capture": np.zeros((2, n)),
        "alpha_carry": np.zeros((2, n)),
        "price_martingale": np.zeros((2, n)),
        "running_penalty": np.zeros((2, n)),
        "fills": np.zeros((2, n)),
        "abs_q_integral": np.zeros((2, n)),
        "q2_integral": np.zeros((2, n)),
        "min_ask": np.full((2, n), np.inf),
        "min_bid": np.full((2, n), np.inf),
    }


def simulate(config: RunConfig) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if config.model == "toy":
        params = toy_legacy.P
        depth_fn = toy_depths
        sigma = params.sigma
        qmax = params.qmax
        gamma = params.gamma
        terminal_penalty_coeff = 0.0
    else:
        params = fast_legacy.P
        depth_fn = fast_alpha_depths
        sigma = params.sigma_price
        qmax = params.qmax
        gamma = params.gamma
        terminal_penalty_coeff = params.alpha_l

    lookup = DepthLookup(
        depth_fn,
        qmax=qmax,
        i_max=config.policy_i_max,
        i_step=config.policy_i_step,
        right_mode=config.right_mode,
    )
    rng = make_rngs(config.seed)
    n = config.paths
    steps = int(round(config.horizon / config.dt))
    record_every = max(1, int(round(config.record_dt / config.dt)))
    sqrt_dt = np.sqrt(config.dt)
    q = np.zeros((2, n), dtype=np.int32)
    state_i = np.zeros(n)
    alpha = np.zeros(n)
    mid = np.zeros(n)
    cash = np.zeros((2, n))
    path = init_path_arrays(n)
    diag = CouplingDiagnostics()
    off_grid = 0
    boundary_visits = np.zeros(2, dtype=np.int64)

    record_times: list[float] = []
    record_mean_pnl: list[np.ndarray] = []
    record_mean_penalized: list[np.ndarray] = []
    record_mean_absq: list[np.ndarray] = []

    alpha_edges = np.linspace(config.alpha_min, config.alpha_max, config.alpha_bins + 1)
    alpha_centers = 0.5 * (alpha_edges[:-1] + alpha_edges[1:])
    conditional_count = np.zeros(config.alpha_bins, dtype=np.int64)
    conditional_ask_count = np.zeros((2, config.alpha_bins), dtype=np.int64)
    conditional_bid_count = np.zeros((2, config.alpha_bins), dtype=np.int64)
    conditional_ask_sum = np.zeros((2, config.alpha_bins))
    conditional_bid_sum = np.zeros((2, config.alpha_bins))
    conditional_alpha_under = 0
    conditional_alpha_over = 0

    start = time.monotonic()
    progress_every = max(1, steps // 20)
    for step in range(steps):
        q_before = q.copy()
        boundary_visits += np.count_nonzero(np.abs(q_before) == qmax, axis=1)
        lambda_ask0 = params.bar_lambda * (1.0 + np.tanh(state_i))
        lambda_bid0 = params.bar_lambda * (1.0 - np.tanh(state_i))

        asks = np.empty((2, n))
        bids = np.empty((2, n))
        for p_idx in range(2):
            asks[p_idx], bids[p_idx], clipped = lookup.depths(
                p_idx, q_before[p_idx], state_i
            )
            off_grid += clipped
        diag.negative_depth += int(
            np.count_nonzero(asks < 0.0) + np.count_nonzero(bids < 0.0)
        )
        path["min_ask"] = np.minimum(path["min_ask"], asks)
        path["min_bid"] = np.minimum(path["min_bid"], bids)

        ask_mean = np.stack(
            [fill_mean(lambda_ask0, asks[p], params.k, config.dt) for p in range(2)]
        )
        bid_mean = np.stack(
            [fill_mean(lambda_bid0, bids[p], params.k, config.dt) for p in range(2)]
        )
        ask_left, ask_right, ask_diag = coupled_capped_poisson(
            ask_mean[0],
            ask_mean[1],
            q_before[0] + qmax,
            q_before[1] + qmax,
            rng["ask_common"],
            rng["ask_left"],
            rng["ask_right"],
        )
        bid_left, bid_right, bid_diag = coupled_capped_poisson(
            bid_mean[0],
            bid_mean[1],
            qmax - q_before[0],
            qmax - q_before[1],
            rng["bid_common"],
            rng["bid_left"],
            rng["bid_right"],
        )
        diag.update(ask_diag)
        diag.update(bid_diag)
        ask_fill = np.stack((ask_left, ask_right))
        bid_fill = np.stack((bid_left, bid_right))
        path["fills"] += ask_fill + bid_fill
        path["abs_q_integral"] += np.abs(q_before) * config.dt
        path["q2_integral"] += q_before.astype(float) ** 2 * config.dt
        phi = 0.5 * gamma * sigma**2
        path["running_penalty"] += phi * q_before.astype(float) ** 2 * config.dt

        if config.model == "toy":
            ask_exec = np.where(np.isfinite(asks), asks, 0.0)
            bid_exec = np.where(np.isfinite(bids), bids, 0.0)
            cash += (mid[None, :] + ask_exec) * ask_fill
            cash -= (mid[None, :] - bid_exec) * bid_fill
            path["spread_capture"] += ask_exec * ask_fill + bid_exec * bid_fill
            q += -ask_fill + bid_fill
            price_noise = sigma * sqrt_dt * rng["price"].standard_normal(n)
            path["price_martingale"] += q.astype(float) * price_noise
            mid += price_noise
            state_i += (
                -params.beta * state_i * config.dt
                + params.eta * sqrt_dt * rng["imbalance"].standard_normal(n)
            )
        else:
            alpha_before = alpha.copy()
            price_noise = sigma * sqrt_dt * rng["price"].standard_normal(n)
            ask_exec = np.where(np.isfinite(asks), asks, 0.0)
            bid_exec = np.where(np.isfinite(bids), bids, 0.0)
            path["spread_capture"] += ask_exec * ask_fill + bid_exec * bid_fill
            path["alpha_carry"] += q_before.astype(float) * alpha_before * config.dt
            path["price_martingale"] += q_before.astype(float) * price_noise
            q += -ask_fill + bid_fill

            aggregate_buy = rng["aggregate_buy"].poisson(lambda_ask0 * config.dt)
            aggregate_sell = rng["aggregate_sell"].poisson(lambda_bid0 * config.dt)
            alpha += (
                -params.zeta * alpha * config.dt
                + params.eta_alpha * sqrt_dt * rng["alpha_noise"].standard_normal(n)
                + params.epsilon * aggregate_buy
                - params.epsilon * aggregate_sell
            )
            state_i += (
                -params.beta * state_i * config.dt
                + params.eta * sqrt_dt * rng["imbalance"].standard_normal(n)
            )

        if (step + 1) % record_every == 0 or step == steps - 1:
            if config.model == "toy":
                total_now = cash + q.astype(float) * mid[None, :]
            else:
                total_now = (
                    path["spread_capture"]
                    + path["alpha_carry"]
                    + path["price_martingale"]
                )
            terminal_penalty_now = terminal_penalty_coeff * q.astype(float) ** 2
            penalized_now = total_now - path["running_penalty"] - terminal_penalty_now
            record_times.append((step + 1) * config.dt)
            record_mean_pnl.append(np.mean(total_now, axis=1))
            record_mean_penalized.append(np.mean(penalized_now, axis=1))
            record_mean_absq.append(np.mean(np.abs(q), axis=1))

            if config.model == "fast-alpha":
                bin_idx = np.digitize(alpha, alpha_edges) - 1
                conditional_alpha_under += int(np.count_nonzero(bin_idx < 0))
                conditional_alpha_over += int(np.count_nonzero(bin_idx >= config.alpha_bins))
                valid = (bin_idx >= 0) & (bin_idx < config.alpha_bins)
                conditional_count += np.bincount(
                    bin_idx[valid], minlength=config.alpha_bins
                )
                for p_idx in range(2):
                    a_depth, b_depth, _ = lookup.depths(p_idx, q[p_idx], state_i)
                    ask_valid = valid & np.isfinite(a_depth)
                    bid_valid = valid & np.isfinite(b_depth)
                    conditional_ask_count[p_idx] += np.bincount(
                        bin_idx[ask_valid], minlength=config.alpha_bins
                    )
                    conditional_bid_count[p_idx] += np.bincount(
                        bin_idx[bid_valid], minlength=config.alpha_bins
                    )
                    conditional_ask_sum[p_idx] += np.bincount(
                        bin_idx[ask_valid],
                        weights=a_depth[ask_valid],
                        minlength=config.alpha_bins,
                    )
                    conditional_bid_sum[p_idx] += np.bincount(
                        bin_idx[bid_valid],
                        weights=b_depth[bid_valid],
                        minlength=config.alpha_bins,
                    )

        if (step + 1) % progress_every == 0 or step == steps - 1:
            elapsed = time.monotonic() - start
            fraction = (step + 1) / steps
            eta_seconds = elapsed * (1.0 / fraction - 1.0)
            print(
                f"[{config.model}] {100*fraction:5.1f}% "
                f"elapsed={elapsed/60:.1f}m eta={eta_seconds/60:.1f}m",
                flush=True,
            )

    if config.model == "toy":
        terminal_pnl = cash + q.astype(float) * mid[None, :]
    else:
        terminal_pnl = (
            path["spread_capture"] + path["alpha_carry"] + path["price_martingale"]
        )
    terminal_penalty = terminal_penalty_coeff * q.astype(float) ** 2
    penalized_pnl = terminal_pnl - path["running_penalty"] - terminal_penalty

    path.update(
        {
            "terminal_pnl": terminal_pnl,
            "penalized_pnl": penalized_pnl,
            "terminal_penalty": terminal_penalty,
            "terminal_inventory": q,
            "avg_abs_inventory": path["abs_q_integral"] / config.horizon,
            "avg_q2": path["q2_integral"] / config.horizon,
            "record_time": np.asarray(record_times),
            "record_mean_pnl": np.asarray(record_mean_pnl).T,
            "record_mean_penalized": np.asarray(record_mean_penalized).T,
            "record_mean_absq": np.asarray(record_mean_absq).T,
            "alpha_bin_centers": alpha_centers,
            "alpha_bin_counts": conditional_count,
            "conditional_ask_counts": conditional_ask_count,
            "conditional_bid_counts": conditional_bid_count,
            "conditional_ask_by_alpha": np.divide(
                conditional_ask_sum,
                conditional_ask_count,
                out=np.full_like(conditional_ask_sum, np.nan),
                where=conditional_ask_count > 0,
            ),
            "conditional_bid_by_alpha": np.divide(
                conditional_bid_sum,
                conditional_bid_count,
                out=np.full_like(conditional_bid_sum, np.nan),
                where=conditional_bid_count > 0,
            ),
        }
    )
    coupling_counts = asdict(diag)
    quoted = max(1, diag.quoted)
    metadata = {
        "elapsed_seconds": time.monotonic() - start,
        "off_policy_grid_evaluations": off_grid,
        "policy_evaluations": int(2 * n * steps),
        "boundary_visits": boundary_visits.tolist(),
        "conditional_alpha_underflow": conditional_alpha_under,
        "conditional_alpha_overflow": conditional_alpha_over,
        "coupling": coupling_counts,
        "coupling_shares": {
            "negative_depth": diag.negative_depth / quoted,
            "raw_multiple_fills": diag.raw_multiple_fills / quoted,
            "capped_fills": diag.capped_fills / quoted,
            "saturated_shortcut": diag.saturated_shortcut / quoted,
        },
    }
    return path, metadata


def summarize(
    config: RunConfig,
    path: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    gamma = toy_legacy.P.gamma if config.model == "toy" else fast_legacy.P.gamma
    policy_modes = ("srn", config.right_mode)
    individual: dict[str, dict[str, float]] = {}
    for p_idx, name in enumerate(policy_modes):
        pnl = path["terminal_pnl"][p_idx]
        pen = path["penalized_pnl"][p_idx]
        ce, ess, max_share = stable_ce(pnl, gamma)
        individual[f"{p_idx}_{name}"] = {
            "terminal_pnl_mean": float(np.mean(pnl)),
            "terminal_pnl_std": float(np.std(pnl, ddof=1)),
            "terminal_pnl_se": standard_error(pnl),
            "terminal_pnl_q05": float(np.quantile(pnl, 0.05)),
            "terminal_pnl_q50": float(np.quantile(pnl, 0.50)),
            "terminal_pnl_q95": float(np.quantile(pnl, 0.95)),
            "penalized_pnl_mean": float(np.mean(pen)),
            "penalized_pnl_se": standard_error(pen),
            "certainty_equivalent": ce,
            "ce_weight_ess": ess,
            "ce_weight_max_share": max_share,
            "fills_mean": float(np.mean(path["fills"][p_idx])),
            "avg_abs_inventory_mean": float(np.mean(path["avg_abs_inventory"][p_idx])),
            "avg_q2_mean": float(np.mean(path["avg_q2"][p_idx])),
            "terminal_abs_inventory_mean": float(
                np.mean(np.abs(path["terminal_inventory"][p_idx]))
            ),
            "negative_min_ask_share": float(np.mean(path["min_ask"][p_idx] < 0.0)),
            "negative_min_bid_share": float(np.mean(path["min_bid"][p_idx] < 0.0)),
        }

    metric_names = (
        "terminal_pnl",
        "penalized_pnl",
        "spread_capture",
        "alpha_carry",
        "price_martingale",
        "running_penalty",
        "terminal_penalty",
        "fills",
        "avg_abs_inventory",
        "avg_q2",
        "terminal_inventory",
    )
    paired = {name: paired_summary(path[name][1], path[name][0]) for name in metric_names}

    rng = make_rngs(config.seed)["bootstrap"]
    left = path["terminal_pnl"][0]
    right = path["terminal_pnl"][1]
    ce_differences = np.empty(config.bootstrap)
    for b in range(config.bootstrap):
        idx = rng.integers(0, config.paths, config.paths)
        ce_left, _, _ = stable_ce(left[idx], gamma)
        ce_right, _, _ = stable_ce(right[idx], gamma)
        ce_differences[b] = ce_right - ce_left
    ce_left, _, _ = stable_ce(left, gamma)
    ce_right, _, _ = stable_ce(right, gamma)
    paired_ce = {
        "left_ce": ce_left,
        "right_ce": ce_right,
        "paired_difference": ce_right - ce_left,
        "bootstrap_se": float(np.std(ce_differences, ddof=1))
        if config.bootstrap > 1
        else 0.0,
        "bootstrap_ci95_low": float(np.quantile(ce_differences, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(ce_differences, 0.975)),
        "bootstrap_repetitions": config.bootstrap,
    }
    return {
        "config": asdict(config),
        "policy_modes": list(policy_modes),
        "individual": individual,
        "paired": paired,
        "paired_certainty_equivalent": paired_ce,
        "diagnostics": metadata,
    }


def write_outputs(
    output_dir: Path,
    config: RunConfig,
    path: dict[str, np.ndarray],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = config.model.replace("-", "_")
    np.savez_compressed(output_dir / f"{stem}_paired_paths.npz", **path)
    (output_dir / f"{stem}_paired_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    rows = []
    for metric, values in summary["paired"].items():
        rows.append({"metric": metric, **values})
    with (output_dir / f"{stem}_paired_differences.csv").open("w", newline="") as handle:
        fields = [
            "metric",
            "left_mean",
            "right_mean",
            "paired_difference_mean",
            "paired_difference_se",
            "paired_ci95_low",
            "paired_ci95_high",
            "paired_correlation",
            "unpaired_difference_se",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    if config.model == "fast-alpha":
        with (output_dir / "fast_alpha_conditional_depths_by_alpha.csv").open(
            "w", newline=""
        ) as handle:
            fields = [
                "alpha_center",
                "count",
                "ask_current_count",
                "ask_leading_cara_count",
                "bid_current_count",
                "bid_leading_cara_count",
                "ask_current",
                "ask_leading_cara",
                "bid_current",
                "bid_leading_cara",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for j, center in enumerate(path["alpha_bin_centers"]):
                writer.writerow(
                    {
                        "alpha_center": center,
                        "count": int(path["alpha_bin_counts"][j]),
                        "ask_current_count": int(path["conditional_ask_counts"][0, j]),
                        "ask_leading_cara_count": int(path["conditional_ask_counts"][1, j]),
                        "bid_current_count": int(path["conditional_bid_counts"][0, j]),
                        "bid_leading_cara_count": int(path["conditional_bid_counts"][1, j]),
                        "ask_current": path["conditional_ask_by_alpha"][0, j],
                        "ask_leading_cara": path["conditional_ask_by_alpha"][1, j],
                        "bid_current": path["conditional_bid_by_alpha"][0, j],
                        "bid_leading_cara": path["conditional_bid_by_alpha"][1, j],
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("toy", "fast-alpha"), required=True)
    parser.add_argument("--paths", type=int, default=2000)
    parser.add_argument("--horizon", type=float, default=10000.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--record-dt", type=float, default=10.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--right-mode", choices=("srn", "leading_cara"), default="leading_cara"
    )
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--policy-i-max", type=float, default=9.0)
    parser.add_argument("--policy-i-step", type=float, default=0.01)
    parser.add_argument("--alpha-min", type=float, default=-0.12)
    parser.add_argument("--alpha-max", type=float, default=0.12)
    parser.add_argument("--alpha-bins", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "imagens_tex" / "cara_aware_closed_form_comparison" / "paired_mc",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_seed = toy_legacy.P.seed if args.model == "toy" else fast_legacy.P.seed
    config = RunConfig(
        model=args.model,
        paths=args.paths,
        horizon=args.horizon,
        dt=args.dt,
        record_dt=args.record_dt,
        seed=default_seed if args.seed is None else args.seed,
        right_mode=args.right_mode,
        bootstrap=args.bootstrap,
        policy_i_max=args.policy_i_max,
        policy_i_step=args.policy_i_step,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_bins=args.alpha_bins,
    )
    if config.paths <= 0 or config.horizon <= 0.0 or config.dt <= 0.0:
        raise ValueError("paths, horizon, and dt must be positive")
    steps = int(round(config.horizon / config.dt))
    if not np.isclose(steps * config.dt, config.horizon, rtol=0.0, atol=1e-12):
        raise ValueError("horizon/dt must be an integer for exact normalization")
    if config.right_mode not in ("srn", "leading_cara"):
        raise ValueError(config.right_mode)
    path, metadata = simulate(config)
    summary = summarize(config, path, metadata)
    write_outputs(args.output_dir.resolve(), config, path, summary)
    print(json.dumps(summary["paired"], indent=2), flush=True)
    print(json.dumps(summary["paired_certainty_equivalent"], indent=2), flush=True)
    print(f"Wrote paired MC outputs to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
