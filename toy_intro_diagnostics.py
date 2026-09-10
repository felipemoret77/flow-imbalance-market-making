#!/usr/bin/env python3
"""Introductory diagnostics for the AS_OU_5 flow-imbalance toy model.

These figures are pedagogical: they show how the OU imbalance factor I_t
changes the side composition of order flow, why the total zero-depth activity
is constant across imbalance levels, and how the exponential depth sensitivity
behaves.  Notation matches the paper: imbalance state I, side index a (buy MOs
hitting the ask) / b (sell MOs hitting the bid), and zero-depth intensities
Lambda^{a,0}(i)=bar_lambda(1+tanh i), Lambda^{b,0}(i)=bar_lambda(1-tanh i).

All legends are placed in the upper-right corner for a uniform layout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
# Use the same output override and relative-path convention as the main toy
# study.  Keep ASOU5_OUTPUT_DIR as a compatibility fallback for older runs.
out_env = os.environ.get(
    "ASOU5_CARA_OUTPUT_DIR",
    os.environ.get("ASOU5_OUTPUT_DIR", "imagens_tex/toy_model_T10000_qmax60"),
)
OUT = Path(out_env)
if not OUT.is_absolute():
    OUT = ROOT / OUT
OUT.mkdir(parents=True, exist_ok=True)

# Uniform legend style: upper-right, opaque box so it reads over the curves.
LEGEND_KW = dict(loc="upper right", framealpha=0.95)


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Params:
    # Paper toy-model parameters.
    bar_lambda: float = env_float("ASOU5_BAR_LAMBDA", 0.9)
    k: float = env_float("ASOU5_K", 2.0)
    beta: float = env_float("ASOU5_BETA", 0.0125)
    eta: float = env_float("ASOU5_ETA", 0.32)
    seed: int = env_int("ASOU5_SEED", 54321)

    @property
    def delta0(self) -> float:
        return 1.0 / self.k


P = Params()


def lambda_a_zero(i: np.ndarray) -> np.ndarray:
    return P.bar_lambda * (1.0 + np.tanh(i))


def lambda_b_zero(i: np.ndarray) -> np.ndarray:
    return P.bar_lambda * (1.0 - np.tanh(i))


def simulate_ou_path(T: float = 2000.0, dt: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(P.seed)
    steps = int(round(T / dt))
    t = np.linspace(0.0, T, steps + 1)
    i = np.zeros(steps + 1)
    sqrt_dt = np.sqrt(dt)
    for n in range(steps):
        i[n + 1] = i[n] - P.beta * i[n] * dt + P.eta * sqrt_dt * rng.standard_normal()
    return t, i


def plot_sample_path() -> None:
    t, i = simulate_ou_path()
    la = lambda_a_zero(i)
    lb = lambda_b_zero(i)
    total = la + lb

    fig, ax = plt.subplots(3, 1, figsize=(10.5, 7.2), sharex=True)
    ax[0].plot(t, i, color="#4c72b0", lw=1.5)
    ax[0].axhline(0.0, color="black", lw=0.9, alpha=0.45)
    ax[0].set_ylabel(r"imbalance $I_t$")
    ax[0].set_title("OU imbalance sample path and induced side intensities")

    ax[1].plot(t, la, color="#55a868", lw=1.45, label=r"$\Lambda^{a,0}(I_t)$")
    ax[1].plot(t, lb, color="#c44e52", lw=1.45, label=r"$\Lambda^{b,0}(I_t)$")
    ax[1].set_ylabel("zero-depth intensity")
    ax[1].legend(fontsize=9, ncol=2, **LEGEND_KW)

    ax[2].plot(t, total, color="#222222", lw=1.7, label=r"$\Lambda^{a,0}+\Lambda^{b,0}$")
    ax[2].axhline(2.0 * P.bar_lambda, color="#8172b2", lw=1.2, ls="--", label=r"$2\bar\Lambda$")
    ax[2].plot(t, np.tanh(i), color="#dd8452", lw=1.2, alpha=0.85, label=r"$\tanh(I_t)$")
    ax[2].set_ylabel("total / imbalance")
    ax[2].set_xlabel("time")
    ax[2].legend(fontsize=9, ncol=3, **LEGEND_KW)

    for a in ax:
        a.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "toy_intro_ou_lambda_sample_path.png", dpi=180)
    plt.close(fig)


def plot_constant_activity() -> None:
    i = np.linspace(-4.0, 4.0, 501)
    la0 = lambda_a_zero(i)
    lb0 = lambda_b_zero(i)
    depths = [0.0, P.delta0, 2.0 * P.delta0, 5.0 * P.delta0]

    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.5))
    ax[0].plot(i, la0, color="#55a868", lw=2.2, label=r"$\Lambda^{a,0}(i)$")
    ax[0].plot(i, lb0, color="#c44e52", lw=2.2, label=r"$\Lambda^{b,0}(i)$")
    ax[0].plot(i, la0 + lb0, color="#222222", lw=2.1, ls="--", label=r"$2\bar\Lambda$")
    ax[0].axvline(0.0, color="black", lw=0.9, alpha=0.35)
    ax[0].set_title("Side composition changes, total activity does not")
    ax[0].set_xlabel(r"imbalance state $i$")
    ax[0].set_ylabel("zero-depth intensity")
    ax[0].legend(fontsize=9, **LEGEND_KW)

    for depth in depths:
        total = (la0 + lb0) * np.exp(-P.k * depth)
        ax[1].plot(i, total, lw=2.0, label=rf"$\delta={depth:.2f}$")
    ax[1].set_title(r"At common depth, total is flat in $i$, scaled by $e^{-k\delta}$")
    ax[1].set_xlabel(r"imbalance state $i$")
    ax[1].set_ylabel(r"$\Lambda^{a}(i,\delta)+\Lambda^{b}(i,\delta)$")
    ax[1].legend(fontsize=9, **LEGEND_KW)

    for a in ax:
        a.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "toy_intro_constant_activity.png", dpi=180)
    plt.close(fig)


def plot_depth_decay() -> None:
    depth = np.linspace(0.0, 3.0, 401)
    i_values = [-2.0, 0.0, 2.0]

    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.5))
    colors = ["#c44e52", "#4c72b0", "#55a868"]
    for i, color in zip(i_values, colors):
        la = lambda_a_zero(np.array([i]))[0] * np.exp(-P.k * depth)
        lb = lambda_b_zero(np.array([i]))[0] * np.exp(-P.k * depth)
        ax[0].plot(depth, la, color=color, lw=2.0, label=rf"buy flow $a$, $i={i:g}$")
        ax[0].plot(depth, lb, color=color, lw=1.6, ls="--", label=rf"sell flow $b$, $i={i:g}$")

    total = 2.0 * P.bar_lambda * np.exp(-P.k * depth)
    ax[1].plot(depth, total, color="#222222", lw=2.6)
    ax[1].axvline(P.delta0, color="#8172b2", lw=1.4, ls="--", label=r"$\delta_0=1/k$")
    ax[1].set_title("Common-depth total decays exponentially")
    ax[1].set_xlabel(r"common depth $\delta$")
    ax[1].set_ylabel(r"$2\bar\Lambda e^{-k\delta}$")
    ax[1].legend(fontsize=9, **LEGEND_KW)

    ax[0].set_title("Side intensities decay with quoted depth")
    ax[0].set_xlabel(r"depth $\delta$")
    ax[0].set_ylabel(r"$\Lambda^{\ell}(i,\delta)=\Lambda^{\ell,0}(i)\,e^{-k\delta}$")
    ax[0].legend(fontsize=7, ncol=2, **LEGEND_KW)

    for a in ax:
        a.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "toy_intro_depth_decay.png", dpi=180)
    plt.close(fig)


def main() -> None:
    plot_sample_path()
    plot_constant_activity()
    plot_depth_decay()
    print(f"Wrote intro diagnostics to {OUT}")


if __name__ == "__main__":
    main()
