"""Regression tests for the paired closed-form Monte Carlo harness."""

from __future__ import annotations

import unittest

import numpy as np

from closed_form_tests.cara_aware_closed_forms import (
    fast_alpha_depths,
    toy_depths,
)
from closed_form_tests.paired_cara_closed_form_mc import (
    DepthLookup,
    RunConfig,
    coupled_capped_poisson,
    simulate,
)


def small_config(model: str) -> RunConfig:
    return RunConfig(
        model=model,
        paths=48,
        horizon=1.0,
        dt=0.05,
        record_dt=0.25,
        seed=12345,
        right_mode="srn",
        bootstrap=10,
        policy_i_max=9.0,
        policy_i_step=0.01,
        alpha_min=-0.12,
        alpha_max=0.12,
        alpha_bins=16,
    )


class PairedHarnessTests(unittest.TestCase):
    def test_lookup_matches_direct_formula(self) -> None:
        rng = np.random.default_rng(7)
        q = rng.integers(-59, 60, 2000)
        i = rng.uniform(-6.0, 6.0, 2000)
        for depth_fn in (toy_depths, fast_alpha_depths):
            lookup = DepthLookup(depth_fn, 60, 9.0, 0.01, "leading_cara")
            for p_idx, mode in enumerate(("srn", "leading_cara")):
                ask, bid, clipped = lookup.depths(p_idx, q, i)
                exact_ask, exact_bid = depth_fn(q, i, risk_mode=mode)
                self.assertEqual(clipped, 0)
                self.assertLess(float(np.max(np.abs(ask - exact_ask))), 3e-5)
                self.assertLess(float(np.max(np.abs(bid - exact_bid))), 3e-5)

    def test_identity_pair_is_bitwise_equal(self) -> None:
        for model in ("toy", "fast-alpha"):
            paths, _ = simulate(small_config(model))
            for key in (
                "terminal_pnl",
                "penalized_pnl",
                "spread_capture",
                "alpha_carry",
                "price_martingale",
                "running_penalty",
                "fills",
                "terminal_inventory",
                "avg_abs_inventory",
            ):
                np.testing.assert_array_equal(paths[key][0], paths[key][1])

    def test_pathwise_decomposition(self) -> None:
        for model in ("toy", "fast-alpha"):
            config = small_config(model)
            config = RunConfig(**{**config.__dict__, "right_mode": "leading_cara"})
            paths, _ = simulate(config)
            reconstructed = (
                paths["spread_capture"]
                + paths["alpha_carry"]
                + paths["price_martingale"]
            )
            np.testing.assert_allclose(paths["terminal_pnl"], reconstructed, atol=1e-11)

    def test_common_component_preserves_poisson_marginals(self) -> None:
        n = 200_000
        mean_left = np.full(n, 0.25)
        mean_right = np.full(n, 0.90)
        cap = np.full(n, 100)
        left, right, _ = coupled_capped_poisson(
            mean_left,
            mean_right,
            cap,
            cap,
            np.random.default_rng(1),
            np.random.default_rng(2),
            np.random.default_rng(3),
        )
        self.assertAlmostEqual(float(np.mean(left)), 0.25, delta=0.004)
        self.assertAlmostEqual(float(np.mean(right)), 0.90, delta=0.006)
        self.assertGreater(float(np.corrcoef(left, right)[0, 1]), 0.45)


if __name__ == "__main__":
    unittest.main()
