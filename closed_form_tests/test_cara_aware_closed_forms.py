"""Regression tests for the isolated CARA-aware closed-form API."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/asou5_cara_aware_tests_mpl")

import cara_glft_yaware_study as toy_legacy
from alpha_studies import alpha_reduced_cole_hopf as fast_legacy
from closed_form_tests.cara_aware_closed_forms import (
    fast_alpha_depths,
    fast_alpha_profiles,
    toy_depths,
    toy_profiles,
)


def _comparison_grid() -> tuple[np.ndarray, np.ndarray]:
    q = np.array([-59, -17, -5, 0, 10, 23, 59], dtype=int)[:, None]
    i = np.linspace(-6.0, 6.0, 31)[None, :]
    return np.broadcast_arrays(q, i)


class CaraAwareClosedFormTests(unittest.TestCase):
    def test_toy_srn_regression_against_legacy(self) -> None:
        q, i = _comparison_grid()
        ask, bid = toy_depths(q, i, risk_mode="srn")
        ask_legacy, bid_legacy = toy_legacy.y_curvature_refined_depths(q, i)
        np.testing.assert_allclose(ask, ask_legacy, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(bid, bid_legacy, rtol=2e-14, atol=2e-14)

    def test_fast_alpha_srn_regression_against_legacy(self) -> None:
        q, i = _comparison_grid()
        ask, bid = fast_alpha_depths(q, i, risk_mode="srn")
        ask_legacy, bid_legacy = fast_legacy.toy_h_sc_cubic_depths(q, i)
        np.testing.assert_allclose(ask, ask_legacy, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(bid, bid_legacy, rtol=2e-14, atol=2e-14)

    def test_quote_reflection_symmetry(self) -> None:
        q = np.array([-27, -9, -1, 0, 4, 13, 29], dtype=int)[:, None]
        i = np.linspace(-5.75, 5.75, 25)[None, :]
        q, i = np.broadcast_arrays(q, i)
        for depth_fn in (toy_depths, fast_alpha_depths):
            for risk_mode in ("srn", "leading_cara"):
                with self.subTest(depth_fn=depth_fn.__name__, risk_mode=risk_mode):
                    ask, bid = depth_fn(q, i, risk_mode=risk_mode)
                    reflected_ask, reflected_bid = depth_fn(
                        -q, -i, risk_mode=risk_mode
                    )
                    np.testing.assert_allclose(
                        ask, reflected_bid, rtol=2e-13, atol=2e-13
                    )
                    np.testing.assert_allclose(
                        bid, reflected_ask, rtol=2e-13, atol=2e-13
                    )

    def test_analytic_shift_derivative(self) -> None:
        i = np.array(
            [-5.5, -3.0, -1.25, -0.2, 0.0, 0.2, 1.25, 3.0, 5.5]
        )
        step = 1e-6
        for profile_fn in (toy_profiles, fast_alpha_profiles):
            with self.subTest(profile_fn=profile_fn.__name__):
                analytic = profile_fn(i, risk_mode="srn").dshift_di
                finite_difference = (
                    profile_fn(i + step, risk_mode="srn").shift
                    - profile_fn(i - step, risk_mode="srn").shift
                ) / (2.0 * step)
                np.testing.assert_allclose(
                    analytic, finite_difference, rtol=2e-8, atol=2e-9
                )

    def test_baseline_origin_risk_factors(self) -> None:
        toy = toy_profiles(0.0, risk_mode="leading_cara")
        fast = fast_alpha_profiles(0.0, risk_mode="leading_cara")

        np.testing.assert_allclose(
            toy.risk_factor, 1.0467034132211996, rtol=2e-14, atol=2e-14
        )
        np.testing.assert_allclose(
            fast.risk_factor, 1.5063357851396093, rtol=2e-14, atol=2e-14
        )
        np.testing.assert_allclose(
            toy.dshift_di, 0.28984982595386666, rtol=2e-14, atol=2e-14
        )
        np.testing.assert_allclose(
            fast.dshift_di, 1.0561125778914918, rtol=2e-14, atol=2e-14
        )

    def test_leading_cara_changes_only_curvature_and_cubic(self) -> None:
        i = np.linspace(-5.0, 5.0, 101)
        for profile_fn in (toy_profiles, fast_alpha_profiles):
            with self.subTest(profile_fn=profile_fn.__name__):
                srn = profile_fn(i, risk_mode="srn")
                leading = profile_fn(i, risk_mode="leading_cara")

                np.testing.assert_array_equal(leading.shift, srn.shift)
                np.testing.assert_array_equal(leading.theta_flow, srn.theta_flow)
                np.testing.assert_array_equal(leading.theta_full, srn.theta_full)
                np.testing.assert_array_equal(leading.m_flow, srn.m_flow)
                np.testing.assert_allclose(
                    leading.r,
                    srn.r * leading.risk_factor,
                    rtol=2e-14,
                    atol=2e-14,
                )
                np.testing.assert_allclose(
                    leading.s,
                    (leading.r**2 / 6.0) * np.tanh(leading.theta_full),
                    rtol=2e-14,
                    atol=2e-14,
                )
                self.assertTrue(np.all(leading.risk_factor >= 1.0))
                np.testing.assert_array_equal(
                    srn.risk_factor, np.ones_like(i)
                )

    def test_inventory_boundary_masks(self) -> None:
        qmax = 60
        q = np.array([-qmax, qmax], dtype=int)
        i = np.zeros_like(q, dtype=float)
        for depth_fn in (toy_depths, fast_alpha_depths):
            for risk_mode in ("srn", "leading_cara"):
                with self.subTest(depth_fn=depth_fn.__name__, risk_mode=risk_mode):
                    ask, bid = depth_fn(q, i, risk_mode=risk_mode)
                    self.assertTrue(np.isinf(ask[0]) and np.isfinite(bid[0]))
                    self.assertTrue(np.isfinite(ask[1]) and np.isinf(bid[1]))

    def test_invalid_risk_mode(self) -> None:
        for profile_fn in (toy_profiles, fast_alpha_profiles):
            with self.subTest(profile_fn=profile_fn.__name__):
                with self.assertRaisesRegex(ValueError, "unknown risk_mode"):
                    profile_fn(0.0, risk_mode="exact_cara")


if __name__ == "__main__":
    unittest.main()
