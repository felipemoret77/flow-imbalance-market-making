"""Regression tests for the production slow-regime sc-cubic formula.

The historical file with this name was an exploratory plotting campaign.  It
solved the regime RHJB at import time and carried a stale, gated copy of the
closed form.  These tests call the production formula directly, do not solve an
RHJB, and do not write figures.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/asou5_regime_tests_mpl")
os.environ.setdefault("ASOU5_REGIME_OUTPUT_DIR", "/tmp/asou5_regime_tests_outputs")

from regime_studies import regime_fast_alpha_ergodic_cole_hopf as regime


def _reference_depths(
    q: np.ndarray,
    imbalance: np.ndarray,
    regime_idx: np.ndarray,
    *,
    gate_regime: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent transcription of the published baseline formula."""
    p = regime.P
    q = np.asarray(q, dtype=float)
    imbalance = np.asarray(imbalance, dtype=float)
    z = regime.Z_LEVELS[np.asarray(regime_idx, dtype=int)]

    inventory_rate = 2.0 * p.p_glft
    flow_scale = np.sqrt(p.beta) / p.eta
    flow_amplitude = ((p.rho / p.k) / (p.rho + 2.0 * p.beta)) / flow_scale
    h_flow = flow_amplitude * np.tanh(flow_scale * imbalance)
    theta_flow = imbalance - p.k * h_flow
    curvature = inventory_rate * np.sqrt(
        np.cosh(imbalance) / np.cosh(theta_flow)
    )
    curvature_multiplier = curvature / inventory_rate

    h_alpha = regime.LOCAL_ANSATZ["alpha_loading"] * np.tanh(imbalance)
    h_regime = regime.LOCAL_ANSATZ["regime_loading"] * z
    gate = 1.0 + np.tanh(flow_scale * imbalance) ** 2 * (
        curvature_multiplier - 1.0
    )
    shift = curvature_multiplier * h_flow + gate * h_alpha
    shift += gate * h_regime if gate_regime else h_regime

    theta_full = imbalance - p.k * shift
    cubic = curvature**2 * np.tanh(theta_full) / 6.0
    ask = (
        p.delta0
        + shift
        - curvature * (2.0 * q - 1.0) / (2.0 * p.k)
        + cubic * (q * q - q + 1.0 / 3.0) / p.k
    )
    bid = (
        p.delta0
        - shift
        + curvature * (2.0 * q + 1.0) / (2.0 * p.k)
        - cubic * (q * q + q + 1.0 / 3.0) / p.k
    )
    ask = np.where(q > -p.qmax, ask, np.inf)
    bid = np.where(q < p.qmax, bid, np.inf)
    return ask, bid


class RegimeSelfConsistentCubicTests(unittest.TestCase):
    def test_default_matches_ungated_reference(self) -> None:
        q = np.array([-59, -13, 0, 8, 27, 59], dtype=int)
        imbalance = np.array([-5.0, -2.25, -0.4, 0.8, 2.75, 5.5])
        regime_idx = np.array([0, 2, 1, 0, 2, 1], dtype=int)

        actual = regime.regime_sc_cubic_depths(q, imbalance, regime_idx)
        expected = _reference_depths(
            q, imbalance, regime_idx, gate_regime=False
        )
        np.testing.assert_allclose(actual[0], expected[0], rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(actual[1], expected[1], rtol=2e-14, atol=2e-14)

    def test_gated_variant_is_explicit_and_not_the_default(self) -> None:
        q = np.array([-10, 0, 10, -10, 0, 10], dtype=int)
        imbalance = np.array([-4.0, -2.5, -1.5, 1.5, 2.5, 4.0])
        regime_idx = np.array([0, 2, 0, 2, 0, 2], dtype=int)

        ungated = regime.regime_sc_cubic_depths(q, imbalance, regime_idx)
        gated = regime.regime_sc_cubic_depths(
            q, imbalance, regime_idx, gate_regime=True
        )
        gated_reference = _reference_depths(
            q, imbalance, regime_idx, gate_regime=True
        )

        np.testing.assert_allclose(
            gated[0], gated_reference[0], rtol=2e-14, atol=2e-14
        )
        np.testing.assert_allclose(
            gated[1], gated_reference[1], rtol=2e-14, atol=2e-14
        )
        self.assertGreater(float(np.max(np.abs(ungated[0] - gated[0]))), 1e-5)
        self.assertGreater(float(np.max(np.abs(ungated[1] - gated[1]))), 1e-5)

        neutral = np.full_like(regime_idx, regime.NEUTRAL_IDX)
        neutral_ungated = regime.regime_sc_cubic_depths(q, imbalance, neutral)
        neutral_gated = regime.regime_sc_cubic_depths(
            q, imbalance, neutral, gate_regime=True
        )
        np.testing.assert_array_equal(neutral_ungated[0], neutral_gated[0])
        np.testing.assert_array_equal(neutral_ungated[1], neutral_gated[1])

    def test_quote_reflection_symmetry(self) -> None:
        q = np.array([-35, -9, -1, 0, 7, 18, 42], dtype=int)
        imbalance = np.array([-5.2, -2.7, -0.3, 0.0, 1.1, 3.4, 5.8])
        regime_idx = np.array([0, 2, 0, 1, 2, 0, 2], dtype=int)

        for gate_regime in (False, True):
            with self.subTest(gate_regime=gate_regime):
                ask, bid = regime.regime_sc_cubic_depths(
                    q, imbalance, regime_idx, gate_regime=gate_regime
                )
                reflected_ask, reflected_bid = regime.regime_sc_cubic_depths(
                    -q,
                    -imbalance,
                    2 - regime_idx,
                    gate_regime=gate_regime,
                )
                np.testing.assert_allclose(
                    ask, reflected_bid, rtol=2e-13, atol=2e-13
                )
                np.testing.assert_allclose(
                    bid, reflected_ask, rtol=2e-13, atol=2e-13
                )

    def test_inventory_boundary_masks(self) -> None:
        q = np.array([-regime.P.qmax, regime.P.qmax], dtype=int)
        imbalance = np.zeros(2)
        regime_idx = np.full(2, regime.NEUTRAL_IDX, dtype=int)
        ask, bid = regime.regime_sc_cubic_depths(q, imbalance, regime_idx)

        self.assertTrue(np.isinf(ask[0]) and np.isfinite(bid[0]))
        self.assertTrue(np.isfinite(ask[1]) and np.isinf(bid[1]))


if __name__ == "__main__":
    unittest.main()
