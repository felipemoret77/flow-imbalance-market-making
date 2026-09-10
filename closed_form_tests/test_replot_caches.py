"""Lightweight round-trip tests for the post-Monte-Carlo replot caches."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "alpha_studies"))
sys.path.insert(0, str(ROOT / "regime_studies"))

import alpha_reduced_cole_hopf as A  # noqa: E402
import regime_fast_alpha_ergodic_cole_hopf as R  # noqa: E402


ALPHA_CACHE_KEYS = {
    "schema_version",
    "policy_names",
    "conditional_y_centers",
    "conditional_y_mean",
    "conditional_y_counts",
    "conditional_alpha_centers",
    "conditional_alpha_mean",
    "conditional_alpha_counts",
    "lifetime_q_centers",
    "lifetime_q_counts",
    "lifetime_abs_centers",
    "lifetime_abs_counts",
    "lifetime_n_observations",
    "terminal_total",
    "terminal_decomposed",
    "sample_times",
    "sample_price",
    "sample_i",
    "sample_alpha",
    "sample_inventory",
}


class ReplotCacheTests(unittest.TestCase):
    def test_alpha_lifetime_histogram_reduction_is_lossless(self) -> None:
        inventory = np.array([-2, -1, -1, 0, 1, 2, 2, 2], dtype=int)
        reduced = A.make_lifetime_inventory_histograms({"GLFT": inventory})
        q_counts = reduced["q_counts"]["GLFT"]
        abs_counts = reduced["abs_counts"]["GLFT"]
        self.assertEqual(int(np.sum(q_counts)), inventory.size)
        self.assertEqual(int(np.sum(abs_counts)), inventory.size)
        centers = np.asarray(reduced["q_centers"])
        self.assertEqual(int(q_counts[centers == 2][0]), 3)

    def test_alpha_replot_cache_round_trip(self) -> None:
        policies = ["GLFT", "Alpha/Y-aware"]
        lifetime_inventory = {
            "GLFT": np.array([-3, -1, -1, 0, 2], dtype=int),
            "Alpha/Y-aware": np.array([-2, 0, 1, 1, 3, 3], dtype=int),
        }
        histograms = A.make_lifetime_inventory_histograms(lifetime_inventory)
        paths = {
            "terminal_total": {
                name: np.arange(6, dtype=float) + 10.0 * j
                for j, name in enumerate(policies)
            },
            "terminal_decomposed": {
                name: -np.arange(6, dtype=float) - 20.0 * j
                for j, name in enumerate(policies)
            },
            "y_centers": np.array([-1.5, -0.25, 0.75]),
            "inventory_by_y": {
                name: np.array([0.25, -1.5, 2.75]) + 7.0 * j
                for j, name in enumerate(policies)
            },
            "inventory_by_y_counts": {
                name: np.array([4, 7, 11]) + 13 * j
                for j, name in enumerate(policies)
            },
            "alpha_centers": np.array([-0.3, -0.05, 0.1, 0.4]),
            "inventory_by_alpha": {
                name: np.array([-2.5, 0.5, 1.25, 4.0]) - 9.0 * j
                for j, name in enumerate(policies)
            },
            "inventory_by_alpha_counts": {
                name: np.array([3, 5, 8, 12]) + 17 * j
                for j, name in enumerate(policies)
            },
            "lifetime_inventory_histograms": histograms,
            "sample": {
                "times": np.array([0.0, 0.1, 0.4, 0.9, 1.5]),
                "price": np.array([100.0, 100.2, 99.8, 100.7, 101.1]),
                "y": np.array([-1.2, -0.4, 0.3, 1.1, 0.6]),
                "alpha": np.array([0.03, -0.02, 0.08, -0.11, 0.04]),
                "inventory": {
                    name: np.array([-2, 0, 1, -1, 3]) + 6 * j
                    for j, name in enumerate(policies)
                },
            },
        }
        expected_raw = {
            "schema_version": np.asarray(A.REPLOT_CACHE_SCHEMA_VERSION, dtype=np.int64),
            "policy_names": np.asarray(policies),
            "conditional_y_centers": paths["y_centers"],
            "conditional_y_mean": np.stack(
                [paths["inventory_by_y"][name] for name in policies]
            ),
            "conditional_y_counts": np.stack(
                [paths["inventory_by_y_counts"][name] for name in policies]
            ),
            "conditional_alpha_centers": paths["alpha_centers"],
            "conditional_alpha_mean": np.stack(
                [paths["inventory_by_alpha"][name] for name in policies]
            ),
            "conditional_alpha_counts": np.stack(
                [paths["inventory_by_alpha_counts"][name] for name in policies]
            ),
            "lifetime_q_centers": histograms["q_centers"],
            "lifetime_q_counts": np.stack(
                [histograms["q_counts"][name] for name in policies]
            ),
            "lifetime_abs_centers": histograms["abs_centers"],
            "lifetime_abs_counts": np.stack(
                [histograms["abs_counts"][name] for name in policies]
            ),
            "lifetime_n_observations": np.asarray(
                [histograms["n_observations"][name] for name in policies]
            ),
            "terminal_total": np.stack(
                [paths["terminal_total"][name] for name in policies]
            ),
            "terminal_decomposed": np.stack(
                [paths["terminal_decomposed"][name] for name in policies]
            ),
            "sample_times": paths["sample"]["times"],
            "sample_price": paths["sample"]["price"],
            "sample_i": paths["sample"]["y"],
            "sample_alpha": paths["sample"]["alpha"],
            "sample_inventory": np.stack(
                [paths["sample"]["inventory"][name] for name in policies]
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            cache = A.save_replot_data(paths, Path(tmp) / "alpha.npz")
            with np.load(cache, allow_pickle=False) as raw:
                self.assertEqual(set(raw.files), ALPHA_CACHE_KEYS)
                self.assertEqual(raw["policy_names"].dtype.kind, "U")
                for key, expected in expected_raw.items():
                    self.assertFalse(raw[key].dtype.hasobject, key)
                    np.testing.assert_array_equal(raw[key], expected, err_msg=key)
            restored = A.load_replot_data(cache)
        np.testing.assert_array_equal(restored["y_centers"], paths["y_centers"])
        np.testing.assert_array_equal(restored["alpha_centers"], paths["alpha_centers"])
        for restored_key, source_key in (
            ("inventory_by_y", "inventory_by_y"),
            ("inventory_by_y_counts", "inventory_by_y_counts"),
            ("inventory_by_alpha", "inventory_by_alpha"),
            ("inventory_by_alpha_counts", "inventory_by_alpha_counts"),
            ("terminal_total", "terminal_total"),
            ("terminal_decomposed", "terminal_decomposed"),
        ):
            self.assertEqual(list(restored[restored_key]), policies)
            for name in policies:
                np.testing.assert_array_equal(
                    restored[restored_key][name], paths[source_key][name]
                )

        restored_histograms = restored["lifetime_inventory_histograms"]
        for center_key in ("q_centers", "abs_centers"):
            np.testing.assert_array_equal(
                restored_histograms[center_key], histograms[center_key]
            )
        for counts_key in ("q_counts", "abs_counts"):
            self.assertEqual(list(restored_histograms[counts_key]), policies)
            for name in policies:
                np.testing.assert_array_equal(
                    restored_histograms[counts_key][name], histograms[counts_key][name]
                )
        self.assertEqual(restored_histograms["n_observations"], histograms["n_observations"])

        for key, source_key in (
            ("times", "times"),
            ("price", "price"),
            ("y", "y"),
            ("alpha", "alpha"),
        ):
            np.testing.assert_array_equal(
                restored["sample"][key], paths["sample"][source_key]
            )
        self.assertEqual(list(restored["sample"]["inventory"]), policies)
        for name in policies:
            np.testing.assert_array_equal(
                restored["sample"]["inventory"][name],
                paths["sample"]["inventory"][name],
            )

    def test_alpha_legacy_lifetime_inventory_fallback(self) -> None:
        legacy = {
            "GLFT": np.array([-2, -1, 0, 0, 3], dtype=int),
            "Alpha/Y-aware": np.array([-1, 1, 1, 2], dtype=int),
        }
        expected = A.make_lifetime_inventory_histograms(legacy)
        resolved = A._resolve_lifetime_inventory_histograms(
            {"lifetime_inventory": legacy}
        )
        for key in ("q_centers", "abs_centers"):
            np.testing.assert_array_equal(resolved[key], expected[key])
        for key in ("q_counts", "abs_counts"):
            for name in legacy:
                np.testing.assert_array_equal(resolved[key][name], expected[key][name])
        self.assertEqual(resolved["n_observations"], expected["n_observations"])

    def test_regime_terminal_pnl_cache_round_trip(self) -> None:
        policies = ["GLFT", "Regime-aware"]
        paths = {
            "terminal_total": {
                name: np.arange(7, dtype=float) + j for j, name in enumerate(policies)
            },
            "terminal_decomposed": {
                name: np.arange(7, dtype=float) - j for j, name in enumerate(policies)
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            old_out = R.OUT
            R.OUT = Path(tmp)
            try:
                cache = R.save_terminal_pnl_replot_data(paths)
                restored = R.load_terminal_pnl_replot_data(cache)
            finally:
                R.OUT = old_out
        self.assertEqual(list(restored["terminal_total"]), policies)
        self.assertEqual(list(restored["terminal_decomposed"]), policies)
        for name in policies:
            np.testing.assert_array_equal(
                restored["terminal_total"][name],
                paths["terminal_total"][name],
            )
            np.testing.assert_array_equal(
                restored["terminal_decomposed"][name],
                paths["terminal_decomposed"][name],
            )


if __name__ == "__main__":
    unittest.main()
