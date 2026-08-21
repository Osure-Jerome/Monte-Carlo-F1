"""Tests for StatisticsEngine — CI, win probability, sensitivity, KDE."""

import numpy as np
import pytest

from f1strategist.statistics.statistics_engine import StatisticsEngine

S = StatisticsEngine()


class TestMeanCI:
    def test_known_values(self):
        times = np.array([100.0, 102.0, 104.0])
        out = S.mean_ci(times)
        assert out["mean_s"] == pytest.approx(102.0)
        assert out["n"] == 3
        assert out["ci_low_s"] < out["mean_s"] < out["ci_high_s"]

    def test_ci_narrows_with_n(self):
        rng = np.random.default_rng(0)
        small = S.mean_ci(rng.normal(100, 5, 100))
        large = S.mean_ci(rng.normal(100, 5, 10_000))
        small_span = small["ci_high_s"] - small["ci_low_s"]
        large_span = large["ci_high_s"] - large["ci_low_s"]
        assert large_span < small_span

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            S.mean_ci(np.array([]))


class TestWinProbability:
    def test_dominated_strategy(self):
        a = np.array([100.0] * 1000)
        b = np.array([200.0] * 1000)
        assert S.win_probability(a, b) == pytest.approx(1.0)

    def test_identical_strategies(self):
        a = np.array([100.0] * 1000)
        assert S.win_probability(a, a.copy()) == pytest.approx(0.0)  # a < b is never true

    def test_mismatched_sizes_rejected(self):
        with pytest.raises(ValueError):
            S.win_probability(np.array([1.0]), np.array([1.0, 2.0]))


class TestSensitivity:
    def test_adjustment_formula(self):
        """adjusted = stored - (old * count) + (new * count)"""
        times = np.array([100.0, 200.0])
        counts = np.array([1, 2])
        adjusted = S.sensitivity(times, counts, old_pit_s=22.0, new_pit_s=30.0)
        expected = np.array([100 - 22 + 30, 200 - 44 + 60])
        np.testing.assert_allclose(adjusted, expected)

    def test_no_op_when_unchanged(self):
        times = np.array([100.0, 200.0])
        counts = np.array([1, 2])
        np.testing.assert_allclose(
            S.sensitivity(times, counts, 22.0, 22.0), times
        )


class TestKDE:
    def test_shapes_and_monotonicity(self):
        sample = np.random.default_rng(0).normal(100, 5, 1000)
        x, density = S.kde(sample)
        assert x.shape == density.shape
        assert density.min() >= 0

    def test_too_few_samples_rejected(self):
        with pytest.raises(ValueError):
            S.kde(np.array([1.0]))
