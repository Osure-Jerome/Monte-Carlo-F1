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


class TestWinRateSweep:
    """Day-4 sensitivity: win probability vs shared pit-stop loss (Sprint 3)."""

    def test_same_strategy_is_flat_at_half(self):
        rng = np.random.default_rng(3)
        times_a = rng.normal(100, 5, 20_000)
        times_b = rng.normal(100, 5, 20_000)  # same distribution ~ 50/50
        counts = np.full_like(times_a, 2, dtype=np.int64)
        losses = np.linspace(20.0, 30.0, 11)
        wins = S.win_rate_vs_pit_loss(times_a, times_b, counts, counts, 22.0, losses)
        assert wins.shape == losses.shape
        # Equal pit counts -> the loss shift cancels -> win rate is flat, ~50/50.
        assert np.allclose(wins, wins[0], atol=0.005)
        assert abs(wins[0] - 0.5) < 0.05

    def test_lower_pit_loss_preferred_when_b_pits_less(self):
        # B does 1 stop, A does 2: cheaper pits -> B benefits, P(A) drops.
        rng = np.random.default_rng(4)
        times_a = rng.normal(6000, 60, 40_000)  # equal base means
        times_b = rng.normal(6000, 60, 40_000)
        counts_a = np.full_like(times_a, 2, dtype=np.int64)
        counts_b = np.full_like(times_b, 1, dtype=np.int64)
        losses = np.linspace(20.0, 30.0, 21)
        wins = S.win_rate_vs_pit_loss(times_a, times_b, counts_a, counts_b, 22.0, losses)
        # P(A) must fall as pit-stop loss rises (A is hit twice as hard).
        assert wins[0] > wins[-1]

    def test_no_op_when_loss_unchanged(self):
        rng = np.random.default_rng(5)
        times = rng.normal(100, 4, 5000)
        counts = np.full_like(times, 2, dtype=np.int64)
        wins = S.win_rate_vs_pit_loss(times, times + 1, counts, counts, 22.0,
                                      np.array([22.0]))
        np.testing.assert_allclose(wins, S.win_probability(times, times + 1))


class TestWinRateGrid:
    """Day-4 sensitivity heatmap over asymmetric (loss_A, loss_B) (Sprint 3)."""

    def test_shape_and_diagonal_symmetry(self):
        rng = np.random.default_rng(6)
        n = 10_000
        times_a = rng.normal(100, 3, n)
        times_b = rng.normal(100, 3, n)
        counts_a = np.full(n, 2, dtype=np.int64)
        counts_b = np.full(n, 1, dtype=np.int64)
        losses = np.linspace(20.0, 30.0, 9)
        grid = S.win_rate_grid(times_a, times_b, counts_a, counts_b,
                               22.0, losses, losses)
        assert grid.shape == (9, 9)
        # Monotone in each axis: P(A) rises when A's pits get cheaper.
        assert np.all(np.diff(grid, axis=1) <= 1e-9)      # along loss_A (cols)
        assert np.all(np.diff(grid, axis=0) >= -1e-9)      # along loss_B (rows)
