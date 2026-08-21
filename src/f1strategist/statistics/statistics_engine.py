"""StatisticsEngine — stateless statistics (all methods @staticmethod).

Provides:
    - ``mean_ci`` — mean, std, 95 % CI of finishing times (FR-9).
    - ``win_probability`` — vectorised head-to-head win probability (FR-10).
    - ``sensitivity`` — pit-stop-loss recomputation without re-simulation (FR-14).
    - ``kde`` — smooth PDF overlay via SciPy ``gaussian_kde`` (FR-13).

Pure NumPy/SciPy operations keep every call comfortably sub-second at 100k rows
(NFR-3), and the engine is never touched during slider interaction.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde  # type: ignore[import-untyped]


class StatisticsEngine:
    """Stateless statistical helpers."""

    # 97.5th percentile of the standard normal -> 95 % two-sided CI.
    _Z = 1.96

    @staticmethod
    def mean_ci(times: np.ndarray) -> dict[str, float]:
        """Mean, std and 95 % confidence interval of a finishing-time array.

        Args:
            times: Array of total finishing times (s).

        Returns:
            ``{"mean_s", "std_s", "ci_low_s", "ci_high_s", "n"}``.
        """
        times = np.asarray(times, dtype=np.float64)
        n = times.size
        if n == 0:
            raise ValueError("Cannot compute stats on an empty array")
        mean = float(times.mean())
        std = float(times.std(ddof=1)) if n > 1 else 0.0
        margin = StatisticsEngine._Z * std / np.sqrt(n)
        return {
            "mean_s": mean,
            "std_s": std,
            "ci_low_s": mean - margin,
            "ci_high_s": mean + margin,
            "n": int(n),
        }

    @staticmethod
    def std(times: np.ndarray) -> float:
        """Sample standard deviation of a finishing-time array."""
        return StatisticsEngine.mean_ci(times)["std_s"]

    @staticmethod
    def win_probability(times_a: np.ndarray, times_b: np.ndarray) -> float:
        """Fraction of paired runs where Strategy A finishes before B (FR-10).

        Uses ``np.mean(a < b)`` — fully vectorised, stable to ±1 % at 100k
        iterations (NFR-4) thanks to pairing by ``sim_index``.
        """
        a = np.asarray(times_a, dtype=np.float64)
        b = np.asarray(times_b, dtype=np.float64)
        if a.size != b.size:
            raise ValueError(
                f"Paired win probability requires equal sizes, got {a.size} vs {b.size}"
            )
        if a.size == 0:
            raise ValueError("Cannot compute win probability on empty arrays")
        return float(np.mean(a < b))

    @staticmethod
    def sensitivity(
        times: np.ndarray,
        pit_counts: np.ndarray,
        old_pit_s: float,
        new_pit_s: float,
    ) -> np.ndarray:
        """Re-derive total times under a different pit-stop-loss assumption.

        ``adjusted = stored_time - (old_pit_s * pit_count) + (new_pit_s * pit_count)``

        This is the whole trick behind FR-14 / NFR-3: the slider recomputes
        from the denormalised ``pit_stop_count`` column — pure in-memory NumPy,
        no re-simulation, no SQLite touch.
        """
        times = np.asarray(times, dtype=np.float64)
        counts = np.asarray(pit_counts, dtype=np.float64)
        if times.size != counts.size:
            raise ValueError("times and pit_counts must be the same length")
        return times - (old_pit_s * counts) + (new_pit_s * counts)

    @staticmethod
    def kde(sample: np.ndarray, n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """Smooth kernel-density estimate for the PDF overlay (FR-13).

        Returns ``(x_grid, density)``.
        """
        sample = np.asarray(sample, dtype=np.float64)
        if sample.size < 2:
            raise ValueError("KDE requires at least two samples")
        try:
            kernel = gaussian_kde(sample)
        except Exception as exc:  # degenerate input (zero bandwidth)
            raise ValueError(f"Cannot estimate density: {exc}") from exc
        lo, hi = float(sample.min()), float(sample.max())
        span = hi - lo if hi > lo else 1.0
        x = np.linspace(lo - 0.05 * span, hi + 0.05 * span, n_points)
        return x, kernel(x)
