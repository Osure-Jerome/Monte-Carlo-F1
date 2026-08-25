"""RunResult / BatchResult — single-race and batch outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from f1strategist.output.lap_result import LapResult


@dataclass(frozen=True, slots=True)
class RunResult:
    """The outcome of a single simulated race.

    Attributes:
        sim_index: 0-based index of this run within its batch.
        total_time_s: Total race time, including pit-stop penalties.
        pit_stop_count: Number of tyre changes performed (denormalised onto the
            DB row so the sensitivity slider never touches lap detail).
        sc_laps: Number of laps affected by a Safety Car.
        lap_trace: Optional per-lap detail; populated only for sampled runs
            (``sim_index < 100``) to bound storage (FR-15).
    """

    sim_index: int
    total_time_s: float
    pit_stop_count: int = 0
    sc_laps: int = 0
    lap_trace: Optional[tuple[LapResult, ...]] = None


@dataclass(slots=True)
class BatchResult:
    """Wraps all ``RunResult`` objects for one experiment.

    Exposes vectorised arrays for the statistics layer.
    """

    strategy: object  # RaceStrategy
    runs: tuple[RunResult, ...] = field(default_factory=tuple)

    @property
    def total_times(self) -> np.ndarray:
        """Vectorised total finishing times (s) for every run."""
        return np.asarray([r.total_time_s for r in self.runs], dtype=np.float64)

    @property
    def pit_stop_counts(self) -> np.ndarray:
        """Vectorised pit-stop counts for every run."""
        return np.asarray([r.pit_stop_count for r in self.runs], dtype=np.int64)

    @property
    def sc_laps(self) -> np.ndarray:
        """Vectorised Safety-Car-affected lap counts for every run."""
        return np.asarray([r.sc_laps for r in self.runs], dtype=np.int64)

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    def sample_traces(self) -> list[tuple[LapResult, ...]]:
        """The sampled lap traces (one per eligible run)."""
        return [r.lap_trace for r in self.runs if r.lap_trace is not None]
