"""LapResult — a per-lap snapshot of one simulated run."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LapResult:
    """One lap of a race simulation.

    ``__slots__`` keeps memory bounded for the sampled lap traces
    (``sim_index < 100``, capped at ~14k rows per batch).
    """

    lap_number: int
    lap_time_s: float
    cumulative_time_s: float
    tyre_age: int
    fuel_remaining_kg: float
    safety_car: bool
    stint_index: int
