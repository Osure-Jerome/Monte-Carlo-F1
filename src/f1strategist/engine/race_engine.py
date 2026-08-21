"""RaceEngine — the pure simulation core.

A deterministic, side-effect-free function of
``(strategy, track, car, seed) -> RunResult``. No I/O, no globals, no shared
mutable state. Each call creates its own ``np.random.Generator`` seeded with the
provided ``seed``, which makes results reproducible (NFR-6), the engine trivially
testable (NFR-8) and safe to parallelise (NFR-2).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from f1strategist.config.track import SC_DELTA_S, Track
from f1strategist.config.car import Car
from f1strategist.strategy.race_strategy import RaceStrategy
from f1strategist.output.lap_result import LapResult
from f1strategist.output.run_result import RunResult


class RaceEngine:
    """Pure Monte Carlo race simulator.

    Args:
        track: The circuit to race on.
        car: The car / driver physics configuration.
        pit_stop_loss_s: Fixed time penalty (s) for each tyre change.
    """

    def __init__(self, track: Track, car: Car, pit_stop_loss_s: float = 22.0) -> None:
        self.track = track
        self.car = car
        self.pit_stop_loss_s = pit_stop_loss_s

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def simulate_race(
        self,
        strategy: RaceStrategy,
        seed: int,
        sample_lap_trace: bool = False,
        sim_index: int = 0,
    ) -> RunResult:
        """Run one full race for ``strategy`` with a fixed ``seed``.

        Args:
            strategy: The stint plan to simulate.
            seed: Random seed for this run (NFR-6 reproducibility).
            sample_lap_trace: If True, build a full per-lap trace
                (only for ``sim_index < 100`` in a batch — FR-15).
            sim_index: 0-based index of this run within its batch.

        Returns:
            A ``RunResult`` containing total time, pit count, SC laps and
            (optionally) the sampled lap trace.
        """
        rng = np.random.Generator(np.random.PCG64(seed))
        lap_trace: list[LapResult] = []
        cumulative = 0.0
        pit_stop_count = 0
        sc_laps = 0

        stints = list(strategy.stints)
        stint_idx = 0
        laps_in_stint = 0
        tyre_age = 0
        tyre_compound = stints[0].tyre_compound
        fuel_remaining = self.car.fuel_load_kg

        for lap in range(1, self.track.total_laps + 1):
            # --- Tyre change? The current stint is exhausted. ---
            if laps_in_stint >= stints[stint_idx].stint_laps and stint_idx < len(stints) - 1:
                stint_idx += 1
                tyre_compound = stints[stint_idx].tyre_compound
                tyre_age = 0
                laps_in_stint = 0
                pit_stop_count += 1
                cumulative += self.pit_stop_loss_s

            # --- Simulate this lap (FR-2, FR-3, FR-6, FR-7). ---
            lap_time, tyre_age_inc, sc = self._simulate_lap(
                rng=rng,
                tyre_compound=tyre_compound,
                tyre_age=tyre_age,
                fuel_remaining=fuel_remaining,
            )

            cumulative += lap_time
            sc_laps += int(sc)
            tyre_age += tyre_age_inc
            laps_in_stint += 1
            fuel_remaining = max(0.0, fuel_remaining - self.car.fuel_burn_per_lap)

            if sample_lap_trace:
                lap_trace.append(
                    LapResult(
                        lap_number=lap,
                        lap_time_s=round(lap_time, 6),
                        cumulative_time_s=round(cumulative, 6),
                        tyre_age=tyre_age,
                        fuel_remaining_kg=round(fuel_remaining, 4),
                        safety_car=bool(sc),
                        stint_index=stint_idx,
                    )
                )

        return RunResult(
            sim_index=sim_index,
            total_time_s=cumulative,
            pit_stop_count=pit_stop_count,
            sc_laps=sc_laps,
            lap_trace=tuple(lap_trace) if sample_lap_trace else None,
        )

    # ------------------------------------------------------------------
    # Inner loop (FR-2, FR-3, FR-6, FR-7)
    # ------------------------------------------------------------------
    def _simulate_lap(
        self,
        rng: np.random.Generator,
        tyre_compound,
        tyre_age: int,
        fuel_remaining: float,
    ) -> tuple[float, int, bool]:
        """Compute one lap's time.

        Sequence:
          1. Draw Bernoulli(sc_probability) for a Safety Car trigger (FR-7).
          2. If SC: fixed pace delta, tyre degradation suspended.
          3. Else: base lap time + degradation (FR-2) + fuel penalty (FR-3)
             + Gaussian driver noise (FR-6).

        Returns:
            ``(lap_time_s, tyre_age_increment, safety_car_triggered)``.
        """
        sc = bool(rng.random() < self.track.sc_probability)
        if sc:
            lap_time = self.track.base_lap_time_s + SC_DELTA_S
            tyre_age_increment = 0  # degradation suspended under SC
        else:
            lap_time = (
                self.track.base_lap_time_s
                + tyre_compound.degradation(tyre_age)
                + self.car.fuel_time_penalty(fuel_remaining)
                + float(rng.normal(0.0, self.car.driver_sigma))
            )
            tyre_age_increment = 1
        return lap_time, tyre_age_increment, sc
