"""RaceEngine — the pure simulation core (Sprint 2: stochastic + vectorised).

A deterministic, side-effect-free function of
``(strategy, track, car, seed) -> RunResult``. No I/O, no globals, no shared
mutable state. Each call creates its own ``np.random.Generator`` seeded with the
provided ``seed``, which makes results reproducible (NFR-6), the engine trivially
testable (NFR-8) and safe to parallelise (NFR-2).

Sprint 2 changes (Monte Carlo F1 Project Strategy.pdf, Weeks 2-3):
  * Gaussian driver noise is drawn on **every** lap so the sequential and the
    vectorised batch paths consume identical RNG streams (Day 1).
  * Safety Car is a **state machine**: a Bernoulli trigger puts the field under
    SC for ``sc_duration_laps`` laps, during which degradation is suspended
    (no tyre wear) and a fixed ``sc_delta_s`` is added per lap (Day 2).
  * ``simulate_batch`` vectorises the hot lap loop over NumPy arrays, removing
    the per-run Python loop (Day 3): O(L) Python iterations over N-element
    arrays instead of O(L * N) scalar loop iterations.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from f1strategist.config.track import Track
from f1strategist.config.car import Car
from f1strategist.strategy.race_strategy import RaceStrategy
from f1strategist.output.lap_result import LapResult
from f1strategist.output.run_result import BatchResult, RunResult


class RaceEngine:
    """Pure Monte Carlo race simulator.

    Args:
        track: The circuit to race on.
        car: The car / driver physics configuration.
        pit_stop_loss_s: Fixed time penalty (s) for each tyre change.
        sc_duration_laps: How many laps a Safety Car stays deployed once
            triggered (Sprint 2 state machine, default 3).
        sc_delta_s: Fixed per-lap pace delta (s) under the Safety Car
            (Sprint 2 default 5 s, vs. the 25 s single-lap model of Sprint 1).
    """

    def __init__(
        self,
        track: Track,
        car: Car,
        pit_stop_loss_s: float = 22.0,
        sc_duration_laps: int = 3,
        sc_delta_s: float = 5.0,
    ) -> None:
        if sc_duration_laps <= 0:
            raise ValueError(
                f"sc_duration_laps must be > 0, got {sc_duration_laps}"
            )
        self.track = track
        self.car = car
        self.pit_stop_loss_s = pit_stop_loss_s
        self.sc_duration_laps = int(sc_duration_laps)
        self.sc_delta_s = sc_delta_s

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

        Draws exactly one Bernoulli SC candidate and one Gaussian noise value
        per lap, in the same order as ``simulate_batch`` — so
        ``simulate_batch(N=1, master_seed=s)`` reproduces
        ``simulate_race(seed=s)`` (verified by test).

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
        sc_remaining = 0
        tyre_compound = stints[0].tyre_compound
        fuel_remaining = self.car.fuel_load_kg

        for lap in range(1, self.track.total_laps + 1):
            # --- Tyre change? The current stint is exhausted. ---
            if (
                laps_in_stint >= stints[stint_idx].stint_laps
                and stint_idx < len(stints) - 1
            ):
                stint_idx += 1
                tyre_compound = stints[stint_idx].tyre_compound
                tyre_age = 0
                laps_in_stint = 0
                pit_stop_count += 1
                cumulative += self.pit_stop_loss_s

            # --- Simulate this lap (FR-2, FR-3, FR-6, FR-7 + SC state machine). ---
            lap_time, tyre_age_inc, sc, sc_remaining = self._simulate_lap(
                rng=rng,
                tyre_compound=tyre_compound,
                tyre_age=tyre_age,
                fuel_remaining=fuel_remaining,
                sc_remaining=sc_remaining,
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

    def simulate_batch(
        self,
        strategy: RaceStrategy,
        n_iterations: int = 100_000,
        master_seed: int = 42,
        sample_count: int = 100,
        sim_index_offset: int = 0,
    ) -> BatchResult:
        """Run ``n_iterations`` races in one vectorised batch (Sprint 2, Day 3).

        The outer loop is over laps (``L`` Python iterations); every inner
        computation is a NumPy operation over the full run dimension ``N``.
        Pit-stop boundaries, tyre compounds and fuel levels are deterministic
        per lap, so they are precomputed as length-``L`` arrays. Only the SC
        state and tyre age evolve per run, tracked as length-``N`` arrays.

        ``master_seed`` makes the whole batch reproducible (NFR-6); the RNG
        draw order per lap is ``(SC candidate, noise)``, identical to
        ``simulate_race``.

        Args:
            strategy: Strategy to simulate.
            n_iterations: Number of Monte Carlo runs in this batch.
            master_seed: Seed for the batch RNG (NFR-6 reproducibility).
            sample_count: How many leading runs keep a full lap trace
                (``sim_index < sample_count``, FR-15 sampling policy).
            sim_index_offset: Global ``sim_index`` of the first run in this
                batch (used when a larger run is split across workers).

        Returns:
            A ``BatchResult`` of all runs.
        """
        N = n_iterations
        if N <= 0:
            raise ValueError(f"n_iterations must be > 0, got {N}")
        L = self.track.total_laps
        base = self.track.base_lap_time_s
        p = self.track.sc_probability
        sigma = self.car.driver_sigma
        fuel_effect = self.car.fuel_time_effect
        fuel_burn = self.car.fuel_burn_per_lap
        fuel_load = self.car.fuel_load_kg
        rng = np.random.Generator(np.random.PCG64(master_seed))

        stints = list(strategy.stints)
        n_pits = len(stints) - 1

        # --- Deterministic per-lap plan (independent of SC randomness). ---
        stint_ends = np.cumsum([s.stint_laps for s in stints])            # (S,)
        laps_1 = np.arange(1, L + 1)                                      # 1-based
        stint_idx_lap = np.searchsorted(stint_ends, laps_1, side="left")  # (L,)
        is_pit_lap = np.zeros(L, dtype=bool)                              # (L,)
        for end in stint_ends[:-1]:                                       # 0-based idx
            is_pit_lap[end] = True                                        # lap end+1

        compounds = [stints[k].tyre_compound for k in stint_idx_lap]
        alpha_lap = np.array([c.deg_coeff for c in compounds])
        thresh_lap = np.array([c.cliff_threshold for c in compounds], dtype=np.int64)
        beta_lap = alpha_lap * np.array([c.cliff_multiplier for c in compounds])
        base_deg_lap = alpha_lap * thresh_lap**2

        fuel_level_lap = np.maximum(0.0, fuel_load - fuel_burn * (laps_1 - 1))
        fuel_pen_lap = fuel_effect * fuel_level_lap                       # (L,)

        # --- Per-run evolving state (all vectorised). ---
        totals = np.zeros(N)
        sc_laps = np.zeros(N, dtype=np.int64)
        age = np.zeros(N, dtype=np.int64)
        sc_remaining = np.zeros(N, dtype=np.int64)

        # --- Lap-trace buffers (first K runs only, FR-15). ---
        K = min(sample_count, N)
        trace_lap_time = np.zeros((K, L))
        trace_cumulative = np.zeros((K, L))
        trace_tyre_age = np.zeros((K, L), dtype=np.int64)
        trace_sc = np.zeros((K, L), dtype=bool)
        trace_stint = np.zeros((K, L), dtype=np.int64)
        trace_fuel = np.zeros((K, L))
        trace_totals = np.zeros(K)

        for li in range(L):
            # RNG: one SC candidate + one noise draw per run per lap.
            sc_candidate = rng.random(N)
            noise = rng.normal(0.0, sigma, N)

            # --- SC state machine (per run). ---
            trigger = (sc_remaining == 0) & (sc_candidate < p)
            sc_remaining[trigger] = self.sc_duration_laps
            in_sc = sc_remaining > 0
            sc_remaining[in_sc] -= 1

            # --- Pit stop before this lap (deterministic lap indices). ---
            if is_pit_lap[li]:
                totals += self.pit_stop_loss_s
                trace_totals += self.pit_stop_loss_s
                age[:] = 0

            # --- Degradation: vectorised piecewise-quadratic cliff. ---
            a = alpha_lap[li]
            t = thresh_lap[li]
            b = beta_lap[li]
            d0 = base_deg_lap[li]
            deg = np.where(age <= t, a * age * age, d0 + b * (age - t) ** 2)
            deg[in_sc] = 0.0  # no wear under SC

            lap_time = base + self.sc_delta_s * in_sc + (
                deg + fuel_pen_lap[li] + noise
            ) * (~in_sc)

            totals += lap_time
            sc_laps += in_sc
            age[~in_sc] += 1

            if K:
                trace_lap_time[:, li] = lap_time[:K]
                trace_totals += lap_time[:K]
                trace_cumulative[:, li] = trace_totals
                trace_tyre_age[:, li] = age[:K]
                trace_sc[:, li] = in_sc[:K]
                trace_stint[:, li] = stint_idx_lap[li]
                trace_fuel[:, li] = np.maximum(0.0, fuel_load - fuel_burn * (li + 1))

        # --- Assemble RunResult objects (cheap; traces only for first K). ---
        runs: list[RunResult] = []
        for i in range(N):
            if i < K:
                lap_trace = tuple(
                    LapResult(
                        lap_number=int(li) + 1,
                        lap_time_s=round(float(trace_lap_time[i, li]), 6),
                        cumulative_time_s=round(float(trace_cumulative[i, li]), 6),
                        tyre_age=int(trace_tyre_age[i, li]),
                        fuel_remaining_kg=round(float(trace_fuel[i, li]), 4),
                        safety_car=bool(trace_sc[i, li]),
                        stint_index=int(trace_stint[i, li]),
                    )
                    for li in range(L)
                )
            else:
                lap_trace = None
            runs.append(
                RunResult(
                    sim_index=sim_index_offset + i,
                    total_time_s=float(totals[i]),
                    pit_stop_count=n_pits,
                    sc_laps=int(sc_laps[i]),
                    lap_trace=lap_trace,
                )
            )

        return BatchResult(strategy=strategy, runs=tuple(runs))

    # ------------------------------------------------------------------
    # Inner loop (FR-2, FR-3, FR-6, FR-7 + Sprint 2 SC state machine)
    # ------------------------------------------------------------------
    def _simulate_lap(
        self,
        rng: np.random.Generator,
        tyre_compound,
        tyre_age: int,
        fuel_remaining: float,
        sc_remaining: int,
    ) -> tuple[float, int, bool, int]:
        """Compute one lap's time under the Sprint 2 SC state machine.

        Sequence (same RNG consumption as the vectorised batch):
          1. Draw a Bernoulli SC candidate (always consumed).
          2. Draw Gaussian driver noise (always consumed, used only if no SC).
          3. If SC is active: fixed per-lap pace delta, no tyre wear.
          4. Else: base lap time + degradation (FR-2) + fuel penalty (FR-3)
             + noise (FR-6).

        Returns:
            ``(lap_time_s, tyre_age_increment, safety_car_active, sc_remaining_next)``.
        """
        sc_candidate = rng.random()
        noise = rng.normal(0.0, self.car.driver_sigma)

        if sc_remaining == 0 and sc_candidate < self.track.sc_probability:
            sc_remaining = self.sc_duration_laps
        in_sc = sc_remaining > 0
        if in_sc:
            sc_remaining -= 1

        if in_sc:
            lap_time = self.track.base_lap_time_s + self.sc_delta_s
            tyre_age_increment = 0  # degradation suspended under SC
        else:
            lap_time = (
                self.track.base_lap_time_s
                + tyre_compound.degradation(tyre_age)
                + self.car.fuel_time_penalty(fuel_remaining)
                + noise
            )
            tyre_age_increment = 1
        return lap_time, tyre_age_increment, bool(in_sc), sc_remaining
