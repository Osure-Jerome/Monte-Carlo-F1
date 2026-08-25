"""Sprint 2 tests — stochastic layer, vectorised batch, parallel scaling.

Covers the Day-3 checkpoint (vectorised totals == loop-reference totals on the
same seed), the Safety Car state machine (Day 2), reproducibility (NFR-6) and
the multiprocessing split (Day 4).
"""

from __future__ import annotations

import numpy as np
import pytest

from f1strategist.config.car import Car
from f1strategist.config.track import Track
from f1strategist.config.tyre_compound import TyreCompound
from f1strategist.engine.montecarlo import MonteCarloRunner
from f1strategist.engine.race_engine import RaceEngine
from f1strategist.strategy.race_strategy import RaceStrategy

MONZA = Track("Monza", 82.0, 53, pit_lane_loss_s=22.0, sc_probability=0.12)
CAR = Car(
    name="Default",
    fuel_load_kg=90.0,
    fuel_burn_per_lap=1.7,
    fuel_time_effect=0.03,
    driver_sigma=0.15,
)

SOFT = TyreCompound("Soft", 0.04, 18)
MEDIUM = TyreCompound("Medium", 0.025, 23)
HARD = TyreCompound("Hard", 0.015, 30)

STRATEGY = "Soft:18,Medium:20,Hard:15"  # 53 laps, 2 pit stops


def make_strategy(desc: str = STRATEGY, name: str = "A") -> RaceStrategy:
    compounds = [SOFT, MEDIUM, HARD]
    return RaceStrategy.from_description(name, desc, compounds, total_laps=MONZA.total_laps)


def _loop_reference(engine: RaceEngine, strategy: RaceStrategy, n: int, seed: int):
    """Python-loop reference for the Day-3 checkpoint.

    Generates randomness **in the same order** as ``simulate_batch`` (one SC
    candidate array and one noise array per lap), then iterates every run with
    plain Python loops. The vectorised implementation must reproduce these
    totals exactly.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    L = engine.track.total_laps
    stints = list(strategy.stints)
    duration = engine.sc_duration_laps

    # Pre-generate per-lap arrays in the same draw order as the vectorised batch.
    sc_all = np.empty((L, n))
    noise_all = np.empty((L, n))
    for li in range(L):
        sc_all[li] = rng.random(n)
        noise_all[li] = rng.normal(0.0, engine.car.driver_sigma, n)

    totals = np.zeros(n)
    sc_laps = np.zeros(n, dtype=np.int64)
    for i in range(n):
        stint_idx = 0
        laps_in_stint = 0
        age = 0
        sc_remaining = 0
        compound = stints[0].tyre_compound
        fuel = engine.car.fuel_load_kg
        total = 0.0
        sc_count = 0
        for li in range(L):
            # pit before lap
            if (
                laps_in_stint >= stints[stint_idx].stint_laps
                and stint_idx < len(stints) - 1
            ):
                stint_idx += 1
                compound = stints[stint_idx].tyre_compound
                age = 0
                laps_in_stint = 0
                total += engine.pit_stop_loss_s
            # draw (same per-lap order as vectorised batch)
            cand = sc_all[li, i]
            noise = noise_all[li, i]
            if sc_remaining == 0 and cand < engine.track.sc_probability:
                sc_remaining = duration
            in_sc = sc_remaining > 0
            if in_sc:
                sc_remaining -= 1
            if in_sc:
                total += engine.track.base_lap_time_s + engine.sc_delta_s
            else:
                total += (
                    engine.track.base_lap_time_s
                    + compound.degradation(age)
                    + engine.car.fuel_time_penalty(fuel)
                    + noise
                )
                age += 1
            sc_count += int(in_sc)
            laps_in_stint += 1
            fuel = max(0.0, fuel - engine.car.fuel_burn_per_lap)
        totals[i] = total
        sc_laps[i] = sc_count
    return totals, sc_laps


class TestVectorisedBatch:
    def test_batch_is_deterministic(self):
        engine = RaceEngine(MONZA, CAR)
        strategy = make_strategy()
        b1 = engine.simulate_batch(strategy, n_iterations=500, master_seed=42)
        b2 = engine.simulate_batch(strategy, n_iterations=500, master_seed=42)
        assert np.array_equal(b1.total_times, b2.total_times)
        assert np.array_equal(b1.sc_laps, b2.sc_laps)

    def test_batch_matches_sequential_single_run(self):
        """simulate_batch(N=1) must reproduce simulate_race for the same seed."""
        engine = RaceEngine(MONZA, CAR)
        strategy = make_strategy()
        for seed in (1, 7, 4242):
            batch = engine.simulate_batch(strategy, n_iterations=1, master_seed=seed)
            single = engine.simulate_race(strategy, seed=seed)
            assert batch.total_times[0] == pytest.approx(single.total_time_s)
            assert batch.sc_laps[0] == single.sc_laps

    def test_vectorized_matches_loop_reference(self):
        """Day 3 checkpoint: vectorised == loop version on the same seed."""
        engine = RaceEngine(MONZA, CAR)
        strategy = make_strategy()
        for seed in (42, 1234):
            batch = engine.simulate_batch(strategy, n_iterations=200, master_seed=seed)
            ref_totals, ref_sc = _loop_reference(engine, strategy, 200, seed)
            assert np.allclose(batch.total_times, ref_totals, atol=1e-9)
            assert np.array_equal(batch.sc_laps, ref_sc)

    def test_batch_pit_stop_count_and_indices(self):
        engine = RaceEngine(MONZA, CAR)
        strategy = make_strategy()  # 3 stints -> 2 stops
        batch = engine.simulate_batch(strategy, n_iterations=300, master_seed=9)
        assert batch.n_runs == 300
        assert np.all(batch.pit_stop_counts == 2)
        assert [r.sim_index for r in batch.runs] == list(range(300))

    def test_batch_lap_trace_consistency(self):
        engine = RaceEngine(MONZA, CAR)
        strategy = make_strategy()
        batch = engine.simulate_batch(strategy, n_iterations=250, master_seed=3)
        traces = batch.sample_traces()
        assert len(traces) == 100  # FR-15 sampling policy
        for trace in traces:
            assert len(trace) == MONZA.total_laps
        first = traces[0]
        assert first[-1].cumulative_time_s == pytest.approx(batch.runs[0].total_time_s)


SC_HIGH = Track("Monza", 82.0, 53, pit_lane_loss_s=22.0, sc_probability=0.5)


class _StubRNG:
    """Deterministic RNG stub returning queued values (state-machine tests)."""

    def __init__(self, randoms, normals):
        self._r = iter(randoms)
        self._n = iter(normals)

    def random(self):
        return next(self._r)

    def normal(self, *args, **kwargs):
        return next(self._n)


class TestSafetyCarStateMachine:
    def test_sc_state_machine_trigger_duration(self):
        """A trigger holds the SC for exactly sc_duration_laps, then releases."""
        engine = RaceEngine(MONZA, CAR, sc_duration_laps=3, sc_delta_s=5.0)
        # The candidate triggers only on the first lap (0.0 < 0.12); later
        # candidates are above threshold but must be ignored while SC is active.
        stub = _StubRNG(randoms=[0.0, 0.99, 0.99, 0.99], normals=[0.0] * 4)
        age = 0
        sc_remaining = 0
        sc_flags: list[bool] = []
        for _ in range(4):
            _t, inc, sc, sc_remaining = engine._simulate_lap(
                rng=stub,
                tyre_compound=SOFT,
                tyre_age=age,
                fuel_remaining=90.0,
                sc_remaining=sc_remaining,
            )
            sc_flags.append(sc)
            age += inc
        assert sc_flags == [True, True, True, False]  # exactly 3 SC laps
        assert age == 1  # tyres only aged on the single non-SC lap

    def test_sc_suspends_tyre_wear(self):
        """No tyre wear under SC: age is frozen and SC laps cost base + delta."""
        engine = RaceEngine(SC_HIGH, CAR, sc_duration_laps=3, sc_delta_s=5.0)
        strategy = make_strategy("Hard:53", "A")
        result = engine.simulate_race(strategy, seed=5, sample_lap_trace=True)
        for prev, cur in zip(result.lap_trace, result.lap_trace[1:]):
            if prev.safety_car and cur.safety_car:
                assert cur.tyre_age == prev.tyre_age  # frozen under SC
            elif not prev.safety_car and not cur.safety_car:
                assert cur.tyre_age == prev.tyre_age + 1
        for lap in result.lap_trace:
            if lap.safety_car:
                assert lap.lap_time_s == pytest.approx(MONZA.base_lap_time_s + 5.0)

    def test_sc_trigger_rate_matches_probability(self):
        """Day 2 checkpoint: SC-affected lap rate ≈ 3p / (1 + 2p).

        A trigger can only occur while ``sc_remaining == 0`` (no re-triggering
        under an active SC), so the stationary fraction of SC-affected laps is
        ``3p / (1 + 2p)`` (Markov chain on the remaining-SC-laps state).
        """
        engine = RaceEngine(MONZA, CAR, sc_duration_laps=3, sc_delta_s=5.0)
        strategy = make_strategy()
        batch = engine.simulate_batch(strategy, n_iterations=10_000, master_seed=11)
        p = MONZA.sc_probability
        expected = MONZA.total_laps * 3 * p / (1 + 2 * p)  # ≈ 15.4
        mean = batch.sc_laps.mean()
        assert expected * 0.95 < mean < expected * 1.05
        assert batch.sc_laps.max() <= MONZA.total_laps


class TestParallelRunner:
    def test_parallel_batch_shape_and_ordering(self):
        engine = RaceEngine(MONZA, CAR)
        runner = MonteCarloRunner(engine, n_workers=2)
        strategy = make_strategy()
        batch = runner.run(strategy, n_iterations=200, master_seed=1, parallel=True)
        assert batch.n_runs == 200
        assert [r.sim_index for r in batch.runs] == list(range(200))
        assert len(batch.sample_traces()) == 100

    def test_parallel_is_reproducible(self):
        engine = RaceEngine(MONZA, CAR)
        runner = MonteCarloRunner(engine, n_workers=2)
        strategy = make_strategy()
        b1 = runner.run(strategy, n_iterations=500, master_seed=7, parallel=True)
        b2 = runner.run(strategy, n_iterations=500, master_seed=7, parallel=True)
        assert np.array_equal(b1.total_times, b2.total_times)

    def test_invalid_n_iterations(self):
        engine = RaceEngine(MONZA, CAR)
        with pytest.raises(ValueError):
            engine.simulate_batch(make_strategy(), n_iterations=0)
        with pytest.raises(ValueError):
            MonteCarloRunner(engine).run(make_strategy(), n_iterations=0)
