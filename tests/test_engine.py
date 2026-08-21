"""Tests for the RaceEngine — determinism (NFR-6), pit logic, sanity (NFR-1)."""

import pytest

from f1strategist.config.track import Track
from f1strategist.config.car import Car
from f1strategist.config.tyre_compound import TyreCompound
from f1strategist.engine.race_engine import RaceEngine
from f1strategist.strategy.stint import Stint
from f1strategist.strategy.race_strategy import RaceStrategy

SOFT = TyreCompound("Soft", 0.04, 18)
MEDIUM = TyreCompound("Medium", 0.03, 24)
HARD = TyreCompound("Hard", 0.02, 30)

MONZA = Track("Monza", base_lap_time_s=82.0, total_laps=53,
              pit_lane_loss_s=22.0, sc_probability=0.12)
CAR = Car("default", fuel_load_kg=110.0, fuel_burn_per_lap=1.5,
          fuel_time_effect=0.03, driver_sigma=0.15)


def make_strategy(desc: str, name: str = "test") -> RaceStrategy:
    return RaceStrategy.from_description(name, desc, [SOFT, MEDIUM, HARD],
                                         total_laps=MONZA.total_laps)


class TestReproducibility:
    def test_same_seed_same_result(self):
        """NFR-6: identical output for a fixed seed."""
        engine = RaceEngine(MONZA, CAR)
        strat = make_strategy("Soft:18,Medium:20,Hard:15")
        r1 = engine.simulate_race(strat, seed=123)
        r2 = engine.simulate_race(strat, seed=123)
        assert r1.total_time_s == pytest.approx(r2.total_time_s)
        assert r1.pit_stop_count == r2.pit_stop_count

    def test_different_seed_different_result(self):
        """Different seeds should (overwhelmingly) diverge."""
        engine = RaceEngine(MONZA, CAR)
        strat = make_strategy("Soft:18,Medium:20,Hard:15")
        r1 = engine.simulate_race(strat, seed=1)
        r2 = engine.simulate_race(strat, seed=2)
        assert r1.total_time_s != pytest.approx(r2.total_time_s)


class TestPitLogic:
    def test_pit_count_matches_strategy_stints(self):
        engine = RaceEngine(MONZA, CAR)
        strat = make_strategy("Soft:18,Medium:20,Hard:15")  # 3 stints -> 2 stops
        result = engine.simulate_race(strat, seed=7)
        assert result.pit_stop_count == len(strat.stints) - 1

    def test_single_stint_no_pits(self):
        engine = RaceEngine(MONZA, CAR)
        strat = make_strategy("Hard:53")
        result = engine.simulate_race(strat, seed=7)
        assert result.pit_stop_count == 0

    def test_lap_trace_length_and_sampling(self):
        engine = RaceEngine(MONZA, CAR)
        strat = make_strategy("Soft:18,Medium:20,Hard:15")
        sampled = engine.simulate_race(strat, seed=7, sample_lap_trace=True)
        plain = engine.simulate_race(strat, seed=7, sample_lap_trace=False)
        assert len(sampled.lap_trace) == MONZA.total_laps
        assert plain.lap_trace is None

    def test_lap_trace_matches_total_time(self):
        engine = RaceEngine(MONZA, CAR)
        strat = make_strategy("Soft:18,Medium:20,Hard:15")
        result = engine.simulate_race(strat, seed=7, sample_lap_trace=True)
        cumulative = result.lap_trace[-1].cumulative_time_s
        assert cumulative == pytest.approx(result.total_time_s)


class TestSanity:
    def test_total_time_is_sane(self):
        """A full race should land roughly in the 60-100 minute window."""
        engine = RaceEngine(MONZA, CAR)
        strat = make_strategy("Soft:18,Medium:20,Hard:15")
        result = engine.simulate_race(strat, seed=1)
        assert 3000 < result.total_time_s < 6500  # 50-110 min for 53 laps

    def test_sc_laps_bound_by_race_length(self):
        engine = RaceEngine(MONZA, CAR)
        strat = make_strategy("Soft:18,Medium:20,Hard:15")
        result = engine.simulate_race(strat, seed=99)
        assert 0 <= result.sc_laps <= MONZA.total_laps

    def test_single_deterministic_run_is_fast(self):
        """NFR-1: a single deterministic run should complete quickly."""
        import time

        engine = RaceEngine(MONZA, CAR)
        strat = make_strategy("Soft:18,Medium:20,Hard:15")
        start = time.perf_counter()
        for _ in range(200):
            engine.simulate_race(strat, seed=42)
        elapsed = time.perf_counter() - start
        assert elapsed / 200 < 0.2  # < 200 ms per run
