"""Integration tests for SimulationRepository (save -> load round trip)."""

import numpy as np
import pytest

from f1strategist.config.track import Track
from f1strategist.config.car import Car
from f1strategist.config.tyre_compound import TyreCompound
from f1strategist.engine.race_engine import RaceEngine
from f1strategist.engine.montecarlo import MonteCarloRunner
from f1strategist.repository.simulation_repository import SimulationRepository
from f1strategist.strategy.race_strategy import RaceStrategy

SOFT = TyreCompound("Soft", 0.04, 18)
MEDIUM = TyreCompound("Medium", 0.03, 24)
HARD = TyreCompound("Hard", 0.02, 30)
MONZA = Track("Monza", 82.0, 53, 22.0, 0.12)
CAR = Car("default", 110.0, 1.5, 0.03, 0.15)


def make_strategy(desc: str, name: str) -> RaceStrategy:
    return RaceStrategy.from_description(name, desc, [SOFT, MEDIUM, HARD],
                                         total_laps=MONZA.total_laps)


@pytest.fixture()
def tmp_db(tmp_path):
    return SimulationRepository(tmp_path / "test_results.db")


class TestRepository:
    def test_round_trip(self, tmp_db):
        engine = RaceEngine(MONZA, CAR)
        runner = MonteCarloRunner(engine, n_workers=1)
        sa = make_strategy("Soft:18,Medium:20,Hard:15", "A")
        sb = make_strategy("Medium:35,Hard:18", "B")

        batch_a = runner.run(sa, n_iterations=200, master_seed=1, parallel=False)
        batch_b = runner.run(sb, n_iterations=200, master_seed=2, parallel=False)

        with tmp_db as repo:
            repo.initialize()
            batch_id = repo.save_batch(
                sa, sb, MONZA, CAR,
                n_iterations=200, master_seed=42,
                pit_stop_loss_s=22.0,
                batches=(batch_a, batch_b),
            )

        # Re-open and load
        with tmp_db as repo:
            loaded = repo.load_runs(batch_id)
            assert loaded["n_iterations"] == 200
            assert len(loaded["runs_a"]) == 200
            assert len(loaded["runs_b"]) == 200

            times_a = np.array([r.total_time_s for r in loaded["runs_a"]])
            expected_a = batch_a.total_times
            np.testing.assert_allclose(times_a, expected_a)

    def test_batch_listing(self, tmp_db):
        engine = RaceEngine(MONZA, CAR)
        runner = MonteCarloRunner(engine, n_workers=1)
        sa = make_strategy("Hard:53", "A")
        sb = make_strategy("Medium:53", "B")
        ba = runner.run(sa, n_iterations=50, master_seed=1, parallel=False)
        bb = runner.run(sb, n_iterations=50, master_seed=2, parallel=False)

        with tmp_db as repo:
            repo.initialize()
            repo.save_batch(sa, sb, MONZA, CAR, 50, 42, 22.0, (ba, bb))
            batches = repo.list_batches()
            assert len(batches) == 1
            assert batches[0]["track"] == "Monza"

    def test_missing_batch_raises(self, tmp_db):
        with tmp_db as repo:
            repo.initialize()
            with pytest.raises(KeyError):
                repo.load_runs(999)


class TestDatabaseLoadPath:
    """Sprint 3 Day-2 read path: lap traces + descriptors + idempotent save."""

    def _save_one(self, tmp_db, n=60):
        engine = RaceEngine(MONZA, CAR)
        runner = MonteCarloRunner(engine, n_workers=1)
        sa = make_strategy("Soft:18,Medium:20,Hard:15", "A")
        sb = make_strategy("Medium:35,Hard:18", "B")
        ba = runner.run(sa, n_iterations=n, master_seed=7, parallel=False)
        bb = runner.run(sb, n_iterations=n, master_seed=7, parallel=False)
        with tmp_db as repo:
            repo.initialize()
            batch_id = repo.save_batch(sa, sb, MONZA, CAR, n, 7, 22.0, (ba, bb))
        return sa, sb, ba, batch_id

    def test_load_batch_round_trip(self, tmp_db):
        sa, sb, ba, batch_id = self._save_one(tmp_db)
        with tmp_db as repo:
            payload = repo.load_batch(batch_id)
        assert payload["n_iterations"] == 60
        assert len(payload["runs_a"]) == 60
        assert payload["strategy_a_meta"]["desc"] == "Soft:18,Medium:20,Hard:15"
        assert payload["strategy_b_meta"]["desc"] == "Medium:35,Hard:18"
        # Lap traces restored for the sampled runs of both strategies.
        assert len(payload["traces_a"]) == 60      # sample_count=100 > n=60
        assert len(payload["traces_a"][0][1]) == MONZA.total_laps
        times = np.array([r.total_time_s for r in payload["runs_a"]])
        np.testing.assert_allclose(times, ba.total_times)

    def test_load_lap_traces_content(self, tmp_db):
        _, _, ba, batch_id = self._save_one(tmp_db)
        with tmp_db as repo:
            a_id = repo.load_runs(batch_id)["strategy_a_id"]
            traces = repo.load_lap_traces(batch_id, a_id)
        # First returned trace is run sim_index=0 and equals the in-memory one.
        trace = traces[0][1]
        expected = ba.runs[0].lap_trace
        assert len(trace) == len(expected)
        assert trace[5].cumulative_time_s == pytest.approx(
            expected[5].cumulative_time_s)

    def test_find_batch_dedupes(self, tmp_db):
        sa, sb, ba, batch_id = self._save_one(tmp_db)
        with tmp_db as repo:
            found = repo.find_batch("Monza", 60, 7, "A", "B")
        assert found == batch_id
        with tmp_db as repo:
            assert repo.find_batch("Monza", 60, 8, "A", "B") is None
