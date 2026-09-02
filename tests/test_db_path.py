"""Unit tests for ``resolve_db_path`` (Cloud read-only fallback, Sprint 3 fix).

Streamlit Cloud can mount the repo read-only, making the default
``<repo>/data/results.db`` unwritable. The resolver must fall back to a fully
writable location under the OS temp dir without leaving probe files behind.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from f1strategist.config.car import Car
from f1strategist.config.track import Track
from f1strategist.config.tyre_compound import TyreCompound
from f1strategist.engine.montecarlo import MonteCarloRunner
from f1strategist.engine.race_engine import RaceEngine
from f1strategist.repository.simulation_repository import SimulationRepository
from f1strategist.repository.simulation_repository import resolve_db_path
from f1strategist.strategy.race_strategy import RaceStrategy

SOFT = TyreCompound("Soft", 0.04, 18)
MEDIUM = TyreCompound("Medium", 0.03, 24)
HARD = TyreCompound("Hard", 0.02, 30)
MONZA = Track("Monza", 82.0, 53, 22.0, 0.12)
CAR = Car("default", 110.0, 1.5, 0.03, 0.15)


class TestResolveDbPath:
    def test_returns_writable_preferred(self, tmp_path):
        target = tmp_path / "results.db"
        resolved = resolve_db_path(target)
        assert resolved == target
        # No probe files are left behind.
        leftovers = [
            p for p in tmp_path.iterdir()
            if p.name.startswith(".f1strategist_probe_")
        ]
        assert leftovers == []

    def test_falls_back_to_temp_when_preferred_read_only(self, tmp_path):
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            pytest.skip("read-only-directory test needs a non-root user")
        read_only = tmp_path / "readonly_data"
        read_only.mkdir()
        read_only.chmod(0o500)
        try:
            resolved = resolve_db_path(read_only / "results.db")
        finally:
            read_only.chmod(0o700)  # let tmp_path cleanup succeed
        assert resolved != read_only / "results.db"
        assert resolved.name == "results.db"
        temp_root = Path(tempfile.gettempdir()).resolve()
        assert temp_root in resolved.resolve().parents

    def test_default_target_is_repo_data_dir(self):
        # Sanity: with no args the resolver points at <repo>/data/results.db
        # (which is writable on a normal developer machine).
        resolved = resolve_db_path()
        assert resolved.name == "results.db"
        assert resolved.parent.name == "data"


class TestRepositoryAtFallbackPath:
    """End-to-end save/load at a temp-style path (as used in read-only hosts)."""

    def test_save_and_load_round_trip(self, tmp_path):
        db = resolve_db_path(tmp_path / "cloud_results" / "results.db")
        engine = RaceEngine(MONZA, CAR)
        runner = MonteCarloRunner(engine, n_workers=1)
        sa = RaceStrategy.from_description(
            "A", "Soft:18,Medium:20,Hard:15", [SOFT, MEDIUM, HARD],
            MONZA.total_laps)
        sb = RaceStrategy.from_description(
            "B", "Medium:53", [MEDIUM], MONZA.total_laps)
        ba = runner.run(sa, n_iterations=40, master_seed=1, parallel=False)
        bb = runner.run(sb, n_iterations=40, master_seed=2, parallel=False)

        with SimulationRepository(db) as repo:
            repo.initialize()
            batch_id = repo.save_batch(sa, sb, MONZA, CAR, 40, 42, 22.0,
                                       (ba, bb))
        with SimulationRepository(db) as repo:
            payload = repo.load_batch(batch_id)
        assert len(payload["runs_a"]) == 40
        np.testing.assert_allclose(
            np.array([r.total_time_s for r in payload["runs_a"]]),
            ba.total_times,
        )
        assert payload["strategy_a_meta"]["desc"] == "Soft:18,Medium:20,Hard:15"
