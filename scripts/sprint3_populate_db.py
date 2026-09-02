"""Sprint 3, Day 2 deliverable — populate ``data/results.db`` with 100k-run output.

Runs the Sprint 2 Monte Carlo engine once per experiment, persists each batch to
SQLite via ``SimulationRepository`` (idempotently — an identical stored batch is
skipped), then proves the Day-2 checkpoint:

    * a strategy_name-filtered query over 100k stored runs returns well under 1 s;
    * two Monaco batches at different seeds agree on win probability within 1 %
      (success metric: stable to +/- 1 %).

Usage:
    python scripts/sprint3_populate_db.py [--reset]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from f1strategist.config.loader import load_tracks, load_compounds, load_cars
from f1strategist.config.track import Track
from f1strategist.engine.race_engine import RaceEngine
from f1strategist.engine.montecarlo import MonteCarloRunner
from f1strategist.repository.simulation_repository import SimulationRepository
from f1strategist.statistics.statistics_engine import StatisticsEngine
from f1strategist.strategy.race_strategy import RaceStrategy

DB_PATH = ROOT / "data" / "results.db"

#: (track, n_iterations, seed, strategy_a, strategy_b)
EXPERIMENTS = [
    ("Monza", 100_000, 42, "Soft:18,Medium:20,Hard:15", "Medium:35,Hard:18"),
    ("Monaco", 100_000, 42, "Soft:18,Medium:25,Hard:35", "Medium:40,Hard:38"),
    ("Monaco", 50_000, 43, "Soft:18,Medium:25,Hard:35", "Medium:40,Hard:38"),
]


def build_result(strategy, track, car, n, seed):
    engine = RaceEngine(track, car, pit_stop_loss_s=track.pit_lane_loss_s)
    runner = MonteCarloRunner(engine)
    t0 = time.perf_counter()
    batch = runner.run(strategy, n_iterations=n, master_seed=seed, parallel=True)
    dt = time.perf_counter() - t0
    return batch, dt


def query_timing(repo: SimulationRepository, batch_id: int) -> float:
    """Strategy_name-filtered query timing over a stored 100k batch (Day 2)."""
    t0 = time.perf_counter()
    row = repo.connection.execute(
        """SELECT s.name, COUNT(*), AVG(sr.total_time_s)
           FROM simulation_run sr
           JOIN simulation_batch b ON b.id = sr.batch_id
           JOIN strategy s ON s.id = sr.strategy_id
           WHERE b.id = ? AND s.name = ?
           GROUP BY s.name""",
        (batch_id, "A"),
    ).fetchone()
    return (time.perf_counter() - t0) * 1000.0, row


def main() -> int:
    if "--reset" in sys.argv and DB_PATH.exists():
        DB_PATH.unlink()
        print(f"Removed existing {DB_PATH} (--reset)")

    tracks = load_tracks()
    compounds = load_compounds()
    cars = load_cars()
    car = cars[0]
    stats = StatisticsEngine()

    print(f"Seeding {DB_PATH}  (data file is git-ignored; regenerable)\n")
    with SimulationRepository(DB_PATH) as repo:
        repo.initialize()
        ids: dict[tuple, int] = {}
        for track_name, n, seed, desc_a, desc_b in EXPERIMENTS:
            track = Track.from_name(track_name, tracks)
            strategy_a = RaceStrategy.from_description(
                "A", desc_a, compounds, total_laps=track.total_laps)
            strategy_b = RaceStrategy.from_description(
                "B", desc_b, compounds, total_laps=track.total_laps)

            existing = repo.find_batch(track.name, n, seed,
                                       strategy_a.name, strategy_b.name)
            if existing is not None:
                print(f"  = batch already stored (Monaco/Monza n={n:,} seed={seed}) "
                      f"-> #{existing}; skipping simulation")
                ids[(track_name, n, seed)] = existing
                continue

            print(f"  simulating {track_name} n={n:,} seed={seed} "
                  f"(A={desc_a} vs B={desc_b}) ...")
            batch_a, dt_a = build_result(strategy_a, track, car, n, seed)
            batch_b, dt_b = build_result(strategy_b, track, car, n, seed)

            t0 = time.perf_counter()
            batch_id = repo.save_batch(
                strategy_a, strategy_b, track, car,
                n_iterations=n, master_seed=seed,
                pit_stop_loss_s=track.pit_lane_loss_s,
                batches=(batch_a, batch_b),
            )
            save_s = time.perf_counter() - t0
            win = stats.win_probability(batch_a.total_times, batch_b.total_times)
            print(f"    -> batch #{batch_id}: saved in {save_s:.1f}s | "
                  f"P(A)={win * 100:.1f}% | sim {dt_a:.1f}s + {dt_b:.1f}s")
            ids[(track_name, n, seed)] = batch_id

        print("\n  Day-2 checkpoint: strategy_name-filtered query timing")
        for (track_name, n, seed), batch_id in ids.items():
            ms, row = query_timing(repo, batch_id)
            print(f"    batch #{batch_id} ({track_name}, n={n:,}, seed={seed}): "
                  f"{ms:.1f} ms  ->  {row}")

        # Stability pair: Monaco 100k seed 42 vs Monaco 50k seed 43 (within ~1%)
        mono42 = ids.get(("Monaco", 100_000, 42))
        mono43 = ids.get(("Monaco", 50_000, 43))
        if mono42 is not None and mono43 is not None:
            a42 = repo.load_runs(mono42)
            a43 = repo.load_runs(mono43)
            w42 = stats.win_probability(
                np_times(a42["runs_a"]), np_times(a42["runs_b"]))
            w43 = stats.win_probability(
                np_times(a43["runs_a"]), np_times(a43["runs_b"]))
            print("\n  Success metric (NFR-4): Monaco win-probability stability")
            print(f"    seed 42 @100k -> P(A) = {w42 * 100:.1f}%")
            print(f"    seed 43 @ 50k -> P(A) = {w43 * 100:.1f}%")
            print(f"    |Δ| = {abs(w42 - w43) * 100:.2f} pp  "
                  f"(target <= 1 pp)  -> {'PASS' if abs(w42 - w43) <= 0.01 else 'FAIL'}")

    print("\nDone. Load batches in the dashboard under *Saved experiments*.")
    return 0


def np_times(runs):
    import numpy as np
    return np.asarray([r.total_time_s for r in runs], dtype=np.float64)


if __name__ == "__main__":
    raise SystemExit(main())
