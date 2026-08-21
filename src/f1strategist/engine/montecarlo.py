"""MonteCarloRunner — parallel batch execution over multiprocessing.Pool."""

from __future__ import annotations

import multiprocessing
import os
from functools import partial
from typing import Optional

import numpy as np

from f1strategist.engine.race_engine import RaceEngine
from f1strategist.output.run_result import BatchResult, RunResult
from f1strategist.strategy.race_strategy import RaceStrategy

#: Number of runs for which a full lap trace is kept (FR-15 sampling policy).
SAMPLE_COUNT = 100


def _run_one(
    engine: RaceEngine,
    strategy: RaceStrategy,
    task: tuple[int, int, bool],
) -> RunResult:
    """Module-level worker function (picklable for multiprocessing)."""
    sim_index, seed, sample = task
    return engine.simulate_race(
        strategy,
        seed=seed,
        sample_lap_trace=sample,
        sim_index=sim_index,
    )


class MonteCarloRunner:
    """Distributes ``(engine, strategy, seed_i)`` tuples across a Pool.

    Each worker runs one strategy for one seed — embarrassingly parallel work.
    Workers never touch I/O; the parent assembles the ``BatchResult``.
    """

    def __init__(self, engine: RaceEngine, n_workers: Optional[int] = None) -> None:
        self.engine = engine
        self.n_workers = n_workers or os.cpu_count() or 1

    def run(
        self,
        strategy: RaceStrategy,
        n_iterations: int = 100_000,
        master_seed: int = 42,
        sample_count: int = SAMPLE_COUNT,
        parallel: bool = True,
    ) -> BatchResult:
        """Run ``n_iterations`` races for ``strategy``.

        Args:
            strategy: Strategy to simulate.
            n_iterations: Number of Monte Carlo runs.
            master_seed: Seed for the deterministic per-run seed sequence (NFR-6).
            sample_count: How many runs (by ``sim_index``) keep a full lap trace.
            parallel: Use multiprocessing when True; sequential otherwise
                (handy for tests / tiny N).

        Returns:
            A ``BatchResult`` of all runs.
        """
        if n_iterations <= 0:
            raise ValueError(f"n_iterations must be > 0, got {n_iterations}")

        # Deterministic per-run seeds derived from the master seed (NFR-6).
        seeds = np.random.default_rng(master_seed).integers(
            0, 2**31 - 1, size=n_iterations
        )
        tasks = [
            (sim_index, int(seed), sim_index < sample_count)
            for sim_index, seed in enumerate(seeds)
        ]

        # ``partial`` pickles the engine/strategy ONCE; tasks are tiny tuples.
        worker = partial(_run_one, self.engine, strategy)
        if parallel and n_iterations > 1 and self.n_workers > 1:
            with multiprocessing.Pool(self.n_workers) as pool:
                runs = pool.map(worker, tasks)
        else:
            runs = [worker(task) for task in tasks]

        return BatchResult(strategy=strategy, runs=tuple(runs))
