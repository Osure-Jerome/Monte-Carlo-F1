"""MonteCarloRunner — Sprint 2 parallel scaling over vectorised chunks.

A 100k-iteration batch is split into ``n_workers`` chunks; each worker runs the
fully NumPy-vectorised ``RaceEngine.simulate_batch`` on its slice of the batch
(so per-worker work is O(L) vector ops over N/n_workers runs, not a Python lap
loop over 100k runs). The parent combines the chunk ``BatchResult`` objects into
one result. Workers never touch I/O; the parent assembles the final batch.
"""

from __future__ import annotations

import multiprocessing
import os
from functools import partial
from typing import Optional

import numpy as np

from f1strategist.engine.race_engine import RaceEngine
from f1strategist.output.run_result import BatchResult
from f1strategist.strategy.race_strategy import RaceStrategy

#: Number of runs (by global ``sim_index``) that keep a full lap trace
#: (FR-15 sampling policy).
SAMPLE_COUNT = 100


def _run_chunk(
    engine: RaceEngine,
    strategy: RaceStrategy,
    chunk_seed: int,
    chunk_size: int,
    sim_index_offset: int,
    sample_count: int,
) -> BatchResult:
    """Module-level worker function (picklable for multiprocessing).

    Runs a vectorised ``simulate_batch`` on one slice of the experiment.
    """
    return engine.simulate_batch(
        strategy,
        n_iterations=chunk_size,
        master_seed=chunk_seed,
        sample_count=sample_count,
        sim_index_offset=sim_index_offset,
    )


class MonteCarloRunner:
    """Splits a batch across a ``multiprocessing.Pool`` of vectorised workers.

    Each worker processes ``~n_iterations / n_workers`` runs with the
    NumPy-vectorised engine; the parent concatenates the chunk results. Chunk
    seeds are derived deterministically from ``master_seed`` and the chunk
    index, so a fixed configuration is reproducible (NFR-6).
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
            master_seed: Seed for the deterministic per-chunk seed sequence
                (NFR-6).
            sample_count: How many runs (by global ``sim_index``) keep a full
                lap trace.
            parallel: Use multiprocessing when True; sequential otherwise
                (handy for tests / tiny N).

        Returns:
            A ``BatchResult`` of all runs.
        """
        if n_iterations <= 0:
            raise ValueError(f"n_iterations must be > 0, got {n_iterations}")

        n_chunks = 1
        if parallel and n_iterations > 1 and self.n_workers > 1:
            n_chunks = min(self.n_workers, n_iterations)

        base, rem = divmod(n_iterations, n_chunks)
        sizes = [base + (1 if i < rem else 0) for i in range(n_chunks)]
        offsets = [sum(sizes[:i]) for i in range(n_chunks)]
        seeds = [
            int(np.random.default_rng(master_seed + i).integers(0, 2**31 - 1))
            for i in range(n_chunks)
        ]
        tasks = [
            (seeds[i], sizes[i], offsets[i], max(0, sample_count - offsets[i]))
            for i in range(n_chunks)
        ]

        worker = partial(_run_chunk, self.engine, strategy)
        if n_chunks > 1:
            with multiprocessing.Pool(n_chunks) as pool:
                chunk_results = pool.starmap(worker, tasks)
        else:
            chunk_results = [worker(*tasks[0])]

        runs = tuple(r for chunk in chunk_results for r in chunk.runs)
        return BatchResult(strategy=strategy, runs=runs)
