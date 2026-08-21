"""GeneticOptimiser — hand-rolled genetic algorithm (no DEAP).

Fitness = mean finishing time from a small-N engine run; lower is better.
The GA writes its winning strategy back through the same ``RaceStrategy`` /
``strategy_stint`` schema so it can be compared head-to-head against human
strategies (FR-17, FR-18, FR-19). Hand-rolling selection / crossover / mutation
keeps the optimisation-theory link concrete for the portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from f1strategist.engine.race_engine import RaceEngine
from f1strategist.strategy.race_strategy import RaceStrategy
from f1strategist.strategy.stint import Stint


@dataclass(slots=True)
class GAConfig:
    """Hyper-parameters for the genetic search."""

    population_size: int = 100
    generations: int = 50
    fitness_runs: int = 1000       # Monte Carlo runs per fitness evaluation
    tournament_size: int = 5
    mutation_sigma: float = 2.0    # Gaussian stdev added to stint lengths
    crossover_rate: float = 0.8
    elitism: int = 2               # keep the N best unchanged
    max_stints: int = 3
    master_seed: int = 7


@dataclass(slots=True)
class GAHistory:
    """Convergence trace for charting (FR-18)."""

    best_fitness: list[float] = field(default_factory=list)
    avg_fitness: list[float] = field(default_factory=list)


class GeneticOptimiser:
    """Searches stint-length combinations minimising average finishing time.

    Chromosome encoding: list of ``max_stints`` integers summing to the track's
    lap count (after crossover/mutation the chromosome is re-normalised).
    """

    def __init__(
        self,
        engine: RaceEngine,
        compounds: list,
        config: Optional[GAConfig] = None,
    ) -> None:
        self.engine = engine
        self.compounds = list(compounds)
        if not self.compounds:
            raise ValueError("At least one tyre compound is required")
        self.config = config or GAConfig()

    # ------------------------------------------------------------------
    # Chromosome <-> strategy
    # ------------------------------------------------------------------
    def _chromosome_to_strategy(self, chromosome: list[int], name: str) -> RaceStrategy:
        stints = tuple(
            Stint(tyre_compound=self.compounds[i % len(self.compounds)], stint_laps=int(laps))
            for i, laps in enumerate(chromosome)
        )
        return RaceStrategy(
            name=name,
            stints=stints,
            source="ga",
            total_laps=self.engine.track.total_laps,
        )

    # ------------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------------
    def fitness(self, chromosome: list[int]) -> float:
        """Mean finishing time over a small Monte Carlo batch (lower is better)."""
        strategy = self._chromosome_to_strategy(chromosome, name="ga-fitness")
        batch = self._quick_batch(strategy)
        return float(batch.total_times.mean())

    def _quick_batch(self, strategy: RaceStrategy):
        # Sequential path for determinism and to avoid pool overhead at small N.
        from f1strategist.engine.montecarlo import MonteCarloRunner

        runner = MonteCarloRunner(self.engine, n_workers=1)
        return runner.run(
            strategy,
            n_iterations=self.config.fitness_runs,
            master_seed=self.config.master_seed,
            parallel=False,
        )

    # ------------------------------------------------------------------
    # GA operators
    # ------------------------------------------------------------------
    def generate_population(self, rng: np.random.Generator) -> list[list[int]]:
        """Random chromosomes summing exactly to the race length."""
        total = self.engine.track.total_laps
        population = []
        for _ in range(self.config.population_size):
            cuts = sorted(rng.integers(1, total, size=self.config.max_stints - 1))
            bounds = [0, *[int(c) for c in cuts], total]
            chromosome = [bounds[i + 1] - bounds[i] for i in range(self.config.max_stints)]
            chromosome = [max(1, int(x)) for x in chromosome]
            chromosome = self._renormalise(chromosome, total)
            population.append(chromosome)
        return population

    @staticmethod
    def _renormalise(chromosome: list[int], target: int) -> list[int]:
        """Clamp to >= 1 and re-sum exactly to ``target``."""
        chromosome = [max(1, int(x)) for x in chromosome]
        delta = target - sum(chromosome)
        i = 0
        while delta != 0:
            if delta > 0:
                chromosome[i % len(chromosome)] += 1
                delta -= 1
            else:
                if chromosome[i % len(chromosome)] > 1:
                    chromosome[i % len(chromosome)] -= 1
                    delta += 1
            i += 1
        return chromosome

    def tournament_select(
        self, rng: np.random.Generator, population: list[list[int]],
        fitnesses: list[float],
    ) -> list[int]:
        """Pick the best of ``tournament_size`` random chromosomes."""
        indices = rng.choice(len(population), size=self.config.tournament_size, replace=False)
        best = min(indices, key=lambda i: fitnesses[i])
        return population[best]

    def crossover(self, rng: np.random.Generator, a: list[int], b: list[int]) -> list[int]:
        """Blend crossover: average stint lengths, then re-normalise."""
        blended = [int(round(0.5 * (x + y))) for x, y in zip(a, b)]
        return self._renormalise(blended, self.engine.track.total_laps)

    def mutate(self, rng: np.random.Generator, chromosome: list[int]) -> list[int]:
        """Add Gaussian noise to stint lengths, then re-normalise."""
        mutated = [
            int(round(x + rng.normal(0.0, self.config.mutation_sigma)))
            for x in chromosome
        ]
        return self._renormalise(mutated, self.engine.track.total_laps)

    # ------------------------------------------------------------------
    # Evolution loop
    # ------------------------------------------------------------------
    def run(self) -> tuple[RaceStrategy, GAHistory]:
        """Run the GA and return ``(best_strategy, convergence_history)``."""
        rng = np.random.Generator(np.random.PCG64(self.config.master_seed))
        total = self.engine.track.total_laps
        population = self.generate_population(rng)
        history = GAHistory()

        for generation in range(self.config.generations):
            fitnesses = [self.fitness(c) for c in population]
            best_idx = int(np.argmin(fitnesses))
            history.best_fitness.append(float(fitnesses[best_idx]))
            history.avg_fitness.append(float(np.mean(fitnesses)))

            next_gen = [population[best_idx]]  # elitism: always keep the best
            # Additional elitism slots
            order = np.argsort(fitnesses)
            for idx in order[1 : self.config.elitism]:
                next_gen.append(population[int(idx)])

            while len(next_gen) < self.config.population_size:
                parent_a = self.tournament_select(rng, population, fitnesses)
                if rng.random() < self.config.crossover_rate:
                    parent_b = self.tournament_select(rng, population, fitnesses)
                    child = self.crossover(rng, parent_a, parent_b)
                else:
                    child = list(parent_a)
                if rng.random() < 0.5:
                    child = self.mutate(rng, child)
                next_gen.append(child)

            population = next_gen[: self.config.population_size]

        final_fitnesses = [self.fitness(c) for c in population]
        best_chromosome = population[int(np.argmin(final_fitnesses))]
        best = self._chromosome_to_strategy(best_chromosome, name="GA-optimal")
        # sanity: still sums to the race length
        assert sum(best.stint_laps) == total
        return best, history
