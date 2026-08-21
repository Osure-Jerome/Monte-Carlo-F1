"""Tests for the GeneticOptimiser — encoding, operators, basic evolution."""

import numpy as np
import pytest

from f1strategist.config.track import Track
from f1strategist.config.car import Car
from f1strategist.config.tyre_compound import TyreCompound
from f1strategist.engine.race_engine import RaceEngine
from f1strategist.optimiser.genetic_optimiser import GAConfig, GeneticOptimiser

SOFT = TyreCompound("Soft", 0.04, 18)
MEDIUM = TyreCompound("Medium", 0.03, 24)
HARD = TyreCompound("Hard", 0.02, 30)
MONZA = Track("Monza", 82.0, 53, 22.0, 0.12)
CAR = Car("default", 110.0, 1.5, 0.03, 0.15)


@pytest.fixture()
def optimiser():
    engine = RaceEngine(MONZA, CAR)
    config = GAConfig(population_size=10, generations=2, fitness_runs=20,
                      max_stints=3, master_seed=3)
    return GeneticOptimiser(engine, [SOFT, MEDIUM, HARD], config=config)


class TestChromosome:
    def test_population_chromosomes_sum_to_race_length(self, optimiser):
        rng = np.random.default_rng(0)
        population = optimiser.generate_population(rng)
        assert len(population) == optimiser.config.population_size
        for chromosome in population:
            assert sum(chromosome) == MONZA.total_laps
            assert all(x >= 1 for x in chromosome)

    def test_renormalise_preserves_sum(self, optimiser):
        target = MONZA.total_laps
        for _ in range(100):
            chromosome = list(np.random.default_rng(_).integers(1, 30, size=3))
            renormalised = optimiser._renormalise(chromosome, target)
            assert sum(renormalised) == target
            assert all(x >= 1 for x in renormalised)

    def test_crossover_and_mutation_preserve_sum(self, optimiser):
        rng = np.random.Generator(np.random.PCG64(1))
        population = optimiser.generate_population(rng)
        child = optimiser.crossover(rng, population[0], population[1])
        assert sum(child) == MONZA.total_laps
        mutant = optimiser.mutate(rng, child)
        assert sum(mutant) == MONZA.total_laps


class TestEvolution:
    def test_run_returns_valid_strategy_and_history(self, optimiser):
        best, history = optimiser.run()
        assert best.source == "ga"
        assert sum(best.stint_laps) == MONZA.total_laps
        assert len(history.best_fitness) == optimiser.config.generations
        assert len(history.avg_fitness) == optimiser.config.generations
