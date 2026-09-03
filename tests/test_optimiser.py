"""Tests for the GeneticOptimiser — encoding, operators, basic evolution."""

import numpy as np
import pytest

from f1strategist.config.track import Track
from f1strategist.config.car import Car
from f1strategist.config.tyre_compound import TyreCompound
from f1strategist.engine.race_engine import RaceEngine
from f1strategist.engine.montecarlo import MonteCarloRunner
from f1strategist.optimiser.genetic_optimiser import GAConfig, GeneticOptimiser
from f1strategist.repository.simulation_repository import SimulationRepository
from f1strategist.strategy.race_strategy import RaceStrategy

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


class TestBehaviour:
    """Sprint 4: progress hook, determinism (NFR-6) and beating a naive baseline."""

    def test_progress_callback_fires_every_generation(self, optimiser):
        calls = []
        best, history = optimiser.run(
            progress_callback=lambda gen, b, a: calls.append((gen, b, a)))
        assert len(calls) == optimiser.config.generations
        assert calls[0][0] == 0
        assert calls[-1][0] == optimiser.config.generations - 1
        bests = [b for _, b, _ in calls]
        assert bests[-1] <= bests[0] + 1e-9  # elitism -> non-increasing best

    def test_deterministic_reproduction(self):
        def run_once():
            engine = RaceEngine(MONZA, CAR)
            cfg = GAConfig(population_size=8, generations=3, fitness_runs=300,
                           max_stints=3, master_seed=11)
            ga = GeneticOptimiser(engine, [SOFT, MEDIUM, HARD], cfg)
            best, history = ga.run()
            return best.describe(), history.best_fitness, history.avg_fitness

        assert run_once() == run_once()

    def test_ga_beats_naive_soft_stint(self):
        engine = RaceEngine(MONZA, CAR)
        cfg = GAConfig(population_size=10, generations=6, fitness_runs=1500,
                       max_stints=3, master_seed=5)
        ga = GeneticOptimiser(engine, [SOFT, MEDIUM, HARD], cfg)
        best, history = ga.run()
        assert sum(best.stint_laps) == MONZA.total_laps
        # Naive: run almost the whole race on Soft (degrades hard past 18 laps).
        naive = [MONZA.total_laps - 2, 1, 1]
        assert ga.fitness(list(best.stint_laps)) <= ga.fitness(naive)


class TestPersistence:
    """GA winners persist through the same strategy schema (source='ga')."""

    def test_ga_winner_persists_and_reloads_with_source_ga(self, tmp_path):
        engine = RaceEngine(MONZA, CAR)
        cfg = GAConfig(population_size=8, generations=2, fitness_runs=200,
                       max_stints=3, master_seed=3)
        ga = GeneticOptimiser(engine, [SOFT, MEDIUM, HARD], cfg)
        best, _ = ga.run()
        human = RaceStrategy.from_description(
            "human", "Soft:18,Medium:20,Hard:15", [SOFT, MEDIUM, HARD],
            total_laps=MONZA.total_laps)
        runner = MonteCarloRunner(engine)
        b_ga = runner.run(best, n_iterations=40, master_seed=42, parallel=False)
        b_human = runner.run(human, n_iterations=40, master_seed=42, parallel=False)

        db = tmp_path / "results.db"
        with SimulationRepository(db) as repo:
            repo.initialize()
            batch_id = repo.save_batch(best, human, MONZA, CAR, 40, 42, 22.0,
                                       (b_ga, b_human))
        with SimulationRepository(db) as repo:
            payload = repo.load_batch(batch_id)
        assert payload["strategy_a_meta"]["name"] == "GA-optimal"
        assert payload["strategy_a_meta"]["source"] == "ga"
        assert payload["strategy_b_meta"]["source"] == "user"
