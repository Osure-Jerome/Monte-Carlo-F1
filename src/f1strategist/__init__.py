"""f1strategist — Stochastic F1 Race Strategist.

Monte Carlo simulation engine + interactive dashboard for F1 pit-stop strategy.
"""

__version__ = "0.1.0"

__all__ = [
    "TyreCompound",
    "Track",
    "Car",
    "Stint",
    "RaceStrategy",
    "RaceEngine",
    "MonteCarloRunner",
    "StatisticsEngine",
    "LapResult",
    "RunResult",
    "BatchResult",
    "SimulationRepository",
    "GeneticOptimiser",
    "load_tracks",
    "load_compounds",
    "load_cars",
]

from f1strategist.config.tyre_compound import TyreCompound
from f1strategist.config.track import Track
from f1strategist.config.car import Car
from f1strategist.config.loader import load_tracks, load_compounds, load_cars
from f1strategist.strategy.stint import Stint
from f1strategist.strategy.race_strategy import RaceStrategy
from f1strategist.output.lap_result import LapResult
from f1strategist.output.run_result import RunResult, BatchResult
from f1strategist.engine.race_engine import RaceEngine
from f1strategist.engine.montecarlo import MonteCarloRunner
from f1strategist.statistics.statistics_engine import StatisticsEngine
from f1strategist.repository.simulation_repository import SimulationRepository
from f1strategist.optimiser.genetic_optimiser import GeneticOptimiser
