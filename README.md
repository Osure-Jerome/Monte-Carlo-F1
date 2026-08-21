# Stochastic F1 Race Strategist

A **Monte Carlo simulation engine and interactive dashboard** that models Formula 1 pit-stop
strategy under uncertainty. Given two or more tyre-stint plans, the system runs each strategy
thousands of times — injecting realistic randomness from tyre degradation, fuel burn, driver
inconsistency, and Safety Car events — and produces statistically robust win-probability estimates.

> Part of a BSc Mathematics & Computer Science portfolio project.

---

## Architecture

See [`docs/architecture.md`](docs/architecture.md), [`docs/database_schema.md`](docs/database_schema.md)
and [`docs/PRD.md`](docs/PRD.md) for the full system design, database layout and product
requirements. This repository implements the package layout proposed in `docs/architecture.md`.

```
src/f1strategist/
├── config/        # TyreCompound, Track, Car + JSON config loader
├── strategy/      # Stint, RaceStrategy (validated, normalised)
├── engine/        # RaceEngine (pure), MonteCarloRunner (multiprocessing)
├── statistics/    # StatisticsEngine (mean/CI/win-probability/sensitivity)
├── output/        # LapResult, RunResult, BatchResult (__slots__)
├── optimiser/     # GeneticOptimiser (bonus, hand-rolled GA)
└── repository/    # SimulationRepository (SQLite context manager)
```

## Technology Stack

| Layer | Choice |
|---|---|
| Language | Python 3.10+ |
| Core math | NumPy, SciPy |
| Parallelism | `multiprocessing.Pool` |
| Dashboard | Streamlit |
| Charts | Plotly (Matplotlib for scratch validation) |
| Database | SQLite |
| Optimisation | Hand-rolled Genetic Algorithm |
| Container | Docker |

## Quick Start

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

# 2. Install the package + dependencies
pip install -e .

# 3. Run the test suite
pytest

# 4. Run a head-to-head comparison from the CLI
python -m f1strategist.cli --track Monaco \
    --strategy-a "Soft:18,Medium:25,Hard:35" \
    --strategy-b "Medium:40,Hard:38" \
    --iterations 10000

# 5. Launch the dashboard
streamlit run dashboard.py
```

## Status

- [x] `docs/` — architecture, database schema, PRD
- [x] Package layout scaffolded (all layers present, core engine implemented)
- [ ] Sprint 1 — deterministic physics engine validation (see build order in `docs/architecture.md`)
- [ ] Sprint 2 — stochastic layer + 100k scaling
- [ ] Sprint 3 — dashboard wired to persisted results
- [ ] Sprint 4 (bonus) — genetic-algorithm optimisation
