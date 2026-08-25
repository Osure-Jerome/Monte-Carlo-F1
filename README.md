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
- [x] Sprint 1 — deterministic physics engine validation
- [x] Sprint 2 — stochastic layer + 100k scaling
- [ ] Sprint 3 — dashboard wired to persisted results
- [ ] Sprint 4 (bonus) — genetic-algorithm optimisation

## Sprint 2 (delivered)

- **Stochastic layer** — Gaussian driver noise drawn every lap (FR-6), plus a
  **Safety-Car state machine**: a per-lap Bernoulli trigger (track
  `sc_probability`) deploys the SC for `sc_duration_laps` laps (default 3),
  during which degradation is suspended and a fixed `sc_delta_s` (default 5 s)
  is added per lap. See `docs/PRD.md` and `notes/speedup.md`.
- **100k scaling via NumPy vectorisation** — `RaceEngine.simulate_batch` runs
  the lap loop over N-element arrays (~6× faster than the scalar loop at 1k
  runs; ~14.4 s → ~1.9 s for 100k runs). `MonteCarloRunner` splits batches into
  vectorised chunks across a `multiprocessing.Pool` (tested, reproducible).
- **Artifacts** — `notebooks/sprint2_histogram.png` (Day 1 bell curve),
  `notes/speedup.md` (Day 3/4 benchmarks), `data/sprint2/*.csv` (Day 5 100k
  results), `tests/test_batch.py` (Sprint 2 verification incl. the Day-3
  loop-vs-vectorised checkpoint).

New CLI flags:

```bash
python -m f1strategist.cli --track Monaco \
    --strategy-a "Soft:18,Medium:25,Hard:35" \
    --strategy-b "Medium:40,Hard:38" \
    --iterations 100000 --parallel \
    --sc-duration 3 --sc-delta 5 \
    --csv data/sprint2/sprint2_100k.csv   # per-run totals (Day 5)
```
