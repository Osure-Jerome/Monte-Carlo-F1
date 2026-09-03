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
- [x] Sprint 3 — interactive dashboard (PDF overlay, sensitivity slider, win-rate line,
      sensitivity heatmap, lap traces) wired to persisted SQLite results
- [x] Sprint 3 deploy — live URL + demo video; read-only Cloud mounts fall back to an
      ephemeral temp DB so persist/Saved experiments still work
- [x] Sprint 4 (bonus) — genetic-algorithm optimisation: dashboard GA mode with
      convergence chart + AI-vs-human head-to-head; GA beats baseline ≥ 0.5 s at 100k

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

## Sprint 3 (delivered)

The dashboard reads **only** from in-memory results (or stored SQLite rows) while you
interact — it never re-runs physics on a widget change (NFR-3, NFR-8).

- **Saved experiments (SQLite)** — every persisted batch is browsable under
  *Saved experiments* and can be rendered straight from `data/results.db`, with no
  re-simulation. See `scripts/sprint3_populate_db.py` to seed the database with the
  full 100k-run Sprint 2 output:

  ```bash
  python scripts/sprint3_populate_db.py          # append missing experiments
  python scripts/sprint3_populate_db.py --reset  # rebuild data/results.db from scratch
  ```

  Day-2 checkpoint verified on the seeded database: a `strategy_name`-filtered query
  over a 100k-run batch returns in **~70 ms** (target: < 1 s).
- **PDF overlay (Day 3)** — overlapping `probability density` histograms with a
  SciPy `gaussian_kde` smooth line per strategy.
- **Sensitivity slider (Day 4)** — a 20–30 s pit-stop-loss slider re-derives every
  chart in-memory from the stored `pit_stop_count` column (sub-second, no physics).
- **Win-rate sensitivity line & 2-D sensitivity heatmap** — P(A beats B) across the
  shared pit loss, plus a heatmap over asymmetric (A × B) pit losses → **4 chart
  types** in total (PDF overlay, lap traces, win-rate line, sensitivity heatmap).
- **Lap-trace plot (FR-15)** — representative lap-time trace from sampled runs
  (`sim_index < 100`), restored from the database on load.
- **Repo fix** — `save_batch` now persists `tyre_compound` + `strategy_stint` rows, so
  stored strategies keep their full stint descriptions.
- **Tests** — added statistics sweep/grid unit tests and repository DB-load integration
  tests (53 total pass).

```bash
# Re-seed the DB with the Sprint-2 100k output, then launch
python scripts/sprint3_populate_db.py --reset
streamlit run dashboard.py            # -> http://localhost:8501
```

## Sprint 4 (delivered) — genetic-algorithm optimisation (bonus)

A hand-rolled GA (no DEAP) searches **stint lengths** over the Soft/Medium/Hard
compounds to minimise mean finishing time (FR-17). Chromosome = stint lengths
summing to the lap count; tournament selection → blend crossover → Gaussian
mutation → re-normalisation. A deterministic master seed (NFR-6) means the same
configuration always reproduces the same optimum.

- **GA mode in the dashboard** — switch *Mode → GA optimisation*, pick a human
  baseline + search size and watch a live progress bar. Results:
  - **convergence chart** — best & average fitness per generation (FR-18);
  - the discovered stint plan and its **GA-vs-human advantage**;
  - the full head-to-head analysis (PDF overlay, win-rate line, sensitivity
    heatmap, lap traces) of GA-optimal vs the human baseline (FR-19 / G6).
- **CLI** — scriptable search + 100k validation (`python -m f1strategist.cli ga --help`):

  ```bash
  python -m f1strategist.cli ga --track Monaco \
      --population 40 --generations 20 --fitness-runs 2000 \
      --iterations 100000 --persist          # saves the batch to results.db
  ```

- **Validation (Sprint 4 exit criterion — 100k head-to-head, seed 42)**:

  | Track | GA-optimal | Human baseline | GA advantage | P(GA wins) |
  |---|---|---|---|---|
  | Monaco | Soft:21,Medium:26,Hard:31 | Medium:40,Hard:38 | **+20.72 s** | 76.1 % |
  | Monza  | Soft:15,Medium:17,Hard:21 | Medium:35,Hard:18 | **+88.12 s** | 98.3 % |

  → exceeds the ≥ 0.5 s acceptance target on both tracks (GA batches #5/#6 in
  `data/results.db`, browsable under *Saved experiments*).
- **Tests** — progress hook, deterministic reproduction, beating a naive
  Soft-only stint, GA-winner persistence with `source='ga'` (61 total pass).

```bash
streamlit run dashboard.py     # Mode -> GA optimisation, or use the CLI GA above
```

## Deploy (Day 5 — manual steps)

1. Push this branch to GitHub: `git push -u origin Sprint-3`.
2. Go to <https://share.streamlit.io> (or streamlit.io/cloud) → **Create app** →
   point at this repo, branch `Sprint-3`, file `dashboard.py`.

   > No extra setup needed: `dashboard.py` bootstraps the `src/` layout onto
   > `sys.path` itself, so it imports `f1strategist` even though Streamlit Cloud
   > only installs `requirements.txt` (and not `pip install -e .`).
   >
   > If Cloud mounts the repo read-only, the app auto-falls back to a writable
   > **ephemeral** temp DB (`<tmpdir>/f1strategist/results.db`) and labels it as
   > such in the sidebar — *persist* / *Saved experiments* still work for the
   > session. To keep a persistent, pre-seeded DB on Cloud instead, point
   > `SimulationRepository` at a writable external volume and seed it via
   > `scripts/sprint3_populate_db.py`.
3. Paste the resulting public URL below, then open it in an **incognito window** to
   confirm it runs with no local dependencies.
4. Record a ~3-minute walkthrough (Loom/YouTube) and embed the link below.

> **Live URL:** _https://monte-carlo-f1-k8s4iiduvfvch6wm3fecnd.streamlit.app/_
>
> **Demo video:** _https://drive.google.com/uc?id=1KqGGRSYbfEzHuMyFk-q0ru4jh9CUZ1Ud&export=download_
