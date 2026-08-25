# System Architecture — Stochastic F1 Race Strategist

> **Source:** Phases 1-2 Blueprint.docx · **Status:** Approved for implementation
> **Scope:** Phases 1 & 2 (requirements, system design, component design)

---

## 1. Architectural Overview

The system is a **Monte Carlo race-simulation engine** paired with an **interactive analytics dashboard**. It is built from four decoupled layers that communicate through clean, typed interfaces:

```
┌────────────────────────────────────────────────────────────────────────┐
│                         PRESENTATION LAYER                              │
│                    Streamlit Dashboard (dashboard.py)                    │
│              selectors · sliders · Plotly charts · GA view              │
│                     reads ONLY from SQLite; never runs                  │
│                        simulations directly                              │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ read-many (query)
┌──────────────────────────────────▼─────────────────────────────────────┐
│                         DATA LAYER                                      │
│                  SQLite (results.db) + SimulationRepository              │
│         write-once-per-experiment boundary between compute & display     │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ write-once (executemany, 1 txn)
┌──────────────────────────────────▼─────────────────────────────────────┐
│                    SIMULATION / DOMAIN LAYER                             │
│        RaceEngine (pure)  ·  MonteCarloRunner (multiprocessing.Pool)    │
│   pure function: (strategy, track, car, seed) → RunResult               │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ candidate strategies / winner
┌──────────────────────────────────▼─────────────────────────────────────┐
│                       OPTIMISATION LAYER (Bonus)                         │
│              GeneticOptimiser — feeds RaceStrategy into engine            │
│                      writes winner back via same schema                   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Cross-cutting layers** (not shown above, consumed by the domain layer):

- **Configuration layer** — `TyreCompound`, `Track`, `Car` (static, frozen dataclasses)
- **Statistics layer** — `StatisticsEngine` (stateless, pure NumPy/SciPy functions)
- **Strategy layer** — `Stint`, `RaceStrategy` (validated strategy objects)

---

## 2. Layer Responsibilities

### 2.1 Presentation Layer — Streamlit Dashboard
- **File:** `dashboard.py`
- Provides a web UI to: select a track, define 2 strategies, trigger a simulation batch, adjust the pit-stop-loss slider, and view PDF-overlay / lap-trace / GA convergence charts.
- **Constraint (NFR-3, NFR-8):** reads *only* from SQLite via `SimulationRepository`. It never triggers physics computation on a widget callback. Slider updates are in-memory NumPy recomputations over pre-loaded arrays.
- Swappable: the engine is importable independently, so a future FastAPI + React frontend is possible with no engine changes.

### 2.2 Data Layer — SQLite + SimulationRepository
- **File:** `repository.py` (context manager wrapping `sqlite3.Connection`)
- `save_batch(...)` writes one batch in a single transaction (`executemany`).
- `load_runs(...)` returns stored `RunResult` objects for analysis/visualisation.
- Indexed for `O(log n)` win-probability queries and sub-second interaction budget (NFR-5).

### 2.3 Simulation / Domain Layer — RaceEngine + MonteCarloRunner
- **Files:** `race_engine.py`, `montecarlo.py`
- `RaceEngine.simulate_race(strategy, seed) → RunResult` — a **pure function**: no I/O, no globals, no shared mutable state. Uses `np.random.Generator` seeded per call (NFR-6 reproducibility).
- `RaceEngine.simulate_batch(strategy, n_iterations, master_seed) → BatchResult` — the Sprint 2 **vectorised** path: the lap loop runs once per lap over N-element NumPy arrays (SC state and tyre age evolve per run). Totals are identical to the scalar loop on the same seed (Day-3 checkpoint, `tests/test_batch.py`).
- `MonteCarloRunner` splits a batch into `~n_iterations / n_workers` vectorised chunks across a `multiprocessing.Pool` via `starmap` (NFR-2). On the Sprint-2 hardware the fully vectorised engine already finishes 100k runs in ~2 s sequentially, so pool overhead dominates at that size — a documented Amdahl crossover (`notes/speedup.md`).

### 2.4 Configuration Layer — TyreCompound, Track, Car
- Frozen dataclasses; one instance per entity. Loaded from config entries (NFR-12 Open/Closed: adding a compound/track = new config entry only).
- `TyreCompound.degradation(age)` encapsulates the piecewise-quadratic cliff formula, so the engine calls one method with no type-switching.
- `Track.sc_probability` is track-specific (Monza 0.12, Monaco 0.30).

### 2.5 Statistics Layer — StatisticsEngine
- **File:** `statistics.py`
- Stateless (`@staticmethod`): `mean_ci`, `std`, `win_probability` (vectorised `np.mean(a < b)`), `kde` (SciPy gaussian_kde), `sensitivity(runs, new_pit_s)`.
- Provides NFR-4 stability (±1 % win probability at 100k) and NFR-3 sub-second slider recomputation.

### 2.6 Optimisation Layer — GeneticOptimiser (Bonus, Sprint 4)
- **File:** `optimiser.py`
- Hand-rolled GA (no DEAP): population of `RaceStrategy` objects, fitness = mean finishing time from a small-N engine run. Selection (tournament) → crossover → Gaussian mutation → re-normalisation.
- Outputs a winner `RaceStrategy` stored through the same `strategy` / `strategy_stint` schema, enabling head-to-head comparison against human strategies (FR-19).

---

## 3. Class Design Summary

| Layer | Class | Responsibility |
|---|---|---|
| Config | `TyreCompound` | Degradation params; `degradation(age) → float` |
| Config | `Track` | Base lap time, pit-lane loss, SC probability; `from_name()` factory |
| Config | `Car` | Fuel load/burn, fuel time effect, driver σ; `fuel_time_penalty(lap)` |
| Strategy | `Stint` | Frozen dataclass: `TyreCompound` + lap count; self-validating |
| Strategy | `RaceStrategy` | Ordered `Stint` list + metadata; validates total laps vs track |
| Output | `LapResult` | Per-lap snapshot; `__slots__` for memory efficiency |
| Output | `RunResult` | Total time, SC laps, pit count, optional `lap_trace`; `__slots__` |
| Output | `BatchResult` | Wraps all runs; `total_times() → np.ndarray` |
| Engine | `RaceEngine` | Pure `simulate_race(strategy, seed) → RunResult` |
| Engine | `MonteCarloRunner` | Parallel distribution over `multiprocessing.Pool` |
| Statistics | `StatisticsEngine` | `mean_ci`, `std`, `win_probability`, `kde`, `sensitivity` |
| Infra | `SimulationRepository` | SQLite context manager; `save_batch` / `load_runs` |
| Bonus | `GeneticOptimiser` | Hand-rolled GA; `run() → RaceStrategy` |

**Engine purity is the architectural guarantee** that delivers NFR-6 (reproducibility), NFR-8 (testability), and NFR-2 (parallel safety) simultaneously.

---

## 4. Simulation Inner Loop — `_simulate_lap()` (Sprint 2)

Executed `L × N` times per batch. Sequence per lap (per run):

```
1. Draw  u ~ Bernoulli(sc_probability)          # consumed every lap
2. Draw  v ~ N(0, driver_sigma)                 # consumed every lap (keeps batch RNG identical)
3. IF the SC state machine is active:
       lap_time = base_lap_time + sc_delta_s    # default +5 s per SC lap
       tyre_age_increment = 0                   # degradation suspended under SC
4. ELIF u < sc_probability:
       enter the SC for sc_duration_laps laps (default 3); this lap is SC lap 1
       -> same as step 3
5. ELSE (normal racing lap):
       lap_time = base_lap_time
                + compound.degradation(tyre_age)
                + car.fuel_time_penalty(fuel_remaining)
                + v
       tyre_age_increment = 1
6. Build LapResult (only if run ∈ sampled set, sim_index < 100)
```

The Safety Car is a **state machine** (Sprint 2 Day 2): one Bernoulli trigger
deploys it for `sc_duration_laps` laps, re-triggering is disabled while it is
active, and tyre degradation is frozen for the whole deployment. Stationary
SC-affected lap fraction = `3p / (1 + 2p)` (verified in `tests/test_batch.py`).

**Strategic implication of step 3/4:** a late Safety Car neutralises the
pit-stop cost — the team can change tires "for free". This is intentionally
modelled and is a headline differentiator of the engine.

### Tire degradation formula (TyreCompound.degradation)

```
Δt_tyre(age) = α · age²                         if age ≤ cliff_threshold
             = α·threshold² + β·(age−threshold)²   otherwise

α = deg_coeff (per compound)   β = α × 3 (cliff multiplier)
```

Example (Soft: α = 0.04, cliff = lap 18, β = 0.12):

| Lap age | Formula | Penalty (s) |
|---|---|---|
| 5 | 0.04 × 25 | 1.0 |
| 12 | 0.04 × 144 | 5.8 |
| 18 | 0.04 × 324 (cliff) | 13.0 |
| 22 | 13.0 + 0.12 × 16 | 14.9 |
| 25 | 13.0 + 0.12 × 49 | 18.9 |

---

## 5. Data Flow (Run-time Sequences)

### 5.1 Batch simulation (user clicks "Run Simulation")
```
Dashboard → SimulationRepository.load_config (tracks, compounds)
          → build RaceStrategy objects from UI form
          → MonteCarloRunner.run(batch spec)          # engine never touched by UI
          → per-worker: RaceEngine.simulate_batch(strategy, chunk_size, chunk_seed)
          → BatchResult assembled from chunk results
          → SimulationRepository.save_batch(batch)    # 1 transaction
          → StatisticsEngine computes stats
          → Dashboard renders PDF overlay / lap traces
```

### 5.2 Sensitivity slider (pit-stop loss 20–30 s) — NO re-simulation
```
Slider change → load batch once (cached in session)
              → StatisticsEngine.sensitivity(runs, new_pit_s)
                     adjusted = stored_total − (old_pit_s × pit_count) + (new_pit_s × pit_count)
              → win_probability = mean(adj_A < adj_B)      # pure NumPy, < 1 s
              → update chart
```
The `pit_stop_count` column is denormalised onto each `simulation_run` row to make this possible without touching `lap_result` or re-running physics.

---

## 6. Concurrency & Parallelism Model

- **Granularity:** one worker = one **vectorised chunk** of `~n_iterations/n_workers` runs (Sprint 2); the lap loop is vectorised over the chunk's run dimension. Work is embarrassingly parallel, so `multiprocessing.Pool.starmap` distributes `(chunk_seed, chunk_size, offset)` tuples.
- **Determinism:** the batch RNG draws per lap in run-major order (`rng.random(n)` then `rng.normal(0, σ, n)`), so `simulate_batch(N=1, seed)` reproduces `simulate_race(seed)` exactly; chunk seeds derive from one master seed (NFR-6).
- **I/O:** workers never touch SQLite; the parent process performs the single bulk write. No shared mutable state → no GIL contention on NumPy vectorised math.
- **Why not Ray / threading:** Ray adds a scheduler we don't need for one machine; the GIL makes threads unsuitable for Python numeric loops. Stdlib multiprocessing gives the target (NFR-2) with zero extra dependencies. Sprint-2 vectorisation alone reaches ~2 s / 100k runs sequentially, so multiprocessing is retained for very large N / many-core hosts (`notes/speedup.md`).

---

## 7. Proposed Package Layout

```
monte-carlo-f1/
├── docs/                      # architecture, schema, PRD, maths derivations (NFR-10)
├── src/f1strategist/
│   ├── __init__.py
│   ├── config/
│   │   ├── tyre_compound.py   # TyreCompound
│   │   ├── track.py           # Track
│   │   └── car.py             # Car
│   ├── strategy/
│   │   ├── stint.py           # Stint
│   │   └── race_strategy.py   # RaceStrategy
│   ├── engine/
│   │   ├── race_engine.py     # RaceEngine
│   │   └── montecarlo.py      # MonteCarloRunner
│   ├── statistics/
│   │   └── statistics_engine.py
│   ├── output/
│   │   ├── lap_result.py      # LapResult (__slots__)
│   │   └── run_result.py      # RunResult, BatchResult (__slots__)
│   ├── optimiser/
│   │   └── genetic_optimiser.py
│   └── repository/
│       └── simulation_repository.py
├── config/
│   ├── tracks.json            # NFR-12: add track = new entry
│   └── compounds.json         # NFR-12: add compound = new entry
├── data/
│   └── results.db             # SQLite (gitignored)
├── dashboard.py               # Streamlit entrypoint
├── tests/
│   ├── test_engine.py
│   ├── test_statistics.py
│   ├── test_repository.py
│   └── test_optimiser.py
├── notebooks/                 # scratch validation plots
├── Dockerfile
└── README.md
```

---

## 8. Deployment Architecture

- **Local/CI:** `python -m venv .venv` + `pip install -r requirements.txt`; containerised via Dockerfile (NFR-11).
- **Cloud (target):** Streamlit Cloud — stateless app container, SQLite DB either bundled (read-only seed data) or on attached volume/persistent store. App must run from a clean environment with no local dependencies (NFR-9).
- **Runtime budget:** batch runs execute **outside** the web request path (one-shot CLI or on-demand worker) so 100k-iteration jobs never block the dashboard.

---

## 9. Key Design Decisions & Rationale

| # | Decision | Rationale |
|---|---|---|
| 1 | Engine is a pure function of `(strategy, track, car, seed)` | Enables NFR-6, NFR-8, parallel safety (NFR-2) simultaneously |
| 2 | SQLite as write-once / read-many boundary | Satisfies NFR-5 without a server process; keeps UI sub-second (NFR-3) |
| 3 | Piecewise-quadratic degradation + cliff multiplier | Matches real tyre "cliff" behaviour; single `degradation()` method (NFR-12) |
| 4 | SC probability on `Track`, degradation suspended during SC | Replicates the real strategic value of a late Safety Car |
| 5 | `lap_result` sampled only for `sim_index < 100` | Bounds storage at ~7,000 rows/batch while preserving FR-15 traces |
| 6 | `pit_stop_count` denormalised on `simulation_run` | Enables sub-second sensitivity recomputation (FR-14 / NFR-3) |
| 7 | Strategies normalised into `strategy` + `strategy_stint` rows | GA writes novel strategies through the same schema (FR-17) |
| 8 | Hand-rolled GA instead of DEAP | Portfolio value; makes optimisation syllabus link concrete |
| 9 | NumPy vectorisation + multiprocessing | Hits NFR-1 (<200 ms/run) and NFR-2 (≤90 s/100k) without compiled bindings; Sprint-2 vectorisation alone reaches ~2 s/100k (notes/speedup.md) |

---

## 10. Extension Points (Open/Closed)

- **New tyre compound** → add entry to `compounds.json` + optionally a row in `tyre_compound`. No engine change.
- **New track** → add entry to `tracks.json` + row in `track`. No engine change.
- **N strategies** → schema already supports it; expansion is a UI concern only (currently 2 for head-to-head framing).
- **New sensitivity dimension** (e.g. fuel-effect slider) → requires richer denormalisation or re-simulation; explicitly out of scope for Sprint 1 (see Open Questions).

---

## 11. Build Order (Sprint 1 — completed)

1. `TyreCompound`, `Track`, `Car` frozen dataclasses + unit tests
2. `Stint`, `RaceStrategy` with validation
3. `RaceEngine._simulate_lap()` + `simulate_race()` with fixed-seed reproducibility test (NFR-6)
4. Benchmark single run for NFR-1 (< 200 ms)
5. `MonteCarloRunner` + 100k benchmark for NFR-2 (≤ 90 s)
6. `SimulationRepository` + first end-to-end integration test
7. `StatisticsEngine` + NFR-4 stability verification
8. Streamlit dashboard — the *last* layer, reading only from SQLite

> The database and UI are built last — the engine is validated mathematically before any presentation concerns are introduced.

## 12. Sprint 2 — Stochastic layer + 100k scaling (completed)

1. Day 1 — Gaussian driver noise drawn every lap (FR-6) + 1,000-run histogram bell curve (`notebooks/sprint2_histogram.png`)
2. Day 2 — Safety-Car state machine (`sc_duration_laps`, `sc_delta_s`) with degradation suspension
3. Day 3 — `simulate_batch` vectorised over NumPy arrays; identical-totals checkpoint vs. the scalar loop
4. Day 4 — `MonteCarloRunner` chunked parallel scaling + sequential-vs-parallel measurements (`notes/speedup.md`)
5. Day 5 — 100k statistics, ±1 % stability verification, per-run CSV export (`--csv`)

Deliverables: `tests/test_batch.py`, `scripts/sprint2_deliverables.py`, `notes/speedup.md`, `data/sprint2/*.csv`.
