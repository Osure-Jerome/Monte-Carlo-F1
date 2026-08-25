# Product Requirements Document (PRD) — Stochastic F1 Race Strategist

| | |
|---|---|
| **Product** | Stochastic F1 Race Strategist |
| **Version** | 1.0 (Phases 1–2) |
| **Author** | Jerome Maunga |
| **Date** | August 2026 |
| **Status** | Approved for implementation; **Sprints 1–2 delivered** (Sprint 3 dashboard in progress) |
| **Related docs** | `docs/architecture.md`, `docs/database_schema.md`, `Monte Carlo F1 Project Strategy.pdf`, `Phases 1-2 Blueprint.docx` |

---

## 1. Product Summary

The **Stochastic F1 Race Strategist** is a Monte Carlo simulation engine and interactive dashboard that models Formula 1 pit-stop strategy under uncertainty. Given two or more tire-stint plans, the system runs each strategy thousands of times — injecting realistic randomness from tyre degradation, fuel burn, driver inconsistency, and Safety Car events — and produces statistically robust win-probability estimates that reveal **which strategy is genuinely superior, not just lucky**.

The product serves a dual purpose:

1. **User value** — a digital "race engineer" for amateur analysts, sim-racers, and F1 fans: pick a track, define two pit-stop strategies, and get a probabilistic head-to-head verdict in seconds.
2. **Portfolio value** — a concrete application of probability theory, stochastic modelling, optimisation theory, parallel computing, and database systems for a BSc Mathematics & Computer Science degree.

---

## 2. Problem Statement

In modern motorsports, strategic pit-stop calls often decide races. Yet amateur analysts and sim-racers rely on **static spreadsheets or gut feeling**. The core problem is threefold:

1. **Static models fail** — fixed lap-time calculators ignore the stochastic nature of racing (driver inconsistency, random Safety Car deployments).
2. **Tyre physics are simplified** — most models treat degradation as linear, whereas real tyres exhibit non-linear "cliff" drop-offs.
3. **No probabilistic output** — a single "estimated race time" is statistically useless without a confidence interval or probability of success against rival strategies.

**Solution:** a Monte Carlo engine running tens of thousands of race iterations that outputs a *distribution* of outcomes — mean, standard deviation, 95 % CI, and head-to-head win probability — rather than a single deterministic number.

---

## 3. Goals & Non-Goals

### 3.1 Goals (this release — Phases 1 & 2 scope)

- **G1.** Deterministic, validated physics engine simulating a full race with tyre degradation, fuel burn, pit penalties, and driver noise.
- **G2.** Stochastic layer scaling to 100,000 iterations per strategy on an 8-core consumer machine in ≤ 90 s.
- **G3.** Statistically sound output: mean, std, 95 % CI, and stable (±1 %) head-to-head win probabilities.
- **G4.** Interactive dashboard comparing two strategies side-by-side with PDF-overlay plots, lap traces, and a pit-stop-loss sensitivity slider that updates in < 1 s **without re-running physics**.
- **G5.** Reproducible, testable, decoupled, containerisable architecture (pure engine, SQLite boundary, Streamlit shell).
- **G6 (Bonus).** Genetic-algorithm mode discovering a strategy that beats a human baseline by ≥ 0.5 s on average.

### 3.2 Non-Goals (explicitly out of scope for this release)

- Full race-dynamics fidelity (DRS, tyre temperature ODE integration, weather, fuel strategy interplay, multi-car traffic).
- Live/real-time telemetry ingestion.
- More than **two** user-defined strategies in the UI (schema supports N; expansion is a UI concern).
- Multi-sensitivity sliders (e.g. fuel-effect) — a single pit-stop-loss slider only.
- Multi-machine distributed compute (Ray, clusters).
- Monetisation, accounts, or multi-tenancy.

---

## 4. Target Users & Personas

| Persona | Description | Primary need |
|---|---|---|
| **Amateur F1 Analyst** | Follows the sport closely, writes strategy posts/threads | Quantify whether a 1-stop or 2-stop plan is genuinely better, with a defensible probability |
| **Sim-Racer** | Competing in league racing with pit-stop rules | Pre-race strategy selection with confidence intervals, not guesses |
| **F1 Fan (curious)** | Non-technical; wants an answer, not code | Interpret the result without instructions (NFR-7) |
| **Recruiter / Lecturer** | Evaluates the portfolio piece | See rigorous engineering: tests, docs, reproducibility, clean architecture |

**Key usability requirement (NFR-7):** a non-technical user must be able to run a comparison and interpret the result with zero external instructions.

---

## 5. User Stories

| ID | User story | Priority |
|---|---|---|
| US-1 | As an analyst, I can select a track (Monza or Monaco) so simulations use realistic base lap times and Safety Car rates. | P0 |
| US-2 | As an analyst, I can define two strategies as ordered stints (compound + laps) so I can compare plans like 1-stop vs 2-stop. | P0 |
| US-3 | As a user, I can trigger a simulation batch of 100k iterations per strategy and see progress. | P0 |
| US-4 | As a user, I can view overlapping PDF plots of finishing-time distributions for both strategies. | P0 |
| US-5 | As a user, I can see the head-to-head win probability ("Strategy A wins 67.3% ± 1.1%") with a 95 % confidence interval. | P0 |
| US-6 | As a user, I can adjust a pit-stop-loss slider (20–30 s) and watch win probabilities update in under a second, without re-running the simulation. | P0 |
| US-7 | As a user, I can view at least one lap trace (lap time vs lap number) for a representative run of each strategy. | P1 |
| US-8 | As a user, I can compare the AI-discovered strategy against my strategies once GA mode completes. | P1 (Bonus) |
| US-9 | As an analyst, I can watch the GA's best/average fitness converge over generations. | P1 (Bonus) |
| US-10 | As a developer, I can re-run any batch with the same seed and get identical results. | P0 (NFR-6) |
| US-11 | As a developer, I can query previously stored results without re-running the simulation. | P0 (FR-11) |

---

## 6. Functional Requirements

### 6.1 Simulation Core

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | Simulate a full race distance (configurable lap count, e.g. 70) for a given pit-stop strategy. | P0 |
| FR-2 | Model tyre degradation as a non-linear (piecewise quadratic) function of tyre age, including a cliff effect where lap-time loss accelerates sharply past a threshold. | P0 |
| FR-3 | Model fuel burn per lap and its effect on lap time (heavier car = slower lap). | P0 |
| FR-4 | Apply a pit-stop time penalty (configurable, default 22–25 s) whenever a strategy calls for a tyre change. | P0 |
| FR-5 | Support at least three tyre compounds (Soft, Medium, Hard) with distinct degradation coefficients. | P0 |
| FR-6 | Inject random lap-time variance (driver inconsistency) using a Gaussian distribution. | P0 |
| FR-7 | Randomly trigger Safety Car events per lap using a track-specific Bernoulli probability, suspending tyre degradation and adding a fixed time delta for affected laps. | P0 |
| FR-8 | Run a single strategy through the simulation engine N times (default 100,000) and record total race time for each run. | P0 |

### 6.2 Statistical Analysis

| ID | Requirement | Priority |
|---|---|---|
| FR-9 | Compute mean, standard deviation, and 95 % confidence interval of finishing time for each simulated strategy. | P0 |
| FR-10 | Compute head-to-head win probability between two strategies (Strategy A beats Strategy B in X % of paired runs). | P0 |
| FR-11 | Persist simulation output so results can be re-queried without re-running the simulation. | P0 |

### 6.3 User-Facing Dashboard

| ID | Requirement | Priority |
|---|---|---|
| FR-12 | Provide a web UI allowing a user to select a track, define 2 strategies, and trigger a simulation batch. | P0 |
| FR-13 | Display overlapping probability-density plots comparing finishing-time distributions across strategies. | P0 |
| FR-14 | Allow a user to adjust pit-stop loss via a slider and see updated win probabilities without re-running the physics simulation (recompute from stored data). | P0 |
| FR-15 | Visualise at least one individual lap trace (lap time vs lap number) for a representative run of each strategy. | P1 |
| FR-16 | Let the user select from at least two track profiles (e.g. Monza, Monaco) with different base lap times and SC rates. | P0 |

### 6.4 Optimisation — Bonus Scope

| ID | Requirement | Priority |
|---|---|---|
| FR-17 | Provide a genetic-algorithm mode that searches stint-length combinations to discover a strategy minimising average finishing time. | P1 |
| FR-18 | Display the GA's convergence over generations (best fitness, average fitness) as a chart. | P1 |
| FR-19 | Allow the AI-discovered strategy to be compared head-to-head against user-defined strategies in the dashboard. | P1 |

---

## 7. Non-Functional Requirements

| ID | Category | Requirement | Acceptance target |
|---|---|---|---|
| NFR-1 | Performance | A single deterministic race simulation completes quickly. | < 200 ms |
| NFR-2 | Performance | A full batch of 100k Monte Carlo iterations completes on an 8-core consumer machine via multiprocessing. | ≤ 90 s |
| NFR-3 | Performance | Dashboard sensitivity slider interactions update without full re-simulation. | < 1 s |
| NFR-4 | Statistical | Win-probability estimates are stable across repeated runs of the same configuration at 100k iterations. | ±1 % |
| NFR-5 | Scalability | The data layer handles millions of stored lap-time rows without query times degrading below the interaction budget. | < 1 s |
| NFR-6 | Reliability | The simulation engine produces identical output for a fixed random seed (deterministic reproducibility). | byte-identical totals |
| NFR-7 | Usability | A non-technical user can run a comparison and interpret the result without external instructions. | first-run task success |
| NFR-8 | Maintainability | Core simulation logic is decoupled from the UI layer — engine importable and testable independently of Streamlit. | engine tests run headless |
| NFR-9 | Portability | The dashboard runs from a clean cloud environment with no local-machine dependencies. | works on Streamlit Cloud / Docker |
| NFR-10 | Documentation | Mathematical derivations documented separately from code. | `docs/` with derivations |
| NFR-11 | Reproducibility | The project is containerisable (Docker) so the exact environment can be recreated by a third party. | `docker build && docker run` |
| NFR-12 | Extensibility | Adding a new tyre compound or track requires only a new config entry, not core engine modifications. | config-driven, no code change |

---

## 8. Product Design Specifications

### 8.1 Simulation Parameters (defaults)

| Parameter | Default | Notes |
|---|---|---|
| Race length | 70 laps | configurable per track |
| Pit-stop loss | 22–25 s | slider range 20–30 s in dashboard |
| Tyre compounds | Soft / Medium / Hard | distinct `deg_coeff`, `cliff_threshold` |
| Degradation model | piecewise quadratic | `α·age²` pre-cliff; `α·t² + 3α·(age−t)²` post-cliff |
| Driver noise σ | 0.15 s | per-lap Gaussian (Sprint 2 Day 1); may become track-specific later |
| SC state machine | 3-lap duration, +5 s/lap | Sprint 2 Day 2: per-lap Bernoulli trigger (`sc_probability`); degradation suspended while active; `sc_duration_laps` / `sc_delta_s` configurable |
| SC probability | Monza 0.12 / Monaco 0.30 | per-lap Bernoulli, track-specific; stationary SC-affected lap fraction = `3p/(1+2p)` |
| Iterations | 100,000 | per strategy |

### 8.2 Dashboard Screens & Interactions

**Screen 1 — Simulation Setup**
- Track dropdown (Monza, Monaco), race length display.
- Strategy builder: 2 strategies, each an ordered list of stints (compound + lap count), with validation that stint lengths sum to total laps.
- "Run Simulation" button → batch runs → progress indicator.

**Screen 2 — Results & Analysis**
- **Head-to-head card:** "Strategy A wins 67.3% ± 1.1%", mean ± CI for both strategies.
- **PDF overlay chart** (Plotly): overlapping histograms with `histnorm='probability density'` + smooth `gaussian_kde` lines.
- **Pit-stop-loss sensitivity slider (20–30 s):** live win-probability recomputation from stored data (in-memory NumPy; no re-simulation).
- **Lap trace chart (FR-15):** lap time vs lap number for a representative run of each strategy (sample set `sim_index < 100`).

**Screen 3 — GA Optimisation (Bonus)**
- Run GA (100 population, ~50 generations), convergence chart (best + average fitness), and "compare AI vs human" head-to-head result.

### 8.3 Output Reporting Standard

- Every statistical claim shown to the user includes its uncertainty: win probability with margin (e.g. `±1.1%`) or CI on finishing times.
- Results persist and can be re-queried (FR-11); the dashboard never silently re-runs expensive batches on widget interactions.

---

## 9. Data & Analytics Requirements

- **Persistent store:** SQLite (`data/results.db`) — see `docs/database_schema.md` for the full schema.
- **Key entities:** `track`, `tyre_compound`, `car`, `strategy`, `strategy_stint`, `simulation_batch`, `simulation_run`, `lap_result`.
- **Critical indexed query:** `(batch_id, strategy_id, total_time_s)` for O(log n) win-probability and CI computations.
- **Sampling policy:** full lap detail stored only for `sim_index < 100` per batch (~14k lap rows per 2-strategy batch), bounding storage while preserving FR-15 traces.
- **Reproducibility:** each batch stores `master_seed`; each run derives its own per-seed Generator (NFR-6).
- **Telemetry/analytics (non-goal this release):** no user tracking, no product analytics; only internal experiment metadata (params, seeds, status, timestamps).

---

## 10. UI / UX Requirements

| ID | Requirement |
|---|---|
| UX-1 | All numeric strategy decisions are presented with their uncertainty (win probability margin or CI) — never a bare number. |
| UX-2 | Strategy definition form validates stints sum to the race length and shows inline errors. |
| UX-3 | The pit-stop-loss slider updates the win-probability card and chart in < 1 s with no spinner/delay over the stored data. |
| UX-4 | Charts are interactive (Plotly hover/zoom) and readable at default sizes. |
| UX-5 | The app provides at least 3 distinct chart types (PDF overlay, sensitivity/win-rate, lap traces) per the portfolio success criteria. |
| UX-6 | Clear loading/status states for batch runs (running → complete → results). |
| UX-7 | A non-technical user completes a comparison without external instructions (NFR-7) — plain-language labels and a short inline explanation of the win-probability result. |

---

## 11. Acceptance Criteria (Definition of Done)

1. **FR-1…FR-11, FR-12…FR-16 all pass** their requirement checks as listed in §6 with P0 priority.
2. **NFR-1** — single deterministic run < 200 ms (benchmarked, recorded in `notes/`).
3. **NFR-2** — 100k × 2 strategies on 8 cores ≤ 90 s (speedup table in `notes/speedup.md`).
4. **NFR-3** — slider interaction updates in < 1 s over 100k rows, verified by measurement.
5. **NFR-4** — win probability stable within ±1 % across repeated 100k runs of identical config.
6. **NFR-6** — fixed-seed integration test reproduces identical totals; committed as a test.
7. **NFR-9 / NFR-11** — app runs from Docker and from a clean Streamlit Cloud deploy.
8. **Portfolio bar:** docs contain maths derivations (`docs/`); README embeds a 3-minute demo video; Git history shows daily commits tagged `sprintN-complete`.
9. **Bonus (Sprint 4):** GA-discovered strategy beats the human baseline by ≥ 0.5 s average finishing time in a 100k head-to-head.

---

## 12. Release Plan & Milestones

| Milestone | Scope | Exit criteria |
|---|---|---|
| **M0 — Setup** | Python 3.10+, venv, Git/GitHub, deps (numpy, pandas, scipy, streamlit, plotly, matplotlib) | Empty commit pushed; env reproducible |
| **Sprint 1 — Deterministic Engine** | `TyreCompound`, `Track`, `Car`, `Stint`, `RaceStrategy`, `RaceEngine.simulate_race` (deterministic) | Sum of lap times sane (~90–100 min); manual calculator spot-checks pass; tagged `sprint1-complete` |
| **Sprint 2 — Stochastic + Scaling** | Gaussian noise, Safety Car state machine, NumPy vectorisation, `multiprocessing.Pool`, `StatisticsEngine` (mean/std/CI/win-prob) | 100k runs ≤ 90 s; win probability stable ±1 %; tagged `sprint2-complete` |
| **Sprint 3 — Dashboard** | SQLite repository + `schema.sql`, Streamlit UI (setup, results, PDF overlay, slider, lap traces), deploy to Streamlit Cloud | Sub-second slider; live public URL; demo video; tagged `sprint3-complete` |
| **Sprint 4 (Bonus) — GA** | Chromosome encoding, fitness, tournament selection, crossover/mutation, convergence plot, AI-vs-human comparison | GA strategy beats baseline ≥ 0.5 s; tagged `sprint4-complete` |

---

## 13. Risks & Open Questions

| Risk / question | Impact | Mitigation / decision |
|---|---|---|
| `TyreCompound` as Enum vs runtime dataclass | NFR-12 extensibility | Enum for Sprint 1 (safe MVP); migrate to config-driven dataclass in Sprint 2 |
| Driver σ as global constant vs track-specific | Realism of street-circuit variance | Keep constant for Sprint 1; revisit in calibration |
| Sensitivity generality limited to one slider | FR-14 scope | Fuel-effect slider requires richer denormalisation or re-simulation — deferred |
| Bulk write: single transaction vs chunked | Peak RAM on 200k-row inserts | Start single-transaction `executemany`; benchmark, chunk if needed |
| SC realism (fixed 25 s delta vs 3-lap neutralisation) | Fidelity vs complexity | **Resolved (Sprint 2):** 3-lap state machine with per-lap +5 s delta and degradation suspension implemented and verified (`tests/test_batch.py`) |
| Win-probability pairing method | Statistical validity | Paired runs by `sim_index` (same seeds for A and B) — reduces variance and is the honest head-to-head comparison |

---

## 14. Success Metrics

| Metric | Target |
|---|---|
| Single deterministic run time | < 200 ms |
| 100k × 2-strategy batch wall time (8-core) | ≤ 90 s |
| Slider update latency (100k rows, in memory) | < 1 s |
| Win-probability repeatability | ±1 % at 100k iterations |
| Storage bound per batch (`lap_result`) | ~14k rows |
| Chart types in dashboard | ≥ 3 |
| GA vs human-baseline advantage (bonus) | ≥ 0.5 s average |
| Reproducibility | fixed seed → identical output (test-enforced) |

---

## 15. Out of Scope (Future Backlog)

- DRS / tyre-temperature ODE / weather simulation
- Multi-car race-order replay with live leaderboard animation
- >2 strategy comparison in the UI and multi-sensitivity sliders
- Distributed compute (Ray / multi-machine), cloud DB (Postgres)
- User accounts, saved histories, and product analytics
