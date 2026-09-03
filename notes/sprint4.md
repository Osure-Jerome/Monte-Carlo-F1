# Sprint 4 (Bonus) — Genetic-Algorithm Optimisation — Delivery Notes

Goal (PRD G6 / FR-17..19): stop hand-crafting rival strategies — let a genetic
algorithm *search* the stint-length space for the plan that minimises mean
finishing time, prove it beats a human baseline, and surface both the search and
the head-to-head in the dashboard.

## What was built

### GA core (already scaffolded, now finished & wired)
`src/f1strategist/optimiser/genetic_optimiser.py` — hand-rolled, no DEAP:

- **Chromosome encoding** — `max_stints` (default 3) stint lengths summing
  exactly to the track's lap count; compounds follow the Soft → Medium → Hard
  order (stint *i* → `compounds[i % len]`), matching how drivers open a race.
- **Fitness** — mean finishing time of `fitness_runs` (default 2000) Monte Carlo
  runs on the *vectorised sequential* engine (deterministic, no pool overhead at
  small N). One eval ≈ 0.05–0.07 s (Monza/Monaco at 2000 runs).
- **Operators** — tournament selection (`tournament_size=5`), blend crossover
  (average stint lengths), Gaussian mutation (`mutation_sigma=2.0` laps), then a
  `_renormalise` step that clamps ≥ 1 and re-sums to the exact lap count.
- **Evolution** — elitism (keep top-`elitism` unchanged) + a deterministic PCG64
  RNG seeded by `config.master_seed` (NFR-6: same config → same optimum).
- **New in Sprint 4** — optional `progress_callback(generation, best, avg)`
  invoked every generation so the dashboard can render a live progress bar.

### CLI subcommand
`python -m f1strategist.cli ga [--track --population --generations --fitness-runs
--ga-seed --iterations --parallel --persist --json --csv]`
Runs the GA, then validates the winner head-to-head against a human baseline
(default: one-stop style plan per track) at `--iterations` (default 100 000),
optionally persisting the batch to `data/results.db` and dumping convergence to
CSV.

### Dashboard GA mode
New sidebar **Mode** selector: *Race comparison* (existing Sprint 3 flow, now
refactored behind a shared `_render_result`) and *GA optimisation*:

- Sidebar: human baseline editor + population / generations / fitness-runs /
  head-to-head-iterations number inputs + **Run Genetic Optimisation**.
- Live progress bar + per-generation best/avg caption while the search runs.
- Results panel: discovered strategy, best fitness, **GA advantage vs human**,
  search effort, FR-18 **convergence chart** (best + average fitness), then the
  full FR-19 head-to-head analysis via the shared renderer (PDF overlay, win-rate
  line, sensitivity heatmap, lap traces).
- Optional persist → GA-vs-human batch appears under *Saved experiments*
  (GA winner stored through the same strategy schema with `source='ga'`).

## Validation — Sprint 4 exit criterion (≥ 0.5 s advantage at 100k)

GA profile: population 40 × 20 generations × 2000 fitness runs, `master_seed=7`
for the search; head-to-head at **100 000 iterations / strategy, seed 42**
(vectorised sequential; wall-clock ≈ 1 min 43 s for BOTH tracks incl. search):

| Track | GA-optimal stint plan | Human baseline | GA advantage (s) | P(GA wins) |
|---|---|---|---|---|
| Monaco (78 laps) | Soft:21, Medium:26, Hard:31 | Medium:40, Hard:38 | **+20.72** | 76.1 % |
| Monza (53 laps)  | Soft:15, Medium:17, Hard:21 | Medium:35, Hard:18 | **+88.12** | 98.3 % |

Both **PASS** the ≥ 0.5 s acceptance bar (exit criterion #9 in `docs/PRD.md`).
Batches #5 (Monaco) and #6 (Monza) are stored in `data/results.db`.

Convergence: Monaco gen0 best 6149.98 s → gen19 best 6143.00 s; Monza gen0 best
4615.39 s → gen19 best 4605.14 s — i.e. the search lands on (and slightly beats)
the hand-tuned Sprint-3 "A" strategies, while crushing the naive one-stop "B"
baseline. GA-optimal (6142.949 s) < hand-tuned A (6144.57 s at 100k) on Monaco.

## Tests

`tests/test_optimiser.py` now covers: encoding invariants (all chromosomes sum to
the race length; renormalise/crossover/mutation preserve the sum), evolution
validity + non-increasing best fitness, **progress-callback firing every
generation**, **deterministic reproduction** (same config → identical history and
optimum), **GA beats a naive Soft-mostly stint**, and **persistence of the GA
winner with `source='ga'`** through `SimulationRepository`.

Headless dashboard smoke (`scripts/dashboard_smoke.py`) now exercises three flows:
Run Simulation, Load-from-database, and the full GA-optimisation flow in AppTest.

Full suite: **61 tests pass**.

## Runtime notes

- GA wall-clock (40×20×2000, sequential fitness): ≈ 38 s (Monza) / ≈ 55 s (Monaco);
  the dashboard GA mode is the same deterministic sequential engine (Streamlit
  runs on a worker thread where multiprocessing forkserver pools cannot start).
  The 100k head-to-head validation adds a few seconds per track.
- The search is intentionally modest (fitness noise ≪ effect size on this smooth
  stint-balancing landscape); tournament selection tolerates the remaining noise.
