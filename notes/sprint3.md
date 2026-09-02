# Sprint 3 — Interactive Dashboard (Week 4) — Delivery Notes

Goal: turn the terminal tool into a professional interactive web app for non-coders,
persisting results in SQLite and letting users interrogate them without re-running physics.

## Day 1 — Streamlit setup  (delivered)
- `dashboard.py` — title, Track dropdown (Monza/Monaco), Monte-Carlo-iteration input,
  editable Strategy A/B stint editors validated against the track's lap count.
- Config-driven tracks (NFR-12) replace the blueprint's free-form "Race Length" input:
  total laps come from `config/tracks.json`, and strategies must sum exactly to them.
- Verified via `scripts/dashboard_smoke.py` (headless Streamlit AppTest).

## Day 2 — Database connection  (delivered + verified)
- One experiment is persisted once via `SimulationRepository.save_batch` in a single
  transaction: `simulation_run` (100k totals + pit counts + SC laps, indexed) and sampled
  `lap_result` detail (sim_index < 100, FR-15).
- `scripts/sprint3_populate_db.py` seeds `data/results.db` (git-ignored, regenerable):

  | batch | track | n | seed | stored P(A) |
  |---|---|---|---|---|
  | #1 | Monza  | 100 000 | 42 | 98.9 % |
  | #2 | Monaco | 100 000 | 42 | 72.6 % |
  | #3 | Monaco |  50 000 | 43 | 72.9 % |

- **Day-2 checkpoint**: strategy_name-filtered query over a 100k batch → **~70 ms**
  (target < 1 s). Columns incl. `lap_number`, `lap_time_s`, `cumulative_time_s`;
  `strategy_name` reachable via the indexed `(batch_id, strategy_id, total_time_s)` join.
- Dashboard *Saved experiments* loader reads batches back via `load_batch` (totals +
  pit counts + restored lap traces + stint descriptors) — charts render with **no
  simulation**, the Day-2 read path.
- **Fix found while testing**: `save_batch` was not persisting `tyre_compound` rows, so
  `strategy_stint` (and stint descriptions) were silently empty. `_upsert_strategy` now
  upserts compounds first.

## Day 3 — PDF overlay  (delivered)
- Overlapping Plotly histograms, `histnorm='probability density'`, + SciPy
  `gaussian_kde` smooth lines (sub-sampled to 25k so the render stays sub-second).
- The two Monaco distributions separate clearly: A 2-stop beats B 1-stop 72.6 %.

## Day 4 — Sensitivity slider  (delivered + verified sub-second)
- 20–30 s slider re-derives *every* chart from stored `pit_stop_count` arrays in-memory
  (`StatisticsEngine.sensitivity`) — no re-simulation, no DB touch on drag.
- New `win_rate_vs_pit_loss` (line) and `win_rate_grid` (2-D heatmap) statistics — pure,
  unit-tested. The app now shows **4 chart types** (PDF overlay, win-rate line, sensitivity
  heatmap, lap traces) vs the 3 required by the success metrics.

## Day 5 — Deployment & demo  (partially manual — code is deploy-ready)
- `.streamlit/config.toml`, Dockerfile, `requirements.txt` already present; README deploy
  steps added with placeholder slots.
- Remaining manual steps (need a Streamlit Cloud account + a video host):
  1. `git push -u origin Sprint-3`
  2. Deploy on <https://share.streamlit.io> → repo, branch `Sprint-3`, file `dashboard.py`
  3. Incognito check of the public URL
  4. Record/embed a 3-minute demo video in the README
- Local launch verified: `streamlit run dashboard.py` on :8501, AppTest covers the full
  run-and-render flow.

## Verification
- `pytest` → 53 passed (incl. new `TestWinRateSweep`, `TestWinRateGrid`,
  `TestDatabaseLoadPath`).
- `scripts/dashboard_smoke.py` → initial render + full Run-Simulation flow OK.
- Monte Carlo stability (NFR-4): Monaco P(A) = 72.6 % (seed 42, 100k) vs
  72.9 % (seed 43, 50k) → |Δ| = 0.24 pp ≤ 1 pp.

## Deployment note — dashboard runs the vectorised engine sequentially
Streamlit executes a script on its own worker thread, where `multiprocessing`
forkserver pools cannot start reliably (observed in the test harness and true on
Streamlit Cloud). `dashboard.py` therefore runs the vectorised **sequential**
engine (Sprint 2) on the Run button — fast at the default 50k — while the heavy
100k seeding keeps parallel execution via `scripts/sprint3_populate_db.py` and the
CLI. Same engine, identical results; only the parallelism differs.
