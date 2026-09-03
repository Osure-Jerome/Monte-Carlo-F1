"""Stochastic F1 Race Strategist — Sprint 4 interactive dashboard.

Two modes (NFR-3 / NFR-8 — the engine is never re-touched on widget changes):

  1. **Race comparison** — "Run Simulation" computes one batch per strategy in
     memory; optionally persists it to SQLite. "Saved experiments" loads a
     stored batch *from the database* (Day-2 read path).
  2. **GA optimisation** (Sprint 4, bonus) — a hand-rolled genetic algorithm
     searches stint lengths to minimise mean finishing time (FR-17), plots
     best/average fitness per generation (FR-18) and races the winner head-
     to-head against your human baseline (FR-19, G6).

Charts (success metric: >= 3 chart types):
  - PDF overlay          : overlapping density histograms + gaussian_kde
  - Win-rate sensitivity : P(A beats B) vs shared pit-stop loss (line)
  - Sensitivity heatmap  : P(A beats B) over asymmetric (loss_A x loss_B)
  - Lap-trace plot       : representative lap time trace (FR-15)
  - GA convergence       : best + average fitness vs generation (FR-18)

Run with:
    streamlit run dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# The package uses a ``src/`` layout. Streamlit Cloud installs only
# ``requirements.txt`` (which intentionally does NOT ``pip install -e .``), so
# ``f1strategist`` would be un-importable there. Bootstrap the repo's ``src/``
# directory onto sys.path to make this entrypoint work regardless of whether
# the package is pip-installed.
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import streamlit as st

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - optional in lightweight runtimes
    go = None  # type: ignore[assignment]

from f1strategist.config.loader import load_tracks, load_compounds, load_cars
from f1strategist.config.track import Track
from f1strategist.engine.race_engine import RaceEngine
from f1strategist.engine.montecarlo import MonteCarloRunner
from f1strategist.optimiser.genetic_optimiser import GAConfig, GeneticOptimiser
from f1strategist.output.lap_result import LapResult
from f1strategist.output.run_result import BatchResult
from f1strategist.repository.simulation_repository import (
    SimulationRepository,
    resolve_db_path,
)
from f1strategist.statistics.statistics_engine import StatisticsEngine
from f1strategist.strategy.race_strategy import RaceStrategy

st.set_page_config(page_title="Stochastic F1 Race Strategist", layout="wide")

DEFAULT_ITERATIONS = 50_000
DEFAULT_SEED = 42  # NFR-6: one master seed for BOTH strategies (paired runs)
#: Prefer the repo ``data/`` dir; fall back to a writable temp DB when the
#: mount is read-only (e.g. Streamlit Cloud), so persist works there too.
_REPO_DATA_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = resolve_db_path(_REPO_DATA_DIR / "results.db")
#: True when the repo data dir was unwritable and we fell back to a temp DB.
DB_EPHEMERAL = DB_PATH.parent != _REPO_DATA_DIR
RESULT_KEY = "s3_result"
GA_KEY = "s4_ga_result"
#: Deterministic GA search seed (NFR-6): same config -> same optimum.
GA_SEED = 7

DEFAULT_STRATEGIES = {
    "Monza": {"A": "Soft:18,Medium:20,Hard:15", "B": "Medium:35,Hard:18"},
    "Monaco": {"A": "Soft:18,Medium:25,Hard:35", "B": "Medium:40,Hard:38"},
}

PIT_SWEEP = np.linspace(20.0, 30.0, 13)  # 13 x 13 heatmap grid (keeps it sub-second)


# ---------------------------------------------------------------------------
# Pure chart builders (unit/headless testable)
# ---------------------------------------------------------------------------
def _subsample(x: np.ndarray, cap: int = 25_000) -> np.ndarray:
    """Deterministic stride subsample so gaussian_kde stays sub-second."""
    x = np.asarray(x, dtype=np.float64)
    stride = max(1, x.size // cap)
    return x[::stride]


def pdf_figure(times_a, times_b, label_a: str, label_b: str) -> "go.Figure":
    """Overlapping density histograms + smooth KDE lines (Day 3 / FR-13)."""
    stats = StatisticsEngine()
    ta = np.asarray(times_a, dtype=np.float64)
    tb = np.asarray(times_b, dtype=np.float64)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=ta, histnorm="probability density",
                               name=label_a, opacity=0.55, nbinsx=120))
    fig.add_trace(go.Histogram(x=tb, histnorm="probability density",
                               name=label_b, opacity=0.55, nbinsx=120))
    for arr, label, color in ((ta, label_a, "#EF553B"), (tb, label_b, "#636EFA")):
        x_k, y_k = stats.kde(_subsample(arr))
        fig.add_trace(go.Scatter(x=x_k, y=y_k, mode="lines", name=f"{label} KDE",
                                 line=dict(width=2, color=color)))
    fig.update_layout(barmode="overlay",
                      title="Finishing-time distributions (PDF overlay)",
                      xaxis_title="Total race time (s)", yaxis_title="Density")
    return fig


def winrate_figure(times_a, times_b, counts_a, counts_b, base_pit: float,
                   losses: np.ndarray) -> "go.Figure":
    """P(A beats B) as the shared pit-stop loss sweeps 20..30 s (Day 4)."""
    wins = StatisticsEngine.win_rate_vs_pit_loss(
        times_a, times_b, counts_a, counts_b, base_pit, losses)
    fig = go.Figure(go.Scatter(x=losses, y=wins, mode="lines+markers",
                               line=dict(width=3, color="#2CA02C")))
    fig.add_hline(y=0.5, line_dash="dot", line_color="grey")
    fig.update_layout(title="Win-rate sensitivity — Strategy A vs shared pit-stop loss",
                      xaxis_title="Pit-stop loss (s)", yaxis_title="P(A beats B)",
                      yaxis=dict(range=[0, 1]))
    return fig


def heatmap_figure(times_a, times_b, counts_a, counts_b, base_pit: float,
                   losses: np.ndarray) -> "go.Figure":
    """2-D win-probability heatmap over asymmetric pit losses (A x B)."""
    grid = StatisticsEngine.win_rate_grid(times_a, times_b, counts_a, counts_b,
                                          base_pit, losses, losses)
    fig = go.Figure(go.Heatmap(
        x=np.round(losses, 2), y=np.round(losses, 2), z=grid,
        zmin=0.0, zmax=1.0, colorscale="RdYlGn",
        colorbar=dict(title="P(A beats B)")))
    fig.update_layout(title="Sensitivity heatmap — A vs B pit-stop loss (s)",
                      xaxis_title="Strategy A pit-stop loss (s)",
                      yaxis_title="Strategy B pit-stop loss (s)")
    return fig


def trace_figure(traces_a, traces_b, label_a: str, label_b: str) -> "go.Figure":
    """Lap-time vs lap for the first sampled run of each strategy (FR-15)."""
    fig = go.Figure()
    plotted = False
    for traces, label in ((traces_a, label_a), (traces_b, label_b)):
        if not traces:
            continue
        trace = traces[0]  # first stored run's full lap trace
        laps = [lap.lap_number for lap in trace]
        times = [lap.lap_time_s for lap in trace]
        fig.add_trace(go.Scatter(x=laps, y=times, mode="lines", name=label))
        plotted = True
    if plotted:
        fig.update_layout(title="Representative lap trace (lap time vs lap)",
                          xaxis_title="Lap", yaxis_title="Lap time (s)")
    return fig


def convergence_figure(best_fitness: list, avg_fitness: list) -> "go.Figure":
    """Best + average fitness per generation (FR-18 / US-9)."""
    gens = list(range(1, len(best_fitness) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gens, y=best_fitness, mode="lines+markers",
                             name="Best fitness",
                             line=dict(width=3, color="#2CA02C")))
    fig.add_trace(go.Scatter(x=gens, y=avg_fitness, mode="lines",
                             name="Average fitness",
                             line=dict(width=1.5, dash="dot", color="#7f7f7f")))
    fig.update_layout(title="GA convergence — mean finishing time per generation",
                      xaxis_title="Generation", yaxis_title="Mean time (s)")
    return fig


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------
def _arrays(runs) -> dict[str, np.ndarray]:
    return {
        "times": np.asarray([r.total_time_s for r in runs], dtype=np.float64),
        "counts": np.asarray([r.pit_stop_count for r in runs], dtype=np.int64),
        "sc_laps": np.asarray([r.sc_laps for r in runs], dtype=np.int64),
    }


def result_from_batches(batch_a: BatchResult, batch_b: BatchResult,
                        label_a: str, label_b: str, track: Track,
                        base_pit: float, source: str) -> dict:
    """Package two in-memory batches into the renderable result dict."""
    a, b = _arrays(batch_a.runs), _arrays(batch_b.runs)
    return {
        "times_a": a["times"], "times_b": b["times"],
        "counts_a": a["counts"], "counts_b": b["counts"],
        "sc_a": a["sc_laps"], "sc_b": b["sc_laps"],
        "label_a": label_a, "label_b": label_b,
        "track_name": track.name, "total_laps": track.total_laps,
        "base_lap_time_s": track.base_lap_time_s,
        "base_pit": float(base_pit),
        "n_iterations": batch_a.n_runs,
        "source": source,
        "traces_a": batch_a.sample_traces(),
        "traces_b": batch_b.sample_traces(),
    }


def result_from_database(batch: dict, summary: dict, base_pit: float) -> dict:
    """Package a ``SimulationRepository.load_batch`` payload into the result dict."""
    a, b = _arrays(batch["runs_a"]), _arrays(batch["runs_b"])
    meta_a, meta_b = batch["strategy_a_meta"], batch["strategy_b_meta"]
    return {
        "times_a": a["times"], "times_b": b["times"],
        "counts_a": a["counts"], "counts_b": b["counts"],
        "sc_a": a["sc_laps"], "sc_b": b["sc_laps"],
        "label_a": f"{meta_a['name']} · {meta_a['desc']}",
        "label_b": f"{meta_b['name']} · {meta_b['desc']}",
        "track_name": batch["track_name"],
        "total_laps": 0,
        "base_lap_time_s": 0.0,
        "base_pit": float(base_pit),
        "n_iterations": batch["n_iterations"],
        "source": (f"loaded from database batch #{batch['batch_id']} · "
                   f"seed {batch['master_seed']} · {summary['created_at']}"),
        "traces_a": [tr for _, tr in batch["traces_a"]],
        "traces_b": [tr for _, tr in batch["traces_b"]],
    }


def _persist(batch_a, batch_b, strategy_a, strategy_b, track, car,
             iterations: int) -> tuple[int, bool]:
    """Write an experiment idempotently; returns (batch_id, inserted)."""
    with SimulationRepository(DB_PATH) as repo:
        repo.initialize()
        existing = repo.find_batch(track.name, iterations, DEFAULT_SEED,
                                   strategy_a.name, strategy_b.name)
        if existing is not None:
            return existing, False
        batch_id = repo.save_batch(strategy_a, strategy_b, track, car,
                                   n_iterations=iterations,
                                   master_seed=DEFAULT_SEED,
                                   pit_stop_loss_s=track.pit_lane_loss_s,
                                   batches=(batch_a, batch_b))
        return batch_id, True


def _render_result(result: dict) -> None:
    """Metrics + charts for a packaged result dict (shared by both modes).

    All slider / widget changes are in-memory NumPy recomputations over the
    stored arrays — the physics engine is never re-run (NFR-3).
    """
    stats = StatisticsEngine()
    base_times_a, base_times_b = result["times_a"], result["times_b"]
    counts_a, counts_b = result["counts_a"], result["counts_b"]
    base_pit = result["base_pit"]

    with st.expander(f"Experiment context — {result['source']}", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Track", result["track_name"])
        c2.metric("Iterations", f"{result['n_iterations']:,}")
        c3.metric("Simulated pit loss", f"{base_pit:g}s")
        c4.metric("Mean SC laps (A / B)",
                  f"{result['sc_a'].mean():.1f} / {result['sc_b'].mean():.1f}")

    # --- Day 4: sensitivity slider drives every chart below (no re-sim) ---
    new_pit = st.slider("Pit-stop loss (seconds)", 20.0, 30.0, float(base_pit), 0.5)

    times_a = base_times_a + counts_a.astype(np.float64) * (new_pit - base_pit)
    times_b = base_times_b + counts_b.astype(np.float64) * (new_pit - base_pit)

    ci_a = stats.mean_ci(times_a)
    ci_b = stats.mean_ci(times_b)
    win_a = stats.win_probability(times_a, times_b)

    if abs(new_pit - base_pit) > 1e-9:
        st.info(f"Adjusted for pit-stop loss {new_pit:.1f}s — in-memory recompute from "
                f"stored pit counts, no physics re-run (Δ = {new_pit - base_pit:+.1f}s).")

    h1, h2, h3 = st.columns(3)
    h1.metric(f"{result['label_a']} mean", f"{ci_a['mean_s']:.2f}s",
              delta=f"±{ci_a['mean_s'] - ci_a['ci_low_s']:.2f}s")
    h2.metric(f"{result['label_b']} mean", f"{ci_b['mean_s']:.2f}s",
              delta=f"±{ci_b['mean_s'] - ci_b['ci_low_s']:.2f}s")
    h3.metric("Strategy A win probability", f"{win_a * 100:.1f}%",
              delta=f"{100 * (win_a - 0.5):+.1f}pp")

    # --- PDF overlay ---
    st.plotly_chart(pdf_figure(times_a, times_b, result["label_a"], result["label_b"]),
                    width="stretch")

    # --- Win-rate line + sensitivity heatmap ---
    left, right = st.columns(2)
    with left:
        st.plotly_chart(winrate_figure(base_times_a, base_times_b, counts_a, counts_b,
                                       base_pit, PIT_SWEEP), width="stretch")
    with right:
        st.plotly_chart(heatmap_figure(base_times_a, base_times_b, counts_a, counts_b,
                                       base_pit, PIT_SWEEP), width="stretch")

    # --- Lap traces (FR-15) ---
    traces_a = result["traces_a"]
    traces_b = result["traces_b"]
    if traces_a or traces_b:
        st.plotly_chart(trace_figure(traces_a, traces_b, result["label_a"],
                                     result["label_b"]), width="stretch")
    else:
        st.warning("No lap traces stored for this batch — run with sampling "
                   "(sim_index < 100) to see the lap-trace chart.")


def _saved_experiments_ui() -> None:
    """Sidebar block to load a stored experiment straight from SQLite (Day 2)."""
    st.sidebar.markdown("#### Saved experiments (SQLite)")
    if DB_EPHEMERAL:
        st.sidebar.caption(
            "⚠️ Repo `data/` is read-only on this host — storing to an **ephemeral "
            f"temp DB** (`{DB_PATH.parent.name}`). Batches last for this session."
        )
    try:
        with SimulationRepository(DB_PATH) as repo:
            repo.initialize()
            saved = repo.list_batches()
    except Exception:
        saved = []

    if saved:
        by_id = {b["batch_id"]: b for b in saved}
        labels = {
            b["batch_id"]: f"#{b['batch_id']} · {b['track']} · {b['n_iterations']:,} iters"
                           f" · seed {b['master_seed']} · {b['strategy_a']} vs {b['strategy_b']}"
            for b in saved
        }
        selected_id = st.sidebar.selectbox(
            "Stored experiments", list(by_id.keys()),
            format_func=lambda i: labels[i])
        if st.sidebar.button("Load selected from database"):
            with SimulationRepository(DB_PATH) as repo:
                repo.initialize()
                payload = repo.load_batch(selected_id)
            result = result_from_database(payload, by_id[selected_id],
                                          payload["pit_stop_loss_s"])
            st.session_state[RESULT_KEY] = result
    else:
        st.sidebar.caption("No experiments stored yet — run and tick *persist*, "
                           "or seed via `scripts/sprint3_populate_db.py`.")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
tracks = load_tracks()
compounds = load_compounds()
cars = load_cars()
car = cars[0]

st.sidebar.title("🏎️ Stochastic F1 Strategist")
mode = st.sidebar.radio(
    "Mode",
    ["Race comparison", "GA optimisation"],
    help="Race comparison runs/loads any two strategies. GA optimisation "
         "(Sprint 4 bonus) searches the optimal stint plan with a genetic "
         "algorithm, then races it head-to-head against your human baseline.",
)

track_name = st.sidebar.selectbox("Track", ["Monza", "Monaco"])
track = Track.from_name(track_name, tracks)

st.sidebar.caption(
    f"**{track.name}** — {track.total_laps} laps · base lap {track.base_lap_time_s:.0f}s · "
    f"SC p={track.sc_probability:.2f} · pit loss {track.pit_lane_loss_s:g}s")

if mode == "Race comparison":
    iterations = st.sidebar.number_input(
        "Monte Carlo iterations", min_value=1_000, max_value=500_000,
        value=int(min(DEFAULT_ITERATIONS, 50_000)), step=10_000)

    def strategy_editor(key: str) -> RaceStrategy:
        default = DEFAULT_STRATEGIES[track_name][key]
        st.sidebar.markdown(f"### Strategy {key}")
        desc = st.sidebar.text_input(
            f"Strategy {key} stints (Compound:laps, ...)", value=default)
        try:
            return RaceStrategy.from_description(
                key, desc, compounds, total_laps=track.total_laps)
        except ValueError as exc:
            st.sidebar.error(f"{exc} — using default {default!r}")
            return RaceStrategy.from_description(
                key, default, compounds, total_laps=track.total_laps)

    strategy_a = strategy_editor("A")
    strategy_b = strategy_editor("B")

    run_clicked = st.sidebar.button("Run Simulation", type="primary")
    persist = st.sidebar.checkbox(
        "Persist this experiment to results.db", value=False,
        help="Lets it appear under Saved experiments.")
else:
    # --- GA optimisation (Sprint 4 bonus) controls ---
    baseline_default = DEFAULT_STRATEGIES[track_name]["B"]
    st.sidebar.markdown("### Human baseline (vs GA)")
    baseline_desc = st.sidebar.text_input(
        "Baseline stints (Compound:laps, ...)", value=baseline_default)
    ga_population = st.sidebar.number_input(
        "GA population", min_value=10, max_value=120, value=40, step=10)
    ga_generations = st.sidebar.number_input(
        "GA generations", min_value=5, max_value=60, value=20, step=5)
    ga_fitness_runs = st.sidebar.number_input(
        "Fitness runs / evaluation", min_value=500, max_value=5_000,
        value=2_000, step=500)
    ga_h2h = st.sidebar.number_input(
        "Head-to-head iterations", min_value=1_000, max_value=100_000,
        value=20_000, step=10_000)
    ga_run_clicked = st.sidebar.button("Run Genetic Optimisation", type="primary")
    persist_ga = st.sidebar.checkbox(
        "Persist GA head-to-head to results.db", value=False,
        help="Lets the AI-vs-human batch appear under Saved experiments.")

st.sidebar.divider()
_saved_experiments_ui()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Monte Carlo F1 Race Strategist")

if mode == "Race comparison":
    if run_clicked:
        # Interactive sessions run on Streamlit's worker thread, where
        # multiprocessing (forkserver) pools cannot start reliably — including on
        # Streamlit Cloud. Use the vectorised sequential engine (Sprint 2): fast
        # at the default 50k. Heavy 100k seeding runs in parallel via
        # scripts/sprint3_populate_db.py / the CLI instead.
        engine = RaceEngine(track, car, pit_stop_loss_s=track.pit_lane_loss_s)
        runner = MonteCarloRunner(engine)
        with st.spinner(f"Running {2 * iterations:,} simulations (vectorised)..."):
            batch_a = runner.run(strategy_a, n_iterations=iterations,
                                 master_seed=DEFAULT_SEED, parallel=False)
            batch_b = runner.run(strategy_b, n_iterations=iterations,
                                 master_seed=DEFAULT_SEED, parallel=False)

        result = result_from_batches(
            batch_a, batch_b,
            f"{strategy_a.name} · {strategy_a.describe()}",
            f"{strategy_b.name} · {strategy_b.describe()}",
            track, track.pit_lane_loss_s, "simulated just now")
        if persist:
            try:
                batch_id, inserted = _persist(batch_a, batch_b, strategy_a,
                                              strategy_b, track, car, iterations)
                status = (f"Persisted to results.db as batch #{batch_id}"
                          if inserted else
                          f"Already stored — batch #{batch_id} (not duplicated)")
                result["source"] = status
                st.success(f"Simulation complete. {status}.")
            except Exception as exc:  # pragma: no cover - defensive DB guard
                st.warning(f"Simulation complete, but persisting failed: {exc}")
        else:
            st.success("Simulation complete. Results cached in session — the slider "
                       "below never re-runs physics.")
        st.session_state[RESULT_KEY] = result

    result = st.session_state.get(RESULT_KEY)
    if result is None:
        st.info("Configure two strategies in the sidebar and press **Run Simulation**, "
                "or pick a **Saved experiment** to render it straight from the database.")
        st.markdown("Race comparison renders the PDF overlay, win-rate sensitivity, "
                    "sensitivity heatmap and lap traces. The dashboard reads **only** "
                    "from in-memory/DB results while you interact (NFR-3); batches can "
                    "be persisted to `data/results.db`. Switch to *GA optimisation* for "
                    "the Sprint 4 genetic-algorithm mode.")
        st.stop()
    _render_result(result)
else:
    # -------------------- GA optimisation (Sprint 4 bonus) --------------------
    if ga_run_clicked:
        try:
            human = RaceStrategy.from_description(
                "human", baseline_desc, compounds, total_laps=track.total_laps)
        except ValueError as exc:
            st.error(f"Invalid baseline strategy: {exc}")
        else:
            engine = RaceEngine(track, car, pit_stop_loss_s=track.pit_lane_loss_s)
            runner = MonteCarloRunner(engine)
            config = GAConfig(population_size=int(ga_population),
                              generations=int(ga_generations),
                              fitness_runs=int(ga_fitness_runs),
                              master_seed=GA_SEED)
            optimiser = GeneticOptimiser(engine, compounds, config)
            progress = st.progress(0.0, text="Searching stint plans…")
            caption = st.empty()

            def _on_generation(generation: int, best: float, avg: float) -> None:
                progress.progress(
                    (generation + 1) / config.generations,
                    text=f"Generation {generation + 1}/{config.generations} — "
                         f"best {best:.2f}s",
                )
                if generation % 5 == 0 or generation == config.generations - 1:
                    caption.caption(
                        f"gen {generation + 1}: best {best:.2f}s · avg {avg:.2f}s")

            best_strategy, history = optimiser.run(progress_callback=_on_generation)
            caption.caption("Search complete — running the head-to-head validation…")
            ga_batch = runner.run(best_strategy, n_iterations=int(ga_h2h),
                                  master_seed=DEFAULT_SEED, parallel=False)
            human_batch = runner.run(human, n_iterations=int(ga_h2h),
                                     master_seed=DEFAULT_SEED, parallel=False)
            advantage_s = float(np.mean(human_batch.total_times) -
                                np.mean(ga_batch.total_times))
            source = (f"GA-optimal vs human baseline · {config.population_size} pop × "
                      f"{config.generations} gen · GA seed {GA_SEED} · "
                      f"head-to-head N={int(ga_h2h):,}")
            result = result_from_batches(
                ga_batch, human_batch,
                f"GA-optimal · {best_strategy.describe()}",
                f"human · {human.describe()}",
                track, track.pit_lane_loss_s, source)
            if persist_ga:
                try:
                    batch_id, inserted = _persist(
                        ga_batch, human_batch, best_strategy, human,
                        track, car, int(ga_h2h))
                    status = (f"Persisted to results.db as batch #{batch_id}"
                              if inserted else
                              f"Already stored — batch #{batch_id} (not duplicated)")
                    result["source"] += f" · {status}"
                    st.success(f"Optimisation complete. {status}.")
                except Exception as exc:  # pragma: no cover - defensive DB guard
                    st.warning(f"Optimisation complete, but persisting failed: {exc}")
            else:
                st.success("Optimisation complete — results cached in session "
                           "(persist them to see the batch under Saved experiments).")
            st.session_state[GA_KEY] = {
                "best_desc": best_strategy.describe(),
                "best_fitness_s": history.best_fitness[-1],
                "gen0_s": history.best_fitness[0],
                "advantage_s": advantage_s,
                "population": config.population_size,
                "generations": config.generations,
                "best": history.best_fitness,
                "avg": history.avg_fitness,
                "result": result,
            }

    ga = st.session_state.get(GA_KEY)
    if ga is None:
        st.info("Configure a human baseline and GA size in the sidebar, then press "
                "**Run Genetic Optimisation**.")
        st.markdown("Sprint 4 (bonus) — the hand-rolled GA (no DEAP) searches stint "
                    "lengths over Soft/Medium/Hard, minimising mean finishing time "
                    "(FR-17); convergence is charted (FR-18) and the winner is raced "
                    "head-to-head against your baseline (FR-19 / G6).")
        st.stop()

    st.subheader("🧬 GA-discovered strategy")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Discovered strategy", ga["best_desc"])
    m2.metric("Best fitness (mean time)", f"{ga['best_fitness_s']:.2f}s")
    m3.metric("GA advantage vs human", f"{ga['advantage_s']:+.2f}s")
    m4.metric("Search effort", f"{ga['population']} pop × {ga['generations']} gen")

    st.caption("GA searched a deterministic seed (NFR-6) — same settings reproduce the "
               "same optimum. The convergence chart below is FR-18; the head-to-head "
               "charts under it are the AI strategy vs your human baseline (FR-19).")
    st.plotly_chart(convergence_figure(ga["best"], ga["avg"]), width="stretch")
    _render_result(ga["result"])
