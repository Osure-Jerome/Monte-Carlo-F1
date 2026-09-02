"""Stochastic F1 Race Strategist — Sprint 3 interactive dashboard.

Flow (NFR-3 / NFR-8 — the engine is never touched during UI interaction):
  1. "Run Simulation" computes one batch per strategy in memory; optionally
     persists it to SQLite via ``SimulationRepository`` (Day-2 path).
  2. "Saved experiments" loads a stored batch *from the database* — proving the
     Day-2 read path: no physics re-run, charts render from stored rows.
  3. Every slider / widget change is an in-memory NumPy recomputation over the
     stored ``pit_stop_count`` columns (Day-4 sensitivity) — no re-simulation.

Charts (success metric: >= 3 chart types):
  - PDF overlay          : overlapping density histograms + gaussian_kde (Day 3)
  - Win-rate sensitivity : P(A beats B) vs shared pit-stop loss (line)
  - Sensitivity heatmap  : P(A beats B) over asymmetric (loss_A x loss_B)
  - Lap-trace plot       : representative lap time trace (FR-15)

Run with:
    streamlit run dashboard.py
"""

from __future__ import annotations

from pathlib import Path

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
from f1strategist.output.lap_result import LapResult
from f1strategist.output.run_result import BatchResult
from f1strategist.repository.simulation_repository import SimulationRepository
from f1strategist.statistics.statistics_engine import StatisticsEngine
from f1strategist.strategy.race_strategy import RaceStrategy

st.set_page_config(page_title="Stochastic F1 Race Strategist", layout="wide")

DEFAULT_ITERATIONS = 50_000
DEFAULT_SEED = 42  # NFR-6: one master seed for BOTH strategies (paired runs)
DB_PATH = Path(__file__).resolve().parent / "data" / "results.db"
RESULT_KEY = "s3_result"

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


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🏎️ Stochastic F1 Strategist")

track_name = st.sidebar.selectbox("Track", ["Monza", "Monaco"])
iterations = st.sidebar.number_input(
    "Monte Carlo iterations", min_value=1_000, max_value=500_000,
    value=int(min(DEFAULT_ITERATIONS, 50_000)), step=10_000)

tracks = load_tracks()
compounds = load_compounds()
cars = load_cars()
track = Track.from_name(track_name, tracks)
car = cars[0]

st.sidebar.caption(
    f"**{track.name}** — {track.total_laps} laps · base lap {track.base_lap_time_s:.0f}s · "
    f"SC p={track.sc_probability:.2f} · pit loss {track.pit_lane_loss_s:g}s")


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
persist = st.sidebar.checkbox("Persist this experiment to results.db",
                              value=False, help="Lets it appear under Saved experiments.")
st.sidebar.divider()

# --- Saved experiments (Day 2: read path from SQLite) ---
st.sidebar.markdown("#### Saved experiments (SQLite)")
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
# Main area
# ---------------------------------------------------------------------------
st.title("Monte Carlo F1 Race Strategist")

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
    st.markdown("Sprint 3 — charts appear here: PDF overlay, win-rate sensitivity, "
                "sensitivity heatmap and lap traces. The dashboard reads **only** "
                "from in-memory/DB results while you interact (NFR-3); batches can "
                "be persisted to `data/results.db`.")
    st.stop()

# ---------------------------------------------------------------------------
# Results rendering (pure in-memory recomputation — never touches the engine)
# ---------------------------------------------------------------------------
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

# --- Day 4: sensitivity slider drives every chart below (no re-simulation) ---
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

# --- Day 3: PDF overlay ---
st.plotly_chart(pdf_figure(times_a, times_b, result["label_a"], result["label_b"]),
                width="stretch")

# --- Day 4: win-rate line + sensitivity heatmap (3rd & 4th chart types) ---
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
