"""Stochastic F1 Race Strategist — Streamlit dashboard.

Reads ONLY from SQLite via ``SimulationRepository``; it never triggers physics
computation on a widget callback (NFR-3, NFR-8). Slider updates are in-memory
NumPy recomputations over pre-loaded arrays.

Run with:
    streamlit run dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - optional dependency in lightweight runtimes
    go = None

from f1strategist.config.loader import load_tracks, load_compounds, load_cars
from f1strategist.config.track import Track
from f1strategist.engine.race_engine import RaceEngine
from f1strategist.engine.montecarlo import MonteCarloRunner
from f1strategist.output.run_result import BatchResult
from f1strategist.repository.simulation_repository import SimulationRepository
from f1strategist.statistics.statistics_engine import StatisticsEngine
from f1strategist.strategy.race_strategy import RaceStrategy

st.set_page_config(page_title="Stochastic F1 Race Strategist", layout="wide")

# ---------------------------------------------------------------------------
# Session state — the batch is loaded ONCE; slider updates never touch the DB.
# ---------------------------------------------------------------------------
DEFAULT_ITERATIONS = 100_000
DEFAULT_SEED = 42  # NFR-6: one master seed for BOTH strategies (paired runs)
DB_PATH = Path(__file__).parent / "data" / "results.db"

#: Track-specific defaults so a strategy always sums to the race length.
DEFAULT_STRATEGIES = {
    "Monza": {"A": "Soft:18,Medium:20,Hard:15", "B": "Medium:35,Hard:18"},
    "Monaco": {"A": "Soft:18,Medium:25,Hard:35", "B": "Medium:40,Hard:38"},
}


# ---------------------------------------------------------------------------
# Sidebar: configuration
# ---------------------------------------------------------------------------
st.sidebar.title("🏎️ Stochastic F1 Strategist")
track_name = st.sidebar.selectbox("Track", ["Monza", "Monaco"])
iterations = st.sidebar.number_input(
    "Monte Carlo iterations", min_value=1_000, max_value=500_000,
    value=min(DEFAULT_ITERATIONS, 50_000), step=10_000,
)

tracks = load_tracks()
compounds = load_compounds()
cars = load_cars()
track = Track.from_name(track_name, tracks)
car = cars[0]

st.sidebar.markdown(
    f"**{track.name}** — {track.total_laps} laps · base lap "
    f"{track.base_lap_time_s:.0f}s · SC p={track.sc_probability:.2f}"
)

def strategy_editor(label: str) -> RaceStrategy:
    default = DEFAULT_STRATEGIES[track_name][label]
    st.sidebar.markdown(f"### {label}")
    desc = st.sidebar.text_input(f"{label} stints (Compound:laps, ...)", value=default)
    try:
        return RaceStrategy.from_description(
            label, desc, compounds, total_laps=track.total_laps
        )
    except ValueError as exc:
        st.sidebar.error(f"{exc} — using default {default!r}")
        return RaceStrategy.from_description(
            label, default, compounds, total_laps=track.total_laps
        )

strategy_a = strategy_editor("Strategy A")
strategy_b = strategy_editor("Strategy B")

st.sidebar.markdown(
    f"A: `{strategy_a.describe()}` · B: `{strategy_b.describe()}`"
)

run_clicked = st.sidebar.button("Run Simulation", type="primary")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Monte Carlo F1 Race Strategist")

if run_clicked:
    with st.spinner(f"Running {2 * iterations:,} simulations..."):
        engine = RaceEngine(track, car, pit_stop_loss_s=track.pit_lane_loss_s)
        runner = MonteCarloRunner(engine)
        batch_a = runner.run(
            strategy_a, n_iterations=iterations, master_seed=DEFAULT_SEED, parallel=True
        )
        batch_b = runner.run(
            strategy_b, n_iterations=iterations, master_seed=DEFAULT_SEED, parallel=True
        )
        st.session_state["batch_a"] = batch_a
        st.session_state["batch_b"] = batch_b
        st.session_state["master_seed"] = DEFAULT_SEED
        st.session_state["track_name"] = track.name
        st.session_state["iterations"] = iterations
    st.success("Simulation complete. Results cached in session — the slider below never re-runs physics.")

if "batch_a" in st.session_state:
    stats = StatisticsEngine()
    batch_a: BatchResult = st.session_state["batch_a"]
    batch_b: BatchResult = st.session_state["batch_b"]
    times_a = batch_a.total_times
    times_b = batch_b.total_times

    ci_a, ci_b = stats.mean_ci(times_a), stats.mean_ci(times_b)
    win_a = stats.win_probability(times_a, times_b)

    # --- Head-to-head card ---
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{strategy_a.describe()} mean", f"{ci_a['mean_s']:.2f}s",
              delta=f"±{ci_a['mean_s'] - ci_a['ci_low_s']:.2f}s")
    c2.metric(f"{strategy_b.describe()} mean", f"{ci_b['mean_s']:.2f}s",
              delta=f"±{ci_b['mean_s'] - ci_b['ci_low_s']:.2f}s")
    c3.metric("Strategy A win probability", f"{win_a * 100:.1f}%",
              delta=f"±{100 * abs(win_a - 0.5):.1f}pp")

    # --- Sensitivity slider (FR-14 / NFR-3): no re-simulation ---
    old_pit = track.pit_lane_loss_s
    new_pit = st.slider("Pit-stop loss (seconds)", 20.0, 30.0, float(old_pit), 0.5)
    if abs(new_pit - old_pit) > 1e-9:
        adj_a = stats.sensitivity(times_a, batch_a.pit_stop_counts, old_pit, new_pit)
        adj_b = stats.sensitivity(times_b, batch_b.pit_stop_counts, old_pit, new_pit)
        win_a = stats.win_probability(adj_a, adj_b)
        times_a, times_b = adj_a, adj_b
        st.info(f"Adjusted for pit-stop loss {new_pit:.1f}s (in-memory, no re-simulation).")

    # --- PDF overlay (FR-13) ---
    x_a, dens_a = stats.kde(times_a)
    x_b, dens_b = stats.kde(times_b)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=times_a, histnorm="probability density",
                               name=f"A {strategy_a.describe()}", opacity=0.55))
    fig.add_trace(go.Scatter(x=x_a, y=dens_a, mode="lines",
                             name="A KDE", line=dict(width=2)))
    fig.add_trace(go.Histogram(x=times_b, histnorm="probability density",
                               name=f"B {strategy_b.describe()}", opacity=0.55))
    fig.add_trace(go.Scatter(x=x_b, y=dens_b, mode="lines",
                             name="B KDE", line=dict(width=2)))
    fig.update_layout(title="Finishing-time distributions (PDF overlay)",
                      xaxis_title="Total race time (s)", yaxis_title="Density",
                      barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)

    # --- Lap trace (FR-15): first sampled run of each strategy ---
    traces_a = batch_a.sample_traces()
    traces_b = batch_b.sample_traces()
    if traces_a and traces_b:
        trace_fig = go.Figure()
        for trace, label in ((traces_a[0], f"A {strategy_a.describe()}"),
                             (traces_b[0], f"B {strategy_b.describe()}")):
            laps = [lap.lap_number for lap in trace]
            times = [lap.lap_time_s for lap in trace]
            trace_fig.add_trace(go.Scatter(x=laps, y=times, mode="lines", name=label))
        trace_fig.update_layout(title="Representative lap trace (lap time vs lap)",
                                xaxis_title="Lap", yaxis_title="Lap time (s)")
        st.plotly_chart(trace_fig, use_container_width=True)
    else:
        st.warning("No lap traces stored — run with default sampling (first 100 runs) to see traces.")

    # --- Persist to SQLite (write-once per experiment) ---
    if st.button("Save batch to database"):
        seed = st.session_state.get("master_seed", DEFAULT_SEED)
        with SimulationRepository(DB_PATH) as repo:
            repo.initialize()
            batch_id = repo.save_batch(
                strategy_a, strategy_b, track, car,
                n_iterations=int(len(times_a)), master_seed=seed,
                pit_stop_loss_s=float(old_pit),
                batches=(batch_a, batch_b),
            )
            st.success(f"Saved batch #{batch_id} to {DB_PATH}.")
else:
    st.info("Configure two strategies in the sidebar and press **Run Simulation**.")
    st.markdown(
        "This dashboard reads **only** from in-memory results while you interact. "
        "Batches can be persisted to SQLite (`data/results.db`) for later re-querying."
    )
