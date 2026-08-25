"""f1strategist.cli — head-to-head comparison from the terminal.

Example:
    python -m f1strategist.cli --track Monaco \
        --strategy-a "Soft:18,Medium:25,Hard:35" \
        --strategy-b "Medium:40,Hard:38" \
        --iterations 10000 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import sys

from f1strategist.config.loader import load_tracks, load_compounds, load_cars
from f1strategist.config.track import Track
from f1strategist.engine.race_engine import RaceEngine
from f1strategist.engine.montecarlo import MonteCarloRunner
from f1strategist.statistics.statistics_engine import StatisticsEngine
from f1strategist.strategy.race_strategy import RaceStrategy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="f1strategist",
        description="Monte Carlo F1 pit-stop strategy comparison",
    )
    parser.add_argument("--track", default="Monaco", help="Track name (default: Monaco)")
    parser.add_argument(
        "--strategy-a",
        default="Soft:18,Medium:25,Hard:35",
        help="Strategy A as Compound:laps,Compound:laps,... (default matches Monaco's 78 laps)",
    )
    parser.add_argument(
        "--strategy-b",
        default="Medium:40,Hard:38",
        help="Strategy B as Compound:laps,Compound:laps,... (default matches Monaco's 78 laps)",
    )
    parser.add_argument("--iterations", type=int, default=10_000, help="Monte Carlo runs per strategy")
    parser.add_argument("--seed", type=int, default=42, help="Master seed (NFR-6 reproducibility)")
    parser.add_argument("--pit-loss", type=float, default=None, help="Pit-stop loss in seconds")
    parser.add_argument(
        "--sc-duration", type=int, default=3,
        help="Safety Car state-machine duration in laps (Sprint 2, default 3)",
    )
    parser.add_argument(
        "--sc-delta", type=float, default=5.0,
        help="Per-lap time delta (s) under the Safety Car (Sprint 2, default 5.0)",
    )
    parser.add_argument("--parallel", action="store_true", help="Use multiprocessing")
    parser.add_argument("--json", action="store_true", help="Emit results as JSON")
    parser.add_argument(
        "--csv", metavar="PATH", default=None,
        help="Write per-run results to a CSV file (Sprint 2 Day 5 deliverable)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    tracks = load_tracks()
    compounds = load_compounds()
    cars = load_cars()
    track = Track.from_name(args.track, tracks)
    car = cars[0]
    pit_loss = args.pit_loss if args.pit_loss is not None else track.pit_lane_loss_s

    try:
        strategy_a = RaceStrategy.from_description(
            "A", args.strategy_a, compounds, total_laps=track.total_laps
        )
        strategy_b = RaceStrategy.from_description(
            "B", args.strategy_b, compounds, total_laps=track.total_laps
        )
    except ValueError as exc:
        print(f"error: invalid strategy: {exc}", file=sys.stderr)
        return 2

    engine = RaceEngine(
        track, car,
        pit_stop_loss_s=pit_loss,
        sc_duration_laps=args.sc_duration,
        sc_delta_s=args.sc_delta,
    )
    runner = MonteCarloRunner(engine)
    batch_a = runner.run(
        strategy_a, n_iterations=args.iterations, master_seed=args.seed,
        parallel=args.parallel,
    )
    batch_b = runner.run(
        strategy_b, n_iterations=args.iterations, master_seed=args.seed,
        parallel=args.parallel,
    )

    stats = StatisticsEngine()
    times_a, times_b = batch_a.total_times, batch_b.total_times
    ci_a, ci_b = stats.mean_ci(times_a), stats.mean_ci(times_b)
    win_a = stats.win_probability(times_a, times_b)

    if args.json:
        payload = {
            "track": track.name,
            "strategy_a": strategy_a.describe(),
            "strategy_b": strategy_b.describe(),
            "iterations": args.iterations,
            "seed": args.seed,
            "pit_stop_loss_s": pit_loss,
            "stats_a": {
                "mean_s": ci_a["mean_s"],
                "ci_low_s": ci_a["ci_low_s"],
                "ci_high_s": ci_a["ci_high_s"],
            },
            "stats_b": {
                "mean_s": ci_b["mean_s"],
                "ci_low_s": ci_b["ci_low_s"],
                "ci_high_s": ci_b["ci_high_s"],
            },
            "win_probability_a": win_a,
            "sc_model": {
                "duration_laps": engine.sc_duration_laps,
                "delta_s": engine.sc_delta_s,
            },
        }
        print(json.dumps(payload, indent=2))
        if args.csv:
            _write_csv(args.csv, batch_a, batch_b)
        return 0

    print(f"Track: {track.name} ({track.total_laps} laps) | Pit loss: {pit_loss}s")
    print(f"SC model: {engine.sc_duration_laps}-lap state machine, +{engine.sc_delta_s:g}s/lap")
    print(f"  {strategy_a.name}  {strategy_a.describe()}  "
          f"mean={ci_a['mean_s']:.3f}s  CI=[{ci_a['ci_low_s']:.3f}, {ci_a['ci_high_s']:.3f}]")
    print(f"  {strategy_b.name}  {strategy_b.describe()}  "
          f"mean={ci_b['mean_s']:.3f}s  CI=[{ci_b['ci_low_s']:.3f}, {ci_b['ci_high_s']:.3f}]")
    print(f"Strategy A wins {win_a * 100:.1f}% ± {100 * abs(win_a - 0.5):.1f}pp (paired, N={args.iterations})")
    if args.csv:
        _write_csv(args.csv, batch_a, batch_b)
    return 0


def _write_csv(path: str, batch_a, batch_b) -> None:
    """Write per-run totals for both strategies to ``path`` (Sprint 2 Day 5)."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sim_index", "strategy", "total_time_s", "pit_stop_count", "sc_laps"])
        for batch in (batch_a, batch_b):
            for run in batch.runs:
                writer.writerow(
                    [run.sim_index, batch.strategy.name, run.total_time_s,
                     run.pit_stop_count, run.sc_laps]
                )
    print(f"wrote {len(batch_a.runs) + len(batch_b.runs)} rows to {path}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
