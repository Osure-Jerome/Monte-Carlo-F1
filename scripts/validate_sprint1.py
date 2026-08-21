"""Sprint 1 validation script (blueprint Day 5 checkpoint).

Manual spot-checks:
    1. Does the lap time decrease by roughly ~0.05 s per lap as fuel burns off?
    2. Does the tyre cliff cause a ~2 s spike when a stint is over-extended?
    3. Does the deterministic engine reproduce the blueprint's degradation table?

Run:
    python scripts/validate_sprint1.py
"""

from __future__ import annotations

import numpy as np

from f1strategist.config.loader import load_tracks, load_compounds, load_cars
from f1strategist.config.track import Track
from f1strategist.engine.race_engine import RaceEngine
from f1strategist.strategy.race_strategy import RaceStrategy


def fuel_burn_check(engine: RaceEngine, laps: int) -> float:
    """Run with zero degradation / zero noise to isolate fuel burn alone."""
    fuel_per_lap = engine.car.fuel_burn_per_lap
    effect = engine.car.fuel_time_effect
    start = engine.car.fuel_load_kg

    # Total fuel penalty at lap i uses remaining fuel = start - burn * i.
    lap_times = []
    fuel = start
    for lap in range(1, laps + 1):
        lap_times.append(engine.car.fuel_time_penalty(fuel))
        fuel = max(0.0, fuel - fuel_per_lap)
    per_lap_delta = np.diff(lap_times).mean()
    expected_per_lap = -effect * fuel_per_lap
    print(f"  fuel: mean lap-to-lap delta = {per_lap_delta:.4f}s "
          f"(expected ≈ {expected_per_lap:.4f}s)")
    return float(per_lap_delta)


def cliff_check(soft, hard) -> float:
    """Compare degradation at threshold vs just beyond it."""
    spike = soft.degradation(soft.cliff_threshold + 1) - soft.degradation(soft.cliff_threshold)
    print(f"  cliff: Soft degradation jump across cliff = {spike:.3f}s")
    return float(spike)


def main() -> None:
    tracks = load_tracks()
    compounds = load_compounds()
    cars = load_cars()
    track = Track.from_name("Monza", tracks)
    car = cars[0]
    engine = RaceEngine(track, car)

    soft = next(c for c in compounds if c.name == "Soft")
    hard = next(c for c in compounds if c.name == "Hard")

    print("== Sprint 1 validation ==")
    print(f"Track: {track.name} ({track.total_laps} laps) | Car: {car.name}")

    print("\n[1] Fuel burn effect")
    fuel_burn_check(engine, track.total_laps)

    print("\n[2] Tyre cliff")
    cliff_check(soft, hard)

    print("\n[3] Blueprint degradation table (Soft: alpha=0.04, cliff=18)")
    for age in (5, 12, 18, 22, 25):
        print(f"  age {age:>2} -> {soft.degradation(age):7.2f}s")

    print("\n[4] Deterministic reproducibility (NFR-6)")
    strat_a = RaceStrategy.from_description(
        "A", "Soft:18,Medium:20,Hard:15", compounds, total_laps=track.total_laps
    )
    r1 = engine.simulate_race(strat_a, seed=4242)
    r2 = engine.simulate_race(strat_a, seed=4242)
    print(f"  same seed -> totals equal: {r1.total_time_s == r2.total_time_s} "
          f"({r1.total_time_s:.3f}s)")
    print(f"  pit stops: {r1.pit_stop_count}, SC laps: {r1.sc_laps}")

    print("\nAll spot-checks complete.")


if __name__ == "__main__":
    main()
