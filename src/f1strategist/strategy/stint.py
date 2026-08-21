"""Stint — a single tyre stint within a RaceStrategy."""

from __future__ import annotations

from dataclasses import dataclass

from f1strategist.config.tyre_compound import TyreCompound


@dataclass(frozen=True, slots=True)
class Stint:
    """One stint: run ``tyre_compound`` for ``stint_laps`` laps.

    Self-validating — a stint with a non-positive lap count is rejected at
    construction time.
    """

    tyre_compound: TyreCompound
    stint_laps: int

    def __post_init__(self) -> None:
        if self.stint_laps <= 0:
            raise ValueError(f"stint_laps must be > 0, got {self.stint_laps}")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Stint(tyre_compound={self.tyre_compound.name!r}, stint_laps={self.stint_laps})"
