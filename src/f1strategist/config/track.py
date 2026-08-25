"""Track — circuit characteristics, including a track-specific Safety Car rate."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Track:
    """A circuit definition.

    Attributes:
        name: Circuit name, e.g. ``"Monza"``.
        base_lap_time_s: Lap time (s) with no degradation / fuel / noise.
        total_laps: Race distance in laps.
        pit_lane_loss_s: Fixed pit-stop time penalty (s).
        sc_probability: Per-lap Bernoulli probability of a Safety Car trigger.
            (The SC state machine — duration and per-lap delta — is configured
            on the ``RaceEngine`` per Sprint 2.)
    """

    name: str
    base_lap_time_s: float
    total_laps: int
    pit_lane_loss_s: float = 22.0
    sc_probability: float = 0.12

    def __post_init__(self) -> None:
        if self.base_lap_time_s <= 0:
            raise ValueError(f"base_lap_time_s must be > 0, got {self.base_lap_time_s}")
        if self.total_laps <= 0:
            raise ValueError(f"total_laps must be > 0, got {self.total_laps}")
        if not 0.0 <= self.sc_probability <= 1.0:
            raise ValueError(f"sc_probability must be in [0, 1], got {self.sc_probability}")

    @classmethod
    def from_name(cls, name: str, tracks: "list[Track]") -> "Track":
        """Factory helper: look up a track by name from a loaded config list."""
        for track in tracks:
            if track.name.lower() == name.lower():
                return track
        raise KeyError(f"Unknown track {name!r}; available: {[t.name for t in tracks]}")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Track(name={self.name!r}, base_lap_time_s={self.base_lap_time_s}, "
            f"total_laps={self.total_laps}, pit_lane_loss_s={self.pit_lane_loss_s}, "
            f"sc_probability={self.sc_probability})"
        )
