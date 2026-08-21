"""Car — physics constants for fuel burn and driver inconsistency."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Car:
    """A car / driver configuration.

    Attributes:
        name: Identifier, e.g. ``"default"``.
        fuel_load_kg: Fuel on board at race start (kg).
        fuel_burn_per_lap: Fuel consumed per lap (kg).
        fuel_time_effect: Lap-time penalty per kg of fuel remaining (s / kg).
        driver_sigma: Standard deviation of per-lap Gaussian driver noise (s).
    """

    name: str
    fuel_load_kg: float
    fuel_burn_per_lap: float
    fuel_time_effect: float = 0.03
    driver_sigma: float = 0.15

    def __post_init__(self) -> None:
        if self.fuel_load_kg < 0:
            raise ValueError(f"fuel_load_kg must be >= 0, got {self.fuel_load_kg}")
        if self.fuel_burn_per_lap <= 0:
            raise ValueError(f"fuel_burn_per_lap must be > 0, got {self.fuel_burn_per_lap}")
        if self.fuel_time_effect < 0:
            raise ValueError(f"fuel_time_effect must be >= 0, got {self.fuel_time_effect}")
        if self.driver_sigma < 0:
            raise ValueError(f"driver_sigma must be >= 0, got {self.driver_sigma}")

    def fuel_time_penalty(self, fuel_remaining_kg: float) -> float:
        """Lap-time penalty (s) for carrying ``fuel_remaining_kg`` of fuel.

        Heavier car = slower lap. Fuel burns off each lap, so the penalty
        decreases through the race (FR-3).
        """
        return max(0.0, self.fuel_time_effect * fuel_remaining_kg)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Car(name={self.name!r}, fuel_load_kg={self.fuel_load_kg}, "
            f"fuel_burn_per_lap={self.fuel_burn_per_lap}, "
            f"fuel_time_effect={self.fuel_time_effect}, driver_sigma={self.driver_sigma})"
        )
