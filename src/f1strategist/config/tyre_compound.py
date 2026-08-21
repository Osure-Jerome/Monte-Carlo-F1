"""TyreCompound — degradation parameters and the cliff model.

Implements the piecewise-quadratic degradation formula from the blueprint:

    delta_tyre(age) = alpha * age**2                         if age <= cliff_threshold
                    = alpha * t**2 + beta * (age - t)**2     otherwise

where ``t = cliff_threshold`` and ``beta = alpha * cliff_multiplier`` (default 3.0).

Adding a new tyre compound requires only a new config entry (NFR-12) — the engine
calls ``degradation(age)`` without ever switching on compound type.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TyreCompound:
    """A tyre compound with a non-linear, cliffed degradation curve.

    Attributes:
        name: Compound name, e.g. ``"Soft"``.
        deg_coeff: Quadratic degradation coefficient ``alpha`` (s / lap^2).
        cliff_threshold: Lap age at which degradation accelerates (``t``).
        cliff_multiplier: ``beta = deg_coeff * cliff_multiplier`` beyond the cliff.
    """

    name: str
    deg_coeff: float
    cliff_threshold: int
    cliff_multiplier: float = 3.0

    def __post_init__(self) -> None:
        if self.deg_coeff < 0:
            raise ValueError(f"deg_coeff must be >= 0, got {self.deg_coeff}")
        if self.cliff_threshold <= 0:
            raise ValueError(f"cliff_threshold must be > 0, got {self.cliff_threshold}")
        if self.cliff_multiplier <= 0:
            raise ValueError(f"cliff_multiplier must be > 0, got {self.cliff_multiplier}")

    def degradation(self, age: int) -> float:
        """Lap-time penalty (seconds) for a tyre at a given age (in laps)."""
        if age <= 0:
            return 0.0
        if age <= self.cliff_threshold:
            return self.deg_coeff * age * age
        base = self.deg_coeff * self.cliff_threshold**2
        beta = self.deg_coeff * self.cliff_multiplier
        return base + beta * (age - self.cliff_threshold) ** 2

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"TyreCompound(name={self.name!r}, deg_coeff={self.deg_coeff}, "
            f"cliff_threshold={self.cliff_threshold}, cliff_multiplier={self.cliff_multiplier})"
        )
