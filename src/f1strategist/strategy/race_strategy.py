"""RaceStrategy — an ordered list of stints with metadata.

Strategies are normalised (one ``Stint`` per row in the DB) so the Genetic
Optimiser can write novel strategies through exactly the same schema as
human-defined ones (FR-17).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from f1strategist.strategy.stint import Stint


@dataclass(frozen=True, slots=True)
class RaceStrategy:
    """A pit-stop plan: an ordered list of ``Stint`` objects.

    Attributes:
        name: Human-readable label, e.g. ``"1-stop"``.
        stints: Ordered stints that make up the whole race.
        source: Provenance — ``"user"``, ``"ga"`` or ``"system"``.
        total_laps: Race length this strategy is validated against. If set, the
            stint lengths must sum exactly to it (validated on construction).
    """

    name: str
    stints: tuple[Stint, ...] = field(default_factory=tuple)
    source: str = "user"
    total_laps: int | None = None

    def __post_init__(self) -> None:
        if not self.stints:
            raise ValueError("A strategy must contain at least one stint")
        if self.source not in ("user", "ga", "system"):
            raise ValueError(f"Unknown source {self.source!r}")
        if self.total_laps is not None:
            if self.total_laps <= 0:
                raise ValueError(f"total_laps must be > 0, got {self.total_laps}")
            if self.total_laps != sum(s.stint_laps for s in self.stints):
                raise ValueError(
                    f"Stints sum to {sum(s.stint_laps for s in self.stints)} laps, "
                    f"expected {self.total_laps}"
                )

    @property
    def stint_laps(self) -> tuple[int, ...]:
        """Tuple of lap counts, one per stint (GA chromosome encoding)."""
        return tuple(s.stint_laps for s in self.stints)

    @property
    def compound_names(self) -> tuple[str, ...]:
        """Tuple of compound names, one per stint."""
        return tuple(s.tyre_compound.name for s in self.stints)

    def describe(self) -> str:
        """Compact human-readable description, e.g. ``Soft:18,Medium:25,Hard:27``."""
        return ",".join(f"{s.tyre_compound.name}:{s.stint_laps}" for s in self.stints)

    @classmethod
    def from_description(
        cls,
        name: str,
        description: str,
        compounds: "list[object]",
        total_laps: int | None = None,
        source: str = "user",
    ) -> "RaceStrategy":
        """Build a strategy from a compact string like ``Soft:18,Medium:25,Hard:27``.

        Args:
            name: Strategy label.
            description: Comma-separated ``Compound:laps`` segments.
            compounds: Loaded list of ``TyreCompound`` objects to resolve names.
            total_laps: Optional race length to validate against.
            source: Provenance of the strategy.
        """
        by_name = {c.name.lower(): c for c in compounds}
        stints: list[Stint] = []
        for segment in description.replace(" ", "").split(","):
            compound_name, _, laps = segment.partition(":")
            if not compound_name or not laps:
                raise ValueError(f"Malformed stint segment {segment!r}")
            compound = by_name.get(compound_name.lower())
            if compound is None:
                raise ValueError(
                    f"Unknown compound {compound_name!r}; available: "
                    f"{[c.name for c in compounds]}"
                )
            stints.append(Stint(tyre_compound=compound, stint_laps=int(laps)))
        return cls(name=name, stints=tuple(stints), source=source, total_laps=total_laps)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"RaceStrategy(name={self.name!r}, stints=[{self.describe()}])"
