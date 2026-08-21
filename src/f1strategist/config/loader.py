"""Configuration loader — reads tracks / compounds / cars from JSON (NFR-12).

Adding a new track or tyre compound = a new JSON entry; no engine change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from f1strategist.config.tyre_compound import TyreCompound
from f1strategist.config.track import Track
from f1strategist.config.car import Car

#: Default config directory relative to the repository root.
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


def _read_config(filename: str, config_dir: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_dir) if config_dir is not None else DEFAULT_CONFIG_DIR
    if not path.is_absolute():
        path = Path.cwd() / path
    with (path / filename).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_tracks(config_dir: str | Path | None = None) -> list[Track]:
    """Load all tracks from ``tracks.json``."""
    data = _read_config("tracks.json", config_dir)
    return [Track(**entry) for entry in data["tracks"]]


def load_compounds(config_dir: str | Path | None = None) -> list[TyreCompound]:
    """Load all tyre compounds from ``compounds.json``."""
    data = _read_config("compounds.json", config_dir)
    return [TyreCompound(**entry) for entry in data["compounds"]]


def load_cars(config_dir: str | Path | None = None) -> list[Car]:
    """Load all cars from ``car.json``."""
    data = _read_config("car.json", config_dir)
    return [Car(**entry) for entry in data["cars"]]
