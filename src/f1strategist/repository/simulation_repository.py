"""SimulationRepository — SQLite context manager (write-once / read-many).

The database is the boundary between computation and presentation:
    - ``save_batch`` writes one experiment in a single transaction.
    - ``load_*`` returns stored results for the dashboard.

Schema is defined in ``schema.sql`` (repo root); an embedded fallback keeps the
repository importable even when the file is not on disk (e.g. pip installs).
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Optional

from f1strategist.output.lap_result import LapResult
from f1strategist.output.run_result import BatchResult, RunResult
from f1strategist.strategy.race_strategy import RaceStrategy

_SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schema.sql"

#: Embedded schema fallback (kept in sync with ``schema.sql``).
_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS track (
    id              INTEGER PRIMARY KEY,
    name            TEXT    NOT NULL UNIQUE,
    base_lap_time_s REAL    NOT NULL,
    total_laps      INTEGER NOT NULL CHECK (total_laps > 0),
    pit_lane_loss_s REAL    NOT NULL DEFAULT 22.0,
    sc_probability  REAL    NOT NULL CHECK (sc_probability BETWEEN 0 AND 1)
);

CREATE TABLE IF NOT EXISTS tyre_compound (
    id               INTEGER PRIMARY KEY,
    name             TEXT    NOT NULL UNIQUE,
    deg_coeff        REAL    NOT NULL CHECK (deg_coeff >= 0),
    cliff_threshold  INTEGER NOT NULL CHECK (cliff_threshold > 0),
    cliff_multiplier REAL    NOT NULL DEFAULT 3.0
);

CREATE TABLE IF NOT EXISTS car (
    id               INTEGER PRIMARY KEY,
    name             TEXT    NOT NULL UNIQUE,
    fuel_load_kg     REAL    NOT NULL CHECK (fuel_load_kg >= 0),
    fuel_burn_per_lap REAL   NOT NULL CHECK (fuel_burn_per_lap >= 0),
    fuel_time_effect REAL    NOT NULL DEFAULT 0.0,
    driver_sigma     REAL    NOT NULL CHECK (driver_sigma >= 0)
);

CREATE TABLE IF NOT EXISTS strategy (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    source      TEXT    NOT NULL DEFAULT 'user'
                        CHECK (source IN ('user', 'ga', 'system')),
    track_id    INTEGER NOT NULL REFERENCES track(id),
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (name, track_id, source)
);

CREATE TABLE IF NOT EXISTS strategy_stint (
    id               INTEGER PRIMARY KEY,
    strategy_id      INTEGER NOT NULL REFERENCES strategy(id) ON DELETE CASCADE,
    stint_index      INTEGER NOT NULL CHECK (stint_index >= 0),
    tyre_compound_id INTEGER NOT NULL REFERENCES tyre_compound(id),
    stint_laps       INTEGER NOT NULL CHECK (stint_laps > 0),
    UNIQUE (strategy_id, stint_index)
);

CREATE TABLE IF NOT EXISTS simulation_batch (
    id              INTEGER PRIMARY KEY,
    track_id        INTEGER NOT NULL REFERENCES track(id),
    car_id          INTEGER NOT NULL REFERENCES car(id),
    strategy_a_id   INTEGER NOT NULL REFERENCES strategy(id),
    strategy_b_id   INTEGER NOT NULL REFERENCES strategy(id),
    n_iterations    INTEGER NOT NULL DEFAULT 100000 CHECK (n_iterations > 0),
    pit_stop_loss_s REAL    NOT NULL DEFAULT 22.0,
    master_seed     INTEGER NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'running'
                            CHECK (status IN ('running', 'complete', 'failed')),
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    completed_at    TEXT
);

CREATE TABLE IF NOT EXISTS simulation_run (
    id             INTEGER PRIMARY KEY,
    batch_id       INTEGER NOT NULL REFERENCES simulation_batch(id) ON DELETE CASCADE,
    strategy_id    INTEGER NOT NULL REFERENCES strategy(id),
    sim_index      INTEGER NOT NULL CHECK (sim_index >= 0),
    total_time_s   REAL    NOT NULL,
    pit_stop_count INTEGER NOT NULL DEFAULT 0,
    sc_laps        INTEGER NOT NULL DEFAULT 0,
    UNIQUE (batch_id, strategy_id, sim_index)
);

CREATE INDEX IF NOT EXISTS idx_run_batch_strategy_time
    ON simulation_run (batch_id, strategy_id, total_time_s);

CREATE INDEX IF NOT EXISTS idx_run_batch ON simulation_run (batch_id);

CREATE TABLE IF NOT EXISTS lap_result (
    id                INTEGER PRIMARY KEY,
    run_id            INTEGER NOT NULL REFERENCES simulation_run(id) ON DELETE CASCADE,
    lap_number        INTEGER NOT NULL CHECK (lap_number >= 1),
    lap_time_s        REAL    NOT NULL,
    cumulative_time_s REAL    NOT NULL,
    tyre_age          INTEGER NOT NULL CHECK (tyre_age >= 0),
    fuel_remaining_kg REAL    NOT NULL,
    safety_car        INTEGER NOT NULL DEFAULT 0 CHECK (safety_car IN (0, 1)),
    stint_index       INTEGER NOT NULL CHECK (stint_index >= 0),
    UNIQUE (run_id, lap_number)
);

CREATE INDEX IF NOT EXISTS idx_lap_run ON lap_result (run_id);
"""


def _writable_dir(directory: Path) -> bool:
    """Probe a directory for real write access (mkdir + create + unlink)."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        fd, probe = tempfile.mkstemp(prefix=".f1strategist_probe_", dir=str(directory))
        os.close(fd)
        Path(probe).unlink()
        return True
    except OSError:
        return False


def resolve_db_path(preferred: str | Path | None = None, db_name: str = "results.db") -> Path:
    """Pick a writable SQLite location, falling back to the OS temp dir.

    Serverless runtimes (e.g. Streamlit Cloud) can mount the repo read-only, so
    the default ``<repo>/data/results.db`` may be unwritable. This resolver
    probes it and, if needed, returns ``<tempdir>/f1strategist/results.db`` — an
    ephemeral but fully writable location, so *persist* and *Saved experiments*
    keep working for the session.
    """
    preferred_path = Path(preferred) if preferred is not None else Path(
        Path(__file__).resolve().parents[3] / "data" / db_name
    )
    if _writable_dir(preferred_path.parent):
        return preferred_path
    fallback_dir = Path(tempfile.gettempdir()) / "f1strategist"
    if not _writable_dir(fallback_dir):
        raise RuntimeError(
            f"No writable directory for the results database (tried "
            f"{preferred_path.parent} and {fallback_dir})"
        )
    return fallback_dir / preferred_path.name


class SimulationRepository:
    """Context manager wrapping a ``sqlite3.Connection``."""

    def __init__(self, db_path: str | Path = "data/results.db") -> None:
        self.db_path = Path(db_path)
        self.connection: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------
    def __enter__(self) -> "SimulationRepository":
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.execute("PRAGMA foreign_keys = ON")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.connection is not None:
            if exc_type is None:
                self.connection.commit()
            else:
                self.connection.rollback()
            self.connection.close()
            self.connection = None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """Create tables/indexes if they do not exist yet."""
        sql = self._load_schema()
        self._conn().executescript(sql)

    @staticmethod
    def _load_schema() -> str:
        if _SCHEMA_PATH.exists():
            return _SCHEMA_PATH.read_text(encoding="utf-8")
        return _SCHEMA_SQL

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _conn(self) -> sqlite3.Connection:
        if self.connection is None:
            raise RuntimeError(
                "SimulationRepository must be used as a context manager"
            )
        return self.connection

    @staticmethod
    def _get_or_insert(conn: sqlite3.Connection, table: str, **values: Any) -> int:
        """Insert a row if absent; return its id either way."""
        # Lookup by the unique natural key (first column named 'name' or pair).
        key_cols = {"track": ("name",), "tyre_compound": ("name",), "car": ("name",)}
        key_cols.setdefault(table, tuple(values.keys()))
        cols = list(key_cols[table])
        where = " AND ".join(f"{c} = ?" for c in cols)
        row = conn.execute(
            f"SELECT id FROM {table} WHERE {where}", [values[c] for c in cols]
        ).fetchone()
        if row is not None:
            return int(row[0])
        placeholders = ", ".join("?" for _ in values)
        col_list = ", ".join(values.keys())
        cur = conn.execute(
            f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})",
            list(values.values()),
        )
        return int(cur.lastrowid)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_batch(
        self,
        strategy_a: RaceStrategy,
        strategy_b: RaceStrategy,
        track,
        car,
        n_iterations: int,
        master_seed: int,
        pit_stop_loss_s: float,
        batches: tuple[BatchResult, BatchResult],
    ) -> int:
        """Persist one head-to-head experiment in a single transaction.

        Returns the new ``simulation_batch.id``.
        """
        conn = self._conn()
        conn.execute("BEGIN")

        track_id = self._get_or_insert(
            conn, "track", name=track.name, base_lap_time_s=track.base_lap_time_s,
            total_laps=track.total_laps, pit_lane_loss_s=track.pit_lane_loss_s,
            sc_probability=track.sc_probability,
        )
        car_id = self._get_or_insert(
            conn, "car", name=car.name, fuel_load_kg=car.fuel_load_kg,
            fuel_burn_per_lap=car.fuel_burn_per_lap,
            fuel_time_effect=car.fuel_time_effect, driver_sigma=car.driver_sigma,
        )

        strat_a_id = self._upsert_strategy(conn, strategy_a, track_id)
        strat_b_id = self._upsert_strategy(conn, strategy_b, track_id)

        cur = conn.execute(
            """INSERT INTO simulation_batch
               (track_id, car_id, strategy_a_id, strategy_b_id,
                n_iterations, pit_stop_loss_s, master_seed, status, completed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'complete', datetime('now'))""",
            (track_id, car_id, strat_a_id, strat_b_id, n_iterations,
             pit_stop_loss_s, master_seed),
        )
        batch_id = int(cur.lastrowid)

        for result, strategy_id in ((batches[0], strat_a_id), (batches[1], strat_b_id)):
            run_rows = [
                (batch_id, strategy_id, r.sim_index, r.total_time_s,
                 r.pit_stop_count, r.sc_laps)
                for r in result.runs
            ]
            conn.executemany(
                """INSERT INTO simulation_run
                   (batch_id, strategy_id, sim_index, total_time_s,
                    pit_stop_count, sc_laps)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                run_rows,
            )

            # Lap traces: only for sampled runs (sim_index < 100).
            lap_rows = []
            for r in result.runs:
                if r.lap_trace is None:
                    continue
                run_id = conn.execute(
                    "SELECT id FROM simulation_run WHERE batch_id=? AND "
                    "strategy_id=? AND sim_index=?",
                    (batch_id, strategy_id, r.sim_index),
                ).fetchone()[0]
                lap_rows.extend(
                    (run_id, lap.lap_number, lap.lap_time_s, lap.cumulative_time_s,
                     lap.tyre_age, lap.fuel_remaining_kg, int(lap.safety_car),
                     lap.stint_index)
                    for lap in r.lap_trace
                )
            if lap_rows:
                conn.executemany(
                    """INSERT INTO lap_result
                       (run_id, lap_number, lap_time_s, cumulative_time_s,
                        tyre_age, fuel_remaining_kg, safety_car, stint_index)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    lap_rows,
                )

        conn.commit()
        return batch_id

    def _upsert_strategy(
        self, conn: sqlite3.Connection, strategy: RaceStrategy, track_id: int
    ) -> int:
        # Tyre compounds must exist before strategy_stint rows can reference them.
        compound_ids: dict[str, int] = {}
        for stint in strategy.stints:
            compound = stint.tyre_compound
            compound_ids[compound.name] = self._get_or_insert(
                conn, "tyre_compound", name=compound.name,
                deg_coeff=compound.deg_coeff,
                cliff_threshold=compound.cliff_threshold,
                cliff_multiplier=compound.cliff_multiplier,
            )

        row = conn.execute(
            "SELECT id FROM strategy WHERE name=? AND track_id=? AND source=?",
            (strategy.name, track_id, strategy.source),
        ).fetchone()
        if row is not None:
            return int(row[0])

        cur = conn.execute(
            "INSERT INTO strategy (name, source, track_id) VALUES (?, ?, ?)",
            (strategy.name, strategy.source, track_id),
        )
        strategy_id = int(cur.lastrowid)
        conn.executemany(
            """INSERT INTO strategy_stint
               (strategy_id, stint_index, tyre_compound_id, stint_laps)
               VALUES (?, ?, ?, ?)""",
            [
                (strategy_id, idx, compound_ids[stint.tyre_compound.name],
                 stint.stint_laps)
                for idx, stint in enumerate(strategy.stints)
            ],
        )
        return strategy_id

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def find_batch(
        self,
        track_name: str,
        n_iterations: int,
        master_seed: int,
        strategy_a_name: str,
        strategy_b_name: str,
    ) -> Optional[int]:
        """Return the id of an identical stored batch, or ``None``.

        Lets the dashboard / populate script persist an experiment
        idempotently instead of duplicating identical 100k-run batches.
        """
        row = self._conn().execute(
            """SELECT b.id
               FROM simulation_batch b
               JOIN track t ON t.id = b.track_id
               JOIN strategy sa ON sa.id = b.strategy_a_id
               JOIN strategy sb ON sb.id = b.strategy_b_id
               WHERE t.name = ? AND b.n_iterations = ? AND b.master_seed = ?
                 AND sa.name = ? AND sb.name = ?
               ORDER BY b.id LIMIT 1""",
            (track_name, n_iterations, master_seed,
             strategy_a_name, strategy_b_name),
        ).fetchone()
        return int(row[0]) if row else None

    def list_batches(self) -> list[dict[str, Any]]:
        """Summary rows for every stored batch (newest first)."""
        conn = self._conn()
        rows = conn.execute(
            """SELECT b.id, b.created_at, t.name AS track, b.n_iterations,
                      b.master_seed, b.status,
                      sa.name AS strategy_a, sb.name AS strategy_b
               FROM simulation_batch b
               JOIN track t ON t.id = b.track_id
               JOIN strategy sa ON sa.id = b.strategy_a_id
               JOIN strategy sb ON sb.id = b.strategy_b_id
               ORDER BY b.id DESC"""
        ).fetchall()
        return [
            {
                "batch_id": r[0], "created_at": r[1], "track": r[2],
                "n_iterations": r[3], "master_seed": r[4], "status": r[5],
                "strategy_a": r[6], "strategy_b": r[7],
            }
            for r in rows
        ]

    def load_runs(self, batch_id: int) -> dict[str, Any]:
        """Load a batch's metadata + runs for both strategies (without laps).

        The dashboard loads this once per batch, then all slider updates are
        in-memory NumPy operations (NFR-3).
        """
        conn = self._conn()
        batch = conn.execute(
            """SELECT b.id, b.n_iterations, b.pit_stop_loss_s, b.master_seed,
                      t.name, t.base_lap_time_s, t.total_laps, t.pit_lane_loss_s,
                      t.sc_probability, b.strategy_a_id, b.strategy_b_id
               FROM simulation_batch b JOIN track t ON t.id = b.track_id
               WHERE b.id = ?""",
            (batch_id,),
        ).fetchone()
        if batch is None:
            raise KeyError(f"No batch with id {batch_id}")

        runs_a = self._load_strategy_runs(conn, batch_id, batch[9])
        runs_b = self._load_strategy_runs(conn, batch_id, batch[10])
        return {
            "batch_id": batch[0],
            "n_iterations": batch[1],
            "pit_stop_loss_s": batch[2],
            "master_seed": batch[3],
            "track_name": batch[4],
            "strategy_a_id": batch[9],
            "strategy_b_id": batch[10],
            "runs_a": runs_a,
            "runs_b": runs_b,
        }

    @staticmethod
    def _load_strategy_runs(
        conn: sqlite3.Connection, batch_id: int, strategy_id: int
    ) -> tuple[RunResult, ...]:
        rows = conn.execute(
            """SELECT sim_index, total_time_s, pit_stop_count, sc_laps
               FROM simulation_run
               WHERE batch_id = ? AND strategy_id = ?
               ORDER BY sim_index""",
            (batch_id, strategy_id),
        ).fetchall()
        return tuple(
            RunResult(sim_index=r[0], total_time_s=r[1],
                      pit_stop_count=r[2], sc_laps=r[3])
            for r in rows
        )

    def load_lap_traces(
        self, batch_id: int, strategy_id: int
    ) -> list[tuple[int, tuple[LapResult, ...]]]:
        """Reconstruct sampled lap traces for one strategy in a batch.

        Only runs that were saved with full lap detail (the ``sim_index < 100``
        FR-15 sampling policy) come back. Returns ``[(sim_index, lap_trace), ...]``
        ordered by ``sim_index`` so the dashboard can draw the lap-trace chart
        from stored data without re-simulating.
        """
        conn = self._conn()
        rows = conn.execute(
            """SELECT sr.sim_index, lr.lap_number, lr.lap_time_s,
                      lr.cumulative_time_s, lr.tyre_age, lr.fuel_remaining_kg,
                      lr.safety_car, lr.stint_index
               FROM lap_result lr
               JOIN simulation_run sr ON sr.id = lr.run_id
               WHERE sr.batch_id = ? AND sr.strategy_id = ?
               ORDER BY sr.sim_index, lr.lap_number""",
            (batch_id, strategy_id),
        ).fetchall()
        traces: dict[int, list[LapResult]] = {}
        for r in rows:
            trace = traces.setdefault(int(r[0]), [])
            trace.append(
                LapResult(
                    lap_number=int(r[1]),
                    lap_time_s=float(r[2]),
                    cumulative_time_s=float(r[3]),
                    tyre_age=int(r[4]),
                    fuel_remaining_kg=float(r[5]),
                    safety_car=bool(r[6]),
                    stint_index=int(r[7]),
                )
            )
        return [(idx, tuple(laps)) for idx, laps in sorted(traces.items())]

    def _strategy_meta(self, conn: sqlite3.Connection, strategy_id: int) -> dict[str, Any]:
        """Name, source and stint description for a stored strategy."""
        row = conn.execute(
            "SELECT name, source FROM strategy WHERE id = ?", (strategy_id,)
        ).fetchone()
        stints = conn.execute(
            """SELECT tc.name, ss.stint_laps
               FROM strategy_stint ss
               JOIN tyre_compound tc ON tc.id = ss.tyre_compound_id
               WHERE ss.strategy_id = ?
               ORDER BY ss.stint_index""",
            (strategy_id,),
        ).fetchall()
        return {
            "name": row[0],
            "source": row[1],
            "desc": ",".join(f"{c}:{laps}" for c, laps in stints),
        }

    def load_batch(self, batch_id: int) -> dict[str, Any]:
        """One-stop dashboard load: metadata + runs + lap traces for a batch.

        Returns everything the UI needs to render from SQLite (the Sprint 3
        Day-2 read path) — totals, pit counts, SC laps, strategy descriptors
        and the sampled lap traces — so no physics re-run is required.
        """
        info = self.load_runs(batch_id)
        conn = self._conn()
        info["strategy_a_meta"] = self._strategy_meta(conn, info["strategy_a_id"])
        info["strategy_b_meta"] = self._strategy_meta(conn, info["strategy_b_id"])
        info["traces_a"] = self.load_lap_traces(batch_id, info["strategy_a_id"])
        info["traces_b"] = self.load_lap_traces(batch_id, info["strategy_b_id"])
        return info
