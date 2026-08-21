-- Stochastic F1 Race Strategist — canonical SQLite schema.
-- Canonical copy: see docs/database_schema.md for design rationale.
-- The SimulationRepository ships an embedded fallback kept in sync with this file.

PRAGMA foreign_keys = ON;

-- ── CONFIGURATION ENTITIES (small, static) ────────────────────────────
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

-- ── STRATEGY ENTITIES (normalised; GA writes here too) ────────────────
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

-- ── OUTPUT ENTITIES (large, bulk-generated) ───────────────────────────
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
    pit_stop_count INTEGER NOT NULL DEFAULT 0,   -- denormalised for sensitivity slider
    sc_laps        INTEGER NOT NULL DEFAULT 0,
    UNIQUE (batch_id, strategy_id, sim_index)
);

-- O(log n) win-probability / head-to-head queries (NFR-3, NFR-5)
CREATE INDEX IF NOT EXISTS idx_run_batch_strategy_time
    ON simulation_run (batch_id, strategy_id, total_time_s);

CREATE INDEX IF NOT EXISTS idx_run_batch ON simulation_run (batch_id);

-- Full per-lap detail, sampled only for sim_index < 100 (FR-15)
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
