# Database Layout — Stochastic F1 Race Strategist

> **Source:** Phases 1-2 Blueprint.docx · **Status:** Approved for implementation
> **Engine:** SQLite (file-based, zero-config) · **DB file:** `data/results.db`

---

## 1. Design Principles

1. **Separate static config from bulk output.** Configuration entities (tracks, compounds, cars, strategies) are small and read frequently; output entities (runs, laps) are large and generated in bulk. They live in separate tables so hot output tables are never locked behind config reads.
2. **Write-once, read-many.** A batch is written once in a single transaction; all subsequent dashboard interaction is read-only or in-memory NumPy recomputation. This satisfies NFR-3 (sub-second slider) and NFR-5 (millions of rows without degraded queries).
3. **Denormalise only where it buys interactive performance.** `pit_stop_count` is copied onto every run so the sensitivity slider (FR-14) never touches `lap_result`.
4. **Bound storage on per-lap detail.** Full lap traces are stored only for the first 100 runs per batch (the lap-chart sample), capping `lap_result` at ~7,000 rows/batch.
5. **Normalise strategies.** `strategy` + `strategy_stint` relational rows mean the Genetic Optimiser writes novel strategies through exactly the same schema as human-defined ones (FR-17).

---

## 2. Entity-Relationship Overview

```
track 1 ────< simulation_batch >──── 1 car
                │      │
                │      └──────────── 1 strategy (strategy_a)
                │      └──────────── 1 strategy (strategy_b)
                │
simulation_batch 1 ────< simulation_run >──── 1 strategy
                              │
                              └──── 1 ────< lap_result >  (sim_index < 100 only)

strategy 1 ────< strategy_stint >──── 1 tyre_compound
```

Key relationships:
- **simulation_batch** — one experiment: a track × car × two strategies × N iterations.
- **simulation_run** — one race within a batch; carries `total_time_s` + denormalised `pit_stop_count`.
- **lap_result** — per-lap detail, sampled (`sim_index < 100`), linked to one run.
- **strategy_stint** — ordered stints; links a strategy to tyre compounds and lap counts.

---

## 3. Full SQL Schema

```sql
PRAGMA foreign_keys = ON;

-- ──────────────────────────────────────────────────────────────
-- CONFIGURATION ENTITIES (small, static, seeded from JSON)
-- ──────────────────────────────────────────────────────────────

CREATE TABLE track (
    id              INTEGER PRIMARY KEY,
    name            TEXT    NOT NULL UNIQUE,          -- 'Monza', 'Monaco'
    base_lap_time_s REAL    NOT NULL,                 -- seconds, no-degradation lap
    total_laps      INTEGER NOT NULL CHECK (total_laps > 0),   -- e.g. 70
    pit_lane_loss_s REAL    NOT NULL DEFAULT 22.0,    -- fixed pit delta
    sc_probability  REAL    NOT NULL CHECK (sc_probability BETWEEN 0 AND 1)
);

CREATE TABLE tyre_compound (
    id               INTEGER PRIMARY KEY,
    name             TEXT    NOT NULL UNIQUE,         -- 'Soft' | 'Medium' | 'Hard'
    deg_coeff        REAL    NOT NULL CHECK (deg_coeff >= 0),  -- alpha
    cliff_threshold  INTEGER NOT NULL CHECK (cliff_threshold > 0), -- laps
    cliff_multiplier REAL    NOT NULL DEFAULT 3.0     -- beta = alpha * multiplier
);

CREATE TABLE car (
    id               INTEGER PRIMARY KEY,
    name             TEXT    NOT NULL UNIQUE,         -- e.g. 'default'
    fuel_load_kg     REAL    NOT NULL CHECK (fuel_load_kg >= 0),
    fuel_burn_per_lap REAL   NOT NULL CHECK (fuel_burn_per_lap >= 0),
    fuel_time_effect REAL    NOT NULL DEFAULT 0.0,    -- seconds per kg
    driver_sigma     REAL    NOT NULL CHECK (driver_sigma >= 0)  -- Gaussian noise
);

-- ──────────────────────────────────────────────────────────────
-- STRATEGY ENTITIES (normalised; GA writes here too)
-- ──────────────────────────────────────────────────────────────

CREATE TABLE strategy (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    source      TEXT    NOT NULL DEFAULT 'user'
                        CHECK (source IN ('user', 'ga', 'system')),
    track_id    INTEGER NOT NULL REFERENCES track(id),
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (name, track_id, source)
);

CREATE TABLE strategy_stint (
    id               INTEGER PRIMARY KEY,
    strategy_id      INTEGER NOT NULL REFERENCES strategy(id) ON DELETE CASCADE,
    stint_index      INTEGER NOT NULL CHECK (stint_index >= 0),   -- 0-based order
    tyre_compound_id INTEGER NOT NULL REFERENCES tyre_compound(id),
    stint_laps       INTEGER NOT NULL CHECK (stint_laps > 0),
    UNIQUE (strategy_id, stint_index)
);

-- ──────────────────────────────────────────────────────────────
-- OUTPUT ENTITIES (large, bulk-generated)
-- ──────────────────────────────────────────────────────────────

CREATE TABLE simulation_batch (
    id              INTEGER PRIMARY KEY,
    track_id        INTEGER NOT NULL REFERENCES track(id),
    car_id          INTEGER NOT NULL REFERENCES car(id),
    strategy_a_id   INTEGER NOT NULL REFERENCES strategy(id),
    strategy_b_id   INTEGER NOT NULL REFERENCES strategy(id),
    n_iterations    INTEGER NOT NULL DEFAULT 100000 CHECK (n_iterations > 0),
    pit_stop_loss_s REAL    NOT NULL DEFAULT 22.0,     -- snapshot of input param
    master_seed     INTEGER NOT NULL,                  -- reproducibility (NFR-6)
    status          TEXT    NOT NULL DEFAULT 'running'
                            CHECK (status IN ('running', 'complete', 'failed')),
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    completed_at    TEXT
);

CREATE TABLE simulation_run (
    id             INTEGER PRIMARY KEY,
    batch_id       INTEGER NOT NULL REFERENCES simulation_batch(id) ON DELETE CASCADE,
    strategy_id    INTEGER NOT NULL REFERENCES strategy(id),
    sim_index      INTEGER NOT NULL CHECK (sim_index >= 0),   -- 0..N-1 in batch
    total_time_s   REAL    NOT NULL,
    pit_stop_count INTEGER NOT NULL DEFAULT 0,   -- denormalised for sensitivity slider
    sc_laps        INTEGER NOT NULL DEFAULT 0,   -- number of SC-affected laps
    UNIQUE (batch_id, strategy_id, sim_index)
);

-- O(log n) win-probability / head-to-head queries (NFR-3, NFR-5)
CREATE INDEX idx_run_batch_strategy_time
    ON simulation_run (batch_id, strategy_id, total_time_s);

-- Bulk load of all runs for a batch
CREATE INDEX idx_run_batch ON simulation_run (batch_id);

CREATE TABLE lap_result (
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

-- Per-run trace retrieval for lap-chart visualisation (FR-15)
CREATE INDEX idx_lap_run ON lap_result (run_id);
```

---

## 4. Schema Decisions vs. Requirements Traceability

| Design choice | Requirement | How it satisfies it |
|---|---|---|
| `idx_run_batch_strategy_time` on `(batch_id, strategy_id, total_time_s)` | NFR-3, NFR-5 | O(log n) win-probability recomputation and sorted range scans |
| `lap_result` only for `sim_index < 100` | FR-15, NFR-5 | Caps table at 100 × laps rows/batch (~7k) while preserving traces |
| `pit_stop_count` denormalised on `simulation_run` | FR-14, NFR-3 | Sensitivity slider recomputes totals in memory; no `lap_result` touch |
| `strategy_stint` relational rows | FR-17 | GA writes novel strategies through the same schema as human ones |
| `master_seed` on batch; unique `(batch_id, strategy_id, sim_index)` | NFR-6 | Every run is reproducible and addressable |
| Config tables seeded from JSON config | NFR-12 | Adding a compound/track is a config entry, not a schema change |

---

## 5. Data Volume & Retention Estimates

| Table | Rows / batch (N = 100k, laps = 70) | Notes |
|---|---|---|
| `simulation_batch` | 1 | one transaction per experiment |
| `simulation_run` | 2 × N = 200,000 | 100k per strategy (2 strategies) |
| `lap_result` | 2 × 100 × 70 = 14,000 | sampled (`sim_index < 100`), both strategies |
| `strategy` / `strategy_stint` | ~2 + stints | ~2–6 stints per strategy |

- **1,000 batches** → ~200 M run rows, ~14 M lap rows. With the indexes above and SQLite's integer-optimised B-tree, typical batch queries stay far under the 1-second interaction budget (NFR-5).
- **Retention policy:** config rows are effectively immutable reference data; output rows can be pruned by `batch.created_at` if disk becomes a concern (no application impact — a batch is self-contained).

---

## 6. Reference Queries

**Q1 — Win probability between two strategies in a batch (NFR-3):**
```sql
-- Strategy A beats B: fraction of paired runs where A.total_time < B.total_time
SELECT
    AVG(CASE WHEN a.total_time_s < b.total_time_s THEN 1.0 ELSE 0.0 END) AS win_prob_a
FROM simulation_run a
JOIN simulation_run b
  ON a.batch_id = b.batch_id
 AND a.sim_index = b.sim_index
 AND b.strategy_id = :strategy_b_id
WHERE a.batch_id = :batch_id
  AND a.strategy_id = :strategy_a_id;
```

**Q2 — Mean / std / 95 % CI of finishing times for a strategy:**
```sql
SELECT COUNT(*)                              AS n,
       AVG(total_time_s)                     AS mean_time,
       STDEV(total_time_s)                   AS std_time,       -- SQLite: see note
       AVG(total_time_s) + 1.96 * STDEV(total_time_s) / SQRT(COUNT(*)) AS ci_upper,
       AVG(total_time_s) - 1.96 * STDEV(total_time_s) / SQRT(COUNT(*)) AS ci_lower
FROM simulation_run
WHERE batch_id = :batch_id AND strategy_id = :strategy_id;
```
> Note: SQLite lacks a built-in `STDEV`; the statistics layer computes CI in NumPy after a bulk load (vectorised, sub-second), which is the primary path (NFR-3). SQL aggregates serve ad-hoc checks only.

**Q3 — Lap trace for a representative run (FR-15):**
```sql
SELECT lap_number, lap_time_s, cumulative_time_s, tyre_age,
       fuel_remaining_kg, safety_car, stint_index
FROM lap_result
WHERE run_id = :run_id            -- pick sim_index = 0..99 run for each strategy
ORDER BY lap_number;
```

**Q4 — Pit-stop count distribution for the sensitivity slider (FR-14):**
```sql
SELECT pit_stop_count, COUNT(*) AS runs
FROM simulation_run
WHERE batch_id = :batch_id AND strategy_id = :strategy_id
GROUP BY pit_stop_count;
```

---

## 7. Write Strategy & Transactions

- **One transaction per batch.** `save_batch()` inserts all `simulation_run` rows with `executemany` inside a single `BEGIN/COMMIT`. At 200k rows this is fast and keeps the write atomic — a failed batch is fully rolled back.
- **Chunked fallback:** if peak RAM becomes a concern at scale, chunked writes are a drop-in alternative (see Open Questions in the blueprint) — the schema is identical.
- **Foreign keys ON** (`PRAGMA foreign_keys = ON`) so cascading deletes keep sampled `lap_result` and `strategy_stint` consistent.

---

## 8. Migration Strategy

- Schema is versioned in a single `schema.sql` + a `schema_version` pragma/user_version.
- Because config and output are separate, most evolutions are **additive** (new compound, new track, new column) and require no data migration.
- Any breaking change (e.g. richer denormalisation for a fuel-effect slider) is handled by a numbered migration script applied at startup (idempotent, tracked in `user_version`).
