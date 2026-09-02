"""Headless dashboard smoke test (Streamlit AppTest) — Sprint 3.

Covers both data paths of the dashboard:
  1. "Run Simulation"  -> engine -> MonteCarloRunner -> charts (no DB write by default)
  2. "Load selected from database" -> SimulationRepository.load_batch -> charts

Run:
    python scripts/dashboard_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "dashboard.py"


def _report_failures(at: AppTest, stage: str) -> bool:
    if at.exception:
        print(f"=== EXCEPTION after {stage} ===")
        for exc in at.exception:
            print(exc.value)
            print(getattr(exc, "stack_trace", ""))
        return True
    return False


def main() -> int:
    at = AppTest.from_file(str(DASHBOARD), default_timeout=240)
    at.run()
    if _report_failures(at, "initial render"):
        return 1

    print("Initial render OK — widgets:")
    for sb in at.selectbox:
        print(f"  selectbox: {sb.label!r} = {sb.value!r}")
    for ni in at.number_input:
        print(f"  number_input: {ni.label!r} = {ni.value!r}")

    # ---- Flow 1: Run Simulation (persist is unchecked by default) ----
    iters = next(ni for ni in at.number_input if "iterations" in ni.label.lower())
    iters.set_value(1_000)
    run_btn = next(b for b in at.button if b.label == "Run Simulation")
    run_btn.click()
    at.run()
    if _report_failures(at, "Run Simulation"):
        return 1

    print("Run-Simulation flow OK:")
    for s in at.success:
        print(f"  success: {s.value[:90]}")
    print(f"  metrics rendered: {len(at.metric)} (expect >= 4)")
    if len(at.metric) < 4:
        print("  FAIL: too few metrics")
        return 1

    # ---- Flow 2: Load a stored batch from SQLite (Day-2 read path) ----
    saved_sb = next((sb for sb in at.selectbox if "Stored experiments" in sb.label), None)
    if saved_sb is None:
        print("No Saved-experiments selector present — is data/results.db empty?")
        return 0  # DB absent (e.g. fresh clone) is acceptable in CI

    options = saved_sb.options
    print(f"Saved experiments present: {len(options)} batch(es) -> {list(options)}")
    # Pick the Monaco 100k batch if available, else the first.
    target = 2 if 2 in options else options[0]
    saved_sb.set_value(target)
    load_btn = next(b for b in at.button if b.label == "Load selected from database")
    load_btn.click()
    at.run()
    if _report_failures(at, "Load from database"):
        return 1
    print(f"Load-from-database flow OK (batch #{target}); metrics: {len(at.metric)}")
    if len(at.metric) < 4:
        print("  FAIL: too few metrics after load")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
