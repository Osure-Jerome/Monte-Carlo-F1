"""Headless dashboard smoke test (Streamlit AppTest).

Verifies the dashboard imports and renders without exceptions, then clicks
"Run Simulation" at a reduced iteration count to exercise the full Sprint 2
batch path (engine -> MonteCarloRunner -> statistics -> charts) end to end.

Run:
    python scripts/dashboard_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "dashboard.py"


def main() -> int:
    at = AppTest.from_file(str(DASHBOARD), default_timeout=180)
    at.run()

    if at.exception:
        print("=== EXCEPTION on initial render ===")
        for exc in at.exception:
            print(exc.value)
            print(getattr(exc, "stack_trace", ""))
        return 1

    print("Initial render OK — widgets present:")
    for sb in at.selectbox:
        print(f"  selectbox: {sb.label!r} = {sb.value!r}")
    for ni in at.number_input:
        print(f"  number_input: {ni.label!r} = {ni.value!r}")

    # Reduce iterations so the flow stays fast, then click Run Simulation.
    iters = next(ni for ni in at.number_input if "iterations" in ni.label.lower())
    iters.set_value(1_000)
    run_btn = next(b for b in at.button if b.label == "Run Simulation")
    run_btn.click()
    at.run()

    if at.exception:
        print("=== EXCEPTION after Run Simulation ===")
        for exc in at.exception:
            print(exc.value)
            print(getattr(exc, "stack_trace", ""))
        return 1

    print("Run Simulation flow OK. Success line:")
    for md in at.markdown:
        if "Simulation complete" in md.value:
            print(f"  {md.value[:80]}")
    print("Metrics shown:")
    for m in at.metric:
        print(f"  {m.label}: {m.value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
