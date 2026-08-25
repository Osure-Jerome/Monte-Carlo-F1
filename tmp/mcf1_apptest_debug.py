"""Reproduce the dashboard startup error headlessly via Streamlit AppTest."""
import traceback
from streamlit.testing.v1 import AppTest

at = AppTest.from_file(
    "/home/jerome-maunga/Desktop/Monte Carlo F1/Monte-Carlo-F1/dashboard.py",
    default_timeout=30,
)
at.run()

if at.exception:
    print("=== EXCEPTION ===")
    for exc in at.exception:
        print(exc.value)
        print("--- traceback ---")
        print(exc.stack_trace)
else:
    print("No exception on first run.")

# Inspect the actual widget values
print("=== widget introspection ===")
for i, sb in enumerate(at.selectbox):
    print(f"selectbox[{i}] label={sb.label!r} value={sb.value!r} type={type(sb.value)}")
for i, ni in enumerate(at.number_input):
    print(f"number_input[{i}] label={ni.label!r} value={ni.value!r} type={type(ni.value)}")
