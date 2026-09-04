"""Shared test fixtures and helpers."""

# Enables the `pytester` fixture (an inner, isolated pytest run) -- used by
# tests/testing/test_live_trial.py to exercise pytest_generate_tests/
# pytest_runtest_makereport hooks, which can't be driven by calling them
# directly the way a plain fixture-factory's logic can.
pytest_plugins = ["pytester"]
