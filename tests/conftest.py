"""Shared test fixtures and helpers."""

# Enables the `pytester` fixture (an inner, isolated pytest run), used
# by tests/testing/test_live_trial.py to exercise its pytest hooks.
pytest_plugins = ["pytester"]
