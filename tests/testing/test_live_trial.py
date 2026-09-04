"""Regression coverage for `emc2p.testing.live_trial.make_live_trial_fixture`.

Uses `pytester` (an inner, isolated pytest run) rather than calling the
factory's own functions directly: `pytest_generate_tests`/
`pytest_runtest_makereport` are pytest hooks that only do anything as
part of a real collection+run cycle, unlike `responder_fixture`'s plain
fixture logic (see test_responder_fixture.py, which needs no such
inner-run machinery).
"""

import csv
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

_INNER_CONFTEST = """
from pathlib import Path

from emc2p.testing.live_trial import make_live_trial_fixture, pytest_runtest_makereport  # noqa: F401

live_trial, pytest_generate_tests = make_live_trial_fixture(
    results_dir=Path(__file__).parent / "results",
    repo_root=Path(__file__).parent,
    env_var="TEST_LIVE_TRIALS",
)
"""


def _results_csv(pytester: pytest.Pytester, test_name: str) -> Path:
    return pytester.path / "results" / f"{test_name}.csv"


def _read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


class TestDefaultReps:
    def test_default_env_var_unset_runs_once_and_records_one_passing_row(self, pytester: pytest.Pytester):
        pytester.makeconftest(_INNER_CONFTEST)
        pytester.makepyfile(
            """
            def test_always_passes(live_trial):
                assert live_trial == 1
            """
        )
        result = pytester.runpytest()
        result.assert_outcomes(passed=1)

        rows = _read_rows(_results_csv(pytester, "test_always_passes"))
        assert len(rows) == 1
        assert rows[0]["passed"] == "True"
        assert rows[0]["error_log"] == ""
        assert rows[0]["commit"]  # non-empty: "unknown" fallback or a real hash
        assert rows[0]["timestamp"]


class TestRepeatedTrials:
    def test_env_var_n_runs_n_trials_and_records_n_rows(self, pytester: pytest.Pytester, monkeypatch):
        pytester.makeconftest(_INNER_CONFTEST)
        pytester.makepyfile(
            """
            def test_always_passes(live_trial):
                pass
            """
        )
        monkeypatch.setenv("TEST_LIVE_TRIALS", "3")
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=3)
        result.stdout.fnmatch_lines(
            ["*test_always_passes*trial1*", "*test_always_passes*trial2*", "*test_always_passes*trial3*"]
        )

        rows = _read_rows(_results_csv(pytester, "test_always_passes"))
        assert len(rows) == 3
        assert all(row["passed"] == "True" for row in rows)

    def test_a_failing_trial_records_passed_false_with_a_nonempty_error_log(self, pytester: pytest.Pytester):
        pytester.makeconftest(_INNER_CONFTEST)
        pytester.makepyfile(
            """
            def test_always_fails(live_trial):
                assert False, "deliberate failure for the harness's own test"
            """
        )
        result = pytester.runpytest()
        result.assert_outcomes(failed=1)

        rows = _read_rows(_results_csv(pytester, "test_always_fails"))
        assert len(rows) == 1
        assert rows[0]["passed"] == "False"
        assert "deliberate failure for the harness's own test" in rows[0]["error_log"]

    def test_multiple_trials_all_append_to_the_same_csv_not_one_per_trial(self, pytester: pytest.Pytester, monkeypatch):
        """`request.node.originalname` (not `.name`, which carries the
        `[trial2]` parametrize suffix) is what names the CSV file -- all
        trials of one logical test share one file."""
        pytester.makeconftest(_INNER_CONFTEST)
        pytester.makepyfile(
            """
            def test_always_passes(live_trial):
                pass
            """
        )
        monkeypatch.setenv("TEST_LIVE_TRIALS", "2")
        pytester.runpytest()
        results_dir = pytester.path / "results"
        csv_files = list(results_dir.glob("*.csv"))
        assert len(csv_files) == 1
        assert csv_files[0].name == "test_always_passes.csv"
