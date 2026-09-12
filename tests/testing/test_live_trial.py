"""Regression coverage for `emc2p.testing.live_trial.make_live_trial_fixture`.

Uses `pytester` (an inner, isolated pytest run), since
`pytest_generate_tests`/`pytest_runtest_makereport` only do anything as
part of a real collection+run cycle. Inner test bodies are modeled on a
write-accuracy check (recording a car's parking location, the actual
motivating case) but kept deterministic and API-cost-free.
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

_TEST_NAME = "test_write_accuracy_records_correct_location"

_CORRECT_WRITE_TEST_BODY = f"""
def {_TEST_NAME}(live_trial):
    # Stand-in for a live model's update_registry write -- a real
    # write-accuracy test asks a model to record this; here it's just a
    # fixed "recorded" value so this illustration stays fast and free.
    recorded_location = "parked in parking_spot_1"
    assert "parking_spot_1" in recorded_location
"""


def _results_csv(pytester: pytest.Pytester, test_name: str = _TEST_NAME) -> Path:
    return pytester.path / "results" / f"{test_name}.csv"


def _read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


class TestDefaultReps:
    def test_default_env_var_unset_runs_once_and_records_one_passing_row(self, pytester: pytest.Pytester):
        pytester.makeconftest(_INNER_CONFTEST)
        pytester.makepyfile(
            f"""
            def {_TEST_NAME}(live_trial):
                assert live_trial.number == 1
                recorded_location = "parked in parking_spot_1"
                assert "parking_spot_1" in recorded_location
            """
        )
        result = pytester.runpytest()
        result.assert_outcomes(passed=1)

        rows = _read_rows(_results_csv(pytester))
        assert len(rows) == 1
        assert rows[0]["passed"] == "True"
        assert rows[0]["live_test_id"] == ""  # never set -- optional
        assert rows[0]["commit"]  # non-empty: "unknown" fallback or a real hash
        assert rows[0]["timestamp"]

    def test_records_the_live_test_id_the_test_body_sets(self, pytester: pytest.Pytester):
        """A real test sets this to its own session trace path -- the CSV
        row points at the full detail rather than duplicating it."""
        pytester.makeconftest(_INNER_CONFTEST)
        pytester.makepyfile(
            f"""
            def {_TEST_NAME}(live_trial):
                live_trial.id = "session-trace-abc123.jsonl"
            """
        )
        result = pytester.runpytest()
        result.assert_outcomes(passed=1)

        rows = _read_rows(_results_csv(pytester))
        assert rows[0]["live_test_id"] == "session-trace-abc123.jsonl"


class TestRepeatedTrials:
    def test_env_var_n_runs_n_trials_and_records_n_rows(self, pytester: pytest.Pytester, monkeypatch):
        pytester.makeconftest(_INNER_CONFTEST)
        pytester.makepyfile(_CORRECT_WRITE_TEST_BODY)
        monkeypatch.setenv("TEST_LIVE_TRIALS", "3")
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=3)
        result.stdout.fnmatch_lines(
            [f"*{_TEST_NAME}*trial1*", f"*{_TEST_NAME}*trial2*", f"*{_TEST_NAME}*trial3*"]
        )

        rows = _read_rows(_results_csv(pytester))
        assert len(rows) == 3
        assert all(row["passed"] == "True" for row in rows)

    def test_a_write_accuracy_misfire_records_passed_false(self, pytester: pytest.Pytester):
        """A write-accuracy misfire: the model writes a plausible-looking
        but wrong spot (see story-simulator#55)."""
        pytester.makeconftest(_INNER_CONFTEST)
        pytester.makepyfile(
            f"""
            def {_TEST_NAME}(live_trial):
                recorded_location = "parked on the street"  # wrong spot -- a misfire
                assert "parking_spot_1" in recorded_location, (
                    f"expected car_a parked in parking_spot_1, got {{recorded_location!r}}"
                )
            """
        )
        result = pytester.runpytest()
        result.assert_outcomes(failed=1)

        rows = _read_rows(_results_csv(pytester))
        assert len(rows) == 1
        assert rows[0]["passed"] == "False"

    def test_multiple_trials_all_append_to_the_same_csv_not_one_per_trial(
        self, pytester: pytest.Pytester, monkeypatch
    ):
        """`originalname` (not `.name`, which carries the `[trial2]` suffix)
        names the CSV file, so all trials of one test share one file."""
        pytester.makeconftest(_INNER_CONFTEST)
        pytester.makepyfile(_CORRECT_WRITE_TEST_BODY)
        monkeypatch.setenv("TEST_LIVE_TRIALS", "2")
        pytester.runpytest()
        results_dir = pytester.path / "results"
        csv_files = list(results_dir.glob("*.csv"))
        assert len(csv_files) == 1
        assert csv_files[0].name == f"{_TEST_NAME}.csv"
