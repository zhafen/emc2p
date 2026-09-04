"""Regression coverage for `emc2p.testing.live_trial.make_live_trial_fixture`.

Uses `pytester` (an inner, isolated pytest run) rather than calling the
factory's own functions directly: `pytest_generate_tests`/
`pytest_runtest_makereport` are pytest hooks that only do anything as
part of a real collection+run cycle, unlike `responder_fixture`'s plain
fixture logic (see test_responder_fixture.py, which needs no such
inner-run machinery).

The inner test bodies below are modeled on story-simulator's own
write-accuracy live tests (`tests/location_write_harness.py`,
`tests/test_human_validated/test_scenarios/parking/test_scenario.py`) --
the actual motivating use case for `live_trial`: asking a model to record
where a car parked, then checking the write landed on the right spot,
repeated across many trials to read a pass rate instead of one pass/fail.
Kept deterministic and API-cost-free here (a fixed "recorded" string
stands in for a real model's write) since this file is only testing the
harness mechanism itself, not any model's actual judgment quality.
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
        """A real write-accuracy test sets this to its own session trace
        path (e.g. `live_trial.id = str(session.trace_path)`) -- the CSV
        row then points at where the full turn-by-turn detail lives,
        instead of duplicating it into the CSV itself."""
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
        """A write-accuracy misfire, standing in for the real thing (see
        story-simulator#55's "wrong-but-real component"/invented-location
        threads): the model writes a plausible-looking but wrong spot.
        """
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
        """`request.node.originalname` (not `.name`, which carries the
        `[trial2]` parametrize suffix) is what names the CSV file -- all
        trials of one logical test share one file, so a write-accuracy
        test's pass rate can be read straight off one CSV's row count."""
        pytester.makeconftest(_INNER_CONFTEST)
        pytester.makepyfile(_CORRECT_WRITE_TEST_BODY)
        monkeypatch.setenv("TEST_LIVE_TRIALS", "2")
        pytester.runpytest()
        results_dir = pytester.path / "results"
        csv_files = list(results_dir.glob("*.csv"))
        assert len(csv_files) == 1
        assert csv_files[0].name == f"{_TEST_NAME}.csv"
