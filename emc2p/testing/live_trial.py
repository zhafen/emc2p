"""Run a live test multiple times ("trials") and record each one's
outcome to a CSV, so flaky live-model behavior reads as a pass rate
across many runs instead of a single pass/fail.

Generic across downstream projects (matching ``responder_fixture.py``'s
factory-returns-a-fixture pattern) -- wire it into a project's own root
``conftest.py``::

    from emc2p.testing.live_trial import make_live_trial_fixture, pytest_runtest_makereport  # noqa: F401

    live_trial, pytest_generate_tests = make_live_trial_fixture(
        results_dir=REPO_ROOT / ".live_trial_results",
        repo_root=REPO_ROOT,
        env_var="STORY_SIM_LIVE_TRIALS",
    )

A test taking ``live_trial`` as a parameter reruns ``env_var``-many
times (default 1). The fixture yields a :class:`LiveTrial` (``.number``,
and a settable ``.id`` for the test's own pointer to where this trial's
full detail lives, e.g. a session trace path -- optional, blank if
unset). Each trial appends one row (``commit``/``passed``/
``live_test_id``/``timestamp``) to ``<results_dir>/<test name>.csv``.

``pytest_generate_tests`` (parametrizes the trial count at collection
time) and ``pytest_runtest_makereport`` (stashes each phase's
``TestReport`` on the item so a fixture can see pass/fail -- exported
directly since it isn't live-trial-specific) must both be assigned under
those exact names in a ``conftest.py`` for pytest to discover them as hooks.
"""

from __future__ import annotations

import csv
import dataclasses
import os
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterator

import pytest

_CSV_FIELDS = ["commit", "passed", "live_test_id", "timestamp"]


@dataclasses.dataclass
class LiveTrial:
    """The `live_trial` fixture's own yielded value.

    `number` is the 1-indexed trial number. `id`, optionally set by the
    test, is recorded verbatim as the CSV's `live_test_id`.
    """

    number: int
    id: str = ""


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Stash each phase's `TestReport` onto `item` as `rep_<phase>`, so a
    fixture's teardown can read `item.rep_call` for the run outcome.
    """
    outcome = yield
    report = outcome.get_result()
    setattr(item, f"rep_{report.when}", report)


def _git_commit(repo_root: Path) -> str:
    """Best-effort current commit hash for `repo_root`; "unknown" on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _sanitize_filename(name: str) -> str:
    """Filesystem-safe stem for a per-test CSV filename."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def _append_row(csv_path: Path, *, commit: str, passed: bool, live_test_id: str, timestamp: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(
            {"commit": commit, "passed": passed, "live_test_id": live_test_id, "timestamp": timestamp}
        )


def make_live_trial_fixture(
    *,
    results_dir: Path,
    repo_root: Path,
    env_var: str,
    default_reps: int = 1,
    fixture_name: str = "live_trial",
) -> tuple[Callable, Callable]:
    """Build a `(live_trial_fixture, pytest_generate_tests)` pair for
    repeated, CSV-recorded live-test runs.

    `results_dir`: where each test's `<test name>.csv` lands.
    `repo_root`: repo `git rev-parse HEAD` is read from, for the `commit` column.
    `env_var`: trial count, read once at collection time (falls back to `default_reps`, 1).
    `fixture_name`: override if a project already has its own `live_trial`.

    Both return values must be assigned under exactly those names in a
    project's `conftest.py` -- pytest only discovers `pytest_generate_tests` by that name.
    """

    def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
        if fixture_name not in metafunc.fixturenames:
            return
        n = int(os.environ.get(env_var, default_reps))
        metafunc.parametrize(
            fixture_name,
            range(1, n + 1),
            indirect=True,
            ids=[f"trial{i}" for i in range(1, n + 1)],
        )

    def _live_trial(request: pytest.FixtureRequest) -> Iterator[LiveTrial]:
        trial = LiveTrial(number=getattr(request, "param", 1))
        yield trial
        report = getattr(request.node, "rep_call", None)
        passed = bool(report is not None and report.passed)
        csv_path = results_dir / f"{_sanitize_filename(request.node.originalname or request.node.name)}.csv"
        _append_row(
            csv_path,
            commit=_git_commit(repo_root),
            passed=passed,
            live_test_id=trial.id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    live_trial = pytest.fixture(_live_trial, name=fixture_name)
    return live_trial, pytest_generate_tests
