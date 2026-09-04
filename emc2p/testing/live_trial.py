"""Pytest infrastructure for running a live test multiple times ("trials")
and recording each trial's own outcome to a CSV, so flaky live-model
behavior can be read as a pass rate across many runs over time instead of
a single pass/fail per invocation.

Generic across downstream projects, matching this package's other
``testing/`` modules (``responder_fixture.py``'s factory-returns-a-fixture
pattern in particular): nothing here knows about a downstream project's
own scenario/responder vocabulary or repo layout. A downstream project
wires this in via its own root ``conftest.py``::

    from emc2p.testing.live_trial import make_live_trial_fixture, pytest_runtest_makereport  # noqa: F401

    live_trial, pytest_generate_tests = make_live_trial_fixture(
        results_dir=REPO_ROOT / ".live_trial_results",
        repo_root=REPO_ROOT,
        env_var="STORY_SIM_LIVE_TRIALS",
    )

Any live test that wants trial recording/repetition then adds
``live_trial`` to its own parameter list (the fixture yields the
1-indexed trial number, if a test wants to log or branch on it, but most
tests can just ignore it):

    @pytest.mark.live
    def test_scenario_resolves_correctly(self, tmp_path, live_trial):
        ...

Running with ``STORY_SIM_LIVE_TRIALS=5 uv run pytest ... -m live`` then
collects and runs that test 5 times (``test_foo[trial1]`` ..
``test_foo[trial5]``); a bare invocation (env var unset) runs it once,
same as before this fixture existed. Each trial appends one row --
``commit``/``passed``/``error_log``/``timestamp`` -- to
``<results_dir>/<test name>.csv`` (one file per logical test, shared
across all its trials; create ``results_dir`` as a gitignored directory,
matching this project's own ``.live_test_traces/`` convention).

Two module-level pieces are needed, not just a fixture, since neither
half of the job is something a fixture can do alone:

- "How many times to collect this test" is a collection-time decision
  (``pytest_generate_tests``), decided once per test session before any
  fixture runs.
- "Did this specific trial pass" needs pytest's own run outcome, which
  isn't otherwise visible from inside a fixture -- ``pytest_runtest_makereport``
  stashes each phase's ``TestReport`` onto the test item (the same
  stash-the-report-on-the-item idiom the pytest docs themselves describe:
  https://docs.pytest.org/en/stable/example/simple.html#making-test-result-information-available-in-fixtures).
  That hook has nothing live-trial-specific about it -- any fixture
  wanting a test's own outcome needs the exact same stash -- so it's a
  plain module-level function here, not produced by the factory: a
  downstream project defining its own copy of this same hook elsewhere
  would collide with this one, so this is the one instance to import
  instead of writing a second.

Per-test override of the trial count (e.g. a single test always running
5 trials regardless of the env var) isn't supported -- every test
requesting ``live_trial`` in one session runs the same number of trials,
resolved once at collection time.
"""

from __future__ import annotations

import csv
import os
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterator

import pytest

_CSV_FIELDS = ["commit", "passed", "error_log", "timestamp"]


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Stash each phase's own `TestReport` onto `item` as `rep_<phase>`
    (`rep_setup`/`rep_call`/`rep_teardown`), so a fixture's teardown can
    read `item.rep_call` -- the run outcome isn't otherwise visible from
    inside a fixture. See this module's own docstring for why this is a
    plain function, not something `make_live_trial_fixture` produces.
    """
    outcome = yield
    report = outcome.get_result()
    setattr(item, f"rep_{report.when}", report)


def _git_commit(repo_root: Path) -> str:
    """Best-effort current commit hash for `repo_root`; "unknown" if it
    can't be read (a shallow clone missing HEAD, a non-git checkout, an
    unexpected git failure, ...) -- never raises, since a trial's own
    pass/fail must still be recorded either way.
    """
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


def _append_row(csv_path: Path, *, commit: str, passed: bool, error_log: str, timestamp: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({"commit": commit, "passed": passed, "error_log": error_log, "timestamp": timestamp})


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

    `results_dir`: where each test's own `<test name>.csv` lands --
    typically a downstream project's own gitignored directory.
    `repo_root`: passed to `git rev-parse HEAD` for the CSV's `commit`
    column -- the downstream project's own repo (whose code is actually
    under test), not emc2p's own.
    `env_var`: how many trials to run per test requesting this fixture,
    read once at collection time (e.g. `STORY_SIM_LIVE_TRIALS=5`) --
    falls back to `default_reps` (1, today's plain single-run behavior)
    when unset, same resolution style as
    `responder_fixture.make_responder_fixture`.
    `fixture_name`: the registered fixture's own name -- override if a
    project already has its own `live_trial`-named fixture.

    Both returned callables must be assigned at module level in the
    downstream project's own root `conftest.py` under exactly these
    names (`live_trial`/`pytest_generate_tests`, or whatever local names
    match how the project's tests request/pytest discovers them) --
    `pytest_generate_tests` is a hook pytest only discovers by that exact
    name in a `conftest.py`.
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

    def _live_trial(request: pytest.FixtureRequest) -> Iterator[int]:
        trial_number = getattr(request, "param", 1)
        yield trial_number
        report = getattr(request.node, "rep_call", None)
        if report is None:
            passed = False
            error_log = "no test report captured (setup/collection error?)"
        else:
            passed = bool(report.passed)
            error_log = "" if passed else report.longreprtext
        csv_path = results_dir / f"{_sanitize_filename(request.node.originalname or request.node.name)}.csv"
        _append_row(
            csv_path,
            commit=_git_commit(repo_root),
            passed=passed,
            error_log=error_log,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    live_trial = pytest.fixture(_live_trial, name=fixture_name)
    return live_trial, pytest_generate_tests
