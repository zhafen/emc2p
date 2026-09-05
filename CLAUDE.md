# CLAUDE.md - AI Assistant Instructions for emc2p

See README.md for what this project is and its development basics
(`uv`, `uv run pytest`, the `.githooks` pre-commit hook).

## Commit and push immediately, don't batch or wait for a prompt

Commit (and push) each meaningful change as soon as it's made, rather
than batching several turns' edits together or waiting for explicit
go-ahead first. This project is worked on primarily through ephemeral
cloud/remote sessions now, not long-running local ones -- uncommitted
work sitting in a container risks being lost outright when it's
reclaimed, and diff-based tooling (e.g. the `PreToolUse` hooks in
`.claude/hooks/`) is keyed off git's own state, so batching multiple
turns' changes into one eventual commit can also cause such a check to
silently miss what actually changed turn-by-turn.

## Three test tiers, by cost

- `uv run pytest` (default, no flags) -- fast (~20s). Tests that go
  through the real load_manifest/from_manifest pipeline (Hamilton driver
  construction dominates regardless of manifest size), spin up a real
  MCP-server subprocess, or otherwise individually take 1s+ are marked
  `slow` and excluded by default (see the `slow` marker in
  pyproject.toml). Use this while iterating before a commit.
- `uv run pytest -m "not live"` -- the full suite, `slow` tests included
  but `live` excluded. This is also what CI runs. Run this before merging
  a PR, or after touching something with genuinely broad reach (a
  `registry.py`/`utils.py` helper, a builtins component schema, a base
  class most tests construct through, or anything in `load_manifest.py`/
  `export_manifest.py`'s shared dataflow plumbing). Passing `-m` on the
  command line fully replaces the default marker filter from `addopts`,
  so this isn't additive with "not slow and not live" -- it's a
  completely separate expression that happens to still exclude live.
- `uv run pytest -m live` -- the live-model tests, a separate axis from
  slow/full entirely (real API calls, cost real usage, flaky by nature).
  Run these deliberately and sparingly, never as a side effect of a
  broader invocation -- a bare `uv run pytest` (no `-m`) in an
  environment with a working API key will actually execute them; the
  default `addopts` filter is what keeps that from happening.

**Scope test runs to what changed even within the fast tier; don't run
the full default suite on every edit either.** While iterating on a
specific function/module, run just the test file(s) that exercise it
(e.g. `uv run pytest tests/test_dataflows/test_resolve_paths.py -q`),
narrowing further with `-k` when chasing one failing test. When unsure
what a change affects, grep for the changed function/class name across
`tests/` first to find the real blast radius, rather than defaulting to
a broader run as a substitute for that.

Downstream repos (iacs, story-simulator) pin a specific emc2p commit; a
change here doesn't need to trigger their test suites too unless you're
about to bump their pin to pick it up.
