# CLAUDE.md - AI Assistant Instructions for emc2p

See README.md for what this project is and its development basics
(`uv`, `uv run pytest`, the `.githooks` pre-commit hook).

## Scope test runs to what changed; don't run the full suite on every edit

The full suite (`uv run pytest -q`) takes several minutes. Running it
after each small change during iterative debugging repeatedly pays that
cost for no benefit, since most edits only affect one or two test files.

- While iterating on a specific function/module, run just the test
  file(s) that exercise it (e.g. `uv run pytest tests/test_dataflows/
  test_resolve_paths.py -q`), narrowing further with `-k` when chasing
  one failing test.
- When unsure what a change affects, grep for the changed function/class
  name across `tests/` first to find the real blast radius, rather than
  defaulting to the full suite as a substitute for that.
- Reserve a full-suite run for right before committing, right before
  pushing, or after touching something with genuinely broad reach (a
  `registry.py`/`utils.py` helper, a builtins component schema, a base
  class most tests construct through, or anything in `load_manifest.py`/
  `export_manifest.py`'s shared dataflow plumbing).
- Downstream repos (iacs, story-simulator) pin a specific emc2p commit;
  a change here doesn't need to trigger their test suites too unless
  you're about to bump their pin to pick it up.
