"""Live regression checking the *write* stage in isolation, over emc2p's
own minimal MCP server (`emc2p.mcp_server`).

Given the already-decided outcome directly (no reasoning required), does
the model still call update_registry correctly: right component type,
right field, right value, on the right entity?

Generic (no downstream project's own domain concepts -- no event/
advance_simulation loop, just open_registry/view_registry/view_entity/
update_registry): three plain widget entities, one `status` component
(tests/data/scenarios/status_board/manifest.yaml), nothing to reason
about, so a failure here isolates the write step itself rather than
whether the model can also work out what to write.
"""

from pathlib import Path

import pytest

from tests.test_human_validated.helpers import HeadlessSession, _load_registrar
from emc2p.testing.registry_checks import unexpected_components

REPO_ROOT = Path(__file__).parent.parent.parent
STATUS_BOARD_SCENARIO_DIR = REPO_ROOT / "tests" / "data" / "scenarios" / "status_board"

# The scenario seed's own starting component (description) plus the one
# field a correct write is supposed to add to each widget.
_EXPECTED_STATUS_COMPONENTS = {"description", "status"}

_WRITE_GIVEN_INSTRUCTION = (
    "Call open_registry with database_url={database_url} and manifest_dir={manifest_dir} "
    "to open a registry. Each widget's status is already decided -- don't reason about "
    "anything, just record it via update_registry: widget_a's status is \"active\", "
    "widget_b's status is \"idle\", widget_c's status is \"broken\". Don't narrate "
    "anything back to me."
)


@pytest.mark.live
def test_write_accuracy_given_assignment(tmp_path):
    """Given each widget's status directly, does update_registry end up
    with exactly that -- no misattributed component, no dropped write?

    Reads the database only after the session (and with it, the
    subprocess-held emc2p-mcp server's own connection) has closed --
    unlike a Postgres-backed registry, DuckDB allows only one process to
    hold a connection to a given file at a time.
    """
    db_path = tmp_path / "test.duckdb"

    with HeadlessSession() as session:
        print(f"HeadlessSession trace: {session.trace_path}")
        narration = session.send_turn(
            _WRITE_GIVEN_INSTRUCTION.format(
                database_url=f"duckdb:///{db_path}", manifest_dir=str(STATUS_BOARD_SCENARIO_DIR)
            )
        )

    r = _load_registrar(db_path, narration)

    assert r.get_current_value("status", "value", "widget_a") == "active", (
        f"widget_a was told its status is active -- narration: {narration!r}"
    )
    assert r.get_current_value("status", "value", "widget_b") == "idle", (
        f"widget_b was told its status is idle -- narration: {narration!r}"
    )
    assert r.get_current_value("status", "value", "widget_c") == "broken", (
        f"widget_c was told its status is broken -- narration: {narration!r}"
    )

    for alias in ("widget_a", "widget_b", "widget_c"):
        misfires = unexpected_components(r, alias, _EXPECTED_STATUS_COMPONENTS)
        assert not misfires, f"{alias} got unexpected components {misfires} -- narration: {narration!r}"
