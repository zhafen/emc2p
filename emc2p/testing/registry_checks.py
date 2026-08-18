"""Generic helpers for checking what a live MCP client actually wrote to an
emc2p registry -- used alongside `headless_session.HeadlessSession` to turn
a model's narration into concrete, checkable registry state.

Nothing here knows about any downstream project's own component schema
(e.g. story-simulator's `location`/`event_state`) -- a caller names the
component types/fields it cares about itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import ibis

if TYPE_CHECKING:
    from emc2p.registrar import Registrar

# Intrinsic emc2p bookkeeping component types every entity has a row in
# regardless of what anyone wrote -- entity_id (alias/filepath/hash) and
# component_type (which types this entity declares/uses). Not something a
# model ever chooses to write, so always excluded from
# `unexpected_components`'s result rather than making every caller repeat
# this same exclusion in its own `expected` set.
_ALWAYS_PRESENT_META_COMPONENTS = {"entity_id", "component_type"}


def schema_exists(dsn: str, schema: str) -> bool:
    """Whether `schema` (an ibis/database-level namespace, not a table's
    column structure) already exists on the database at `dsn`.

    For a live test whose model narrates a plausible-sounding scene
    without ever actually opening/creating its own registry (confirmed
    happening in practice with a weak model), this lets a caller assert
    that state was actually created for the exact location it expected,
    before trying to load it -- rather than either silently reading
    whichever schema a connection falls through to when the named one
    doesn't exist, or hitting a confusing raw `ibis` error partway through
    loading a nonexistent schema's tables.
    """
    con = ibis.connect(dsn)
    return schema in con.list_databases()


def unexpected_components(registrar: "Registrar", alias: str, expected: set[str]) -> set[str]:
    """Component types currently recorded on `alias` beyond `expected`
    (plus the always-present meta types, see
    `_ALWAYS_PRESENT_META_COMPONENTS`).

    Names *what* got written instead of the intended field when a
    targeted check on one field fails, rather than just noting that the
    intended field is empty -- e.g. a model recording a fact under an
    invented or wrong component name instead of the one a caller expected.
    `expected` should cover both the scenario's starting components and
    whatever a fully correct write is supposed to add; any component type
    showing up outside that set is a plausible misattribution candidate,
    not proof of one -- a legitimate write could add components beyond
    `expected` for reasons unrelated to any bug.
    """
    return set(registrar.view_entity_df(alias)) - expected - _ALWAYS_PRESENT_META_COMPONENTS


def write_time(registrar: "Registrar", component_type: str, alias: str) -> float | None:
    """The `time` value `component_type` was last written for `alias`
    under its own `time_dimension`-flagged field, or None if nothing's
    been recorded yet.

    Useful as order evidence in a live test that doesn't pace itself one
    step at a time: comparing these recorded times across entities/
    component types confirms each write happened when a caller expected
    it to (e.g. stamped with a particular event's own end_time), not
    guessed independent of it -- regardless of how many actual
    conversation turns the model took to get there.
    """
    df = registrar.view_df(component_type, aliases=alias)
    if df.empty:
        return None
    return float(df.iloc[-1][f"{component_type}.time"])
