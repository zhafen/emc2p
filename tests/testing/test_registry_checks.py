"""Regression coverage for `emc2p.testing.registry_checks`.

Reproduces, mechanically and deterministically (no LLM, no live model
calls, DuckDB in-memory), a read-path hazard: a time_dimension'd
component field written twice for the same entity, at two different
`time` values, whose `view_df`/`view` row order doesn't match write
order. `get_current_value`/`view_current` still resolve the right one
(they sort explicitly, see `Registry._current_table`); a reader that
instead trusts positional row order can silently return the wrong row,
including against a real Postgres-backed registry, where row order isn't
guaranteed to match write order.
"""

import pytest

from emc2p.registrar import Registrar
from emc2p.testing.registry_checks import write_time

# Both tests here synthesize a large number of out-of-order writes -- each
# individually takes several seconds.
pytestmark = pytest.mark.slow

_SOURCE_KEY = "session"


def _write(registrar: Registrar, alias: str, value: str, time: int) -> None:
    registrar.update(
        yaml_strings={
            _SOURCE_KEY: f"{alias}:\n    - description:\n        value: {value}\n        time: {time}\n"
        }
    )


def _force_physical_row_order_by_time_desc(registrar: Registrar, component_type: str) -> None:
    """Rewrite `component_type`'s table with its rows in `time DESC`
    physical order, with no `ORDER BY` on the query that reads it back.

    `Registry._view` (what `view_df`/`view` call) never sorts its own
    output -- a real Postgres-backed registry can return two rows for the
    same alias in either order, regardless of which was written first
    (see `write_time`'s own docstring). This helper constructs that exact
    "physical order disagrees with time order" state directly, rather
    than depending on a specific backend's own query-planner quirks
    (which aren't the same across backends, and aren't the point under
    test) -- the point under test is `write_time`'s own read, not any one
    backend's scrambling behavior.
    """
    con = registrar.registry._con
    con.raw_sql(
        f"CREATE TABLE {component_type}_reordered AS "
        f"SELECT * FROM {component_type} ORDER BY time DESC"
    )
    con.raw_sql(f"DROP TABLE {component_type}")
    con.raw_sql(f"ALTER TABLE {component_type}_reordered RENAME TO {component_type}")


def test_write_time_survives_out_of_order_rows():
    """`write_time` must return the *latest* time_dimension value, not
    whichever row a plain, unsorted `view_df` happens to return last.
    """
    r = Registrar()
    _write(r, "widget", "earlier_value", 1020)
    _write(r, "widget", "later_value", 1030)
    _force_physical_row_order_by_time_desc(r, "description")

    # Sanity check: the forced reorder actually did put the earlier-time
    # row last, positionally -- otherwise this test isn't exercising
    # anything (see the module docstring for why this matters).
    raw = r.registry.view_df("description", aliases="widget")
    assert list(raw["description.time"]) == [1030, 1020], (
        "the reorder helper didn't actually invert physical row order -- "
        f"this test needs it to, got {list(raw['description.time'])}"
    )

    assert r.get_current_value("description", "value", "widget") == "later_value", (
        "view_current's explicit max(time) resolution should be unaffected by row order"
    )
    assert write_time(r, "description", "widget") == 1030


def test_write_time_matches_current_value_across_many_out_of_order_writes():
    """Same shape of check, several writes deep, so a fix that merely
    special-cases two rows wouldn't quietly pass.
    """
    r = Registrar()
    for value, time in [("a", 10), ("b", 30), ("c", 20), ("d", 50), ("e", 40)]:
        _write(r, "widget", value, time)
    _force_physical_row_order_by_time_desc(r, "description")

    assert r.get_current_value("description", "value", "widget") == "d"
    assert write_time(r, "description", "widget") == 50
