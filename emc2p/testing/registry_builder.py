"""Component-first Registry construction, shared by every downstream
project's own test suite (this module was moved here specifically because
emc2p's and iacs's own tests/conftest.py files had drifted into
byte-identical duplicates of it -- see docs/manifest/history.yaml for that
history).

Used to also carry a lenient ``pandas.testing.assert_allclose`` monkeypatch
for comparing round-tripped component tables. That patch masked a real
export/reload fidelity gap (see docs/manifest/history.yaml:
project_history.assert_allclose_patch_replaced_by_assert_components_equal)
and has been replaced by ``emc2p.testing.expected_fixtures.
assert_components_equal``/``assert_registries_equal``, which compare every
common column instead of just ``component_type``/``modifier``.
"""

import pandas as pd
import ibis

from emc2p.registry import Registry


def make_registry(components: dict[str, list[dict]]) -> Registry:
    """Create a Registry from component-first data.

    Args:
        components: Dict mapping component type names to lists of row dicts.
            Each row dict should include "entity_id" plus any component fields.

    Returns:
        A Registry backed by a DuckDB connection.
    """
    conn = ibis.duckdb.connect()
    comp_tables = {}
    for comp_type, rows in components.items():
        df = pd.DataFrame(rows)
        conn.create_table(comp_type, df)
        comp_tables[comp_type] = conn.table(comp_type)
    return Registry(conn, comp_tables)
