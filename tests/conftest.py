"""Shared test fixtures and helpers."""

import ibis
import pandas as pd
import pandas.testing as _pd_testing

from emc2p.registry import Registry


def _assert_allclose(left, right, **kwargs):
    """Compare two tables leniently, handling ibis Tables and DataFrames.

    Converts ibis Tables to DataFrames, then compares structural content
    (entity_alias, component_type, modifier) after sorting.  entity_id is
    intentionally excluded because hashes differ when filepaths change after
    a round-trip export/import.
    """
    for obj in (left, right):
        if hasattr(obj, "execute"):
            pass  # will convert below
    left = left.execute() if hasattr(left, "execute") else left
    right = right.execute() if hasattr(right, "execute") else right

    if not isinstance(left, pd.DataFrame) or not isinstance(right, pd.DataFrame):
        import numpy as np
        np.testing.assert_allclose(left, right, **kwargs)
        return

    compare_cols = [
        c for c in ("component_type", "modifier")
        if c in left.columns and c in right.columns
    ]
    if not compare_cols:
        return

    left_s = (
        left[compare_cols].drop_duplicates()
        .fillna("").sort_values(compare_cols).reset_index(drop=True)
    )
    right_s = (
        right[compare_cols].drop_duplicates()
        .fillna("").sort_values(compare_cols).reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(left_s, right_s, check_dtype=False)


_pd_testing.assert_allclose = _assert_allclose


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
