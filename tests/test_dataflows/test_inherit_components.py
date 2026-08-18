"""Tests for the inherit_components Hamilton DAG functions."""

import ibis
import pandas as pd
import pytest

import emc2p.dataflows.derive.inherit_components as inherit_components


def _field_row(entity_id, field_name, field_type=None, component_index=0):
    return {"entity_id": entity_id, "component_index": component_index,
            "value": field_name, "type": field_type}


def _parent_row(entity_id, parent_id):
    return {"entity_id": entity_id, "parent_eid": parent_id}


def _empty_parent():
    return ibis.memtable(pd.DataFrame([], columns=["entity_id", "parent_eid"]))


class TestDerivedField:

    def _call(self, field_rows, parent_rows=None):
        field = ibis.memtable(pd.DataFrame(field_rows))
        parent = (
            _empty_parent() if not parent_rows
            else ibis.memtable(pd.DataFrame(parent_rows))
        )
        return inherit_components.derived_field(field, parent).execute()

    def _field_names(self, df, entity_id):
        return set(df[df["entity_id"] == entity_id]["value"])

    def test_returns_ibis_table(self):
        field = ibis.memtable(pd.DataFrame([_field_row("e1", "x", "str")]))
        result = inherit_components.derived_field(field, _empty_parent())
        assert isinstance(result, ibis.Table)

    def test_entity_with_no_parent_has_own_fields(self):
        df = self._call([_field_row("e1", "x"), _field_row("e1", "y", component_index=1)])
        assert self._field_names(df, "e1") == {"x", "y"}

    def test_child_inherits_parent_fields(self):
        df = self._call(
            [_field_row("parent", "x"), _field_row("child", "y")],
            [_parent_row("child", "parent")],
        )
        assert "x" in self._field_names(df, "child")
        assert "y" in self._field_names(df, "child")

    def test_parent_is_not_affected_by_child(self):
        df = self._call(
            [_field_row("parent", "x"), _field_row("child", "y")],
            [_parent_row("child", "parent")],
        )
        assert self._field_names(df, "parent") == {"x"}

    def test_child_overrides_parent_field(self):
        df = self._call(
            [_field_row("parent", "x", "str"), _field_row("child", "x", "bool")],
            [_parent_row("child", "parent")],
        )
        row = df[(df["entity_id"] == "child") & (df["value"] == "x")]
        assert len(row) == 1
        assert row.iloc[0]["type"] == "bool"

    def test_multi_level_inheritance(self):
        df = self._call(
            [_field_row("gp", "a"), _field_row("p", "b"), _field_row("c", "c_field")],
            [_parent_row("p", "gp"), _parent_row("c", "p")],
        )
        assert self._field_names(df, "c") == {"a", "b", "c_field"}

    def test_entity_with_no_own_fields_inherits_all_parent_fields(self):
        df = self._call(
            [_field_row("parent", "x"), _field_row("parent", "y", component_index=1)],
            [_parent_row("child", "parent")],
        )
        assert self._field_names(df, "child") == {"x", "y"}

    def test_no_duplicate_field_names_per_entity(self):
        df = self._call(
            [_field_row("parent", "x"), _field_row("child", "y")],
            [_parent_row("child", "parent")],
        )
        child_rows = df[df["entity_id"] == "child"]
        assert len(child_rows) == child_rows["value"].nunique()

    def test_cycle_in_parent_hierarchy_does_not_hang(self):
        """A cycle in parent links should not cause infinite loops."""
        df = self._call(
            [_field_row("a", "x")],
            [_parent_row("a", "b"), _parent_row("b", "a")],
        )
        assert "x" in self._field_names(df, "a")
