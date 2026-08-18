"""Tests for whitespace stripping in the derive_components DAG."""

import pandas as pd
import ibis

from emc2p.dataflows.derive.derive_components import stripped_registry
from tests.conftest import make_registry


def _make_reg_with_field_metadata(comp_type: str, field_name: str, field_entity_id: str):
    """Build a registry with a field row declaring type 'description' for a component."""
    return make_registry({
        "field": [
            {"entity_id": field_entity_id, "value": field_name, "type": "description"},
        ],
        "entity_id": [
            {"value": field_entity_id, "entity_key": comp_type},
        ],
        comp_type: [
            {"entity_id": "e1", field_name: "  padded  ", "other": "  untouched  "},
        ],
    })


class TestStrippedRegistry:

    def test_strips_description_value_column(self):
        reg = make_registry({
            "description": [
                {"entity_id": "e1", "value": "  hello world  \n"},
                {"entity_id": "e2", "value": "\ttabbed\t"},
            ],
        })
        result = stripped_registry(reg)
        df = result._components["description"].to_pandas()
        assert df.loc[df["entity_id"] == "e1", "value"].iloc[0] == "hello world"
        assert df.loc[df["entity_id"] == "e2", "value"].iloc[0] == "tabbed"

    def test_no_description_component_does_not_error(self):
        from emc2p.registry import Registry
        reg = make_registry({})
        result = stripped_registry(reg)
        assert isinstance(result, Registry)

    def test_strips_description_typed_field_in_other_component(self):
        reg = _make_reg_with_field_metadata("mycomp", "summary", "eid_mycomp")
        reg._components["mycomp"] = ibis.memtable(
            pd.DataFrame([{"entity_id": "e1", "summary": "  padded  ", "other": "  untouched  "}])
        )
        result = stripped_registry(reg)
        df = result._components["mycomp"].to_pandas()
        assert df.loc[0, "summary"] == "padded"
        assert df.loc[0, "other"] == "  untouched  "

    def test_non_string_values_are_unchanged(self):
        reg = make_registry({
            "description": [
                {"entity_id": "e1", "value": "  text  "},
                {"entity_id": "e2", "value": None},
            ],
        })
        result = stripped_registry(reg)
        df = result._components["description"].to_pandas()
        assert df.loc[df["entity_id"] == "e1", "value"].iloc[0] == "text"
        assert pd.isna(df.loc[df["entity_id"] == "e2", "value"].iloc[0])

    def test_no_field_or_entity_id_table_does_not_error(self):
        reg = make_registry({
            "description": [{"entity_id": "e1", "value": "  text  "}],
        })
        result = stripped_registry(reg)
        df = result._components["description"].to_pandas()
        assert df.loc[0, "value"] == "text"
