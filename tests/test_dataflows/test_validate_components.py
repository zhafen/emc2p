"""Tests for the validate_components Hamilton DAG functions."""

import ibis
import pandas as pd

import emc2p.dataflows.validation.validate_components as validate_components


_EMPTY_FIELD_COLS = ["entity_id", "component_index", "value", "type", "nullable", "default", "range"]
_EMPTY_ENTITY_ID_COLS = ["value", "entity_key", "path", "alias"]


def _make_entity_id_table(rows: list[dict]) -> ibis.Table:
    if not rows:
        return ibis.memtable(pd.DataFrame(columns=_EMPTY_ENTITY_ID_COLS).astype(str))
    return ibis.memtable(pd.DataFrame(rows))


def _make_field_table(rows: list[dict]) -> ibis.Table:
    if not rows:
        return ibis.memtable(pd.DataFrame(columns=_EMPTY_FIELD_COLS).astype(str))
    return ibis.memtable(pd.DataFrame(rows))


def _make_component_table(rows: list[dict]) -> ibis.Table:
    return ibis.memtable(pd.DataFrame(rows))


def _call(components, field_rows, entity_id_rows):
    validated_field = _make_field_table(field_rows)
    entity_id = _make_entity_id_table(entity_id_rows)
    validated, invalid, _ = validate_components.validation_results(components, validated_field, entity_id)
    return validated, invalid


class TestValidationResults:

    def _entity_id_row(self, entity_id, entity_key):
        return {"value": entity_id, "entity_key": entity_key, "path": f"test:{entity_key}", "alias": entity_key}

    def _field_row(self, entity_id, field_name, field_type=None, nullable=None, default=None, field_range=None):
        return {
            "entity_id": entity_id,
            "component_index": 0,
            "value": field_name,
            "type": field_type,
            "nullable": nullable,
            "default": default,
            "range": field_range,
        }

    def test_returns_tuple(self):
        components = {"desc": _make_component_table([{"entity_id": "e1", "component_index": 0, "value": "hello"}])}
        validated, invalid = _call(components, [], [])
        assert isinstance(validated, dict)
        assert isinstance(invalid, ibis.Table)

    def test_no_schema_component_passes_through(self):
        """Components with no field definitions pass through unchanged."""
        rows = [{"entity_id": "e1", "component_index": 0, "value": "hello"}]
        components = {"desc": _make_component_table(rows)}
        validated, _ = _call(components, [], [])
        assert "desc" in validated
        df = validated["desc"].execute()
        assert df.iloc[0]["value"] == "hello"

    def test_str_field_coercion(self):
        components = {"mycomp": _make_component_table([
            {"entity_id": "e1", "component_index": 0, "name": "Alice"},
        ])}
        field_rows = [self._field_row("eid_mycomp", "name", field_type="str")]
        entity_id_rows = [self._entity_id_row("eid_mycomp", "mycomp")]
        validated, _ = _call(components, field_rows, entity_id_rows)
        df = validated["mycomp"].execute()
        assert df.iloc[0]["name"] == "Alice"

    def test_float_field_cast_from_string(self):
        components = {"scores": _make_component_table([
            {"entity_id": "e1", "component_index": 0, "score": "3.14"},
        ])}
        field_rows = [self._field_row("eid_scores", "score", field_type="float")]
        entity_id_rows = [self._entity_id_row("eid_scores", "scores")]
        validated, _ = _call(components, field_rows, entity_id_rows)
        df = validated["scores"].execute()
        assert abs(df.iloc[0]["score"] - 3.14) < 1e-6

    def test_bool_field_cast_from_string(self):
        components = {"flags": _make_component_table([
            {"entity_id": "e1", "component_index": 0, "active": "True"},
            {"entity_id": "e2", "component_index": 0, "active": "False"},
        ])}
        field_rows = [self._field_row("eid_flags", "active", field_type="bool")]
        entity_id_rows = [self._entity_id_row("eid_flags", "flags")]
        validated, _ = _call(components, field_rows, entity_id_rows)
        df = validated["flags"].execute().sort_values("entity_id").reset_index(drop=True)
        assert df.iloc[0]["active"] is True or df.iloc[0]["active"] == True
        assert df.iloc[1]["active"] is False or df.iloc[1]["active"] == False

    def test_missing_column_added_as_null(self):
        """A typed column missing from the component table is added as null."""
        components = {"things": _make_component_table([
            {"entity_id": "e1", "component_index": 0},
        ])}
        field_rows = [self._field_row("eid_things", "score", field_type="float")]
        entity_id_rows = [self._entity_id_row("eid_things", "things")]
        validated, _ = _call(components, field_rows, entity_id_rows)
        df = validated["things"].execute()
        assert "score" in df.columns
        assert pd.isna(df.iloc[0]["score"])

    def test_default_applied_when_null(self):
        """Default values are filled in for null fields."""
        components = {"items": _make_component_table([
            {"entity_id": "e1", "component_index": 0, "priority": None},
        ])}
        field_rows = [self._field_row("eid_items", "priority", field_type="float", default=0.5)]
        entity_id_rows = [self._entity_id_row("eid_items", "items")]
        validated, _ = _call(components, field_rows, entity_id_rows)
        df = validated["items"].execute()
        assert abs(df.iloc[0]["priority"] - 0.5) < 1e-6

    def test_nullable_violation_collected(self):
        """Null values in non-nullable fields appear in invalid_field."""
        components = {"reqs": _make_component_table([
            {"entity_id": "e1", "component_index": 1, "value": None},
        ])}
        field_rows = [self._field_row("eid_reqs", "value", field_type="float", nullable=False)]
        entity_id_rows = [self._entity_id_row("eid_reqs", "reqs")]
        _, invalid = _call(components, field_rows, entity_id_rows)
        df = invalid.execute()
        assert len(df) == 1
        assert df.iloc[0]["component_type"] == "reqs"
        assert df.iloc[0]["field"] == "value"
        assert df.iloc[0]["error_type"] == "nullable"

    def test_range_violation_collected(self):
        """Values outside a categorical range appear in invalid_field."""
        components = {"food": _make_component_table([
            {"entity_id": "e1", "component_index": 1, "type": "cosmic_horror"},
        ])}
        field_rows = [self._field_row("eid_food", "type", field_type="str", nullable=True, field_range=["wet", "dry"])]
        entity_id_rows = [self._entity_id_row("eid_food", "food")]
        _, invalid = _call(components, field_rows, entity_id_rows)
        df = invalid.execute()
        assert len(df) == 1
        assert df.iloc[0]["error_type"] == "range"
        assert df.iloc[0]["value"] == "cosmic_horror"

    def test_valid_range_produces_no_violation(self):
        """Values within a categorical range produce no violations."""
        components = {"food": _make_component_table([
            {"entity_id": "e1", "component_index": 1, "type": "wet"},
        ])}
        field_rows = [self._field_row("eid_food", "type", field_type="str", nullable=True, field_range=["wet", "dry"])]
        entity_id_rows = [self._entity_id_row("eid_food", "food")]
        _, invalid = _call(components, field_rows, entity_id_rows)
        df = invalid.execute()
        assert len(df) == 0

    def test_no_violations_returns_empty_invalid_field(self):
        """When there are no violations, invalid_field is an empty table with correct schema."""
        components = {"desc": _make_component_table([{"entity_id": "e1", "component_index": 0, "value": "ok"}])}
        _, invalid = _call(components, [], [])
        df = invalid.execute()
        assert df.empty
        assert set(df.columns) >= {"entity_id", "component_index", "component_type", "field", "value", "error_type"}

    def test_all_rows_preserved(self):
        """Validation does not drop any rows from the component table."""
        rows = [
            {"entity_id": f"e{i}", "component_index": i, "name": f"item{i}"}
            for i in range(5)
        ]
        components = {"items": _make_component_table(rows)}
        field_rows = [self._field_row("eid_items", "name", field_type="str")]
        entity_id_rows = [self._entity_id_row("eid_items", "items")]
        validated, _ = _call(components, field_rows, entity_id_rows)
        df = validated["items"].execute()
        assert len(df) == 5

    def test_multiple_components_validated_independently(self):
        """Each component type uses its own field schema."""
        components = {
            "cats": _make_component_table([{"entity_id": "e1", "component_index": 0, "name": "Felix"}]),
            "dogs": _make_component_table([{"entity_id": "e2", "component_index": 0, "breed": "Poodle"}]),
        }
        field_rows = [
            self._field_row("eid_cats", "name", field_type="str"),
            self._field_row("eid_dogs", "breed", field_type="str"),
        ]
        entity_id_rows = [
            self._entity_id_row("eid_cats", "cats"),
            self._entity_id_row("eid_dogs", "dogs"),
        ]
        validated, invalid = _call(components, field_rows, entity_id_rows)
        assert "cats" in validated
        assert "dogs" in validated
        assert invalid.execute().empty

    def test_empty_string_treated_as_null_for_default(self):
        """Empty strings in str fields are treated as null before applying defaults."""
        components = {"items": _make_component_table([
            {"entity_id": "e1", "component_index": 0, "label": ""},
        ])}
        field_rows = [self._field_row("eid_items", "label", field_type="str", default="unknown")]
        entity_id_rows = [self._entity_id_row("eid_items", "items")]
        validated, _ = _call(components, field_rows, entity_id_rows)
        df = validated["items"].execute()
        assert df.iloc[0]["label"] == "unknown"

    def test_multiple_violations_unioned(self):
        """Multiple violations across fields are all collected in invalid_field."""
        components = {"reqs": _make_component_table([
            {"entity_id": "e1", "component_index": 1, "value": None, "type": "unknown"},
        ])}
        field_rows = [
            self._field_row("eid_reqs", "value", field_type="float", nullable=False),
            self._field_row("eid_reqs", "type", field_type="str", nullable=True, field_range=["functional", "quality"]),
        ]
        entity_id_rows = [self._entity_id_row("eid_reqs", "reqs")]
        _, invalid = _call(components, field_rows, entity_id_rows)
        df = invalid.execute()
        assert len(df) == 2
        assert set(df["error_type"]) == {"nullable", "range"}

    def test_unrecognized_entity_key_ignored(self):
        """Components without matching field definitions are passed through."""
        components = {"orphan": _make_component_table([{"entity_id": "e1", "component_index": 0, "value": "x"}])}
        field_rows = [self._field_row("eid_other", "value", field_type="str")]
        entity_id_rows = [self._entity_id_row("eid_other", "other")]  # "other" != "orphan"
        validated, invalid = _call(components, field_rows, entity_id_rows)
        assert "orphan" in validated
        df = validated["orphan"].execute()
        assert df.iloc[0]["value"] == "x"
        assert invalid.execute().empty

    def test_formerly_infra_type_is_validated(self):
        """Infrastructure component types (e.g. "parent") are validated
        like any other component, since users can add invalid records to
        them too.
        """
        components = {"parent": _make_component_table([
            {"entity_id": "e1", "component_index": 0, "parent_eid": None},
        ])}
        field_rows = [self._field_row("eid_parent", "parent_eid", field_type="str", nullable=False)]
        entity_id_rows = [self._entity_id_row("eid_parent", "parent")]
        _, invalid = _call(components, field_rows, entity_id_rows)
        df = invalid.execute()
        assert len(df) == 1
        assert df.iloc[0]["component_type"] == "parent"
        assert df.iloc[0]["error_type"] == "nullable"

    def test_field_itself_not_double_validated(self):
        """"field" in the result is folded straight in from the validated_field
        argument, not re-derived from components["field"] a second time here."""
        components = {"field": _make_field_table([
            {"entity_id": "eid_x", "component_index": 0, "value": "raw_unvalidated_row"},
        ])}
        field_rows = [self._field_row("eid_x", "already_validated_row")]
        validated, _ = _call(components, field_rows, [])
        df = validated["field"].execute()
        assert len(df) == 1
        assert df.iloc[0]["value"] == "already_validated_row"


class TestValidationResultsDeclaredSchemas:
    """Tests for the declared_schemas output: schemas for component types
    declared via a `component_type` tag but with no data rows this batch."""

    def _entity_id_row(self, entity_id, entity_key):
        return {"value": entity_id, "entity_key": entity_key, "path": f"test:{entity_key}", "alias": entity_key}

    def _field_row(self, entity_id, field_name, field_type=None, nullable=None, default=None, field_range=None):
        return {
            "entity_id": entity_id,
            "component_index": 0,
            "value": field_name,
            "type": field_type,
            "nullable": nullable,
            "default": default,
            "range": field_range,
        }

    def _component_type_row(self, entity_id):
        return {"entity_id": entity_id, "component_index": 0, "value": None}

    def _call(self, components, field_rows, entity_id_rows):
        validated_field = _make_field_table(field_rows)
        entity_id = _make_entity_id_table(entity_id_rows)
        return validate_components.validation_results(components, validated_field, entity_id)

    def test_dataless_typed_component_gets_its_own_field_schema(self):
        """A component_type-tagged type with declared fields but no rows
        this batch gets a schema with its real, typed field columns.

        Not the generic fallback a fieldless tag type gets.
        """
        components = {
            "component_type": _make_component_table([self._component_type_row("eid_widget")]),
        }
        field_rows = [
            self._field_row("eid_widget", "width", field_type="float"),
            self._field_row("eid_widget", "height", field_type="float"),
        ]
        entity_id_rows = [self._entity_id_row("eid_widget", "widget")]
        _, _, declared_schemas = self._call(components, field_rows, entity_id_rows)
        assert "widget" in declared_schemas
        schema = declared_schemas["widget"]
        assert schema["width"] == ibis.dtype("float64")
        assert schema["height"] == ibis.dtype("float64")
        assert "value" not in schema

    def test_dataless_fieldless_component_gets_generic_value_column(self):
        """A component_type-tagged type with no field definitions at all
        (a bare tag) falls back to a generic `value` column.

        Matching what the loader gives such a type when it does have rows.
        """
        components = {
            "component_type": _make_component_table([self._component_type_row("eid_marker")]),
        }
        entity_id_rows = [self._entity_id_row("eid_marker", "marker")]
        _, _, declared_schemas = self._call(components, [], entity_id_rows)
        assert "marker" in declared_schemas
        assert set(declared_schemas["marker"].names) == {"entity_id", "component_index", "modifier", "value"}

    def test_component_type_with_data_this_batch_not_declared(self):
        """A component type that already has rows this batch is skipped --
        declared_schemas is only for types with no data of their own yet."""
        components = {
            "component_type": _make_component_table([self._component_type_row("eid_marker")]),
            "marker": _make_component_table([{"entity_id": "e1", "component_index": 0, "value": None}]),
        }
        entity_id_rows = [self._entity_id_row("eid_marker", "marker")]
        _, _, declared_schemas = self._call(components, [], entity_id_rows)
        assert "marker" not in declared_schemas

    def test_no_component_type_table_returns_no_declared_schemas(self):
        """Without a component_type table at all (e.g. a bare validation
        call), there's nothing to declare."""
        components = {"desc": _make_component_table([{"entity_id": "e1", "component_index": 0, "value": "hi"}])}
        _, _, declared_schemas = self._call(components, [], [])
        assert declared_schemas == {}


class TestFieldValidationResults:
    """Tests for field_validation_results, which validates ((field))
    against its own self-referential schema, before field is used to
    validate every other component.
    """

    def _entity_id_row(self, entity_id, entity_key):
        return {"value": entity_id, "entity_key": entity_key, "path": f"test:{entity_key}", "alias": entity_key}

    def _meta_row(self, entity_id, attr_name, field_type=None, nullable=None, default=None, field_range=None, **attrs):
        """A field row defining one of field's own meta-attributes (attached to the entity that defines "field")."""
        return {
            "entity_id": entity_id,
            "component_index": 0,
            "value": attr_name,
            "type": field_type,
            "nullable": nullable,
            "default": default,
            "range": field_range,
            **attrs,
        }

    def _instance_row(self, entity_id, field_name, **attrs):
        """A regular field row (defines a field for some other component type), with meta-attribute values to validate."""
        return {"entity_id": entity_id, "component_index": 0, "value": field_name, **attrs}

    def _call(self, field_rows, entity_id_rows):
        field = _make_field_table(field_rows)
        entity_id = _make_entity_id_table(entity_id_rows)
        components = {"field": field}
        return validate_components.field_validation_results(components, field, entity_id)

    def test_returns_tuple(self):
        entity_id_rows = [self._entity_id_row("eid_field", "field")]
        field_rows = [self._meta_row("eid_field", "value", field_type="str")]
        validated_field, invalid = self._call(field_rows, entity_id_rows)
        assert isinstance(validated_field, ibis.Table)
        assert isinstance(invalid, ibis.Table)

    def test_no_self_schema_passes_through_unchanged(self):
        """When field has no schema defined for itself, field passes through as-is."""
        field_rows = [self._meta_row("eid_other", "name", field_type="str")]
        entity_id_rows = [self._entity_id_row("eid_other", "other")]  # entity_key != "field"
        validated_field, invalid = self._call(field_rows, entity_id_rows)
        df = validated_field.execute()
        assert len(df) == 1
        assert invalid.execute().empty

    def test_coerces_own_bool_attribute_on_every_row(self):
        """field's own bool-typed meta-attribute is coerced for every row of field,
        not just the rows that define the meta-schema."""
        entity_id_rows = [self._entity_id_row("eid_field", "field")]
        field_rows = [
            self._meta_row("eid_field", "time_dimension", field_type="bool", default=False),
            self._instance_row("eid_status_reading", "as_of", time_dimension="True"),
            self._instance_row("eid_status_reading", "status", time_dimension="False"),
        ]
        validated_field, invalid = self._call(field_rows, entity_id_rows)
        df = validated_field.execute().set_index("value")
        assert bool(df.loc["as_of", "time_dimension"]) is True
        assert bool(df.loc["status", "time_dimension"]) is False
        assert invalid.execute().empty

    def test_missing_bool_attribute_gets_default(self):
        """A row that never set the meta-attribute is filled with its declared default."""
        entity_id_rows = [self._entity_id_row("eid_field", "field")]
        field_rows = [
            self._meta_row("eid_field", "time_dimension", field_type="bool", default=False),
            self._instance_row("eid_status_reading", "status"),
        ]
        validated_field, _ = self._call(field_rows, entity_id_rows)
        df = validated_field.execute().set_index("value")
        assert bool(df.loc["status", "time_dimension"]) is False

    def test_nullable_violation_collected(self):
        """A null meta-attribute with no default and nullable=False is reported."""
        entity_id_rows = [self._entity_id_row("eid_field", "field")]
        field_rows = [
            self._meta_row("eid_field", "units", field_type="str", nullable=False, units="n/a"),
            self._instance_row("eid_status_reading", "as_of"),
        ]
        _, invalid = self._call(field_rows, entity_id_rows)
        df = invalid.execute()
        assert len(df) == 1
        assert df.iloc[0]["component_type"] == "field"
        assert df.iloc[0]["field"] == "units"
        assert df.iloc[0]["error_type"] == "nullable"

    def test_range_violation_collected(self):
        """A meta-attribute value outside its declared range is reported."""
        entity_id_rows = [self._entity_id_row("eid_field", "field")]
        field_rows = [
            self._meta_row("eid_field", "format", field_type="str", field_range=["str", "epoch_time"], format="str"),
            self._instance_row("eid_status_reading", "as_of", format="unparseable"),
        ]
        _, invalid = self._call(field_rows, entity_id_rows)
        df = invalid.execute()
        assert len(df) == 1
        assert df.iloc[0]["error_type"] == "range"
        assert df.iloc[0]["value"] == "unparseable"
