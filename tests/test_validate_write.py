"""Tests for emc2p.validate_write's mechanical write-safety checks."""

import emc2p.validate_write as validate_write


# ---------------------------------------------------------------------------
# referenced_component_types
# ---------------------------------------------------------------------------

class TestReferencedComponentTypes:

    def test_bare_string_marker(self):
        parsed = {"my_type": ["component_type"]}
        assert validate_write.referenced_component_types(parsed) == {"component_type"}

    def test_one_key_dict_component(self):
        parsed = {"car_a": [{"location": {"value": "spot_1"}}]}
        assert validate_write.referenced_component_types(parsed) == {"location"}

    def test_nested_entity_is_recursed_into(self):
        parsed = {"desk": {"drawer": [{"object": {"type": "container"}}]}}
        assert validate_write.referenced_component_types(parsed) == {"object"}

    def test_data_key_is_exempt(self):
        parsed = {"my_type": {"data": "some freeform note"}}
        assert validate_write.referenced_component_types(parsed) == set()


# ---------------------------------------------------------------------------
# component_types_defined_inline
# ---------------------------------------------------------------------------

class TestComponentTypesDefinedInline:

    def test_entity_with_component_type_marker_is_defined_inline(self):
        parsed = {"my_type": ["component_type", {"description": "A thing."}]}
        assert validate_write.component_types_defined_inline(parsed) == {"my_type"}

    def test_entity_without_marker_is_not_defined_inline(self):
        parsed = {"car_a": [{"location": {"value": "spot_1"}}]}
        assert validate_write.component_types_defined_inline(parsed) == set()


# ---------------------------------------------------------------------------
# find_component_named_as_nested_entity
# ---------------------------------------------------------------------------

class TestFindComponentNamedAsNestedEntity:

    def test_known_component_name_as_dict_key_is_flagged(self):
        yaml_string = "car_b:\n    location:\n        - value: parked in parking_spot_1\n"
        result = validate_write.find_component_named_as_nested_entity(yaml_string, {"location"})
        assert result == ["car_b.location"]

    def test_same_shape_as_a_list_item_is_not_flagged(self):
        """The correctly-formed form for the very same component (`location`
        as a list item, not a nested dict key) must not be flagged."""
        yaml_string = "car_b:\n    - object:\n        type: car\n    - location:\n        value: parked in parking_spot_1\n"
        result = validate_write.find_component_named_as_nested_entity(yaml_string, {"location", "object"})
        assert result == []

    def test_legitimate_nested_entity_not_named_after_a_component_is_not_flagged(self):
        yaml_string = "desk:\n    - object:\n        type: desk\n    journal:\n        - object:\n            type: journal\n"
        result = validate_write.find_component_named_as_nested_entity(yaml_string, {"object"})
        assert result == []

    def test_invalid_yaml_returns_empty(self):
        assert validate_write.find_component_named_as_nested_entity("not: [valid: yaml", {"location"}) == []


# ---------------------------------------------------------------------------
# find_unknown_component_types
# ---------------------------------------------------------------------------

class TestFindUnknownComponentTypes:

    def test_unknown_type_is_flagged_with_suggestions(self):
        yaml_string = "car_a:\n    - locaton:\n        value: parked on street\n"
        result = validate_write.find_unknown_component_types(yaml_string, {"location", "object"})
        assert "locaton" in result
        assert "location" in result["locaton"]

    def test_known_type_is_not_flagged(self):
        yaml_string = "car_a:\n    - location:\n        value: parked on street\n"
        result = validate_write.find_unknown_component_types(yaml_string, {"location"})
        assert result == {}

    def test_inline_defined_type_is_not_flagged_as_unknown(self):
        yaml_string = (
            "brand_new_component:\n"
            "    - description: A freshly-declared component type.\n"
            "    - component_type\n"
            "car_a:\n"
            "    - brand_new_component:\n"
            "        value: hello\n"
        )
        # `component_type` is itself a referenced marker in this YAML (the
        # declaration shape itself), so a realistic `known` set must
        # include it too -- only `brand_new_component` is the thing
        # actually being defined inline here.
        result = validate_write.find_unknown_component_types(
            yaml_string, {"description", "component_type"}
        )
        assert result == {}
