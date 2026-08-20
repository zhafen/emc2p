"""Tests for time_filled_registry in the derive_components DAG."""

import pytest

from emc2p.dataflows.derive.derive_components import time_filled_registry
from tests.conftest import make_registry


def _status_reading_registry():
    return make_registry({
        "entity_id": [{"value": "def1", "entity_key": "status_reading"}],
        "field": [
            {"entity_id": "def1", "component_index": 0, "value": "as_of", "time_dimension": True},
            {"entity_id": "def1", "component_index": 0, "value": "status", "time_dimension": False},
        ],
        "status_reading": [
            {"entity_id": "e1", "component_index": 0, "as_of": None, "status": "open"},
            {"entity_id": "e2", "component_index": 0, "as_of": "2024-01-01", "status": "closed"},
        ],
    })


class TestTimeFilledRegistry:
    """Tests for time_filled_registry, which backfills time_dimension fields."""

    def test_no_load_time_is_noop(self):
        registry = _status_reading_registry()
        result = time_filled_registry(registry, load_time=None)
        assert result is registry

    def test_fills_null_time_dimension_values(self):
        registry = _status_reading_registry()
        result = time_filled_registry(registry, load_time="2024-12-25")
        df = result._components["status_reading"].execute()
        assert df.set_index("entity_id").loc["e1", "as_of"] == "2024-12-25"

    def test_does_not_overwrite_existing_values(self):
        registry = _status_reading_registry()
        result = time_filled_registry(registry, load_time="2024-12-25")
        df = result._components["status_reading"].execute()
        assert df.set_index("entity_id").loc["e2", "as_of"] == "2024-01-01"

    def test_leaves_non_time_dimension_fields_untouched(self):
        registry = _status_reading_registry()
        result = time_filled_registry(registry, load_time="2024-12-25")
        df = result._components["status_reading"].execute()
        assert df.set_index("entity_id").loc["e1", "status"] == "open"

    def test_no_time_dimension_column_is_noop(self):
        registry = make_registry({
            "entity_id": [{"value": "def1", "entity_key": "status_reading"}],
            "field": [{"entity_id": "def1", "component_index": 0, "value": "as_of"}],
            "status_reading": [
                {"entity_id": "e1", "component_index": 0, "as_of": None, "status": "open"},
                {"entity_id": "e2", "component_index": 0, "as_of": "2024-01-01", "status": "closed"},
            ],
        })
        result = time_filled_registry(registry, load_time="2024-12-25")
        assert result is registry

    def test_falls_back_to_existing_registry_schema_when_batch_omits_it(self):
        """A component type's time_dimension field can have been declared in
        an earlier update() call rather than this batch's own -- e.g. a
        scenario-local field only ever loaded once, at world-creation time,
        not re-included in every later incremental write."""
        existing_registry = _status_reading_registry()
        batch = make_registry({
            "entity_id": [{"value": "other_def", "entity_key": "other_type"}],
            "field": [
                {"entity_id": "other_def", "component_index": 0, "value": "note", "time_dimension": False},
            ],
            "status_reading": [
                {"entity_id": "e3", "component_index": 0, "as_of": None, "status": "open"},
                {"entity_id": "e4", "component_index": 0, "as_of": "2024-02-02", "status": "closed"},
            ],
        })
        result = time_filled_registry(batch, load_time="2024-12-25", existing_registry=existing_registry)
        df = result._components["status_reading"].execute()
        assert df.set_index("entity_id").loc["e3", "as_of"] == "2024-12-25"
        assert df.set_index("entity_id").loc["e4", "as_of"] == "2024-02-02"

    def test_existing_registry_without_schema_tables_does_not_error(self):
        """A brand-new existing_registry (no prior update() calls yet) has no
        "field"/"entity_id" tables at all -- must not be treated as knowing
        about every component type's schema."""
        existing_registry = make_registry({"status_reading": [{"entity_id": "e9"}]})
        registry = _status_reading_registry()
        result = time_filled_registry(registry, load_time="2024-12-25", existing_registry=existing_registry)
        df = result._components["status_reading"].execute()
        assert df.set_index("entity_id").loc["e1", "as_of"] == "2024-12-25"

    def test_multiple_time_dimension_fields_raises(self):
        registry = make_registry({
            "entity_id": [{"value": "def1", "entity_key": "status_reading"}],
            "field": [
                {"entity_id": "def1", "component_index": 0, "value": "as_of", "time_dimension": True},
                {"entity_id": "def1", "component_index": 0, "value": "also_as_of", "time_dimension": True},
            ],
            "status_reading": [
                {"entity_id": "e1", "component_index": 0, "as_of": None, "status": "open"},
                {"entity_id": "e2", "component_index": 0, "as_of": "2024-01-01", "status": "closed"},
            ],
        })
        with pytest.raises(ValueError, match="status_reading"):
            time_filled_registry(registry, load_time="2024-12-25")
