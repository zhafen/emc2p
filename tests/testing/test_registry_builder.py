"""Unit tests for emc2p.testing.registry_builder.make_registry.

The pandas.testing.assert_allclose monkeypatch this module applies at
import time is exercised implicitly across this whole suite (any test
using pandas.testing.assert_allclose on a Registry-derived table) rather
than tested directly here.
"""

from emc2p.testing.registry_builder import make_registry


class TestMakeRegistry:
    def test_builds_a_registry_with_the_given_component_tables(self):
        registry = make_registry({
            "description": [{"entity_id": "e1", "value": "a thing"}],
        })
        df = registry.get("description").to_pandas()
        assert df["value"].tolist() == ["a thing"]

    def test_supports_multiple_component_types(self):
        registry = make_registry({
            "description": [{"entity_id": "e1", "value": "a thing"}],
            "requirement": [{"entity_id": "e1"}],
        })
        assert set(registry.component_types) >= {"description", "requirement"}
