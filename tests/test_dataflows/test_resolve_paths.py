"""Tests for emc2p.dataflows.derive.resolve_paths."""

from emc2p.dataflows.derive.resolve_paths import (
    components_with_resolved_paths,
    fields_of_type_entity_ref,
    implicit_parent_target_types,
    parent_from_hierarchy,
)
import pandas as pd

from emc2p.registry import Registry
from emc2p.utils import dhash

PARENT_PATH = "example.yaml:the_parent"
CHILD_PATH = "example.yaml:the_parent.the_child"
OTHER_PATH = "example.yaml:something_else"
PARENT_EID = dhash(PARENT_PATH)
CHILD_EID = dhash(CHILD_PATH)
OTHER_EID = dhash(OTHER_PATH)


def _entity_id_rows(*, other_entity=False):
    rows = [
        {"value": PARENT_EID, "path": PARENT_PATH, "entity_key": "the_parent"},
        {"value": CHILD_EID, "path": CHILD_PATH, "entity_key": "the_child"},
        # Component-type definition entities, so fields_of_type_entity_ref
        # can map their "field" rows back to a component_type name.
        {"value": "def_requirement", "path": "builtins.yaml:requirement", "entity_key": "requirement"},
        {"value": "def_solution", "path": "builtins.yaml:solution", "entity_key": "solution"},
        {"value": "def_calls", "path": "builtins.yaml:calls", "entity_key": "calls"},
    ]
    if other_entity:
        rows.append({"value": OTHER_EID, "path": OTHER_PATH, "entity_key": "something_else"})
    return rows


def _field_rows(*type_names):
    return [
        {"entity_id": f"def_{name}", "component_index": 0, "value": "value", "type": "entity_ref"}
        for name in type_names
    ]


def _component_type_rows(*flagged_names, **explicit_flags):
    """Rows for the component_type table declaring each named type's own
    ``implicit_parent`` flag, matching requirement/solution's own
    ``- component_type: {implicit_parent: true}`` declaration in
    auditing.yaml.

    ``flagged_names`` is shorthand for ``implicit_parent=True`` (the
    common case); ``explicit_flags`` lets a test spell out ``False``
    too (e.g. ``_component_type_rows(calls=False)``) -- needed because
    real ``load_manifest`` output stamps ``implicit_parent`` (True or
    False) onto *every* component_type row, not just flagged ones (see
    ``component_type_table``), so a fixture representing "this type has
    no implicit-parent fallback" should include an explicit False row
    rather than omitting the component_type table entirely, which would
    leave it without the ``implicit_parent`` column at all.

    The row's own ``component_type`` column is always the literal string
    "component_type" here, matching real load_manifest.py output: it's an
    instance of the component_type component type itself (the schema
    entity's ``- component_type: {...}`` tag), not an instance of the
    flagged type. ``implicit_parent_target_types`` resolves the flagged
    type's real name via the owning entity's own entity_key in the
    entity_id table (see ``_entity_id_rows``'s ``def_{name}`` -> ``{name}``
    mapping), not from this column.
    """
    flags = {name: True for name in flagged_names}
    flags.update(explicit_flags)
    return [
        {"entity_id": f"def_{name}", "component_index": 0, "component_type": "component_type", "implicit_parent": flag}
        for name, flag in flags.items()
    ]


def _resolve(registry):
    entity_id = registry._components["entity_id"]
    field = registry._components["field"]
    hierarchy = parent_from_hierarchy(entity_id)
    return components_with_resolved_paths(
        entity_id=entity_id,
        components=registry._components,
        fields_of_type_entity_ref=fields_of_type_entity_ref(entity_id, field),
        parent_from_hierarchy=hierarchy,
        implicit_parent_target_types=implicit_parent_target_types(registry._components),
    )


class TestComponentsWithResolvedPaths:
    def test_bare_requirement_defaults_to_parent(self):
        registry = Registry.from_component_rows({
            "entity_id": _entity_id_rows(other_entity=True),
            "field": _field_rows("requirement"),
            "component_type": _component_type_rows("requirement"),
            "requirement": [
                {"entity_id": CHILD_EID, "component_index": 0, "value": None},
                # Non-null sibling row, so pandas/duckdb infers a real
                # string dtype for "value" instead of an all-NULL column.
                {"entity_id": OTHER_EID, "component_index": 0, "value": "something_else"},
            ],
        })
        df = _resolve(registry)["requirement"].to_pandas()
        assert df.set_index("entity_id").loc[CHILD_EID, "value_eid"] == PARENT_EID

    def test_empty_string_requirement_defaults_to_parent(self):
        """The real entity-first manifest loader represents a bare (no
        ``of:`` target) tag's value as ``""``, not ``None``; empty string
        must trigger the same implicit-parent fallback as an actual
        null."""
        registry = Registry.from_component_rows({
            "entity_id": _entity_id_rows(other_entity=True),
            "field": _field_rows("requirement"),
            "component_type": _component_type_rows("requirement"),
            "requirement": [
                {"entity_id": CHILD_EID, "component_index": 0, "value": ""},
                {"entity_id": OTHER_EID, "component_index": 0, "value": "something_else"},
            ],
        })
        df = _resolve(registry)["requirement"].to_pandas()
        assert df.set_index("entity_id").loc[CHILD_EID, "value_eid"] == PARENT_EID

    def test_bare_solution_defaults_to_parent(self):
        registry = Registry.from_component_rows({
            "entity_id": _entity_id_rows(other_entity=True),
            "field": _field_rows("solution"),
            "component_type": _component_type_rows("solution"),
            "solution": [
                {"entity_id": CHILD_EID, "component_index": 0, "value": None},
                {"entity_id": OTHER_EID, "component_index": 0, "value": "something_else"},
            ],
        })
        df = _resolve(registry)["solution"].to_pandas()
        assert df.set_index("entity_id").loc[CHILD_EID, "value_eid"] == PARENT_EID

    def test_explicit_requirement_target_is_not_overridden(self):
        registry = Registry.from_component_rows({
            "entity_id": _entity_id_rows(other_entity=True),
            "field": _field_rows("requirement"),
            "component_type": _component_type_rows("requirement"),
            "requirement": [
                {"entity_id": CHILD_EID, "component_index": 0, "value": "something_else"},
            ],
        })
        df = _resolve(registry)["requirement"].to_pandas()
        assert df.set_index("entity_id").loc[CHILD_EID, "value_eid"] == OTHER_EID

    def test_bare_field_on_unscoped_type_stays_unresolved(self):
        """Only requirement/solution get the implicit-parent fallback -- an
        unrelated entity_ref-typed component with an empty value should
        stay unresolved."""
        registry = Registry.from_component_rows({
            "entity_id": _entity_id_rows(other_entity=True),
            "field": _field_rows("calls"),
            "component_type": _component_type_rows(calls=False),
            "calls": [
                {"entity_id": CHILD_EID, "component_index": 0, "value": None},
                {"entity_id": OTHER_EID, "component_index": 0, "value": "something_else"},
            ],
        })
        df = _resolve(registry)["calls"].to_pandas()
        assert pd.isna(df.set_index("entity_id").loc[CHILD_EID, "value_eid"])

    def test_explicit_dependence_target_resolves(self):
        """dependence works like requirement/solution's own explicit-target
        case: a real value_eid, resolved by entity_ref lookup."""
        registry = Registry.from_component_rows({
            "entity_id": _entity_id_rows(other_entity=True) + [
                {"value": "def_dependence", "path": "builtins.yaml:dependence", "entity_key": "dependence"},
            ],
            "field": _field_rows("dependence"),
            "component_type": _component_type_rows(dependence=False),
            "dependence": [
                {"entity_id": CHILD_EID, "component_index": 0, "value": "something_else"},
            ],
        })
        df = _resolve(registry)["dependence"].to_pandas()
        assert df.set_index("entity_id").loc[CHILD_EID, "value_eid"] == OTHER_EID

    def test_bare_dependence_stays_unresolved(self):
        """Unlike requirement/solution, dependence has no implicit-parent
        fallback -- there's no "candidate dependencies to choose among" the
        way there are candidate solutions to a requirement, so an empty
        value should stay unresolved rather than default to the parent."""
        registry = Registry.from_component_rows({
            "entity_id": _entity_id_rows(other_entity=True) + [
                {"value": "def_dependence", "path": "builtins.yaml:dependence", "entity_key": "dependence"},
            ],
            "field": _field_rows("dependence"),
            "component_type": _component_type_rows(dependence=False),
            "dependence": [
                {"entity_id": CHILD_EID, "component_index": 0, "value": None},
                {"entity_id": OTHER_EID, "component_index": 0, "value": "something_else"},
            ],
        })
        df = _resolve(registry)["dependence"].to_pandas()
        assert pd.isna(df.set_index("entity_id").loc[CHILD_EID, "value_eid"])

    def test_top_level_entity_with_bare_requirement_has_no_parent_to_default_to(self):
        """A top-level entity (no parent) has nothing for
        parent_from_hierarchy to offer -- value_eid should stay unresolved,
        not raise."""
        registry = Registry.from_component_rows({
            "entity_id": [
                {"value": PARENT_EID, "path": PARENT_PATH, "entity_key": "the_parent"},
                {"value": OTHER_EID, "path": OTHER_PATH, "entity_key": "something_else"},
                {"value": "def_requirement", "path": "builtins.yaml:requirement", "entity_key": "requirement"},
            ],
            "field": _field_rows("requirement"),
            "component_type": _component_type_rows("requirement"),
            "requirement": [
                {"entity_id": PARENT_EID, "component_index": 0, "value": None},
                {"entity_id": OTHER_EID, "component_index": 0, "value": "something_else"},
            ],
        })
        df = _resolve(registry)["requirement"].to_pandas()
        assert pd.isna(df.set_index("entity_id").loc[PARENT_EID, "value_eid"])
