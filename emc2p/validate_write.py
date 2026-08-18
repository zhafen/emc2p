"""Mechanical safety checks for a raw entity-first YAML write, run before
merging it into a registry.

Pure functions over already-parsed YAML plus a registry's own known
component types -- nothing about them is story-simulator-specific (see
docs/manifest/history.yaml: validate_write_checks_ported_from_story_simulator
for where they came from). The third check this module's callers also
rely on -- `_find_malformed_entities`, the bare-mapping check -- already
lives natively in `emc2p.dataflows.etl.load_yaml` and runs automatically
inside `Registrar.update()`'s own ETL, so it isn't duplicated here.
"""

import difflib

import yaml

CONSOLIDATION_GUIDANCE = (
    "Review the above for anything that should be consolidated -- e.g. the "
    "same fact recorded twice under different (possibly invented) component "
    "names, or a wrong-but-real component holding data that belongs under the "
    "correct one instead. If you find something, write a correction; if "
    "everything looks right, no action needed."
)
"""Agent-facing instruction for what to do with a `Registry.summarize_components`
report -- kept here, not on `summarize_components` itself, so that method
stays a plain data report and this module stays the one place holding
guidance on how an agent should act on write-safety/data-quality concerns.
Composed onto the report by `commands.cmd_review_components`."""


def referenced_component_types(parsed: dict) -> set[str]:
    """Collect every component-type name referenced anywhere in `parsed`
    (already-`yaml.safe_load`-ed entity-first YAML) -- the component-list
    keys/bare-string markers at whatever nesting depth they actually
    appear, not the entity-path segments above them.

    Recurse through dict-of-dicts-or-lists (entity nesting), and once an
    actual list is reached (a component list), that's where real
    component-type names live -- either a bare string marker (e.g.
    `component_type`) or the single key of a one-key dict (e.g.
    `{"location": {"value": ...}}`). Assumes `parsed` is already
    well-formed (no bare-mapping mistakes) -- call this only after
    `load_yaml._find_malformed_entities` finds nothing, e.g. after a
    successful `Registrar.update()`.
    """
    types: set[str] = set()

    def walk(value) -> None:
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    types.add(item)
                elif isinstance(item, dict):
                    types.update(k for k in item if k != "data")
            return
        if isinstance(value, dict):
            for key, sub_value in value.items():
                if key != "data":
                    walk(sub_value)

    for alias, value in parsed.items():
        if alias != "data":
            walk(value)
    return types


def component_types_defined_inline(parsed: dict) -> set[str]:
    """Names of component types being freshly declared in this same YAML
    batch -- an entity named after the type itself, carrying a bare
    `component_type` marker in its own component list, the shape a
    manifest's own builtin component definitions use: `location:\\n    -
    description: ...\\n    - component_type\\n    - field: ...`.
    """
    defined: set[str] = set()
    for alias, value in parsed.items():
        if alias == "data" or not isinstance(value, list):
            continue
        if any(item == "component_type" for item in value):
            defined.add(alias)
    return defined


def find_component_named_as_nested_entity(yaml_string: str, known: set[str]) -> list[str]:
    """Scan for an alias whose value is a dict (a legitimate nested-entity
    shape on its own) where one of that dict's own keys names an
    already-known, real component type -- e.g. `car_b:\\n    location:\\n
    - value: x` (missing the leading `- ` that would make `location` a
    list item, i.e. a component attachment, rather than a dict key, i.e.
    a nested child entity's own alias). The parser reads this shape as
    "car_b has a nested child entity named `location`", not "attach a
    `location` component to car_b" -- structurally identical to a genuine
    nested entity, so the bare-mapping check can't catch it on shape
    alone.

    Mechanical, not semantic, like `find_unknown_component_types`: a
    nested child entity's alias coinciding with a real component type's
    name is vanishingly unlikely to be intentional, so this flags every
    such collision.

    Only checks nesting one level below a top-level alias, matching the
    one live mistake actually observed -- not a general recursive walk,
    since a collision at a deeper nesting level has no evidence behind it
    yet.
    """
    try:
        parsed = yaml.safe_load(yaml_string)
    except yaml.YAMLError:
        return []
    if not isinstance(parsed, dict):
        return []

    suspicious: list[str] = []
    for alias, value in parsed.items():
        if alias == "data" or not isinstance(value, dict):
            continue
        for key, sub_value in value.items():
            if key != "data" and key in known and isinstance(sub_value, list):
                suspicious.append(f"{alias}.{key}")
    return suspicious


def find_unknown_component_types(yaml_string: str, known: set[str]) -> dict[str, list[str]]:
    """Component types this write references that aren't in `known` and
    aren't being defined inline (see `component_types_defined_inline`) --
    mapping each unknown name to its closest known matches, if any, so
    the caller's own error message can suggest a likely-intended real
    name instead of leaving the caller to guess again.

    Best-effort: a parse failure or unexpected top-level shape just finds
    nothing here too, since `Registrar.update()`'s own error handling is
    authoritative for those.

    Deliberately mechanical, not semantic -- catches an invented name
    (`parked_at`) that was never declared anywhere, but not a real,
    already-declared type used for the wrong purpose (e.g. writing a
    car's location under `todo` instead of `location` -- both are
    genuine, known component types, so this check has nothing to object
    to).
    """
    try:
        parsed = yaml.safe_load(yaml_string)
    except yaml.YAMLError:
        return {}
    if not isinstance(parsed, dict):
        return {}

    referenced = referenced_component_types(parsed)
    defined_inline = component_types_defined_inline(parsed)
    unknown = referenced - known - defined_inline
    if not unknown:
        return {}
    return {name: difflib.get_close_matches(name, sorted(known), n=3) for name in sorted(unknown)}
