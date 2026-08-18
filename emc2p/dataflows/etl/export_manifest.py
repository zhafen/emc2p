"""Hamilton DAG for converting component-centered registry data back to entity-centered manifest data.

DAG structure (dependency order):

    registry
        └── components (also extracts entity_id)
                ├── entity_hierarchy (depends on components + entity_id)
                │       └── non_hierarchy_parents (depends on components + entity_hierarchy + entity_id)
                │               └── components_for_export
                │                       └── entity_first_data (depends on components_for_export + entity_id)
                │                               └── condensed_entity_first_data
                │                                       └── hierarchical_entity_first_data
                │                                       └── exported_manifest_filepaths
                └── entity_id (extracted via @extract_fields)

    hierarchical_entity_first_data also depends on entity_hierarchy and entity_id.

The entity_hierarchy node determines which parent relationship defines each
entity's position in the nesting tree (is_primary=True wins; otherwise the
first explicit parent by component_index is used).

The non_hierarchy_parents node collects parent rows NOT used as the hierarchy
parent so they can be round-tripped as explicit ``parent:`` YAML entries.

The components_for_export node replaces the parent table with non-hierarchy
parents only (removing hierarchy-implied parents that would be reconstructed
from nesting on reload).

entity_first_data serialises all component tables to {entity_id: [(idx, entry)]}
(flat, keyed by entity_id).  condensed_entity_first_data collapses single-field
component dicts from ``{type: {field: value}}`` to ``{type: value}``.
hierarchical_entity_first_data uses the entity_hierarchy to nest child entities
under their parents and produces the {filepath: {entity_key: ...}} structure
written to YAML.
"""

from pathlib import Path

import yaml
from hamilton.function_modifiers import extract_fields
import ibis
import ibis.expr.types as ir
import pandas as pd

from ...registry import Registry
from .load_manifest import builtins_tags


INPUT_COMPONENT_TYPES = ["entity_id", "parent"]


def _exportable_mask(filepath: pd.Series) -> pd.Series:
    """True for entities that belong in manifest export.

    Excludes builtin entities (from emc2p's own builtins directory, and any
    other registered via ``load_manifest.register_builtins_dir``) and any
    entity sourced from a Python file. Python-derived components —
    including a docstring's "Components" YAML section (see
    ``emc2p.dataflows.etl.load_python``) — live in source code, not a
    standalone YAML file, so they have nowhere to round-trip to and are
    excluded from export by default.
    """
    is_builtin = pd.Series(False, index=filepath.index)
    for tag in builtins_tags():
        is_builtin |= filepath.str.startswith(f"{tag}.")
    return ~is_builtin & ~filepath.str.endswith(".py")


@extract_fields({ct: ir.Table for ct in INPUT_COMPONENT_TYPES})
def components(registry: Registry, collapse_time_dimension: bool = True) -> dict:
    """Extract all component tables from the registry.

    Returns all components (not just INPUT_COMPONENT_TYPES) because downstream
    nodes such as ``components_for_export`` and ``entity_first_data`` must
    iterate over every component type to reconstruct the full manifest.
    ``parent`` is guaranteed present via a fallback empty table.

    Args:
        collapse_time_dimension: When True (the default), any component type
            with a field flagged ``time_dimension: true`` is collapsed to its
            current row per entity -- the same resolution
            ``Registry.view_current`` uses -- rather than exported as its
            full accumulated history. A manifest export is normally meant to
            round-trip a snapshot of current state (what a save's own YAML
            should read as, e.g. a single door description rather than every
            restatement of it since the world started), not a change log.
            Set False to export every component type in full, uncollapsed,
            regardless of time_dimension -- e.g. for a full-history export
            kept alongside a collapsed current-state one. The full history is
            always still in the registry itself either way, reachable via
            ``view`` even when this defaults to collapsing the export.
    """
    result = {
        comp_type: (
            registry._current_table(comp_type)
            if collapse_time_dimension and registry._time_dimension_field(comp_type) is not None
            else table
        )
        for comp_type, table in registry._components.items()
    }
    if "parent" not in result:
        result["parent"] = ibis.memtable(
            [],
            schema={"entity_id": "string", "component_index": "int64", "modifier": "string", "parent_eid": "string", "is_primary": "boolean"},
        )
    return result


_METADATA_COLS = {"entity_id", "component_index", "modifier", "is_primary"}


def _derived_comp_types(components: dict) -> set[str]:
    """Return component type names marked derived=True in the component_type table."""
    if "component_type" not in components:
        return set()
    ct = components["component_type"]
    if "derived" not in ct.columns:
        return set()
    return set(
        ct.filter(ct["derived"] == True)
        .select("component_type")
        .distinct()
        .execute()["component_type"]
        .tolist()
    )


def _skip_on_export_types(components: dict) -> set[str]:
    """Return component type names marked skip_on_export=True in the component_type table."""
    if "component_type" not in components:
        return set()
    ct = components["component_type"]
    if "skip_on_export" not in ct.columns:
        return set()
    return set(
        ct.filter(ct["skip_on_export"] == True)
        .select("component_type")
        .distinct()
        .execute()["component_type"]
        .tolist()
    )


def entity_hierarchy(parent: ir.Table, entity_id: ir.Table) -> dict[str, str | None]:
    """Determine the hierarchy parent for each user entity.

    Rule: if the entity has exactly one parent with is_primary=True, use that.
    Otherwise use the first parent (lowest component_index). Root entities
    (no parents) map to None.

    Parameters
    ----------
    parent : ir.Table
        The parent component table from the registry.
    entity_id : ir.Table
        The entity spine table extracted from ``components``.

    Returns
    -------
    dict[str, str | None]
        Mapping of child entity_id → parent entity_id (or None for roots).
        Entities with no parents are NOT included in the dict.
    """
    spine_df = entity_id.to_pandas()
    user_rows = spine_df[_exportable_mask(spine_df["filepath"])]
    user_entity_ids = set(user_rows["value"].unique())

    parent_df = parent.execute()

    # Filter to user entities only
    parent_df = parent_df[parent_df["entity_id"].isin(user_entity_ids)].copy()
    if parent_df.empty:
        return {}

    hierarchy: dict[str, str | None] = {}

    for child_eid, group in parent_df.groupby("entity_id"):
        # Check for is_primary column
        if "is_primary" in group.columns:
            primary = group[group["is_primary"] == True]
            if len(primary) == 1:
                hierarchy[child_eid] = primary.iloc[0]["parent_eid"]
                continue

        # Fall back: use the row with the lowest component_index
        if "component_index" in group.columns:
            chosen = group.nsmallest(1, "component_index").iloc[0]
        else:
            chosen = group.iloc[0]
        hierarchy[child_eid] = chosen["parent_eid"]

    return hierarchy


def non_hierarchy_parents(
    parent: ir.Table,
    entity_hierarchy: dict[str, str | None],
    entity_id: ir.Table,
) -> ir.Table:
    """Return parent rows NOT used as the hierarchy parent for each entity.

    Only user entities (non-builtin) are considered. Builtin entity rows are
    always excluded since they should not be exported.

    Parameters
    ----------
    parent : ir.Table
        The parent component table from the registry.
    entity_hierarchy : dict[str, str | None]
        Mapping of child entity_id → parent entity_id as returned by
        ``entity_hierarchy``.
    entity_id : ir.Table
        The entity spine table extracted from ``components``.

    Returns
    -------
    ir.Table
        Parent rows (same schema as the parent component table) for all
        parent entries that were not selected as the hierarchy parent.
        Empty table if there are no non-hierarchy parents.
    """
    parent_df = parent.execute()
    if parent_df.empty:
        return ibis.memtable(parent_df)

    # Only keep rows for user entities
    spine_df = entity_id.to_pandas()
    user_rows = spine_df[_exportable_mask(spine_df["filepath"])]
    user_entity_ids = set(user_rows["value"].unique())

    parent_df = parent_df[parent_df["entity_id"].isin(user_entity_ids)].copy()

    # Build entity_id → entity_key lookup for reconstructing parent value
    spine_df = entity_id.to_pandas()
    id_to_key = (
        spine_df.drop_duplicates("value")
        .set_index("value")["entity_key"]
        .to_dict()
    )

    keep_rows = []
    for child_eid, group in parent_df.groupby("entity_id"):
        chosen_parent_eid = entity_hierarchy.get(child_eid)
        if chosen_parent_eid is None:
            # No hierarchy parent selected — keep all rows
            keep_rows.append(group)
            continue

        # Exclude exactly one row matching the chosen parent_eid
        if "parent_eid" in group.columns:
            match_mask = group["parent_eid"] == chosen_parent_eid
            matches = group[match_mask]
            if not matches.empty:
                # Drop only the first matching row
                first_match_idx = matches.index[0]
                remaining = group.drop(index=first_match_idx)
            else:
                remaining = group
        else:
            remaining = group

        if not remaining.empty:
            keep_rows.append(remaining)

    if not keep_rows:
        return ibis.memtable(parent_df.iloc[0:0])  # empty with same schema

    result_df = pd.concat(keep_rows, ignore_index=True)

    # Reconstruct value column from parent_eid for export
    if "parent_eid" in result_df.columns and "value" not in result_df.columns:
        result_df["value"] = result_df["parent_eid"].map(id_to_key).astype(pd.StringDtype())

    return ibis.memtable(result_df)


def components_for_export(
    components: dict,
    non_hierarchy_parents: ir.Table,
) -> dict:
    """Return updated components dict with parent table replaced by non-hierarchy-only parents.

    If non_hierarchy_parents is empty, the parent key is removed from components.

    Parameters
    ----------
    components : dict
        Dict mapping component type names to ibis Tables.
    non_hierarchy_parents : ir.Table
        Parent rows not used as hierarchy parents, as returned by
        ``non_hierarchy_parents``.

    Returns
    -------
    dict
        Updated components dict.
    """
    result = dict(components)
    nhp_df = non_hierarchy_parents.execute()
    if nhp_df.empty:
        result.pop("parent", None)
    else:
        result["parent"] = non_hierarchy_parents

    return result


def entity_first_data(components_for_export: dict, entity_id: ir.Table) -> dict:
    """Reconstruct the entity-centered nested dict from component tables.

    For each component type, groups rows by entity_id and serializes each row
    as a component entry. Tags (empty value, no other fields) become bare
    strings; all other components become ``{type: {field: value, ...}}``.
    Modifiers (e.g. "of") are appended to the component type key
    (e.g. "solution of"). Builtin entities are excluded.

    Parameters
    ----------
    components_for_export : dict
        Dict mapping component type names to ibis Tables, as returned by
        ``components_for_export``. Must include an ``"entity_id"`` key.
    entity_id : ir.Table
        The entity spine table extracted from ``components``.

    Returns
    -------
    dict
        ``{entity_id: [(component_index, entry), ...]}`` — flat, keyed by
        entity_id, with entries not yet sorted (sorting happens in
        ``hierarchical_entity_first_data``).
    """
    spine_df = entity_id.to_pandas()
    user_rows = spine_df[_exportable_mask(spine_df["filepath"])]
    user_entity_ids = set(user_rows["value"].unique())

    skip_types = _skip_on_export_types(components_for_export)
    derived_types = _derived_comp_types(components_for_export)

    result: dict[str, list] = {}

    for comp_type, table in components_for_export.items():
        if comp_type in skip_types or comp_type in derived_types:
            continue

        df = table.execute()
        if "entity_id" not in df.columns or "component_index" not in df.columns:
            continue

        for _, row in df.iterrows():
            eid = row["entity_id"]
            cidx = int(row["component_index"])
            if eid not in user_entity_ids:
                continue

            modifier = row.get("modifier")
            key = f"{comp_type} {modifier}" if pd.notna(modifier) and modifier else comp_type

            fields = {
                k: v for k, v in row.items()
                if k not in _METADATA_COLS
                and not k.endswith("_eid")
                and not k.startswith("_seq_")
                and pd.notna(v)
            }

            if not fields or (len(fields) == 1 and fields.get("value") == ""):
                entry = key
            else:
                entry = {key: fields}

            result.setdefault(eid, []).append((cidx, entry))

    return result


def condensed_entity_first_data(entity_first_data: dict) -> dict:
    """Condense single-field component entries to a scalar form.

    For any component entry of the form ``{type: {field: value}}`` where the
    inner dict has exactly one field, replaces it with ``{type: value}`` so
    the exported YAML stays concise.

    Parameters
    ----------
    entity_first_data : dict
        Output of ``entity_first_data``: ``{entity_id: [(idx, entry), ...]}``.

    Returns
    -------
    dict
        Same structure, with single-field component dicts collapsed.
    """
    result: dict[str, list] = {}
    for eid, entries in entity_first_data.items():
        condensed = []
        for idx, entry in entries:
            if isinstance(entry, dict):
                (comp_type, fields), = entry.items()
                if isinstance(fields, dict) and len(fields) == 1:
                    (field_name, field_value), = fields.items()
                    if field_name == "value":
                        entry = {comp_type: field_value}
            condensed.append((idx, entry))
        result[eid] = condensed
    return result


def _get_or_create_node(
    path_parts: list[str], hierarchical: dict, node_of: dict
) -> dict:
    """Return the dict node for a given entity path, creating structural nodes as needed.

    ``node_of[path]`` stores the entity's own dict value (after any list→dict conversion).
    If the entity's value is still a list (no children have been added yet), it is
    converted to ``{"data": list}`` on first access as a parent.
    """
    if not path_parts:
        return hierarchical

    path = ".".join(path_parts)
    key = path_parts[-1]

    if path in node_of:
        val = node_of[path]
        if isinstance(val, list):
            parent = _get_or_create_node(path_parts[:-1], hierarchical, node_of)
            parent[key] = {"data": val} if val else {}
            node_of[path] = parent[key]
        return node_of[path]

    parent = _get_or_create_node(path_parts[:-1], hierarchical, node_of)
    if key not in parent:
        parent[key] = {}
    elif isinstance(parent[key], list):
        parent[key] = {"data": parent[key]} if parent[key] else {}
    node_of[path] = parent[key]
    return node_of[path]


def hierarchical_entity_first_data(
    condensed_entity_first_data: dict,
    entity_hierarchy: dict[str, str | None],  # noqa: ARG001 — used for DAG wiring
    entity_id: ir.Table,
) -> dict:
    """Nest entity components using the entity path stored in the registry.

    Takes components from ``condensed_entity_first_data`` and nests child
    entities under their parents using the dot-separated path.  When an entity
    has both its own components and child entities, its components go under a
    ``"data"`` key.

    ``entity_hierarchy`` is accepted for DAG dependency ordering (it ensures
    non-hierarchy parents are already filtered out of
    ``condensed_entity_first_data``'s source dict) but the actual nesting is
    path-based so virtual parent nodes (path segments with no corresponding
    entity) are handled automatically.

    Parameters
    ----------
    condensed_entity_first_data : dict
        Flat mapping ``{entity_id: [(component_index, entry), ...]}`` as
        returned by ``condensed_entity_first_data``.
    entity_hierarchy : dict[str, str | None]
        Unused directly; accepted so Hamilton routes this node after
        ``entity_hierarchy`` is computed.
    entity_id : ir.Table
        The entity spine table extracted from ``components``.

    Returns
    -------
    dict
        ``{filepath: {entity_key: ...}}`` where nested entities are dicts
        containing a ``"data"`` key (own components) and child-entity keys.
    """
    spine_df = entity_id.to_pandas()
    user_rows = spine_df[_exportable_mask(spine_df["filepath"])]
    user_entity_ids = set(user_rows["value"].unique())

    id_to_filepath: dict[str, str] = {}
    id_to_path_in_file: dict[str, str] = {}

    for _, row in user_rows.iterrows():
        eid = str(row["value"])
        filepath = str(row["filepath"])
        full_path = str(row["path"])
        id_to_filepath[eid] = filepath
        id_to_path_in_file[eid] = full_path[len(filepath) + 1:]

    # Resolve component lists per entity_id (already sorted by condensed_entity_first_data)
    entity_components: dict[str, list] = {
        eid: [e for _, e in sorted(entries, key=lambda x: x[0])]
        for eid, entries in condensed_entity_first_data.items()
    }

    filepath_entities: dict[str, list[tuple[str, str]]] = {}
    for eid in user_entity_ids:
        if eid in id_to_filepath:
            filepath_entities.setdefault(id_to_filepath[eid], []).append(
                (eid, id_to_path_in_file[eid])
            )

    result: dict[str, dict] = {}
    for filepath, entity_list in filepath_entities.items():
        entity_list.sort(key=lambda x: (len(x[1].split(".")), x[1]))

        hierarchical: dict = {}
        node_of: dict[str, object] = {}

        for eid, path_in_file in entity_list:
            parts = path_in_file.split(".")
            entity_key = parts[-1]
            comps = entity_components.get(eid, [])

            if len(parts) == 1:
                hierarchical[entity_key] = comps
                node_of[path_in_file] = hierarchical[entity_key]
            else:
                parent_node = _get_or_create_node(parts[:-1], hierarchical, node_of)
                parent_node[entity_key] = comps
                node_of[path_in_file] = parent_node[entity_key]

        result[filepath] = hierarchical

    return result


def exported_manifest_filepaths(hierarchical_entity_first_data: dict, output_dir: str = "") -> list[str]:
    """Save hierarchical_entity_first_data to one EC file per original source filepath.

    Parameters
    ----------
    hierarchical_entity_first_data : dict
        Mapping of original filepath → entity dict, as returned by
        ``hierarchical_entity_first_data``.
    output_dir : str
        Directory to write the EC files into.  When empty (the
        default), each file is written back to its original source path,
        normalising formatting in-place.

    Returns
    -------
    list[str]
        The saved filepaths, sorted for determinism.
    """
    saved = []
    for source_filepath, entities in hierarchical_entity_first_data.items():
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            dest = out / Path(source_filepath).with_suffix(".yaml").name
        else:
            dest = Path(source_filepath)
            dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            yaml.dump(entities, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        saved.append(str(dest))
    return sorted(saved)


FINAL_VAR = "exported_manifest_filepaths"
