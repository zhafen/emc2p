import pandas as pd
import ibis
import ibis.expr.types as ir
from hamilton.function_modifiers import extract_fields

from ...registry import Registry


INPUT_COMPONENT_TYPES = ["field", "parent"]


@extract_fields({ct: ir.Table for ct in INPUT_COMPONENT_TYPES})
def components(registry: Registry) -> dict:
    """Give access to the components needed by this dataflow."""
    return {ct: registry.get(ct) for ct in INPUT_COMPONENT_TYPES}


def derived_field(field: ir.Table, parent: ir.Table) -> ir.Table:
    """Component definitions inherit fields from their parents, but can override them.
    The derived_field table contains the results of applying this inheritance.

    For each entity the full set of fields is resolved by a BFS walk up the
    parent chain.  The child's own definition of a field always beats any
    ancestor's definition of a field with the same name (``value`` column).
    The output has the same columns as ``field`` but one row per
    ``(entity_id, field_name)`` pair, where ``entity_id`` is the entity that
    *has* the field (directly or through inheritance).

    Parameters
    ----------
    field : ir.Table
        The type-coerced field component table.
    parent : ir.Table
        Parent–child relationships with columns ``entity_id`` and ``parent_eid``.

    Returns
    -------
    ir.Table
        One row per (entity, field_name) with the winning field definition.
    """
    df_field = field.execute()
    df_parent = parent.execute()

    # ── entity_own_fields: entity_id -> {field_name: row_dict} ───────────
    entity_own_fields: dict[str, dict[str, dict]] = {}
    for _, row in df_field.iterrows():
        eid = row["entity_id"]
        fname = row.get("value")
        if pd.isna(fname) or not fname:
            continue
        entity_own_fields.setdefault(str(eid), {})[str(fname)] = row.to_dict()

    # ── parent_map: entity_id -> [parent_id, ...] ─────────────────────────
    parent_map: dict[str, list[str]] = {}
    for _, row in df_parent.iterrows():
        eid, pid = row.get("entity_id"), row.get("parent_eid")
        if pd.notna(eid) and pd.notna(pid):
            parent_map.setdefault(str(eid), []).append(str(pid))

    # ── BFS resolver: child fields beat parent fields ─────────────────────
    def resolve_fields(start_id: str) -> dict[str, dict]:
        resolved: dict[str, dict] = {}
        visited: set[str] = set()
        queue = [start_id]
        while queue:
            eid = queue.pop(0)
            if eid in visited:
                continue
            visited.add(eid)
            for fname, frow in entity_own_fields.get(eid, {}).items():
                if fname not in resolved:
                    resolved[fname] = frow
            queue.extend(p for p in parent_map.get(eid, []) if p not in visited)
        return resolved

    # ── Emit one row per (entity, field_name) ─────────────────────────────
    all_entities = (
        set(entity_own_fields)
        | set(parent_map)
        | {p for parents in parent_map.values() for p in parents}
    )

    result_rows = []
    for entity_id in sorted(all_entities):
        own_fields = entity_own_fields.get(entity_id, {})
        own_field_names = set(own_fields)
        own_indices = [r.get("component_index", 0) for r in own_fields.values()]
        next_index = max(own_indices, default=0) + 1

        for fname, frow in resolve_fields(entity_id).items():
            if fname in own_field_names:
                result_rows.append({**frow, "entity_id": entity_id})
            else:
                result_rows.append(
                    {**frow, "entity_id": entity_id, "component_index": next_index}
                )
                next_index += 1

    if result_rows:
        result_df = (
            pd.DataFrame(result_rows)
            .drop_duplicates(subset=["entity_id", "value"])
            .reset_index(drop=True)
        )
    else:
        result_df = df_field.iloc[0:0].copy()

    return ibis.memtable(result_df)


def derived_registry(registry: Registry, derived_field: ir.Table) -> Registry:
    """Store the inheritance-resolved field table as "field".

    "field" is overwritten here so that anything reading "field" downstream
    gets the inheritance-resolved set, not just each entity's own
    directly-declared fields.
    """
    registry.update({"field": derived_field})
    return registry


FINAL_VAR = "derived_registry"
