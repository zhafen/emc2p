import pandas as pd
import ibis
import ibis.expr.types as ir
from hamilton.function_modifiers import extract_fields

from ...registry import Registry
from ...utils import candidate_entity_ids, dhash


INPUT_COMPONENT_TYPES = ["field", "entity_id"]


@extract_fields({ct: ir.Table for ct in INPUT_COMPONENT_TYPES})
def components(registry: Registry) -> dict:
    """Give access to the components needed by this dataflow.

    Returns all registry components (not just INPUT_COMPONENT_TYPES) so that
    ``components_with_resolved_paths`` can resolve entity_ref fields across
    every component type.
    """
    return registry._components


def fields_of_type_entity_ref(entity_id: ir.Table, field: ir.Table) -> dict[str, list[str]]:
    """Return a mapping of component_type -> [field_names] for entity_ref fields.

    Parameters
    ----------
    field : ir.Table
        The field component table from the registry.
    entity_id : ir.Table
        The entity_id component table from the registry.

    Returns
    -------
    dict[str, list[str]]
        E.g. ``{"solution": ["target"]}``
    """
    derived_field_df = field.to_pandas()
    entity_id_df = entity_id.to_pandas()

    entity_ref_fields = derived_field_df[derived_field_df["type"] == "entity_ref"]
    id_to_key = entity_id_df.set_index("value")["entity_key"]

    result: dict[str, list[str]] = {}
    for _, row in entity_ref_fields.iterrows():
        comp_type = id_to_key.get(row["entity_id"])
        if comp_type:
            result.setdefault(comp_type, []).append(row["value"])
    return result


@extract_fields(dict(parent=ir.Table))
def components_with_resolved_paths(
    entity_id: ir.Table,
    components: dict,
    fields_of_type_entity_ref: dict[str, list[str]],
) -> dict:
    """Return all components, with entity_ref fields resolved to ``{field}_eid`` columns.

    Every component from the registry is included in the result.  For component
    types listed in ``fields_of_type_entity_ref``, each named field gets a
    companion ``{field}_eid`` column containing the resolved entity ID
    (``None`` if 0 or 2+ candidates match).

    Parameters
    ----------
    entity_id : ir.Table
        The entity_id component table from the registry.
    components : dict
        The full components dict from the registry.
    fields_of_type_entity_ref : dict[str, list[str]]
        Mapping of component_type -> list of field names with entity_ref type.

    Returns
    -------
    dict
        All components, with resolved ``{field}_eid`` columns added where applicable.
    """
    entity_id_df = entity_id.to_pandas()

    result: dict = dict(components)
    for comp_type, field_names in fields_of_type_entity_ref.items():
        if comp_type not in result:
            continue
        df = result[comp_type].to_pandas()
        for field_name in field_names:
            if field_name not in df.columns:
                continue

            def resolve(val, _df=entity_id_df):
                if pd.isna(val):
                    return None
                candidates = candidate_entity_ids(str(val), _df)
                return candidates[0] if len(candidates) == 1 else None

            df[f"{field_name}_eid"] = df[field_name].apply(resolve).astype("string")
        result[comp_type] = ibis.memtable(df)
    return result


def parent_from_hierarchy(entity_id: ir.Table) -> ir.Table:
    """Convert the entity paths in entity_id_table into parent-child relationships.

    Produces hierarchy-implied parent-child rows: every nested entity (whose
    entity path contains a dot after the file-id separator) is a child of the
    entity at the path one level up.

    Parameters
    ----------
    entity_id : ir.Table
        One row per entity with columns ``value``, ``path``, ``entity_key``, ``filepath``.

    Returns
    -------
    ir.Table
        A table with columns ``entity_id``, ``component_index``, ``modifier``,
        ``parent_eid``, ``is_primary``.
    """
    df_spine = entity_id.to_pandas()
    df_spine = df_spine.rename(columns={"value": "entity_id"})
    df_spine["entity_path"] = df_spine["path"]

    def has_parent(entity_path):
        sep = entity_path.find(":")
        name_part = entity_path[sep + 1:] if sep != -1 else entity_path
        return "." in name_part

    def get_parent_path(entity_path):
        sep = entity_path.find(":")
        if sep != -1:
            file_id, name_part = entity_path[:sep], entity_path[sep + 1:]
            return f"{file_id}:{name_part.rsplit('.', 1)[0]}"
        return entity_path.rsplit(".", 1)[0]

    spine_pairs = df_spine[["entity_id", "entity_path"]].dropna().drop_duplicates()
    nested = spine_pairs[spine_pairs["entity_path"].apply(has_parent)].copy()

    if nested.empty:
        hierarchy = pd.DataFrame(
            [],
            columns=["entity_id", "component_index", "modifier", "parent_eid", "is_primary"],
        )
    else:
        nested["parent_eid"] = nested["entity_path"].apply(
            lambda ep: dhash(get_parent_path(ep))
        )
        nested["component_index"] = -1
        nested["modifier"] = pd.NA
        nested["is_primary"] = True
        hierarchy = nested[
            ["entity_id", "component_index", "modifier", "parent_eid", "is_primary"]
        ].drop_duplicates()

    return ibis.memtable(hierarchy)


def updated_parent(
    entity_id: ir.Table, parent: ir.Table, parent_from_hierarchy: ir.Table
) -> ir.Table:
    """Combine the parents from the hierarchy with the ones from the resolved components."""
    df_spine = entity_id.to_pandas()
    df_spine = df_spine.rename(columns={"value": "entity_id"})
    key_to_id = (
        df_spine[["entity_id", "entity_key"]]
        .dropna()
        .drop_duplicates(subset=["entity_id", "entity_key"])
        .drop_duplicates(subset=["entity_key"])
        .set_index("entity_key")["entity_id"]
        .to_dict()
    )

    df_parent = parent.to_pandas()
    df_hierarchy = parent_from_hierarchy.to_pandas()

    if not df_parent.empty and "value" in df_parent.columns:
        df_exp = df_parent.copy()
        df_exp = df_exp.dropna(subset=["value"])
        df_exp["parent_eid"] = df_exp["value"].map(key_to_id)
        df_exp["is_primary"] = False
        df_exp = df_exp.dropna(subset=["parent_eid"])
        explicit_cols = ["entity_id", "component_index", "modifier", "parent_eid", "is_primary"]
        for col in explicit_cols:
            if col not in df_exp.columns:
                df_exp[col] = pd.NA
        explicit = df_exp[explicit_cols].drop_duplicates()
    else:
        explicit = pd.DataFrame(
            [],
            columns=["entity_id", "component_index", "modifier", "parent_eid", "is_primary"],
        )

    combined = (
        pd.concat([df_hierarchy, explicit], ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    combined["entity_id"] = combined["entity_id"].astype(pd.StringDtype())
    combined["modifier"] = combined["modifier"].astype(pd.StringDtype())
    combined["parent_eid"] = combined["parent_eid"].astype(pd.StringDtype())
    combined["component_index"] = combined["component_index"].astype("int64")
    return ibis.memtable(combined)


def resolved_registry(
    registry: Registry,
    updated_parent: ir.Table,
    components_with_resolved_paths: dict,
) -> Registry:
    """Store resolved paths and updated parent back into the registry."""
    registry.update({**components_with_resolved_paths, "parent": updated_parent})
    return registry


FINAL_VAR = "resolved_registry"
