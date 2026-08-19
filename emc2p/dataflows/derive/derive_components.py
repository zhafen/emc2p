from typing import Any

import ibis
import pandas as pd
from hamilton.function_modifiers import subdag, source

from ...registry import Registry
from . import inherit_components, resolve_paths, resolve_same_as


def same_as_resolved_registry(registry: Registry, existing_registry: Registry = None) -> Registry:
    return resolve_same_as.same_as_resolved_registry(registry, existing_registry)


@subdag(resolve_paths, inputs={"registry": source("same_as_resolved_registry")}, config={})
def resolved_registry(resolved_registry: Registry) -> Registry:
    return resolved_registry


def stripped_registry(resolved_registry: Registry) -> Registry:
    """Strip leading/trailing whitespace from description-typed fields.

    Strips the ``value`` column of the ``description`` component and any field
    columns in other component tables whose declared type is ``description``.
    """
    components = resolved_registry._components
    updated: dict = {}

    field_df = components["field"].to_pandas() if "field" in components else pd.DataFrame()
    entity_id_df = components["entity_id"].to_pandas() if "entity_id" in components else pd.DataFrame()

    fields_by_comp: dict[str, list[str]] = {}
    if not field_df.empty and not entity_id_df.empty and "type" in field_df.columns:
        desc_fields = field_df[field_df["type"] == "description"]
        id_to_key = entity_id_df.set_index("value")["entity_key"]
        for _, row in desc_fields.iterrows():
            comp_type = id_to_key.get(row["entity_id"])
            if comp_type:
                fields_by_comp.setdefault(comp_type, []).append(row["value"])

    if "description" in components:
        original = components["description"]
        df = original.to_pandas().copy()
        if "value" in df.columns:
            df["value"] = df["value"].apply(lambda v: v.strip() if isinstance(v, str) else v)
            # Schema passed explicitly, not inferred: an all-NULL column here
            # has no dtype for pandas/ibis to infer, which DuckDB rejects.
            updated["description"] = ibis.memtable(df, schema=original.schema())

    for comp_type, field_names in fields_by_comp.items():
        if comp_type not in components:
            continue
        original = updated.get(comp_type) or components[comp_type]
        df = original.to_pandas().copy()
        for col in field_names:
            if col in df.columns:
                df[col] = df[col].apply(lambda v: v.strip() if isinstance(v, str) else v)
        updated[comp_type] = ibis.memtable(df, schema=original.schema())

    resolved_registry.update(updated)
    return resolved_registry


@subdag(inherit_components, inputs={"registry": source("stripped_registry")}, config={})
def field_derived_registry(derived_registry: Registry) -> Registry:
    return derived_registry


def time_filled_registry(field_derived_registry: Registry, load_time: Any = None) -> Registry:
    """Backfill null time_dimension fields with load_time.

    Used when loading a manifest that represents a snapshot as of a known
    point in time: the field flagged ``time_dimension: true`` in a component
    type's schema is set to ``load_time`` wherever it is still null. Values
    that are already set are left untouched. A no-op when ``load_time`` is
    not given.

    Parameters
    ----------
    field_derived_registry : Registry
        The inheritance-resolved registry, with "field" already validated
        and type-coerced against its own schema (so ``time_dimension`` is a
        real bool, not a raw string) by the preceding validate_components
        pass.
    load_time : Any, optional
        The point in time this load represents, e.g. a timestamp or date
        string.

    Returns
    -------
    Registry
        ``field_derived_registry``, with time_dimension fields backfilled.

    Raises
    ------
    ValueError
        If a component type has more than one time_dimension field.
    """
    if load_time is None:
        return field_derived_registry

    components = field_derived_registry._components
    df_field = components["field"].execute()
    if "time_dimension" not in df_field.columns:
        return field_derived_registry

    df_entity = components["entity_id"].execute()
    key_by_eid = df_entity.set_index("value")["entity_key"]

    time_fields: dict[str, str] = {}
    for _, row in df_field.iterrows():
        if not row["time_dimension"]:
            continue
        fname = row.get("value")
        if pd.isna(fname):
            continue
        ctype = key_by_eid.get(row["entity_id"])
        if ctype is None:
            continue
        fname = str(fname)
        existing = time_fields.get(ctype)
        if existing is not None and existing != fname:
            raise ValueError(
                f"Component type {ctype!r} has multiple time_dimension "
                f"fields {sorted({existing, fname})}; only one is allowed."
            )
        time_fields[ctype] = fname

    updated: dict = {}
    for ctype, fname in time_fields.items():
        table = components.get(ctype)
        if table is None or fname not in table.columns:
            continue
        df = table.execute()
        if df[fname].isna().any():
            df[fname] = df[fname].fillna(load_time)
            updated[ctype] = ibis.memtable(df)

    field_derived_registry.update(updated)
    return field_derived_registry


FINAL_VAR = "time_filled_registry"
