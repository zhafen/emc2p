"""Derive step: resolve ``same_as`` components and rebase their owners.

Runs before every other derive step so that inheritance, entity_ref
resolution, and everything downstream sees ``same_as``-declared entities
already folded into their target's identity, not a disconnected new one.
"""
import ibis
import pandas as pd

from ...registry import Registry
from ...utils import candidate_entity_ids


def same_as_resolved_registry(registry: Registry, existing_registry: Registry = None) -> Registry:
    """Rebase entities that declare a ``same_as`` component onto their target.

    A ``same_as`` component declares that its owner entity is identical to an
    already-registered entity, targeted either by ``target_entity_id`` (an
    exact hash) or by ``value`` (an entity_ref path, resolved by substring
    matching). Every component row belonging to the owner — across every
    component type, including its own ``entity_id`` spine row — is rewritten
    onto the target's entity_id, so the rows merge into the target entity
    instead of the owner's own freshly-derived, disconnected identity.

    Parameters
    ----------
    registry : Registry
        The freshly loaded/validated registry for just this update's input.
    existing_registry : Registry, optional
        The registry already accumulated from prior updates, consulted so
        that ``same_as`` can target an entity from an earlier update, not
        just one in the current batch. Omitted (e.g. a bare ``base_etl`` run
        with no prior registry) means only same-batch targets resolve.

    Returns
    -------
    Registry
        ``registry``, with same_as owners' rows rebased onto their targets.

    Raises
    ------
    ValueError
        If a ``same_as`` component's target can't be resolved to exactly one
        entity_id.
    """
    components = registry._components
    if "same_as" not in components:
        return registry

    same_as_df = components["same_as"].to_pandas()
    if same_as_df.empty:
        return registry

    new_entity_df = components["entity_id"].to_pandas()
    if existing_registry is not None:
        existing_entity_df = existing_registry.get("entity_id").to_pandas()
        combined_entity_df = pd.concat(
            [existing_entity_df, new_entity_df], ignore_index=True
        )
    else:
        combined_entity_df = new_entity_df

    rebase_map: dict[str, str] = {}
    for _, row in same_as_df.iterrows():
        owner = row["entity_id"]
        target_hash = row.get("target_entity_id")
        if pd.notna(target_hash):
            target = str(target_hash)
            if not (combined_entity_df["value"] == target).any():
                raise ValueError(
                    f"same_as on {owner!r} targets entity_id {target!r}, "
                    "which does not exist in the registry."
                )
        else:
            path_text = row.get("value")
            if pd.isna(path_text):
                raise ValueError(
                    f"same_as on {owner!r} has neither `value` nor "
                    "`target_entity_id` set."
                )
            candidates = [
                c for c in candidate_entity_ids(str(path_text), combined_entity_df)
                if c != owner
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"same_as on {owner!r} referencing {path_text!r} resolved "
                    f"to {len(candidates)} entities; expected exactly 1."
                )
            target = candidates[0]
        rebase_map[owner] = target

    def _rebase(eid):
        return rebase_map.get(eid, eid)

    updated: dict = {}
    for comp_type, table in components.items():
        if comp_type == "entity_id":
            continue
        df = table.to_pandas()
        if "entity_id" not in df.columns or not df["entity_id"].isin(rebase_map).any():
            continue
        df = df.copy()
        df["entity_id"] = df["entity_id"].map(_rebase)
        # Pass the original schema explicitly: a column that's entirely NULL
        # in this batch (e.g. a time_dimension field not yet backfilled) has
        # no non-null values for pandas/ibis to infer a dtype from, and
        # DuckDB rejects creating a table with an untyped NULL column.
        updated[comp_type] = ibis.memtable(df, schema=table.schema())

    entity_df = new_entity_df[~new_entity_df["value"].isin(rebase_map)]
    updated["entity_id"] = ibis.memtable(entity_df, schema=components["entity_id"].schema())

    registry.update(updated)
    return registry


FINAL_VAR = "same_as_resolved_registry"
