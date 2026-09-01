import hashlib

import pandas as pd

# format_guide.yaml self-documents the EC format spec using the same
# component-tagging conventions as real data, but isn't project data.
FORMAT_GUIDE_BASENAME = "format_guide.yaml"


def non_format_guide_ids(entity_id_table: pd.DataFrame, ids: set) -> set:
    """Return `ids` with any entity sourced from format_guide.yaml removed."""
    if "filepath" not in entity_id_table.columns:
        return ids
    excluded = set(
        entity_id_table.loc[
            entity_id_table["filepath"].astype(str).str.endswith(FORMAT_GUIDE_BASENAME),
            "value",
        ]
    )
    return ids - excluded


def dhash(path: str) -> str:
    """Return a deterministic 12-char hex hash."""
    return hashlib.sha256(path.encode()).hexdigest()[:12]

def get_id(filepath: str, path: str) -> str:
    """Get the ID from the filepath and path within the file."""

    return dhash(f"{filepath}:{path}")


def flagged_component_type_names(components: dict, flag_column: str) -> set[str]:
    """Return names of component types whose own schema entity declares
    ``<flag_column>: true`` on a ``- component_type: {...}`` tag (e.g.
    ``entity_id``'s own ``skip_on_export: true``, or ``requirement``'s own
    ``implicit_parent: true`` in auditing.yaml).

    A component type's schema entity marks itself by owning a
    ``- component_type: {<flag_column>: true}`` tag component. In the
    ``component_type`` meta table, that tag's own row always has
    ``component_type == "component_type"`` -- it is an instance of the
    ``component_type`` component type itself, not of the type it flags --
    so the flagged type's real name is NOT that row's own ``component_type``
    column; it's the owning entity's own ``entity_key``, found by joining
    the row's ``entity_id`` against the ``entity_id`` table.

    Parameters
    ----------
    components : dict
        Dict mapping component type names to ibis Tables, as found on a
        Registry/registrar's ``_components`` or an equivalent ``components``
        dict. For a registry built via ``load_manifest``, both
        ``"component_type"`` and ``"entity_id"`` are always present
        (``registry()`` requires both as inputs) and ``flag_column`` is
        always a real column on ``component_type`` for any of its
        schema-declared flags (see ``load_manifest.component_type_table``,
        which derives the flag columns from ``component_type``'s own
        declared ``field``s rather than a hardcoded list) -- but a registry
        built by hand for a test (see ``emc2p.testing.registry_builder.
        make_registry``) may omit tables/columns it doesn't need, so this
        still returns ``set()`` rather than raising when either is missing.
    flag_column : str
        The boolean column on ``component_type`` to filter by, e.g.
        ``"skip_on_export"``, ``"derived"``, ``"implicit_parent"``.
    """
    if "component_type" not in components or "entity_id" not in components:
        return set()
    ct, entity_id = components["component_type"], components["entity_id"]
    if flag_column not in ct.columns:
        return set()
    flagged = ct.filter(ct.component_type == "component_type", ct[flag_column])
    return set(flagged.join(entity_id, flagged.entity_id == entity_id.value).entity_key.execute())


def candidate_entity_ids(user_path: str, entity_id_table: pd.DataFrame) -> list[str]:
    """Return the entity ID(s) `user_path` identifies.

    Tries three resolutions in order, from most to least exact, each one
    a fallback for when the previous finds nothing:

    1. An exact match against `value` (the entity_id hash itself) —
       `user_path` is returned unchanged as the sole candidate if it's
       already a valid entity_id. Lets any `entity_ref`-typed field (or
       `same_as.value`) reference an entity precisely by hash, the same
       as `same_as.target_entity_id` does explicitly, without a caller
       needing that separate field.
    2. An exact match against `alias`, when `entity_id_table` has one
       (every real entity_id table does; callers that pass a bare
       `value`/`path` table — e.g. the older, alias-less tests for this
       function — just skip to the substring fallback). This isn't just
       an optimization: a container entity's own alias is always a
       substring of its descendants' full paths too (e.g.
       `"make_cats_happy"` vs. `"make_cats_happy.feed_and_water_cats"`),
       so without it, `user_path` naming a container by its own alias
       would resolve as ambiguous with its own children instead of
       uniquely to the container.
    3. Every entity ID whose full `path` contains `user_path` as a
       substring.
    """
    hash_matches = entity_id_table.loc[entity_id_table["value"] == user_path, "value"].tolist()
    if hash_matches:
        return hash_matches
    if "alias" in entity_id_table.columns:
        alias_matches = entity_id_table.loc[entity_id_table["alias"] == user_path, "value"].tolist()
        if alias_matches:
            return alias_matches
    mask = entity_id_table["path"].str.contains(user_path, regex=False)
    return entity_id_table.loc[mask, "value"].tolist()
