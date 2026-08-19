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
