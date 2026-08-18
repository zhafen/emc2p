"""Hamilton DAG for parsing entity-first data from EC file text."""

import yaml


def _find_malformed_entities(entities: dict) -> list[str]:
    """Scan a parsed entity-first dict for components given as a bare YAML
    mapping instead of the required list form (`alias:\\n    component:\\n
    value: x` instead of `alias:\\n    - component:\\n        value: x`).

    Parameters
    ----------
    entities : dict
        A parsed entity-first dict, as returned per-file by
        `raw_entity_first_data`.

    Returns
    -------
    list[str]
        Dotted paths (relative to this dict) of malformed components, empty
        if none found.
    """
    # Downstream flattening treats any dict value as a nested-entity
    # container rather than raising on this mistake, so it silently
    # produces zero rows instead of merging -- this is the only thing
    # that catches it.
    malformed: list[str] = []

    def check(path: str, value) -> None:
        # A dict value is ambiguous on its own: legitimate nested-entity
        # container, or this mistake. Distinguished by recursing only
        # while every value at this level is itself a list, dict, or
        # null -- a bare non-null scalar means this level was meant as a
        # component list. Null counts as valid too: a bare `key:` with
        # nothing after the colon is a real, componentless placeholder
        # entity (e.g. `iacs_manifest/iacs.yaml`'s
        # `required_functionality:` block) -- treating it as malformed
        # broke real data the first time this shipped. `"data"` is
        # exempt everywhere, matching EC's own freeform-notes convention.
        if value is None or isinstance(value, list):
            return
        if isinstance(value, dict) and value and all(
            v is None or isinstance(v, (list, dict))
            for k, v in value.items() if k != "data"
        ):
            for key, sub_value in value.items():
                if key != "data":
                    check(f"{path}.{key}", sub_value)
            return
        malformed.append(path)

    for alias, value in entities.items():
        if alias != "data":
            check(alias, value)
    return malformed


def raw_entity_first_data(raw_yaml_strings: dict[str, str]) -> dict:
    """Parse raw YAML text into entity-first dicts, keyed by file identifier.

    Raises if any file's entities include a component given as a bare
    mapping instead of a list -- see `_find_malformed_entities`. A parse
    failure or non-dict top-level shape is left to normal YAML/downstream
    error handling rather than checked here.

    Parameters
    ----------
    raw_yaml_strings : dict[str, str]
        A dict keyed by file identifier, where each value is raw YAML text.

    Returns
    -------
    dict
        A dict keyed by file identifier, where each value is the dict of
        entities loaded from that file's YAML text.
    """
    result = {}
    for file_id, content in raw_yaml_strings.items():
        parsed = yaml.safe_load(content) or {}
        if isinstance(parsed, dict):
            malformed = _find_malformed_entities(parsed)
            if malformed:
                raise ValueError(
                    "Each entity's components must be a YAML list (a leading `- ` "
                    "before each component name), e.g. `alias:\n    - component:\n"
                    "        value: x` -- a bare mapping (`alias:\n    component:\n"
                    "        value: x`) is silently ignored rather than merged. "
                    f"Malformed in {file_id}: {', '.join(malformed)}."
                )
        result[file_id] = parsed
    return result


FINAL_VAR = "raw_entity_first_data"
