import pandas as pd
import pytest

from emc2p.utils import candidate_entity_ids


@pytest.fixture
def entity_id_df():
    return pd.DataFrame([
        {"value": "aaa111aaa111", "path": "examples/minimal2/minimal2.yaml:core_requirement.first_subrequirement"},
        {"value": "bbb222bbb222", "path": "examples/minimal2/minimal2.yaml:core_requirement.second_subrequirement"},
        {"value": "ccc333ccc333", "path": "examples/minimal2/minimal2.yaml:core_requirement"},
        {"value": "ddd444ddd444", "path": "examples/minimal2/minimal2.yaml:my_infrastructure"},
    ])


def test_exact_full_path_match(entity_id_df):
    result = candidate_entity_ids(
        "core_requirement.first_subrequirement", entity_id_df
    )
    assert result == ["aaa111aaa111"]


def test_partial_substring_match(entity_id_df):
    result = candidate_entity_ids("first_subrequirement", entity_id_df)
    assert result == ["aaa111aaa111"]


def test_no_match_returns_empty(entity_id_df):
    result = candidate_entity_ids("nonexistent_entity", entity_id_df)
    assert result == []


def test_ambiguous_matches_returns_multiple(entity_id_df):
    result = candidate_entity_ids("subrequirement", entity_id_df)
    assert set(result) == {"aaa111aaa111", "bbb222bbb222"}


@pytest.fixture
def entity_id_df_with_alias():
    return pd.DataFrame([
        {
            "value": "ccc333ccc333",
            "alias": "core_requirement",
            "path": "examples/minimal2/minimal2.yaml:core_requirement",
        },
        {
            "value": "aaa111aaa111",
            "alias": "core_requirement.first_subrequirement",
            "path": "examples/minimal2/minimal2.yaml:core_requirement.first_subrequirement",
        },
    ])


def test_exact_alias_match_preferred_over_substring(entity_id_df_with_alias):
    result = candidate_entity_ids("core_requirement.first_subrequirement", entity_id_df_with_alias)
    assert result == ["aaa111aaa111"]


def test_container_alias_resolves_despite_being_path_prefix_of_child(entity_id_df_with_alias):
    """A container's own alias is a substring of its child's path too
    ("core_requirement" vs "core_requirement.first_subrequirement").

    The exact-alias match must resolve it unambiguously to the
    container despite that.
    """
    result = candidate_entity_ids("core_requirement", entity_id_df_with_alias)
    assert result == ["ccc333ccc333"]


def test_falls_back_to_substring_when_no_exact_alias_match(entity_id_df_with_alias):
    result = candidate_entity_ids("first_subrequirement", entity_id_df_with_alias)
    assert result == ["aaa111aaa111"]


def test_exact_hash_match_preferred_over_alias_and_substring(entity_id_df_with_alias):
    """A raw entity_id hash resolves to itself, even before the
    alias/path resolutions are tried.

    So any entity_ref-typed field (or same_as.value) can reference an
    entity precisely by hash, the same as same_as.target_entity_id
    does explicitly.
    """
    result = candidate_entity_ids("aaa111aaa111", entity_id_df_with_alias)
    assert result == ["aaa111aaa111"]
