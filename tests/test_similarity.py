"""Tests for emc2p.similarity's cosine-similarity ranking and entity search.

No real embedding calls in most of this file: rank_by_similarity/
find_similar_entities tests use a small deterministic word-overlap fake
embed_fn (see _fake_embed below) -- the same "no real model calls"
approach tests/agents/test_tool_calling_loop.py already uses for
litellm.acompletion. These are ranking-mechanics tests, not a test of
any embedding model's actual semantic judgment. Only
TestEmbedWithLitellm exercises litellm.embedding's call shape, and only
with litellm's own call monkeypatched.
"""

import pytest

from emc2p.registrar import Registrar
from emc2p.similarity import (
    cosine_similarity,
    embed_with_litellm,
    find_duplicate_entities,
    find_duplicate_groups,
    find_similar_entities,
    pairwise_similarities,
    rank_by_similarity,
)
from tests.conftest import make_registry


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic word-count embedding: cosine similarity between two
    texts increases with word overlap. Vocabulary is every distinct word
    across `texts`, so results are only meaningful within one call --
    exactly how rank_by_similarity/find_similar_entities use it (query +
    all candidates embedded together in a single embed_fn call)."""
    vocab = sorted({word for text in texts for word in text.lower().split()})
    return [[text.lower().split().count(word) for word in vocab] for text in texts]


class TestCosineSimilarity:
    def test_identical_vectors_are_maximally_similar(self):
        assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_orthogonal_vectors_have_zero_similarity(self):
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_vectors_are_maximally_dissimilar(self):
        assert cosine_similarity([1, 2], [-1, -2]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero_not_a_crash(self):
        assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0
        assert cosine_similarity([1, 2, 3], [0, 0, 0]) == 0.0
        assert cosine_similarity([0, 0], [0, 0]) == 0.0


class TestRankBySimilarity:
    def _candidates(self):
        return {
            "cat": "feed the cat and water the cat",
            "car": "change the car oil",
            "unrelated": "quarterly database migration report",
        }

    def test_ranks_most_overlapping_text_first(self):
        ranked = rank_by_similarity("cat food", self._candidates(), _fake_embed)
        assert ranked[0][0] == "cat"

    def test_returns_all_candidates_ranked_by_default(self):
        ranked = rank_by_similarity("cat food", self._candidates(), _fake_embed)
        assert {cid for cid, _ in ranked} == set(self._candidates())

    def test_top_k_truncates(self):
        ranked = rank_by_similarity("cat food", self._candidates(), _fake_embed, top_k=1)
        assert len(ranked) == 1
        assert ranked[0][0] == "cat"

    def test_empty_candidates_returns_empty_list(self):
        assert rank_by_similarity("anything", {}, _fake_embed) == []

    def test_ties_broken_by_candidate_id(self):
        # Identical text -> identical embeddings -> tied score, so id order decides.
        candidates = {"z_candidate": "same text here", "a_candidate": "same text here"}
        ranked = rank_by_similarity("same text here", candidates, _fake_embed)
        assert [cid for cid, _ in ranked] == ["a_candidate", "z_candidate"]

    def test_embed_fn_called_once_with_query_and_all_candidates(self):
        calls = []

        def counting_embed(texts):
            calls.append(list(texts))
            return _fake_embed(texts)

        rank_by_similarity("query text", {"a": "text a", "b": "text b"}, counting_embed)
        assert calls == [["query text", "text a", "text b"]]


class TestFindSimilarEntities:
    def _registry(self):
        return make_registry({
            "description": [
                {"entity_id": "e1", "value": "feed the cat every morning"},
                {"entity_id": "e2", "value": "change the car oil regularly"},
                {"entity_id": "e3", "value": "quarterly database migration report"},
                {"entity_id": "e4", "value": None},  # no text -- should be skipped
            ],
        })

    def test_ranks_entities_by_description_similarity(self):
        results = find_similar_entities(self._registry(), "cat food", _fake_embed)
        assert results[0][0] == "e1"

    def test_skips_entities_with_no_text(self):
        results = find_similar_entities(self._registry(), "cat food", _fake_embed)
        assert "e4" not in [eid for eid, _ in results]

    def test_excludes_given_entity_ids(self):
        results = find_similar_entities(
            self._registry(), "cat food", _fake_embed, exclude_entity_ids=["e1"]
        )
        assert "e1" not in [eid for eid, _ in results]

    def test_top_k_limits_results(self):
        results = find_similar_entities(self._registry(), "cat food", _fake_embed, top_k=1)
        assert len(results) == 1

    def test_works_with_a_registrar_too(self):
        # Registrar.get is registry.get forwarded -- confirm the duck-typed
        # `registry.get(component_type)` call in find_similar_entities works
        # against a Registrar the same way it does a bare Registry.
        registrar = Registrar(self._registry())
        results = find_similar_entities(registrar, "cat food", _fake_embed)
        assert results[0][0] == "e1"


def _vector_embed(vectors: dict[str, tuple[float, ...]]):
    """A fake embed_fn with hand-picked vectors, keyed by text (candidate
    texts are just their own id in tests using this) -- lets a test
    control exact cosine similarities instead of relying on word overlap."""
    def embed_fn(texts: list[str]) -> list[list[float]]:
        return [list(vectors[text]) for text in texts]
    return embed_fn


class TestPairwiseSimilarities:
    def test_reports_each_unordered_pair_once(self):
        candidates = {"a": "apple", "b": "banana", "c": "cherry"}
        pairs = pairwise_similarities(candidates, _fake_embed)
        assert len(pairs) == 3  # 3 choose 2
        assert len({frozenset((a, b)) for a, b, _ in pairs}) == 3

    def test_threshold_filters_pairs(self):
        candidates = {
            "a": "apple pie recipe",
            "b": "apple pie recipe",
            "c": "unrelated database migration",
        }
        pairs = pairwise_similarities(candidates, _fake_embed, threshold=0.99)
        assert [(id_a, id_b) for id_a, id_b, _ in pairs] == [("a", "b")]

    def test_fewer_than_two_candidates_returns_empty(self):
        assert pairwise_similarities({}, _fake_embed) == []
        assert pairwise_similarities({"a": "text"}, _fake_embed) == []

    def test_embed_fn_called_once_with_every_candidate(self):
        calls = []

        def counting_embed(texts):
            calls.append(list(texts))
            return _fake_embed(texts)

        pairwise_similarities({"a": "text a", "b": "text b", "c": "text c"}, counting_embed)
        assert calls == [["text a", "text b", "text c"]]


class TestFindDuplicateGroups:
    def test_groups_transitively_even_when_the_endpoints_alone_would_not(self):
        # cos(a,b) ~= cos(b,c) ~= 0.71 (>= threshold), cos(a,c) = 0.0 (well
        # below) -- a and c only end up together because b bridges them.
        vectors = {"a": (1, 0, 0), "b": (1, 1, 0), "c": (0, 1, 0), "d": (0, 0, 1)}
        candidates = {cid: cid for cid in vectors}
        groups = find_duplicate_groups(candidates, _vector_embed(vectors), threshold=0.7)
        assert groups == [["a", "b", "c"]]

    def test_isolated_candidate_is_excluded_from_output(self):
        vectors = {"a": (1, 0, 0), "b": (1, 1, 0), "c": (0, 1, 0), "d": (0, 0, 1)}
        candidates = {cid: cid for cid in vectors}
        groups = find_duplicate_groups(candidates, _vector_embed(vectors), threshold=0.7)
        assert all("d" not in group for group in groups)

    def test_default_threshold_groups_near_identical_text(self):
        candidates = {
            "a": "feed the cat every morning",
            "b": "feed the cat every morning",
            "c": "change the car oil regularly",
        }
        assert find_duplicate_groups(candidates, _fake_embed) == [["a", "b"]]

    def test_no_duplicates_returns_empty_list(self):
        candidates = {"a": "apples", "b": "oranges", "c": "database migration"}
        assert find_duplicate_groups(candidates, _fake_embed, threshold=0.99) == []

    def test_groups_sorted_by_first_member(self):
        vectors = {"z": (1, 0), "y": (1, 0), "x": (0, 1)}
        candidates = {cid: cid for cid in vectors}
        groups = find_duplicate_groups(candidates, _vector_embed(vectors), threshold=0.5)
        assert groups == [["y", "z"]]


class TestFindDuplicateEntities:
    def _registry(self):
        return make_registry({
            "description": [
                {"entity_id": "e1", "value": "feed the cat every morning"},
                {"entity_id": "e2", "value": "feed the cat every morning"},
                {"entity_id": "e3", "value": "change the car oil regularly"},
                {"entity_id": "e4", "value": None},
            ],
        })

    def test_groups_duplicate_entities(self):
        assert find_duplicate_entities(self._registry(), _fake_embed) == [["e1", "e2"]]

    def test_skips_entities_with_no_text(self):
        groups = find_duplicate_entities(self._registry(), _fake_embed)
        assert all("e4" not in group for group in groups)

    def test_excludes_given_entity_ids(self):
        groups = find_duplicate_entities(
            self._registry(), _fake_embed, exclude_entity_ids=["e1"]
        )
        assert groups == []


class TestEmbedWithLitellm:
    def test_calls_litellm_and_extracts_embeddings(self, monkeypatch):
        pytest.importorskip("litellm", reason="requires the 'agents' extra")
        import litellm

        captured = {}

        def fake_embedding(model, input):
            captured["model"] = model
            captured["input"] = input
            return {"data": [{"embedding": [1.0, 2.0]}, {"embedding": [3.0, 4.0]}]}

        monkeypatch.setattr(litellm, "embedding", fake_embedding)

        result = embed_with_litellm(["a", "b"], model="test-model")

        assert result == [[1.0, 2.0], [3.0, 4.0]]
        assert captured == {"model": "test-model", "input": ["a", "b"]}
