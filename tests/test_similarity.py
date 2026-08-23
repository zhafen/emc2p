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
    find_similar_entities,
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
