"""Find entities (or arbitrary text) similar to a query by embedding
cosine similarity.

Embedding-provider-agnostic: every function takes an ``embed_fn``
(``list[str] -> list[list[float]]``) rather than calling a specific
provider directly, so callers can inject a fast, deterministic fake for
tests (see ``tests/test_similarity.py``) instead of making real network
calls. ``embed_with_litellm`` is the default production ``embed_fn``,
gated behind the "agents" extra (litellm) the same way
``emc2p.agents.tool_calling_loop`` already is.

General-purpose, not entity-placement-specific: the use case driving this
(deciding which existing entity a new, naturally-described one is most
related to, in order to link it in appropriately) is just one caller of
``find_similar_entities``, not baked into it. It's the retrieval half of
a retrieval-narrowed-then-agent-judged placement workflow: this module
only ranks candidates by similarity, it doesn't decide anything.

``pairwise_similarities``/``find_duplicate_groups`` extend the same
primitives to the "considering multiple entities at once" case: rather
than ranking candidates against one query, they compare every candidate
against every other one to surface likely duplicates -- one entity/type
definition that's really the same thing as another, described twice.
``find_duplicate_groups`` closes matches transitively (A~B, B~C -> one
group of {A, B, C}) via union-find, the same approach an entity-
deduplication pipeline typically needs to resolve transitive duplicates
before finalizing a merge, rather than leaving the caller to reconstruct
groups from a flat pile of pairwise edges by hand.

Similarity math is plain numpy (already an effective dependency via
pandas/ibis, so this doesn't add a new one) rather than a vector-database/
ANN library: candidate counts here are architecture-manifest-sized (an
emc2p/iacs/story-simulator project's own entities, not a general text
corpus), so an in-memory brute-force comparison is the right-sized tool
-- an ANN index earns its cost at a scale this module never sees.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

EmbedFn = Callable[[list[str]], list[list[float]]]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length vectors, in [-1, 1].

    0.0 if either vector is all-zero -- an all-zero vector has no
    direction to compare, so "no similarity claim" is more honest than
    an arbitrary tie-break value.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def _pairwise_cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between every row of ``matrix`` and every other
    row, as one vectorized matrix product -- the shared core of
    ``rank_by_similarity``/``pairwise_similarities``' own all-pairs math.

    Row pairs where either row is all-zero score 0.0 (see
    ``cosine_similarity``), not NaN from a division by zero.
    """
    norms = np.linalg.norm(matrix, axis=1)
    norm_products = np.outer(norms, norms)
    with np.errstate(invalid="ignore", divide="ignore"):
        similarities = (matrix @ matrix.T) / norm_products
    return np.where(norm_products == 0, 0.0, similarities)


def rank_by_similarity(
    query: str,
    candidates: dict[str, str],
    embed_fn: EmbedFn,
    *,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Rank ``candidates`` (id -> text) by cosine similarity to
    ``query``'s embedding, most similar first.

    ``embed_fn`` is called once with the query and every candidate text
    together, so a batching ``embed_fn`` only pays one round trip rather
    than one per candidate.

    Parameters
    ----------
    query : str
    candidates : dict[str, str]
        Candidate id -> text to compare against ``query``.
    embed_fn : EmbedFn
    top_k : int | None
        Truncate to the top ``top_k`` results. ``None`` returns all of
        them, ranked.

    Returns
    -------
    list[tuple[str, float]]
        (candidate_id, similarity) pairs, most similar first. Ties
        broken by candidate id for a stable, reproducible order.
    """
    if not candidates:
        return []
    ids = list(candidates)
    texts = [query] + [candidates[i] for i in ids]
    embeddings = np.asarray(embed_fn(texts), dtype=float)
    similarities = _pairwise_cosine_similarity(embeddings)[0, 1:]
    scored = list(zip(ids, similarities.tolist()))
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored[:top_k] if top_k is not None else scored


def _entity_texts(
    registry,
    *,
    component_type: str,
    field: str,
    exclude_entity_ids: Sequence[str] = (),
) -> dict[str, str]:
    """entity_id -> ``component_type.field`` text for every entity in
    ``registry`` that has one, skipping ``exclude_entity_ids`` and
    entities missing the field entirely (e.g. a tag component with no
    description) -- there's no text to embed for those."""
    table = registry.get(component_type)
    df = table.execute() if hasattr(table, "execute") else table
    exclude = set(exclude_entity_ids)
    return {
        str(row["entity_id"]): str(row[field])
        for _, row in df.iterrows()
        if str(row["entity_id"]) not in exclude and not pd.isna(row[field])
    }


def find_similar_entities(
    registry,
    query: str,
    embed_fn: EmbedFn,
    *,
    component_type: str = "description",
    field: str = "value",
    top_k: int = 10,
    exclude_entity_ids: Sequence[str] = (),
) -> list[tuple[str, float]]:
    """Entity IDs in ``registry`` most similar to ``query``, ranked by
    cosine similarity between ``query``'s embedding and each entity's
    ``component_type.field`` text embedding (default: ``description.value``).

    Entities missing ``component_type.field`` (e.g. a tag component with
    no description) are silently skipped -- there's no text to embed.

    Parameters
    ----------
    registry : Registry | Registrar
        Anything exposing ``.get(component_type)`` -- both work.
    query : str
        Free-text description of what to find similar entities to.
    embed_fn : EmbedFn
        ``embed_with_litellm`` is the default production choice; tests
        should inject a fast, deterministic fake instead of making real
        API calls (see this module's own docstring).
    component_type, field : str
        Which component table/column holds each entity's comparison text.
    top_k : int
    exclude_entity_ids : Sequence[str]
        Entity IDs to skip -- e.g. the entity being placed itself, if
        it's already in the registry.

    Returns
    -------
    list[tuple[str, float]]
        (entity_id, similarity) pairs, most similar first.
    """
    candidates = _entity_texts(
        registry, component_type=component_type, field=field, exclude_entity_ids=exclude_entity_ids
    )
    return rank_by_similarity(query, candidates, embed_fn, top_k=top_k)


def pairwise_similarities(
    candidates: dict[str, str],
    embed_fn: EmbedFn,
    *,
    threshold: float | None = None,
) -> list[tuple[str, str, float]]:
    """Cosine similarity between every pair of ``candidates``, most
    similar first.

    ``embed_fn`` is called once with every candidate text, regardless of
    how many candidates there are -- one round trip, not one per pair.

    Parameters
    ----------
    candidates : dict[str, str]
        Candidate id -> text.
    embed_fn : EmbedFn
    threshold : float | None
        Drop any pair scoring below this. ``None`` keeps every pair.

    Returns
    -------
    list[tuple[str, str, float]]
        (id_a, id_b, similarity) triples, each unordered pair reported
        once (id_a < id_b), most similar first. Ties broken by
        (id_a, id_b) for a stable, reproducible order.
    """
    ids = list(candidates)
    if len(ids) < 2:
        return []
    embeddings = np.asarray(embed_fn([candidates[i] for i in ids]), dtype=float)
    similarities = _pairwise_cosine_similarity(embeddings)
    pairs = []
    for i, id_a in enumerate(ids):
        for j in range(i + 1, len(ids)):
            score = float(similarities[i, j])
            if threshold is None or score >= threshold:
                pairs.append((id_a, ids[j], score))
    pairs.sort(key=lambda triple: (-triple[2], triple[0], triple[1]))
    return pairs


def find_duplicate_groups(
    candidates: dict[str, str],
    embed_fn: EmbedFn,
    *,
    threshold: float = 0.9,
) -> list[list[str]]:
    """Group ``candidates`` into likely-duplicate clusters.

    Any two candidates scoring >= ``threshold`` land in the same group,
    and groups are transitively closed via union-find -- if A~B and B~C
    both clear the threshold, A/B/C end up in one group even if A and C
    alone wouldn't have. A candidate with no match above threshold isn't
    included in the output at all (nothing to report -- it's not a
    duplicate of anything here).

    Parameters
    ----------
    candidates : dict[str, str]
        Candidate id -> text.
    embed_fn : EmbedFn
    threshold : float
        Similarity above which two candidates count as duplicates.
        Cosine similarity between two independently-worded descriptions
        of the same thing is rarely much above 0.9, and everyday
        unrelated text rarely reaches it, so 0.9 is a reasonable
        starting point -- tune per embedding model/domain.

    Returns
    -------
    list[list[str]]
        Each likely-duplicate group (2+ candidate ids, sorted), sorted
        by their first (smallest) id, for a stable, reproducible order.
    """
    pairs = pairwise_similarities(candidates, embed_fn, threshold=threshold)
    parent = {cid: cid for cid in candidates}

    def find(cid: str) -> str:
        while parent[cid] != cid:
            parent[cid] = parent[parent[cid]]
            cid = parent[cid]
        return cid

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for id_a, id_b, _ in pairs:
        union(id_a, id_b)

    groups: dict[str, list[str]] = {}
    for cid in candidates:
        groups.setdefault(find(cid), []).append(cid)

    return sorted(
        (sorted(members) for members in groups.values() if len(members) > 1),
        key=lambda members: members[0],
    )


def find_duplicate_entities(
    registry,
    embed_fn: EmbedFn,
    *,
    component_type: str = "description",
    field: str = "value",
    threshold: float = 0.9,
    exclude_entity_ids: Sequence[str] = (),
) -> list[list[str]]:
    """Group entities in ``registry`` into likely-duplicate clusters, by
    cosine similarity between their ``component_type.field`` text
    (default: ``description.value``). See ``find_duplicate_groups`` for
    the grouping/threshold semantics; ``find_similar_entities`` for what
    ``component_type``/``field``/``exclude_entity_ids`` mean here.
    """
    candidates = _entity_texts(
        registry, component_type=component_type, field=field, exclude_entity_ids=exclude_entity_ids
    )
    return find_duplicate_groups(candidates, embed_fn, threshold=threshold)


def embed_with_litellm(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Default production ``embed_fn``: batched embeddings via litellm
    (any provider it supports).

    Import kept local so this module's pure-Python ranking functions stay
    usable without the "agents" extra -- only this one function needs
    litellm installed.
    """
    import litellm

    response = litellm.embedding(model=model, input=texts)
    return [item["embedding"] for item in response["data"]]
