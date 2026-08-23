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
(deciding which existing requirement/solution entity a new, naturally-
described entity is most related to -- see iacs's own
``requirement_solution_modeling_discussion``) is just one caller of
``find_similar_entities``, not baked into it. It's the retrieval half of
a retrieval-narrowed-then-agent-judged placement workflow: this module
only ranks candidates by similarity, it doesn't decide anything.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import pandas as pd

EmbedFn = Callable[[list[str]], list[list[float]]]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length vectors, in [-1, 1].

    0.0 if either vector is all-zero -- an all-zero vector has no
    direction to compare, so "no similarity claim" is more honest than
    an arbitrary tie-break value.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


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
    embeddings = embed_fn(texts)
    query_embedding, candidate_embeddings = embeddings[0], embeddings[1:]
    scored = [
        (cid, cosine_similarity(query_embedding, cemb))
        for cid, cemb in zip(ids, candidate_embeddings)
    ]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored[:top_k] if top_k is not None else scored


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
    table = registry.get(component_type)
    df = table.execute() if hasattr(table, "execute") else table
    exclude = set(exclude_entity_ids)
    candidates = {
        str(row["entity_id"]): str(row[field])
        for _, row in df.iterrows()
        if str(row["entity_id"]) not in exclude and not pd.isna(row[field])
    }
    return rank_by_similarity(query, candidates, embed_fn, top_k=top_k)


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
