#!/usr/bin/env python3
"""Find likely-duplicate component-type definitions in emc2p's own
builtins (emc2p/builtins/*.yaml), using emc2p.similarity.find_duplicate_entities
over every builtin entity's description.value text.

Real semantic embeddings need a configured provider (embed_with_litellm,
this script's default -- needs e.g. OPENAI_API_KEY/VOYAGE_API_KEY set,
whichever env var the chosen model needs; see litellm's own docs).
Without one, --offline falls back to a simple bag-of-words embedding
(lowercased, punctuation-stripped, stopword-filtered word counts) --
good enough to catch near-duplicate *wording*, not real semantic near-
duplicates phrased completely differently.

Run from the repo root:
    uv run python scripts/find_duplicate_component_types.py [--offline] [--threshold 0.9]
"""

from __future__ import annotations

import argparse
import string
import tempfile

from emc2p.registrar import Registrar
from emc2p.similarity import embed_with_litellm, find_duplicate_entities

_STOPWORDS = {
    "a", "an", "the", "of", "or", "and", "to", "in", "on", "for", "is",
    "are", "with", "by", "its", "this", "that", "as", "be", "not", "it",
    "at", "from", "into", "if", "eg", "ie", "when", "than",
}
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _tokenize(text: str) -> list[str]:
    words = text.lower().translate(_PUNCT_TABLE).split()
    return [w for w in words if w not in _STOPWORDS]


def offline_embed(texts: list[str]) -> list[list[float]]:
    """Bag-of-words fallback: no external calls, no API key needed.
    Catches near-duplicate *wording*, not real semantic paraphrase --
    use embed_with_litellm (this script's default) for that."""
    tokenized = [_tokenize(t) for t in texts]
    vocab = sorted({word for tokens in tokenized for word in tokens})
    return [[tokens.count(word) for word in vocab] for tokens in tokenized]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline", action="store_true",
        help="Use the bag-of-words fallback instead of a real embedding provider.",
    )
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()

    # emc2p/builtins is *always* auto-included by Registrar.from_manifest
    # (see load_manifest.py's own docstring) -- passing it again as an
    # input dir would double-load every entity in it. An empty temp dir
    # gets just the auto-included builtins, nothing more.
    with tempfile.TemporaryDirectory() as empty_dir:
        registrar = Registrar.from_manifest([empty_dir])
    embed_fn = offline_embed if args.offline else embed_with_litellm

    groups = find_duplicate_entities(registrar, embed_fn, threshold=args.threshold)

    eids = registrar.get("entity_id").execute()
    path_of = dict(zip(eids["value"].astype(str), eids["path"].astype(str)))
    desc = registrar.get("description").execute()
    text_of = dict(zip(desc["entity_id"].astype(str), desc["value"].astype(str)))

    if not groups:
        print(f"No likely-duplicate component definitions found (threshold={args.threshold}).")
        return

    print(f"{len(groups)} likely-duplicate group(s) (threshold={args.threshold}):\n")
    for group in groups:
        print("  Group:")
        for eid in group:
            name = path_of.get(eid, eid).split(":", 1)[-1]
            text = text_of.get(eid, "")
            print(f"    - {name}: {text[:80]!r}")
        print()


if __name__ == "__main__":
    main()
