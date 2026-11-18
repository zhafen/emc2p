"""Build the mobile entity-graph viewer from emc2p's builtins + docs/manifest.

Usage: uv run python apps/entity_graph/build.py OUT.html [MANIFEST_DIR ...]
(OUT ending in .json writes the raw nodes/edges instead.)
"""
import sys

from emc2p.entity_graph import write_graph
from emc2p.registrar import Registrar

if __name__ == "__main__":
    out, manifests = sys.argv[1], sys.argv[2:] or ["docs/manifest"]
    print(f"wrote {write_graph(Registrar.from_manifest(manifests).registry, out)}")
