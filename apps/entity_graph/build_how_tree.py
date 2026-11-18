"""Build the drill-down how-chain viewer from a registry's how/how_link links.

Usage: uv run python apps/entity_graph/build_how_tree.py OUT.html ROOT_KEY MANIFEST_DIR [MANIFEST_DIR ...]
e.g. for story-simulator: ... out.html how_chain_mission
  ../story-simulator/docs/how_chain ../story-simulator/story_simulator
  ../story-simulator/tests ../story-simulator/scripts
"""
import sys
from pathlib import Path

from emc2p.entity_graph import registry_to_how_tree, render_how_tree_html
from emc2p.registrar import Registrar

if __name__ == "__main__":
    out, root, manifests = sys.argv[1], sys.argv[2], sys.argv[3:]
    tree = registry_to_how_tree(Registrar.from_manifest(manifests).registry, root=root)
    Path(out).write_text(render_how_tree_html(tree))
    print(f"wrote {out} ({len(tree['nodes'])} nodes)")
