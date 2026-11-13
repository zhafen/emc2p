"""Build the mobile entity-graph viewer from emc2p's builtins + docs/manifest.

Usage: uv run python apps/entity_graph/build.py OUT.html [MANIFEST_DIR ...]
"""
import json
import math
import sys
from pathlib import Path

from emc2p.registrar import Registrar

HERE = Path(__file__).parent


def clean(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if hasattr(v, "item"):
        v = v.item()
    return v.isoformat() if hasattr(v, "isoformat") else v


def export(manifest):
    reg = Registrar.from_manifest(manifest).registry
    nodes = {
        r.value: dict(id=r.value, name=r.display_key, alias=clean(r.display_alias),
                      path=r.path, file=clean(r.filepath), components=[])
        for r in reg.get("entity_id").to_pandas().itertuples()
    }
    edges = []
    for t in reg.component_types:
        if t in ("entity_id", "invalid_field"):
            continue
        for row in reg.get(t).to_pandas().to_dict("records"):
            eid = row.pop("entity_id")
            f = {k: clean(v) for k, v in row.items()
                 if not k.startswith("_") and k != "component_index" and clean(v) is not None}
            if not f.get("modifier"):
                f.pop("modifier", None)
            for k, v in f.items():
                if k.endswith("_eid") and v in nodes:
                    edges.append(dict(source=eid, target=v, type=t))
            if eid in nodes:
                nodes[eid]["components"].append(dict(type=t, fields=f))
    return dict(nodes=list(nodes.values()), edges=edges)


if __name__ == "__main__":
    out, manifests = sys.argv[1], sys.argv[2:] or ["docs/manifest"]
    data = json.dumps(export(manifests), default=str).replace("</", "<\\/")
    html = (HERE / "template.html").read_text().replace("/*__DATA__*/null", data)
    Path(out).write_text(html)
    print(f"wrote {out}")
