"""Extend the web outward from the lateral family: each lateral relative's main inputs (one layer
back, on its own top corpus examples), with forward dumps for blind labelling.

  python -m bonsai_interp.probe.lateral_out 9/3688 +   ->  lateral_out_nodes.json, lateral_out_request.json
"""
import sys

import torch

from bonsai_interp.env import empty_cache, limit_memory
from bonsai_interp.model import load
from bonsai_interp.probe.quick import one_back
from bonsai_interp.probe.web import forward_dump
from bonsai_interp.probe.workspace import Workspace


def run(ws, k=3):
    spec = ws.load("lateral_spec.json")
    known = {nd["unit"] for nd in ws.load("web_nodes.json")["nodes"]} | {ws.load("web_nodes.json")["center"]}
    limit_memory(0.8)
    model, tok, rec = load(ws.model)
    nodes, edges = {}, []
    for nd in spec["nodes"]:
        if not nd.get("lateral"):
            continue
        u = f"L{nd['L']}/N{nd['n']}"
        feeders = one_back(model, rec, nd["L"], nd["n"], 0, k_ex=6, k_cand=32, k_show=k + 2)
        kept = [f for f in feeders if f["unit"] not in known][:k]
        print(f"{nd['name']:<13} {u}: feeders {[(f['unit'], round(f['drop'], 3), f['peaks_pos'][:3]) for f in feeders]}")
        for f in kept:
            L, n = map(int, f["unit"][1:].split("/N"))
            nodes.setdefault(f["unit"], {"unit": f["unit"], "L": L, "n": n, "pole": "+", "roles": ["feeds lateral"], "via": u,
                                         "examples": forward_dump(L, n)})
            edges.append({"from": f["unit"], "to": u, "kind": "feeds lateral", "w": f["drop"]})
        empty_cache()
    ws.save("lateral_out_nodes.json", {"nodes": list(nodes.values()), "edges": edges})
    ws.save("lateral_out_request.json", {"nodes": [{"unit": v["unit"], "pole": "+", "examples": v["examples"]} for v in nodes.values()]})
    print(len(nodes), "new nodes")


if __name__ == "__main__":
    L, n = map(int, sys.argv[1].split("/"))
    run(Workspace(L, n, sys.argv[2] if len(sys.argv) > 2 else "+"))
