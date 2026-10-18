"""Stage 8 (web): crawl the probed pole's neighbourhood breadth-first.

  discover   readers (zero-ablation, boost, weights-only; per branch, kept if >= 2 methods
             agree), siblings (other main inputs of each reader = lateral relatives),
             co-firing neurons on the probe's sentences, and second-hop upstream feeders.
             Writes web_nodes.json (nodes, edges, forward examples for labelling).

  python -m bonsai_interp.probe.web discover 9/3688 +
"""
import json
import sys

import numpy as np
import torch

from bonsai_interp import explore
from bonsai_interp.env import DEVICE, empty_cache, limit_memory
from bonsai_interp.inverse.linear import Weights
from bonsai_interp.model import StopForward, load
from bonsai_interp.probe import textinv
from bonsai_interp.probe.neighbors import ids_and_peak
from bonsai_interp.probe.quick import one_back
from bonsai_interp.probe.workspace import Workspace

DOWN = 12          # layers after the probed one
TOP_RANK = 60      # a reader must be in the top-60 of at least two methods
N_READERS = 6
N_SIBLINGS = 4
N_COFIRE = 10


def _zero_or_set(a, n, pos, value):
    a = a.clone()
    if pos is None:
        a[..., n] = value
    else:
        a[:, pos, n] = value
    return a


@torch.no_grad()
def reader_effects(model, rec, examples, L, n, pmax, scan, window=3):
    """Per later neuron: mean signed change (fraction of its pole max) under zeroing the probed
    neuron everywhere, and under setting it to 2x its corpus max at the peak position."""
    layers = list(range(L + 1, min(L + 1 + DOWN, model.config.num_hidden_layers)))
    acc = {"zero": {l: [] for l in layers}, "boost": {l: [] for l in layers}}
    for ids, pos in examples:
        x = torch.tensor(np.asarray(ids[: pos + window])[None], device=DEVICE)
        res = {}
        for cond, fn in (("clean", None), ("zero", lambda a: _zero_or_set(a, n, None, 0.0)),
                         ("boost", lambda a: _zero_or_set(a, n, pos, 2 * pmax))):
            rec.mode, rec.layers, rec.stop_at = "act", set(layers), layers[-1]
            rec.interventions = {L: fn} if fn else {}
            try:
                model.model(x)
            except StopForward:
                pass
            rec.mode, rec.stop_at, rec.interventions = None, None, {}
            res[cond] = {l: rec.store.pop(l)[0, pos: pos + window].float() for l in layers}
            rec.clear()
        for l in layers:
            for cond, sign in (("zero", 1.0), ("boost", -1.0)):
                d = (res["clean"][l] - res[cond][l]) * sign  # contribution of the probed neuron / of the boost
                idx = d.abs().argmax(0)
                dd = d.gather(0, idx[None])[0].cpu().numpy()
                pm = np.where(dd >= 0, scan["top_val"][l, 0, 0], scan["top_val"][l, 1, 0]).astype(np.float64)
                acc[cond][l].append(dd / np.maximum(pm, 1e-6))
    return layers, {c: np.stack([np.mean(acc[c][l], 0) for l in layers]) for c in acc}


def weight_scores(L, n, layers):
    """|cos| between the probed neuron's write direction and each later neuron's gate/up read rows."""
    W = Weights("bonsai1")
    d = W.get(f"model.layers.{L}.mlp.down_proj.weight")[:, n]
    d = d / d.norm()
    out = []
    for l in layers:
        g = W.get(f"model.layers.{l}.post_attention_layernorm.weight")
        cg = (W.get(f"model.layers.{l}.mlp.gate_proj.weight") * g) @ d
        cu = (W.get(f"model.layers.{l}.mlp.up_proj.weight") * g) @ d
        ng = (W.get(f"model.layers.{l}.mlp.gate_proj.weight") * g).norm(dim=1)
        nu = (W.get(f"model.layers.{l}.mlp.up_proj.weight") * g).norm(dim=1)
        out.append(torch.maximum((cg / ng).abs(), (cu / nu).abs()).numpy())
    return np.stack(out)


def rank_set(arr, k):
    flat = np.abs(arr).ravel()
    return set(np.argsort(-flat)[:k].tolist())


@torch.no_grad()
def cofire(model, tok, rec, texts, L, n, layers):
    """Pearson r between the probed pole's per-sentence max activation and every neuron's, on the
    probe's own sentences (after a neutral preamble)."""
    pre = textinv.PREAMBLES[0]
    k = len(tok(pre).input_ids)
    maxes = {l: [] for l in layers}
    for s in range(0, len(texts), 8):
        enc = tok([pre + t for t in texts[s:s + 8]], return_tensors="pt", padding=True).to(DEVICE)
        rec.mode, rec.layers, rec.stop_at = "act", set(layers), max(layers)
        try:
            model.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        except StopForward:
            pass
        rec.mode, rec.stop_at = None, None
        m = enc.attention_mask.bool().clone()
        m[:, :k] = False
        for l in layers:
            a = rec.store.pop(l).float()
            maxes[l].append(torch.where(m[..., None], a, torch.full_like(a, -1e9)).max(1).values.cpu())
        rec.clear()
    A = {l: torch.cat(v).numpy() for l, v in maxes.items()}
    y = A[L][:, n]
    y = (y - y.mean()) / (y.std() + 1e-9)
    out = []
    for l in layers:
        X = A[l]
        Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
        r = (Xz * y[:, None]).mean(0)
        if l == L:
            r[n] = 0
        for j in np.argsort(-r)[:N_COFIRE]:
            out.append({"unit": f"L{l}/N{int(j)}", "L": l, "n": int(j), "r": float(r[j])})
    out.sort(key=lambda r: -r["r"])
    return out[:N_COFIRE]


def forward_dump(L, n, k=8):
    return {pole: [t[-220:] for *_, t in explore.examples("bonsai1", L, n, pi, k, 90, 15)] for pole, pi in (("+", 0), ("-", 1))}


def discover(ws: Workspace):
    limit_memory(0.8)
    model, tok, rec = load(ws.model)
    tok.padding_side = "right"
    scan = explore.scan(ws.model)
    L, n = ws.L, ws.n
    pmax = float(scan["top_val"][L, 0, 0, n])
    spec = ws.load("neighbors_request.json")
    nodes, edges = {}, []

    def add(unit, role, pole, **extra):
        if unit == f"L{L}/N{n}":
            return
        nd = nodes.setdefault(unit, {"unit": unit, "L": int(unit[1:].split("/")[0]), "n": int(unit.split("N")[-1]),
                                     "roles": [], "pole": pole})
        if role not in nd["roles"]:
            nd["roles"].append(role)
        nd.update(extra)

    # ---- known first-hop nodes (from neighbors.json / quick.json)
    q = ws.load("quick.json")["neighbours"]
    for nb in q:
        add(nb["unit"], {"main input (74%)": "input", "second input (61%)": "input", "context gate (pain)": "gate"}.get(nb["role"], "reader"),
            nb["pole"], known_label=nb["label"], quick_existing=nb["survives"])

    # ---- 1. readers, three methods, per branch
    readers = {}
    for branch in ("psychological", "anatomical"):
        ex = []
        for t in spec["branches"][branch]:
            ids, pos, _ = ids_and_peak(model, tok, rec, t, L, n, 1.0)
            ex.append((ids, pos))
        layers, eff = reader_effects(model, rec, ex, L, n, pmax, scan)
        readers[branch] = eff
    wts = weight_scores(L, n, layers)
    N = model.config.intermediate_size
    psy = readers["psychological"]
    sets = [rank_set(psy["zero"], TOP_RANK), rank_set(psy["boost"], TOP_RANK), rank_set(wts, TOP_RANK)]
    votes = {}
    for s in sets:
        for i in s:
            votes[i] = votes.get(i, 0) + 1
    cand = [i for i, v in votes.items() if v >= 2]
    cand.sort(key=lambda i: -abs(psy["boost"].ravel()[i]))
    reader_rows = []
    for i in cand[:N_READERS]:
        li, j = divmod(i, N)
        l = layers[li]
        b = float(psy["boost"][li, j])
        unit = f"L{l}/N{j}"
        row = {"unit": unit, "zero": float(psy["zero"][li, j]), "boost": b, "cos": float(wts[li, j]),
               "zero_anat": float(readers["anatomical"]["zero"][li, j]), "votes": votes[i]}
        reader_rows.append(row)
        add(unit, "reader", "+" if b >= 0 else "-")
        edges.append({"from": f"L{L}/N{n}", "to": unit, "kind": "reads", "w": b, "detail": row})
    print("readers:", [(r["unit"], round(r["boost"], 3), round(r["zero"], 3), round(r["cos"], 3), r["votes"]) for r in reader_rows])

    # ---- 2. siblings: other main inputs of each reader
    for r in reader_rows:
        Lr, nr = int(r["unit"][1:].split("/")[0]), int(r["unit"].split("N")[-1])
        pole_r = 0 if r["boost"] >= 0 else 1
        feeders = one_back(model, rec, Lr, nr, pole_r, k_ex=6, k_cand=32, k_show=N_SIBLINGS + 2)
        kept = [f for f in feeders if f["unit"] != f"L{L}/N{n}"][:N_SIBLINGS]
        r["feeders"] = feeders
        for f in kept:
            add(f["unit"], "sibling", "+", via=r["unit"])
            edges.append({"from": f["unit"], "to": r["unit"], "kind": "feeds reader", "w": f["drop"]})
        print(f"siblings via {r['unit']}:", [(f['unit'], round(f['drop'], 3)) for f in feeders])
        empty_cache()

    # ---- 3. co-firing on the probe's own sentences
    sol = ws.load("solo.json")
    texts = [x["text"] for x in sol["rows"]] + [x["text"] for x in sol["holdout"]["rows"]]
    cf = cofire(model, tok, rec, texts, L, n, list(range(max(1, L - 4), L + 6)))
    for c in cf:
        add(c["unit"], "co-fires", "+", r=c["r"])
        edges.append({"from": f"L{L}/N{n}", "to": c["unit"], "kind": "co-fires", "w": c["r"]})
    print("co-firing:", [(c["unit"], round(c["r"], 2)) for c in cf])

    # ---- 4. second-hop upstream: strong feeders of nerve's inputs and gate
    for nb in q:
        if nb["role"].startswith("strongest reader"):
            continue
        for f in nb["feeders"]:
            if f["drop"] >= 0.04:
                add(f["unit"], "hop-2 upstream", "+", via=nb["unit"])
                edges.append({"from": f["unit"], "to": nb["unit"], "kind": "feeds", "w": f["drop"]})
    # first-hop edges for completeness
    nbj = ws.load("neighbors.json")
    for r in nbj["upstream"][:2]:
        edges.append({"from": r["unit"], "to": f"L{L}/N{n}", "kind": "feeds", "w": r["drop_mean"]})
    edges.append({"from": "L8/N3858", "to": f"L{L}/N{n}", "kind": "gates", "w": -0.25})

    # ---- forward dumps for labelling
    for u, nd in nodes.items():
        nd["examples"] = forward_dump(nd["L"], nd["n"])
        nd["max"] = {"+": float(scan["top_val"][nd["L"], 0, 0, nd["n"]]), "-": float(scan["top_val"][nd["L"], 1, 0, nd["n"]])}
    ws.save("web_nodes.json", {"center": f"L{L}/N{n}", "nodes": list(nodes.values()), "edges": edges, "readers": reader_rows,
                               "cofire": cf})
    print(f"{len(nodes)} nodes, {len(edges)} edges -> web_nodes.json")


if __name__ == "__main__":
    L, n = map(int, sys.argv[2].split("/"))
    ws = Workspace(L, n, sys.argv[3] if len(sys.argv) > 3 else "+")
    {"discover": discover}[sys.argv[1]](ws)
