"""Analysis for the neighbourhood web.

  sources   Claude-written vs Bonsai-written tests on the same labels: fire rate, false-positive
            rate, mean activation, and Bonsai's own perplexity on each source's sentences.
  bank      every web neuron's activation on the probed neuron's sentence bank (+ held-out),
            so each node can be drawn as an iris of the same sentences.
  edges     direct coupling among a cluster: zero one node, measure the others, on given texts.

  python -m bonsai_interp.probe.web_analysis sources|bank 9/3688 +
"""
import json
import sys

import numpy as np
import torch

from bonsai_interp import explore
from bonsai_interp.device import DEVICE
from bonsai_interp.model import load
from bonsai_interp.probe import textinv
from bonsai_interp.probe.workspace import Workspace

META = ("feature detector", "the model", "detector", "language model")


@torch.no_grad()
def perplexity(model, tok, texts, batch=16):
    out = []
    tok.padding_side = "right"
    for s in range(0, len(texts), batch):
        enc = tok(texts[s:s + batch], return_tensors="pt", padding=True).to(DEVICE)
        logits = model(**enc).logits.float()
        lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, enc.input_ids[:, 1:, None])[..., 0]
        m = enc.attention_mask[:, 1:].float()
        out += (-(lp * m).sum(1) / m.sum(1)).exp().tolist()
    return out


def sources(ws: Workspace):
    cl = {n["unit"]: n for n in ws.load("web_quick_claude.json")["neighbours"]}
    for n in ws.load("quick.json")["neighbours"]:  # first-hop nodes were tested with Claude sentences earlier
        cl.setdefault(n["unit"], n)
    bo = {n["unit"]: n for n in ws.load("web_quick_bonsai.json")["neighbours"]}
    rows = []
    for u in bo:
        if u not in cl:
            continue
        c, b = cl[u], bo[u]
        f = lambda xs: float(np.mean([x["act"] >= 0.3 for x in xs])) if xs else None
        m = lambda xs: float(np.mean([x["act"] for x in xs])) if xs else None
        rows.append({"unit": u, "label": c["label"], "confidence": c.get("confidence"),
                     "claude_fire": f(c["fire"]), "bonsai_fire": f(b["fire"]),
                     "claude_fp": f(c["quiet"]), "bonsai_fp": f(b["quiet"]),
                     "claude_mean_fire": m(c["fire"]), "bonsai_mean_fire": m(b["fire"]),
                     "claude_survives": c["survives"], "bonsai_survives": b["survives"],
                     "bonsai_meta_leaks": sum(any(k in x["text"].lower() for k in META) for x in b["fire"] + b["quiet"])})
    # perplexity of each source's sentences under the model being probed
    model, tok, rec = load(ws.model)
    ctexts = [x["text"] for u in bo if u in cl for x in cl[u]["fire"] + cl[u]["quiet"]]
    btexts = [x["text"] for u in bo if u in cl for x in bo[u]["fire"] + bo[u]["quiet"]]
    pc, pb = perplexity(model, tok, ctexts), perplexity(model, tok, btexts)
    d = lambda k: [r[k] for r in rows if r[k] is not None]
    summary = {
        "n_labels": len(rows),
        "fire_rate": {"claude": float(np.mean(d("claude_fire"))), "bonsai": float(np.mean(d("bonsai_fire")))},
        "false_pos_rate": {"claude": float(np.mean(d("claude_fp"))), "bonsai": float(np.mean(d("bonsai_fp")))},
        "mean_fire_activation": {"claude": float(np.mean(d("claude_mean_fire"))), "bonsai": float(np.mean(d("bonsai_mean_fire")))},
        "labels_surviving": {"claude": sum(r["claude_survives"] for r in rows), "bonsai": sum(r["bonsai_survives"] for r in rows)},
        "agree_on_verdict": sum(r["claude_survives"] == r["bonsai_survives"] for r in rows),
        "paired_fire_bonsai_higher": sum(r["bonsai_fire"] > r["claude_fire"] for r in rows),
        "paired_fire_claude_higher": sum(r["claude_fire"] > r["bonsai_fire"] for r in rows),
        "perplexity_median": {"claude": float(np.median(pc)), "bonsai": float(np.median(pb))},
        "bonsai_meta_leak_sentences": sum(r["bonsai_meta_leaks"] for r in rows),
    }
    ws.save("web_sources.json", {"summary": summary, "rows": rows})
    print(json.dumps(summary, indent=1))


@torch.no_grad()
def bank(ws: Workspace):
    W = ws.load("web_nodes.json")
    scan = explore.scan(ws.model)
    units = [W["center"]] + [n["unit"] for n in W["nodes"]]
    poles = {W["center"]: ws.pole, **{n["unit"]: n["pole"] for n in W["nodes"]}}
    nodes = []
    for u in units:
        L, n = int(u[1:].split("/")[0]), int(u.split("N")[-1])
        p = 0 if poles[u] == "+" else 1
        nodes.append((u, L, n, 1.0 if p == 0 else -1.0, float(scan["top_val"][L, p, 0, n])))
    model, tok, rec = load(ws.model)
    tok.padding_side = "right"
    ev = textinv.CircuitEvaluator(model, tok, rec, nodes, batch=8)
    sol = ws.load("solo.json")
    rows = [{"text": r["text"], "kind": r["kind"], "rel": r.get(sol["holdout"]["hypothesis"], 0), "rel_neg": r.get("rel_neg", 0)}
            for r in sol["rows"]] + \
           [{"text": r["text"], "kind": "held-out", "rel": 0.95 if r["predict"] == "fire" else 0.15, "rel_neg": 0} for r in sol["holdout"]["rows"]]
    acts = ev([r["text"] for r in rows])
    ws.save("web_bank.json", {"units": units, "rows": rows, "acts": [[a[u] for u in units] for a in acts]})
    A = np.array([[a[u] for u in units] for a in acts])
    print("mean |act| per unit on nerve's bank (top 12):",
          sorted(zip(units, np.round(np.abs(A).mean(0), 3)), key=lambda t: -t[1])[:12])


if __name__ == "__main__":
    L, n = map(int, sys.argv[2].split("/"))
    ws = Workspace(L, n, sys.argv[3] if len(sys.argv) > 3 else "+")
    {"sources": sources, "bank": bank}[sys.argv[1]](ws)
