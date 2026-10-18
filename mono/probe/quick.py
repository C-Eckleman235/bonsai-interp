"""Quick check of a neighbour neuron: three cheap ways to break its label.

  tokens     weights-only: which vocabulary tokens push the pole directly through the
             embedding path (most meaningful for early layers)
  one back   on the neuron's own top corpus examples, which earlier neurons feed it
             (attribution, then zeroing each candidate), with each feeder's peak tokens
  falsify    the pre-registered 8 fire / 8 quiet sentences from quick_tests.json

  python -m mono.probe.quick 9/3688 +    (reads probes/<id>/quick_tests.json, writes quick.json)
"""
import re
import sys

import numpy as np
import torch

from mono import explore
from mono.inverse.linear import Weights, token_weights
from mono.inverse.neuron import NeuronSite
from mono.model import load
from mono.probe import textinv
from mono.probe.neighbors import single_ablation
from mono.probe.workspace import Workspace

PEAK = re.compile(r"⟦(.*?)⟧")


def peaks(L, n, pole, k=5):
    return [PEAK.search(t).group(1).replace("⏎", "\\n") for *_, t in explore.examples("bonsai1", L, n, pole, k, 0, 0)]


def one_back(model, rec, L, n, pole, k_ex=6, k_cand=24, k_show=4):
    if L == 0:
        return []
    s, toks = explore.scan("bonsai1"), explore.corpus()
    sign = 1.0 if pole == 0 else -1.0
    sites, attrs = [], []
    for j in range(k_ex):
        w, p = s["top_win"][L, pole, j, n], int(s["top_pos"][L, pole, j, n])
        site = NeuronSite(model, rec, toks[w].astype(np.int64), L, n, p)
        attr, y = site.attribution()
        attrs.append(np.concatenate([attr[l] for l in range(L)]) * sign / max(abs(y), 1e-6))
        sites.append(site)
    N = model.config.intermediate_size
    cand = [(int(i // N), int(i % N)) for i in np.argsort(-np.mean(attrs, 0))[:k_cand]]
    drops = np.mean([1 - np.array(single_ablation(site, cand)) for site in sites], 0)
    out = []
    for i in np.argsort(-drops)[:k_show]:
        l, m = cand[i]
        out.append({"unit": f"L{l}/N{m}", "drop": float(drops[i]), "peaks_pos": peaks(l, m, 0), "peaks_neg": peaks(l, m, 1)})
    return out


def run(ws: Workspace, tests=("quick_tests.json",), out="quick.json", with_feeders=True, need_fire=None):
    spec = {"neighbours": [nb for t in tests for nb in ws.require(t, "write and commit the tests first")["neighbours"]]}
    torch.cuda.set_per_process_memory_fraction(0.8)
    model, tok, rec = load(ws.model)
    tok.padding_side = "right"
    W = Weights(ws.model)
    scan = explore.scan(ws.model)
    results = []
    for nb in spec["neighbours"]:
        L, n = map(int, nb["unit"][1:].split("/N"))
        pole = 0 if nb["pole"] == "+" else 1
        # 1) direct token drivers (gate open and up positive for the + pole)
        T = token_weights(W, L, n)
        score = np.minimum(T[0], T[1]) if pole == 0 else np.minimum(T[0], -T[1])
        direct = [tok.decode([int(i)]) for i in np.argsort(-score)[:15]]
        # 2) one layer back
        feeders = one_back(model, rec, L, n, pole) if with_feeders else []
        need = need_fire if need_fire is not None else nb.get("need_fire", 5)
        # 3) falsification
        pmax, omax = float(scan["top_val"][L, pole, 0, n]), float(scan["top_val"][L, 1 - pole, 0, n])
        ev = textinv.TextEvaluator(model, tok, rec, L, n, pmax, omax, 1.0 if pole == 0 else -1.0)
        fire, quiet = (ev(nb["fire"]) if nb["fire"] else []), (ev(nb["quiet"]) if nb["quiet"] else [])
        n_fire = sum(r["act"] >= 0.3 for r in fire)
        n_fp = sum(r["act"] >= 0.3 for r in quiet)
        res = {**{k: nb.get(k) for k in ("unit", "pole", "role", "label", "confidence")}, "need_fire": need,
               "direct_tokens": direct, "feeders": feeders,
               "fire": [{"text": t, **r} for t, r in zip(nb["fire"], fire)],
               "quiet": [{"text": t, **r} for t, r in zip(nb["quiet"], quiet)],
               "n_fire": n_fire, "n_false_pos": n_fp, "n_written": [len(nb["fire"]), len(nb["quiet"])],
               "survives": bool(n_fire >= need and n_fp <= 1)}
        results.append(res)
        print(f"\n===== {nb['unit']}{nb['pole']} ({nb.get('role', '')}): {nb['label']}")
        print(f"  VERDICT: {'SURVIVES' if res['survives'] else 'FAILS'}  fire {n_fire}/{len(nb['fire'])} (need {need})  false positives {n_fp}/{len(nb['quiet'])}")
        print(f"  direct tokens: {direct}")
        for f in feeders:
            print(f"  feeder {f['unit']:<11} drop {f['drop']:.2f}  + peaks {f['peaks_pos']}  - peaks {f['peaks_neg']}")
        if len(spec["neighbours"]) <= 6:
            for r in sorted(res["fire"] + res["quiet"], key=lambda r: -r["act"]):
                tag = "fire " if r in res["fire"] else "quiet"
                print(f"    {tag} {r['act']:.2f} peak={r['peak_token']!r:<12} {r['text'][:70]}")
        torch.cuda.empty_cache()
    ws.save(out, {"neighbours": results})


if __name__ == "__main__":
    L, n = map(int, sys.argv[1].split("/"))
    run(Workspace(L, n, sys.argv[2] if len(sys.argv) > 2 else "+"))
