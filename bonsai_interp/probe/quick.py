"""Quick check of a neighbour neuron: three cheap ways to break its label.

  tokens     weights-only: which vocabulary tokens push the pole directly through the
             embedding path (most meaningful for early layers)
  one back   on the neuron's own top corpus examples, which earlier neurons feed it
             (attribution, then zeroing each candidate), with each feeder's peak tokens
  falsify    the pre-registered 8 fire / 8 quiet sentences from quick_tests.json

  python -m bonsai_interp.probe.quick 9/3688 +    (reads probes/<id>/quick_tests.json, writes quick.json)
"""
import re
import sys

import numpy as np
import torch

from bonsai_interp import explore
from bonsai_interp.env import empty_cache, limit_memory
from bonsai_interp.inverse.linear import Weights, token_weights
from bonsai_interp.inverse.neuron import NeuronSite
from bonsai_interp.model import load
from bonsai_interp.probe import textinv
from bonsai_interp.probe.neighbors import single_ablation
from bonsai_interp.probe.workspace import Workspace

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


def one_forward(model, rec, L, n, pole, k_ex=6, layers_down=12, k_cand=24, k_show=4, window=3, min_level=0.1):
    """The readers of a pole, the downstream mirror of one_back: in the contexts where THIS pole fires (its top k_ex
    corpus examples), which later poles lose the most of their activation when it is zeroed.

    For every neuron in the next `layers_down` layers, at the peak and the `window - 1` tokens after it:
      screen  the signed change (clean - zeroed), normalised by the corpus max of the pole it moves toward; the k_cand
              largest mean changes are the candidates, each with the pole this one pushes it toward;
      drop    the share of that reader pole's activation lost when this neuron is zeroed, averaged over the examples
              where the reader is actually on (>= min_level of its corpus max). The same quantity as a `feeds` weight
              (share of the target's activation lost), measured in the source's contexts, which is where a sparse
              source matters (on a reader's own top examples a sparse source is usually silent).
    Only readers that are on in at least one example are kept; `level` is how strongly on (fraction of their max).
    """
    from bonsai_interp.env import DEVICE
    from bonsai_interp.model import StopForward
    s, toks = explore.scan("bonsai1"), explore.corpus()
    layers = list(range(L + 1, min(L + 1 + layers_down, model.config.num_hidden_layers)))
    if not layers:
        return []

    def forward(ids, ablate):
        rec.mode, rec.layers, rec.stop_at = "act", set(layers), layers[-1]
        rec.interventions = {L: (lambda a: _zero(a, n))} if ablate else {}
        try:
            with torch.no_grad():
                model.model(torch.tensor(ids[None], device=DEVICE))
        except StopForward:
            pass
        finally:
            rec.mode, rec.stop_at, rec.interventions = None, None, {}
        out = {l: rec.store.pop(l)[0] for l in layers}
        rec.clear()
        return out

    pos_max = np.stack([s["top_val"][l, 0, 0] for l in layers]).astype(np.float64)        # (layers, N)
    neg_max = np.abs(np.stack([s["top_val"][l, 1, 0] for l in layers]).astype(np.float64))
    clean_w, abl_w = [], []  # per example: (layers, window, N), signed raw activations
    for j in range(k_ex):
        if not np.isfinite(s["top_val"][L, pole, j, n]):
            continue
        w, p = s["top_win"][L, pole, j, n], int(s["top_pos"][L, pole, j, n])
        ids = toks[w][: p + window].astype(np.int64)
        c, a_ = forward(ids, False), forward(ids, True)
        clean_w.append(np.stack([c[l][p: p + window].float().cpu().numpy() for l in layers]))
        abl_w.append(np.stack([a_[l][p: p + window].float().cpu().numpy() for l in layers]))
    if not clean_w:
        return []
    C, A = np.stack(clean_w), np.stack(abl_w)                    # (ex, layers, window, N)
    D = C - A
    pick = np.abs(D).argmax(2)[:, :, None, :]                   # the window position of largest change, per example
    d = np.take_along_axis(D, pick, 2)[:, :, 0, :]              # (ex, layers, N)
    eff = np.mean(d / np.where(d >= 0, pos_max[None], neg_max[None]).clip(1e-6), 0)   # (layers, N)
    N = eff.shape[1]
    out = []
    for i in np.argsort(-np.abs(eff), axis=None)[:k_cand]:
        li, m = int(i // N), int(i % N)
        rp = 0 if eff[li, m] > 0 else 1
        sg, mx = (1.0, pos_max[li, m]) if rp == 0 else (-1.0, neg_max[li, m])
        levels, losses = [], []
        for e in range(C.shape[0]):
            q = int(np.argmax(C[e, li, :, m] * sg))             # where the reader pole is most on in this example
            clean, abl = C[e, li, q, m] * sg, A[e, li, q, m] * sg
            if clean >= min_level * mx:
                levels.append(clean / mx)
                losses.append(1 - abl / clean)
        if losses:
            l = layers[li]
            out.append({"unit": f"L{l}/N{m}{'+' if rp == 0 else '-'}", "L": l, "n": m, "pole": "+" if rp == 0 else "-",
                        "drop": float(np.mean(losses)), "level": float(np.mean(levels)), "n_on": len(losses),
                        "screen": float(eff[li, m]), "peaks_pos": peaks(l, m, 0), "peaks_neg": peaks(l, m, 1)})
    out.sort(key=lambda r: -r["drop"])
    return out[:k_show]


def _zero(a, n):
    a = a.clone()
    a[..., n] = 0
    return a


def run(ws: Workspace, tests=("quick_tests.json",), out="quick.json", with_feeders=True, need_fire=None):
    spec = {"neighbours": [nb for t in tests for nb in ws.require(t, "write and commit the tests first")["neighbours"]]}
    limit_memory(0.8)
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
        empty_cache()
    ws.save(out, {"neighbours": results})


if __name__ == "__main__":
    L, n = map(int, sys.argv[1].split("/"))
    run(Workspace(L, n, sys.argv[2] if len(sys.argv) > 2 else "+"))
