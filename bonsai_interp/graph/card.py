"""Neuron cards: a compact, label-free view of any of the 442,368 neurons, from the corpus scan and the weights.

  python -m bonsai_interp.graph.play card --neuron bonsai1/L9/N3688+ [--k 6] [--out card.json|card.md]

A card holds, for both poles: the top corpus examples with the peak token in ⟦ ⟧, the pole's corpus max,
its activation density (share of all scanned tokens above 0.1 and 0.3 of that max, read from the scan's
log-binned histograms), and the sign-regime split. It adds the direct logit effect of the neuron's output
direction (final norm gain, then the unembedding; weights only, read on the CPU) and flags for units an
8 + 8 test can't pass (dense, positional). No GPU and no model load: a card takes a few seconds.
"""
import json
from functools import lru_cache

import numpy as np

from bonsai_interp import explore
from bonsai_interp.graph.contrib import NEURON, ROOT

# flags (calibrated on the poles contributors marked dense / positional, against held token poles)
DENSE_10 = 0.10     # >= 10% of tokens above 0.1 of max: an 8 + 8 test can hardly pass
BROAD_10 = 0.01     # >= 1%: near misses will often fire weakly
POSITIONAL = 0.5    # >= half of the top 32 peaks within the first 16 tokens of a window, or one position


def parse(neuron):
    """'bonsai1/L9/N3688+' or 'L9/N3688' or '9/3688' -> (model, L, n, pole or None)."""
    m = NEURON.match(neuron)
    if m:
        return m["model"], int(m["L"]), int(m["n"]), m["pole"]
    s = neuron.removeprefix("bonsai1/").replace("L", "").replace("N", "")
    pole = s[-1] if s[-1] in "+-" else None
    L, n = map(int, s.rstrip("+-").split("/"))
    return "bonsai1", L, n, pole


def _above(h, lower, upper, t):
    """Count in a log-binned histogram above threshold t, splitting the straddling bin log-uniformly."""
    b = int(np.searchsorted(lower, t, side="right")) - 1  # lower[b] <= t < lower[b+1]
    n = h[b + 1:].sum()
    lo, hi = lower[b], upper[b]
    if lo > 0 and np.isfinite(hi) and hi > t:
        n += h[b] * (np.log(hi) - np.log(t)) / (np.log(hi) - np.log(lo))
    elif lo == 0 and hi > t:
        n += h[b] * (hi - t) / hi
    return float(n)


def pole_stats(name, L, n, pole):
    s = explore.scan(name)
    pi, regs = (0, [0, 1]) if pole == "+" else (1, [2, 3])
    mx = float(s["top_val"][L, pi, 0, n])
    h = s["hist"][L, n]  # (4 regimes, NB); bin b holds |a| in (bins[b-1], bins[b]]
    bins = s["bins"].astype(np.float64)
    lower, upper = np.concatenate([[0], bins]), np.concatenate([bins, [np.inf]])
    total = float(h.sum())
    pole_h = h[regs].sum(0)
    dens = {f: _above(pole_h, lower, upper, f * mx) / total for f in (0.1, 0.3)}
    split = {}
    for r in regs:  # regime shares among this pole's activations above 0.1 of max
        split[explore.REGIMES[r]] = _above(h[r], lower, upper, 0.1 * mx)
    tot = sum(split.values())
    split = {k: round(v / tot, 3) if tot else None for k, v in split.items()}
    pos = s["top_pos"][L, pi, :, n]
    pos = pos[np.isfinite(s["top_val"][L, pi, :, n])]
    vals, counts = np.unique(pos, return_counts=True)
    early, mode_share = float((pos < 16).mean()) if len(pos) else 0.0, float(counts.max() / len(pos)) if len(pos) else 0.0
    return {"max": round(mx, 3), "density": {"0.1": round(dens[0.1], 5), "0.3": round(dens[0.3], 5)},
            "regimes": split, "peak_positions": {"first16": round(early, 3), "commonest": int(vals[counts.argmax()]) if len(pos) else None,
                                                  "commonest_share": round(mode_share, 3)}}


def flags(st):
    out = []
    d = st["density"]["0.1"]
    if d >= DENSE_10:
        out.append(f"dense: {d:.1%} of tokens reach 0.1 of max; an 8 + 8 test can hardly pass (consider unit_kind 'dense / positional')")
    elif d >= BROAD_10:
        out.append(f"broad: {d:.1%} of tokens reach 0.1 of max; near misses will often fire weakly")
    pp = st["peak_positions"]
    if pp["first16"] >= POSITIONAL or pp["commonest_share"] >= POSITIONAL:
        out.append(f"positional: top peaks cluster at window position {pp['commonest']} "
                   f"({pp['commonest_share']:.0%}; {pp['first16']:.0%} within the first 16 tokens)")
    if st["regimes"]:
        echo = max(st["regimes"], key=lambda k: st["regimes"][k] or 0)
        if echo in ("-x- echo", "-g +u leak") and (st["regimes"][echo] or 0) >= 0.6:
            out.append(f"gate-closed regime dominates ({echo} {st['regimes'][echo]:.0%})")
    return out


@lru_cache(None)
def _weights(name):
    """(down_proj column reader, final norm gain in fp32, unembedding in fp16) on the CPU, from the
    unpacked safetensors or from the 1-bit GGUF (bonsai_interp.gguf_weights), whichever the folder holds."""
    from bonsai_interp.model import MODELS
    path = MODELS[name]
    if not (path / "model.safetensors.index.json").exists():
        from bonsai_interp import gguf_weights
        ts = gguf_weights.by_name(path)
        col = lambda L, n: gguf_weights.column(ts[f"model.layers.{L}.mlp.down_proj.weight"], n)
        return col, gguf_weights.dense(ts["model.norm.weight"]), gguf_weights.dense(ts["lm_head.weight"]).half()
    from safetensors import safe_open
    idx = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    def get(t):
        with safe_open(str(path / idx[t]), "pt", device="cpu") as h:
            return h.get_tensor(t)
    def col(L, n):
        t = f"model.layers.{L}.mlp.down_proj.weight"
        with safe_open(str(path / idx[t]), "pt", device="cpu") as h:
            return h.get_slice(t)[:, n:n + 1][:, 0].float()
    return col, get("model.norm.weight").float(), get("lm_head.weight")


def direct_logits(name, L, n, k=12):
    """Tokens the + pole's output direction promotes / suppresses directly (the - pole does the reverse):
    W_U (g * d), d = down_proj column n, g = final RMSNorm gain. Same as probe/induce.py `direct`."""
    col, g, wu = _weights(name)
    d = col(L, n)
    logits = _matvec(wu, g * d)
    tok = explore.tokenizer()
    top, bot = logits.topk(k), (-logits).topk(k)
    return {"promotes": [[tok.decode([i]), round(v, 3)] for i, v in zip(top.indices.tolist(), top.values.tolist())],
            "suppresses": [[tok.decode([i]), round(-v, 3)] for i, v in zip(bot.indices.tolist(), bot.values.tolist())],
            "note": "for the + pole, per unit activation, before the final RMS normalisation; the - pole swaps the lists"}


def _matvec(w, v, chunk=16384):
    """fp16 matrix x fp32 vector in chunks (keeps memory at a few hundred MB)."""
    import torch
    return torch.cat([w[i:i + chunk].float() @ v for i in range(0, w.shape[0], chunk)])


def card(neuron, k=6, logits=True, before=40, after=8):
    name, L, n, pole = parse(neuron)
    out = {"neuron": f"{name}/L{L}/N{n}", "focus": pole, "poles": {}}
    for p in ("+", "-"):
        st = pole_stats(name, L, n, p)
        ex = explore.examples(name, L, n, 0 if p == "+" else 1, k, before=before, after=after)
        st["examples"] = [{"act": round(v, 3), "frac": round(v / st["max"] * (1 if p == "+" else -1), 3) if st["max"] else None,
                           "regime": r, "text": t} for v, r, w, pos, t in ex]
        st["flags"] = flags(st)
        out["poles"][p] = st
    if logits:
        try:
            out["direct"] = direct_logits(name, L, n)
        except (FileNotFoundError, KeyError, OSError) as e:
            out["direct"] = {"error": f"weights unavailable: {e.__class__.__name__}"}
    rec = record(name, L, n)
    out["records"] = {p: {"status": r["status"], "label": r.get("label"), "short": r.get("short")} for p, r in rec.items()}
    return out


def record(name, L, n):
    """Derived graph records for the two poles, if any ({pole: record})."""
    out = {}
    for p, suf in (("+", "pos"), ("-", "neg")):
        f = ROOT / "graph" / "neurons" / name / f"L{L}_N{n}_{suf}.json"
        if f.exists():
            out[p] = json.loads(f.read_text(encoding="utf-8"))
    return out


def render(c):
    lines = [f"=== {c['neuron']}  (card: scan + weights; not a tested label)"
             + (f"   >>> you asked about the {c['focus']} pole: it is shown first <<<" if c.get("focus") else "")]
    for p in sorted(c["poles"], key=lambda p: p != (c["focus"] or "+")):
        st = c["poles"][p]
        rec = c["records"].get(p)
        lines.append(f"--- {p} pole{' (ASKED)' if p == c.get('focus') else ''}   max {st['max']:.2f}   density >0.1·max {st['density']['0.1']:.3%}  >0.3·max {st['density']['0.3']:.3%}"
                     f"   regimes {', '.join(f'{k} {v:.0%}' for k, v in st['regimes'].items() if v is not None)}")
        lines.append(f"    record: {rec['status']}: {rec['label']}" if rec else "    record: none (unsurveyed)")
        for f in st["flags"]:
            lines.append(f"    FLAG {f}")
        for e in st["examples"]:
            lines.append(f"    {e['act']:8.2f} {e['regime']:>10} | {e['text'][-110:]}")
    d = c.get("direct")
    if d and "error" not in d:
        lines.append("--- direct logit effect of the output direction (+ pole; the - pole reverses it)")
        lines.append("    promotes:   " + "  ".join(repr(t) for t, _ in d["promotes"]))
        lines.append("    suppresses: " + "  ".join(repr(t) for t, _ in d["suppresses"]))
    elif d:
        lines.append(f"--- direct logit effect: {d['error']}")
    return "\n".join(lines)


def main(a):
    if len(a.neuron) > 1:  # several cards in one process: the scan and the weights load once
        from pathlib import Path
        texts = [render(card(nid, k=a.k, logits=not a.no_logits)) for nid in a.neuron]
        if a.out:  # many cards are long: agents whose tools truncate output read the file in pieces instead
            Path(a.out).write_text("\n\n".join(texts) + "\n", encoding="utf-8", newline="\n")
            print(f"wrote {len(texts)} cards to {a.out} (one '=== ' header per neuron)")
        else:
            print("\n\n".join(texts))
        return
    c = card(a.neuron[0], k=a.k, logits=not a.no_logits)
    print(render(c))
    if a.out:
        from pathlib import Path
        p = Path(a.out)
        p.write_text(json.dumps(c, ensure_ascii=False, indent=1) + "\n" if p.suffix == ".json" else render(c) + "\n",
                     encoding="utf-8", newline="\n")
        print(f"wrote {p}")


def compact(c, k=4, pre=70, post=24, tokens=8):
    """A small form of a card for the survey site: about 1.5 KB per neuron.
    {"+"/"-": {"m": max, "d": [density 0.1, 0.3], "r": [gate-open share, gate-closed share], "f": [flag words],
               "x": [[act, before, peak token, after], ...]}, "u": [[promoted by +], [suppressed by +]]}"""
    out = {}
    for p, st in c["poles"].items():
        xs = []
        for e in st["examples"][:k]:
            a, rest = e["text"].split("⟦", 1)
            b, after = rest.split("⟧", 1)
            xs.append([round(e["act"], 2), a[-pre:], b, after[:post]])
        out[p] = {"m": round(st["max"], 2), "d": [st["density"]["0.1"], st["density"]["0.3"]],
                  "r": [v if v is not None else 0 for v in st["regimes"].values()],
                  "f": [f.split(":")[0].split(" ")[0] for f in st["flags"]], "x": xs}
    d = c.get("direct") or {}
    if "promotes" in d:
        out["u"] = [[t for t, _ in d["promotes"][:tokens]], [t for t, _ in d["suppresses"][:tokens]]]
    return out
