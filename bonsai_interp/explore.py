"""Forward view of a neuron from a corpus scan: top examples per pole, regime stats.

Usage: python -m bonsai_interp.explore --model bonsai1 --neuron 18/4242 [--pole +] [--k 12]
"""
import argparse
from functools import lru_cache

import numpy as np
from transformers import AutoTokenizer

from bonsai_interp.model import MODELS, ROOT

REGIMES = ["+x+", "-x- echo", "+g -u", "-g +u leak"]


@lru_cache(None)
def scan(name):
    return dict(np.load(ROOT / "data" / f"scan_{name}.npz"))


@lru_cache(None)
def corpus():
    return np.load(ROOT / "data" / "corpus.npz")["tokens"]


@lru_cache(None)
def tokenizer():
    return AutoTokenizer.from_pretrained(MODELS["bonsai1"])


def examples(name, L, n, pole=0, k=12, before=48, after=8):
    """[(activation, regime, window, pos, text with the peak token in ⟦ ⟧)]"""
    s, tok, toks = scan(name), tokenizer(), corpus()
    out = []
    for j in range(k):
        v = s["top_val"][L, pole, j, n]
        if not np.isfinite(v):
            break
        w, p, r = s["top_win"][L, pole, j, n], int(s["top_pos"][L, pole, j, n]), s["top_reg"][L, pole, j, n]
        ids = toks[w]
        text = tok.decode(ids[max(0, p - before):p]) + "⟦" + tok.decode(ids[p:p + 1]) + "⟧" + \
            tok.decode(ids[p + 1:p + 1 + after])
        out.append((float(v) * (1 if pole == 0 else -1), REGIMES[r], int(w), p, text.replace("\n", "⏎")))
    return out


def regime_stats(name, L, n):
    """Share of each regime among activations above 10% / 30% of each pole's max."""
    s = scan(name)
    h, bins = s["hist"][L, n], s["bins"]  # (4, NB): bin b holds |a| in [bins[b-1], bins[b])
    lower = np.concatenate([[0], bins])
    res = {}
    for pole, regs in (("+", [0, 1]), ("-", [2, 3])):
        mx = abs(s["top_val"][L, 0 if pole == "+" else 1, 0, n])
        for frac in (0.1, 0.3):
            mask = lower >= frac * mx
            counts = h[regs][:, mask].sum(1)
            tot = counts.sum()
            res[f"{pole}pole>{int(frac * 100)}%max"] = {
                "count": int(tot),
                **{REGIMES[r]: round(float(c) / tot, 3) if tot else None for r, c in zip(regs, counts)}}
        res[f"{pole}max"] = float(mx)
    return res


def show(name, L, n, poles=("+", "-"), k=12):
    print(f"=== {name} L{L}/N{n}")
    for key, v in regime_stats(name, L, n).items():
        print(f"  {key}: {v}")
    for pole in poles:
        print(f"--- {pole} pole")
        for v, r, w, p, text in examples(name, L, n, 0 if pole == "+" else 1, k):
            print(f"{v:9.2f} {r:>10} | {text}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bonsai1")
    ap.add_argument("--neuron", required=True)
    ap.add_argument("--pole", default="+-")
    ap.add_argument("--k", type=int, default=12)
    a = ap.parse_args()
    L, n = map(int, a.neuron.split("/"))
    show(a.model, L, n, tuple(a.pole), a.k)
