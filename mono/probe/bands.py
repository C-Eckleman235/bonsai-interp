"""Full-distribution view of a few neurons: every token's activation over a corpus slice, so the
weak and middle activations (not only the top 32) can be sampled, annotated and tested.

  scan     record the given neurons on every token of N windows -> data/bands_<tag>.npz
  sample   for a neuron and pole, draw contexts from activation bands (fractions of corpus max)

  python -m mono.probe.bands scan --units 9/3688,10/6901,... --windows 6000 --tag web
"""
import argparse
import re

import numpy as np
import torch

from mono import explore
from mono.model import ROOT, StopForward, load

BANDS = [(0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 10.0)]


@torch.no_grad()
def scan(units, windows, tag, batch=8):
    toks = explore.corpus()[:windows]
    model, tok, rec = load("bonsai1")
    layers = sorted({L for L, _ in units})
    out = {u: np.zeros((len(toks), toks.shape[1]), np.float16) for u in units}
    rec.mode, rec.layers, rec.stop_at = "act", set(layers), layers[-1]
    for s in range(0, len(toks), batch):
        x = torch.from_numpy(toks[s:s + batch].astype(np.int64)).cuda()
        try:
            model.model(x)
        except StopForward:
            pass
        for L, n in units:
            out[(L, n)][s:s + batch] = rec.store[L][:, :, n].float().cpu().numpy().astype(np.float16)
        rec.clear()
        if (s // batch) % 100 == 0:
            print(f"{s + batch}/{len(toks)}", flush=True)
    rec.mode, rec.stop_at = None, None
    np.savez(ROOT / "data" / f"bands_{tag}.npz", windows=windows,
             **{f"L{L}_N{n}": a for (L, n), a in out.items()})
    print("saved", ROOT / "data" / f"bands_{tag}.npz")


def load_bands(tag):
    return np.load(ROOT / "data" / f"bands_{tag}.npz")


def sample(tag, L, n, pole, k=30, seed=0, before=60, after=12, bands=BANDS):
    """{band: {"count": tokens in band, "examples": [(frac, text with ⟦peak⟧)]}} for one pole.
    At most one example per window, so a single document cannot dominate a band."""
    d = load_bands(tag)
    a = d[f"L{L}_N{n}"].astype(np.float32) * (1 if pole == "+" else -1)
    pmax = float(explore.scan("bonsai1")["top_val"][L, 0 if pole == "+" else 1, 0, n])
    f = a / pmax
    toks, tk = explore.corpus(), explore.tokenizer()
    rng = np.random.default_rng(seed)
    res = {}
    for lo, hi in bands:
        w, p = np.where((f >= lo) & (f < hi))
        key = f"{lo:.2f}-{hi:.2f}" if hi < 10 else f"≥{lo:.2f}"
        ex = []
        if len(w):
            order = rng.permutation(len(w))
            seen = set()
            for i in order:
                if w[i] in seen:
                    continue
                seen.add(w[i])
                ids = toks[w[i]]
                q = int(p[i])
                text = tk.decode(ids[max(0, q - before):q]) + "⟦" + tk.decode(ids[q:q + 1]) + "⟧" + tk.decode(ids[q + 1:q + 1 + after])
                ex.append({"frac": round(float(f[w[i], q]), 3), "text": text.replace("\n", "⏎")})
                if len(ex) >= k:
                    break
        res[key] = {"count": int(len(w)), "examples": ex}
    res["_total_tokens"] = int(a.size)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--units", default="9/3688")
    ap.add_argument("--windows", type=int, default=6000)
    ap.add_argument("--tag", default="web")
    a = ap.parse_args()
    units = [tuple(map(int, u.split("/"))) for u in a.units.split(",")]
    if a.stage == "scan":
        scan(units, a.windows, a.tag)
