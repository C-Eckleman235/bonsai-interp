"""Does quantization-aware training preserve neuron identity?

Runs the same token windows through Qwen3-8B, Ternary-Bonsai-8B and Bonsai-8B,
then for each sampled layer compares neuron i in one model with neuron i in
another (same-index Pearson r) and with its best match among all neurons of
the same layer. If same-index r is high and equals the best match, neuron
indices can be compared directly across the precision ladder.

Usage: python -m bonsai_interp.identity --windows 16
"""
import argparse
import json

import numpy as np
import torch

from bonsai_interp.device import DEVICE, empty_cache
from bonsai_interp.model import ROOT, load

LAYERS = [1, 5, 9, 13, 17, 21, 25, 29, 33, 35]
CACHE = ROOT / "data" / "identity"


@torch.no_grad()
def record(name, tokens):
    model, tok, rec = load(name)
    rec.mode, rec.layers = "act", set(LAYERS)
    acts = {l: [] for l in LAYERS}
    for i in range(0, len(tokens), 4):
        ids = torch.from_numpy(tokens[i:i + 4].astype(np.int64)).to(DEVICE)
        model(ids)
        for l in LAYERS:
            acts[l].append(rec.store[l].reshape(-1, rec.store[l].shape[-1]).to(torch.float16).cpu())
        rec.clear()
    del model
    empty_cache()
    return {l: torch.cat(v) for l, v in acts.items()}


def zscore(x):
    x = x.to(DEVICE).float()
    x = x - x.mean(0)
    return x / x.norm(dim=0).clamp_min(1e-6)


def compare(a, b):
    za, zb = zscore(a), zscore(b)
    same = (za * zb).sum(0)
    best = torch.empty_like(same)
    best_idx = torch.empty_like(same, dtype=torch.long)
    for s in range(0, za.shape[1], 2048):  # chunk the 12288 x 12288 correlation matrix
        c = za[:, s:s + 2048].T @ zb
        best[s:s + 2048], best_idx[s:s + 2048] = c.max(1)
    idx = torch.arange(len(same), device=same.device)
    return {
        "same_r_median": same.median().item(),
        "same_r_p10": same.quantile(0.1).item(),
        "frac_same_r_gt_0.5": (same > 0.5).float().mean().item(),
        "frac_best_match_is_same_index": (best_idx == idx).float().mean().item(),
        "best_r_median": best.median().item(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=16)
    a = ap.parse_args()
    corpus = np.load(ROOT / "data" / "corpus.npz")
    tokens = corpus["tokens"][-a.windows:]  # tail windows; the scan uses the head
    acts = {m: record(m, tokens) for m in ["qwen", "ternary", "bonsai1"]}
    out = {}
    for x, y in [("qwen", "ternary"), ("qwen", "bonsai1"), ("ternary", "bonsai1")]:
        for l in LAYERS:
            out[f"{x}~{y}/L{l}"] = r = compare(acts[x][l], acts[y][l])
            print(f"{x:>7}~{y:<8} L{l:<2} same-index r median {r['same_r_median']:.3f} "
                  f"(p10 {r['same_r_p10']:.3f}), best match = same index {r['frac_best_match_is_same_index']:.1%}",
                  flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "identity.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
