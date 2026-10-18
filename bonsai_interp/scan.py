"""Corpus scan: per-pole top-K windows and sign-regime histograms for every neuron.

For each MLP neuron (layer, index) and each pole (+/-) we keep the K windows with
the most extreme activation (one example per window, at that window's peak
position), plus a log-binned histogram of |activation| split by the four
(sign gate, sign up) regimes:
  regime 0: g>0, u>0  -> positive pole, gate open   (+x+)
  regime 1: g<0, u<0  -> positive pole, gate closed (-x-, the essay's "echo")
  regime 2: g>0, u<0  -> negative pole, gate open
  regime 3: g<0, u>0  -> negative pole, gate closed ("sign-flipped leak")

Usage: python -m bonsai_interp.scan --model bonsai1 --windows 20000
"""
import argparse
import time

import numpy as np
import torch

from bonsai_interp.device import DEVICE
from bonsai_interp.model import ROOT, load

K = 32
BINS = torch.logspace(-3, 4, 57)  # |a| from 1e-3 to 1e4, 8 bins/decade
NB = len(BINS) + 1


class LayerStats:
    def __init__(self, n: int, device=DEVICE):
        self.top_val = torch.full((2, K, n), -float("inf"), device=device)
        self.top_win = torch.full((2, K, n), -1, dtype=torch.int32, device=device)
        self.top_pos = torch.zeros((2, K, n), dtype=torch.int16, device=device)
        self.top_reg = torch.zeros((2, K, n), dtype=torch.uint8, device=device)
        self.hist = torch.zeros((n, 4, NB), dtype=torch.int64, device=device)
        self.bins = BINS.to(device)
        self.n = n

    def update(self, g, u, a, win0: int):
        B, T, n = a.shape
        a = a.float()
        regime = ((g < 0).to(torch.uint8) * 1 + (u < 0).to(torch.uint8) * 2)
        # map (g<0, u<0) bits to the regime numbering in the module docstring
        regime = torch.tensor([0, 3, 2, 1], device=a.device, dtype=torch.uint8)[regime.long()]

        b = torch.bucketize(a.abs(), self.bins)  # [B,T,n]
        flat = (torch.arange(n, device=a.device) * 4 * NB + regime.long() * NB + b).reshape(-1)
        self.hist += torch.bincount(flat, minlength=n * 4 * NB).reshape(n, 4, NB)

        for pole, sgn in ((0, 1.0), (1, -1.0)):
            s = a * sgn
            v, p = s.max(1)  # [B, n] peak per window
            r = regime.gather(1, p[:, None, :]).squeeze(1)
            wins = torch.arange(win0, win0 + B, device=a.device, dtype=torch.int32)[:, None].expand(B, n)
            allv = torch.cat([self.top_val[pole], v])
            top, idx = allv.topk(K, dim=0)
            self.top_val[pole] = top
            self.top_win[pole] = torch.cat([self.top_win[pole], wins]).gather(0, idx)
            self.top_pos[pole] = torch.cat([self.top_pos[pole], p.to(torch.int16)]).gather(0, idx)
            self.top_reg[pole] = torch.cat([self.top_reg[pole], r]).gather(0, idx)

    def to_numpy(self):
        return {k: getattr(self, k).cpu().numpy() for k in ("top_val", "top_win", "top_pos", "top_reg", "hist")}


@torch.no_grad()
def scan(name: str, n_windows: int, batch: int, corpus_path=ROOT / "data" / "corpus.npz", tag=""):
    tokens = np.load(corpus_path)["tokens"][:n_windows]
    model, tok, rec = load(name)
    n_layers = len(model.model.layers)
    stats = [LayerStats(model.config.intermediate_size) for _ in range(n_layers)]

    # update stats inside the hook so a whole model's activations are never held at once
    cur = {"win0": 0}

    class Sink(dict):
        def __setitem__(self, layer, gua):
            stats[layer].update(*gua, cur["win0"])

    rec.mode, rec.store = "full", Sink()
    t0 = time.time()
    for i in range(0, len(tokens), batch):
        cur["win0"] = i
        model.model(torch.from_numpy(tokens[i:i + batch].astype(np.int64)).to(DEVICE))  # skip lm_head
        if (i // batch) % 50 == 0:
            done = i + batch
            el = time.time() - t0
            print(f"{done}/{len(tokens)} windows, {done * tokens.shape[1] / el:,.0f} tok/s, "
                  f"eta {el / done * (len(tokens) - done) / 60:.1f} min", flush=True)

    out = ROOT / "data" / f"scan_{name}{tag}.npz"
    arrs = [s.to_numpy() for s in stats]
    np.savez(out, **{k: np.stack([a[k] for a in arrs]) for k in arrs[0]}, bins=BINS.numpy(),
             n_windows=len(tokens))
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bonsai1")
    ap.add_argument("--windows", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--corpus", default=str(ROOT / "data" / "corpus.npz"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    scan(a.model, a.windows, a.batch, a.corpus, a.tag)
