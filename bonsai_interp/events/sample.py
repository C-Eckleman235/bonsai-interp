"""A uniform sample of every pole's strong activations over the parsed and lexed windows (GPU, one pass, all 36 layers).

  python -m bonsai_interp.events.sample run [--limit W] [--out FILE]   # -> data/events/bonsai1.events.npz
  python -m bonsai_interp.events.sample stats                          # CPU: counts per layer

The windows are those the grammar pass parsed (12,000 English windows of the selection split, grammar/parse.py) and those
sweeplib@5 lexed (every code window of the selection split, codesyn/lex.py), so any condition defined on the parse or the lex
can be evaluated at every sampled activation. Activations are read at full window context (1,024 tokens) from position
RULE["min_pos"] on (window starts are an artifact, FINDINGS 2026-10-18).

A strong activation is >= RULE["strong"] (0.3) of the pole's corpus maximum (the scan's), in the pole's sign. Per pole the
pass keeps the K = 64 strong activations with the smallest hash of their (window, position): a bottom-k sample, uniform over
the pole's strong activations and shared between poles (two poles that fire at one position keep it or drop it together,
when both have enough). It also counts every pole's strong activations by domain. Weak poles (corpus max < 0.5) are skipped.

Per pole (layer, sign, neuron): ids (K) int32 = window * 1024 + position (-1 empty), vals (K) float16 = activation / pole max,
count (int64, all strong activations), count_dom (English, code).
"""
import argparse
import time
from pathlib import Path

import numpy as np

from bonsai_interp.auto import context_tier as ct

ROOT = ct.ROOT
OUT = ROOT / "data" / "events"
K = 64
BUF = 1_000_000  # buffered candidates per layer before a compaction
N, LAYERS, T = 12288, 36, 1024


def windows():
    from bonsai_interp.grammar import parse as PA
    en = np.load(ROOT / "data" / "grammar" / "bonsai1.parse.select.npz")["windows"]
    code = np.load(ROOT / "data" / "codesyn" / "bonsai1.lex.select.npz")["windows"]
    return np.unique(np.concatenate([en, code])).astype(np.int64), set(map(int, en)), PA


def _hash(w):
    """Per position of window w: a deterministic uniform in [0, 1) (the bottom-k priority)."""
    return np.random.default_rng([97, int(w)]).random(T).astype(np.float32)


class Sampler:
    def __init__(self, device):
        import torch
        self.t = torch
        mp = ct.RULE["min_pos"]
        self.mp = mp
        pm = np.stack([np.stack(ct._pole_max(L)) for L in range(LAYERS)]).astype(np.float32)  # (L, 2, N)
        self.pm = torch.tensor(pm, device=device)
        self.ok = torch.tensor((pm >= ct.RULE["min_max"]).reshape(LAYERS, 2 * N), device=device)  # weak poles are skipped
        self.h = torch.full((LAYERS, K, 2 * N), np.inf, device=device)
        self.ids = torch.full((LAYERS, K, 2 * N), -1, dtype=torch.int32, device=device)
        self.vals = torch.zeros((LAYERS, K, 2 * N), dtype=torch.float16, device=device)
        self.count = torch.zeros((LAYERS, 2, 2 * N), dtype=torch.int64, device=device)  # [L, domain (en, code), pole]
        self.tau = torch.full((LAYERS, 2 * N), np.inf, device=device)  # per pole: the K-th smallest kept priority
        self.buf, self.nbuf = [[] for _ in range(LAYERS)], [0] * LAYERS
        self.device = device
        self.batch = None  # (hash (B*T'), ids (B*T'), domain (B,)) for the batch being read

    def set_batch(self, ws, en):
        t = self.t
        mp = self.mp
        h = np.stack([_hash(w)[mp:] for w in ws]).ravel()
        ids = (np.asarray(ws)[:, None] * T + np.arange(mp, T)[None, :]).ravel().astype(np.int32)
        dom = np.array([0 if int(w) in en else 1 for w in ws])
        self.batch = (t.tensor(h, device=self.device), t.tensor(ids, device=self.device),
                      t.tensor(np.repeat(dom, T - mp), device=self.device))

    def hook(self, L):
        t = self.t

        def fn(a):
            h, ids, dom = self.batch
            v = a[:, self.mp:].reshape(-1, N).float()  # (B*T', N)
            x = t.cat([v, -v], 1) / t.cat([self.pm[L, 0], self.pm[L, 1]])[None, :]  # (B*T', 2N), as a share of the pole max
            S = (x >= ct.RULE["strong"]) & self.ok[L][None, :]
            for d in (0, 1):
                self.count[L, d] += S[dom == d].sum(0)
            r, col = (S & (h[:, None] < self.tau[L][None, :])).nonzero(as_tuple=True)
            buf = self.buf[L]
            buf.append((col.int(), h[r], ids[r], x[r, col].half()))
            self.nbuf[L] += len(r)
            if self.nbuf[L] > BUF:
                self.compact(L)
            return a
        return fn

    def compact(self, L):
        """Merge layer L's buffered candidates into its kept bottom-K per pole and lower the pole thresholds."""
        t = self.t
        if not self.buf[L]:
            return
        kh = self.h[L]
        kr, kc = t.isfinite(kh).nonzero(as_tuple=True)
        col = t.cat([kc.int()] + [b[0] for b in self.buf[L]])
        hh = t.cat([kh[kr, kc]] + [b[1] for b in self.buf[L]])
        ii = t.cat([self.ids[L][kr, kc]] + [b[2] for b in self.buf[L]])
        vv = t.cat([self.vals[L][kr, kc]] + [b[3] for b in self.buf[L]])
        self.buf[L], self.nbuf[L] = [], 0
        o = t.argsort(col.double() * 2 + hh.double())
        col, hh, ii, vv = col[o], hh[o], ii[o], vv[o]
        start = t.searchsorted(col, t.arange(2 * N, device=self.device, dtype=col.dtype))
        rank = t.arange(len(col), device=self.device) - start[col]
        k = rank < K
        self.h[L].fill_(np.inf)
        self.h[L][rank[k], col[k].long()] = hh[k]
        self.ids[L][rank[k], col[k].long()] = ii[k]
        self.vals[L][rank[k], col[k].long()] = vv[k]
        self.tau[L] = self.h[L][K - 1]

    def save(self, path, ws, minutes):
        for L in range(LAYERS):
            self.compact(L)
        c = lambda z: z.cpu().numpy()  # noqa: E731
        ids = c(self.ids).reshape(LAYERS, K, 2, N).transpose(0, 2, 3, 1)  # (L, 2, N, K)
        vals = c(self.vals).reshape(LAYERS, K, 2, N).transpose(0, 2, 3, 1)
        h = c(self.h).reshape(LAYERS, K, 2, N).transpose(0, 2, 3, 1)
        ids = np.where(np.isfinite(h), ids, -1)
        cnt = c(self.count).reshape(LAYERS, 2, 2, N).transpose(0, 2, 3, 1)  # (L, 2 sign, N, 2 domain)
        np.savez_compressed(path, windows=np.asarray(ws, np.int32), ids=ids.astype(np.int32), vals=vals, count_dom=cnt,
                            K=K, min_pos=self.mp, strong=ct.RULE["strong"], minutes=minutes)


def run(a):
    import torch
    from bonsai_interp import explore
    from bonsai_interp.env import DEVICE, limit_memory
    from bonsai_interp.model import StopForward
    ws, en, _ = windows()
    if a.limit:
        ws = ws[:a.limit]
    toks = explore.corpus()
    OUT.mkdir(parents=True, exist_ok=True)
    path = Path(a.out) if a.out else OUT / "bonsai1.events.npz"
    limit_memory(0.8)
    model, tok, rec = ct._gpu_load()
    smp = Sampler(DEVICE)
    rec.mode, rec.layers, rec.stop_at = None, None, None
    rec.interventions = {L: smp.hook(L) for L in range(LAYERS)}
    t0, B = time.time(), a.batch
    print(f"sampling {len(ws)} windows ({len(en & set(map(int, ws)))} English) x {T} tokens, all {LAYERS} layers", flush=True)
    try:
        for b0 in range(0, len(ws), B):
            ct._stop_requested()
            wb = ws[b0:b0 + B]
            smp.set_batch(wb, en)
            x = torch.from_numpy(toks[wb].astype(np.int64)).to(DEVICE)
            rec.stop_at = LAYERS - 1
            try:
                with torch.no_grad():
                    model.model(input_ids=x)
            except StopForward:
                pass
            done = b0 + len(wb)
            if (b0 // B) % 50 == 0 or done == len(ws):
                el = time.time() - t0
                print(f"  {done}/{len(ws)} windows, {done * T / max(el, 1e-9):.0f} tok/s, eta {(len(ws) - done) * el / done / 60:.0f} min",
                      flush=True)
    finally:
        rec.interventions, rec.stop_at = {}, None
    ct._gpu_idle("events")
    smp.save(path, ws, round((time.time() - t0) / 60, 1))
    print(f"wrote {path} in {(time.time() - t0) / 60:.1f} min", flush=True)


def load(path=None):
    z = np.load(path or OUT / "bonsai1.events.npz")
    return {k: z[k] for k in z.files}


def stats(a):
    z = load(a.out)
    full = (z["ids"] >= 0).sum(-1)
    cnt = z["count_dom"].sum(-1)
    for L in range(LAYERS):
        print(f"L{L}: poles with >= 32 sampled {int((full[L] >= 32).sum())}, median strong count {int(np.median(cnt[L][cnt[L] > 0]))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "stats"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out")
    ap.add_argument("--batch", type=int, default=2)
    a = ap.parse_args()
    {"run": run, "stats": stats}[a.cmd](a)


if __name__ == "__main__":
    main()
