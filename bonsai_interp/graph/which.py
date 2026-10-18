"""Reverse search: from text or a token to the neuron poles that fire on it.

  python -m bonsai_interp.graph.play which "It takes a lot of nerve to say that."
      one forward pass; for each token, the poles across all 36 layers with the highest activation as a
      fraction of that pole's corpus max (both poles), marking the ones that have graph records
  python -m bonsai_interp.graph.play which --token " nerve"
      the poles whose top-32 corpus peaks most often land on this token (from the token index)

The token index (data/token_index_bonsai1.npz) is derived from the corpus scan and the corpus alone, so it
is rebuilt on first use (about a minute, CPU) when it isn't on disk. It is a library artifact, never
committed (DATA_POLICY.md). It also holds per-pole density and peak-position statistics, which `which`
uses to hide dense and positional poles: they top every token and say nothing about it.
"""
import hashlib
import json
import sys
import time

import numpy as np

from bonsai_interp import explore
from bonsai_interp.graph import card as cards
from bonsai_interp.graph.contrib import ROOT

NAME = "bonsai1"
INDEX = ROOT / "data" / f"token_index_{NAME}.npz"
INDEX_VERSION = 1
NL, NN = 36, 12288


def _pid(L, p, n):
    return (L * 2 + p) * NN + n


def _unpid(pid):
    pid = int(pid)
    return pid // (2 * NN), (pid // NN) % 2, pid % NN


def build_index(out=INDEX):
    """token -> poles from the scan's peak tokens, plus per-pole density (vectorised over all 884,736 poles)."""
    t0 = time.time()
    s, toks = explore.scan(NAME), explore.corpus()
    val, win, pos = s["top_val"], s["top_win"], s["top_pos"]  # (36, 2, 32, 12288)
    valid = np.isfinite(val) & (win >= 0)
    peak = np.where(valid, toks[np.clip(win, 0, None), pos.astype(np.int64)], 0).astype(np.int64)
    # rank axis outermost, so np.unique's first occurrence is the best (lowest) rank of a (pole, token) pair
    shape = (32, NL, 2, NN)
    pid = np.broadcast_to(((np.arange(NL)[:, None, None] * 2 + np.arange(2)[None, :, None]) * NN + np.arange(NN)), shape)
    rank = np.broadcast_to(np.arange(32)[:, None, None, None], shape)
    sel = valid.transpose(2, 0, 1, 3)
    pid_f, rank_f, tok_f = pid[sel], rank[sel], peak.transpose(2, 0, 1, 3)[sel]
    key = tok_f * (NL * 2 * NN) + pid_f
    uk, first, count = np.unique(key, return_index=True, return_counts=True)
    best = rank_f[first]
    keep = (count >= 2) | (best == 0)  # a token seen once, below the top example, is noise
    uk, count, best = uk[keep], count[keep], best[keep]
    tok_u, pid_u = uk // (NL * 2 * NN), uk % (NL * 2 * NN)
    # per-pole density above 0.1 / 0.3 of max and peak-position clustering (same rules as the card)
    hist, bins = s["hist"], s["bins"].astype(np.float64)
    lower, upper = np.concatenate([[0], bins]), np.concatenate([bins, [np.inf]])
    total = hist[0, 0].sum()
    dens = np.zeros((2, NL, 2, NN), np.float32)
    for p, regs in ((0, [0, 1]), (1, [2, 3])):
        mx = val[:, p, 0, :].astype(np.float64)  # (36, NN)
        for L_ in range(NL):
            h = hist[L_][:, regs].sum(1).astype(np.float64)  # (NN, NB)
            rc = np.cumsum(h[:, ::-1], 1)[:, ::-1]  # rc[:, b] = count in bins >= b
            rc = np.concatenate([rc, np.zeros((NN, 1))], 1)
            for fi, f in enumerate((0.1, 0.3)):
                t = f * mx[L_]
                b = np.clip(np.searchsorted(lower, t, side="right") - 1, 0, len(lower) - 1)
                lo, hi = lower[b], upper[b]
                with np.errstate(divide="ignore", invalid="ignore"):  # share of the straddling bin above t, log-uniform
                    part = np.where(np.isfinite(hi) & (lo > 0), (np.log(hi) - np.log(t)) / (np.log(hi) - np.log(lo)),
                                    np.where(lo == 0, (hi - t) / hi, 0))
                part = np.nan_to_num(np.clip(part, 0, 1))
                hb = h[np.arange(NN), b]
                dens[fi, L_, p] = (rc[np.arange(NN), b + 1] + hb * part) / total
    early = ((pos < 16) & valid).sum(2) / np.maximum(valid.sum(2), 1)  # (36, 2, NN)
    np.savez_compressed(out, version=INDEX_VERSION, tok=tok_u.astype(np.int32), pid=pid_u.astype(np.int32),
                        count=count.astype(np.uint8), best=best.astype(np.uint8), pole_max=val[:, :, 0, :].astype(np.float32),
                        dens10=dens[0], dens30=dens[1], early=early.astype(np.float16))
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"built {out.relative_to(ROOT).as_posix()} in {time.time() - t0:.0f} s: {len(tok_u):,} (token, pole) pairs, "
          f"{out.stat().st_size / 1e6:.1f} MB, sha256 {h}")
    return out


def load_index(rebuild=False):
    if rebuild or not INDEX.exists():
        print("building the token index from the scan (once; about a minute) ...", file=sys.stderr)
        build_index()
    d = np.load(INDEX)
    if int(d["version"]) != INDEX_VERSION:
        build_index()
        d = np.load(INDEX)
    return d


def _records():
    """{(L, n, '+'/'-'): index entry} for every pole with a graph record."""
    idx = json.loads((ROOT / "graph" / "index.json").read_text(encoding="utf-8"))["neurons"]
    out = {}
    for nid, r in idx.items():
        _, L, n, p = cards.parse(nid)
        out[(L, n, p)] = r
    return out


def _tag(rec):
    if not rec:
        return ""
    return f"  [{rec['status']}: {(rec.get('short') or rec.get('label') or '')[:60]}]"


def _hidden(d, keep_dense, min_max):
    """(36, 2, NN) mask of poles to hide: tiny max, and (unless keep_dense) dense or positional."""
    hide = d["pole_max"] < min_max
    if not keep_dense:
        hide |= (d["dens10"] >= cards.DENSE_10) | (d["early"].astype(np.float32) >= cards.POSITIONAL)
    return hide


def by_token(d, token, k, keep_dense, min_max):
    tk = explore.tokenizer()
    ids = tk(token, add_special_tokens=False).input_ids
    if len(ids) != 1:
        pieces = [tk.decode([i]) for i in ids]
        print(f"{token!r} is {len(ids)} tokens {pieces}; showing the first. Pass one token, e.g. ' nerve' with its leading space.")
    tid = ids[0]
    lo, hi = np.searchsorted(d["tok"], tid, "left"), np.searchsorted(d["tok"], tid, "right")
    pid, count, best = d["pid"][lo:hi], d["count"][lo:hi], d["best"][lo:hi]
    hide = _hidden(d, keep_dense, min_max).ravel()
    ok = ~hide[pid]
    order = np.lexsort((best[ok], -count[ok].astype(int)))
    recs = _records()
    print(f"token {tk.decode([tid])!r} (id {tid}): peak token of {len(pid):,} poles' top examples ({int((~ok).sum())} dense/positional/tiny hidden)")
    print(f"  {'pole':<16} {'top-32 peaks on it':<19} {'best rank':<10} density>0.1·max")
    for i in order[:k]:
        L, p, n = _unpid(pid[ok][i])
        pole = "+-"[p]
        print(f"  {f'L{L}/N{n}{pole}':<16} {f'{int(count[ok][i])}/32':<19} {f'#{int(best[ok][i]) + 1}':<10} "
              f"{float(d['dens10'][L, p, n]):.3%}{_tag(recs.get((L, n, pole)))}")


def by_text(d, text, k, keep_dense, min_max):
    import torch
    from bonsai_interp.model import load
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)
    model, tok, rec = load(NAME)
    dev = model.model.embed_tokens.weight.device
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    rec.mode, rec.layers, rec.stop_at = "act", None, NL - 1
    from bonsai_interp.model import StopForward
    with torch.no_grad():
        try:
            model.model(ids)
        except StopForward:
            pass
    acts = torch.stack([rec.store[L][0].float() for L in range(NL)], 1)  # (T, 36, NN)
    rec.mode, rec.stop_at = None, None
    rec.clear()
    pmax = torch.from_numpy(d["pole_max"]).to(dev)  # (36, 2, NN), both positive
    frac = torch.stack([acts / pmax[:, 0], -acts / pmax[:, 1]], 2)  # (T, 36, 2, NN)
    frac[:, torch.from_numpy(_hidden(d, keep_dense, min_max)).to(dev)] = -float("inf")
    flat = frac.reshape(frac.shape[0], -1)
    top = flat.topk(k, dim=1)
    recs = _records()
    pieces = [tok.decode([i]) for i in ids[0].tolist()]
    print(f"{len(pieces)} tokens; per token, the top {k} poles by activation / pole's corpus max "
          f"(dense, positional and max<{min_max} poles hidden{'' if not keep_dense else ': no, kept'}); [status] = has a graph record")
    for t, piece in enumerate(pieces):
        cells = []
        for v, i in zip(top.values[t].tolist(), top.indices[t].tolist()):
            L, p, n = _unpid(i)
            pole = "+-"[p]
            cells.append(f"L{L}/N{n}{pole} {v:.2f}{_tag(recs.get((L, n, pole)))}")
        print(f"{t:>3} {piece!r:<14} " + cells[0])
        for c in cells[1:]:
            print(" " * 19 + c)
    best = flat.max(0)
    tv = best.values.topk(10)
    print("over the whole text: " + ", ".join(f"L{_unpid(i)[0]}/N{_unpid(i)[2]}{'+-'[_unpid(i)[1]]} {v:.2f} at {pieces[int(best.indices[i])]!r}"
                                         for v, i in zip(tv.values.tolist(), tv.indices.tolist())))
    print("next: python -m bonsai_interp.graph.play card --neuron bonsai1/<pole> to see one, or probe it with your own sentences")


def main(a):
    if not a.text and not a.token:
        sys.exit('give text ("which \\"some text\\"") or --token " nerve"')
    d = load_index(a.rebuild)
    if a.token is not None:
        by_token(d, a.token, a.k, a.keep_dense, a.min_max)
    if a.text:
        by_text(d, a.text, a.k, a.keep_dense, a.min_max)
