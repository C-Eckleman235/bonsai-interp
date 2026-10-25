"""Prediction neurons: found from the weights (an inverse analysis of the output side), tested on the model's own predictions.

1. Direct effect. For each pole, W_U · (final-norm gain ⊙ W_down[:, n]) · sign gives the 20 tokens it promotes most (its
   set S) and its spike: how far the 20th token stands above the vocabulary mean, in standard deviations.
2. Selector (CPU). Take the pole's 32 strongest corpus activations from the scan and count how many are followed by a
   token in S. A pole is a candidate when at least 8 of 32 are, at 10 times or more the rate S's tokens occur in the
   corpus. At most CAP per layer, ranked by that count. The tested set is the tokens of S that follow those activations at
   least twice (at most SEEN_MAX, most frequent first): what the pole promotes and is seen to precede.
3. Test construction (GPU, logits only, no neuron is read). Candidate snippets come from the test split: 24 where the
   corpus continues with a token in S, and 16 that end on the same tokens but continue otherwise. The pole's own top
   windows are excluded. The model reads each snippet after the first preamble. (S below is the tested set.)
     - fire: the model puts at least 0.5 of its next-token probability on S;
     - quiet: the model puts at most 0.05 on S, ending on the same token as a fire snippet when one exists.
   So the near miss keeps the current word and differs only in what the model expects next.
"""
from functools import lru_cache

import numpy as np

from bonsai_interp.auto import context_tier as ct
from bonsai_interp.sweeplib import kinds as K

TOPK, SPIKE_MIN = 20, 4.0
SEL = {"hits": 8, "lift": 10.0}
CAP = 64
SEEN_MAX = 6
POOL = {"fire": 24, "quiet": 16}
P_FIRE, P_QUIET = 0.5, 0.05
N = 12288


def direct_effect(model, L, chunk=512):
    """(S (2, N, TOPK) token ids, spike (2, N)) for layer L, on the GPU."""
    import torch
    WU = model.lm_head.weight  # (V, d)
    g = model.model.norm.weight.float()
    Wd = model.model.layers[L].mlp.down_proj.weight()  # (d, N)
    S = np.zeros((2, N, TOPK), np.int32)
    spike = np.zeros((2, N), np.float32)
    for c0 in range(0, N, chunk):
        cols = (Wd[:, c0:c0 + chunk].float() * g[:, None]).to(WU.dtype)
        E = (WU @ cols).float()  # (V, chunk)
        mu, sd = E.mean(0), E.std(0).clamp_min(1e-9)
        for p, sign in ((0, 1.0), (1, -1.0)):
            v, i = (sign * E).topk(TOPK, dim=0)
            S[p, c0:c0 + chunk] = i.T.cpu().numpy()
            spike[p, c0:c0 + chunk] = ((v[-1] - sign * mu) / sd).cpu().numpy()
        del E
    return S, spike


@lru_cache(None)
def _unigram():
    from bonsai_interp import explore
    t = explore.corpus()
    return np.bincount(t.ravel(), minlength=ct.vocab()["V"]) / t.size


def candidates(L, S, spike, eligible):
    """[(n, pole, hits, lift, tested set)] for layer L, best first, at most CAP."""
    from bonsai_interp import explore
    sc, toks = explore.scan("bonsai1"), explore.corpus()
    uni = _unigram()
    out = []
    for p in (0, 1):
        w, pos = sc["top_win"][L, p].astype(np.int64), sc["top_pos"][L, p].astype(np.int64)  # (32, N)
        ok = (w >= 0) & (pos + 1 < toks.shape[1])
        nxt = np.where(ok, toks[np.clip(w, 0, len(toks) - 1), np.clip(pos + 1, 0, toks.shape[1] - 1)].astype(np.int64), -1)
        for n in np.nonzero(eligible[p] & (spike[p] >= SPIKE_MIN))[0]:
            s = S[p, n]
            hit = np.isin(nxt[:, n], s)
            if hit.sum() < SEL["hits"]:
                continue
            lift = hit.sum() / ok[:, n].sum() / max(uni[s].sum(), 1e-12)
            if lift >= SEL["lift"]:
                # the tested set: the promoted tokens that actually follow the top activations (at least twice), most often first
                t, c = np.unique(nxt[hit, n], return_counts=True)
                seen = [int(x) for x, k in sorted(zip(t, c), key=lambda tc: (-tc[1], tc[0])) if k >= 2][:SEEN_MAX] or [int(t[c.argmax()])]
                out.append((int(n), p, int(hit.sum()), round(float(lift), 1), seen))
    out.sort(key=lambda r: (-r[2], -r[3], r[1], r[0]))
    return out[:CAP]


@lru_cache(None)
def _test_index():
    """Test-split positions (window, p) sorted by the next token and by the current token."""
    from bonsai_interp import explore
    toks = explore.corpus()
    lo = K.SPLITS["test"][0]
    t = toks[lo:].astype(np.int64)
    W, P = np.meshgrid(np.arange(lo, len(toks)), np.arange(K.MIN_POS + K.SNIP - 1, toks.shape[1] - 1), indexing="ij")
    W, P = W.ravel().astype(np.int32), P.ravel().astype(np.int32)
    cur, nxt = t[W - lo, P].astype(np.int32), t[W - lo, P + 1].astype(np.int32)
    del t
    o_n, o_c = np.argsort(nxt, kind="stable").astype(np.int32), np.argsort(cur, kind="stable").astype(np.int32)
    return {"W": W, "P": P, "cur": cur, "nxt": nxt, "by_next": o_n, "next_sorted": nxt[o_n], "by_cur": o_c, "cur_sorted": cur[o_c]}


def _rows(ix, sorted_key, order, vals):
    out = []
    for v in vals:
        a, b = np.searchsorted(ix[sorted_key], [v, v + 1])
        out.append(ix[order][a:b])
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def pool(L, n, p, s):
    """Candidate snippet ends (window, position) for one pole: POOL['fire'] continuing with a token in S, POOL['quiet']
    ending on the same tokens but continuing otherwise. Never in the pole's own 32 top windows. Seeded."""
    from bonsai_interp import explore
    ix, sc = _test_index(), explore.scan("bonsai1")
    rng = np.random.default_rng([L, n, p, 5])
    top = set(sc["top_win"][L, p, :, n].tolist())
    f = _rows(ix, "next_sorted", "by_next", np.unique(s))
    f = f[~np.isin(ix["W"][f], list(top))]
    f = rng.permutation(f)
    fire, seen = [], set()
    for i in f:
        if ix["W"][i] not in seen:
            seen.add(ix["W"][i])
            fire.append(int(i))
        if len(fire) >= POOL["fire"]:
            break
    quiet = []
    for c in rng.permutation(np.unique(ix["cur"][fire])) if fire else []:
        q = _rows(ix, "cur_sorted", "by_cur", [c])
        q = q[~np.isin(ix["nxt"][q], s) & ~np.isin(ix["W"][q], list(top | seen))]
        for i in rng.permutation(q)[:2]:
            if ix["W"][i] not in seen and len(quiet) < POOL["quiet"]:
                seen.add(ix["W"][i])
                quiet.append(int(i))
    return [(int(ix["W"][i]), int(ix["P"][i])) for i in fire], [(int(ix["W"][i]), int(ix["P"][i])) for i in quiet]


def choose(fire, quiet, pS, cur, rng):
    """The 8 + 8 test from the pooled snippets' P(S): fire with P(S) >= P_FIRE, quiet with P(S) <= P_QUIET, quiet on a
    fire snippet's last token first. None when either side has fewer than 8."""
    F = [i for i in fire if pS[i] >= P_FIRE]
    Q = [i for i in quiet if pS[i] <= P_QUIET]
    if len(F) < K.N_TEST or len(Q) < K.N_TEST:
        return None
    F = [F[i] for i in rng.permutation(len(F))[:K.N_TEST]]
    fc = {cur[i] for i in F}
    Q = sorted(Q, key=lambda i: (cur[i] not in fc, rng.random()))[:K.N_TEST]
    return F, Q
