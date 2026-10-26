"""Name a pole's local inputs by upstream poles: the composition hypothesis (CPU, deterministic).

For a target pole B, take its strong peaks (the scan's top examples) and the tokens at the offsets that matter (the
occlusion pilot: the current token and the one before it carry most of a mid-layer pole's peak). For each offset, find
the upstream pole A (layer < B's layer) whose strong activations best cover that token set, using the word screen's full
(token, pole) tables of strong activations:
    recall(A)   = the share of B's peaks whose token at that offset is a strong trigger of A (>= 2 strong activations)
    focus(A)    = the share of A's own strong activations that fall on those tokens
    score(A)    = recall x focus  (A fires on these tokens, and these tokens are what A is about)
The hypothesis for B is then "A0 at the read token AND A1 one token back", which a factorial test can check. When A is
already characterized, B's label composes from A's label: the snowball.

  python -m bonsai_interp.compose.match bonsai1/L10/N5404-
"""
import sys
from collections import Counter
from functools import lru_cache

import numpy as np

N = 12288
NN = 2 * N


@lru_cache(12)
def _table(L):
    """Token-major view of layer L's strong-activation table: (keys = token * NN + pole, sorted; counts; pole totals)."""
    from bonsai_interp.graph.coverage import HITS
    p = HITS / f"bonsai1_L{L}.hits.npz"
    if not p.exists():
        return None
    d = np.load(p)
    k, c = d["uni_keys"], d["uni_counts"]
    o = np.argsort(k, kind="stable")
    return k[o], c[o].astype(np.int64), d["total"].astype(np.int64)


def peaks(L, n, p, k=32):
    from bonsai_interp import explore
    sc, toks = explore.scan("bonsai1"), explore.corpus()
    w, pos = sc["top_win"][L, p, :k, n].astype(int), sc["top_pos"][L, p, :k, n].astype(int)
    ok = (w >= 0) & (pos >= 2)
    return [(int(a), int(b)) for a, b in zip(w[ok], pos[ok])], toks


def best_upstream(tokens, L, top=3, min_count=2):
    """[(score, recall, focus, layer, neuron, sign)] for upstream poles covering this multiset of tokens."""
    S = Counter(tokens)
    scores = {}
    for l in range(L):
        t = _table(l)
        if t is None:
            continue
        keys, cnt, total = t
        hits = {}  # pole -> (examples covered, strong activations on S)
        for tok, mult in S.items():
            a, b = np.searchsorted(keys, [tok * NN, (tok + 1) * NN])
            poles, cs = keys[a:b] % NN, cnt[a:b]
            for pl, c in zip(poles[cs >= min_count], cs[cs >= min_count]):
                e, s = hits.get(pl, (0, 0))
                hits[pl] = (e + mult, s + c)
        n_ex = sum(S.values())
        for pl, (e, s) in hits.items():
            tot = total[pl]
            if tot < 20:
                continue
            rec, foc = e / n_ex, s / tot
            scores[(l, int(pl))] = (rec * foc, rec, foc)
    best = sorted(scores.items(), key=lambda kv: -kv[1][0])[:top]
    return [(round(v[0], 3), round(v[1], 2), round(v[2], 2), l, pl % N, "+" if pl < N else "-") for (l, pl), v in best]


def hypothesis(nid, offsets=(0, 1)):
    L, rest = nid.split("/L")[1].split("/N")
    L, n, p = int(L), int(rest[:-1]), 0 if rest[-1] == "+" else 1
    pk, toks = peaks(L, n, p)
    out = {}
    for off in offsets:
        tokens = [int(toks[w, q - off]) for w, q in pk]
        out[off] = {"tokens": Counter(tokens).most_common(8), "upstream": best_upstream(tokens, L)}
    return out


def main(nids):
    import json
    from pathlib import Path
    from bonsai_interp import explore
    tk = explore.tokenizer()
    ROOT = Path(__file__).resolve().parents[2]
    ix = json.loads((ROOT / "graph" / "index.json").read_text(encoding="utf-8"))["neurons"]
    from bonsai_interp.compose.pilot import label_of
    for nid in nids:
        h = hypothesis(nid)
        print(f"\n{nid}  ({(ix.get(nid) or {}).get('label') or 'unlabelled'})")
        for off, v in h.items():
            toks_s = ", ".join(f"{tk.decode([t])!r}x{c}" for t, c in v["tokens"][:6])
            print(f"  offset {off}: {toks_s}")
            for sc_, rec, foc, l, m, s in v["upstream"]:
                print(f"     score {sc_:.3f} (recall {rec:.0%}, focus {foc:.0%})  L{l}/N{m}{s}  {label_of((l, m, s), ix)}")


if __name__ == "__main__":
    main(sys.argv[1:])


def cover(tokens, L, k=3, min_focus=0.3, min_count=2):
    """Greedy union of up to k upstream poles that together cover the most of B's peaks at one offset, each pole focused
    on these tokens (at least min_focus of its own strong activations): [(layer, neuron, sign, recall gained, focus)], and
    the union's recall. 'specifically' OR 'especially' is two word units; a class no unit covers stays uncovered (a gap)."""
    S = Counter(tokens)
    n_ex = sum(S.values())
    cand = {}  # pole -> (set of tokens it fires on, focus)
    for l in range(L):
        t = _table(l)
        if t is None:
            continue
        keys, cnt, total = t
        on, mass = {}, {}
        for tok in S:
            a, b = np.searchsorted(keys, [tok * NN, (tok + 1) * NN])
            poles, cs = keys[a:b] % NN, cnt[a:b]
            for pl, c in zip(poles[cs >= min_count], cs[cs >= min_count]):
                on.setdefault(pl, set()).add(tok)
                mass[pl] = mass.get(pl, 0) + c
        for pl, ts in on.items():
            if total[pl] >= 20 and mass[pl] / total[pl] >= min_focus:
                cand[(l, int(pl))] = (ts, mass[pl] / total[pl])
    covered, out = set(), []
    for _ in range(k):
        best, gain = None, 0
        for key, (ts, foc) in cand.items():
            g = sum(S[t] for t in ts - covered)
            if g > gain or (g == gain and best is not None and foc > cand[best][1]):
                best, gain = key, g
        if best is None or gain / n_ex < 0.05:
            break
        ts, foc = cand.pop(best)
        covered |= ts
        l, pl = best
        out.append((l, pl % N, "+" if pl < N else "-", round(gain / n_ex, 2), round(foc, 2)))
    return out, round(sum(S[t] for t in covered) / n_ex, 2)
