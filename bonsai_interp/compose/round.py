"""compose@1: the first compositional round. Explain a pole by the upstream word units at the offsets that feed its peaks.

  python -m bonsai_interp.compose.round protocol        # write the manifest; commit it ALONE
  python -m bonsai_interp.compose.round select          # CPU: hypotheses for every residual pole of the target layers
  python -m bonsai_interp.compose.round build           # CPU: held-out 8 + 8 tests; commit the prereg manifest ALONE
  python -m bonsai_interp.compose.round score           # GPU: judge with quick-8+8@2's rule
  (--smoke DIR: no manifests, a few hundred targets; files in DIR)

Why tokens at offsets (FINDINGS 2026-10-26). At their peaks, mid-layer poles are mostly functions of the read token and
the one before it: replacing the read token removes about 99% of a peak, the previous one 78-87% in layers 6-23, and
tokens 10 or more back almost nothing. Zeroing the top attributed upstream neurons removes only about half. Each pole's
peaks are classes (22-24 distinct read tokens in a top 32), so the class at each offset is named by the upstream word
units whose strong activations cover it.

Selection (CPU, deterministic; the scan's top 32 peaks of each target and the word screen's (token, pole) tables of
strong activations, selection windows 0-7,999):
  - For each offset k in {0, 1}: the upstream poles (layers below the target) whose strong triggers (>= 2 strong
    activations) include the tokens of the peaks at offset k. A unit is scored recall x focus:
      recall = the share of the 32 peaks covered;
      focus  = the share of the unit's own strong activations on those tokens.
    A greedy union of up to 3 units, each with focus >= 0.3, then covers the offset.
  - A 'pair' hypothesis (A here, given B one token back) needs:
      - each offset covered on >= 60% of the peaks;
      - both offsets covered together on >= 50%.
  - A 'class' hypothesis (the read token is one of the units' words) needs:
      - the read token covered on >= 70% of the peaks;
      - at least 3 distinct covered tokens (one or two words are the word screen's business).
  - Units must already be characterized (a held survey label, a held word-screen or sweep-library test): a composition
    is a sentence about known pieces, and a later round can use the units this round characterizes.
  - Targets are eligible poles of the target layers that are not assignment-grade already (the residual).
Tests (test windows 16,400 on, never a target's own top windows; 48-token snippets read at the last token; 3 preambles):
  - pair: fire = a token of class A after a token of class B. Quiet = 4 with A without B (the same read token as a fire
    item where possible) and 4 with B without A (a read token of the same word class).
  - class: fire = 8 tokens of the class, at most 2 per token. Quiet = 8 tokens of the same word class outside the class,
    excluding the target's own strong triggers.
  - A unit's trigger set is its tokens with >= 2 strong activations, most first, until they hold 80% of its strong
    activations (at most 40 tokens). Class A is the union of the read-token units' trigger sets restricted to tokens the
    target itself fires on (>= 2 strong activations in the selection windows); class B is the union of the
    previous-token units' trigger sets.
  - Placebo: one hypothesis in ten also gets a test of two quiet sets against each other, to count false holds.
Coverage (graph/SCHEMA.md): the share of the target's strong activations on its class-A tokens. This is exact for
'class'; for 'pair' it is an upper bound, alongside the share of the 32 peaks where both hold.
"""
import argparse
import datetime as dt
import gzip
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from bonsai_interp.auto import context_tier as ct
from bonsai_interp.compose import match as M

ROOT = ct.ROOT
PROTOCOL = "compose@1"
N, NN = 12288, 24576
TARGETS = list(range(6, 12))
SEL = {"min_count": 2, "focus": 0.3, "units": 3, "pair_recall": 0.6, "pair_joint": 0.5, "class_recall": 0.7, "class_distinct": 3,
       "trigger_mass": 0.8, "trigger_max": 40, "cand_per_layer": 8, "cand_total": 16, "cap": 6000, "placebo_every": 10}
SPLIT_TEST = 16400
SNIP = 48
JUDGE = "quick-8+8@2"
CODE = [Path(__file__), Path(M.__file__)]


def _paths(d=None):
    o, m = (Path(d), Path(d)) if d else (ROOT / "data" / "compose", ROOT / "graph" / "auto")
    b = "bonsai1.compose1"
    return {"protocol": m / f"{b}.protocol.json", "prereg": m / f"{b}.prereg.json", "summary": m / f"{b}.summary.json",
            "select": o / f"{b}.select.json.gz", "tests": o / f"{b}.tests.jsonl.gz", "results": o / f"{b}.results.jsonl.gz"}


def _code_sha():
    h = hashlib.sha256()
    for p in CODE:
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def _setup(a, need_prereg=False):
    from bonsai_interp.sweeplib.run import _alone
    if a.smoke:
        Path(a.smoke).mkdir(parents=True, exist_ok=True)
        return _paths(a.smoke)
    P = _paths()
    if not P["protocol"].exists() or not _alone(P["protocol"]):
        sys.exit(f"commit {P['protocol'].relative_to(ROOT)} on its own first (run `protocol`)")
    if json.loads(P["protocol"].read_text(encoding="utf-8"))["code_sha256"] != _code_sha():
        sys.exit("the compose code changed since its protocol was registered: that is a new protocol version")
    if need_prereg and (not P["prereg"].exists() or not _alone(P["prereg"])):
        sys.exit(f"commit {P['prereg'].relative_to(ROOT)} on its own before scoring (run `build`)")
    P["select"].parent.mkdir(parents=True, exist_ok=True)
    return P


def protocol(a):
    for p in CODE:
        if subprocess.run(["git", "status", "--porcelain", "--", str(p)], cwd=ROOT, capture_output=True, text=True).stdout.strip():
            sys.exit(f"commit {p.relative_to(ROOT)} first")
    P = _paths()
    man = {"protocol": PROTOCOL, "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "toolchain_commit": ct._commit(),
           "code_sha256": _code_sha(), "target_layers": TARGETS, "selection": SEL, "test_windows_from": SPLIT_TEST, "snippet": SNIP,
           "judge": JUDGE, "read": "last token, mean over the 3 PREAMBLES", "about": __doc__}
    P["protocol"].write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {P['protocol'].relative_to(ROOT)}: commit it ALONE before running any stage")


# ---------------------------------------------------------------- selection (CPU)

def _upstream(L):
    """Per upstream layer l < L: (sorted keys token * NN + pole for count >= min_count, counts, pole totals)."""
    out = {}
    for l in range(L):
        t = M._table(l)
        if t is None:
            continue
        keys, cnt, total = t
        keep = cnt >= SEL["min_count"]
        out[l] = (keys[keep], cnt[keep], total)
    return out


def _candidates(T, up):
    """T: (n, 32) peak tokens at one offset (-1 = missing). Per target: the best units over all upstream layers, each
    (score, layer, pole index, focus, bitmask of the peaks it covers). Vectorised: one sparse product per layer."""
    import scipy.sparse as sp
    n, K = T.shape
    rows = np.repeat(np.arange(n), K)
    cols = T.ravel()
    ok = cols >= 0
    V = ct.vocab()["V"]
    Q = sp.csr_matrix((np.ones(ok.sum()), (rows[ok], cols[ok])), shape=(n, V))  # multiplicities
    Qb = Q.copy()
    Qb.data[:] = 1
    n_ex = np.maximum((T >= 0).sum(1), 1).astype(np.float64)
    parts = []
    for l, (keys, cnt, total) in up.items():
        tok, pole = keys // NN, keys % NN
        B = sp.csr_matrix((np.ones(len(keys)), (tok, pole)), shape=(V, NN))
        C = sp.csr_matrix((cnt.astype(np.float64), (tok, pole)), shape=(V, NN))
        R = (Q @ B).tocoo()             # peaks covered, per (target, unit)
        F = (Qb @ C).tocsr()            # the unit's strong activations on the target's tokens
        foc = np.asarray(F[R.row, R.col]).ravel() / np.maximum(total[R.col], 1)
        good = (foc >= SEL["focus"]) & (total[R.col] >= 20)
        r, c, f = R.row[good], R.col[good], foc[good]
        sc = R.data[good] / n_ex[r] * f
        o = np.lexsort((-sc, r))  # by target, best first
        r, c, f, sc = r[o], c[o], f[o], sc[o]
        first = np.searchsorted(r, np.arange(n))
        rank = np.arange(len(r)) - first[r]
        keep = rank < SEL["cand_per_layer"]
        r, c, f, sc = r[keep], c[keep], f[keep], sc[keep]
        # which peaks each kept unit covers: (token * NN + unit) in the layer's sorted key list
        q = T[r] * NN + c[:, None]
        pos_ = np.clip(np.searchsorted(keys, q), 0, len(keys) - 1)
        hit = (keys[pos_] == q) & (T[r] >= 0)
        bits = (hit * (1 << np.arange(K, dtype=np.int64))).sum(1)
        parts.append(np.rec.fromarrays([r, np.full(len(r), l), c, f, sc, bits], names="r,l,u,f,sc,bits"))
    best = [[] for _ in range(n)]
    if parts:
        allp = np.concatenate(parts).view(np.recarray)
        allp = allp[np.lexsort((-allp.sc, allp.r))]
        first = np.searchsorted(allp.r, np.arange(n))
        rank = np.arange(len(allp)) - first[allp.r]
        allp = allp[rank < SEL["cand_total"]]
        for r_, l_, u_, f_, sc_, b_ in zip(allp.r, allp.l, allp.u, allp.f, allp.sc, allp.bits):
            best[int(r_)].append((float(sc_), int(l_), int(u_), float(f_), int(b_)))
    return best


def _union(cands, n_ex):
    """Greedy union of up to SEL['units'] candidates by newly covered peaks: (units, covered bitmask)."""
    covered, units = 0, []
    pool = list(cands)
    for _ in range(SEL["units"]):
        gain, pick = 0, None
        for c in pool:
            g = bin(c[4] & ~covered).count("1")
            if g > gain:
                gain, pick = g, c
        if pick is None or gain / n_ex < 0.05:
            break
        covered |= pick[4]
        units.append(pick)
        pool.remove(pick)
    return units, covered


@__import__("functools").lru_cache(64)
def _pole_tokens(l):
    """Pole-major view of layer l's table: (order, starts, tokens, counts) for fast per-unit trigger sets."""
    keys, cnt, total = M._table(l)
    pole = keys % NN
    o = np.argsort(pole, kind="stable")
    starts = np.searchsorted(pole[o], np.arange(NN + 1))
    return (keys[o] // NN).astype(np.int64), cnt[o], starts, total


def trigger_set(l, u):
    """A unit's class: its tokens with >= min_count strong activations, most first, until 80% of its strong mass."""
    toks, cnt, starts, total = _pole_tokens(l)
    a, b = starts[u], starts[u + 1]
    t, c = toks[a:b], cnt[a:b]
    keep = c >= SEL["min_count"]
    t, c = t[keep], c[keep]
    o = np.argsort(-c, kind="stable")
    t, c = t[o], c[o]
    cum = np.cumsum(c) / max(total[u], 1)
    k = min(int(np.searchsorted(cum, SEL["trigger_mass"]) + 1), SEL["trigger_max"], len(t))
    return [int(x) for x in t[:k]]


def select(a):
    P = _setup(a)
    from bonsai_interp import explore
    from bonsai_interp.compose.pilot import characterized
    sc, toks = explore.scan("bonsai1"), explore.corpus()
    char = characterized()
    vb = ct.vocab()
    hyps, t0 = [], time.time()
    layers = TARGETS if not a.smoke else TARGETS[:1]
    for L in layers:
        el = ct.eligibility(L)
        tg = [(n, p) for p in (0, 1) for n in np.nonzero(el[p] == "eligible")[0] if char.get((L, int(n), "+-"[p])) != "assigned"]
        if a.smoke:
            tg = tg[:300]
        up = _upstream(L)
        print(f"L{L}: {len(tg)} residual targets, {len(up)} upstream layers ({time.time() - t0:.0f} s)", flush=True)
        for b0 in range(0, len(tg), 2000):
            chunk = tg[b0:b0 + 2000]
            w = np.stack([sc["top_win"][L, p, :32, n] for n, p in chunk]).astype(np.int64)
            ps = np.stack([sc["top_pos"][L, p, :32, n] for n, p in chunk]).astype(np.int64)
            okm = (w >= 0) & (ps >= 1)
            T0 = np.where(okm, toks[np.clip(w, 0, None), ps].astype(np.int64), -1)
            T1 = np.where(okm, toks[np.clip(w, 0, None), np.clip(ps - 1, 0, None)].astype(np.int64), -1)
            c0, c1 = _candidates(T0, up), _candidates(T1, up)
            for i, (n, p) in enumerate(chunk):
                n_ex = int(okm[i].sum())
                if n_ex < 16:
                    continue
                known = lambda cs: [c for c in cs if char.get((c[1], c[2] % N, "+" if c[2] < N else "-"))]
                u0, cov0 = _union(known(c0[i]), n_ex)  # compose only from units already characterized (the sudoku rule)
                u1, cov1 = _union(known(c1[i]), n_ex)
                r0, r1 = bin(cov0).count("1") / n_ex, bin(cov1).count("1") / n_ex
                joint = bin(cov0 & cov1).count("1") / n_ex
                distinct0 = len({int(T0[i, j]) for j in range(32) if cov0 >> j & 1})
                kind = None
                if u0 and u1 and r0 >= SEL["pair_recall"] and r1 >= SEL["pair_recall"] and joint >= SEL["pair_joint"]:
                    kind = "pair"
                elif u0 and r0 >= SEL["class_recall"] and distinct0 >= SEL["class_distinct"]:
                    kind = "class"
                if not kind:
                    continue
                unit = lambda c: {"pole": f"bonsai1/L{c[1]}/N{c[2] % N}{'+' if c[2] < N else '-'}", "layer": c[1], "index": c[2],
                                  "focus": round(c[3], 3), "char": char.get((c[1], c[2] % N, '+' if c[2] < N else '-'))}
                hyps.append({"target": ct.pid(p, int(n), L), "kind": kind, "score": round(r0 * (r1 if kind == "pair" else 1), 3),
                             "recall0": round(r0, 3), "recall1": round(r1, 3), "joint": round(joint, 3),
                             "units0": [unit(c) for c in u0], "units1": [unit(c) for c in u1] if kind == "pair" else [],
                             "top_windows": sorted({int(x) for x in w[i][w[i] >= 0]})})
            print(f"  L{L}: {b0 + len(chunk)}/{len(tg)} targets, {len(hyps)} hypotheses so far ({time.time() - t0:.0f} s)", flush=True)
    hyps.sort(key=lambda h: (-h["score"], h["target"]))
    kept = hyps[:SEL["cap"]]
    ct._write_gz(P["select"], [{"hypotheses": kept, "n_all": len(hyps), "minutes": round((time.time() - t0) / 60, 1)}])
    print(f"wrote {P['select']}: {len(hyps)} hypotheses ({sum(h['kind'] == 'pair' for h in hyps)} pair, "
          f"{sum(h['kind'] == 'class' for h in hyps)} class); kept the best {len(kept)}", flush=True)


# ---------------------------------------------------------------- tests (CPU)

@__import__("functools").lru_cache(None)
def _test_positions():
    """Test-split positions with their read token and previous token, sorted by read token."""
    from bonsai_interp import explore
    toks = explore.corpus()
    t = toks[SPLIT_TEST:]
    W, Pp = np.meshgrid(np.arange(SPLIT_TEST, len(toks), dtype=np.int32), np.arange(64 + SNIP - 1, toks.shape[1], dtype=np.int32), indexing="ij")
    W, Pp = W.ravel(), Pp.ravel()
    cur = t[W - SPLIT_TEST, Pp].astype(np.int32)
    prev = t[W - SPLIT_TEST, Pp - 1].astype(np.int32)
    o = np.argsort(cur, kind="stable")
    ix = {"W": W[o], "P": Pp[o], "cur": cur[o], "prev": prev[o]}
    # near-miss pools are drawn from a fixed, seeded sample of 2M positions (a full scan per hypothesis is too slow)
    s = np.sort(np.random.default_rng(13).choice(len(o), min(len(o), 2_000_000), replace=False))
    ix["pool"] = s
    return ix


def _rows_for(ix, tokens):
    out = []
    for t in sorted(set(tokens)):
        a, b = np.searchsorted(ix["cur"], [t, t + 1])
        out.append(np.arange(a, b))
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def _text(w, p, cache={}):
    from bonsai_interp import explore
    if (w, p) not in cache:
        cache[(w, p)] = explore.tokenizer().decode(explore.corpus()[w, p - SNIP + 1:p + 1].tolist())
    return cache[(w, p)]


def _pick(rng, ix, rows, k, used, per_token=2, avoid=()):
    out, per = [], {}
    for r in rng.permutation(rows):
        w, c = int(ix["W"][r]), int(ix["cur"][r])
        if w in used or w in avoid or per.get(c, 0) >= per_token:
            continue
        used.add(w)
        per[c] = per.get(c, 0) + 1
        out.append((w, int(ix["P"][r])))
        if len(out) >= k:
            break
    return out


def _own_strong(target):
    """The target's own strong triggers (>= 2 strong activations in the selection windows): never a quiet item."""
    L, rest = target.split("/L")[1].split("/N")
    L, n, p = int(L), int(rest[:-1]), 0 if rest[-1] == "+" else 1
    toks, cnt, starts, total = _pole_tokens(L)
    u = p * N + n
    a, b = starts[u], starts[u + 1]
    return set(int(t) for t, c in zip(toks[a:b], cnt[a:b]) if c >= 2), (toks[a:b], cnt[a:b], int(total[u]))


def build(a):
    P = _setup(a)
    sel = ct._read_gz(P["select"])[0]
    hyps = sel["hypotheses"][:120] if a.smoke else sel["hypotheses"]
    ix, vb = _test_positions(), ct.vocab()
    rows, n_test, n_unt, n_plac = [], 0, 0, 0
    for h_i, h in enumerate(hyps):
        rng = np.random.default_rng(int(hashlib.sha256(h["target"].encode()).hexdigest()[:12], 16))
        own, (own_t, own_c, own_tot) = _own_strong(h["target"])
        # class A: the units' words that the target itself fires on (>= 2 strong activations in the selection windows);
        # a unit's whole trigger set is broader than what this pole responds to
        A = sorted({t for u in h["units0"] for t in trigger_set(u["layer"], u["index"])} & own)
        Bc = sorted({t for u in h["units1"] for t in trigger_set(u["layer"], u["index"])})
        avoid, used = set(h["top_windows"]), set()
        Aset, Bset = set(A), set(Bc)
        coverage = float(sum(c for t, c in zip(own_t, own_c) if int(t) in Aset)) / max(own_tot, 1)
        cand_A = _rows_for(ix, A)
        if h["kind"] == "pair":
            inB = np.isin(ix["prev"][cand_A], Bc)
            fire = _pick(rng, ix, cand_A[inB], 8, used, per_token=2 if len(A) > 3 else 8, avoid=avoid)  # a fixed bigram repeats its word
            q1 = _pick(rng, ix, cand_A[~inB], 4, used, per_token=2 if len(A) > 3 else 8, avoid=avoid)  # the word without the context
            fire_cls = {vb["cls"][t] for t in A}
            other = ix["pool"][np.isin(ix["prev"][ix["pool"]], Bc)]
            other = other[~np.isin(ix["cur"][other], A) & ~np.isin(ix["cur"][other], list(own))]
            other = other[np.isin(np.array([vb["cls"][int(c)] for c in ix["cur"][other]], dtype=object), list(fire_cls))] if len(other) else other
            q2 = _pick(rng, ix, other, 4, used, avoid=avoid)  # the context without the word
            quiet, kinds = q1 + q2, ["word without context"] * len(q1) + ["context without word"] * len(q2)
            ok = len(fire) == 8 and len(q1) == 4 and len(q2) == 4
            label = (f"composition: a word of class A [{', '.join(repr(vb['text'][t]) for t in A[:6])}] right after a word of class B "
                     f"[{', '.join(repr(vb['text'][t]) for t in Bc[:6])}]")
        else:
            fire = _pick(rng, ix, cand_A, 8, used, avoid=avoid)
            cls = {vb["cls"][t] for t in A}
            pool = ix["pool"]
            pool = pool[~np.isin(ix["cur"][pool], A) & ~np.isin(ix["cur"][pool], list(own))]
            sample = rng.choice(pool, min(len(pool), 200000), replace=False)
            sample = sample[np.isin(np.array([vb["cls"][int(c)] for c in ix["cur"][sample]], dtype=object), list(cls))]
            quiet = _pick(rng, ix, sample, 8, used, avoid=avoid)
            kinds = ["same word class, outside the class"] * len(quiet)
            ok = len(fire) == 8 and len(quiet) == 8
            label = f"composition: a word of the class [{', '.join(repr(vb['text'][t]) for t in A[:8])}] (the words of upstream units)"
        units = {"A": [u["pole"] for u in h["units0"]], "B": [u["pole"] for u in h["units1"]]}
        base = {"type": "test", "target": h["target"], "kind": h["kind"], "label": label, "units": units, "class_A": A, "class_B": Bc,
                "select": {k: h[k] for k in ("score", "recall0", "recall1", "joint")}, "coverage_A": round(coverage, 4)}
        if not ok:
            rows.append({**base, "class": "untestable", "why": f"fire {len(fire)}, quiet {len(quiet)} (8 + 8 needed)"})
            n_unt += 1
            continue
        rows.append({**base, "class": "tested", "fire": [_text(*x) for x in fire], "quiet": [_text(*x) for x in quiet], "quiet_kind": kinds})
        n_test += 1
        if h_i % SEL["placebo_every"] == 0:
            pq = _pick(rng, ix, cand_A[~np.isin(ix["prev"][cand_A], Bc)] if h["kind"] == "pair" else sample, 16, used, avoid=avoid)
            if len(pq) == 16:
                rows.append({**base, "class": "placebo", "fire": [_text(*x) for x in pq[:8]], "quiet": [_text(*x) for x in pq[8:]],
                             "quiet_kind": ["placebo"] * 8})
                n_plac += 1
        if (h_i + 1) % 500 == 0:
            print(f"  {h_i + 1}/{len(hyps)} hypotheses built", flush=True)
    sha = ct._write_gz(P["tests"], rows)
    from bonsai_interp.sweeplib.run import _alone
    pre = {"protocol": PROTOCOL, "protocol_commit": _alone(P["protocol"]) if not a.smoke else None,
           "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "tests_file": P["tests"].name, "tests_sha256": sha,
           "select_sha256": hashlib.sha256(gzip.open(P["select"], "rb").read()).hexdigest(),
           "counts": {"tested": n_test, "untestable": n_unt, "placebo": n_plac}}
    P["prereg"].write_text(json.dumps(pre, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(pre["counts"]), f"\nwrote {P['tests']} and {P['prereg']}: commit the manifest ALONE before scoring", flush=True)


# ---------------------------------------------------------------- scoring (GPU)

def score(a):
    P = _setup(a, need_prereg=True)
    from bonsai_interp.env import limit_memory
    from bonsai_interp.graph.protocols import PROTOCOLS
    from bonsai_interp.probe.textinv import PREAMBLES
    from bonsai_interp.sweeplib.run import _pm, read_last
    pre = json.loads(P["prereg"].read_text(encoding="utf-8"))
    blob = gzip.open(P["tests"], "rb").read()
    if hashlib.sha256(blob).hexdigest() != pre["tests_sha256"]:
        sys.exit("the tests file does not match its pre-registered sha256")
    rows = [json.loads(x) for x in blob.decode("utf-8").splitlines()]
    tests = [r for r in rows if r["class"] in ("tested", "placebo")]
    want = {}
    for r in tests:
        L, rest = r["target"].split("/L")[1].split("/N")
        key = (int(L), 0 if rest[-1] == "+" else 1, int(rest[:-1]))
        for t in r["fire"] + r["quiet"]:
            want.setdefault(t, set()).add(key)
    by_top = {}
    for t, ks in want.items():
        by_top.setdefault(max(k[0] for k in ks), []).append(t)
    pm = _pm()
    limit_memory(0.8)
    model, tok, rec = ct._gpu_load()
    t0, val = time.time(), {}
    print(f"scoring {len(tests)} tests on {len(want)} snippets x {len(PREAMBLES)} preambles", flush=True)
    for top in sorted(by_top):
        ts = sorted(by_top[top])
        for b in range(0, len(ts), 256):
            sl = ts[b:b + 256]
            acc = np.zeros((len(sl), top + 1, N), np.float32)
            for pre_text in PREAMBLES:
                acc += read_last(model, tok, rec, sl, pre_text, a.batch, top) / len(PREAMBLES)
            for i, t in enumerate(sl):
                for L, p, n in want[t]:
                    val[(t, L, p, n)] = round(float(acc[i, L, n] / pm[L, 0, n] if p == 0 else -acc[i, L, n] / pm[L, 1, n]), 4)
        print(f"  texts to layer {top}: {len(ts)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    ct._gpu_idle("compose score")
    rule = PROTOCOLS[JUDGE]["rule"]
    out, tally = [], {}
    for r in tests:
        L, rest = r["target"].split("/L")[1].split("/N")
        key = (int(L), 0 if rest[-1] == "+" else 1, int(rest[:-1]))
        per = [{"set": "fire", "text": t, "act": val[(t, *key)]} for t in r["fire"]] + \
              [{"set": "quiet", "text": t, "act": val[(t, *key)], "kind": k} for t, k in zip(r["quiet"], r["quiet_kind"])]
        res = rule({"per_sentence": per})
        k = f"{'placebo' if r['class'] == 'placebo' else r['kind']}"
        tally.setdefault(k, {}).setdefault(res, 0)
        tally[k][res] += 1
        out.append({**{x: r[x] for x in ("target", "kind", "label", "units", "class_A", "class_B", "select", "coverage_A")},
                    "placebo": r["class"] == "placebo", "outcome": res, "per_sentence": per})
    sha = ct._write_gz(P["results"], out)
    held = [o for o in out if o["outcome"] == "held" and not o["placebo"]]
    summ = {"protocol": PROTOCOL, "scored": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "toolchain_commit": ct._commit(),
            "outcomes": tally, "poles_held": len({o["target"] for o in held}),
            "held_assignment_grade": sum(o["coverage_A"] >= 0.5 for o in held),
            "held_by_layer": {L: sum(o["target"].startswith(f"bonsai1/L{L}/") for o in held) for L in TARGETS},
            "held_with_characterized_units": sum(any(True for _ in o["units"]["A"]) for o in held),
            "minutes": round((time.time() - t0) / 60, 1), "results_file": P["results"].name, "results_sha256": sha}
    P["summary"].write_text(json.dumps(summ, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: summ[k] for k in ("outcomes", "poles_held", "held_assignment_grade", "minutes")}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["protocol", "select", "build", "score"])
    ap.add_argument("--smoke", help="a scratch directory: no manifests, a few hundred targets")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    {"protocol": protocol, "select": select, "build": build, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
