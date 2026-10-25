"""Coverage: how much of a pole's activity its label accounts for.

A held test shows that a label's trigger reliably fires the pole (sufficiency). Coverage is the other half: the share of the
pole's strong activations that the label describes. The user's rule (2026-10-25): a held label counts as the pole's
assignment only if it covers at least half of the strong activations. Below that it is 'partial': true information about
the pole, not its assignment. "That's information, not assignment."

Two measures:
  1. Token coverage (deterministic, CPU). For labels whose trigger is a set of tokens: the share of the pole's strong
     activations (>= 0.3 of its max; auto-token@2's selection windows 0-7,999, positions >= 64) that fall on a trigger
     token. The source is the word screen's full (token, pole) tables (data/auto2/bonsai1_L*.hits.npz, release
     auto2-data). The trigger comes from a declared `trigger` on the contribution:
       {"words": [...]}   a token whose text, stripped and lower-cased, is one of the words, or a 3+ letter piece of how
                          the tokenizer splits one of them; with "family": true, also a token that starts with a word of
                          4+ letters (break -> breaks, breaking)
       {"pieces": [...]}  a token whose stripped, lower-cased text is exactly the piece (the piece as its own token)
       {"tokens": [...]}  exact token strings, leading space included
     Older labels with no declared trigger are measured from the clean word-like terms they quote (as a word family),
     but only when the label is word-shaped: unit kind lexical, token, word or spelling, or a label that names a quoted
     trigger ("the word 'X'", "the token ' X'", "the letters 'ur'"). This is an estimate; declaring a trigger in a new
     contribution replaces it.
  2. A coverage round (protocol coverage-16@1), for labels whose trigger is a context rather than a token set.
     `play coverage --neuron X` draws 16 of the pole's strong activations at random from held-out windows (seeded by the
     pole and the label's contribution id). The author marks each one "the label describes this" or not, and the share
     marked is the coverage.

`python -m bonsai_interp.graph.coverage build` writes graph/auto/bonsai1.coverage.json (token coverage for every held or
partial label it can measure). fold reads it, and reads the coverage rounds from the contributions.
"""
import argparse
import gzip
import hashlib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "graph" / "auto" / "bonsai1.coverage.json"
HITS = ROOT / "data" / "auto2"
ASSIGN = 0.5          # coverage at or above this makes a held label the pole's assignment
STRONG = 0.3          # strong activation: this fraction of the pole's corpus max (the word screen's threshold)
ROUND_N = 16
N = 12288
WORDISH_KINDS = re.compile(r"^(lexical|token|word|spelling|token ?/ ?(lexical|spelling|subword)|spelling unit|token detector)$", re.I)
# a label that names its trigger: "the word 'X'", "the token ' X'", "the letters 'ur'", "the word piece 'exec'" ...
WORDISH_LABEL = re.compile(r"^\s*(the |a |an )?(english |german |french |spanish |japanese |arabic |russian |chinese |latin |exact |bare |"
                           r"capital )*(word|words|token|tokens|letters|letter string|word piece|sub-?word( piece)?|piece|chunk|stem|kanji|"
                           r"character|name)\s+(piece\s+|token\s+|itself\s+)?['\"‘“]", re.I)
TERM = re.compile(r"^[^\W_]([\w'’.-]{0,23}[^\W_])?$")  # a clean word-like term: letters or digits inside, no spaces
QUOTE = re.compile(r"'([^']{1,30})'|\"([^\"]{1,30})\"|‘([^’]{1,30})’|“([^”]{1,30})”")


def legacy_trigger(label, kind):
    """A trigger read from the quoted terms of a word-shaped label, or None when the label is not word-shaped."""
    if not label or not (WORDISH_KINDS.search(kind or "") or WORDISH_LABEL.match(label)):
        return None
    terms = {next(g for g in m.groups() if g is not None).strip().strip("-") for m in QUOTE.finditer(label)}
    words = sorted(t for t in terms if TERM.match(t))
    return {"words": words, "family": True} if words else None


@lru_cache(None)
def _vocab():
    from bonsai_interp.auto import context_tier as ct
    vb = ct.vocab()
    return vb["text"], [t.strip().lower() for t in vb["text"]]


def trigger_tokens(trigger):
    """Token ids a trigger covers."""
    from bonsai_interp import explore
    text, low = _vocab()
    tok = explore.tokenizer()
    want = set()
    words = {w.strip().lower() for w in trigger.get("words", []) if w.strip()}
    pieces = {p.strip().lower() for p in trigger.get("pieces", []) if p.strip()}
    exact = set(trigger.get("tokens", []))
    for w in list(words):
        for form in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            for t in tok(form, add_special_tokens=False).input_ids:
                if len(low[t]) >= 3:
                    want.add(t)
    stems = tuple(w for w in words if len(w) >= 4) if trigger.get("family") else ()
    for i, (t, lw) in enumerate(zip(text, low)):
        if lw and (lw in words or lw in pieces or (stems and lw.startswith(stems))) or t in exact:
            want.add(i)
    return want


@lru_cache(4)
def _hits(L):
    """(token ids, counts) per pole index (p * N + n) for layer L's strong activations, and the pole totals."""
    p = HITS / f"bonsai1_L{L}.hits.npz"
    if not p.exists():
        return None
    d = np.load(p)
    NN = 2 * N
    k, c = d["uni_keys"], d["uni_counts"]
    pole = k % NN
    o = np.argsort(pole, kind="stable")
    starts = np.searchsorted(pole[o], np.arange(NN + 1))
    return {"tok": (k[o] // NN).astype(np.int64), "cnt": c[o].astype(np.int64), "starts": starts, "total": d["total"].astype(np.int64)}


def token_coverage(nid, trigger):
    """(coverage, strong count) for a pole and a trigger, or None when the pole has no token table (dense or ineligible)."""
    m = re.match(r"^bonsai1/L(\d+)/N(\d+)([+-])$", nid)
    L, n, p = int(m[1]), int(m[2]), 0 if m[3] == "+" else 1
    h = _hits(L)
    if h is None:
        return None
    i = p * N + n
    total = int(h["total"][i])
    if total < 20:  # too few strong activations to say what share a trigger covers
        return None
    a, b = h["starts"][i], h["starts"][i + 1]
    want = trigger_tokens(trigger)
    got = int(h["cnt"][a:b][np.isin(h["tok"][a:b], list(want))].sum()) if want else 0
    return got / total, total


def _label_source(contribs):
    """(label, contribution id, declared trigger, unit kind) of the label a pole's record carries: the latest interpret or
    refine whose own test held."""
    from bonsai_interp.graph.protocols import outcome
    best, kind = None, None
    for c in contribs:
        if c.get("unit_kind"):
            kind = c["unit_kind"]
        if c["kind"] in ("interpret", "qc") and c.get("label"):
            res = [outcome(t) for t in c.get("tests", []) if t.get("counts_for_status", True)]
            res = [r for r in res if r in ("held", "failed")]
            if res and res[-1] == "held":
                best = (c["label"], c["id"], c.get("trigger"), kind)
    return best


def build(a):
    from bonsai_interp.graph import contrib
    by = {}
    for _, c in contrib.load_all():
        by.setdefault(c["neuron"], []).append(c)
    poles, skipped = {}, {"no trigger": 0, "no table": 0}
    for nid, cs in sorted(by.items()):
        if not nid.startswith("bonsai1/"):
            continue
        cs.sort(key=lambda c: (c["created"], c["id"]))
        src = _label_source(cs)
        if not src:
            continue
        label, cid, declared, kind = src
        trig, method = (declared, "declared trigger") if declared else (legacy_trigger(label, kind), "quoted terms of a word-shaped label")
        if not trig:
            skipped["no trigger"] += 1
            continue
        got = token_coverage(nid, trig)
        if got is None:
            skipped["no table"] += 1
            continue
        poles[nid] = {"label_contribution": cid, "coverage": round(got[0], 4), "strong": got[1], "method": method, "trigger": trig}
    tables = sorted(HITS.glob("bonsai1_L*.hits.npz"))
    out = {"about": "token coverage of held labels: the share of each pole's strong activations (>= 0.3 of max, auto-token@2 "
                    "selection windows) on its trigger's tokens; see bonsai_interp/graph/coverage.py",
           "assignment_threshold": ASSIGN, "tables": [p.name for p in tables],
           "tables_sha256": hashlib.sha256(b"".join(hashlib.sha256(p.read_bytes()).digest() for p in tables)).hexdigest(),
           "poles": poles}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    cov = np.array([v["coverage"] for v in poles.values()])
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}: {len(poles)} labels measured, {int((cov >= ASSIGN).sum())} at or above "
          f"{ASSIGN:.0%} (assignment), {int((cov < ASSIGN).sum())} below (partial); not measured: {skipped}")


@lru_cache(None)
def measured():
    """{pole: entry} from graph/auto/bonsai1.coverage.json ({} when absent)."""
    return json.loads(OUT.read_text(encoding="utf-8"))["poles"] if OUT.exists() else {}


# ---------------------------------------------------------------- the coverage round (coverage-16@1)

def draw(nid, label_contribution, k=ROUND_N, windows=1500, batch=4):
    """k of the pole's strong activations, drawn at random from held-out windows (16,400 on), one per window, seeded by the
    pole and the label's contribution id: [{window, position, act, text}] with the peak token marked ⟦ ⟧. GPU."""
    import torch
    from bonsai_interp import explore
    from bonsai_interp.auto import context_tier as ct
    from bonsai_interp.env import DEVICE, limit_memory
    m = re.match(r"^bonsai1/L(\d+)/N(\d+)([+-])$", nid)
    L, n, sign = int(m[1]), int(m[2]), 1.0 if m[3] == "+" else -1.0
    pmax = ct._pole_max(L)[0 if sign > 0 else 1][n]
    toks = explore.corpus()
    seed = int(hashlib.sha256(f"{nid}|{label_contribution}".encode()).hexdigest()[:12], 16)
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.arange(16400, len(toks)))[:windows]
    found = []
    limit_memory(0.8)  # BONSAI_GPU_FRACTION lowers it when the stream shares the card
    # every forward pass takes a background gpu_turn tick (context_tier._gpu_load), so the stream keeps priority
    model, tok, rec = ct._gpu_load()
    for b in range(0, len(order), batch):
        ws = order[b:b + batch]
        x = torch.from_numpy(toks[ws].astype(np.int64)).to(DEVICE)
        a = ct._forward(model, rec, x, L)[:, :, n].float() * sign / pmax  # (B, T)
        a[:, :64] = -1
        for j, pos in zip(*np.nonzero(a.cpu().numpy() >= STRONG)):
            found.append((int(ws[j]), int(pos), float(a[j, pos])))
        if len({w for w, _, _ in found}) >= 3 * k:
            break
    ct._gpu_idle("coverage draw")
    if not found:
        return []
    by_w = {}
    for f in found:
        by_w.setdefault(f[0], []).append(f)
    wins = sorted(by_w)
    pick = [by_w[wins[i]][rng.integers(len(by_w[wins[i]]))] for i in sorted(rng.choice(len(wins), min(k, len(wins)), replace=False))]
    out = []
    for w, p, act in pick:  # one activation per window, so a single long passage cannot fill the sample
        pre, peak, post = tok.decode(toks[w, max(0, p - 40):p]), tok.decode(toks[w, p:p + 1]), tok.decode(toks[w, p + 1:p + 9])
        out.append({"window": w, "position": p, "act": round(act, 3), "text": f"{pre}⟦{peak}⟧{post}"})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="token coverage of every held label it can measure -> graph/auto/bonsai1.coverage.json")
    a = ap.parse_args()
    if a.cmd == "build":
        if not list(HITS.glob("bonsai1_L*.hits.npz")):
            sys.exit("no token tables: gh release download auto2-data --repo moridinamael/bonsai-interp --dir data/auto2 --pattern 'bonsai1_L*.hits.npz'")
        build(a)


if __name__ == "__main__":
    main()
