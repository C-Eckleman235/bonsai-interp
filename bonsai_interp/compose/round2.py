"""compose@2: the second compositional round. Layers 12-17, composed from everything round 1 and the twins characterized.

  python -m bonsai_interp.compose.round2 protocol           # write the manifest; commit it ALONE
  python -m bonsai_interp.compose.round2 select --layer L   # CPU, one target layer per process (run several at once)
  python -m bonsai_interp.compose.round2 merge              # the per-layer hypotheses -> one selection file (best cap)
  python -m bonsai_interp.compose.round2 build              # CPU: held-out tests; commit the prereg manifest ALONE
  python -m bonsai_interp.compose.round2 score              # GPU
  (--smoke DIR: no manifests, a few hundred targets; files in DIR)

The engine is compose@1's (round.py), imported unchanged: the same hypotheses, tests, judge and coverage
(docs/COMPOSE.md). Round 2 differs in four ways:
  1. Targets are the residual eligible poles of layers 13-17. Layer 12 was held out of the word screen by design, so it
     has no token table: it can be neither a target (its own strong tokens cannot be measured) nor a source of units.
  2. Units are everything characterized after round 1. The atlas now marks a pole held when any deterministic pass
     held for it: the word screen, sweeplib@1, the twin batteries (sweeplib@2), grammar (sweeplib@3) or compose@1.
     A held test with measured coverage >= 50% (atlas lib_coverage) counts as assignment-grade, like a survey label.
  3. Upstream tables are filtered to characterized units BEFORE matching. compose@1 matched every unit and dropped the
     uncharacterized ones afterwards, so its per-layer top candidates were partly unusable. Filtering first is faster
     and keeps only usable candidates.
  4. Selection runs one target layer per process; `merge` keeps the best SEL['cap'] over all of them.
"""
import argparse
import datetime as dt
import gzip
import json
import sys
from pathlib import Path

import numpy as np

from bonsai_interp.compose import match as M
from bonsai_interp.compose import pilot as PI
from bonsai_interp.compose import round as R

PROTOCOL = "compose@2"
TARGETS = [L for L in range(12, 18) if L != 12]  # L12 has no word-screen token table (held out by design)
BASE = "bonsai1.compose2"


def _paths(d=None):
    o, m = (Path(d), Path(d)) if d else (R.ROOT / "data" / "compose", R.ROOT / "graph" / "auto")
    return {"protocol": m / f"{BASE}.protocol.json", "prereg": m / f"{BASE}.prereg.json", "summary": m / f"{BASE}.summary.json",
            "select": o / f"{BASE}.select.json.gz", "tests": o / f"{BASE}.tests.jsonl.gz", "results": o / f"{BASE}.results.jsonl.gz"}


def characterized():
    """Round 1's characterized set (pilot.characterized), plus atlas lib_test 'held' with lib_coverage >= 0.5 as
    assignment-grade."""
    from bonsai_interp import atlas
    out = PI.characterized()
    for L in range(36):
        t = atlas.load(L)
        if t is None or "lib_coverage" not in t.column_names:
            continue
        d = t.select(["neuron", "sign", "lib_test", "lib_coverage"]).to_pydict()
        for n, s, lt, lc in zip(d["neuron"], d["sign"], d["lib_test"], d["lib_coverage"]):
            if lt == "held":
                out[(L, n, s)] = "assigned" if (lc or 0) >= 0.5 else out.get((L, n, s), "held")
    return out


def _filtered_upstream(char):
    """compose@1's _upstream with each layer's table cut to characterized units."""
    known = {}
    for (l, n, s) in char:
        known.setdefault(l, []).append(n + (0 if s == "+" else R.N))

    def upstream(L):
        out = {}
        for l in range(L):
            t = M._table(l)
            if t is None or l not in known:
                continue
            keys, cnt, total = t
            keep = (cnt >= R.SEL["min_count"]) & np.isin(keys % R.NN, np.array(known[l]))
            out[l] = (keys[keep], cnt[keep], total)
        return out
    return upstream


def _install(a):
    """Point compose@1's engine at round 2: protocol, targets, paths, code list, characterized set, upstream tables."""
    char = characterized()
    R.PROTOCOL, R.TARGETS = PROTOCOL, ([a.layer] if getattr(a, "layer", None) is not None else TARGETS)
    R.CODE = [Path(R.__file__), Path(M.__file__), Path(__file__)]
    paths = _paths
    if getattr(a, "layer", None) is not None and a.cmd == "select":
        def paths(d=None, _p=_paths):  # one target layer per process: its own selection file
            P = _p(d)
            P["select"] = P["select"].with_name(f"{BASE}.select.L{a.layer}.json.gz")
            return P
    R._paths = paths
    PI.characterized = lambda: char
    R._upstream = _filtered_upstream(char)


def merge(a):
    P = _paths(a.smoke)
    if not a.smoke:
        R._paths = _paths
        R._setup(a)  # the protocol guard
    hyps, n_all, minutes = [], 0, 0.0
    for L in TARGETS:
        p = P["select"].with_name(f"{BASE}.select.L{L}.json.gz")
        if not p.exists():
            sys.exit(f"missing {p.name}: run `select --layer {L}` first")
        s = R.ct._read_gz(p)[0]
        hyps += s["hypotheses"]
        n_all += s["n_all"]
        minutes = max(minutes, s["minutes"])
    hyps.sort(key=lambda h: (-h["score"], h["target"]))
    kept = hyps[:R.SEL["cap"]]
    R.ct._write_gz(P["select"], [{"hypotheses": kept, "n_all": n_all, "minutes": minutes}])
    print(f"wrote {P['select']}: {n_all} hypotheses over layers {TARGETS} "
          f"({sum(h['kind'] == 'pair' for h in hyps)} pair, {sum(h['kind'] == 'class' for h in hyps)} class in the per-layer bests); "
          f"kept the best {len(kept)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["protocol", "select", "merge", "build", "score"])
    ap.add_argument("--layer", type=int, help="select: the one target layer this process runs")
    ap.add_argument("--smoke", help="a scratch directory: no manifests, a few hundred targets")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    if a.cmd == "select" and a.layer not in TARGETS:
        sys.exit(f"select needs --layer, one of {TARGETS}")
    _install(a)
    if a.cmd == "merge":
        return merge(a)
    if a.cmd == "protocol":
        man_extra = {"targets": TARGETS, "units": "characterized after round 1 (survey, word screen, sweeplib@1-3, compose@1)",
                     "upstream": "tables filtered to characterized units before matching", "about2": __doc__}
        R.protocol(a)
        p = _paths()["protocol"]
        man = json.loads(p.read_text(encoding="utf-8"))
        man.update(man_extra)
        p.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        return
    {"select": R.select, "build": R.build, "score": R.score}[a.cmd](a)


if __name__ == "__main__":
    main()
