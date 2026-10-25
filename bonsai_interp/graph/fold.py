"""Fold contributions into neuron records, edges and the site index. Deterministic and idempotent:
the same contributions always give byte-identical output, so CI can check that nobody hand-edited
a record (`--check`), and merging the same contribution twice changes nothing.

  python -m bonsai_interp.graph.fold            rewrite graph/neurons/, graph/edges.json, graph/index.json
  python -m bonsai_interp.graph.fold --check    exit 1 if the committed files differ from the fold

Rules (per neuron pole, contributions in (created, id) order):
  - interpret / qc-refine: the latest test that counts for status decides the round's outcome.
    A held round sets the label. While no label has held, the latest attempt is shown, and three
    failed rounds make the pole 'unresolved'. A failed round after a held label is kept in the
    history but does not replace it (use a 'dispute' QC to challenge a held label).
  - qc confirm/dispute: counted against the current label; more disputes than confirmations since
    the label last changed makes a held pole 'contested'.
  - replicate: appended to 'replications'; never changes the label.
  - unit_kind containing 'dense' makes the pole 'dense'; depth 'full' makes it 'probed'.
  - a held label is the pole's assignment only if it covers at least half of the pole's strong activations
    (bonsai_interp/graph/coverage.py); a held label measured below that is 'partial'. Coverage comes from a declared
    trigger's token measure, else a 'coverage' round on that label, else the token estimate from the label's quoted words.
  - a 'withhold' contribution (sensitive trigger) makes the pole 'withheld', above every other status; its masked
    label and short replace any other, so the content never reaches the index, the map or the stream.
  - list fields (roles, evidence, examples) are unions in order; scalar extras: latest wins.
"""
import argparse
import json
import sys
from collections import defaultdict

from bonsai_interp.graph.contrib import NEURON, ROOT, load_all
from bonsai_interp.graph.protocols import outcome
from bonsai_interp.graph.coverage import ASSIGN, measured

G = ROOT / "graph"
SCALARS = ("short", "verdict", "summary", "offshoots", "depth", "lineage")


def _union(a, b):
    return a + [x for x in (b or []) if x not in a]


LEVELS = ["unexamined", "tested", "re-tested", "checked", "investigated", "replicated"]


def scrutiny(r, contribs):
    """How much evidence a pole has, separate from what the evidence says (status). Derived, never claimed.

    1 tested        one pre-registered test
    2 re-tested     two or more test rounds, or tests written by a second author
    3 checked       a QC by a contributor other than the label's author, or a held-out test
    4 investigated  a full probe, or a blind clean-room investigation
    5 replicated    a full probe plus an independent blind replication
    """
    tests = [t for t in r["tests"] if t.get("protocol") != "cleanroom@1"]
    rounds = len({t["contribution"] for t in tests})
    authors = {t.get("author") or "?" for t in tests}
    first = next((c["contributor"] for c in contribs if c["kind"] == "interpret"), None)
    qc_other = any(c["kind"] == "qc" and c["contributor"] != first for c in contribs)
    held_out = any(t.get("protocol") == "held-out@1" for t in tests)
    full, blind = r.get("depth") == "full", bool(r["replications"])
    level = (5 if full and blind else 4 if full or blind else 3 if qc_other or held_out
             else 2 if rounds >= 2 or len(authors) >= 2 else 1 if tests else 0)
    return {"level": level, "name": LEVELS[level], "test_rounds": rounds,
            "sentences": sum(sum(t.get("n_written") or [0]) for t in tests),
            "contributors": len(r["contributors"]), "qc": sum(r["qc"].values())}


def fold_neuron(neuron, contribs):
    m = NEURON.match(neuron)
    r = {"id": neuron, "model": m["model"], "layer": int(m["L"]), "index": int(m["n"]), "pole": m["pole"],
         "label": None, "short": None, "kind": None, "status": "unlabelled", "depth": "quick",
         "tests": [], "label_history": [], "examples": [], "evidence": [], "roles": [], "contributors": [],
         "contributions": [], "qc": {"confirm": 0, "dispute": 0, "refine": 0}, "replications": []}
    held, failed_rounds, since, held_cid = False, 0, {"confirm": 0, "dispute": 0}, None
    for c in contribs:
        r["contributions"].append(c["id"])
        r["contributors"] = _union(r["contributors"], [c["contributor"]])
        for k in ("roles", "evidence", "examples"):
            r[k] = _union(r[k], c.get(k))
        for k in SCALARS:
            if c.get(k) is not None:
                r[k] = c[k]
        if c.get("unit_kind") is not None:
            r["kind"] = c["unit_kind"]
        tests = [{**t, "contribution": c["id"], "outcome": outcome(t)} for t in c.get("tests", [])]
        r["tests"] += tests
        counted = [t for t in tests if t.get("counts_for_status", True) and t["outcome"] in ("held", "failed")]
        res = counted[-1]["outcome"] if counted else None
        kind, verdict = c["kind"], (c.get("qc") or {}).get("verdict")
        if kind == "coverage":  # a coverage round on the current label: read below, after the label is settled
            continue
        if kind == "withhold":
            r["withheld"] = {"reason": c["withheld"]["reason"], "label": c["label"], "short": c.get("short"), "contribution": c["id"]}
            r["label_history"].append([c["id"], c["label"], "withheld"])
            continue
        if kind == "replicate":
            r["replications"].append({"contribution": c["id"], **{k: c[k] for k in ("label", "verdict", "confidence", "notes") if k in c},
                                      "outcome": tests[-1]["outcome"] if tests else "recorded"})
            continue
        if kind == "qc" and verdict in ("confirm", "dispute"):
            if res:  # the test decides, not the verdict: a confirm whose own test failed counts as a dispute, and vice versa
                got = "confirm" if res == "held" else "dispute"
                r["qc"][got] += 1
                since[got] += 1
            continue  # an exploratory or unscored QC is recorded but confirms or disputes nothing
        if kind == "qc":
            r["qc"]["refine"] += 1
        if kind in ("interpret", "qc") and c.get("label"):
            r["label_history"].append([c["id"], c["label"], res or "recorded"])
            if res == "held":
                r["label"], held, failed_rounds, since, held_cid = c["label"], True, 0, {"confirm": 0, "dispute": 0}, c["id"]
            elif not held:
                r["label"] = c["label"]
                failed_rounds += res == "failed"
    r["coverage"] = _coverage(neuron, held_cid, contribs) if held else None
    r["scrutiny"] = scrutiny(r, contribs)
    if r.get("withheld"):
        w = r["withheld"]
        r["status"], r["label"], r["short"] = "withheld", w["label"], w["short"]
        r["withheld"] = {"reason": w["reason"], "contribution": w["contribution"]}
    elif r["kind"] and "dense" in r["kind"]:
        r["status"] = "dense"
    elif r["depth"] == "full":
        r["status"] = "probed"
    elif held:
        # held means the trigger reliably fires the pole; it is the pole's assignment only if it also covers at least half of the
        # pole's strong activations (coverage.ASSIGN). Below that the label is 'partial': information, not assignment.
        partial = r["coverage"] is not None and r["coverage"]["value"] < ASSIGN
        r["status"] = "contested" if since["dispute"] > since["confirm"] else "partial" if partial else "held"
    elif failed_rounds >= 3:
        r["status"] = "unresolved"
    elif failed_rounds:
        r["status"] = "failed"
    elif r["replications"] and not r["label_history"]:
        r["status"], r["label"] = "replicated", r["replications"][-1].get("label")
    elif r["label"]:
        r["status"] = "probed"  # labelled, but only exploratory or unscored tests so far
    return r


def _coverage(neuron, held_cid, contribs):
    """The current held label's coverage: a token measure from a declared trigger, else the latest coverage round on this
    label, else the token estimate from the label's quoted words (graph/auto/bonsai1.coverage.json). None if unmeasured."""
    tok = measured().get(neuron)
    tok = tok if tok and tok["label_contribution"] == held_cid else None
    rounds = [c for c in contribs if c["kind"] == "coverage" and c.get("label_contribution") == held_cid]
    if tok and tok["method"] == "declared trigger":
        return {"value": tok["coverage"], "method": "declared trigger (token measure)", "strong": tok["strong"]}
    if rounds:
        c = rounds[-1]
        return {"value": c["coverage"], "method": f"coverage round ({sum(c['marks'])}/{len(c['marks'])} described)", "contribution": c["id"]}
    if tok:
        return {"value": tok["coverage"], "method": "estimated from the label's quoted words (token measure)", "strong": tok["strong"]}
    return None


def _history(r):
    """The last six labels tried; on a withheld pole every earlier label is masked too (they may name the content)."""
    h = [x[1:] for x in r["label_history"]][-6:]
    if r.get("withheld"):
        from bonsai_interp.sensitive import mask
        h = [[mask(lab), out] for lab, out in h]
    return h


def fold():
    by_neuron, edges = defaultdict(list), {}
    for _, c in load_all():
        by_neuron[c["neuron"]].append(c)
    files = {}
    records = {}
    for neuron, cs in by_neuron.items():
        cs.sort(key=lambda c: (c["created"], c["id"]))
        r = fold_neuron(neuron, cs)
        records[neuron] = r
        name = f"{r['model']}/L{r['layer']}_N{r['index']}_{'pos' if r['pole'] == '+' else 'neg'}.json"
        files[f"graph/neurons/{name}"] = json.dumps(r, ensure_ascii=False, indent=1) + "\n"
        for c in cs:
            for e in c.get("edges", []):
                key = (e["from"], e["to"], e["kind"], e.get("method", ""))
                edges[key] = {**e, "contribution": c["id"]}  # latest measurement of the same edge wins
    E = [edges[k] for k in sorted(edges)]
    files["graph/edges.json"] = json.dumps({"edges": E}, ensure_ascii=False, indent=1) + "\n"
    # compact index for the site / game: one line per pole, coverage, contributors
    points = defaultdict(int)
    for cs in by_neuron.values():
        for c in cs:
            points[c["contributor"]] += {"interpret": 10, "qc": 6, "replicate": 15, "map": 3, "withhold": 3, "coverage": 4}.get(c["kind"], 1)
    idx = {"neurons": {n: {k: r[k] for k in ("status", "short", "label", "kind", "depth")} | {"qc": r["qc"], "n_contrib": len(r["contributions"]),
                                                                                            "replicated": bool(r["replications"]),
                                                                                            "contributors": r["contributors"],
                                                                                            "t": [[t.get("protocol"), t["outcome"], t.get("n_fire"), t.get("n_false_pos"), t.get("author"), t["contribution"]]
                                                                                                  for t in r["tests"]],
                                                                                            "history": _history(r),
                                                                                            "scrutiny": r["scrutiny"]}
                       | ({"withheld": r["withheld"]["reason"]} if r.get("withheld") else {})
                       | ({"coverage": r["coverage"]["value"]} if r.get("coverage") else {})
                       for n, r in sorted(records.items())},
           "edges": [[e["from"], e["to"], e["kind"]] for e in E],
           "contributors": dict(sorted(points.items(), key=lambda t: (-t[1], t[0]))),
           "models": {"bonsai1": {"layers": 36, "neurons_per_layer": 12288}}}
    files["graph/index.json"] = json.dumps(idx, ensure_ascii=False, separators=(",", ":")) + "\n"
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    files = fold()
    if a.check:
        bad = [p for p, s in files.items() if not (ROOT / p).exists() or (ROOT / p).read_text(encoding="utf-8") != s]
        stale = [p.relative_to(ROOT).as_posix() for p in (G / "neurons").glob("*/*.json") if p.relative_to(ROOT).as_posix() not in files]
        for p in bad + stale:
            print("OUT OF DATE", p)
        print(f"{len(files)} folded files; {len(bad) + len(stale)} differ from the commit (run: python -m bonsai_interp.graph.fold)")
        sys.exit(1 if bad or stale else 0)
    for p in (G / "neurons").glob("*/*.json"):
        if p.relative_to(ROOT).as_posix() not in files:
            p.unlink()
    for p, s in files.items():
        (ROOT / p).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / p).write_text(s, encoding="utf-8", newline="\n")
    print(f"wrote {len(files)} files")


if __name__ == "__main__":
    main()
