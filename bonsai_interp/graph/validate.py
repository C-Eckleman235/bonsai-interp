"""Validate every graph record and edge (no model or torch needed).

  python -m bonsai_interp.graph.validate        exit 1 on any problem

The rules are described in graph/SCHEMA.md.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
G = ROOT / "graph"
STATUS = {"unlabelled", "held", "failed", "unresolved", "dense", "probed"}
DEPTH = {"quick", "full", "blind replication only"}
POLE = {"+", "-"}
ID = re.compile(r"^(?P<model>[a-z0-9_]+)/L(?P<L>\d+)/N(?P<n>\d+)(?P<pole>[+-])$")
REQUIRED = {"id": str, "model": str, "layer": int, "index": int, "pole": str, "status": str, "depth": str,
            "tests": list, "label_history": list, "examples": list, "evidence": list, "roles": list, "contributors": list}
LIMITS = {"bonsai1": (36, 12288)}


def check_record(path, r, errs):
    say = lambda m: errs.append(f"{path.relative_to(ROOT)}: {m}")
    for k, t in REQUIRED.items():
        if not isinstance(r.get(k), t):
            say(f"'{k}' missing or not {t.__name__}")
    if errs and errs[-1].startswith(str(path.relative_to(ROOT))):
        return
    m = ID.match(r["id"])
    if not m:
        return say(f"bad id {r['id']!r} (want model/L<layer>/N<index><+|->)")
    if (m["model"], int(m["L"]), int(m["n"]), m["pole"]) != (r["model"], r["layer"], r["index"], r["pole"]):
        say("id does not match model/layer/index/pole")
    want = f"L{r['layer']}_N{r['index']}_{'pos' if r['pole'] == '+' else 'neg'}.json"
    if path.name != want or path.parent.name != r["model"]:
        say(f"file should be graph/neurons/{r['model']}/{want}")
    if r["pole"] not in POLE:
        say("pole must be + or -")
    if r["model"] in LIMITS:
        nl, nn = LIMITS[r["model"]]
        if not (0 <= r["layer"] < nl and 0 <= r["index"] < nn):
            say("layer/index out of range")
    if r["status"] not in STATUS:
        say(f"status must be one of {sorted(STATUS)}")
    if r["depth"] not in DEPTH:
        say(f"depth must be one of {sorted(DEPTH)}")
    if r["status"] in ("held", "failed", "unresolved") and not r["tests"]:
        say(f"status '{r['status']}' needs at least one test")
    if r["status"] != "unlabelled" and not r.get("label"):
        say("a labelled status needs a 'label'")
    for i, t in enumerate(r["tests"]):
        for k in ("type", "author", "tests_file", "result_file", "prereg_commit"):
            if not t.get(k):
                say(f"tests[{i}] needs '{k}'")
        for k in ("tests_file", "result_file"):
            if t.get(k) and not (ROOT / t[k]).exists():
                say(f"tests[{i}].{k} does not exist: {t[k]}")
        if t.get("type") == "quick-8+8":
            want_ok = t["n_fire"] >= 5 and t["n_false_pos"] <= 1
            if bool(t.get("survives")) != want_ok:
                say(f"tests[{i}] 'survives' disagrees with the 5-of-8 / at-most-1 rule")
    if r["status"] == "held":
        latest = [t for t in r["tests"] if t.get("type") == "quick-8+8" and t.get("author") != "bonsai"]
        if latest and not latest[-1].get("survives"):
            say("status 'held' but the latest quick test failed")


def main():
    errs, ids = [], set()
    files = sorted((G / "neurons").glob("*/*.json"))
    for f in files:
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            errs.append(f"{f.relative_to(ROOT)}: invalid JSON ({e})")
            continue
        check_record(f, r, errs)
        if r.get("id") in ids:
            errs.append(f"{f.relative_to(ROOT)}: duplicate id {r['id']}")
        ids.add(r.get("id"))
    edges = json.loads((G / "edges.json").read_text(encoding="utf-8"))["edges"]
    for i, e in enumerate(edges):
        for k in ("from", "to", "kind", "weight", "method", "source"):
            if k not in e:
                errs.append(f"graph/edges.json: edge {i} needs '{k}'")
        for end in ("from", "to"):
            if e.get(end) and e[end] not in ids:
                errs.append(f"graph/edges.json: edge {i} {end} {e[end]!r} has no neuron record")
    for m in errs:
        print("ERROR", m)
    print(f"{len(files)} records, {len(edges)} edges, {len(errs)} problems")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
