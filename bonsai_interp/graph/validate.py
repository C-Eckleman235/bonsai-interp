"""Validate the graph (no model or torch needed).

  python -m bonsai_interp.graph.validate             contributions only (what a PR is checked against)
  python -m bonsai_interp.graph.validate --strict    also: derived files on disk match the fold exactly (main)

Checks every contribution and pre-registration file against the rules in graph/SCHEMA.md and
DATA_POLICY.md, then checks that the committed neuron records, edges and index are exactly what
`fold` produces from the contributions (so no record was edited by hand).
"""
import json
import sys

from bonsai_interp.graph import contrib, fold

ROOT = contrib.ROOT


def main():
    errs = []
    results = contrib.load_all()
    ids = {}
    for path, c in results:
        errs += contrib.check(path, c)
        key = (path.parent.name, c.get("id"))  # ids are unique per pole folder; every reference to one also names its pole
        if key in ids:
            errs.append(f"duplicate contribution id {c['id']} in {path.parent.name} ({ids[key]} and {path.relative_to(ROOT).as_posix()})")
        ids[key] = path.relative_to(ROOT).as_posix()
    tests = sorted(contrib.C.glob("*/*/*.tests.json"))
    for p in tests:
        errs += contrib.check_tests_file(p)
    stray = [p for p in contrib.C.rglob("*") if p.is_file() and p.suffix not in (".json", ".md")]
    errs += [f"{p.relative_to(ROOT).as_posix()}: only .json and .md files are allowed under graph/contributions" for p in stray]
    files = fold.fold()
    if "--strict" in sys.argv:  # on main, after the merge agent's fold commit: derived files must match exactly
        for p, s in files.items():
            if not (ROOT / p).exists() or (ROOT / p).read_text(encoding="utf-8") != s:
                errs.append(f"{p}: out of date with the contributions (run python -m bonsai_interp.graph.fold and commit)")
        for p in (ROOT / "graph" / "neurons").glob("*/*.json"):
            if p.relative_to(ROOT).as_posix() not in files:
                errs.append(f"{p.relative_to(ROOT).as_posix()}: record with no contributions (records are derived; do not add them by hand)")
    neurons = set(json.loads(files["graph/index.json"])["neurons"])
    for e in json.loads(files["graph/edges.json"])["edges"]:
        for end in ("from", "to"):
            if e[end] not in neurons:
                errs.append(f"edge {e['from']} -> {e['to']} ({e['contribution']}): {e[end]} has no contributions of its own")
    for m in errs:
        print("ERROR", m)
    print(f"{len(results)} contributions, {len(tests)} pre-registration files, {len(neurons)} neuron poles; {len(errs)} problems")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
