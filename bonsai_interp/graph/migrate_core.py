"""One-off (2026-10-18): turn the core team's derived records into contributions, so every record
from now on is folded from the append-only contribution log. Kept for provenance.

  python -m bonsai_interp.graph.migrate_core

Each test round of a record becomes one 'interpret' contribution, dated by its pre-registration
commit. Bonsai-written tests are kept but do not count for status (as before). Full probes become
an interpret contribution with depth 'full'; the blind clean-room run becomes a 'replicate'
contribution; the neighbourhood edges become one 'map' contribution.
"""
import json
import subprocess
from functools import lru_cache

from bonsai_interp.graph.contrib import C, ROOT, folder_for

G = ROOT / "graph"


@lru_cache(None)
def commit_time(h):
    return subprocess.run(["git", "show", "-s", "--format=%cI", h], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def cid(rec, i):
    return f"core-{rec['model']}-l{rec['layer']}-n{rec['index']}-{'pos' if rec['pole'] == '+' else 'neg'}-{i:02d}"


def write(c):
    d = folder_for(c["neuron"])
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{c['id']}.json").write_text(json.dumps(c, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


def main():
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((G / "neurons").glob("*/*.json"))]
    n = 0
    for r in recs:
        base = {"neuron": r["id"], "contributor": "bonsai-interp-core"}
        rounds = []
        for t in r["tests"]:
            t = dict(t)
            t["protocol"] = {"quick-8+8": "quick-8+8@1", "held-out": "held-out@1"}[t.pop("type")]
            if t.get("author") == "bonsai":
                t["counts_for_status"] = False
            label = t.pop("label_tested", r["label"])
            rounds.append((commit_time(t["prereg_commit"]), label, t))
        out = []
        for i, (when, label, t) in enumerate(rounds):
            out.append({"id": cid(r, i), "kind": "interpret", **base, "created": when, "label": label, "tests": [t]})
        if r.get("depth") == "full" and out:
            out[-1].update({"label": r["label"], "depth": "full", "lineage": r.get("label_history"),
                            **{k: r[k] for k in ("verdict", "summary", "offshoots") if r.get(k)}})
        if out:
            last = out[-1]
            last.update({k: r[k] for k in ("short", "examples", "evidence", "roles") if r.get(k)})
            if r.get("kind"):
                last["unit_kind"] = r["kind"]
        for rep in r.get("replications", []) + ([{"label": r["label"], "verdict": r.get("verdict"), "path": r["evidence"][0]}]
                                                 if r.get("depth") == "blind replication only" else []):
            out.append({"id": cid(r, len(out)), "kind": "replicate", **base, "contributor": "neuron-cleanroom-blind-agent",
                        "created": commit_time("5376866"), "label": rep["label"], "verdict": rep.get("verdict"),
                        "confidence": rep.get("confidence"), "evidence": ["runs/cleanroom/blind/REPORT.md"],
                        **({"short": r["short"], "depth": r["depth"], "unit_kind": r.get("kind")} if r.get("depth") == "blind replication only" else {}),
                        "tests": [{"protocol": "cleanroom@1", "tests_file": "runs/cleanroom/predictions.json",
                                   "prereg_commit": "2dae2f0",
                                   "outcome": "partly agrees" if r["pole"] == "+" else "no prior label",
                                   "result_file": "runs/cleanroom/scored.json"}]})
        for c in out:
            write(c)
            n += 1
    edges = json.loads((G / "edges.json").read_text(encoding="utf-8"))["edges"]
    write({"id": "core-bonsai1-l9-n3688-pos-edges", "kind": "map", "neuron": "bonsai1/L9/N3688+",
           "contributor": "bonsai-interp-core", "created": commit_time("3240cdd"),
           "notes": "Neighbourhood web around L9/N3688 (feeds, reads, gates, co-fires, feeds lateral); see probes/bonsai1_L9_N3688_pos.",
           "edges": [{k: e[k] for k in ("from", "to", "kind", "weight", "method", "source")} for e in edges]})
    print(f"wrote {n + 1} contributions under {C.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
