"""Export the probe results into graph records (graph/neurons/<model>/L{L}_N{n}_{pos|neg}.json, graph/edges.json).

  python -m bonsai_interp.graph.export

Sources: the full probes under probes/ (verdict.json, solo.json), the neighbourhood web around
L9/N3688 (via bonsai_interp.probe.web_page.collect), and the blind clean-room run under runs/cleanroom.
Pre-registration commits are looked up in git: the first commit that added each test file.
"""
import json
import subprocess
from functools import lru_cache

from bonsai_interp.model import ROOT
from bonsai_interp.probe import web_page
from bonsai_interp.probe.workspace import Workspace

G = ROOT / "graph"


@lru_cache(None)
def first_commit(path):
    out = subprocess.run(["git", "log", "--diff-filter=A", "--format=%h", "--", str(path)], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    return out[-1] if out else None


def rid(model, L, n, pole):
    return f"{model}/L{L}/N{n}{pole}"


def record(model, L, n, pole, **kw):
    r = {"id": rid(model, L, n, pole), "model": model, "layer": L, "index": n, "pole": pole,
         "label": None, "short": None, "kind": None, "status": "unlabelled", "depth": "quick",
         "tests": [], "label_history": [], "examples": [], "evidence": [], "roles": [], "contributors": ["bonsai-interp core"]}
    r.update(kw)
    return r


def quick_test(ws, unit, tests_files, result_file, author):
    """The latest quick test of `unit` among the given (tests file, result file) rounds."""
    for tf in tests_files:
        if not ws.has(tf) or not ws.has(result_file):
            continue
        spec = {nb["unit"]: nb for nb in ws.load(tf)["neighbours"]}
        res = {nb["unit"]: nb for nb in ws.load(result_file)["neighbours"]}
        if unit in spec and unit in res:
            r = res[unit]
            return {"type": "quick-8+8", "author": author, "tests_file": f"probes/{ws.dir.name}/{tf}",
                    "result_file": f"probes/{ws.dir.name}/{result_file}", "prereg_commit": first_commit(ws.path(tf)),
                    "label_tested": spec[unit]["label"], "n_fire": r["n_fire"], "n_false_pos": r["n_false_pos"],
                    "n_written": r.get("n_written", [len(spec[unit]["fire"]), len(spec[unit]["quiet"])]),
                    "survives": r["survives"]}
    return None


ROUNDS = [  # (tests files, result file, author), oldest first
    (["quick_tests.json"], "quick.json", "claude"),
    ([f"web_tests_claude_{b}.json" for b in range(3)], "web_quick_claude.json", "claude"),
    (["web_tests_bonsai.json"], "web_quick_bonsai.json", "bonsai"),
    (["relabel_tests_0.json", "relabel_tests_1.json"], "relabel_quick.json", "claude"),
    (["relabel2_tests.json"], "relabel2_quick.json", "claude"),
    (["lateral_out_tests.json"], "lateral_out_quick.json", "claude"),
]


def web_records(ws):
    data = web_page.collect(ws)
    out = {}
    for nd in data["nodes"]:
        L, n = nd["L"], nd["n"]
        tests = [t for tf, res, who in ROUNDS for t in [quick_test(ws, nd["unit"], tf, res, who)] if t]
        latest_claude = next((t for t in reversed(tests) if t["author"] == "claude"), None)
        kind = nd.get("kind")
        status = ("dense" if kind and "dense" in kind else
                  "held" if latest_claude and latest_claude["survives"] else
                  "unresolved" if nd["attempts"] >= 3 else "failed" if latest_claude else "unlabelled")
        out[(L, n, nd["pole"])] = record(
            "bonsai1", L, n, nd["pole"], label=nd["label"], short=nd["short"], kind=kind, status=status,
            roles=nd["roles"] + ([f"via {nd['via']}"] if nd.get("via") else []), tests=tests,
            label_history=[[t["author"] + " round", t["label_tested"]] for t in tests],
            examples=[e[-200:] for e in nd["examples"][:4]],
            evidence=[f"probes/{ws.dir.name}/web.html"])
    edges = [{"from": f"bonsai1/{e['from']}{byu.get(e['from'], '+')}", "to": f"bonsai1/{e['to']}{byu.get(e['to'], '+')}",
              "kind": e["kind"], "weight": round(float(e["w"]), 4),
              "method": "single-unit zero ablation" if e["kind"] in ("feeds", "feeds reader", "feeds lateral") else
                        "boost/zero of the source, change in the target" if e["kind"] == "reads" else
                        "pre-registered gate test" if e["kind"] == "gates" else "co-activation over the corpus",
              "source": f"probes/{ws.dir.name}"}
             for byu in [{nd["unit"]: nd["pole"] for nd in data["nodes"]} | {data["center"]: "+"}] for e in data["edges"]]
    return out, edges


def full_record(ws):
    v = ws.load("verdict.json") if ws.has("verdict.json") else {}
    bank, solo = ws.load("bank.json"), ws.load("solo.json")
    ho = solo.get("holdout")
    tests = []
    if ho:
        tests.append({"type": "held-out", "author": "claude", "tests_file": f"probes/{ws.dir.name}/holdout.json",
                      "result_file": f"probes/{ws.dir.name}/solo.json", "prereg_commit": first_commit(ws.path("holdout.json")),
                      "label_tested": bank.get("hypotheses", {}).get(ho["hypothesis"], bank["label"][ws.pole]),
                      "accuracy": round(ho["accuracy"], 3), "auroc": round(ho["auroc"], 3),
                      "fire_rate_predicted_fire": round(ho["fire_rate_predicted_fire"], 3),
                      "fire_rate_predicted_quiet": round(ho["fire_rate_predicted_quiet"], 3)})
    label = v.get("label_history", [[None, bank["label"][ws.pole]]])[-1][1]
    fwd = ws.load("forward.json")
    return record("bonsai1", ws.L, ws.n, ws.pole, label=label, short=v.get("title"), kind=None,
                  status="probed", depth="full", tests=tests, label_history=v.get("label_history", []),
                  verdict=v.get("verdict"), summary=v.get("summary"), offshoots=v.get("offshoots"),
                  examples=[e["text"][-200:] for e in fwd["examples"][ws.pole][:4]],
                  evidence=[f"probes/{ws.dir.name}/dossier.html", f"probes/{ws.dir.name}/"])


def main():
    recs = {}
    nerve = Workspace(9, 3688, "+")
    web, edges = web_records(nerve)
    recs.update(web)
    for L, n in ((9, 3688), (10, 5749), (22, 11024)):
        ws = Workspace(L, n, "+")
        recs[(L, n, "+")] = {**recs.get((L, n, "+"), {}), **full_record(ws)}
    # the blind clean-room replication of L9/N3688, both poles
    blind = ROOT / "runs" / "cleanroom" / "blind" / "summary.json"
    if blind.exists():
        b = json.loads(blind.read_text(encoding="utf-8"))
        recs[(9, 3688, "+")]["replications"] = [{"type": "blind clean room", "path": "runs/cleanroom/blind/REPORT.md",
                                                  "label": b["pole_plus"]["label"], "verdict": b["pole_plus"]["verdict"],
                                                  "confidence": b["pole_plus"]["confidence"]}]
        recs[(9, 3688, "-")] = record("bonsai1", 9, 3688, "-", label=b["pole_minus"]["label"], short="somatosensory physiology",
                                      kind="sparse feature", status="probed", depth="blind replication only",
                                      verdict=b["pole_minus"]["verdict"], evidence=["runs/cleanroom/blind/REPORT.md"],
                                      contributors=["blind clean-room agent (runs/cleanroom)"])
    d = G / "neurons" / "bonsai1"
    d.mkdir(parents=True, exist_ok=True)
    for (L, n, pole), r in sorted(recs.items()):
        (d / f"L{L}_N{n}_{'pos' if pole == '+' else 'neg'}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    (G / "edges.json").write_text(json.dumps({"edges": edges}, ensure_ascii=False, indent=1), encoding="utf-8")
    counts = {}
    for r in recs.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"{len(recs)} neuron records, {len(edges)} edges; status counts {counts}")


if __name__ == "__main__":
    main()
