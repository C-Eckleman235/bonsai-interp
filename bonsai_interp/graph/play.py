"""The contributor's command line: start a contribution, run its pre-registered test, write the results.

  python -m bonsai_interp.graph.play new interpret --neuron bonsai1/L12/N5397- --handle octocat
  python -m bonsai_interp.graph.play new qc --neuron bonsai1/L12/N345+ --handle octocat --verdict confirm
      -> graph/contributions/<model>/<pole folder>/<id>.tests.json   (fill in the label and sentences, then COMMIT)
  python -m bonsai_interp.graph.play draw --k 16 [--seed 7] [--layers 0-35]
      -> k random poles with no record yet, for a sweep (no claims: a duplicate is an independent replication)
  python -m bonsai_interp.graph.play new interpret --neuron <id> <id> ... --handle octocat   one tests file per pole
  python -m bonsai_interp.graph.play run graph/contributions/.../<id>.tests.json
      -> <id>.json with the results (needs the GPU; commit it in a SECOND commit)
  python -m bonsai_interp.graph.play step --neuron bonsai1/L12/N5397- [--k 4]
      -> the strongest feeders of a pole (single-unit zeroing on its top examples): candidates for "step out one neuron"
  python -m bonsai_interp.graph.play look --neuron bonsai1/L12/N5397-      top corpus examples
  python -m bonsai_interp.graph.play probe --neuron bonsai1/L12/N5397- "a sentence" "another"   exploration, nothing recorded
  python -m bonsai_interp.graph.play card --neuron bonsai1/L9/N3688+        card for ANY neuron: examples, density, logit effect, flags
  python -m bonsai_interp.graph.play which "some text"                     reverse search: which poles fire on each token
  python -m bonsai_interp.graph.play which --token " nerve"                  poles whose top corpus peaks are this token
  python -m bonsai_interp.graph.fold                                         refresh the derived records
  python -m bonsai_interp.graph.play doctor                                  is this machine ready? (agents: run first)
  python -m bonsai_interp.graph.play brief interpret --neuron <id> --handle <you>   full mission instructions for an agent

The "run" step follows the protocol named in the tests file (quick-8+8@1 today) and records the
toolchain commit, so the result can be reproduced.
"""
import argparse
import datetime as dt
import json
import secrets
import subprocess
import sys
from pathlib import Path

from bonsai_interp.graph.contrib import NEURON, ROOT, folder_for
from bonsai_interp.graph.methods import SCREENS, detect, minutes_between
from bonsai_interp.graph.protocols import outcome

TEMPLATE_FIRE = ["<sentence that should fire the pole>"] * 8
TEMPLATE_QUIET = ["<near miss: same topic, without the claimed trigger>"] * 8


def _commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def new(a):
    bad = [x for x in a.neuron if not NEURON.match(x)]
    if bad:
        sys.exit(f"neuron must look like bonsai1/L12/N5397+ (model/layer/index and pole): {bad}")
    paths = [_new_one(a, nid) for nid in a.neuron]
    rel = [p.relative_to(ROOT).as_posix() for p in paths]
    what = a.neuron[0] if len(paths) == 1 else f"{len(paths)} poles"
    print(f"wrote {len(paths)} tests file(s):\n  " + "\n  ".join(rel) +
          "\nFill in each label and its 16 sentences, then commit the tests files ALONE (one commit for all of them is fine):\n"
          f"  git add {' '.join(rel)} && git commit -m \"Pre-register {a.kind} tests for {what}\"\n"
          f"then: python -m bonsai_interp.graph.play run {' '.join(rel)}")


def _new_one(a, nid):
    m = NEURON.match(nid)
    cid = f"{dt.date.today().isoformat()}-{a.handle}-{secrets.token_hex(2)}"
    d = folder_for(nid)
    d.mkdir(parents=True, exist_ok=True)
    unit = f"L{m['L']}/N{m['n']}"
    current = None
    rec = ROOT / "graph" / "neurons" / m["model"] / f"{d.name}.json"
    if rec.exists():
        current = json.loads(rec.read_text(encoding="utf-8")).get("label")
    t = {"contribution": cid, "contributor": a.handle, "kind": a.kind, "neuron": nid, "protocol": "quick-8+8@2",
         "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
         **({"qc": {"verdict": a.verdict, "label_under_test": current}} if a.kind == "qc" else {}),
         "neighbours": [{"unit": unit, "pole": m["pole"], "label": current if a.kind == "qc" and a.verdict != "refine" else "<your label>",
                         "fire": TEMPLATE_FIRE, "quiet": TEMPLATE_QUIET}],
         "predictions": "<optional: what you expect, e.g. '6/8 fire, 0 near misses'>",
         "methods": {"agent": "<who did the work: e.g. 'claude-code opus-5.5', 'codex gpt-6.1', 'human'>",
                     "screens": ["look"], "_screens_vocabulary": sorted(SCREENS)}}
    p = d / f"{cid}.tests.json"
    p.write_text(json.dumps(t, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return p


def draw(a):
    """k random poles with no record in the graph. No claims: if two people draw the same pole, both results
    count, and the second is an independent replication."""
    import random
    known = set(json.loads((ROOT / "graph" / "index.json").read_text(encoding="utf-8"))["neurons"])
    lo, hi = map(int, a.layers.split("-"))
    rng, out = random.Random(a.seed), []
    while len(out) < a.k:
        nid = f"bonsai1/L{rng.randint(lo, hi)}/N{rng.randrange(12288)}{rng.choice('+-')}"
        if nid not in known and nid not in out:
            out.append(nid)
    print(" ".join(out))
    if a.handle:  # write the tests files for exactly these poles, so the ids cannot drift between draw and new
        a.neuron, a.kind, a.verdict = out, "interpret", "confirm"
        new(a)


def _check_tests_file(path):
    """(problem or None, tests dict, rel path, prereg commit) for one pre-registration."""
    p = Path(path).resolve()
    rel = p.relative_to(ROOT).as_posix()
    if subprocess.run(["git", "log", "--format=%h", "-1", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip() == "":
        return f"{rel}: not committed yet; commit the pre-registration first", None, rel, None
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode != 0:
        return f"{rel}: has uncommitted edits; a pre-registration must be run exactly as committed", None, rel, None
    t = json.loads(p.read_text(encoding="utf-8"))
    if any("<" in s for nb in t["neighbours"] for s in nb["fire"] + nb["quiet"] + [nb["label"]]):
        return f"{rel}: still has <placeholders>", None, rel, None
    agent = (t.get("methods") or {}).get("agent")
    if not agent or "<" in str(agent):
        return f"{rel}: fill in methods.agent (who did the work), commit, then run", None, rel, None
    prereg = subprocess.run(["git", "log", "--diff-filter=A", "--format=%h", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout.split()[-1]
    return None, t, rel, prereg


def run(a):
    """Score one or more committed pre-registrations with ONE model load (batching is where the time goes:
    loading takes 30-60 s, a 16-sentence test about a second)."""
    checked = [(f, *_check_tests_file(f)) for f in a.tests_files]
    problems = [c[1] for c in checked if c[1]]
    for msg in problems:
        print("SKIP", msg)
    todo = [c for c in checked if not c[1]]
    if not todo:
        sys.exit("nothing to run")
    import torch
    from bonsai_interp import explore
    from bonsai_interp.env import limit_memory
    from bonsai_interp.model import load
    from bonsai_interp.probe import textinv
    limit_memory(0.8)
    models, scans = {}, {}
    for f, _, t, rel, prereg in todo:
        m = NEURON.match(t["neuron"])
        if m["model"] not in models:
            models[m["model"]] = load(m["model"])
            models[m["model"]][1].padding_side = "right"
            scans[m["model"]] = explore.scan(m["model"])
        model, tok, rec = models[m["model"]]
        scan = scans[m["model"]]
        tests = []
        for nb in t["neighbours"]:
            L, n = map(int, nb["unit"][1:].split("/N"))
            pole = 0 if nb["pole"] == "+" else 1
            pmax, omax = float(scan["top_val"][L, pole, 0, n]), float(scan["top_val"][L, 1 - pole, 0, n])
            ev = textinv.TextEvaluator(model, tok, rec, L, n, pmax, omax, 1.0 if pole == 0 else -1.0)
            fire, quiet = ev(nb["fire"]), ev(nb["quiet"])
            per = [{"set": s, "i": i, "act": round(r["act"], 3), "peak": r["peak_token"]} for s, rs in (("fire", fire), ("quiet", quiet)) for i, r in enumerate(rs)]
            nf, fp = sum(r["act"] >= 0.3 for r in fire), sum(r["act"] >= 0.3 for r in quiet)
            test = {"protocol": t["protocol"], "tests_file": rel, "prereg_commit": prereg, "author": a.author or t["contributor"],
                    "n_fire": nf, "n_false_pos": fp, "n_written": [len(fire), len(quiet)], "per_sentence": per}
            tests.append(test)
            fa, qa = [r["act"] for r in fire], [r["act"] for r in quiet]
            auroc = sum((x > y) + 0.5 * (x == y) for x in fa for y in qa) / (len(fa) * len(qa))
            med = sorted(fa)[len(fa) // 2] if len(fa) % 2 else sum(sorted(fa)[len(fa) // 2 - 1: len(fa) // 2 + 1]) / 2
            print(f"{t['neuron']:<22} AUROC {auroc:.2f} (need >= 0.90), median fire {med:.2f} (need >= 0.20), "
                  f"near misses >= 0.30: {fp} (max 1); fire >= 0.30: {nf}/{len(fa)} -> {outcome(test).upper()} under {t['protocol']}"
                  if t["protocol"] == "quick-8+8@2" else
                  f"{t['neuron']:<22} {nb['unit']}{nb['pole']}: fire {nf}/{len(fire)}, near misses firing {fp}/{len(quiet)} "
                  f"-> {outcome(test).upper()} under {t['protocol']}")
        p = Path(f).resolve()
        declared = dict(t.get("methods") or {})
        declared.pop("_screens_vocabulary", None)
        tests_screens = ["quick-test"] + (["second-round"] if len(list(p.parent.glob("*.tests.json"))) > 1 else [])
        meth = {**declared, "screens": sorted(set(declared.get("screens", [])) | set(tests_screens)),
                "detected": detect(ROOT, int(m["L"]), int(m["n"]), m["pole"]), "toolchain_commit": _commit(),
                "minutes": minutes_between(ROOT, rel), "batch_size": len(todo)}
        unknown = [x for x in meth["screens"] if x not in SCREENS]
        if unknown:
            print(f"note: screens not in the shared vocabulary (add them in bonsai_interp/graph/methods.py): {unknown}")
        c = {"id": t["contribution"], "kind": t["kind"], "neuron": t["neuron"], "contributor": t["contributor"],
             "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
             "recipe": {"toolchain_commit": _commit(), "command": f"python -m bonsai_interp.graph.play run {rel}"},
             "label": t["neighbours"][0]["label"], "tests": tests, "methods": meth}
        if t["kind"] == "qc":
            c["qc"] = {"verdict": t["qc"]["verdict"], "label_under_test": t["qc"].get("label_under_test")}
            if t["qc"]["verdict"] != "refine":
                c.pop("label")
        for k in ("short", "unit_kind", "confidence", "notes", "roles", "via"):
            if t.get(k) is not None:
                c[k] = t[k]
        out = p.with_name(p.name.replace(".tests.json", ".json"))
        out.write_text(json.dumps(c, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"   wrote {out.relative_to(ROOT).as_posix()}")
    print(f"\n{len(todo)} scored, {len(problems)} skipped. Commit the results files in a SEPARATE commit from the pre-registrations "
          f"and open a PR with only graph/contributions/ files (the merge agent regenerates the derived graph files).")


def step(a):
    import numpy as np
    import torch
    from bonsai_interp.env import limit_memory
    from bonsai_interp.model import load
    from bonsai_interp.probe.quick import one_back
    m = NEURON.match(a.neuron)
    limit_memory(0.8)
    model, tok, rec = load(m["model"])
    feeders = one_back(model, rec, int(m["L"]), int(m["n"]), 0 if m["pole"] == "+" else 1, k_show=a.k)
    for f in feeders:
        print(f"{f['unit']:<11} drop {f['drop']:.2f}  + peaks {f['peaks_pos']}  - peaks {f['peaks_neg']}")
    print("Pick one, then: python -m bonsai_interp.graph.play new interpret --neuron bonsai1/<unit><pole> ... "
          "and add 'roles': ['feeds <this neuron>'] plus an edge in your results notes.")


def look(a):
    m = NEURON.match(a.neuron)
    subprocess.run([sys.executable, "-m", "bonsai_interp.explore", "--model", m["model"], "--neuron", f"{m['L']}/{m['n']}",
                    "--pole", m["pole"], "--k", "16"], cwd=ROOT)



def probe(a):
    """Exploration only: score sentences on a pole without recording anything. Use it freely before
    pre-registering; the committed test is what counts."""
    import torch
    from bonsai_interp import explore
    from bonsai_interp.env import limit_memory
    from bonsai_interp.model import load
    from bonsai_interp.probe import textinv
    m = NEURON.match(a.neuron)
    if not m:
        sys.exit("neuron must look like bonsai1/L12/N345+")
    texts = list(a.sentences) + ([ln.strip() for ln in Path(a.file).read_text(encoding="utf-8").splitlines() if ln.strip()] if a.file else [])
    if not texts:
        sys.exit("give sentences as arguments or --file (one per line)")
    L, n, pole = int(m["L"]), int(m["n"]), 0 if m["pole"] == "+" else 1
    limit_memory(0.8)
    model, tok, rec = load(m["model"])
    tok.padding_side = "right"
    s = explore.scan(m["model"])
    ev = textinv.TextEvaluator(model, tok, rec, L, n, float(s["top_val"][L, pole, 0, n]), float(s["top_val"][L, 1 - pole, 0, n]),
                               1.0 if pole == 0 else -1.0)
    for t, r in zip(texts, ev(texts)):
        print(f"{r['act']:6.3f}  peak {r['peak_token']!r:<14} {t}")
    print("(exploration only: nothing was recorded. Fractions of the pole's corpus max. A quick-8+8@2 test holds when its 8 fire "
          "sentences beat its 8 near misses (AUROC >= 0.90), the median fire sentence reaches 0.20, and at most 1 near miss reaches 0.30)")


def edge(a):
    """Record measured edges as a 'map' contribution (no test needed; both ends need records of their own)."""
    for end in (a.src, a.dst):
        if not NEURON.match(end):
            sys.exit(f"{end}: use full ids like bonsai1/L8/N1234+")
    cid = f"{dt.date.today().isoformat()}-{a.handle}-{secrets.token_hex(2)}"
    c = {"id": cid, "kind": "map", "neuron": a.dst, "contributor": a.handle,
         "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
         "recipe": {"toolchain_commit": _commit(), "command": a.method},
         "edges": [{"from": a.src, "to": a.dst, "kind": a.kind, "weight": round(a.weight, 4), "method": a.method}]}
    d = folder_for(a.dst)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{cid}.json"
    p.write_text(json.dumps(c, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {p.relative_to(ROOT)}")

def main():
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to a legacy code page
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new")
    n.add_argument("kind", choices=["interpret", "qc"])
    n.add_argument("--neuron", required=True, nargs="+", help="one or more poles; one tests file each")
    n.add_argument("--handle", required=True, help="your GitHub handle")
    n.add_argument("--verdict", choices=["confirm", "dispute", "refine"], default="confirm",
                   help="qc only: confirm = re-test the label as is; refine = propose a better label")
    r = sub.add_parser("run", help="score one or more committed pre-registrations (one model load for all)")
    r.add_argument("tests_files", nargs="+")
    r.add_argument("--author", help="who wrote the sentences, if not you (e.g. 'bonsai' when the model wrote them)")
    dr = sub.add_parser("draw", help="k random poles with no record yet (for a sweep)")
    dr.add_argument("--k", type=int, default=16)
    dr.add_argument("--seed", type=int, help="reproducible draw (default: random)")
    dr.add_argument("--layers", default="0-35")
    dr.add_argument("--handle", help="also write an interpret tests file for each drawn pole (play new), as a sweep does")
    s = sub.add_parser("step")
    s.add_argument("--neuron", required=True)
    s.add_argument("--k", type=int, default=4)
    lk = sub.add_parser("look")
    lk.add_argument("--neuron", required=True)
    pr = sub.add_parser("probe", help="exploration: score sentences on a pole, nothing recorded")
    pr.add_argument("--neuron", required=True)
    pr.add_argument("sentences", nargs="*")
    pr.add_argument("--file", help="one sentence per line")
    e = sub.add_parser("edge", help="record a measured edge between two poles")
    e.add_argument("--from", dest="src", required=True)
    e.add_argument("--to", dest="dst", required=True)
    e.add_argument("--kind", default="feeds", choices=["feeds", "reads", "gates", "co-fires"])
    e.add_argument("--weight", type=float, required=True, help="feeds: share of the target's activation lost when the source is zeroed")
    e.add_argument("--method", default="single-unit zero ablation (play step)")
    e.add_argument("--handle", required=True)
    c = sub.add_parser("card", help="a card for any neuron (labelled or not): examples, density, regimes, logit effect")
    c.add_argument("--neuron", required=True, help="bonsai1/L9/N3688+ (the pole is shown first) or L9/N3688")
    c.add_argument("--k", type=int, default=6, help="top examples per pole")
    c.add_argument("--out", help="also write the card: .json for data, anything else for the text")
    c.add_argument("--no-logits", action="store_true", help="skip the direct logit effect (no weights needed)")
    w = sub.add_parser("which", help="reverse search: text -> poles firing on each token, or --token -> poles")
    w.add_argument("text", nargs="?")
    w.add_argument("--token", help="a token as the model sees it, e.g. ' nerve' (leading space matters)")
    w.add_argument("--k", type=int, default=8, help="poles per token")
    w.add_argument("--min-max", type=float, default=0.5, help="ignore poles whose corpus max is below this (fractions of a tiny max are noise)")
    w.add_argument("--keep-dense", action="store_true", help="also list dense / positional poles (hidden by default)")
    w.add_argument("--rebuild", action="store_true", help="--token: rebuild the token index from the scan")
    d = sub.add_parser("doctor", help="check this machine is ready to play")
    d.add_argument("--quick", action="store_true", help="skip loading the model")
    b = sub.add_parser("brief", help="print the full instructions for a mission (for an AI agent)")
    b.add_argument("kind", choices=["interpret", "sweep", "qc", "step", "blind"])
    b.add_argument("--neuron", help="the pole (every kind but sweep)")
    b.add_argument("--k", type=int, default=16, help="sweep: how many poles")
    b.add_argument("--handle", default="<your-github-handle>")
    b.add_argument("--gmoo", action="store_true", help="include the optional globalMOO step")
    b.add_argument("--depth", choices=["quick", "standard", "deep"], default="quick",
                   help="quick (~5-10 min, the default), standard (~30-45 min) or deep (hours)")
    a = ap.parse_args()
    if a.cmd in ("doctor", "brief"):
        from bonsai_interp.graph import agent
        return getattr(agent, a.cmd)(a)
    if a.cmd == "card":
        from bonsai_interp.graph import card
        return card.main(a)
    if a.cmd == "which":
        from bonsai_interp.graph import which
        return which.main(a)
    {"new": new, "draw": draw, "run": run, "step": step, "look": look, "edge": edge, "probe": probe}[a.cmd](a)


if __name__ == "__main__":
    main()
