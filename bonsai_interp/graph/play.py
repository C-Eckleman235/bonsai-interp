"""The contributor's command line: start a contribution, run its pre-registered test, write the results.

  python -m bonsai_interp.graph.play new interpret --neuron bonsai1/L12/N5397- --handle octocat
  python -m bonsai_interp.graph.play new qc --neuron bonsai1/L9/N3688+ --handle octocat --verdict confirm
      -> graph/contributions/<model>/<pole folder>/<id>.tests.json   (fill in the label and sentences, then COMMIT)
  python -m bonsai_interp.graph.play run graph/contributions/.../<id>.tests.json
      -> <id>.json with the results (needs the GPU; commit it in a SECOND commit)
  python -m bonsai_interp.graph.play step --neuron bonsai1/L12/N5397- [--k 4]
      -> the strongest feeders of a pole (single-unit zeroing on its top examples): candidates for "step out one neuron"
  python -m bonsai_interp.graph.play look --neuron bonsai1/L12/N5397-      top corpus examples
  python -m bonsai_interp.graph.fold                                         refresh the derived records

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

TEMPLATE_FIRE = ["<sentence that should fire the pole>"] * 8
TEMPLATE_QUIET = ["<near miss: same topic, without the claimed trigger>"] * 8


def _commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def new(a):
    m = NEURON.match(a.neuron)
    if not m:
        sys.exit("neuron must look like bonsai1/L12/N5397+ (model/layer/index and pole)")
    cid = f"{dt.date.today().isoformat()}-{a.handle}-{secrets.token_hex(2)}"
    d = folder_for(a.neuron)
    d.mkdir(parents=True, exist_ok=True)
    unit = f"L{m['L']}/N{m['n']}"
    current = None
    rec = ROOT / "graph" / "neurons" / m["model"] / f"{d.name}.json"
    if rec.exists():
        current = json.loads(rec.read_text(encoding="utf-8")).get("label")
    t = {"contribution": cid, "contributor": a.handle, "kind": a.kind, "neuron": a.neuron, "protocol": "quick-8+8@1",
         "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
         **({"qc": {"verdict": a.verdict, "label_under_test": current}} if a.kind == "qc" else {}),
         "neighbours": [{"unit": unit, "pole": m["pole"], "label": current if a.kind == "qc" and a.verdict != "refine" else "<your label>",
                         "fire": TEMPLATE_FIRE, "quiet": TEMPLATE_QUIET}],
         "predictions": "<optional: what you expect, e.g. '6/8 fire, 0 near misses'>"}
    p = d / f"{cid}.tests.json"
    p.write_text(json.dumps(t, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {p.relative_to(ROOT)}\nFill in the label and the 16 sentences, then commit THIS FILE ALONE:\n"
          f"  git add {p.relative_to(ROOT).as_posix()} && git commit -m \"Pre-register {a.kind} test for {a.neuron}\"\n"
          f"then: python -m bonsai_interp.graph.play run {p.relative_to(ROOT).as_posix()}")


def run(a):
    import torch
    from bonsai_interp import explore
    from bonsai_interp.model import load
    from bonsai_interp.probe import textinv
    p = Path(a.tests_file).resolve()
    rel = p.relative_to(ROOT).as_posix()
    if subprocess.run(["git", "log", "--format=%h", "-1", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip() == "":
        sys.exit(f"{rel} is not committed yet. Commit the pre-registration first, then run it.")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode != 0:
        sys.exit(f"{rel} has uncommitted edits. A pre-registration must be run exactly as committed.")
    t = json.loads(p.read_text(encoding="utf-8"))
    if any("<" in s for nb in t["neighbours"] for s in nb["fire"] + nb["quiet"] + [nb["label"]]):
        sys.exit("the tests file still has <placeholders>")
    prereg = subprocess.run(["git", "log", "--diff-filter=A", "--format=%h", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout.split()[-1]
    m = NEURON.match(t["neuron"])
    torch.cuda.set_per_process_memory_fraction(0.8)
    model, tok, rec = load(m["model"])
    tok.padding_side = "right"
    scan = explore.scan(m["model"])
    tests = []
    for nb in t["neighbours"]:
        L, n = map(int, nb["unit"][1:].split("/N"))
        pole = 0 if nb["pole"] == "+" else 1
        pmax, omax = float(scan["top_val"][L, pole, 0, n]), float(scan["top_val"][L, 1 - pole, 0, n])
        ev = textinv.TextEvaluator(model, tok, rec, L, n, pmax, omax, 1.0 if pole == 0 else -1.0)
        fire, quiet = ev(nb["fire"]), ev(nb["quiet"])
        per = [{"set": s, "i": i, "act": round(r["act"], 3), "peak": r["peak_token"]} for s, rs in (("fire", fire), ("quiet", quiet)) for i, r in enumerate(rs)]
        nf, fp = sum(r["act"] >= 0.3 for r in fire), sum(r["act"] >= 0.3 for r in quiet)
        tests.append({"protocol": t["protocol"], "tests_file": rel, "prereg_commit": prereg, "author": a.author or t["contributor"],
                      "n_fire": nf, "n_false_pos": fp, "n_written": [len(fire), len(quiet)], "per_sentence": per})
        print(f"{nb['unit']}{nb['pole']}: fire {nf}/{len(fire)}, near misses firing {fp}/{len(quiet)} -> {'HELD' if nf >= 5 and fp <= 1 else 'FAILED'}")
    c = {"id": t["contribution"], "kind": t["kind"], "neuron": t["neuron"], "contributor": t["contributor"],
         "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
         "recipe": {"toolchain_commit": _commit(), "command": f"python -m bonsai_interp.graph.play run {rel}"},
         "label": t["neighbours"][0]["label"], "tests": tests}
    if t["kind"] == "qc":
        c["qc"] = {"verdict": t["qc"]["verdict"], "label_under_test": t["qc"].get("label_under_test")}
        if t["qc"]["verdict"] != "refine":
            c.pop("label")
    for k in ("short", "unit_kind", "confidence", "notes", "roles", "via"):
        if t.get(k) is not None:
            c[k] = t[k]
    out = p.with_name(p.name.replace(".tests.json", ".json"))
    out.write_text(json.dumps(c, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out.relative_to(ROOT)}. Next: python -m bonsai_interp.graph.fold, then commit the results in a second commit and open a PR.")


def step(a):
    import numpy as np
    import torch
    from bonsai_interp.model import load
    from bonsai_interp.probe.quick import one_back
    m = NEURON.match(a.neuron)
    torch.cuda.set_per_process_memory_fraction(0.8)
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


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new")
    n.add_argument("kind", choices=["interpret", "qc"])
    n.add_argument("--neuron", required=True)
    n.add_argument("--handle", required=True, help="your GitHub handle")
    n.add_argument("--verdict", choices=["confirm", "dispute", "refine"], default="confirm",
                   help="qc only: confirm = re-test the label as is; refine = propose a better label")
    r = sub.add_parser("run")
    r.add_argument("tests_file")
    r.add_argument("--author", help="who wrote the sentences, if not you (e.g. 'bonsai' when the model wrote them)")
    s = sub.add_parser("step")
    s.add_argument("--neuron", required=True)
    s.add_argument("--k", type=int, default=4)
    lk = sub.add_parser("look")
    lk.add_argument("--neuron", required=True)
    a = ap.parse_args()
    {"new": new, "run": run, "step": step, "look": look}[a.cmd](a)


if __name__ == "__main__":
    main()
