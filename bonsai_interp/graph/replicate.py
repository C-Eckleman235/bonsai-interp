"""Spot-replication: re-score a contribution's committed sentences from scratch and compare with what it claims.

  python -m bonsai_interp.graph.replicate <results .json> ...       one model load for all
  python -m bonsai_interp.graph.replicate --pr-base origin/main [--k 2] [--seed <sha>]
      -> picks k results files ADDED since the base (seeded, so re-runs pick the same ones) and checks them

Run by the `checks` workflow on every PR, on a CPU runner: the 1-bit weights make an honest re-score cheap, so a
results file with invented activations cannot merge quietly. Each sentence must match its recorded activation within
TOL (CPU vs the CUDA reference differ by well under that). An outcome that flips while every number agrees is
reported as borderline, not failed. Exit 1 on a mismatch.
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from bonsai_interp.graph.contrib import NEURON, ROOT
from bonsai_interp.graph.protocols import outcome

TOL = 0.03


def added_results(base):
    out = subprocess.run(["git", "diff", "--name-only", "--diff-filter=A", f"{base}...HEAD", "--", "graph/contributions"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    keep = []
    for f in out:
        if f.endswith(".json") and not f.endswith(".tests.json"):
            c = json.loads((ROOT / f).read_text(encoding="utf-8"))
            if any("per_sentence" in t and str(t.get("tests_file", "")).startswith("graph/") for t in c.get("tests", [])):
                keep.append(f)
    return sorted(keep)


def check(files):
    import torch  # noqa: F401
    from bonsai_interp.env import limit_memory
    from bonsai_interp.model import load
    from bonsai_interp.probe import textinv
    scan = np.load(ROOT / "data" / "scan_bonsai1.npz")  # only top_val is read (NpzFile loads keys on access)
    top_val = scan["top_val"]
    limit_memory(0.8)
    model, tok, rec = load("bonsai1")
    tok.padding_side = "right"
    bad = 0
    for f in files:
        c = json.loads((ROOT / f).read_text(encoding="utf-8"))
        for t in c.get("tests", []):
            if "per_sentence" not in t or not str(t.get("tests_file", "")).startswith("graph/"):
                continue
            spec = json.loads((ROOT / t["tests_file"]).read_text(encoding="utf-8"))
            for nb in spec["neighbours"]:
                L, n = map(int, nb["unit"][1:].split("/N"))
                pole = 0 if nb["pole"] == "+" else 1
                ev = textinv.TextEvaluator(model, tok, rec, L, n, float(top_val[L, pole, 0, n]), float(top_val[L, 1 - pole, 0, n]),
                                           1.0 if pole == 0 else -1.0, batch=32)
                rs = ev(nb["fire"] + nb["quiet"])  # one batch per preamble: on a CPU the per-pass weight unpacking is the cost
                got = {("fire", i): r["act"] for i, r in enumerate(rs[:len(nb["fire"])])} |                       {("quiet", i): r["act"] for i, r in enumerate(rs[len(nb["fire"]):])}
                diffs = [(p["set"], p["i"], p["act"], got[(p["set"], p["i"])]) for p in t["per_sentence"] if (p["set"], p["i"]) in got]
                worst = max((abs(a - b) for _, _, a, b in diffs), default=0.0)
                redo = {**t, "per_sentence": [{**p, "act": round(got[(p["set"], p["i"])], 3)} for p in t["per_sentence"] if (p["set"], p["i"]) in got]}
                same = outcome(redo) == outcome(t)
                # the job is catching invented numbers: every sentence must reproduce within TOL. An outcome that flips
                # while the numbers agree is a test on the pass line (e.g. AUROC 0.906 = 58/64 pairs), not a bad result
                ok = worst <= TOL and len(diffs) == len(t["per_sentence"])
                bad += not ok
                print(f"{'OK  ' if ok else 'FAIL'} {f}: {c.get('neuron')} max |diff| {worst:.3f} (tolerance {TOL}); "
                      f"outcome claimed {outcome(t)}, re-scored {outcome(redo)}"
                      + ("" if same or not ok else "  [borderline: the numbers reproduce; the outcome sits on the pass line]"))
                if not ok:
                    for s, i, a, b in sorted(diffs, key=lambda d: -abs(d[2] - d[3]))[:4]:
                        print(f"     {s} #{i}: claimed {a:.3f}, re-scored {b:.3f}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--pr-base", help="pick from the results files added since this base")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", default="0")
    a = ap.parse_args()
    files = list(a.files)
    if a.pr_base:
        pool = added_results(a.pr_base)
        files += random.Random(a.seed).sample(pool, min(a.k, len(pool)))
        print(f"{len(pool)} results files added; re-scoring {len(files)}: {files}")
    if not files:
        print("nothing to replicate")
        return
    bad = check(files)
    print("spot-replication:", "PASS" if not bad else f"FAIL ({bad} test(s) did not reproduce)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
