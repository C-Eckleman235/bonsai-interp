"""Score the pre-registered H4 test (h4_test.json): one coarser job vs several."""
import sys

import numpy as np

from bonsai_interp.probe import textinv
from bonsai_interp.probe.run import setup
from bonsai_interp.probe.workspace import Workspace


def run(ws):
    spec = ws.load("h4_test.json")
    _, _, _, ev = setup(ws)
    out = {"reference": [{"text": t, **r} for t, r in zip(spec["reference_psych_nerve"], ev(spec["reference_psych_nerve"]))], "branches": {}}
    ref = np.mean([r["act"] for r in out["reference"]])
    print(f"reference psychological 'nerve': mean {ref:.2f}")
    for b, d in spec["branches"].items():
        f, q = ev(d["fire"]), ev(d["quiet"])
        fa, qa = [r["act"] for r in f], [r["act"] for r in q]
        res = {"auroc": textinv.auroc(fa, qa), "mean_fire": float(np.mean(fa)), "mean_quiet": float(np.mean(qa)),
               "strong_fire": float(np.mean([a >= 0.3 for a in fa])),
               "fire": [{"text": t, **r} for t, r in zip(d["fire"], f)], "quiet": [{"text": t, **r} for t, r in zip(d["quiet"], q)]}
        out["branches"][b] = res
        print(f"{b:<20} AUROC {res['auroc']:.2f}  mean {res['mean_fire']:.2f} vs {res['mean_quiet']:.2f}  strong {res['strong_fire']:.0%}  "
              f"| peaks: {[r['peak_token'] for r in f]}")
    n_pass = sum(r["auroc"] >= 0.75 for r in out["branches"].values())
    out["verdict"] = "one coarser job" if n_pass == 4 else f"{4 - n_pass} branch(es) failed"
    print("branches passing AUROC >= 0.75:", n_pass, "->", out["verdict"])
    ws.save("h4_results.json", out)


if __name__ == "__main__":
    L, n = map(int, sys.argv[1].split("/"))
    run(Workspace(L, n, sys.argv[2] if len(sys.argv) > 2 else "+"))
