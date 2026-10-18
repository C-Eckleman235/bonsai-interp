"""One comparable set of monosemanticity numbers per probed pole (for calibration across neurons).

  python -m bonsai_interp.probe.scorecard 9/3688 10/5749 22/11024
"""
import json
import sys

from bonsai_interp.probe.workspace import Workspace


def card(ws: Workspace):
    sol = ws.load("solo.json")
    hyp = sol["hypotheses"]
    key = sol["holdout"]["hypothesis"] if "holdout" in sol else max(hyp, key=lambda k: hyp[k]["auroc"] or 0)
    h = hyp[key]
    de = ws.load("invert_de.json") if ws.has("invert_de.json") else None
    off = [r for r in sol["rows"] if r.get(key, r.get("rel_pos", 0)) <= 0.2]
    c = {
        "neuron": ws.id, "label": h["label"], "label_hypothesis": key,
        "bank_auroc": h["auroc"], "bank_spearman": h["spearman"],
        "strong_on": h["strong_on"], "strong_off": h["strong_off"],
        "bank_leak_max": max((r["act"] for r in off), default=None),
        "holdout_auroc": sol.get("holdout", {}).get("auroc"), "holdout_accuracy": sol.get("holdout", {}).get("accuracy"),
        "inverse_leak": de["goals"]["leak"]["y"]["act"] if de and "leak" in de["goals"] else None,
        "inverse_leak_text": de["goals"]["leak"]["text"] if de and "leak" in de["goals"] else None,
        "opposite": sol.get("opposite_pole"), "cross": sol.get("cross"),
    }
    return c


if __name__ == "__main__":
    out = []
    for arg in sys.argv[1:]:
        L, n = map(int, arg.split("/"))
        out.append(card(Workspace(L, n, "+")))
    print(json.dumps(out, ensure_ascii=False, indent=1))
