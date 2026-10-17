"""Side-by-side calibration page for several probed neurons.

  python -m mono.probe.calibration 9/3688 10/5749 22/11024  ->  reports/calibration.html
"""
import json
import sys

from mono.model import ROOT
from mono.probe.workspace import Workspace

TEMPLATE = ROOT / "reports" / "calibration_template.html"


def neuron(ws: Workspace):
    sol, bank, v = ws.load("solo.json"), ws.load("bank.json"), ws.load("verdict.json")
    key = sol["holdout"]["hypothesis"] if "holdout" in sol else "rel_pos"
    de = ws.load("invert_de.json") if ws.has("invert_de.json") else {}
    gm = ws.load("invert_gmoo.json") if ws.has("invert_gmoo.json") else {}
    ind = ws.load("induce.json") if ws.has("induce.json") else {}
    sweep = {s["c"]: s for s in ind.get("sweep", [])}
    h = sol["hypotheses"][key]
    ho = sol.get("holdout", {})
    return {
        "id": ws.id, "title": v["title"], "verdict": v["verdict"], "label": h["label"], "hyp_key": key,
        "bank": [{"rel": r.get(key, r.get("rel_pos", 0)), "rel_neg": r.get("rel_neg", 0), "act": r["act"], "opp": r["opp"], "kind": r["kind"],
                  "text": r["text"], "branch": r.get("branch", "")} for r in sol["rows"]],
        "holdout": [{"act": r["act"], "opp": r["opp"], "predict": r["predict"], "text": r["text"]} for r in ho.get("rows", [])],
        "metrics": {
            "holdout_fire": ho.get("fire_rate_predicted_fire"), "holdout_fp": ho.get("fire_rate_predicted_quiet"),
            "holdout_auroc": ho.get("auroc"), "holdout_acc": ho.get("accuracy"),
            "bank_auroc": h["auroc"], "bank_strong_on": h["strong_on"],
            "opp_auroc": sol["opposite_pole"]["auroc"], "opp_strong_on": sol["opposite_pole"]["strong_on"],
            "cross": sol["cross"]["opposite_on_pole_concept"],
            "leak": de.get("goals", {}).get("leak", {}).get("y", {}).get("act"),
            "leak_text": de.get("goals", {}).get("leak", {}).get("text"),
            "kl_pos20": sweep.get(20, {}).get("kl_from_base"), "kl_neg20": sweep.get(-20, {}).get("kl_from_base"),
            "gains_pos": sweep.get(10, {}).get("top_gain", [])[:6], "gains_neg": sweep.get(-10, {}).get("top_gain", [])[:6],
            "gmoo_levels": [gm.get("goals", {}).get(g, {}).get("satisfied") for g in ("level_25", "level_50", "level_75")],
            "gmoo_level_evals": [gm.get("goals", {}).get(g, {}).get("evals") for g in ("level_25", "level_50", "level_75")],
        },
    }


if __name__ == "__main__":
    data = [neuron(Workspace(*map(int, a.split("/")), "+")) for a in sys.argv[1:]]
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/").replace("�", "\\ufffd")
    out = ROOT / "reports" / "calibration.html"
    out.write_text(TEMPLATE.read_text(encoding="utf-8").replace("__CAL_DATA__", blob), encoding="utf-8")
    print("wrote", out)
