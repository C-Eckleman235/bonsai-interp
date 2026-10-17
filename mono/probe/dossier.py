"""Assemble a neuron's probe results into one self-contained HTML dossier.

  python -m mono.probe.dossier 9/3688 +   ->  probes/<neuron>/dossier.html
"""
import json
import sys

from mono.model import ROOT
from mono.probe.workspace import Workspace

TEMPLATE = ROOT / "reports" / "dossier_template.html"


def opt(ws, name):
    return ws.load(name) if ws.has(name) else None


def build(ws: Workspace):
    fwd, bank, solo = ws.load("forward.json"), ws.load("bank.json"), ws.load("solo.json")
    keep = lambda r: {k: r.get(k) for k in ("text", "kind", "branch", "rel_pos", "rel_h2", "rel_neg", "act", "opp", "peak_token", "peak_char")}
    data = {
        "neuron": ws.id, "L": ws.L, "n": ws.n, "pole": ws.pole, "model": ws.model,
        "max": fwd["max"], "label": bank["label"], "hypotheses": solo.get("hypotheses"),
        "examples": {p: [{"act": e["act"], "text": e["text"]} for e in fwd["examples"][p][:6]] for p in "+-"},
        "bank": [keep(r) for r in solo["rows"]],
        "holdout": solo.get("holdout"),
        "invert": {k: opt(ws, f"invert_{k}.json") for k in ("gmoo", "de", "gmoo_tight", "de_tight")},
        "plan": opt(ws, "gmoo_plan.json"), "verdict": opt(ws, "verdict.json"),
        "induce": opt(ws, "induce.json"), "judgement": opt(ws, "induce_judgement.json"),
    }
    if ws.has("neighbors.json"):
        nb = ws.load("neighbors.json")
        data["neighbors"] = {"upstream": nb["upstream"][:8],
                             "disambiguators": {**{k: v for k, v in nb["disambiguators"].items() if k != "units"},
                                                "units": nb["disambiguators"]["units"][:3]},
                             "downstream_by_branch": {b: r[:5] for b, r in nb.get("downstream_by_branch", {}).items()}}
    data["map"] = opt(ws, "neighborhood.json")
    data["circuit_inv"] = {k: opt(ws, f"circuit_{k}.json") for k in ("de", "gmoo")}
    for k, v in data["invert"].items():
        if v:
            v.pop("evaluated", None)
    # circuit-level backward view from the Tier-1 screen, if this neuron was screened
    screen = ROOT / "data" / f"screen_{ws.model}.jsonl"
    if screen.exists():
        for line in screen.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["L"] == ws.L and r["n"] == ws.n and r["pole"] == ws.pole:
                data["circuit"] = {"keep_curve": r.get("keep_curve"), "units": r["inverse"]["units"],
                                   "round_trip": r["inverse"]["round_trip"], "ablate_top24": r["inverse"]["frac_all_ablated"]}
        ante = ROOT / "runs" / "pilot_antecedents.json"
        key = f"{ws.L}/{ws.n}{ws.pole}"
        if ante.exists() and key in (a := json.loads(ante.read_text(encoding="utf-8"))) and "circuit" in data:
            data["circuit"]["antecedents"] = a[key]
    # partial-byte tokens decode to U+FFFD; keep them as JSON escapes so the file holds none
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/").replace("�", "\\ufffd")
    title = (data["verdict"] or {}).get("title") or f"Neuron {ws.id}"
    html = TEMPLATE.read_text(encoding="utf-8").replace("__TITLE__", title).replace("__DOSSIER_DATA__", blob)
    out = ws.path("dossier.html")
    out.write_text(html, encoding="utf-8")
    return out


if __name__ == "__main__":
    L, n = map(int, sys.argv[1].split("/"))
    print("wrote", build(Workspace(L, n, sys.argv[2] if len(sys.argv) > 2 else "+")))
