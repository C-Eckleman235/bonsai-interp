"""Build the interactive neighbourhood page from the web's result files.

  python -m mono.probe.web_page 9/3688 +   ->  probes/<id>/web.html
"""
import json
import re
import sys

from mono.model import ROOT
from mono.probe.workspace import Workspace

TEMPLATE = ROOT / "reports" / "web_template.html"


def short(label, k=24):
    s = re.split(r"[:(;,]| — | - ", label or "")[0].strip().strip("'\"")
    s = re.sub(r"^(The (token|word|spelling|verb|preposition) )", "", s, flags=re.I)
    return (s[:k - 1] + "…") if len(s) > k else s


def build(ws: Workspace):
    W = ws.load("web_nodes.json")
    spec = ws.load("web_spec.json")  # lateral cluster, claim, texts, caveats, short-name overrides (LLM stage)
    labels = {}
    for b in range(3):
        for nb in ws.load(f"web_tests_claude_{b}.json")["neighbours"]:
            labels[nb["unit"]] = (nb["label"], nb.get("confidence"))
    for nb in ws.load("quick_tests.json")["neighbours"]:
        labels.setdefault(nb["unit"], (nb["label"], None))
    qc = {n["unit"]: n for n in ws.load("web_quick_claude.json")["neighbours"]}
    for n in ws.load("quick.json")["neighbours"]:
        qc.setdefault(n["unit"], n)
    qb = {n["unit"]: n for n in ws.load("web_quick_bonsai.json")["neighbours"]}
    slim = lambda r: r and {"survives": r["survives"], "n_fire": r["n_fire"], "n_false_pos": r["n_false_pos"],
                            "n_written": r.get("n_written", [8, 8])}
    nodes = []
    for nd in W["nodes"]:
        u = nd["unit"]
        lab, conf = labels.get(u, (nd.get("known_label"), None))
        nodes.append({"unit": u, "L": nd["L"], "n": nd["n"], "pole": nd["pole"], "roles": nd["roles"], "via": nd.get("via"),
                      "label": lab, "confidence": conf, "short": spec.get("short", {}).get(u) or short(lab),
                      "lateral": u in spec["lateral"], "claude": slim(qc.get(u)), "bonsai": slim(qb.get(u)),
                      "examples": nd["examples"][nd["pole"]]})
    bank = ws.load("web_bank.json")
    data = {"center": W["center"], "claim": spec["claim"], "nodes": nodes, "edges": W["edges"],
            "lateral": spec["lateral"], "lateral_text": spec.get("lateral_text", ""),
            "lateral_inv": {k: (ws.load(f"lateral_{k}.json") if ws.has(f"lateral_{k}.json") else None) for k in ("de", "gmoo")},
            "sources": ws.load("web_sources.json"), "sources_text": spec.get("sources_text", ""),
            "bank_units": bank["units"], "bank_rows": bank["rows"], "bank_acts": bank["acts"], "caveats": spec["caveats"]}
    if not data["lateral_inv"]["de"]:
        data["lateral_inv"] = None
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/").replace("�", "\\ufffd")
    out = ws.path("web.html")
    out.write_text(TEMPLATE.read_text(encoding="utf-8").replace("__WEB_DATA__", blob), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    L, n = map(int, sys.argv[1].split("/"))
    build(Workspace(L, n, sys.argv[2] if len(sys.argv) > 2 else "+"))
