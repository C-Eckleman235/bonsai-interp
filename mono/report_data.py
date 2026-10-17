"""Collect pilot results (labels + forward examples + backward screen) into one JSON
for the HTML report.  Usage: python -m mono.report_data > runs/report_data.json
"""
import json
import sys

from mono.model import ROOT


def snippet(text, before=110, after=30):
    """Trim around the ⟦peak⟧ token so the peak stays visible."""
    i = text.find("⟦")
    j = text.find("⟧", i)
    start, end = max(0, i - before), min(len(text), j + 1 + after)
    return ("…" if start else "") + text[start:end] + ("…" if end < len(text) else "")


def main():
    interps = json.loads((ROOT / "runs" / "pilot_interps.json").read_text(encoding="utf-8"))
    ante = json.loads((ROOT / "runs" / "pilot_antecedents.json").read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in (ROOT / "data" / "screen_bonsai1.jsonl").read_text(encoding="utf-8").splitlines()]
    out = []
    for key, labels in interps.items():
        if key.startswith("_"):
            continue
        L, n = map(int, key.split("/"))
        neuron = {"id": f"L{L}/N{n}", "L": L, "n": n, "poles": []}
        for pole in "+-":
            r = next((x for x in rows if x["L"] == L and x["n"] == n and x["pole"] == pole), None)
            label, verdict = labels[pole]
            p = {"pole": pole, "label": label, "verdict": verdict}
            if r:
                inv, rt = r["inverse"], r["routes"]
                p.update({
                    "max": r["regimes"][f"{pole}max"],
                    "examples": [{"act": a, "text": snippet(t)} for a, _, t in r["top"][:3]],
                    "routes": rt["route_sizes"], "top_route_share": rt["top_route_share"],
                    "jaccard": rt["jaccard_mean"], "cos": rt["cos_mean"],
                    "ablate_top24": inv["frac_all_ablated"],
                    "round_trip": inv["round_trip"],
                    "keep50": {"frac": inv["min_keep_50"]["frac"], "sum_theta": inv["min_keep_50"]["sum_theta"],
                               "units": [f"L{l}/N{m}" for l, m in inv["min_keep_50"]["units_gt_0.5"]]},
                    "ctx_fidelity": inv.get("ctx_fidelity"),
                    "top_units": inv["units"][:6],
                    "keep_curve": r.get("keep_curve"), "k_half": r.get("k_half"),
                })
            if f"{L}/{n}{pole}" in ante:
                p["antecedents"] = ante[f"{L}/{n}{pole}"]
            neuron["poles"].append(p)
        out.append(neuron)
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
