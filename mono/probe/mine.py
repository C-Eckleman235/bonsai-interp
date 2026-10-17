"""Stage 7 (mine the neighbourhood): look for alternative readings of the probed pole in the
sentences that drive its causal neighbours.

  extract  top corpus sentences of each key upstream input, gate and downstream reader
           (both poles) -> mined_request.json, to be rated for relevance to the current
           label BEFORE evaluation (writes mined_ratings.json: {"ratings": [0..1, ...]})
  eval     run every mined sentence through the probed neuron (after neutral preambles);
           flag sentences that fire the pole although rated off-label -> mined.json

  python -m mono.probe.mine extract 9/3688 +
  python -m mono.probe.mine eval 9/3688 +
"""
import sys

from mono import explore
from mono.probe import forward
from mono.probe.workspace import Workspace


def extract(ws: Workspace, per_pole=5):
    nb, nmap = ws.load("neighbors.json"), ws.load("neighborhood.json")
    units = [(r["unit"], "input") for r in nb["upstream"][:6]] + \
            [(r["unit"], "gate") for r in nb["disambiguators"]["units"][:2]] + \
            [(r["unit"], "reader") for r in (nb.get("downstream_by_branch", {}).get("psychological", []) or nb["downstream"])[:4]]
    rows, seen = [], set()
    for unit, role in units:
        L, n = map(int, unit[1:].split("/N"))
        for pole in (0, 1):
            for a, _, _, _, t in explore.examples(ws.model, L, n, pole, per_pole, 160, 40):
                s = forward.top_sentence(t)
                if len(s) < 25 or s in seen:
                    continue
                seen.add(s)
                rows.append({"text": s, "source_unit": unit, "source_role": role, "source_pole": "+-"[pole],
                             "source_label": nmap.get("unit_labels", {}).get(unit, "")})
    ws.save("mined_request.json", {
        "instructions": "Rate each sentence 0..1 for relevance to the current label of the probed pole "
                        "(the held-out-tested hypothesis in bank.json), judged from meaning only, before evaluation. "
                        "Write mined_ratings.json as {\"ratings\": [r0, r1, ...]} in the same order.",
        "label": ws.load("bank.json")["hypotheses"][ws.load("solo.json")["holdout"]["hypothesis"]],
        "rows": rows})
    print(f"{len(rows)} mined sentences from {len(units)} neighbours -> mined_request.json")


def evaluate(ws: Workspace):
    from mono.probe.run import setup
    req = ws.require("mined_request.json", "run extract first")
    rat = ws.require("mined_ratings.json", "rate mined_request.json first (before evaluation)")["ratings"]
    assert len(rat) == len(req["rows"]), "one rating per mined sentence"
    _, _, _, ev = setup(ws)
    res = ev([r["text"] for r in req["rows"]])
    rows = [{**r, "rel": float(q), "act": x["act"], "opp": x["opp"], "peak_token": x["peak_token"]}
            for r, q, x in zip(req["rows"], rat, res)]
    alt = sorted([r for r in rows if r["act"] >= 0.3 and r["rel"] <= 0.3], key=lambda r: -r["act"])
    by_unit = {}
    for r in rows:
        u = by_unit.setdefault(r["source_unit"], {"role": r["source_role"], "label": r["source_label"], "n": 0, "fire": 0, "max": 0.0})
        u["n"] += 1
        u["fire"] += r["act"] >= 0.3
        u["max"] = max(u["max"], r["act"])
    ws.save("mined.json", {"rows": rows, "alternatives": alt, "by_unit": by_unit})
    for unit, u in by_unit.items():
        print(f"{u['role']:<7} {unit:<11} fires nerve {u['fire']}/{u['n']}  max {u['max']:.2f}  | {u['label'][:60]}")
    print(f"--- off-label sentences that fire the pole (act >= 0.3, rel <= 0.3): {len(alt)}")
    for r in alt:
        print(f"  {r['act']:.2f} via {r['source_unit']} ({r['source_role']}) peak={r['peak_token']!r} | {r['text'][:110]}")


if __name__ == "__main__":
    L, n = map(int, sys.argv[2].split("/"))
    ws = Workspace(L, n, sys.argv[3] if len(sys.argv) > 3 else "+")
    {"extract": extract, "eval": evaluate}[sys.argv[1]](ws)
