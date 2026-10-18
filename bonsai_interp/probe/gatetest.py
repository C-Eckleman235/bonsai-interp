"""Stage 6 (pre-registered circuit test): score gate_test.json predictions.

Each sentence is read after [preamble] + context sentence, for neutral and pain contexts,
and the circuit nodes are measured on the sentence's own tokens. Predictions are scored
exactly as written in gate_test.json; nothing here may be tuned after seeing results.

  python -m bonsai_interp.probe.gatetest 9/3688 +
"""
import sys

import numpy as np

from bonsai_interp import explore
from bonsai_interp.probe import textinv
from bonsai_interp.probe.run import setup
from bonsai_interp.probe.workspace import Workspace

NODES = [("nerve", 9, 3688, "+"), ("token", 3, 6493, "+"), ("concept", 8, 7756, "+"), ("pain", 8, 3858, "+")]


def run(ws: Workspace):
    model, tok, rec, _ = setup(ws)
    spec = ws.require("gate_test.json", "write and commit gate_test.json first")
    scan = explore.scan(ws.model)
    nodes = [(nm, L, n, 1.0 if p == "+" else -1.0, float(scan["top_val"][L, 0 if p == "+" else 1, 0, n])) for nm, L, n, p in NODES]
    meas = spec["measurement"]
    evs = {cond: textinv.CircuitEvaluator(model, tok, rec, nodes,
                                          preambles=[p + c + " " for p in textinv.PREAMBLES for c in meas[f"{cond}_contexts"]])
           for cond in ("neutral", "pain")}
    rows = {}
    for set_name, st in spec["sets"].items():
        sents = st["sentences"] + st.get("exploratory", [])
        res = {cond: ev(sents) for cond, ev in evs.items()}
        rows[set_name] = []
        for i, s in enumerate(sents):
            r = {"text": s, "scored": i < len(st["sentences"]),
                 **{f"{k}_{cond}": res[cond][i][k] for cond in res for k in res[cond][i]}}
            r["ratio"] = r["nerve_pain"] / r["nerve_neutral"] if r["nerve_neutral"] > 1e-6 else None
            rows[set_name].append(r)

    def scored(name):
        return [r for r in rows[name] if r["scored"]]

    def count(rs, f):
        return sum(1 for r in rs if f(r))

    A, B, C, D = scored("A_open_latin"), scored("B_closed_latin"), scored("C_concept_route"), scored("D_controls_latin_anatomical")
    ratio_ok = lambda lo=None, hi=None: (lambda r: r["ratio"] is not None and (lo is None or r["ratio"] >= lo) and (hi is None or r["ratio"] <= hi))
    checks = [
        ("A fires_neutral", count(A, lambda r: r["nerve_neutral"] >= 0.3), 6, len(A)),
        ("A pain_gate (ratio <= 0.5)", count(A, ratio_ok(hi=0.5)), 6, len(A)),
        ("A pain_unit_on", count(A, lambda r: r["pain_pain"] >= 0.3), 6, len(A)),
        ("B token_present", count(B, lambda r: r["token_neutral"] >= 0.4), 5, len(B)),
        ("B pole_silent", count(B, lambda r: r["nerve_neutral"] < 0.3), 5, len(B)),
        ("C concept_on", count(C, lambda r: r["concept_neutral"] >= 0.3), 3, len(C)),
        ("C fires_either_sense", count(C, lambda r: r["nerve_neutral"] >= 0.3), 3, len(C)),
        ("C not_pain_gated (ratio >= 0.5)", count(C, ratio_ok(lo=0.5)), 3, len(C)),
        ("D token_present", count(D, lambda r: r["token_neutral"] >= 0.4), 3, len(D)),
        ("D pole_weak", count(D, lambda r: r["nerve_neutral"] < 0.3), 3, len(D)),
    ]
    mean_a, mean_b = np.mean([r["nerve_neutral"] for r in A]), np.mean([r["nerve_neutral"] for r in B])
    results = [{"prediction": n, "observed": k, "needed": need, "of": tot, "pass": k >= need} for n, k, need, tot in checks]
    results.append({"prediction": "B contrast_with_A (mean B < mean A / 2)", "observed": round(float(mean_b), 3),
                    "needed": round(float(mean_a / 2), 3), "of": None, "pass": bool(mean_b < mean_a / 2)})
    out = {"rows": rows, "results": results}
    ws.save("gate_test_results.json", out)
    for r in results:
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['prediction']:<38} observed {r['observed']} (needed {r['needed']}" +
              (f" of {r['of']})" if r["of"] else ")"))
    for name, rs in rows.items():
        print(f"--- {name}")
        for r in rs:
            print(f"  {'   ' if r['scored'] else 'exp'} nerve {r['nerve_neutral']:.2f}→{r['nerve_pain']:.2f} "
                  f"(ratio {r['ratio'] if r['ratio'] is None else round(r['ratio'], 2)})  token {r['token_neutral']:.2f}  "
                  f"concept {r['concept_neutral']:.2f}  pain {r['pain_neutral']:.2f}→{r['pain_pain']:.2f}  | {r['text'][:60]}")
    return out


if __name__ == "__main__":
    L, n = map(int, sys.argv[1].split("/"))
    run(Workspace(L, n, sys.argv[2] if len(sys.argv) > 2 else "+"))
