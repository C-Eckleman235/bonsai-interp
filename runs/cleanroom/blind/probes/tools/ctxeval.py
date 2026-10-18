"""Context-manipulation test: activation of L9/N3688 on a TARGET sentence's tokens only, after a CONTEXT.

  python probes/tools/ctxeval.py SPEC.json --out OUT.json

SPEC: {"contexts": {"name": "text", ...}, "items": [{"id", "branch", "target"}], "pairs": [[ctxA, ctxB], ...]}
Every item is evaluated after every context (and after no context, "none"). The model's standard neutral
preambles come first; the context is then treated as part of the preamble, so only the target's own
tokens are read. Also reports each context alone (its own tokens) as a leak check.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mono.model import load  # noqa: E402
from mono.probe import textinv  # noqa: E402

L, N, PMAX, NMAX = 9, 3688, 7.65625, 19.0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    torch.cuda.set_per_process_memory_fraction(0.8)
    model, tok, rec = load("bonsai1")
    tok.padding_side = "right"
    ctxs = {"none": "", **spec["contexts"]}
    targets = [it["target"] for it in spec["items"]]
    res = {}
    for name, c in ctxs.items():
        pre = [p + (c + " " if c else "") for p in textinv.PREAMBLES]
        ev_p = textinv.TextEvaluator(model, tok, rec, L, N, PMAX, NMAX, 1.0, preambles=pre)
        ev_n = textinv.TextEvaluator(model, tok, rec, L, N, NMAX, PMAX, -1.0, preambles=pre)
        rp, rn = ev_p(targets), ev_n(targets)
        res[name] = [{"pos": r["act"], "neg": q["act"], "peak_pos": r["peak_token"], "peak_neg": q["peak_token"]} for r, q in zip(rp, rn)]
    base = textinv.TextEvaluator(model, tok, rec, L, N, PMAX, NMAX, 1.0)
    base_n = textinv.TextEvaluator(model, tok, rec, L, N, NMAX, PMAX, -1.0)
    names = list(spec["contexts"])
    alone = {nm: {"pos": base([spec["contexts"][nm]])[0]["act"], "neg": base_n([spec["contexts"][nm]])[0]["act"]} for nm in names}
    print("context alone (own tokens):")
    for nm, v in alone.items():
        print(f"  {nm:<16} +{v['pos']:.3f} -{v['neg']:.3f}")
    print(f"\n{'id':<10}{'branch':<10}" + "".join(f"{n[:11]:>12}" for n in ctxs) + "   (+ pole; - pole in brackets)")
    for i, it in enumerate(spec["items"]):
        print(f"{it['id']:<10}{it['branch']:<10}" + "".join(f"{res[n][i]['pos']:>6.2f}({res[n][i]['neg']:.2f})" for n in ctxs) + f"  {it['target'][:50]}")
    summ = {}
    for A, B in spec["pairs"]:
        for br in sorted({it["branch"] for it in spec["items"]}):
            idx = [i for i, it in enumerate(spec["items"]) if it["branch"] == br]
            dp = [res[A][i]["pos"] - res[B][i]["pos"] for i in idx]
            dn = [res[A][i]["neg"] - res[B][i]["neg"] for i in idx]
            summ[f"{A}-vs-{B}/{br}"] = {"n": len(idx), "mean_dpos": float(np.mean(dp)), "frac_pos_up": float(np.mean(np.array(dp) > 0)),
                                        "mean_dneg": float(np.mean(dn)),
                                        "fire_rate_A": float(np.mean([res[A][i]["pos"] >= 0.3 for i in idx])),
                                        "fire_rate_B": float(np.mean([res[B][i]["pos"] >= 0.3 for i in idx]))}
    print()
    for k, v in summ.items():
        print(f"  {k:<40} n={v['n']} mean d(+)={v['mean_dpos']:+.3f} up {v['frac_pos_up']:.2f}  mean d(-)={v['mean_dneg']:+.3f}  "
              f"fire>=0.3 A {v['fire_rate_A']:.2f} B {v['fire_rate_B']:.2f}")
    Path(a.out).write_text(json.dumps({"alone": alone, "results": res, "summary": summ, "items": spec["items"]},
                                      ensure_ascii=False, indent=1), encoding="utf-8")
