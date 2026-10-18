"""Stage 9 (lateral cluster): how the probed pole relates to its lateral relatives.

  build    a bank for the cluster: the probed neuron's bank plus both authors' 'should fire'
           sentences for every lateral relative; solo activations of every cluster node
  couple   direct coupling: zero one node (at its layer), measure the later nodes, on each
           node's own sentences
  invert   cluster-level inverse questions (lateral_spec.json goals) with SciPy DE or globalMOO

  python -m mono.probe.lateral build|couple|invert 9/3688 + [--solver de|gmoo]
"""
import sys

import numpy as np
import torch

from mono import explore
from mono.inverse import solvers as S
from mono.inverse.problem import Goal
from mono.model import StopForward, load
from mono.probe import textinv
from mono.probe.workspace import Workspace


def nodes_from(ws, spec, scan):
    out = []
    for nd in spec["nodes"]:
        L, n = nd["L"], nd["n"]
        p = 0 if nd["pole"] == "+" else 1
        out.append((nd["name"], L, n, 1.0 if p == 0 else -1.0, float(scan["top_val"][L, p, 0, n])))
    return out


def build(ws):
    spec = ws.load("lateral_spec.json")
    scan = explore.scan(ws.model)
    bank = ws.load("bank.json")
    key = ws.load("solo.json")["holdout"]["hypothesis"]
    sents = [{"text": s["text"], "rel": s.get(key, 0.0), "rel_neg": s.get("rel_neg", 0.0), "kind": s["kind"], "family": "nerve bank"}
             for s in bank["sentences"]]
    claude = {}
    for b in range(3):
        for nb in ws.load(f"web_tests_claude_{b}.json")["neighbours"]:
            claude[nb["unit"]] = nb
    bonsai = {nb["unit"]: nb for nb in ws.load("web_tests_bonsai.json")["neighbours"]}
    meta = ("feature detector", "the model", "detector", "language model")
    for nd in spec["nodes"]:
        u = f"L{nd['L']}/N{nd['n']}"
        if nd.get("lateral"):
            for src, tests in (("claude", claude), ("bonsai", bonsai)):
                for t in tests.get(u, {}).get("fire", []):
                    if not any(k in t.lower() for k in meta):
                        sents.append({"text": t, "rel": nd["rel"], "rel_neg": 0.0, "kind": "lateral", "family": f"{nd['name']} ({src})"})
    model, tok, rec = load(ws.model)
    tok.padding_side = "right"
    ev = textinv.CircuitEvaluator(model, tok, rec, nodes_from(ws, spec, scan))
    acts = ev([s["text"] for s in sents])
    for s, a in zip(sents, acts):
        s["acts"] = a
        s["act"] = max(a.values())  # strength coordinate: strongest cluster response
        s["rel_pos"] = s["rel"]
    ws.save("lateral_bank.json", {"sentences": sents})
    fams = sorted({s["family"] for s in sents})
    print(f"{len(sents)} sentences")
    for f in fams:
        rows = [s for s in sents if s["family"] == f]
        print(f"  {f:<28} n={len(rows):<3} " + " ".join(f"{k}={np.mean([r['acts'][k] for r in rows]):.2f}" for k in rows[0]["acts"]))


@torch.no_grad()
def couple(ws):
    """Zero each node at its own layer; mean change in every later node, on each family's sentences."""
    spec = ws.load("lateral_spec.json")
    scan = explore.scan(ws.model)
    nodes = nodes_from(ws, spec, scan)
    lb = ws.load("lateral_bank.json")["sentences"]
    model, tok, rec = load(ws.model)
    tok.padding_side = "right"
    base = textinv.CircuitEvaluator(model, tok, rec, nodes)
    fams = {}
    for s in lb:
        fams.setdefault(s["family"].split(" (")[0], []).append(s["text"])
    out = {}
    for name, L, n, sign, pm in nodes:
        rec.interventions = {}
        clean = {f: base(ts) for f, ts in fams.items()}
        ev = textinv.CircuitEvaluator(model, tok, rec, nodes)
        zero = lambda a, n=n: a.index_fill(-1, torch.tensor([n], device=a.device), 0.0)
        ev_cache = {}
        for f, ts in fams.items():
            rec.interventions = {L: zero}
            ev.cache = {}
            ev_cache[f] = ev(ts)
            rec.interventions = {}
        for f in fams:
            for other, Lo, *_ in nodes:
                if Lo <= L or other == name:
                    continue
                d = np.mean([c[other] - z[other] for c, z in zip(clean[f], ev_cache[f])])
                out.setdefault(name, {}).setdefault(other, {})[f] = float(d)
        print(f"zero {name:<14}", {o: {f: round(v, 3) for f, v in fs.items() if abs(v) > 0.02} for o, fs in out.get(name, {}).items()})
    ws.save("lateral_coupling.json", out)


def invert(ws, solver):
    spec = ws.load("lateral_spec.json")
    scan = explore.scan(ws.model)
    model, tok, rec = load(ws.model)
    tok.padding_side = "right"
    ev = textinv.CircuitEvaluator(model, tok, rec, nodes_from(ws, spec, scan))
    lb = ws.load("lateral_bank.json")
    man = textinv.SentenceManifold({"sentences": lb["sentences"]}, lb["sentences"], "rel")
    p = textinv.build_circuit_problem(ev, man)
    out = {"solver": solver, "nodes": spec["nodes"], "goals": {}}
    if solver == "gmoo":
        gm = S.GmooInverse(p, f"mono-lateral-{ws.model}-L{ws.L}-N{ws.n}")
        out["gmoo"] = {"model_id": gm.model_id, "trial_id": gm.trial.id, "doe_evals": gm.doe_evals}
    for name, g in spec["goals"].items():
        G = [Goal(*x) if len(x) == 3 else Goal(x[0], x[1]) for x in g["goals"]]
        r = S.differential_evolution(p, G, maxiter=12, popsize=8) if solver == "de" else gm.query(G, max_iter=40)
        out["goals"][name] = {"question": g["question"], "satisfied": r.satisfied, "evals": r.n_evals, "extra": r.extra,
                              **textinv.describe_circuit(p, r.x, r.y)}
        y = out["goals"][name]["y"]
        print(f"[{solver}] {name:<24} sat={r.satisfied!s:<5} evals={r.n_evals:<4} " +
              " ".join(f"{k}={v:.2f}" for k, v in y.items() if k not in ('dose', 'dose_broad')) + f" | {out['goals'][name]['text'][:90]}", flush=True)
    ws.save(f"lateral_{solver}.json", out)


if __name__ == "__main__":
    L, n = map(int, sys.argv[2].split("/"))
    ws = Workspace(L, n, sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else "+")
    stage = sys.argv[1]
    if stage == "invert":
        invert(ws, sys.argv[sys.argv.index("--solver") + 1] if "--solver" in sys.argv else "de")
    else:
        {"build": build, "couple": couple}[stage](ws)
