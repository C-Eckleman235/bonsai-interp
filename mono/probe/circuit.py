"""Stage 5 (circuit inverse): inverse questions about a small circuit around the probed neuron.

Outputs are several neurons at once (the probed pole, its key inputs, a context
disambiguator, its main reader), each as a fraction of its corpus max. Inputs are the
same natural-passage manifold as the single-neuron inverse. Goals ask about the circuit:
can the pole fire without its main input, can the input be present while the pole is
silent (and what does the disambiguator do then), does the reader have other routes.

  probes/<id>/circuit.json   nodes + goals (written by hand / LLM stage)
  python -m mono.probe.circuit 9/3688 + --solver de|gmoo
"""
import argparse
import time

from mono import explore
from mono.inverse import solvers as S
from mono.inverse.problem import Goal
from mono.probe import textinv
from mono.probe.run import setup
from mono.probe.workspace import Workspace


def run(ws, solver, de_maxiter=12, gmoo_iter=40):
    model, tok, rec, _ = setup(ws)
    spec = ws.require("circuit.json", "write circuit.json (nodes + goals)")
    scan = explore.scan(ws.model)
    nodes = []
    for nd in spec["nodes"]:
        L, n = nd["L"], nd["n"]
        pole = 0 if nd["pole"] == "+" else 1
        nodes.append((nd["name"], L, n, 1.0 if nd["pole"] == "+" else -1.0, float(scan["top_val"][L, pole, 0, n])))
    ev = textinv.CircuitEvaluator(model, tok, rec, nodes)
    bank, sol = ws.load("bank.json"), ws.load("solo.json")
    key = sol["holdout"]["hypothesis"] if "holdout" in sol else "rel_pos"
    p = textinv.build_circuit_problem(ev, textinv.SentenceManifold(bank, sol["rows"], key))
    out = {"solver": solver, "nodes": spec["nodes"], "goals": {}, "started": time.strftime("%Y-%m-%d %H:%M")}
    if solver == "gmoo":
        gm = S.GmooInverse(p, f"mono-circuit-{ws.model}-L{ws.L}-N{ws.n}")
        out["gmoo"] = {"model_id": gm.model_id, "project_id": gm.project_id, "trial_id": gm.trial.id, "doe_evals": gm.doe_evals}
    for name, g in spec["goals"].items():
        G = [Goal(*x) if len(x) == 3 else Goal(x[0], x[1]) for x in g["goals"]]
        r = S.differential_evolution(p, G, maxiter=de_maxiter, popsize=8) if solver == "de" else gm.query(G, max_iter=gmoo_iter)
        out["goals"][name] = {"question": g["question"], "satisfied": r.satisfied, "evals": r.n_evals, "extra": r.extra,
                              **textinv.describe_circuit(p, r.x, r.y)}
        y = out["goals"][name]["y"]
        print(f"[{solver}] {name:<22} sat={r.satisfied!s:<5} evals={r.n_evals:<4} " +
              " ".join(f"{k}={v:.2f}" for k, v in y.items()) + f" | {out['goals'][name]['text'][:90]}", flush=True)
    out["forward_passes"] = ev.forward_passes
    ws.save(f"circuit_{solver}.json", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("neuron")
    ap.add_argument("pole", nargs="?", default="+")
    ap.add_argument("--solver", default="de")
    a = ap.parse_args()
    L, n = map(int, a.neuron.split("/"))
    run(Workspace(L, n, a.pole), a.solver)
