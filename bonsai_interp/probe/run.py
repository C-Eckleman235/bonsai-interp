"""Neuron probe: point at a neuron pole, read it forwards, invert it from text, induce it.

  python -m bonsai_interp.probe.run forward  9/3688 +     corpus evidence + bank request
  (fill probes/.../bank.json — an LLM stage)
  python -m bonsai_interp.probe.run solo     9/3688 +     every bank sentence alone: dose-response, leakage
  python -m bonsai_interp.probe.run invert   9/3688 + --solver de     local baseline (SciPy)
  python -m bonsai_interp.probe.run invert   9/3688 + --solver gmoo   one globalMOO model, all goals
  python -m bonsai_interp.probe.run induce   9/3688 +     clamp the neuron: logit effects + generations
"""
import argparse
import time

import numpy as np
import torch

from bonsai_interp.device import limit_memory
from bonsai_interp.inverse import solvers as S
from bonsai_interp.inverse.problem import satisfied
from bonsai_interp.model import load
from bonsai_interp.probe import forward, textinv
from bonsai_interp.probe.workspace import Workspace


def setup(ws):
    limit_memory(0.8)
    model, tok, rec = load(ws.model)
    tok.padding_side = "right"
    fwd = ws.require("forward.json", "run the forward stage first")
    sign = 1.0 if ws.pole == "+" else -1.0
    other = "-" if ws.pole == "+" else "+"
    ev = textinv.TextEvaluator(model, tok, rec, ws.L, ws.n, abs(fwd["max"][ws.pole]), abs(fwd["max"][other]), sign)
    return model, tok, rec, ev


def solo(ws, ev=None):
    bank = ws.require("bank.json", "write bank.json from bank_request.json (LLM stage)")
    ev = ev or setup(ws)[3]
    res = ev([s["text"] for s in bank["sentences"]])
    rel = np.array([s["rel_pos"] for s in bank["sentences"]])
    act = np.array([r["act"] for r in res])
    rows = [{**s, **r} for s, r in zip(bank["sentences"], res)]
    off = [r for r in rows if r["rel_pos"] <= 0.2]
    on = [r for r in rows if r["rel_pos"] >= 0.9]
    out = {
        "rows": rows,
        "spearman_rel_act": textinv.spearman(rel, act),
        "auroc_on_vs_off": textinv.auroc([r["act"] for r in on], [r["act"] for r in off]),
        "leak_max": max(off, key=lambda r: r["act"]) if off else None,
        "strong_off_frac": float(np.mean([r["act"] >= 0.3 for r in off])) if off else None,
        "strong_on_frac": float(np.mean([r["act"] >= 0.3 for r in on])) if on else None,
        "silent_on": [r for r in on if r["act"] <= 0.1],
        "by_kind": {k: {"n": len(v), "mean_act": float(np.mean([r["act"] for r in v])),
                        "max_act": float(max(r["act"] for r in v))}
                    for k in sorted({r["kind"] for r in rows}) for v in [[r for r in rows if r["kind"] == k]]},
    }
    # the opposite pole, scored against its own label (rel_neg) the same way
    rn = np.array([s.get("rel_neg", 0.0) for s in bank["sentences"]])
    opp = np.array([r["opp"] for r in res])
    out["opposite_pole"] = {
        "label": bank["label"]["-" if ws.pole == "+" else "+"],
        "spearman": textinv.spearman(rn, opp), "auroc": textinv.auroc(opp[rn >= 0.9], opp[rn <= 0.2]),
        "strong_on": float(np.mean(opp[rn >= 0.9] >= 0.3)) if (rn >= 0.9).any() else None,
        "strong_off": float(np.mean(opp[rn <= 0.2] >= 0.3)) if (rn <= 0.2).any() else None,
        "leak_max": float(opp[rn <= 0.2].max()) if (rn <= 0.2).any() else None}
    # cross-talk between the poles: does each pole respond to the other pole's concept?
    out["cross"] = {"pole_on_opposite_concept": float(act[rn >= 0.9].mean()) if (rn >= 0.9).any() else None,
                    "opposite_on_pole_concept": float(opp[rel >= 0.9].mean()) if (rel >= 0.9).any() else None}
    # hypothesis competition: how well does each label's relevance rating predict activation?
    hyps = bank.get("hypotheses", {"rel_pos": bank["label"][ws.pole]})
    out["hypotheses"] = {}
    for key, label in hyps.items():
        r_ = np.array([s.get(key, 0.0) for s in bank["sentences"]])
        out["hypotheses"][key] = {
            "label": label, "spearman": textinv.spearman(r_, act),
            "auroc": textinv.auroc(act[r_ >= 0.9], act[r_ <= 0.2]),
            "strong_on": float(np.mean(act[r_ >= 0.9] >= 0.3)) if (r_ >= 0.9).any() else None,
            "strong_off": float(np.mean(act[r_ <= 0.2] >= 0.3)) if (r_ <= 0.2).any() else None}
    # pre-registered held-out test (sentences written before evaluation, with predictions)
    if ws.has("holdout.json"):
        ho = ws.load("holdout.json")
        hres = ev([x["text"] for x in ho["sentences"]])
        hrows = [{**x, **r, "hit": (r["act"] >= 0.3) == (x["predict"] == "fire")} for x, r in zip(ho["sentences"], hres)]
        fire = [r for r in hrows if r["predict"] == "fire"]
        quiet = [r for r in hrows if r["predict"] == "quiet"]
        out["holdout"] = {"hypothesis": ho["hypothesis"], "rows": hrows,
                          "accuracy": float(np.mean([r["hit"] for r in hrows])),
                          "fire_rate_predicted_fire": float(np.mean([r["act"] >= 0.3 for r in fire])),
                          "fire_rate_predicted_quiet": float(np.mean([r["act"] >= 0.3 for r in quiet])),
                          "auroc": textinv.auroc([r["act"] for r in fire], [r["act"] for r in quiet])}
    ws.save("solo.json", out)
    return out


def invert(ws, solver, goals=None, de_maxiter=12, gmoo_iter=40, attach=False, tag=""):
    model, tok, rec, ev = setup(ws)
    bank = ws.require("bank.json", "write bank.json first")
    sol = ws.load("solo.json") if ws.has("solo.json") else solo(ws, ev)
    # the dose axis is the hypothesis whose ratings best predict activation (held-out test reported separately)
    key = sol["holdout"]["hypothesis"] if "holdout" in sol else         max(sol["hypotheses"], key=lambda k: sol["hypotheses"][k]["auroc"] or 0)
    man = textinv.SentenceManifold(bank, sol["rows"], key)
    p = textinv.build_problem(ev, man)
    goals = goals or [g for g in textinv.GOALS if not g.startswith("tight")]
    out = {"solver": solver, "neuron": ws.id, "hypothesis": key, "label": sol["hypotheses"][key]["label"],
           "goals": {}, "started": time.strftime("%Y-%m-%d %H:%M")}
    t0 = time.time()
    if solver == "gmoo" and attach:
        prev = ws.load("invert_gmoo.json")["gmoo"]
        gm = S.GmooInverse.attach(p, prev["model_id"], prev["trial_id"])
        out["gmoo"] = {**prev, "attached": True, "doe_evals_reevaluated": gm.doe_evals}
    elif solver == "gmoo":
        gm = S.GmooInverse(p, f"mono-probe-{ws.model}-L{ws.L}-N{ws.n}{'p' if ws.pole == '+' else 'n'}")
        out["gmoo"] = {"model_id": gm.model_id, "project_id": gm.project_id, "trial_id": gm.trial.id,
                       "doe_evals": gm.doe_evals, "doe_seconds": round(gm.doe_seconds, 1)}
    for g in goals:
        G = textinv.GOALS[g]
        if solver == "de":
            r = S.differential_evolution(p, G, maxiter=de_maxiter, popsize=8)
        elif solver == "lsq_free":
            raise ValueError("text problem is not differentiable")
        elif solver == "gmoo":
            r = gm.query(G, max_iter=gmoo_iter, start="center" if g.startswith("tight") else "best")
        else:
            raise ValueError(solver)
        out["goals"][g] = {"question": textinv.QUESTIONS[g], "satisfied": r.satisfied, "evals": r.n_evals,
                           "seconds": round(r.seconds, 1), "extra": r.extra, **textinv.describe(p, r.x, r.y)}
        print(f"[{solver}] {g:<20} sat={r.satisfied!s:<5} act={r.y[0]:.3f} dose={r.y[2]:.2f} evals={r.n_evals:<4} "
              f"| {out['goals'][g]['text'][:110]}", flush=True)
    out["total_seconds"] = round(time.time() - t0, 1)
    out["forward_passes"] = ev.forward_passes
    out["evaluated"] = [{"y": e["y"], "idx": e["idx"]} for e in p.log]
    ws.save(f"invert_{solver}{tag}.json", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["forward", "solo", "invert", "induce"])
    ap.add_argument("neuron")
    ap.add_argument("pole", nargs="?", default="+")
    ap.add_argument("--solver", default="de")
    ap.add_argument("--goals", default="")
    ap.add_argument("--attach", action="store_true", help="re-use the trained globalMOO trial")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    L, n = map(int, a.neuron.split("/"))
    ws = Workspace(L, n, a.pole)
    if a.stage == "forward":
        forward.run(ws)
    elif a.stage == "solo":
        o = solo(ws)
        print(f"spearman(rel, act)={o['spearman_rel_act']:.3f}  AUROC on-vs-off={o['auroc_on_vs_off']:.3f}  "
              f"strong on={o['strong_on_frac']:.2f} off={o['strong_off_frac']:.2f}")
        for k, v in o["by_kind"].items():
            print(f"  {k:<10} n={v['n']:<3} mean act {v['mean_act']:.3f}  max {v['max_act']:.3f}")
        lm = o["leak_max"]
        print(f"  strongest off-concept sentence: {lm['act']:.3f} '{lm['text'][:90]}' peak={lm['peak_token']!r}")
        print(f"  silent on-concept sentences: {len(o['silent_on'])}")
        op = o["opposite_pole"]
        print(f"  OPPOSITE POLE: spearman {op['spearman']:.3f}  AUROC {op['auroc'] if op['auroc'] is None else round(op['auroc'], 3)}  "
              f"strong on {op['strong_on']}  off {op['strong_off']}  leak max {op['leak_max']}  | {op['label'][:70]}")
        print(f"  cross-talk: pole on opposite concept {o['cross']['pole_on_opposite_concept']}  opposite on pole concept {o['cross']['opposite_on_pole_concept']}")
        for k, h in o["hypotheses"].items():
            print(f"  {k}: spearman {h['spearman']:.3f}  AUROC {h['auroc']:.3f}  strong on {h['strong_on']:.2f} off {h['strong_off']:.2f}  | {h['label'][:70]}")
        if "holdout" in o:
            h = o["holdout"]
            print(f"  HOLDOUT ({h['hypothesis']}): accuracy {h['accuracy']:.2f}  fire-rate predicted-fire {h['fire_rate_predicted_fire']:.2f}  "
                  f"predicted-quiet {h['fire_rate_predicted_quiet']:.2f}  AUROC {h['auroc']:.3f}")
            for r in sorted(h["rows"], key=lambda r: -r["act"]):
                print(f"    {'ok ' if r['hit'] else 'MISS'} {r['predict']:<5} {r['act']:.3f} peak={r['peak_token']!r:<10} {r['text'][:80]}")
    elif a.stage == "invert":
        invert(ws, a.solver, [g for g in a.goals.split(",") if g] or None, attach=a.attach, tag=a.tag)
    elif a.stage == "induce":
        from bonsai_interp.probe import induce
        induce.run(ws)
