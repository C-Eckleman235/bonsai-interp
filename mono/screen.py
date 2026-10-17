"""Tier-1 screening (local only): forward + backward profile of many neurons.

For each (layer, neuron, pole):
  forward   top-K corpus windows from the scan, regime shares
  routes    attribution of the pole's activation to upstream neurons on each of
            the top-E examples; pairwise overlap of the top-k upstream sets and a
            clustering of examples into routes (the essay's "route share")
  inverse   on the strongest example, least_squares over gains on the top-k
            upstream units to hit 25/50/75% of the clean activation (round-trip
            fidelity), and the smallest unit set that keeps 50% (L-BFGS-B on
            Σθ with a retention constraint)
Output: data/screen_<model>.jsonl, one line per neuron-pole.

Usage: python -m mono.screen --model bonsai1 --n 40 --layers 6-30
"""
import argparse
import json
import time

import numpy as np
import torch

from mono import explore
from mono.inverse import solvers as S
from mono.inverse.neuron import NeuronSite, circuit_problem
from mono.inverse.problem import Goal
from mono.model import ROOT, load


def topk_set(attr, k, sign):
    flat = np.concatenate([sign * attr[l] for l in sorted(attr)])
    idx = np.argsort(-flat)[:k]
    n = len(attr[0])
    return {(int(i // n), int(i % n)) for i in idx}, flat


def route_profile(model, rec, L, n, pole, name, n_examples=8, k=24):
    s, toks = explore.scan(name), explore.corpus()
    sign = 1.0 if pole == 0 else -1.0
    sets, vecs, acts = [], [], []
    for j in range(n_examples):
        w, p = s["top_win"][L, pole, j, n], int(s["top_pos"][L, pole, j, n])
        if w < 0:
            break
        site = NeuronSite(model, rec, toks[w].astype(np.int64), L, n, p)
        attr, y = site.attribution()
        st, flat = topk_set(attr, k, sign)  # units that push the activation toward this pole
        sets.append(st)
        vecs.append(flat / (np.linalg.norm(flat) + 1e-9))
        acts.append(y)
    m = len(sets)
    jac = np.array([[len(sets[a] & sets[b]) / len(sets[a] | sets[b]) for b in range(m)] for a in range(m)])
    cos = np.array(vecs) @ np.array(vecs).T
    # routes: connected components at cosine > 0.5 (single linkage)
    parent = list(range(m))
    find = lambda i: i if parent[i] == i else find(parent[i])
    for a in range(m):
        for b in range(a + 1, m):
            if cos[a, b] > 0.5:
                parent[find(a)] = find(b)
    roots = [find(i) for i in range(m)]
    sizes = sorted(np.unique(roots, return_counts=True)[1].tolist(), reverse=True)
    off = ~np.eye(m, dtype=bool)
    return {
        "examples": m,
        "acts": [round(a, 3) for a in acts],
        "jaccard_mean": float(jac[off].mean()) if m > 1 else None,
        "cos_mean": float(cos[off].mean()) if m > 1 else None,
        "route_sizes": sizes,
        "top_route_share": sizes[0] / m,
    }


def inverse_profile(model, rec, L, n, pole, name, k=24):
    s, toks = explore.scan(name), explore.corpus()
    w, p = s["top_win"][L, pole, 0, n], int(s["top_pos"][L, pole, 0, n])
    site = NeuronSite(model, rec, toks[w].astype(np.int64), L, n, p)
    attr, clean = site.attribution()
    full = float(s["top_val"][L, pole, 0, n]) * (1 if pole == 0 else -1)
    prob = circuit_problem(site, attr, k=k)
    prob.scales = np.array([abs(clean) * 0.05 + 1e-3, 1.0, 0.05])
    ones = np.ones(k)
    out = {"clean": clean, "clean_full_window": full, "ctx_fidelity": clean / full,
           "units": [f"L{l}/N{m}" for l, m in prob.units]}
    # all top-k units ablated: how much of the activation do they carry?
    out["frac_all_ablated"] = float(prob.evaluate(np.zeros((1, k)))[0, 2])
    fid = []
    for t in (0.25, 0.5, 0.75):
        r = S.least_squares(prob, [Goal("frac", "~=", t, tol=0.05)], ones * t, max_nfev=25)
        fid.append({"target": t, "achieved": round(float(r.y[2]), 3), "evals": r.n_evals, "kept": round(float(r.y[1]), 2)})
    out["round_trip"] = fid
    # sparsest gain vector keeping half the activation (local, gradient)
    r = S.lbfgs(prob, [Goal("frac", ">=", 0.5, weight=50.0), Goal("kept", "min", weight=1.0)], ones, maxiter=40)
    th = r.x
    out["min_keep_50"] = {"frac": round(float(r.y[2]), 3), "sum_theta": round(float(th.sum()), 2),
                          "units_gt_0.5": [prob.units[j] for j in np.where(th > 0.5)[0]],
                          "evals": r.n_evals}
    return out


def pick_neurons(name, n, layers, seed=0):
    rng = np.random.default_rng(seed)
    s = explore.scan(name)
    picks = []
    while len(picks) < n:
        L = int(rng.integers(layers[0], layers[1] + 1))
        m = int(rng.integers(0, s["top_val"].shape[-1]))
        if np.isfinite(s["top_val"][L, 0, 7, m]):  # needs >= 8 examples
            picks.append((L, m))
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bonsai1")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--layers", default="6-30")
    ap.add_argument("--neurons", default="", help="comma list L/N to screen instead of random picks")
    a = ap.parse_args()
    lo, hi = map(int, a.layers.split("-"))
    neurons = [tuple(map(int, x.split("/"))) for x in a.neurons.split(",")] if a.neurons else \
        pick_neurons(a.model, a.n, (lo, hi))
    torch.cuda.set_per_process_memory_fraction(0.8)  # fail loudly instead of spilling into shared memory
    model, tok, rec = load(a.model)
    out = ROOT / "data" / f"screen_{a.model}.jsonl"
    with open(out, "a", encoding="utf-8") as f:
        for L, n in neurons:
            for pole in (0, 1):
                t0 = time.time()
                row = {"model": a.model, "L": L, "n": n, "pole": "+-"[pole],
                       "regimes": explore.regime_stats(a.model, L, n),
                       "top": [(round(v, 2), r, t) for v, r, _, _, t in explore.examples(a.model, L, n, pole, 10, 32, 4)]}
                row["routes"] = route_profile(model, rec, L, n, pole, a.model)
                row["inverse"] = inverse_profile(model, rec, L, n, pole, a.model)
                row["seconds"] = round(time.time() - t0, 1)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                r = row["routes"]
                print(f"L{L}/N{n}{row['pole']}: routes {r['route_sizes']} jac {r['jaccard_mean']:.2f} | "
                      f"all-top-k ablated -> {row['inverse']['frac_all_ablated']:.2f} | "
                      f"round-trip {[x['achieved'] for x in row['inverse']['round_trip']]} | "
                      f"{row['seconds']}s", flush=True)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
