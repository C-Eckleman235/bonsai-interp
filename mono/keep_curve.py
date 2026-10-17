"""Discrete minimal-circuit size: keep only the top-k upstream neurons (by attribution)
among each pole's top 24, zero the other 24-k, and read the retained activation.
Adds 'keep_curve' and 'k_half' to each row of data/screen_<model>.jsonl.
"""
import json
import sys

import numpy as np
import torch

from mono import explore
from mono.inverse.neuron import NeuronSite, circuit_problem
from mono.model import ROOT, load

name = sys.argv[1] if len(sys.argv) > 1 else "bonsai1"
path = ROOT / "data" / f"screen_{name}.jsonl"
rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
torch.cuda.set_per_process_memory_fraction(0.8)
model, tok, rec = load(name)
s, toks = explore.scan(name), explore.corpus()
for r in rows:
    L, n, pole = r["L"], r["n"], "+-".index(r["pole"])
    w, p = s["top_win"][L, pole, 0, n], int(s["top_pos"][L, pole, 0, n])
    site = NeuronSite(model, rec, toks[w].astype(np.int64), L, n, p)
    attr, _ = site.attribution()
    prob = circuit_problem(site, attr, k=24)
    X = np.tril(np.ones((25, 24)), -1)  # row k keeps the first k units
    frac = prob.evaluate(X)[:, 2]
    r["keep_curve"] = [round(float(f), 3) for f in frac]
    r["k_half"] = next((k for k, f in enumerate(frac) if f >= 0.5), None)
    print(f"L{L}/N{n}{r['pole']}: k_half={r['k_half']} curve={r['keep_curve'][:8]}...{r['keep_curve'][-1]}", flush=True)
    torch.cuda.empty_cache()
path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
