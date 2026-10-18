"""P4D: corpus-wide test of the left-context veto on nerve tokens (bands scan, 6000 windows)."""
import re, sys, json
import numpy as np
sys.path.insert(0, '.')
from mono.probe import bands, textinv
from mono import explore

d = bands.load_bands('n3688'); a = d['L9_N3688'].astype(np.float32)
toks = explore.corpus()[:a.shape[0]]; tk = explore.tokenizer()
ids = {30915, 49481}  # " nerve", " nerves" (BUGFIX: " Nerve" is two tokens " N"+"erve"; its first id matched every " N")
rx = re.compile(r"pain|sensation|numb|tingl|burning|receptor|touch|ache|sensory", re.I)
rows = []
for i, j in zip(*np.where(np.isin(toks, list(ids)))):
    left = tk.decode(toks[i, max(0, j - 40):j])
    rows.append((bool(rx.search(left)), a[i, j] / 7.65625, -a[i, j] / 19.0))
pc = np.array([r[1] for r in rows if r[0]]); nc = np.array([r[1] for r in rows if not r[0]])
pn = np.array([r[2] for r in rows if r[0]]); nn = np.array([r[2] for r in rows if not r[0]])
out = {"n_pain_ctx": len(pc), "n_no_ctx": len(nc), "median_plus_pain_ctx": float(np.median(pc)), "median_plus_no_ctx": float(np.median(nc)),
       "auroc_no_ctx_gt_pain_ctx_plus": textinv.auroc(nc, pc), "median_minus_pain_ctx": float(np.median(pn)), "median_minus_no_ctx": float(np.median(nn)),
       "frac_plus_ge_0.2_no_ctx": float(np.mean(nc >= 0.2)), "frac_plus_ge_0.2_pain_ctx": float(np.mean(pc >= 0.2))}
print(json.dumps(out, indent=1))
json.dump(out, open('probes/tests/nerve_gate_out.json', 'w'), indent=1)
