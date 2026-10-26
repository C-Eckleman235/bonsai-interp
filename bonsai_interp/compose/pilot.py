"""Feasibility pilot for the compositional pass: are mid-layer poles fed by a few characterized poles?

  python -m bonsai_interp.compose.pilot --poles 300 --layers 6-11 --examples 8 --out data/compose/pilot.json

For each sampled pole (non-weak, not dense or positional), on its strongest corpus examples (the scan's top), the
position-resolved attribution (feeders.py) is aggregated into feeder keys (pole, offset band). The pilot reports:
  - concentration: the share of the positive attribution mass on the top 1, 3 and 10 feeder keys;
  - attention: the share by offset band (same token vs carried from earlier positions);
  - readiness: the share on feeders already characterized, either assignment-grade (a survey label covering at least
    50% of the feeder's strong activations, or a word-screen test whose trigger word covers at least 50%) or any held
    test (survey held or partial, word screen, sweep library).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from bonsai_interp.compose import feeders as F

ROOT = Path(__file__).resolve().parents[2]


def characterized():
    """{(L, n, sign): 'assigned' | 'held'} from the survey index and the atlas."""
    from bonsai_interp import atlas
    out = {}
    ix = json.loads((ROOT / "graph" / "index.json").read_text(encoding="utf-8"))["neurons"]
    for nid, v in ix.items():
        if not nid.startswith("bonsai1/"):
            continue
        L, rest = nid.split("/L")[1].split("/N")
        key = (int(L), int(rest[:-1]), rest[-1])
        if v["status"] in ("held", "replicated") and (v.get("coverage") or 0) >= 0.5:
            out[key] = "assigned"
        elif v["status"] in ("held", "partial", "replicated"):
            out[key] = "held"
    for L in range(36):
        t = atlas.load(L)
        if t is None:
            continue
        d = t.select(["neuron", "sign", "test", "hit_coverage", "hit_trigger", "lib_test"]).to_pydict()
        for i, n in enumerate(d["neuron"]):
            key = (L, n, d["sign"][i])
            if key in out and out[key] == "assigned":
                continue
            if d["test"][i] == "held":
                cov = max([c for c, tg in zip(d["hit_coverage"][i] or [], d["hit_trigger"][i] or []) if tg] or [0])
                out[key] = "assigned" if cov >= 0.5 else out.get(key, "held")
            elif d["lib_test"][i] == "held":
                out.setdefault(key, "held")
    return out


def label_of(key, ix, at_cache={}):
    L, n, s = key
    v = ix.get(f"bonsai1/L{L}/N{n}{s}")
    if v and v.get("label"):
        return v["label"][:70]
    from bonsai_interp import atlas
    if L not in at_cache:
        t = atlas.load(L)
        at_cache[L] = (t.select(["neuron", "sign", "bin", "context", "hit_tok"]).to_pydict(), atlas.vocab()) if t is not None else None
    if not at_cache[L]:
        return ""
    d, vb = at_cache[L]
    i = 2 * n + (s == "-")
    words = " ".join(repr(vb.get(w, w)) for w in (d["hit_tok"][i] or [])[:2])
    return f"[{d['bin'][i] or '-'}] {d['context'][i] or words}"[:70]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poles", type=int, default=300)
    ap.add_argument("--layers", default="6-11")
    ap.add_argument("--examples", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "data" / "compose" / "pilot.json"))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    from bonsai_interp import explore
    from bonsai_interp.auto import context_tier as ct
    from bonsai_interp.env import limit_memory
    from bonsai_interp.graph import which
    lo, hi = map(int, a.layers.split("-"))
    ti, sc, toks = which.load_index(), explore.scan("bonsai1"), explore.corpus()
    rng = np.random.default_rng(a.seed)
    cand = []
    for L in range(lo, hi + 1):
        el = ct.eligibility(L)
        for p in (0, 1):
            cand += [(L, int(n), p) for n in np.nonzero(el[p] == "eligible")[0]]
    pick = [cand[i] for i in sorted(rng.choice(len(cand), a.poles, replace=False))]
    char = characterized()
    ix = json.loads((ROOT / "graph" / "index.json").read_text(encoding="utf-8"))["neurons"]
    print(f"{len(pick)} poles from layers {lo}-{hi}; {len(char)} characterized poles known "
          f"({sum(v == 'assigned' for v in char.values())} assignment-grade)", flush=True)
    limit_memory(0.8)
    model, tok, rec = ct._gpu_load()
    out, t0 = [], time.time()
    for j, (L, n, p) in enumerate(pick):
        ct._stop_requested()
        sign = 1.0 if p == 0 else -1.0
        sites, masses = [], []
        for e in range(a.examples):
            w, pos = int(sc["top_win"][L, p, e, n]), int(sc["top_pos"][L, p, e, n])
            if w < 0:
                continue
            y, rows, pm = F.site_contributions(model, rec, toks[w].astype(np.int64), L, n, pos, sign)
            if rows:
                sites.append(rows)
                masses.append(pm)
        if not sites:
            continue
        agg = F.pole_feeders(sites, masses)
        pos_items = sorted(((v, k) for k, v in agg.items() if v > 0), reverse=True)
        mass = float(np.mean(masses))
        share = lambda ks: sum(v for v, k in pos_items[:ks]) / max(mass, 1e-9)
        bands = [sum(v for v, k in pos_items if k[3] == b) / max(mass, 1e-9) for b in range(len(F.BANDS))]
        chars = {"assigned": 0.0, "held": 0.0}
        for v, k in pos_items:
            c = char.get(k[:3])
            if c:
                chars[c] += v / max(mass, 1e-9)
        rec_ = {"pole": f"bonsai1/L{L}/N{n}{'+-'[p]}", "examples": len(sites), "top1": share(1), "top3": share(3), "top10": share(10),
                "bands": bands, "assigned": chars["assigned"], "held": chars["held"],
                "top": [{"feeder": f"L{k[0]}/N{k[1]}{k[2]}", "band": F.BAND_NAMES[k[3]], "share": round(v / max(mass, 1e-9), 3),
                         "char": char.get(k[:3]), "label": label_of(k[:3], ix)} for v, k in pos_items[:5]]}
        out.append(rec_)
        if (j + 1) % 10 == 0:
            print(f"  {j + 1}/{len(pick)} poles, {(time.time() - t0) / 60:.1f} min", flush=True)
    ct._gpu_idle("compose pilot")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    summarize(out)


def summarize(out):
    q = lambda k: np.percentile([r[k] for r in out], [25, 50, 75])
    print(f"\n{len(out)} poles. Share of positive attribution mass (quartiles 25/50/75%):")
    for k, name in (("top1", "top 1 feeder key"), ("top3", "top 3 feeder keys"), ("top10", "top 10 feeder keys"),
                    ("assigned", "on assignment-grade feeders"), ("held", "on feeders with a held (partial) test")):
        print(f"  {name:38s} " + "  ".join(f"{x:6.1%}" for x in q(k)))
    B = np.array([r["bands"] for r in out])
    print("  by offset (median share): " + ", ".join(f"{n} {m:.1%}" for n, m in zip(F.BAND_NAMES, np.median(B, 0))))
    print(f"  poles whose top 3 keys hold >= 50% of the mass: {np.mean([r['top3'] >= .5 for r in out]):.0%}; "
          f"with >= 50% on characterized feeders: {np.mean([r['assigned'] + r['held'] >= .5 for r in out]):.0%}")


if __name__ == "__main__":
    main()
