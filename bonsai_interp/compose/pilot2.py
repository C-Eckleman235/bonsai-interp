"""Pilot, part 2: does knocking out a pole's top attributed feeders, or a few context tokens, remove its activation?

  python -m bonsai_interp.compose.pilot2 --poles 60 --layers 6-11 --examples 4

Per strong example of each sampled pole:
  - attribution: the top terms (layer, neuron, offset) by signed contribution (feeders.site_contributions);
  - ablation: the share of the activation lost when the top 3 / top 10 positive terms are zeroed at their positions;
  - occlusion: the share lost when the token k back is replaced by a filler (k = 0..31), and how many of the most
    damaging offsets it takes to lose half (attention handled as "which earlier tokens does the pole need").
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from bonsai_interp.compose import feeders as F

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poles", type=int, default=60)
    ap.add_argument("--layers", default="6-11")
    ap.add_argument("--examples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--ctx", type=int, default=256, help="tokens of context before the read position (128 for deep layers)")
    ap.add_argument("--out", default=str(ROOT / "data" / "compose" / "pilot2.json"))
    a = ap.parse_args()
    from bonsai_interp import explore
    from bonsai_interp.auto import context_tier as ct
    from bonsai_interp.env import limit_memory
    lo, hi = map(int, a.layers.split("-"))
    sc, toks, tk = explore.scan("bonsai1"), explore.corpus(), explore.tokenizer()
    filler = tk(" the", add_special_tokens=False).input_ids[0]
    rng = np.random.default_rng(a.seed)
    cand = []
    for L in range(lo, hi + 1):
        el = ct.eligibility(L)
        for p in (0, 1):
            cand += [(L, int(n), p) for n in np.nonzero(el[p] == "eligible")[0]]
    pick = [cand[i] for i in sorted(rng.choice(len(cand), a.poles, replace=False))]
    limit_memory(0.8)
    model, tok, rec = ct._gpu_load()
    out, t0 = [], time.time()
    for j, (L, n, p) in enumerate(pick):
        ct._stop_requested()
        sign = 1.0 if p == 0 else -1.0
        for e in range(a.examples):
            w, pos = int(sc["top_win"][L, p, e, n]), int(sc["top_pos"][L, p, e, n])
            if w < 0:
                continue
            ids = toks[w].astype(np.int64)
            y, rows, pm = F.site_contributions(model, rec, ids, L, n, pos, sign, max_ctx=a.ctx, keep=64)
            if not rows or y <= 0:
                continue
            posr = [r for r in rows if r[0] > 0]
            drop = {}
            for k in (1, 3, 10):
                ya = F.ablate(model, rec, ids, L, n, pos, sign, [(l, m, off) for _, l, m, off, _ in posr[:k]], max_ctx=a.ctx)
                drop[k] = (y - ya) / y
            ys = F.occlude(model, rec, ids, L, n, pos, sign, filler=filler, max_ctx=a.ctx)
            occ = np.clip((y - ys) / y, -5, 5)
            order = np.argsort(-occ)
            need = int(np.searchsorted(np.cumsum(occ[order]), 0.5) + 1) if occ[order].cumsum()[-1] >= 0.5 else None
            out.append({"pole": f"L{L}/N{n}{'+-'[p]}", "layer": L, "y": y, "abs_mass": float(sum(abs(r[0]) for r in rows)),
                        "top_attr": [round(r[0], 3) for r in posr[:3]], "drop": drop, "occ": [round(float(x), 3) for x in occ],
                        "tokens_for_half": need, "top_offsets": [int(k) for k in order[:3]]})
        if (j + 1) % 10 == 0:
            print(f"  {j + 1}/{len(pick)} poles, {(time.time() - t0) / 60:.1f} min", flush=True)
    ct._gpu_idle("compose pilot2")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    summarize(out)


def summarize(out):
    q = lambda xs: "  ".join(f"{x:6.1%}" for x in np.percentile(xs, [25, 50, 75]))
    print(f"\n{len(out)} examples from {len({r['pole'] for r in out})} poles (quartiles 25/50/75%)")
    for k in ("1", "3", "10"):
        print(f"  activation lost when the top {k:>2} attributed feeders are zeroed: {q([r['drop'][k] if k in r['drop'] else r['drop'][int(k)] for r in out])}")
    occ = np.array([r["occ"][:32] + [0] * (32 - len(r["occ"])) for r in out])
    print(f"  lost when the read token itself is replaced (offset 0):        {q(occ[:, 0])}")
    print(f"  lost when the previous token is replaced (offset 1):           {q(occ[:, 1])}")
    print(f"  lost by the most damaging earlier token (offsets 1-31):        {q(occ[:, 1:].max(1))}")
    need = [r["tokens_for_half"] for r in out]
    print(f"  tokens whose replacement removes half the activation: 1 token {np.mean([x == 1 for x in need]):.0%}, "
          f"<= 2 {np.mean([x is not None and x <= 2 for x in need]):.0%}, <= 3 {np.mean([x is not None and x <= 3 for x in need]):.0%}, "
          f"never (by single-token knockouts) {np.mean([x is None for x in need]):.0%}")


if __name__ == "__main__":
    main()
