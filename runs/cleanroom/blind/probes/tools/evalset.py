"""Evaluate one or more sentence files on L9/N3688 (both poles) with the standard TextEvaluator.

  python probes/tools/evalset.py IN.json [IN2.json ...] --out OUT.json

Each input is {"name": ..., "sentences": [{"text": ..., (any extra fields)}]} or a plain list of strings.
Prints act (+ pole), opp (- pole), and the peak token of the + pole, and saves rows to OUT.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402

from mono.model import load  # noqa: E402
from mono.probe import textinv  # noqa: E402

L, N = 9, 3688
PMAX, NMAX = 7.65625, 19.0


def peak_neg(ev, texts):
    """Peak token of the - pole: evaluate with the sign flipped (separate cache)."""
    ev2 = textinv.TextEvaluator(ev.model, ev.tok, ev.rec, L, N, NMAX, PMAX, -1.0)
    return ev2(texts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.8)
    model, tok, rec = load("bonsai1")
    tok.padding_side = "right"
    ev = textinv.TextEvaluator(model, tok, rec, L, N, PMAX, NMAX, 1.0)
    results = {}
    for path in a.inputs:
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        sents = spec["sentences"] if isinstance(spec, dict) else spec
        sents = [s if isinstance(s, dict) else {"text": s} for s in sents]
        texts = [s["text"] for s in sents]
        rp, rn = ev(texts), peak_neg(ev, texts)
        rows = [{**s, "pos": round(p["act"], 3), "neg": round(q["act"], 3), "peak_pos": p["peak_token"],
                 "peak_neg": q["peak_token"]} for s, p, q in zip(sents, rp, rn)]
        results[path] = rows
        print(f"\n######## {path}")
        for r in rows:
            tag = " ".join(f"{k}={r[k]}" for k in ("group", "predict") if k in r)
            print(f"  +{r['pos']:.2f} [{r['peak_pos']!r:<12}] -{r['neg']:.2f} [{r['peak_neg']!r:<12}] {tag:<22} {r['text'][:85]}")
    Path(a.out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
