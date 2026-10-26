"""float16 parity study: does BONSAI_DTYPE=float16 reproduce the bfloat16 survey readings and test outcomes?

Free Colab T4s have float16 tensor cores but none for bfloat16, so float16 would run missions ~5x faster there
(docs/COLAB.md). This re-scores a stratified sample of registered quick-8+8 tests exactly as `play run` does
(TextEvaluator: 3 preambles, max over the text's tokens, fraction of the pole's corpus max), once per dtype in
separate processes, and compares. Nothing is written under graph/; output goes to data/fp16/ (gitignored).

  python scripts/fp16_parity.py sample [--n 150]               -> data/fp16/sample.json
  BONSAI_DTYPE=bfloat16 python scripts/fp16_parity.py score    -> data/fp16/score_bfloat16.json (+ _units.npz)
  BONSAI_DTYPE=float16  python scripts/fp16_parity.py score    -> data/fp16/score_float16.json  (+ _units.npz)
  python scripts/fp16_parity.py compare                        -> data/fp16/report.json, printed summary

Each score process takes the machine's batch slot and one GPU turn per forward (bonsai_interp.gpu_turn). Besides the
test sentences it records, per layer, the largest |value| and any inf/NaN in the attention output, MLP activations,
MLP output and residual stream; a full forward (logits) on a subset; every neuron's reading on 16 texts; and timings.
docs/FLOAT16.md has the results.
"""
import argparse
import glob
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get("FP16_OUT") or ROOT / "data" / "fp16")
TAG = "fp16-parity"


def sample(a):
    from bonsai_interp.graph.protocols import _quick_8_8_v2
    pool = defaultdict(list)  # (layer, outcome@2) -> tests
    for f in sorted(glob.glob(str(ROOT / "graph/contributions/bonsai1/*/*.tests.json"))):
        r = Path(f.replace(".tests.json", ".json"))
        if not r.exists():
            continue
        t, res = json.loads(Path(f).read_text(encoding="utf-8")), json.loads(r.read_text(encoding="utf-8"))
        for k, (nb, test) in enumerate(zip(t.get("neighbours", []), res.get("tests", []))):
            if test.get("protocol") not in ("quick-8+8@2", "quick-8+8@3") or "per_sentence" not in test:
                continue
            if len(nb["fire"]) != 8 or len(nb["quiet"]) != 8:
                continue
            L = int(nb["unit"][1:].split("/N")[0])
            pool[(L, _quick_8_8_v2(test))].append({"file": Path(f).relative_to(ROOT).as_posix(), "k": k, "neuron": t["neuron"],
                                                   "unit": nb["unit"], "pole": nb["pole"], "fire": nb["fire"], "quiet": nb["quiet"],
                                                   "recorded": test["per_sentence"], "protocol": test["protocol"],
                                                   "outcome2": _quick_8_8_v2(test)})
    rng = random.Random(a.seed)
    for v in pool.values():
        rng.shuffle(v)
    picked, seen = [], set()
    for oc in ("held", "failed"):  # round-robin over layers, one distinct pole at a time
        want, layers, i = a.n // 2, list(range(36)), 0
        got = 0
        while got < want and any(pool[(L, oc)] for L in layers):
            L = layers[i % 36]
            i += 1
            while pool[(L, oc)]:
                x = pool[(L, oc)].pop()
                if x["unit"] + x["pole"] not in seen:
                    seen.add(x["unit"] + x["pole"])
                    picked.append(x)
                    got += 1
                    break
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sample.json").write_text(json.dumps(picked, ensure_ascii=False, indent=0), encoding="utf-8")
    by = defaultdict(int)
    for x in picked:
        by[x["outcome2"]] += 1
    print(f"{len(picked)} tests ({dict(by)}) over {len({x['unit'].split('/')[0] for x in picked})} layers -> {OUT / 'sample.json'}")


class Stats:
    """Per-layer max |value| and inf/NaN counts over real (unpadded) positions, from forward hooks."""

    def __init__(self, model):
        self.mask, self.hooks = None, []
        self.d = defaultdict(lambda: {"max": 0.0, "nonfinite": 0})
        self.hooks.append(model.model.register_forward_pre_hook(self._pre, with_kwargs=True))
        for i, layer in enumerate(model.model.layers):
            self.hooks.append(layer.self_attn.register_forward_hook(self._mk(i, "attn")))
            self.hooks.append(layer.mlp.down_proj.register_forward_pre_hook(self._mk_in(i, "mlp_act")))
            self.hooks.append(layer.mlp.down_proj.register_forward_hook(self._mk(i, "mlp_out")))
            self.hooks.append(layer.register_forward_hook(self._mk(i, "resid")))
        self.hooks.append(model.model.norm.register_forward_hook(self._mk(99, "final_norm")))

    def _pre(self, mod, args, kwargs):
        self.mask = kwargs.get("attention_mask")

    def _note(self, i, kind, x):
        import torch
        if isinstance(x, (tuple, list)):
            x = x[0]
        if not torch.is_tensor(x) or x.dim() != 3:
            return
        m = self.mask.bool() if self.mask is not None and self.mask.shape == x.shape[:2] else torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        v = x[m]
        e = self.d[f"{i}:{kind}"]
        fin = torch.isfinite(v)
        e["nonfinite"] += int((~fin).sum())
        e["max"] = max(e["max"], float(torch.where(fin, v.abs(), torch.zeros_like(v)).max().float()))

    def _mk(self, i, kind):
        return lambda mod, args, out: self._note(i, kind, out)

    def _mk_in(self, i, kind):
        return lambda mod, args: self._note(i, kind, args[0])

    def remove(self):
        for h in self.hooks:
            h.remove()


def score(a):
    import torch
    from bonsai_interp import explore, gpu_turn
    from bonsai_interp.env import DEVICE, DTYPE, describe
    from bonsai_interp.model import StopForward, load
    from bonsai_interp.probe import textinv

    dt_name = str(DTYPE).removeprefix("torch.")
    tests = json.loads((OUT / "sample.json").read_text(encoding="utf-8"))
    waited = gpu_turn.hold_batch_slot(TAG)
    print(f"batch slot after {waited:.0f} s; {describe()}", flush=True)
    t0 = time.time()
    with gpu_turn.turn(TAG, 1):
        model, tok, rec = load("bonsai1")
        torch.cuda.synchronize()
    load_s = time.time() - t0
    tok.padding_side = "right"
    scan = explore.scan("bonsai1")
    top = scan["top_val"]
    timing = {"load_s": load_s, "score_gpu_s": 0.0, "score_forwards": 0, "score_tokens": 0}

    class Ev(textinv.TextEvaluator):
        def _run(self, texts, pre_len):
            with gpu_turn.turn(TAG, 1):
                s = time.perf_counter()
                out = super()._run(texts, pre_len)
                torch.cuda.synchronize()
                timing["score_gpu_s"] += time.perf_counter() - s
            timing["score_forwards"] += 1
            timing["score_tokens"] += sum(len(x) for x in self.tok(texts).input_ids)
            return out

        def __call__(self, texts):  # textinv.TextEvaluator.__call__, keeping each preamble's reading
            todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
            for s in range(0, len(todo), self.batch):
                chunk = todo[s:s + self.batch]
                per_pre = [self._run([p + t for t in chunk], k) for p, k in zip(self.preambles, self.pre_len)]
                for i, t in enumerate(chunk):
                    rs = [r[i] for r in per_pre]
                    self.cache[t] = {"act": float(np.mean([r["act"] for r in rs])), "raw": [r["act"] for r in rs],
                                     "opp": float(np.mean([r["opp"] for r in rs])), "peak_token": rs[0]["peak_token"]}
            return [self.cache[t] for t in texts]

    stats = Stats(model)
    rows = []
    for j, x in enumerate(tests):
        L, n = map(int, x["unit"][1:].split("/N"))
        pole = 0 if x["pole"] == "+" else 1
        pmax, omax = float(top[L, pole, 0, n]), float(top[L, 1 - pole, 0, n])
        mk = lambda: Ev(model, tok, rec, L, n, pmax, omax, 1.0 if pole == 0 else -1.0)
        ev = mk()
        fire, quiet = ev(x["fire"]), ev(x["quiet"])  # as play run: fire then near misses, one batch of 8 each
        alt = mk()(x["fire"] + x["quiet"])  # one batch of 16: a different padding shape (the batch-shape noise floor)
        rows.append({"fire": fire, "quiet": quiet, "alt": [r["act"] for r in alt]})
        if j % 25 == 0:
            print(f"  [{j + 1}/{len(tests)}] {x['neuron']}", flush=True)
    score_stats = dict(stats.d)

    # Full forward with logits on the first 40 tests' fire sentences (first preamble): late-layer overflow and outputs.
    stats.d.clear()
    rec.mode, rec.stop_at = None, None
    texts = [textinv.PREAMBLES[0] + s for x in tests[:40] for s in x["fire"]]
    full = []
    for s in range(0, len(texts), 8):
        enc = tok(texts[s:s + 8], return_tensors="pt", padding=True).to(DEVICE)
        with gpu_turn.turn(TAG, 1), torch.no_grad():
            lp = torch.log_softmax(model(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits.float(), -1)
            torch.cuda.synchronize()
        for i in range(lp.shape[0]):
            T = int(enc.attention_mask[i].sum())
            v, ix = lp[i, T - 1].topk(10)
            full.append({"top1": lp[i, :T].argmax(-1).tolist(), "last_ids": ix.tolist(), "last_lp": v.tolist(),
                         "nonfinite_logits": int((~torch.isfinite(lp[i, :T])).sum())})
        del lp
    full_stats = dict(stats.d)
    stats.remove()

    # Every neuron's reading (max and min over the probe text's tokens, as fractions of the + and - corpus max) on
    # 16 texts, at batch 4 and at batch 1, so the dtype difference can be set against bfloat16's own batch-shape noise.
    unit_texts = [tests[k]["fire"][0] for k in range(0, len(tests), max(1, len(tests) // 16))][:16]
    pre, pre_len = textinv.PREAMBLES[0], len(tok(textinv.PREAMBLES[0]).input_ids)
    denom = np.maximum(top[:, :, 0, :], 1e-6)  # (36, 2, 12288)
    units = {}
    for bs in (4, 1):
        out = np.zeros((len(unit_texts), 36, 2, 12288), np.float32)
        for s in range(0, len(unit_texts), bs):
            enc = tok([pre + t for t in unit_texts[s:s + bs]], return_tensors="pt", padding=True).to(DEVICE)
            rec.mode, rec.layers, rec.stop_at = "act", None, 35
            with gpu_turn.turn(TAG, 1), torch.no_grad():
                try:
                    model.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
                except StopForward:
                    pass
                m = enc.attention_mask.bool().clone()
                m[:, :pre_len] = False
                for L in range(36):
                    act = rec.store[L].float()
                    hi = torch.where(m[..., None], act, torch.full_like(act, -1e9)).max(1).values
                    lo = torch.where(m[..., None], -act, torch.full_like(act, -1e9)).max(1).values
                    out[s:s + bs, L, 0], out[s:s + bs, L, 1] = hi.cpu().numpy(), lo.cpu().numpy()
                torch.cuda.synchronize()
            rec.mode, rec.stop_at = None, None
            rec.clear()
        units[f"b{bs}"] = out / denom[None]

    # Throughput: 64 test sentences with the first preamble, batch 16, full depth, no hooks, after a warm-up.
    bench = [textinv.PREAMBLES[0] + s for x in tests[:8] for s in x["fire"] + x["quiet"]][:64]
    encs = [tok(bench[s:s + 16], return_tensors="pt", padding=True).to(DEVICE) for s in range(0, 64, 16)]
    ntok = sum(int(e.attention_mask.sum()) for e in encs)
    reps = []
    for r in range(6):
        with gpu_turn.turn(TAG, 1), torch.no_grad():
            s = time.perf_counter()
            for e in encs:
                model.model(input_ids=e.input_ids, attention_mask=e.attention_mask)
            torch.cuda.synchronize()
            reps.append(time.perf_counter() - s)
    timing.update({"bench_tokens": ntok, "bench_s": reps, "bench_tok_per_s": ntok / min(reps[1:]),
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30, "device": describe()})
    np.savez_compressed(OUT / f"score_{dt_name}_units.npz", **units)
    (OUT / f"score_{dt_name}.json").write_text(json.dumps({"dtype": dt_name, "rows": rows, "score_stats": score_stats,
                                                            "full_stats": full_stats, "full": full, "unit_texts": unit_texts,
                                                            "timing": timing}), encoding="utf-8")
    print(f"done {dt_name}: {timing}", flush=True)


def long(a):
    """Corpus windows of 256 tokens (the length of band scans and doctor's speed check): residual headroom over a longer
    context, inf/NaN, next-token agreement, and every neuron's max/min over each window (as fractions of corpus max)."""
    import torch
    from bonsai_interp import explore, gpu_turn
    from bonsai_interp.env import DEVICE, DTYPE, describe
    from bonsai_interp.model import load

    dt_name = str(DTYPE).removeprefix("torch.")
    gpu_turn.hold_batch_slot(TAG)
    heads = DTYPE != torch.float32  # a float32 reference skips lm_head (2.5 GB) to stay within BONSAI_GPU_FRACTION
    with gpu_turn.turn(TAG, 1):
        model, tok, rec = load("bonsai1", lm_head=heads)
    top = explore.scan("bonsai1")["top_val"]
    denom = np.maximum(top[:, :, 0, :], 1e-6)
    corpus = explore.corpus()
    rows = np.random.default_rng(1).choice(len(corpus), a.windows, replace=False)
    x = torch.from_numpy(corpus[np.sort(rows), :256].astype("int64")).to(DEVICE)
    stats = Stats(model)
    units = np.zeros((len(x), 36, 2, 12288), np.float32)
    top1 = []
    bs = a.batch
    for s in range(0, len(x), bs):
        xb = x[s:s + bs]
        rec.mode, rec.layers, rec.stop_at = "act", None, None
        with gpu_turn.turn(TAG, 1), torch.no_grad():
            if heads:
                logits = model(input_ids=xb, attention_mask=torch.ones_like(xb)).logits
            else:
                logits = torch.zeros(xb.shape + (1,), device=DEVICE)
                model.model(input_ids=xb, attention_mask=torch.ones_like(xb))
            top1.append(logits.argmax(-1).cpu().numpy())
            nonfin = int((~torch.isfinite(logits)).sum())
            for L in range(36):
                act = rec.store[L].float()
                units[s:s + bs, L, 0], units[s:s + bs, L, 1] = act.max(1).values.cpu().numpy(), (-act).max(1).values.cpu().numpy()
            torch.cuda.synchronize()
        rec.mode = None
        rec.clear()
        stats.d[f"99:nonfinite_logits"]["nonfinite"] += nonfin
        del logits
    np.savez_compressed(OUT / f"long_{dt_name}{'' if bs == 2 else f'_b{bs}'}.npz", units=units / denom[None], top1=np.concatenate(top1))
    (OUT / f"long_{dt_name}{'' if bs == 2 else f'_b{bs}'}.json").write_text(json.dumps({"dtype": dt_name, "stats": dict(stats.d), "device": describe()}), encoding="utf-8")
    worst = sorted(((v["max"], k) for k, v in stats.d.items()), reverse=True)[:5]
    print(f"done {dt_name}: nonfinite {sum(v['nonfinite'] for v in stats.d.values())}, largest {worst}", flush=True)


def compare_long(a):
    B, H = (np.load(OUT / f"long_{d}.npz") for d in ("bfloat16", "float16"))
    SB, SH = (json.loads((OUT / f"long_{d}.json").read_text(encoding="utf-8"))["stats"] for d in ("bfloat16", "float16"))
    ub, uh = B["units"], H["units"]
    from bonsai_interp import explore
    ok = explore.scan("bonsai1")["top_val"][:, :, 0, :] >= 0.05
    ub, uh = np.where(ok[None], ub, 0.0), np.where(ok[None], uh, 0.0)
    fin = np.isfinite(uh)
    live = ((np.abs(ub) >= 0.2) | (np.abs(uh) >= 0.2)) & fin
    rep = {"windows": int(ub.shape[0]), "fp16_vs_bf16_units": _err(uh[fin], ub[fin]), "where_either>=0.2": _err(uh[live], ub[live]),
           "units_nonfinite_fp16": int((~fin).sum()),
           "max_err_by_layer": {L: round(float(np.abs(uh[:, L] - ub[:, L])[fin[:, L]].max()), 4) for L in range(36)},
           "top1_agree": float((B["top1"] == H["top1"]).mean()),
           "fp16_nonfinite": {k: v["nonfinite"] for k, v in SH.items() if v["nonfinite"]},
           "bf16_nonfinite": {k: v["nonfinite"] for k, v in SB.items() if v["nonfinite"]},
           "largest_bf16": sorted(((v["max"], k) for k, v in SB.items()), reverse=True)[:6],
           "largest_fp16": sorted(((v["max"], k) for k, v in SH.items()), reverse=True)[:6]}
    (OUT / "report_long.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=float))


def _err(a, b):
    d = np.abs(np.asarray(a, float) - np.asarray(b, float))
    return {"n": int(d.size), "max": float(d.max()), "p999": float(np.quantile(d, 0.999)), "p99": float(np.quantile(d, 0.99)),
            "mean": float(d.mean())}


def compare(a):
    from bonsai_interp.graph.protocols import _quick_8_8_v2, v2_stats
    tests = json.loads((OUT / "sample.json").read_text(encoding="utf-8"))
    B, H = (json.loads((OUT / f"score_{d}.json").read_text(encoding="utf-8")) for d in ("bfloat16", "float16"))
    rep = {}

    def acts(R):  # per test: fire 0-7 then near misses 0-7, the order play run records
        return np.array([r["act"] for row in R["rows"] for k in ("fire", "quiet") for r in row[k]])

    def raws(R):
        return np.array([v for row in R["rows"] for k in ("fire", "quiet") for r in row[k] for v in r["raw"]])

    b, h = acts(B), acts(H)
    recd = np.array([r["act"] for x in tests for r in x["recorded"]])
    b_alt = np.array([v for row in B["rows"] for v in row["alt"]])
    h_alt = np.array([v for row in H["rows"] for v in row["alt"]])
    rel = np.abs(h - b) / np.maximum(np.abs(b), 0.05)
    rep["sentence_act"] = {
        "fp16_vs_bf16": _err(h, b), "fp16_vs_bf16_relative(floor 0.05)": _err(rel, 0 * rel),
        "fp16_vs_bf16_per_preamble": _err(raws(H), raws(B)),
        "bf16_batch_shape_noise": _err(b_alt, b), "fp16_batch_shape_noise": _err(h_alt, h),
        "bf16_rerun_vs_recorded(3dp)": _err(b, recd), "fp16_vs_recorded(3dp)": _err(h, recd),
        "peak_token_agree": float(np.mean([rb["peak_token"] == rh["peak_token"] for xb, xh in zip(B["rows"], H["rows"])
                                           for k in ("fire", "quiet") for rb, rh in zip(xb[k], xh[k])])),
        "nonfinite_acts": int((~np.isfinite(h)).sum())}

    def test_of(row):
        return {"per_sentence": [{"set": k, "act": round(r["act"], 3)} for k in ("fire", "quiet") for r in row[k]]}

    agree, flips, per_test = 0, [], []
    for x, rb, rh in zip(tests, B["rows"], H["rows"]):
        tb, th = test_of(rb), test_of(rh)
        ob, oh = _quick_8_8_v2(tb), _quick_8_8_v2(th)
        sb, sh = v2_stats(tb), v2_stats(th)
        L = int(x["unit"][1:].split("/N")[0])
        per_test.append({"neuron": x["neuron"], "L": L, "recorded": x["outcome2"], "bf16": ob, "fp16": oh,
                         "bf16_stats": sb, "fp16_stats": sh})
        agree += ob == oh
        if ob != oh:
            flips.append(per_test[-1])
    rep["outcomes"] = {"n": len(tests), "fp16_agrees_with_bf16": agree, "flips": flips,
                       "bf16_agrees_with_recorded": sum(p["bf16"] == p["recorded"] for p in per_test),
                       "fp16_agrees_with_recorded": sum(p["fp16"] == p["recorded"] for p in per_test),
                       "bf16_vs_recorded_flips": [p for p in per_test if p["bf16"] != p["recorded"]],
                       "stat_diff": {"auroc": _err([p["fp16_stats"][0] for p in per_test], [p["bf16_stats"][0] for p in per_test]),
                                     "median_fire": _err([p["fp16_stats"][1] for p in per_test], [p["bf16_stats"][1] for p in per_test])}}
    from bonsai_interp.graph.replicate import verdict  # CI's spot-replication rule, on 3-decimal recorded values
    vs = defaultdict(int)
    for p, rb, rh in zip(per_test, B["rows"], H["rows"]):
        d = [abs(round(a1["act"], 3) - round(a2["act"], 3)) for k in ("fire", "quiet") for a1, a2 in zip(rb[k], rh[k])]
        vs[verdict(d, 16, p["bf16"] == p["fp16"])] += 1
    rep["outcomes"]["ci_replication_verdict_fp16_vs_bf16"] = dict(vs)
    by_L, noise_L = defaultdict(list), defaultdict(list)
    for x, rb, rh in zip(tests, B["rows"], H["rows"]):
        L = int(x["unit"][1:].split("/N")[0])
        by_L[L] += [abs(a1["act"] - a2["act"]) for k in ("fire", "quiet") for a1, a2 in zip(rb[k], rh[k])]
        noise_L[L] += [abs(a1["act"] - v) for a1, v in zip(rb["fire"] + rb["quiet"], rb["alt"])]
    rep["sentence_err_by_layer_max"] = {L: [round(max(v), 4), round(max(noise_L[L]), 4)] for L, v in sorted(by_L.items())}
    rep["sentence_err_by_layer_max_note"] = "[fp16 vs bf16, bf16 batch-shape noise]"

    def stat_table(R, key):
        return {k: v for k, v in sorted(R[key].items(), key=lambda kv: (int(kv[0].split(":")[0]), kv[0]))}
    for key in ("score_stats", "full_stats"):
        hs, bs = stat_table(H, key), stat_table(B, key)
        rep[key] = {"fp16_nonfinite": {k: v["nonfinite"] for k, v in hs.items() if v["nonfinite"]},
                    "bf16_nonfinite": {k: v["nonfinite"] for k, v in bs.items() if v["nonfinite"]},
                    "largest_bf16": sorted(((v["max"], k) for k, v in bs.items()), reverse=True)[:8],
                    "largest_fp16": sorted(((v["max"], k) for k, v in hs.items()), reverse=True)[:8],
                    "max_resid_by_layer_bf16": {k: round(v["max"], 1) for k, v in bs.items() if k.endswith("resid")},
                    "headroom_vs_65504": 65504 / max(v["max"] for v in hs.values())}
    t1 = sum(sum(a1 == a2 for a1, a2 in zip(fb["top1"], fh["top1"])) for fb, fh in zip(B["full"], H["full"]))
    tn = sum(len(fb["top1"]) for fb in B["full"])
    lpd = []
    for fb, fh in zip(B["full"], H["full"]):
        hm = dict(zip(fh["last_ids"], fh["last_lp"]))
        lpd += [abs(v - hm[i]) for i, v in zip(fb["last_ids"][:5], fb["last_lp"][:5]) if i in hm]
    rep["logits"] = {"positions": tn, "top1_agree": t1 / tn, "last_top1_agree": float(np.mean([fb["last_ids"][0] == fh["last_ids"][0] for fb, fh in zip(B["full"], H["full"])])),
                     "last_top5_logprob_diff": _err(lpd, np.zeros(len(lpd))), "fp16_nonfinite_logits": sum(f["nonfinite_logits"] for f in H["full"])}
    ub, uh = (dict(np.load(OUT / f"score_{d}_units.npz")) for d in ("bfloat16", "float16"))
    from bonsai_interp import explore
    ok = explore.scan("bonsai1")["top_val"][:, :, 0, :] >= 0.05  # 5 pole-cells have a corpus max under 0.05: fractions of ~0 are meaningless
    for u in (ub, uh):
        for k in u:
            u[k] = np.where(ok[None], u[k], 0.0)
    live = (np.abs(ub["b4"]) >= 0.2) | (np.abs(uh["b4"]) >= 0.2)
    rep["all_units"] = {"texts": len(B["unit_texts"]), "fp16_vs_bf16": _err(uh["b4"], ub["b4"]),
                        "bf16_batch_shape_noise": _err(ub["b1"], ub["b4"]), "fp16_batch_shape_noise": _err(uh["b1"], uh["b4"]),
                        "fp16_vs_bf16_where_either>=0.2": _err(uh["b4"][live], ub["b4"][live]),
                        "bf16_noise_where_either>=0.2": _err(ub["b1"][live], ub["b4"][live]),
                        "cross_0.3_fp16_vs_bf16": int(((uh["b4"] >= 0.3) != (ub["b4"] >= 0.3)).sum()),
                        "cross_0.3_bf16_noise": int(((ub["b1"] >= 0.3) != (ub["b4"] >= 0.3)).sum()),
                        "nonfinite_fp16": int((~np.isfinite(uh["b4"])).sum())}
    du = np.abs(uh["b4"] - ub["b4"])
    dn = np.abs(ub["b1"] - ub["b4"])
    rep["all_units"]["max_err_by_layer"] = {L: [round(float(du[:, L].max()), 4), round(float(dn[:, L].max()), 4)] for L in range(36)}
    rep["all_units"]["max_err_by_layer_note"] = "[fp16 vs bf16, bf16 batch-shape noise]"
    rep["timing"] = {d: R["timing"] for d, R in (("bfloat16", B), ("float16", H))}
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("score_stats", "full_stats")}, indent=1, default=float)[:20000])
    for key in ("score_stats", "full_stats"):
        print(key, json.dumps({k: v for k, v in rep[key].items() if k != "max_resid_by_layer_bf16"}, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=150)
    s.add_argument("--seed", type=int, default=0)
    sub.add_parser("score")
    sub.add_parser("compare")
    s = sub.add_parser("long")
    s.add_argument("--windows", type=int, default=32)
    s.add_argument("--batch", type=int, default=2)
    sub.add_parser("compare-long")
    a = ap.parse_args()
    {"sample": sample, "score": score, "compare": compare, "long": long, "compare-long": compare_long}[a.cmd](a)
