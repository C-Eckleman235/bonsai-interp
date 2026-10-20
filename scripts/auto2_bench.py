"""Size auto-token@2's GPU footprint before a run: throughput and this process's peak memory per setting.

  PYTHONPATH=. python scripts/auto2_bench.py [--quick]

A: the corpus passes (1,024-token windows, forward to a layer): batch x SDPA backend at layers 20 and 35.
B: scoring (preamble + 48-token snippets, read at the last token): full vs preamble-cached x batch, at layers 4 and 20,
   on real auto tests (L4's and L20's tests files).
The model is loaded through context_tier's gpu_turn path, so the benchmark yields to the stream like any stage.
Peak memory is torch.cuda.max_memory_allocated for this process (weights included), reset per setting.
"""
import copy
import gzip
import json
import sys
import time

import numpy as np
import torch

from bonsai_interp import explore
from bonsai_interp.auto import context_tier as ct
from bonsai_interp.env import DEVICE, limit_memory
from bonsai_interp.model import StopForward
from bonsai_interp.probe.textinv import PREAMBLES

quick = "--quick" in sys.argv
limit_memory(0.8)
model, tok, rec = ct._gpu_load()
tok.padding_side = "right"
gb = lambda: torch.cuda.max_memory_allocated() / 2**30


def stop_at(L, **kw):
    rec.mode, rec.layers, rec.stop_at = "act", {L}, L
    try:
        with torch.no_grad():
            model.model(**kw)
    except StopForward:
        pass
    finally:
        rec.mode, rec.stop_at = None, None
    rec.clear()


def timed(fn, n):
    fn()  # warm-up
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n, gb()


print(f"weights resident: {torch.cuda.memory_allocated() / 2**30:.2f} GB", flush=True)
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402
toks = explore.corpus()[:16].astype(np.int64)
for L in ((20,) if quick else (20, 35)):
    for name, be in (("default", None), ("efficient", SDPBackend.EFFICIENT_ATTENTION), ("flash", SDPBackend.FLASH_ATTENTION)):
        for B in (1, 2, 4, 8):
            x = torch.from_numpy(toks[:B]).to(DEVICE)
            try:
                with (sdpa_kernel([be]) if be is not None else torch.no_grad()):
                    dt, peak = timed(lambda: stop_at(L, input_ids=x), 3)
                print(f"A layer {L:2d} sdpa={name:9s} batch {B}: {B * 1024 / dt:7,.0f} tok/s  peak {peak:.2f} GB", flush=True)
            except Exception as e:
                print(f"A layer {L:2d} sdpa={name:9s} batch {B}: {type(e).__name__} {str(e)[:70]}", flush=True)
                torch.cuda.empty_cache()
                break

for L in (4, 20):
    rows = [json.loads(x) for x in gzip.open(ct._paths(L)["tests"], "rb").read().decode("utf-8").splitlines()]
    texts = sorted({t for r in rows if r["class"] == "tested" for t in r["fire"] + r["quiet"]})[:512]
    pre = PREAMBLES[0]
    pre_ids = tok(pre).input_ids
    joined = [tok(pre + t).input_ids for t in texts]
    texts_c = [j[len(pre_ids):] for j in joined if j[:len(pre_ids)] == pre_ids]
    for B in (8, 16, 32, 64):
        chunk_full = [pre + t for t in texts[:B]]

        def full():
            enc = tok(chunk_full, return_tensors="pt", padding=True).to(DEVICE)
            stop_at(L, input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        from transformers import DynamicCache
        cache = DynamicCache()
        rec.stop_at = L
        try:
            with torch.no_grad():
                model.model(input_ids=torch.tensor([pre_ids], device=DEVICE), past_key_values=cache, use_cache=True)
        except StopForward:
            pass
        finally:
            rec.stop_at = None
        suf = texts_c[:B]
        T = max(len(s) for s in suf)
        ids = torch.tensor([s + [0] * (T - len(s)) for s in suf], device=DEVICE)
        m = torch.tensor([[1] * (len(pre_ids) + len(s)) + [0] * (T - len(s)) for s in suf], device=DEVICE)

        def cached():
            c = copy.deepcopy(cache)
            c.batch_repeat_interleave(len(suf))
            stop_at(L, input_ids=ids, attention_mask=m, past_key_values=c, use_cache=True)
        for label, fn in (("full", full), ("cached", cached)):
            try:
                dt, peak = timed(fn, 3 if quick else 6)
                print(f"B layer {L:2d} {label:6s} batch {B:2d}: {B / dt:6.1f} snippets/s  peak {peak:.2f} GB", flush=True)
            except Exception as e:
                print(f"B layer {L:2d} {label:6s} batch {B:2d}: {type(e).__name__} {str(e)[:70]}", flush=True)
                torch.cuda.empty_cache()
ct._gpu_idle("benchmark")
print(f"gpu_turn: yielded {ct.YIELD['seconds']:.1f} s during the benchmark")
