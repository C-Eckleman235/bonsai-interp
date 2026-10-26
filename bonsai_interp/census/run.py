"""Drive the census's GPU stages in one process (one batch-slot hold, a gpu_turn tick before every batch).

    python -m bonsai_interp.census.run pilot          # lens check against the HF forward, timings, one calibration
    python -m bonsai_interp.census.run all            # weights -> lens prefix -> links -> steer lens, resumable
    python -m bonsai_interp.census.summary            # CPU: flags, summary, validation (bonsai_interp/census/summary.py)

Outputs (data/census/, never in git; release `census-data`):
    bonsai1.global.npz             null-space spectrum, random-direction nulls, residual norms, steer calibration
    bonsai1_L{L}.weights.npz       out_norm, null_share, logit_gain, freq_r (N,); dla_top (2, N, 20), dla_spike (2, N)
    bonsai1_L{L}.links.npz         rd_* (N, K) readers of each neuron; wrp_* / wrn_* (N, K) writers of the + / − pole
    bonsai1_L{L}.steer.npz         alpha, curve, kl (2, N), top (2, N, 10), dlp (2, N, 10), rand_kl / rand_top1 (2, R)
"""
import argparse
import json
import time

import numpy as np

from bonsai_interp.env import ROOT

OUT = ROOT / "data" / "census"
NAME = "bonsai1"
N_RAND_STEER = 256
STEER_BATCH = 256


def _path(L, kind):
    return OUT / f"{NAME}_L{L}.{kind}.npz"


def _save(path, **arrays):
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def _load():
    from bonsai_interp.auto import context_tier as ct
    from bonsai_interp.env import limit_memory
    limit_memory(0.8)
    return ct._gpu_load(lm_head=True), ct._tick, ct._gpu_idle


def _counts(V):
    from bonsai_interp import explore
    return np.bincount(explore.corpus().ravel().astype(np.int64), minlength=V)


def stage_weights(model, tick, layers):
    from bonsai_interp.census import weights as Wt
    U = Wt.Unembed(model, _counts(model.lm_head.weight.shape[0]))
    rnd = U.random_null(model.config.hidden_size)
    for L in layers:
        if _path(L, "weights").exists():
            continue
        tick()
        t0 = time.time()
        _save(_path(L, "weights"), **Wt.layer_stats(model, U, L))
        print(f"census weights L{L}: {time.time() - t0:.1f} s", flush=True)
    return {"eig": U.eig, "n_seen": U.n_seen, **{f"rand_{k}": v for k, v in rnd.items()}}


def stage_links(model, tick, rB):
    from bonsai_interp.census import connect
    if all(_path(L, "links").exists() for L in range(len(model.model.layers))):
        return
    t0 = time.time()
    res = connect.links(model, rB, tick)
    for L, r in enumerate(res):
        _save(_path(L, "links"), **{f"{pre}_{k}": v for key, pre in (("rd", "rd"), ("wr+", "wrp"), ("wr-", "wrn")) for k, v in r[key].items()})
    print(f"census links: {(time.time() - t0) / 60:.1f} min", flush=True)


def stage_steer(model, lens, tick, layers, seed=0):
    import torch
    for L in layers:
        if _path(L, "steer").exists():
            continue
        t0 = time.time()
        gen = np.random.default_rng(seed + L)
        tick()
        alpha, curve = lens.calibrate(L, gen)
        Wd = model.model.layers[L].mlp.down_proj.weight().float()
        dirs = (Wd / Wd.norm(dim=0, keepdim=True)).T.contiguous()
        del Wd
        N = dirs.shape[0]
        kl = np.zeros((2, N), np.float32)
        top = np.zeros((2, N, 10), np.int32)
        dlp = np.zeros((2, N, 10), np.float16)
        for b0 in range(0, N, STEER_BATCH):
            tick()
            k, i, v = lens.push(L, dirs[b0:b0 + STEER_BATCH], alpha)
            kl[:, b0:b0 + STEER_BATCH], top[:, b0:b0 + STEER_BATCH], dlp[:, b0:b0 + STEER_BATCH] = k, i, v
        del dirs
        r = torch.from_numpy(gen.standard_normal((N_RAND_STEER, model.config.hidden_size)).astype(np.float32))
        r /= r.norm(dim=-1, keepdim=True)
        tick()
        rk, ri, rv = lens.push(L, r, alpha)
        _save(_path(L, "steer"), alpha=np.float32(alpha), curve=np.array(curve, np.float32), kl=kl, top=top, dlp=dlp,
              rand_kl=rk, rand_top1=rv[..., 0].astype(np.float32), rand_top=ri)
        print(f"census steer L{L}: alpha {alpha:.3f}, median KL neuron {np.median(kl):.3f} vs random {np.median(rk):.3f}, "
              f"{time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pilot", "all"])
    ap.add_argument("--layers", default="")
    a = ap.parse_args()
    import torch
    from bonsai_interp.census.lens import Lens, CONTEXTS
    OUT.mkdir(parents=True, exist_ok=True)
    (model, tok, rec), tick, idle = _load()
    nL = len(model.model.layers)
    layers = [int(x) for x in a.layers.split(",")] if a.layers else list(range(nL))
    tick()
    lens = Lens(model, tok)
    err = lens.check()
    print("lens vs HF forward, max |Δ log-prob| over the top 50 tokens per context:", [round(e, 4) for e in err], flush=True)
    hn = lens.hnorm.cpu().numpy()
    d = model.config.hidden_size
    rB = torch.stack([m.float().norm(dim=-1) for m in lens.mid]).mean(0).cpu().numpy() / np.sqrt(d)  # (nL,) RMS before each MLP
    if a.stage == "pilot":
        for L in layers:
            gen = np.random.default_rng(L)
            t0 = time.time()
            alpha, curve = lens.calibrate(L, gen)
            torch.cuda.synchronize()
            t1 = time.time()
            Wd = model.model.layers[L].mlp.down_proj.weight().float()
            dirs = (Wd[:, :STEER_BATCH] / Wd[:, :STEER_BATCH].norm(dim=0, keepdim=True)).T.contiguous()
            k, i, v = lens.push(L, dirs, alpha)
            torch.cuda.synchronize()
            t2 = time.time()
            print(f"L{L}: alpha {alpha:.3f} curve {[(x, round(y, 4)) for x, y in curve]} calib {t1 - t0:.1f} s; "
                  f"one batch of {STEER_BATCH} neurons {t2 - t1:.2f} s -> layer est {(t2 - t1) * 12288 / STEER_BATCH:.0f} s; "
                  f"median KL {np.median(k):.3f}, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
            vb = tok.convert_ids_to_tokens
            for n in range(3):
                print(f"   N{n}+ KL {k[0, n]:.3f}", vb(i[0, n, :6].tolist()), np.round(v[0, n, :6], 2).tolist(), flush=True)
        idle("pilot")
        return
    glob = stage_weights(model, tick, layers)
    torch.cuda.empty_cache()
    tick()
    stage_links(model, tick, rB)
    torch.cuda.empty_cache()
    stage_steer(model, lens, tick, layers)
    meta = {"contexts": list(CONTEXTS), "hnorm": hn, "rms_mid": rB, "lens_check": np.array(err), **glob}
    _save(OUT / f"{NAME}.global.npz", **{k: np.asarray(v) for k, v in meta.items()})
    idle("census")
    print("census done", flush=True)


if __name__ == "__main__":
    main()
