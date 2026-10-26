"""The census's per-pole table, flags, summary and validation (CPU, from data/census/).

    python -m bonsai_interp.census.summary            # writes bonsai1.census.poles.npz and bonsai1.census.summary.json
    python -m bonsai_interp.census.summary pole bonsai1/L35/N123-   # one pole's census entry, tokens decoded

Per pole (arrays (36, 2, N), pole 0 = +, 1 = −). Nulls come from bonsai1.nulls.npz (census/nulls.py): per layer, 1,024
"sign-random" columns with the layer's real 128-group scales and random signs, i.e. what a random 1-bit neuron writes.
    dla_spike      how far the 20th promoted token stands above the vocabulary mean (SD), sweeplib's spike
    dla_pct        share of the layer's sign-random null whose spike is below this pole's (0.5 = typical random)
    freq_r         correlation of the pole's logit effect with log unigram frequency (− pole: −r of the + pole)
    null_share     share of the output direction in the unembedding's null space (the 2 directions below the spectral
                   gap; same for both poles)
    out_norm_z     ||W_down[:, n]|| as a z-score within its layer (same for both poles; set by the 128-neuron group)
    steer_kl       KL(p0 || p) of the calibrated push, mean of the lens contexts
    steer_z        (log KL − mean log KL of random directions) / SD, at the same dose and layer
    steer_agree    share of the steer lens's top 10 tokens that are in the direct effect's top 20
    flags          bit 1 predict, 2 frequency, 4 entropy, 8 steer-strong (definitions in FLAGS)
"""
import gzip
import json
import re
import sys

import numpy as np

from bonsai_interp.census.run import NAME, OUT, _path

N = 12288
FLAGS = {
    "predict": "dla_spike above the layer's sign-random 99.9th percentile and steer_agree >= 0.3 (the push writes the "
               "tokens the weights say it promotes)",
    "frequency": "|freq_r| >= max(0.2, the layer's sign-random 99.9th percentile of |r|)",
    "entropy": "null_share >= 2x the layer's sign-random 99.9th percentile",
    "steer": "steer_z >= 3 (the push moves the next-token distribution far more than a random direction of the same dose)",
}
BITS = {"predict": 1, "frequency": 2, "entropy": 4, "steer": 8}


def build():
    g = dict(np.load(OUT / f"{NAME}.global.npz", allow_pickle=True))
    nL = len(g["rms_mid"])
    nu = np.load(OUT / f"{NAME}.nulls.npz")
    f = {k: np.zeros((nL, 2, N), np.float32) for k in ("dla_spike", "dla_pct", "freq_r", "null_share", "out_norm_z", "steer_kl", "steer_z", "steer_agree")}
    agree_null = []
    thr = {"spike_q999": np.zeros(nL), "freq_abs_q999": np.zeros(nL), "null_q999": np.zeros(nL)}
    per_layer = []
    rng = np.random.default_rng(0)
    for L in range(nL):
        w, s = np.load(_path(L, "weights")), np.load(_path(L, "steer"))
        sn = np.sort(nu[f"sgn{L}_spike"].ravel())
        thr["spike_q999"][L] = np.quantile(sn, 0.999)
        thr["freq_abs_q999"][L] = np.quantile(np.abs(nu[f"sgn{L}_freq_r"]), 0.999)
        thr["null_q999"][L] = np.quantile(nu[f"sgn{L}_null_gap_share"], 0.999)
        f["dla_spike"][L] = w["dla_spike"]
        f["dla_pct"][L] = np.searchsorted(sn, w["dla_spike"].astype(np.float32)) / len(sn)
        f["freq_r"][L] = np.stack([w["freq_r"], -w["freq_r"]])
        f["null_share"][L] = w["null_gap_share"][None]
        z = (w["out_norm"] - w["out_norm"].mean()) / w["out_norm"].std()
        f["out_norm_z"][L] = z[None]
        f["steer_kl"][L] = s["kl"]
        lr = np.log(np.maximum(s["rand_kl"], 1e-12)).ravel()
        f["steer_z"][L] = (np.log(np.maximum(s["kl"], 1e-12)) - lr.mean()) / lr.std()
        S, T = w["dla_top"], s["top"]
        for p in (0, 1):
            f["steer_agree"][L, p] = [np.isin(T[p, n], S[p, n]).mean() for n in range(N)]
            perm = rng.permutation(N)[:2000]  # null: a pole's steer tokens against another neuron's direct effect
            agree_null.append(np.mean([np.isin(T[p, n], S[p, m]).mean() for n, m in zip(range(2000), perm)]))
        per_layer.append({"L": L, "alpha": float(s["alpha"]), "rms_mid": float(g["rms_mid"][L]),
                          "steer_kl_median": float(np.median(s["kl"])), "rand_kl_median": float(np.median(s["rand_kl"])),
                          "rand_kl_q99": float(np.quantile(s["rand_kl"], 0.99)),
                          "neuron_over_rand_q99": float((s["kl"] > np.quantile(s["rand_kl"], 0.99)).mean()),
                          "agree_mean": float(f["steer_agree"][L].mean()), "spike_ge4": float((w["dla_spike"] >= 4).mean()),
                          "spike_over_null_q999": float((w["dla_spike"] > thr["spike_q999"][L]).mean()),
                          "null_spike_median": float(np.median(sn)), "null_spike_q999": float(thr["spike_q999"][L]),
                          "freq_abs_gt_0.2": int((np.abs(w["freq_r"]) > 0.2).sum()), "null_freq_abs_q999": float(thr["freq_abs_q999"][L]),
                          "null_share_max": float(w["null_gap_share"].max()), "null_null_share_q999": float(thr["null_q999"][L]),
                          "out_norm_cv": float(w["out_norm"].std() / w["out_norm"].mean())})
    flags = np.zeros((nL, 2, N), np.uint8)
    T = {k: v[:, None, None] for k, v in thr.items()}
    flags |= np.where((f["dla_spike"] > T["spike_q999"]) & (f["steer_agree"] >= 0.3), BITS["predict"], 0).astype(np.uint8)
    flags |= np.where(np.abs(f["freq_r"]) >= np.maximum(0.2, T["freq_abs_q999"]), BITS["frequency"], 0).astype(np.uint8)
    flags |= np.where(f["null_share"] >= 2 * T["null_q999"], BITS["entropy"], 0).astype(np.uint8)
    flags |= np.where(f["steer_z"] >= 3, BITS["steer"], 0).astype(np.uint8)
    np.savez_compressed(OUT / f"{NAME}.census.poles.npz", flags=flags, **{k: v.astype(np.float16) for k, v in f.items()})
    for r in per_layer:
        L = r["L"]
        r.update({k: int(((flags[L] & b) > 0).sum()) for k, b in BITS.items()})
    rq = {"iso_spike_median": float(np.median(nu["iso_spike"])), "iso_spike_ge4": float((nu["iso_spike"] >= 4).mean()),
          "k_null": int(nu["k_gap"]), "sv_bottom_over_median": (np.sqrt(np.maximum(nu["eig"][:4], 0)) / np.median(np.sqrt(np.maximum(nu["eig"], 0)))).tolist()}
    summ = {"flags": FLAGS, "random": rq, "agree_null": float(np.mean(agree_null)), "per_layer": per_layer,
            "totals": {k: int(((flags & b) > 0).sum()) for k, b in BITS.items()},
            "lens_check": g["lens_check"].tolist(), "contexts": g["contexts"].tolist(),
}
    summ["validation"] = validate(f, flags)
    (OUT / f"{NAME}.census.summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    return summ


def _ids(keys):
    from bonsai_interp.graph import cards
    out = []
    for k in keys:
        _, L, n, p = cards.parse(k)
        out.append((L, 0 if p == "+" else 1, n))
    return out


def _rates(f, flags, idx):
    if not idx:
        return {"n": 0}
    L, p, n = map(np.array, zip(*idx))
    r = {"n": len(idx)}
    r.update({k: round(float(((flags[L, p, n] & b) > 0).mean()), 4) for k, b in BITS.items()})
    r.update({"agree_mean": round(float(f["steer_agree"][L, p, n].mean()), 4), "steer_z_median": round(float(np.median(f["steer_z"][L, p, n])), 3),
              "abs_freq_r_median": round(float(np.median(np.abs(f["freq_r"][L, p, n]))), 4)})
    return r


def validate(f, flags):
    """Do the flags enrich for poles the survey or sweeplib@1 already call prediction- or frequency-type?"""
    from bonsai_interp.env import ROOT
    from bonsai_interp.sweeplib.validate import PATTERNS
    res = [json.loads(l) for l in gzip.open(ROOT / "data" / "sweeplib" / "bonsai1.sweeplib1.results.jsonl.gz", "rt", encoding="utf-8")]
    pred = [r for r in res if r["kind"] == "predict" and not r.get("placebo")]
    ix = json.loads((ROOT / "graph" / "index.json").read_text(encoding="utf-8"))["neurons"]
    held = {k: v for k, v in ix.items() if v.get("status") == "held" and k.startswith("bonsai1/")}
    lab = lambda pat: [k for k, v in held.items() if re.search(pat, (v.get("label") or "").lower())]
    sc = np.load(ROOT / "data" / f"scan_{NAME}.npz")["top_val"][:, :, 0, :]
    nonweak = np.argwhere(np.abs(sc) >= 0.5)
    rng = np.random.default_rng(1)
    rand = [tuple(x) for x in nonweak[rng.choice(len(nonweak), 20000, replace=False)]]
    groups = {
        "random non-weak poles": rand,
        "sweeplib@1 predict held": _ids([r["neuron"] for r in pred if r["outcome"] == "held"]),
        "sweeplib@1 predict failed": _ids([r["neuron"] for r in pred if r["outcome"] == "failed"]),
        "survey held, prediction wording": _ids(lab(PATTERNS["predict"])),
        "survey held, frequency wording": _ids(lab(r"\bfrequen|common (word|token)s?|function words?|high-frequency|stopword")),
        "survey held, all": _ids(list(held)),
        "survey dense / positional": _ids([k for k, v in ix.items() if k.startswith("bonsai1/") and v.get("status") == "dense"]),
    }
    out = {k: _rates(f, flags, v) for k, v in groups.items()}
    for k, v in groups.items():  # same-layer random baseline, since every flag depends on depth
        if v and k != "random non-weak poles":
            Ls = np.array([x[0] for x in v])
            base = [x for x in rand if x[0] in set(Ls.tolist())]
            out[k]["random_same_layers"] = _rates(f, flags, base)
    return out


def pole(nid):
    """One pole's census entry with decoded tokens (what a card would show)."""
    from bonsai_interp import explore
    from bonsai_interp.graph import cards
    _, L, n, p = cards.parse(nid)
    pi = 0 if p == "+" else 1
    tok = explore.tokenizer()
    dec = lambda ids: [tok.decode([int(i)]) for i in ids]
    w, s = np.load(_path(L, "weights")), np.load(_path(L, "steer"))
    k = np.load(_path(L, "links"))
    pre = "wrp" if pi == 0 else "wrn"
    link = lambda idx, G, U: [{"to": f"{NAME}/L{int(i) // N}/N{int(i) % N}{'+' if g * u >= 0 else '-'}", "G": round(float(g), 5), "U": round(float(u), 5)}
                              for i, g, u in zip(idx, G, U) if i >= 0]
    poles = np.load(OUT / f"{NAME}.census.poles.npz")
    fl = int(poles["flags"][L, pi, n])
    return {"neuron": nid, "flags": [k for k, b in BITS.items() if fl & b],
            "dla_promotes": dec(w["dla_top"][pi, n][:10]), "dla_spike": float(w["dla_spike"][pi, n]),
            "steer_promotes": dec(s["top"][pi, n]), "steer_dlp": [round(float(x), 2) for x in s["dlp"][pi, n]],
            "steer_kl": float(s["kl"][pi, n]), "steer_z": float(poles["steer_z"][L, pi, n]),
            "dla_pct": float(poles["dla_pct"][L, pi, n]),
            "freq_r": float(w["freq_r"][n]) * (1 if pi == 0 else -1), "null_share": float(w["null_gap_share"][n]),
            "out_norm_z": float(poles["out_norm_z"][L, pi, n]),
            "readers": link(k["rd_idx"][n][:8], k["rd_G"][n][:8], k["rd_U"][n][:8]),
            "writers": [{**x, "from": x.pop("to")[:-1]} for x in link(k[f"{pre}_idx"][n][:8], k[f"{pre}_G"][n][:8], k[f"{pre}_U"][n][:8])]}


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "pole":
        print(json.dumps(pole(sys.argv[2]), indent=1, ensure_ascii=False))
    else:
        s = build()
        print(json.dumps({k: s[k] for k in ("totals", "random", "agree_null", "validation")}, indent=1))
