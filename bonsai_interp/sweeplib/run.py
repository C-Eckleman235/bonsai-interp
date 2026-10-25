"""The sweep library: deterministic screens for kinds of neuron that a word hit does not explain. No language model writes
or judges anything.

  python -m bonsai_interp.sweeplib.run protocol       # write the protocol manifest: commit it ALONE
  python -m bonsai_interp.sweeplib.run select         # GPU: read every selection battery on every pole; the predict weights
  python -m bonsai_interp.sweeplib.run build          # GPU (logits only): predict tests; write the tests file and its
                                                      # pre-registration manifest: commit that ALONE
  python -m bonsai_interp.sweeplib.run score          # GPU: read the registered tests, judge with quick-8+8@2's rule
  (each takes --smoke DIR: no manifests, small batteries, files in DIR; --smoke DIR --pilot: full batteries)

How it works:
  - Kinds (kinds.py: copy, topic, lang, struct, number) are batteries of minimal pairs that do not depend on the pole. One
    read of a battery gives every layer's activation at each snippet's last token, so one pass scores all 442,368 poles.
  - A pole is selected for a condition when its selection pairs separate:
      - AUROC >= 0.85;
      - median fire >= 0.15 of the pole's corpus max;
      - at most 15% of quiet snippets >= 0.30;
      - the fire snippet beats its own quiet in >= 75% of pairs;
      - the mean paired difference is >= 0.10.
    The rule is looser than the judge on purpose: the held-out test applies the judge's bar. A pilot on the selection split
    (2026-10-24) found 0 placebo selections under this rule and under stricter ones, in every condition.
  - The same rule is run on placebo pairs (two quiet snippets) to count false selections.
  - Selected poles get their condition's 8 + 8 test from held-out windows, written and registered before any of it is read.
  - The test is judged like auto-token@2, with 3 preambles averaged and the read at the last token.
  - Every tested condition also gets a placebo test (8 quiet against 8 quiet) on the same poles, to count false holds.
  - predict (predict.py) is pole-specific: it is selected from the weights and the scan, and tested on the model's own
    next-token probabilities.
"""
import argparse
import datetime as dt
import gzip
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from bonsai_interp.auto import context_tier as ct
from bonsai_interp.sweeplib import kinds as K
from bonsai_interp.sweeplib import predict as PR

ROOT = ct.ROOT
PROTOCOL = "sweeplib@1"
OUT = ROOT / "data" / "sweeplib"
MAN = ROOT / "graph" / "auto"
N, LAYERS = 12288, 36
SELECT = {"auroc": 0.85, "med_fire": 0.15, "quiet_hi": 0.30, "quiet_hi_share": 0.15, "win": 0.75, "mean_diff": 0.10}
JUDGE = "quick-8+8@2"
# poles whose statistics are recorded for every condition (diagnostics): the zoo's copy neurons
WATCH = {w: (int(w.split("/L")[1].split("/")[0]), 0 if w.endswith("+") else 1, int(w.split("/N")[1][:-1])) for w in (
    "bonsai1/L23/N11310+", "bonsai1/L24/N1091+", "bonsai1/L18/N624+", "bonsai1/L19/N4757+", "bonsai1/L26/N9961+",
    "bonsai1/L13/N6073-", "bonsai1/L14/N8520-", "bonsai1/L15/N9881-", "bonsai1/L16/N5346+", "bonsai1/L17/N9491+")}
CODE = [Path(K.__file__), Path(PR.__file__), Path(__file__)]


def _paths(d=None):
    o, m = (Path(d), Path(d)) if d else (OUT, MAN)
    b = "bonsai1.sweeplib1"
    return {"protocol": m / f"{b}.protocol.json", "prereg": m / f"{b}.prereg.json", "summary": m / f"{b}.summary.json",
            "select": o / f"{b}.select.json.gz", "tests": o / f"{b}.tests.jsonl.gz", "results": o / f"{b}.results.jsonl.gz",
            "effect": o / f"{b}.effect.npz"}


def _code_sha():
    h = hashlib.sha256()
    for p in CODE:
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def _alone(p):
    """The first commit touching p, if that commit touched nothing else and p is clean since."""
    first, dirty = ct._committed_alone_clean(p)
    if not first or dirty:
        return None
    files = subprocess.run(["git", "show", "--name-only", "--format=", first], cwd=ROOT, capture_output=True, text=True).stdout.split()
    return first if files == [p.relative_to(ROOT).as_posix()] else None


def _setup(a, need_prereg=False):
    if a.smoke:
        P = _paths(a.smoke)
        Path(a.smoke).mkdir(parents=True, exist_ok=True)
        return P, None
    P = _paths()
    if not P["protocol"].exists() or not _alone(P["protocol"]):
        sys.exit(f"commit {P['protocol'].relative_to(ROOT)} on its own before running any stage (run `protocol` first)")
    man = json.loads(P["protocol"].read_text(encoding="utf-8"))
    if man["code_sha256"] != _code_sha():
        sys.exit("the sweep library's code changed since its protocol was registered: that is a new protocol version")
    if need_prereg and (not P["prereg"].exists() or not _alone(P["prereg"])):
        sys.exit(f"commit {P['prereg'].relative_to(ROOT)} on its own before scoring (run `build` first)")
    OUT.mkdir(parents=True, exist_ok=True)
    return P, man


def protocol(a):
    for p in CODE:
        if subprocess.run(["git", "status", "--porcelain", "--", str(p)], cwd=ROOT, capture_output=True, text=True).stdout.strip():
            sys.exit(f"commit {p.relative_to(ROOT)} first: the manifest pins the committed code")
    P = _paths()
    man = {"protocol": PROTOCOL, "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "toolchain_commit": ct._commit(),
           "code_sha256": _code_sha(), "layers": list(range(LAYERS)), "splits": K.SPLITS, "snippet": K.SNIP, "min_pos": K.MIN_POS,
           "per_domain": K.PER_DOMAIN, "pairs": {"select": K.N_SELECT, "min_select": K.MIN_SELECT, "test": K.N_TEST, "per_key": K.PER_KEY},
           "kinds": {k: {"about": v.about, "conditions": v.labels} for k, v in K.KINDS.items()},
           "predict": {"about": PR.__doc__, "topk": PR.TOPK, "spike_min": PR.SPIKE_MIN, "select": PR.SEL, "cap_per_layer": PR.CAP,
                       "pool": PR.POOL, "p_fire": PR.P_FIRE, "p_quiet": PR.P_QUIET},
           "selection": SELECT, "excluded": "weak poles (corpus max < 0.5)", "judge": JUDGE, "read": "last token, mean over the 3 PREAMBLES",
           "placebo": "per condition: selection on quiet-vs-quiet pairs (false selections) and an 8 + 8 quiet-vs-quiet test on the "
                      "selected poles (false holds)"}
    P["protocol"].write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {P['protocol'].relative_to(ROOT)}: commit it ALONE before running any stage")


# ---------------------------------------------------------------- the GPU reader

def _text(s, cache={}):
    if s.w * 4096 + s.p not in cache:
        from bonsai_interp import explore
        cache[s.w * 4096 + s.p] = explore.tokenizer().decode(s.ids.tolist())
    return cache[s.w * 4096 + s.p]


def read_last(model, tok, rec, texts, pre, B, top=LAYERS - 1, cached=True):
    """(len(texts), top + 1, N) float32: every layer's MLP activation at each text's last token, read after `pre`.
    cached: the preamble's keys and values are computed once and reused for every batch (the same token inputs as encoding
    pre + text together; a text whose joined encoding splits differently at the boundary is read in full instead)."""
    import copy as _copy

    import torch
    from transformers import DynamicCache

    from bonsai_interp.env import DEVICE
    from bonsai_interp.model import StopForward
    tok.padding_side = "right"
    out = np.empty((len(texts), top + 1, N), np.float32)
    pre_ids = tok(pre).input_ids
    joined = tok([pre + t for t in texts]).input_ids
    P_ = len(pre_ids)
    cache = None
    if cached:
        cache = DynamicCache()
        rec.mode, rec.interventions, rec.stop_at = None, {}, top
        try:
            with torch.no_grad():
                model.model(input_ids=torch.tensor([pre_ids], device=DEVICE), past_key_values=cache, use_cache=True)
        except StopForward:
            pass
        finally:
            rec.stop_at = None
    fast = [i for i in range(len(texts)) if cached and joined[i][:P_] == pre_ids and len(joined[i]) > P_]
    slow = sorted(set(range(len(texts))) - set(fast))

    def run(idx, use_cache):
        seqs = [joined[i][P_:] if use_cache else joined[i] for i in idx]
        T = max(map(len, seqs))
        ids = torch.full((len(idx), T), tok.pad_token_id or 0, dtype=torch.long)
        mask = torch.zeros((len(idx), T), dtype=torch.long)
        for j, q in enumerate(seqs):
            ids[j, :len(q)], mask[j, :len(q)] = torch.tensor(q), 1
        ids, mask = ids.to(DEVICE), mask.to(DEVICE)
        last = mask.sum(1) - 1
        ar = torch.arange(len(idx), device=DEVICE)
        got = {}

        def grab(L):
            return lambda act: (got.__setitem__(L, act[ar, last].float()), act)[1]
        kw = {}
        if use_cache:
            c = _copy.deepcopy(cache)
            c.batch_repeat_interleave(len(idx))
            kw = {"past_key_values": c, "use_cache": True}
            mask = torch.cat([torch.ones((len(idx), P_), dtype=mask.dtype, device=DEVICE), mask], 1)
        rec.mode, rec.interventions, rec.stop_at = None, {L: grab(L) for L in range(top + 1)}, top
        try:
            with torch.no_grad():
                model.model(input_ids=ids, attention_mask=mask, **kw)
        except StopForward:
            pass
        finally:
            rec.interventions, rec.stop_at = {}, None
        out[idx] = torch.stack([got[L] for L in range(top + 1)], 1).cpu().numpy()

    for group, use_cache in ((fast, True), (slow, False)):
        b = 0
        while b < len(group):
            try:
                run(group[b:b + B], use_cache)
            except torch.OutOfMemoryError:
                if B == 1:
                    raise
                B = max(1, B // 2)
                torch.cuda.empty_cache()
                print(f"  out of GPU memory: batch {B} from here", flush=True)
                continue
            b += B
    return out


def read_probs(model, tok, texts, pre, sets, B):
    """Next-token probability mass on each text's own token set, after `pre` (model logits only; no neuron is read)."""
    import torch
    from bonsai_interp.env import DEVICE
    tok.padding_side = "right"
    out = np.empty(len(texts), np.float32)
    for b in range(0, len(texts), B):
        chunk = texts[b:b + B]
        enc = tok([pre + t for t in chunk], return_tensors="pt", padding=True).to(DEVICE)
        last = enc.attention_mask.sum(1) - 1
        with torch.no_grad():
            h = model.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask).last_hidden_state
            pr = model.lm_head(h[torch.arange(len(chunk), device=DEVICE), last]).float().softmax(-1)
        idx = torch.tensor(np.stack(sets[b:b + B]), device=DEVICE, dtype=torch.long)
        out[b:b + len(chunk)] = pr.gather(1, idx).sum(1).cpu().numpy()
    return out


@__import__("functools").lru_cache(None)
def _pm():
    return np.stack([ct._pole_max(L) for L in range(LAYERS)]).astype(np.float32)  # (L, 2, N)


def _stats(x, fi, qi):
    """Selection statistics per pole from aligned pairs (rows fi of x against rows qi), one layer at a time so memory stays
    small: x (n, L, N) raw activations -> dict of (L, 2, N)."""
    n, pm = len(fi), _pm()
    out = {k: np.zeros((x.shape[1], 2, N), np.float32) for k in ("auroc", "med_fire", "quiet_hi_share", "win", "mean_diff")}
    for L in range(x.shape[1]):
        a = x[:, L]
        v = np.stack([a / pm[L, 0], -a / pm[L, 1]], 1)  # (n_rows, 2, N): fractions of each pole's corpus max
        f, q = v[fi], v[qi]
        r = np.concatenate([f, q]).argsort(0, kind="stable").argsort(0, kind="stable")[:n].sum(0) + n
        out["auroc"][L] = (r - n * (n + 1) / 2) / (n * n)
        out["med_fire"][L] = np.median(f, 0)
        out["quiet_hi_share"][L] = (q >= SELECT["quiet_hi"]).mean(0)
        out["win"][L] = (f > q).mean(0)
        out["mean_diff"][L] = (f - q).mean(0)
    return out


def _selected(st, ok):
    return ok & (st["auroc"] >= SELECT["auroc"]) & (st["med_fire"] >= SELECT["med_fire"]) & \
        (st["quiet_hi_share"] <= SELECT["quiet_hi_share"]) & (st["win"] >= SELECT["win"]) & (st["mean_diff"] >= SELECT["mean_diff"])


def _not_weak():
    from bonsai_interp.graph import which
    return which.load_index()["pole_max"] >= ct.RULE["min_max"]  # (L, 2, N)


def _pid(L, p, n):
    return ct.pid(p, n, L)


def select(a):
    P, man = _setup(a)
    from bonsai_interp.env import limit_memory
    from bonsai_interp.probe.textinv import PREAMBLES
    ok = _not_weak()
    limit_memory(0.8)
    model, tok, rec = ct._gpu_load(lm_head=True)
    t0, res = time.time(), {"kinds": {}, "predict": {}}
    for kind in [k for k in (a.kinds or K.KINDS) if k in K.KINDS]:
        bat = K.battery(kind, "select")
        res["kinds"][kind] = {}
        for c in K.KINDS[kind].labels:
            pairs, plac = bat[c], bat["placebo:" + c]
            if a.smoke and not a.pilot:
                pairs, plac = pairs[:12], plac[:12]
            entry = {"pairs": len(pairs), "strata": {}}
            for _, _, st in pairs:
                entry["strata"][st] = entry["strata"].get(st, 0) + 1
            if len(pairs) < (4 if a.smoke and not a.pilot else K.MIN_SELECT):
                entry["why"] = f"only {len(pairs)} selection pairs (needs {K.MIN_SELECT})"
                res["kinds"][kind][c] = entry
                print(f"{kind} {c}: untestable ({entry['why']})", flush=True)
                continue
            texts = [_text(s) for f, q, _ in pairs for s in (f, q)] + [_text(s) for f, q, _ in plac for s in (f, q)]
            x = read_last(model, tok, rec, texts, PREAMBLES[0], a.batch)
            m = 2 * len(pairs)
            st = _stats(x, np.arange(0, m, 2), np.arange(1, m, 2))
            sel = _selected(st, ok)
            stp = _stats(x, np.arange(m, len(x), 2), np.arange(m + 1, len(x), 2)) if len(plac) >= 4 else None
            del x
            if a.smoke and a.pilot:  # a pilot keeps every pole's statistics, for calibrating the rule offline
                np.savez_compressed(Path(a.smoke) / f"stats.{kind}.{c.replace('/', '_')}.npz",
                                    **{k: v.astype(np.float16) for k, v in st.items()},
                                    **({"placebo_" + k: v.astype(np.float16) for k, v in stp.items()} if stp else {}))
            entry["placebo_pairs"] = len(plac)
            entry["placebo_selected"] = int(_selected(stp, ok).sum()) if stp else None
            Ls, ps, ns = np.nonzero(sel)
            entry["selected"] = [{"neuron": _pid(int(L), int(p), int(n)), **{k: round(float(v[L, p, n]), 3) for k, v in st.items()}}
                                 for L, p, n in zip(Ls, ps, ns)]
            entry["by_layer"] = np.bincount(Ls, minlength=LAYERS).tolist()
            # diagnostics, not selections: the 10 best-separating poles by mean paired difference, and the watch list
            md = np.where(ok, st["mean_diff"], -np.inf).ravel()
            top = np.argsort(-md)[:10]
            entry["best"] = [{"neuron": _pid(*map(int, np.unravel_index(i, ok.shape))),
                              **{k: round(float(v.ravel()[i]), 3) for k, v in st.items()}} for i in top]
            entry["watch"] = {w: {k: round(float(v[L, p, n]), 3) for k, v in st.items()} for w, (L, p, n) in WATCH.items()}
            res["kinds"][kind][c] = entry
            print(f"{kind} {c}: {len(pairs)} pairs, {len(entry['selected'])} poles selected, placebo {entry['placebo_selected']} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    if not a.kinds or "predict" in a.kinds:
        eff = {}
        for L in ((0, 20) if a.smoke and not a.pilot else range(LAYERS)):
            ct._stop_requested()
            S, spike = PR.direct_effect(model, L)
            eff[f"S{L}"], eff[f"spike{L}"] = S, spike.astype(np.float16)
            cand = PR.candidates(L, S, spike, ok[L])
            res["predict"][str(L)] = [{"neuron": _pid(L, p, n), "hits": h, "lift": lf, "spike": round(float(spike[p, n]), 2),
                                       "set": seen, "promoted": S[p, n].tolist()} for n, p, h, lf, seen in cand]
            print(f"predict L{L}: {len(cand)} candidates ({time.time() - t0:.0f} s)", flush=True)
        np.savez_compressed(P["effect"], **eff)
    ct._gpu_idle("select")
    res["minutes"] = round((time.time() - t0) / 60, 1)
    ct._write_gz(P["select"], [res])
    print(f"wrote {P['select']} in {res['minutes']} min", flush=True)


def build(a):
    """The tests: battery conditions with selected poles get their held-out 8 + 8 and a placebo 8 + 8; predict candidates get
    snippets chosen by the model's next-token probabilities. Writes the tests file and the prereg manifest."""
    P, man = _setup(a)
    sel = ct._read_gz(P["select"])[0]
    rows = []
    for kind, conds in sel["kinds"].items():
        bat = K.battery(kind, "test")
        for c, e in conds.items():
            if not e.get("selected"):
                continue
            pairs, plac = bat[c], bat.get("placebo:" + c, [])
            item = {"type": "items", "kind": kind, "condition": c, "label": K.KINDS[kind].labels[c]}
            if len(pairs) < K.N_TEST:
                rows.append({**item, "class": "untestable", "why": f"only {len(pairs)} held-out pairs"})
                continue
            rows.append({**item, "class": "tested", "fire": [_text(f) for f, _, _ in pairs], "quiet": [_text(q) for _, q, _ in pairs],
                         "quiet_kind": [st for _, _, st in pairs], "windows": [[f.w, f.p, q.w, q.p] for f, q, _ in pairs]})
            if len(plac) >= K.N_TEST:
                rows.append({"type": "items", "kind": kind, "condition": "placebo:" + c, "label": "placebo", "class": "placebo",
                             "fire": [_text(f) for f, _, _ in plac], "quiet": [_text(q) for _, q, _ in plac], "quiet_kind": ["placebo"] * len(plac)})
            for r in e["selected"]:
                rows.append({"type": "test", "neuron": r["neuron"], "kind": kind, "condition": c,
                             "select": {k: r[k] for k in ("auroc", "med_fire", "quiet_hi_share", "win", "mean_diff")}})
    if sel.get("predict"):
        rows += _build_predict(a, sel["predict"])
    sha = ct._write_gz(P["tests"], rows)
    counts = {"items": sum(r["type"] == "items" and r["class"] == "tested" for r in rows),
              "battery_tests": sum(r["type"] == "test" for r in rows),
              "predict_tests": sum(r["type"] == "predict" and r["class"] == "tested" for r in rows),
              "predict_untestable": sum(r["type"] == "predict" and r["class"] == "untestable" for r in rows)}
    pre = {"protocol": PROTOCOL, "protocol_commit": _alone(P["protocol"]) if not a.smoke else None,
           "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "tests_file": P["tests"].name, "tests_sha256": sha,
           "select_file": P["select"].name, "select_sha256": hashlib.sha256(gzip.open(P["select"], "rb").read()).hexdigest(), "counts": counts}
    P["prereg"].write_text(json.dumps(pre, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(counts), f"\nwrote {P['tests']} and {P['prereg']}: commit the manifest ALONE before scoring", flush=True)


def _build_predict(a, cand):
    from bonsai_interp import explore
    from bonsai_interp.env import limit_memory
    from bonsai_interp.probe.textinv import PREAMBLES
    toks = explore.corpus()
    jobs = []
    for L, cs in sorted(cand.items(), key=lambda kv: int(kv[0])):
        for r in cs[:3] if a.smoke and not a.pilot else cs:
            n, p = int(r["neuron"].split("/N")[1][:-1]), 0 if r["neuron"].endswith("+") else 1
            s = np.array(r["set"])
            fire, quiet = PR.pool(int(L), n, p, s)
            jobs.append((r, s, fire, quiet))
    texts, sets, where = [], [], []
    for j, (r, s, fire, quiet) in enumerate(jobs):
        for side, ends in (("fire", fire), ("quiet", quiet)):
            for w, pp in ends:
                texts.append(_text(K.Snip(w, pp, None, toks[w, pp - K.SNIP + 1:pp + 1].astype(np.int64))))
                sets.append(s)
                where.append((j, side, w, pp))
    print(f"predict: {len(jobs)} candidates, {len(texts)} pooled snippets to read for next-token probabilities", flush=True)
    limit_memory(0.8)
    model, tok, rec = ct._gpu_load(lm_head=True)
    t0 = time.time()
    pS = read_probs(model, tok, texts, PREAMBLES[0], sets, a.batch)
    ct._gpu_idle("predict pools")
    print(f"  read in {(time.time() - t0) / 60:.1f} min", flush=True)
    rows, by_job = [], {}
    for k, (j, side, w, pp) in enumerate(where):
        by_job.setdefault(j, {"fire": [], "quiet": []})[side].append(k)
    for j, (r, s, fire, quiet) in enumerate(jobs):
        L = int(r["neuron"].split("/L")[1].split("/")[0])
        n = int(r["neuron"].split("/N")[1][:-1])
        rng = np.random.default_rng([L, n, 6])
        bj = by_job.get(j, {"fire": [], "quiet": []})
        cur = {k: int(toks[where[k][2], where[k][3]]) for k in bj["fire"] + bj["quiet"]}
        ch = PR.choose(bj["fire"], bj["quiet"], pS, cur, rng)
        vb = ct.vocab()
        lab = "prediction: fires where the model is about to predict " + ", ".join(repr(vb["text"][t]) for t in r["set"][:6]) + \
              " (the tokens it promotes most), not on the same words where it expects something else"
        base = {"type": "predict", "neuron": r["neuron"], "kind": "predict", "condition": "predict", "label": lab,
                "select": {k: r[k] for k in ("hits", "lift", "spike")}, "set": r["set"]}
        if ch is None:
            rows.append({**base, "class": "untestable", "why": f"{sum(pS[k] >= PR.P_FIRE for k in bj['fire'])} pooled snippets with "
                         f"P(S) >= {PR.P_FIRE}, {sum(pS[k] <= PR.P_QUIET for k in bj['quiet'])} with <= {PR.P_QUIET} (8 each needed)"})
            continue
        F, Q = ch
        rows.append({**base, "class": "tested", "fire": [texts[k] for k in F], "quiet": [texts[k] for k in Q],
                     "quiet_kind": ["same word, other prediction" if cur[k] in {cur[i] for i in F} else "other prediction" for k in Q],
                     "p_set": {"fire": [round(float(pS[k]), 3) for k in F], "quiet": [round(float(pS[k]), 3) for k in Q]}})
    return rows


def score(a):
    P, man = _setup(a, need_prereg=True)
    from bonsai_interp.env import limit_memory
    from bonsai_interp.graph.protocols import PROTOCOLS
    from bonsai_interp.probe.textinv import PREAMBLES
    pre = json.loads(P["prereg"].read_text(encoding="utf-8"))
    blob = gzip.open(P["tests"], "rb").read()
    if hashlib.sha256(blob).hexdigest() != pre["tests_sha256"]:
        sys.exit("the tests file does not match its pre-registered sha256")
    rows = [json.loads(x) for x in blob.decode("utf-8").splitlines()]
    items = {(r["kind"], r["condition"]): r for r in rows if r["type"] == "items" and r["class"] in ("tested", "placebo")}
    tests = []  # (row, fire texts, quiet texts, quiet kinds, placebo of)
    for r in rows:
        if r["type"] == "test":
            it = items.get((r["kind"], r["condition"]))
            if it:
                tests.append((r, it, None))
                pl = items.get((r["kind"], "placebo:" + r["condition"]))
                if pl:
                    tests.append(({**r, "condition": "placebo:" + r["condition"], "placebo": True}, pl, r["condition"]))
        elif r["type"] == "predict" and r["class"] == "tested":
            tests.append((r, r, None))
    want = {}  # text -> {(L, p, n)}
    for r, it, _ in tests:
        L = int(r["neuron"].split("/L")[1].split("/")[0])
        n, p = int(r["neuron"].split("/N")[1][:-1]), 0 if r["neuron"].endswith("+") else 1
        for t in it["fire"] + it["quiet"]:
            want.setdefault(t, set()).add((L, p, n))
    # group the texts by the deepest layer they are needed at, so a text read only for layer 3 stops at layer 3
    by_top = {}
    for t, ps in want.items():
        by_top.setdefault(max(L for L, _, _ in ps), []).append(t)
    pm = _pm()
    limit_memory(0.8)
    model, tok, rec = ct._gpu_load()
    t0, val = time.time(), {}
    print(f"scoring {len(tests)} tests on {len(want)} snippets x {len(PREAMBLES)} preambles", flush=True)
    for top in sorted(by_top):
        ts = sorted(by_top[top])
        for b in range(0, len(ts), 256):  # in slices, so memory stays bounded
            sl = ts[b:b + 256]
            acc = np.zeros((len(sl), top + 1, N), np.float32)
            for pre_text in PREAMBLES:
                acc += read_last(model, tok, rec, sl, pre_text, a.batch, top) / len(PREAMBLES)
            for i, t in enumerate(sl):
                for L, p, n in want[t]:
                    v = acc[i, L, n] / pm[L, 0, n] if p == 0 else -acc[i, L, n] / pm[L, 1, n]
                    val[(t, L, p, n)] = round(float(v), 4)
        print(f"  texts to layer {top}: {len(ts)} ({time.time() - t0:.0f} s)", flush=True)
    ct._gpu_idle("score")
    rule = PROTOCOLS[JUDGE]["rule"]
    out, tally = [], {}
    for r, it, plof in tests:
        L = int(r["neuron"].split("/L")[1].split("/")[0])
        n, p = int(r["neuron"].split("/N")[1][:-1]), 0 if r["neuron"].endswith("+") else 1
        per = [{"set": "fire", "text": t, "act": val[(t, L, p, n)]} for t in it["fire"]] + \
              [{"set": "quiet", "text": t, "act": val[(t, L, p, n)], "kind": k} for t, k in zip(it["quiet"], it["quiet_kind"])]
        res = rule({"per_sentence": per})
        key = ("placebo:" if plof else "") + r["kind"]
        tally.setdefault(key, {}).setdefault(res, 0)
        tally[key][res] += 1
        out.append({"neuron": r["neuron"], "kind": r["kind"], "condition": r["condition"], "label": it["label"] if not plof else "placebo",
                    "outcome": res, "per_sentence": per, "select": r.get("select"), "placebo": bool(plof)})
    sha = ct._write_gz(P["results"], out)
    held = [o for o in out if o["outcome"] == "held" and not o["placebo"]]
    by_cond = {}
    for o in out:
        e = by_cond.setdefault(f"{o['kind']}/{o['condition']}", {"held": 0, "failed": 0})
        e[o["outcome"]] = e.get(o["outcome"], 0) + 1
    summ = {"protocol": PROTOCOL, "scored": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "toolchain_commit": ct._commit(),
            "prereg_commit": (_alone(P["prereg"]) or "")[:12] if not a.smoke else None, "outcomes": tally, "by_condition": by_cond,
            "poles_held": len({o["neuron"] for o in held}), "held_by_layer": np.bincount([int(o["neuron"].split("/L")[1].split("/")[0]) for o in held],
                                                                                       minlength=LAYERS).tolist(),
            "minutes": round((time.time() - t0) / 60, 1), "results_file": P["results"].name, "results_sha256": sha}
    P["summary"].write_text(json.dumps(summ, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: summ[k] for k in ("outcomes", "poles_held", "minutes")}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["protocol", "select", "build", "score"])
    ap.add_argument("--smoke", help="a scratch directory: no manifests, small batteries")
    ap.add_argument("--pilot", action="store_true", help="with --smoke: the full batteries (an exploratory run on the selection split)")
    ap.add_argument("--kinds", nargs="*", help="select only these kinds (and 'predict')")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    {"protocol": protocol, "select": select, "build": build, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
