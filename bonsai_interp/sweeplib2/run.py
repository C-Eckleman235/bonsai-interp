"""sweeplib@2: twin batteries seeded by word hits. Pass 2 builds on pass 1: every word the word screen (auto-token@2)
found becomes a battery of one-swap twins (twins.py), and every pole that shares the word is scored from one read.

  python -m bonsai_interp.sweeplib2.run protocol   # CPU: the seed words and the rule; commit the manifest ALONE
  python -m bonsai_interp.sweeplib2.run select     # GPU: read the selection twins (windows 0-15,999), select poles
  python -m bonsai_interp.sweeplib2.run build      # CPU: held-out twins (windows 16,400 on) for the selected claims;
                                                   # commit the prereg manifest ALONE
  python -m bonsai_interp.sweeplib2.run score      # GPU: read the registered tests, judge, measure coverage
  (--smoke DIR: no manifests, files in DIR; --words K: the first K seed words; build/score --split select exercise the
  code on the selection split, so nothing of the test split is read before registration)

Rule (pinned by the protocol manifest):
  - Seeds: every (pole, word) where the word is one of the pole's stage-1 triggers (atlas hit_tok with hit_trigger) and
    is a whole-word token (a leading space and 2 or more letters). The atlas has a row for every pole the word screen
    tabulated, so this includes every pole whose auto-token@2 test held. Words are ranked by their number of seed poles;
    the first WORDS are run.
  - Selection, per (pole, claim), on the claim's selection pairs (twins.N_SELECT, at least twins.MIN_SELECT): sweeplib@1's
    rule (AUROC >= 0.85, median fire >= 0.15 of the pole's max, at most 15% of quiet >= 0.30, fire beats its own quiet in
    >= 75% of pairs, mean paired difference >= 0.10), read at the last token after the first standard preamble. Only the
    word's own seed poles are scored, and weak poles (corpus max < 0.5) never.
  - Placebo: the same rule on the claim's placebo pairs (base vs base2, or ctl vs ctl2), on the same poles, counts false
    selections; every tested claim also gets an 8 + 8 placebo test on the same poles (false holds).
  - Test: 8 + 8 held-out twins per (word, claim), judged with quick-8+8@2 at the last token, mean over the 3 preambles.
  - Coverage of a held claim: the share of the pole's strong activations (word screen tables, windows 0-7,999) the claim
    covers. Token-set claims are measured exactly (graph.coverage.token_coverage with a token trigger). Context claims
    (sentence position, code slot) get an estimate: the seed's token coverage times the context's share of the seed's
    strong activations (its share of the seed's occurrences times its strong rate in the selection reads, against the
    other contexts of the same axis); their upper bound is the seed's token coverage.
"""
import argparse
import datetime as dt
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from bonsai_interp.auto import context_tier as ct
from bonsai_interp.sweeplib import run as R1
from bonsai_interp.sweeplib2 import twins as TW

ROOT = ct.ROOT
PROTOCOL = "sweeplib@2"
OUT = ROOT / "data" / "sweeplib2"
MAN = ROOT / "graph" / "auto"
N, LAYERS = 12288, 36
WORDS = 300
SELECT, JUDGE = R1.SELECT, R1.JUDGE
STRONG = 0.3
BANDS = [(0, 5), (6, 11), (12, 17), (18, 23), (24, 29), (30, 35)]
CODE = [Path(TW.__file__), Path(__file__)]
# the token set a held claim covers: "seed" (the seed token), "+case", "+space", "+piece", "+infl"; None for context claims
COVERS = {"form/any-case": "+case", "form/this-case": "seed", "form/no-space": "+space", "form/space-only": "seed",
          "piece/inside": "+piece", "piece/own": "seed", "infl/family": "+infl", "infl/this-form": "seed"}
CONTEXT = {"pos/initial": ("initial", ("initial", "mid")), "pos/mid": ("mid", ("initial", "mid")),
           "slot/comment": ("comment", ("comment", "string", "code", "prose")), "slot/string": ("string", ("comment", "string", "code", "prose")),
           "slot/code": ("code", ("comment", "string", "code", "prose")), "slot/prose": ("prose", ("comment", "string", "code", "prose"))}


def _paths(d=None):
    o, m = (Path(d), Path(d)) if d else (OUT, MAN)
    b = "bonsai1.sweeplib2"
    return {"protocol": m / f"{b}.protocol.json", "prereg": m / f"{b}.prereg.json", "summary": m / f"{b}.summary.json",
            "seeds": o / f"{b}.seeds.json.gz", "select": o / f"{b}.select.json.gz", "stats": o / f"{b}.pilot_stats.npz",
            "tests": o / f"{b}.tests.jsonl.gz", "results": o / f"{b}.results.jsonl.gz"}


def _code_sha():
    h = hashlib.sha256()
    for p in CODE:
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def _sha_gz(p):
    return hashlib.sha256(gzip.open(p, "rb").read()).hexdigest()


def _amendments(P):
    """Code versions allowed besides the registered one: entries of the committed-alone amendments file, each a fix that
    changes no selection, test or judgement (a speed or crash fix), with its reason."""
    p = P["protocol"].with_name(P["protocol"].name.replace(".protocol.", ".amendments."))
    if not p.exists() or not R1._alone(p):
        return set()
    return {e["code_sha256"] for e in json.loads(p.read_text(encoding="utf-8"))}


def _setup(a, need_prereg=False):
    if a.smoke:
        Path(a.smoke).mkdir(parents=True, exist_ok=True)
        return _paths(a.smoke), None
    P = _paths()
    if not P["protocol"].exists() or not R1._alone(P["protocol"]):
        sys.exit(f"commit {P['protocol'].relative_to(ROOT)} on its own before running any stage (run `protocol` first)")
    man = json.loads(P["protocol"].read_text(encoding="utf-8"))
    if _code_sha() not in {man["code_sha256"]} | _amendments(P):
        sys.exit("sweeplib2's code changed since its protocol was registered and the change is not a committed amendment: "
                 "that is a new protocol version")
    if _sha_gz(P["seeds"]) != man["seeds_sha256"]:
        sys.exit("the seeds file does not match the protocol")
    if need_prereg and (not P["prereg"].exists() or not R1._alone(P["prereg"])):
        sys.exit(f"commit {P['prereg'].relative_to(ROOT)} on its own before scoring (run `build` first)")
    OUT.mkdir(parents=True, exist_ok=True)
    return P, man


# ---------------------------------------------------------------- seeds

def seeds():
    """[{tok, text, poles: [[L, p, n, auto2 test, hit_coverage, bin]]}], words ranked by seed poles (ties: token id)."""
    from bonsai_interp import atlas
    vb = ct.vocab()
    by = {}
    for L in range(LAYERS):
        t = atlas.load(L)
        if t is None:
            continue
        c = t.select(["neuron", "sign", "hit_tok", "hit_coverage", "hit_trigger", "test", "bin"]).to_pydict()
        for i in range(t.num_rows):
            for tk, cv, tr in zip(c["hit_tok"][i] or [], c["hit_coverage"][i] or [], c["hit_trigger"][i] or []):
                if tr and TW.is_seed_token(tk, vb):
                    by.setdefault(int(tk), []).append([L, 0 if c["sign"][i] == "+" else 1, int(c["neuron"][i]), c["test"][i],
                                                        round(float(cv), 4), c["bin"][i]])
    return [{"tok": t, "text": vb["text"][t], "poles": by[t]} for t in sorted(by, key=lambda t: (-len(by[t]), t))]


def protocol(a):
    import subprocess
    for p in CODE:
        if subprocess.run(["git", "status", "--porcelain", "--", str(p)], cwd=ROOT, capture_output=True, text=True).stdout.strip():
            sys.exit(f"commit {p.relative_to(ROOT)} first: the manifest pins the committed code")
    P = _paths()
    OUT.mkdir(parents=True, exist_ok=True)
    S = seeds()
    ct._write_gz(P["seeds"], S)
    run = S[:WORDS]
    man = {"protocol": PROTOCOL, "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "toolchain_commit": ct._commit(),
           "code_sha256": _code_sha(), "about": __doc__.split("\n\n")[0],
           "seeds_file": P["seeds"].name, "seeds_sha256": _sha_gz(P["seeds"]), "seed_words": len(S),
           "seed_pole_words": sum(len(w["poles"]) for w in S), "words_run": len(run), "pole_words_run": sum(len(w["poles"]) for w in run),
           "first_words": [w["text"] for w in run[:40]], "last_word_poles": len(run[-1]["poles"]) if run else 0,
           "splits": TW.SPLITS, "snippet": TW.SNIP, "pairs": {"select": TW.N_SELECT, "min_select": TW.MIN_SELECT, "test": TW.N_TEST},
           "sets": TW.__doc__.split("read at their last token.")[1].split("A claim is")[0].strip(),
           "claims": {c: {"axis": v[0], "type": v[1], "fire": v[2], "quiet": list(v[3]), "placebo": PROTOCOL + ":" + v[4]} for c, v in TW.CLAIMS.items()},
           "selection": SELECT, "excluded": "weak poles (corpus max < 0.5); poles not seeded by the word",
           "judge": JUDGE, "read": "last token; selection after PREAMBLES[0], tests mean over the 3 PREAMBLES",
           "coverage": {"assignment": 0.5, "token claims": COVERS, "context claims": {k: v[0] for k, v in CONTEXT.items()},
                        "strong": STRONG}}
    P["protocol"].write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(S)} seed words, {man['seed_pole_words']} (pole, word) seeds; running the first {len(run)} words "
          f"({man['pole_words_run']} seeds, the last with {man['last_word_poles']} poles)")
    print(f"wrote {P['seeds']} and {P['protocol'].relative_to(ROOT)}: commit the manifest ALONE before any stage")


def _load_seeds(P, a):
    if not P["seeds"].exists():
        if not a.smoke:
            sys.exit("no seeds file: run `protocol`")
        ct._write_gz(P["seeds"], seeds())
    S = ct._read_gz(P["seeds"])
    return S[:a.words or WORDS]


# ---------------------------------------------------------------- reading

def _read(texts, cols, tops, pres, B, fake=False):
    """texts: {word: [text]}; cols: {word: (Ls, ns)}; tops: {word: deepest layer}. -> {word: (len(texts), len(cols)) raw
    activations at the last token, mean over the preambles `pres`}. Words that need the same depth share batches.
    fake: random numbers instead of a read (a CPU check of the code around it; smoke runs only)."""
    if fake:
        rng = np.random.default_rng(0)
        return {w: rng.normal(0, 0.5, (len(t), len(cols[w][0]))).astype(np.float32) for w, t in texts.items()}, 0.0
    from bonsai_interp.env import limit_memory
    limit_memory(0.8)
    model, tok, rec = ct._gpu_load()
    out = {w: np.zeros((len(t), len(cols[w][0])), np.float32) for w, t in texts.items()}
    by_top = {}
    for w in texts:
        by_top.setdefault(tops[w], []).extend((w, i) for i in range(len(texts[w])))
    t0, done, total = time.time(), 0, sum(len(t) for t in texts.values())
    for top in sorted(by_top):
        jobs = by_top[top]
        for b in range(0, len(jobs), 256):
            sl = jobs[b:b + 256]
            tx = [texts[w][i] for w, i in sl]
            acc = np.zeros((len(sl), top + 1, N), np.float32)
            for pre in pres:
                acc += R1.read_last(model, tok, rec, tx, pre, B, top) / len(pres)
            for k, (w, i) in enumerate(sl):
                Ls, ns = cols[w]
                out[w][i] = acc[k, Ls, ns]
            done += len(sl)
            el = time.time() - t0
            print(f"  read {done}/{total} snippets ({done / el:.1f}/s, eta {(total - done) / max(done / el, 1e-9) / 60:.1f} min)", flush=True)
    ct._gpu_idle("sweeplib2 read")
    return out, (time.time() - t0) / 60


def _cols(poles):
    """Unique (L, n) of a word's poles, and each pole's column."""
    u = sorted({(p[0], p[2]) for p in poles})
    ix = {k: i for i, k in enumerate(u)}
    return (np.array([k[0] for k in u], np.int64), np.array([k[1] for k in u], np.int64)), [ix[(p[0], p[2])] for p in poles]


def _frac(raw, poles, col):
    """(n texts, n poles): each pole's activation as a fraction of its corpus max (the - pole reads -activation)."""
    pm = R1._pm()
    L = np.array([p[0] for p in poles]); s = np.array([p[1] for p in poles]); n = np.array([p[2] for p in poles])
    sign = np.where(s == 0, 1.0, -1.0)
    return raw[:, col] * sign / pm[L, s, n]


def _stats(F, Q):
    n = len(F)
    r = np.concatenate([F, Q]).argsort(0, kind="stable").argsort(0, kind="stable")[:n].sum(0) + n
    return {"auroc": (r - n * (n + 1) / 2) / (n * n), "med_fire": np.median(F, 0), "quiet_hi_share": (Q >= SELECT["quiet_hi"]).mean(0),
            "win": (F > Q).mean(0), "mean_diff": (F - Q).mean(0)}


def _selected(st):
    return (st["auroc"] >= SELECT["auroc"]) & (st["med_fire"] >= SELECT["med_fire"]) & (st["quiet_hi_share"] <= SELECT["quiet_hi_share"]) & \
        (st["win"] >= SELECT["win"]) & (st["mean_diff"] >= SELECT["mean_diff"])


def _texts(bat):
    """Per word: unique texts, and per set the text index of each item."""
    T, where = {}, {}
    for w, b in bat.items():
        seen, lst, wh = {}, [], {}
        for s, items in b["sets"].items():
            wh[s] = []
            for it in items:
                x = TW.text(it)
                if x not in seen:
                    seen[x] = len(lst)
                    lst.append(x)
                wh[s].append(seen[x])
        T[w], where[w] = lst, wh
    return T, where


def _nid(L, p, n):
    return ct.pid(p, n, L)


# ---------------------------------------------------------------- select

def select(a):
    P, man = _setup(a)
    from bonsai_interp.graph import which
    from bonsai_interp.probe.textinv import PREAMBLES
    S = _load_seeds(P, a)
    ok = which.load_index()["pole_max"] >= ct.RULE["min_max"]
    S = [{**w, "poles": [p for p in w["poles"] if ok[p[0], p[1], p[2]]]} for w in S]
    S = [w for w in S if w["poles"]]
    t0 = time.time()
    bat = TW.battery(tuple(w["tok"] for w in S), "select", TW.N_SELECT)
    T, where = _texts(bat)
    print(f"{len(S)} words, {sum(len(w['poles']) for w in S)} seed poles, {sum(len(t) for t in T.values())} snippets "
          f"(battery built in {time.time() - t0:.0f} s)", flush=True)
    cols, colix = {}, {}
    for w in S:
        cols[w["tok"]], colix[w["tok"]] = _cols(w["poles"])
    raw, minutes = _read(T, cols, {w["tok"]: max(p[0] for p in w["poles"]) for w in S}, PREAMBLES[:1], a.batch, a.smoke and a.fake)
    res, pilot = {"words": [], "minutes_read": round(minutes, 1)}, {}
    for w in S:
        t, poles = w["tok"], w["poles"]
        b, wh = bat[t], where[t]
        V = _frac(raw[t], poles, colix[t])
        entry = {"tok": t, "text": w["text"], "info": b["info"], "counts": b["counts"], "sets": {s: len(v) for s, v in b["sets"].items()},
                 "poles": len(poles), "claims": {}}
        plac = {}
        for kind in ("ctx", "swap"):
            pp, f = TW.placebo_pairs(b["sets"], kind)
            if len(pp) >= TW.MIN_SELECT:
                st = _stats(V[[wh[f][i] for i, _, _, _ in pp]], V[[wh[q][j] for _, q, j, _ in pp]])
                plac[kind] = _selected(st)
                if a.pilot:
                    pilot[f"{t}|placebo:{kind}"] = np.stack([st[k] for k in sorted(st)]).astype(np.float16)
        # each context's strong rate per pole (for coverage estimates of context claims)
        rate = {s: (V[wh[s]] >= STRONG).mean(0) for s in ("base", "initial", "comment", "string", "code", "prose") if b["sets"].get(s)}
        for c, (axis, typ, f, qs, pk) in TW.CLAIMS.items():
            pr = TW.pairs(b["sets"], c)
            e = {"pairs": len(pr)}
            if len(pr) < TW.MIN_SELECT:
                e["why"] = f"only {len(pr)} selection pairs"
                entry["claims"][c] = e
                continue
            st = _stats(V[[wh[f][i] for i, _, _, _ in pr]], V[[wh[q][j] for _, q, j, _ in pr]])
            sel = _selected(st)
            if a.pilot:
                pilot[f"{t}|{c}"] = np.stack([st[k] for k in sorted(st)]).astype(np.float16)
            e["placebo_selected"] = int(plac[pk].sum()) if pk in plac else None
            e["selected"] = [{"neuron": _nid(*poles[j][:3]), "seed": poles[j][3:], **{k: round(float(v[j]), 3) for k, v in st.items()},
                              "rate": {s: round(float(r[j]), 3) for s, r in rate.items()}} for j in np.flatnonzero(sel)]
            entry["claims"][c] = e
        res["words"].append(entry)
    res["minutes"] = round((time.time() - t0) / 60, 1)
    ct._write_gz(P["select"], [res])
    if a.pilot:
        np.savez_compressed(P["stats"], **pilot)
    tot = {}
    for e in res["words"]:
        for c, x in e["claims"].items():
            d = tot.setdefault(c, [0, 0, 0])
            d[0] += len(x.get("selected", []))
            d[1] += x.get("placebo_selected") or 0
            d[2] += "selected" in x
    for c, (s, pl, nw) in tot.items():
        print(f"  {c:16s} words testable {nw:4d}  selected {s:6d}  placebo {pl:5d}")
    print(f"wrote {P['select']} in {res['minutes']} min", flush=True)


# ---------------------------------------------------------------- build

def build(a):
    """Held-out twins for every (word, claim) with a selected pole, a placebo 8 + 8 per (word, placebo kind), the tests file
    and the prereg manifest. Texts only: nothing is read."""
    P, man = _setup(a)
    split = a.split or "test"
    if split != "test" and not a.smoke:
        sys.exit("--split is for smoke runs")
    sel = ct._read_gz(P["select"])[0]
    words = [e for e in sel["words"] if any(x.get("selected") for x in e["claims"].values())]
    bat = TW.battery(tuple(e["tok"] for e in words), split, TW.N_TEST)
    rows = []
    for e in words:
        t, b = e["tok"], bat[e["tok"]]
        info = b["info"]
        kinds = set()
        for c, x in e["claims"].items():
            if not x.get("selected"):
                continue
            axis, typ, f, qs, pk = TW.CLAIMS[c]
            pr = TW.pairs(b["sets"], c)
            item = {"type": "items", "tok": t, "word": e["text"], "claim": c, "axis": axis, "label": TW.label(c, info)}
            if len(pr) < TW.N_TEST:
                rows.append({**item, "class": "untestable", "why": f"only {len(pr)} held-out pairs"})
                continue
            pr = pr[:TW.N_TEST]
            rows.append({**item, "class": "tested", "fire": [TW.text(b["sets"][f][i]) for i, _, _, _ in pr],
                         "quiet": [TW.text(b["sets"][q][j]) for _, q, j, _ in pr], "quiet_kind": [s for _, _, _, s in pr],
                         "items": [[*b["sets"][f][i], *b["sets"][q][j]] for i, q, j, _ in pr], "placebo": pk})
            kinds.add(pk)
            for r in x["selected"]:
                rows.append({"type": "test", "neuron": r["neuron"], "tok": t, "claim": c,
                             "select": {k: r[k] for k in ("auroc", "med_fire", "quiet_hi_share", "win", "mean_diff")}})
        for pk in sorted(kinds):
            pp, f = TW.placebo_pairs(b["sets"], pk)
            if len(pp) >= TW.N_TEST:
                pp = pp[:TW.N_TEST]
                rows.append({"type": "items", "tok": t, "word": e["text"], "claim": "placebo:" + pk, "axis": "placebo", "label": "placebo",
                             "class": "placebo", "fire": [TW.text(b["sets"][f][i]) for i, _, _, _ in pp],
                             "quiet": [TW.text(b["sets"][q][j]) for _, q, j, _ in pp], "quiet_kind": ["placebo"] * len(pp)})
    sha = ct._write_gz(P["tests"], rows)
    counts = {"words": len(words), "items": sum(r["type"] == "items" and r["class"] == "tested" for r in rows),
              "untestable": sum(r["type"] == "items" and r["class"] == "untestable" for r in rows),
              "placebo_items": sum(r["type"] == "items" and r["class"] == "placebo" for r in rows),
              "tests": sum(r["type"] == "test" for r in rows)}
    pre = {"protocol": PROTOCOL, "protocol_commit": R1._alone(P["protocol"]) if not a.smoke else None, "split": split,
           "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "tests_file": P["tests"].name, "tests_sha256": sha,
           "select_file": P["select"].name, "select_sha256": _sha_gz(P["select"]), "counts": counts}
    P["prereg"].write_text(json.dumps(pre, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(counts), f"\nwrote {P['tests']} and {P['prereg']}: commit the manifest ALONE before scoring", flush=True)


# ---------------------------------------------------------------- score

def _cover_tokens(claim, info):
    vb = ct.vocab()
    s, c = info["text"], info["core"]
    kind = COVERS[claim]
    toks = [s]
    if kind == "+case":
        toks += [v for _, v in info["case"]]
    elif kind == "+space" or kind == "+piece":
        toks += [c]
    elif kind == "+infl":
        toks += [vb["text"][i] for _, _, i in info["infl"]]
    return toks


def _coverage(nid, claim, info, counts, rate):
    """(coverage, bound, method) of a held claim on its pole."""
    from bonsai_interp.graph import coverage as CV
    if claim in COVERS:
        got = CV.token_coverage(nid, {"tokens": _cover_tokens(claim, info)})
        return (None, None, "no token table") if got is None else (round(got[0], 4), round(got[0], 4), "token set")
    got = CV.token_coverage(nid, {"tokens": [info["text"]]})
    if got is None:
        return None, None, "no token table"
    own, parts = CONTEXT[claim]
    # the natural share of each context among the seed's occurrences (code slots scaled from the classified sample)
    code = counts.get("code_total", 0) / max(counts.get("code_sampled", 0) - counts.get("code_unclear", 0), 1)
    share = {"initial": counts.get("initial", 0), "mid": counts.get("mid", 0), "prose": counts.get("prose_total", 0),
             **{k: counts.get(k, 0) * code for k in ("comment", "string", "code")}}
    r = {k: rate.get("base" if k == "mid" else k) for k in parts}
    known = [v for v in r.values() if v is not None]
    if r.get(own) is None or not known:
        return None, round(got[0], 4), "context: no strong rate"
    fill = float(np.mean(known))
    num = share[own] * r[own]
    den = sum(share[k] * (fill if r[k] is None else r[k]) for k in parts)
    return (round(got[0] * num / den, 4) if den > 0 else 0.0), round(got[0], 4), "context estimate: token coverage x the context's share of strong"


def score(a):
    P, man = _setup(a, need_prereg=True)
    from bonsai_interp.graph.protocols import PROTOCOLS
    from bonsai_interp.probe.textinv import PREAMBLES
    pre = json.loads(P["prereg"].read_text(encoding="utf-8"))
    blob = gzip.open(P["tests"], "rb").read()
    if hashlib.sha256(blob).hexdigest() != pre["tests_sha256"]:
        sys.exit("the tests file does not match its pre-registered sha256")
    rows = [json.loads(x) for x in blob.decode("utf-8").splitlines()]
    sel = {e["tok"]: e for e in ct._read_gz(P["select"])[0]["words"]}
    items = {(r["tok"], r["claim"]): r for r in rows if r["type"] == "items" and r["class"] in ("tested", "placebo")}
    tests = []
    for r in rows:
        if r["type"] == "test" and (r["tok"], r["claim"]) in items:
            it = items[(r["tok"], r["claim"])]
            tests.append((r, it, False))
            if (r["tok"], "placebo:" + it["placebo"]) in items:
                tests.append((r, items[(r["tok"], "placebo:" + it["placebo"])], True))
    # per word: its texts and the poles read on them
    T, cols, tops, poles_of = {}, {}, {}, {}
    for r, it, _ in tests:
        T.setdefault(r["tok"], {})
        for x in it["fire"] + it["quiet"]:
            T[r["tok"]].setdefault(x, len(T[r["tok"]]))
        L = int(r["neuron"].split("/L")[1].split("/")[0])
        poles_of.setdefault(r["tok"], {})[r["neuron"]] = (L, 0 if r["neuron"].endswith("+") else 1, int(r["neuron"].split("/N")[1][:-1]))
    poles, colix = {}, {}
    for t in T:
        poles[t] = list(poles_of[t].items())
        cols[t], cx = _cols([p for _, p in poles[t]])
        colix[t] = dict(zip([k for k, _ in poles[t]], cx))
        tops[t] = max(p[0] for _, p in poles[t])
    print(f"scoring {len(tests)} tests on {sum(len(v) for v in T.values())} snippets x {len(PREAMBLES)} preambles", flush=True)
    raw, minutes = _read({t: list(v) for t, v in T.items()}, cols, tops, PREAMBLES, a.batch, a.smoke and a.fake)
    pm = R1._pm()
    rule = PROTOCOLS[JUDGE]["rule"]
    out = []
    for r, it, plac in tests:
        t = r["tok"]
        L, p, n = poles_of[t][r["neuron"]]
        v = raw[t][:, colix[t][r["neuron"]]] * (1.0 if p == 0 else -1.0) / pm[L, p, n]
        fa = [round(float(v[T[t][x]]), 4) for x in it["fire"]]
        qa = [round(float(v[T[t][x]]), 4) for x in it["quiet"]]
        res = rule({"per_sentence": [{"set": "fire", "act": x} for x in fa] + [{"set": "quiet", "act": x} for x in qa]})
        F, Q = np.array(fa), np.array(qa)
        o = {"neuron": r["neuron"], "layer": L, "tok": t, "word": it["word"], "claim": r["claim"], "axis": TW.CLAIMS[r["claim"]][0],
             "label": it["label"] if not plac else "placebo", "placebo": plac, "outcome": res,
             "auroc": round(float(((F[:, None] > Q[None]) + 0.5 * (F[:, None] == Q[None])).mean()), 3), "med_fire": round(float(np.median(F)), 3),
             "fire": fa, "quiet": qa, "select": r["select"]}
        if res == "held" and not plac:
            o["_cov"] = (r, t)
        out.append(o)
    # coverage of the held claims, layer by layer: graph.coverage caches 4 layer tables, and a table costs ~6 s to load,
    # so test order (by word, every layer mixed) reloaded a table for almost every held claim (amendment 1)
    for o in sorted((o for o in out if "_cov" in o), key=lambda o: o["layer"]):
        r, t = o.pop("_cov")
        e = sel[t]
        sr = next((s for s in e["claims"][r["claim"]]["selected"] if s["neuron"] == r["neuron"]), {})
        o["coverage"], o["coverage_bound"], o["coverage_method"] = _coverage(r["neuron"], r["claim"], e["info"], e["counts"], sr.get("rate", {}))
        o["seed"] = sr.get("seed")
    sha = ct._write_gz(P["results"], out)
    summ = summarize(out, sel)
    summ.update({"protocol": PROTOCOL, "scored": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "toolchain_commit": ct._commit(),
                 "prereg_commit": (R1._alone(P["prereg"]) or "")[:12] if not a.smoke else None, "minutes_read": round(minutes, 1),
                 "results_file": P["results"].name, "results_sha256": sha})
    P["summary"].write_text(json.dumps(summ, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: summ[k] for k in ("by_axis", "poles_held", "coverage")}, indent=1))


def summarize(out, sel):
    def band(L):
        return next(f"{a}-{b}" for a, b in BANDS if a <= L <= b)
    by_axis, by_claim, by_band = {}, {}, {}
    for o in out:
        k = "placebo" if o["placebo"] else "real"
        for d, key in ((by_axis, o["axis"]), (by_claim, o["claim"]), (by_band, band(o["layer"]))):
            e = d.setdefault(key, {"real": {}, "placebo": {}})
            e[k][o["outcome"]] = e[k].get(o["outcome"], 0) + 1
    held = [o for o in out if o["outcome"] == "held" and not o["placebo"]]
    cov = np.array([o["coverage"] for o in held if o.get("coverage") is not None])
    # previously partial: the seed's auto-token@2 test did not hold, or the word covered < 50% of the pole's strong activations
    part = [o for o in held if o.get("seed") and (o["seed"][0] != "held" or (o["seed"][1] or 0) < 0.5)]
    part_poles = {o["neuron"] for o in part}
    gained = {o["neuron"] for o in part if (o.get("coverage") or 0) >= 0.5}
    hist = np.histogram(cov, bins=[0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0001])[0].tolist() if len(cov) else []
    selected = {c: sum(len(e["claims"].get(c, {}).get("selected", [])) for e in sel.values()) for c in TW.CLAIMS}
    placebo_sel = {c: sum(e["claims"].get(c, {}).get("placebo_selected") or 0 for e in sel.values()) for c in TW.CLAIMS}
    return {"by_axis": by_axis, "by_claim": by_claim, "by_band": by_band, "selected": selected, "placebo_selected": placebo_sel,
            "tests": sum(not o["placebo"] for o in out), "held": len(held), "poles_held": len({o["neuron"] for o in held}),
            "words_with_held": len({o["tok"] for o in held}),
            "partial_seeds_sharpened": {"held_tests": len(part), "poles": len(part_poles), "poles_reaching_assignment": len(gained)},
            "coverage": {"measured": int(len(cov)), "median": round(float(np.median(cov)), 4) if len(cov) else None,
                         "at_or_above_0.5": int((cov >= 0.5).sum()), "bins": [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0], "hist": hist}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["protocol", "select", "build", "score"])
    ap.add_argument("--smoke", help="a scratch directory: no manifests")
    ap.add_argument("--pilot", action="store_true", help="select: save every pole's statistics (calibration)")
    ap.add_argument("--words", type=int, help="the first K seed words only")
    ap.add_argument("--split", choices=["select", "test"], help="build: the split to draw from (smoke runs: select)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--fake", action="store_true", help="with --smoke: random numbers instead of GPU reads (CPU check)")
    a = ap.parse_args()
    {"protocol": protocol, "select": select, "build": build, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
