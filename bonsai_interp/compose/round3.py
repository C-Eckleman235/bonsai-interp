"""compose@3: cover poles with conditions. Each pole's sampled strong activations (events/sample.py) are matched against
the condition library (conds.py: domain gates, part of speech, morphology, grammatical roles, code syntax roles); a
hypothesis is selected only when it already covers at least half of the pole, so a hold is assignment-grade by
construction (the survey's 50% rule, docs/HARNESS.md).

  python -m bonsai_interp.compose.round3 protocol    # write the manifest; commit it ALONE
  python -m bonsai_interp.compose.round3 select      # CPU: hypotheses from the event sample (selection split)
  python -m bonsai_interp.compose.round3 build       # CPU: held-out 8 + 8 tests from the test split; commit the prereg ALONE
  python -m bonsai_interp.compose.round3 score       # GPU: grammar/run.py's scorer (quick-8+8@2, 3 preambles)
  python -m bonsai_interp.compose.round3 report      # CPU
  (--smoke DIR: no manifests, files in DIR, at most --max hypotheses; its pairs come from the selection split)

Hypotheses, per eligible pole not already assignment-grade, with >= SEL["min_events"] sampled strong activations in at
least SEL["min_windows"] windows:
  1. Condition c: its coverage (share of the pole's sampled activations where c holds) >= SEL["min_cov"] and its lift
     (coverage / c's base rate over the sampled windows) >= SEL["min_lift"]. The pole's c is the one with the highest
     lift among those.
  2. Tokens: T = the fewest of the pole's commonest tokens (where c holds) that cover SEL["token_share"] of those
     activations, at most SEL["max_tokens"]. When c and T together still cover >= SEL["min_cov"] of the pole, the
     hypothesis is the conjunction "a token of T where c holds" (kind conj), tested as the same token with c on against c
     off: it holds only if c matters beyond the token. Otherwise it is c alone (kind class), tested against tokens of the
     same corpus-frequency band where c fails, in the same domain, and in c's natural complement where its pass defines
     one (sweeplib@3/4/5's quiet side: present against past verbs, plain against superlative adjectives, the same word in
     another grammatical role, code tokens outside the code role).
  Hypotheses are ranked by coverage x log(lift); the best SEL["cap"] are tested. Each test has a placebo (two snippets
  where c fails, same keys) read on the same pole.
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
from bonsai_interp.compose import conds as CD
from bonsai_interp.events import sample as EV
from bonsai_interp.grammar import cons as C
from bonsai_interp.grammar import run as GR
from bonsai_interp.sweeplib import kinds as SK

ROOT = ct.ROOT
PROTOCOL, KIND, BASE = "compose@3", "cover", "bonsai1.compose3"
OUT = ROOT / "data" / "compose"
SEL = {"min_events": 32, "min_windows": 8, "min_cov": 0.5, "min_lift": 2.0, "token_share": 0.8, "max_tokens": 8, "cap": 6000}
PER_KEY = {"conj": {"select": 4, "test": 2}, "class": {"select": 16, "test": 2}}
CAP = {"conj": 24, "class": 400}
CODE = [Path(__file__), Path(CD.__file__), Path(EV.__file__)]


def _paths(d=None):
    o, m = (Path(d), Path(d)) if d else (OUT, ROOT / "graph" / "auto")
    return {"protocol": m / f"{BASE}.protocol.json", "prereg": m / f"{BASE}.prereg.json", "summary": m / f"{BASE}.summary.json",
            "select": o / f"{BASE}.select.json.gz", "tests": o / f"{BASE}.tests.jsonl.gz", "results": o / f"{BASE}.results.jsonl.gz",
            "coverage": o / f"{BASE}.coverage.jsonl.gz", "report": o / f"{BASE}.report.json",
            "events": EV.OUT / "bonsai1.events.npz", "parse_select": OUT / "unused", "parse_test": OUT / "unused"}


def _install():
    """Point grammar/run.py's scorer at this round."""
    GR._paths, GR.PROTOCOL, GR.KIND = _paths, PROTOCOL, KIND
    GR.CODE = list(CODE)


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def protocol(a):
    _install()
    for p in CODE:
        if subprocess.run(["git", "status", "--porcelain", "--", str(p)], cwd=ROOT, capture_output=True, text=True).stdout.strip():
            sys.exit(f"commit {p.relative_to(ROOT)} first: the manifest pins the committed code")
    P = _paths()
    man = {"protocol": PROTOCOL, "kind": KIND, "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "toolchain_commit": ct._commit(), "code_sha256": GR._code_sha(), "about": __doc__,
           "events": {"file": P["events"].name, "sha256": _sha(P["events"]), "K": EV.K, "doc": EV.__doc__},
           "conditions": CD.names(), "condition_doc": CD.__doc__, "selection": SEL, "pairs": {"per_key": PER_KEY, "cap": CAP,
           "test": C.N_TEST, "snippet": C.SNIP, "min_pos": C.MIN_POS}, "judge": GR.JUDGE,
           "read": "last token, mean over the 3 PREAMBLES (grammar/run.py score, unchanged)",
           "test_split": "English windows parsed by the grammar pass (bonsai1.parse.test.npz) and code windows lexed by sweeplib@5 "
                         "(bonsai1.lex.test.npz)",
           "coverage": "the selected hypothesis's share of the pole's sampled strong activations (selection split)"}
    P["protocol"].write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {P['protocol'].relative_to(ROOT)}: commit it ALONE before running any stage")


def _setup(a, need_prereg=False):
    _install()
    if a.smoke:
        Path(a.smoke).mkdir(parents=True, exist_ok=True)
        P = _paths(a.smoke)
        P["events"] = Path(a.events) if a.events else _paths()["events"]
        return P, None
    return GR._setup(a, need_prereg)


def _assigned():
    from bonsai_interp.compose import round2 as R2
    return {k for k, v in R2.characterized().items() if v == "assigned"}


def select(a):
    P, man = _setup(a)
    if man and man["events"]["sha256"] != _sha(P["events"]):
        sys.exit("the event sample differs from the one the protocol registered")
    from bonsai_interp import explore
    t0 = time.time()
    z = EV.load(P["events"])
    ids, mp = z["ids"], int(z["min_pos"])
    toks = explore.corpus()
    nm = CD.names()
    base = CD.base_rates(mp)
    U, inv = np.unique(ids[ids >= 0], return_inverse=True)
    print(f"{len(U):,} distinct sampled positions; evaluating {len(nm)} conditions", flush=True)
    CU = CD.at(U)  # (U, conditions)
    tokU = toks[U // 1024, U % 1024].astype(np.int64)
    uidx = np.full(ids.shape, -1, np.int64)
    uidx[ids >= 0] = inv
    done = _assigned()
    hyps = []
    for L in range(EV.LAYERS):
        elig = ct.eligibility(L)
        for p in (0, 1):
            ui = uidx[L, p]  # (N, K)
            n_ev = (ui >= 0).sum(1)
            for n in np.nonzero((n_ev >= SEL["min_events"]) & (elig[p] == "eligible"))[0]:
                if (L, int(n), "+" if p == 0 else "-") in done:
                    continue
                u = ui[n][ui[n] >= 0]
                if len(np.unique(U[u] // 1024)) < SEL["min_windows"]:
                    continue  # one or two documents: a condition would look general when it is only the document
                cm = CU[u]  # (k, conditions)
                cov = cm.mean(0)
                lift = np.where(base > 0, cov / np.maximum(base, 1e-12), 0)
                ok = (cov >= SEL["min_cov"]) & (lift >= SEL["min_lift"])
                if not ok.any():
                    continue
                c = int(np.argmax(np.where(ok, lift, -1)))
                on = cm[:, c]
                tk, cnt = np.unique(tokU[u][on], return_counts=True)
                o = np.argsort(-cnt, kind="stable")
                cum = np.cumsum(cnt[o]) / on.sum()
                k = int(np.searchsorted(cum, SEL["token_share"]) + 1)
                T = sorted(int(x) for x in tk[o][:k]) if k <= SEL["max_tokens"] else None
                ccov = float(np.isin(tokU[u], T).__and__(on).mean()) if T else 0.0
                kind = "conj" if T and ccov >= SEL["min_cov"] else "class"
                hyps.append({"neuron": ct.pid(p, int(n), L), "condition": nm[c], "kind": kind, "tokens": T if kind == "conj" else None,
                             "coverage": round(ccov if kind == "conj" else float(cov[c]), 3), "cond_coverage": round(float(cov[c]), 3),
                             "lift": round(float(lift[c]), 2), "events": int(len(u)),
                             "score": round(float((ccov if kind == "conj" else cov[c]) * np.log(lift[c])), 4)})
        if L % 6 == 5:
            print(f"  through layer {L}: {len(hyps)} hypotheses ({time.time() - t0:.0f} s)", flush=True)
    hyps.sort(key=lambda h: (-h["score"], h["neuron"]))
    kept = hyps[:a.max or SEL["cap"]]
    ct._write_gz(P["select"], [{"hypotheses": kept, "n_all": len(hyps), "minutes": round((time.time() - t0) / 60, 1),
                                "base_rates": dict(zip(nm, np.round(base, 5).tolist()))}])
    by = {}
    for h in kept:
        by[(h["kind"], h["condition"].split(":")[0])] = by.get((h["kind"], h["condition"].split(":")[0]), 0) + 1
    print(f"wrote {P['select']}: {len(hyps)} hypotheses, kept {len(kept)}; by kind and family {by}", flush=True)


def _test_masks(split="test"):
    """Per source (English parse, code lex) of a split: windows, token ids, {condition: fire mask}, the natural complements,
    and where a snippet may end (the token ends its word / is lexed and visible). A smoke run builds from the selection
    split, so the test split stays unread until the tests are registered."""
    from bonsai_interp import explore
    from bonsai_interp.codesyn import lex as X
    from bonsai_interp.grammar import parse as PA
    toks = explore.corpus()
    out = []
    for path, fn, kind in ((ROOT / "data" / "grammar" / f"bonsai1.parse.{split}.npz", CD._english, "english"),
                           (ROOT / "data" / "codesyn" / f"bonsai1.lex.{split}.npz", CD._code, "code")):
        z = PA.load(path) if kind == "english" else X.load(path)
        ids = toks[z["windows"]].astype(np.int64)
        M = fn(z, ids)
        Mq = fn(z, ids, quiet=True)
        if kind == "english":
            can = z["last"] & (z["pos"] > 0)
        else:
            T_ = ids.shape[1]
            can = z["ok"] & (z["start"] >= 0) & (np.arange(T_)[None, :] - z["start"] <= C.SNIP - 2)
        can[:, :C.MIN_POS + C.SNIP - 1] = False
        out.append((kind, z["windows"], ids, M, Mq, can))
    return out


def battery(hyps, split="test"):
    """{item key: [(fire Snip, quiet Snip, stratum)], "placebo:<key>": [...]} on a split (the test split, or the selection
    split for a smoke run)."""
    from bonsai_interp import explore
    from bonsai_interp.grammar import posmorph as PM
    toks, dom = explore.corpus(), ct._domains()
    band = PM._band()
    src = _test_masks(split)
    out = {}
    for key, (c, T) in sorted(hyps.items()):
        var = "conj" if T else "class"
        Fk, Fw, Fp, Qk, Qw, Qp = [], [], [], [], [], []
        for kind, win, ids, M, Mq, can in src:
            if c not in M:
                continue  # the condition's parse does not apply here (no fire snippet; no quiet from another domain)
            f = M[c] & can
            q = ~M[c] & can
            if not T and c in Mq:  # class: the condition's natural complement (past against present verbs, ...)
                q &= Mq[c]
            if T:
                tm = np.isin(ids, T)
                f, q = f & tm, q & tm
            fw, fp = np.nonzero(f)
            qw, qp = np.nonzero(q)
            kf, kq = ids[fw, fp], ids[qw, qp]
            if not T:
                kf, kq = band[kf], band[kq]
            Fk += kf.tolist(); Fw += win[fw].tolist(); Fp += fp.tolist()  # noqa: E702
            Qk += kq.tolist(); Qw += win[qw].tolist(); Qp += qp.tolist()  # noqa: E702
        seed = [19, sum(map(ord, key)) % 100003, 1]
        rng = np.random.default_rng(seed + [5])
        F = C._cap(rng, Fk, Fw, Fp, CAP[var])
        Q = C._cap(rng, Qk, Qw, Qp, CAP[var], ["off"] * len(Qk))
        pairs = SK._pairs(F, Q, C.N_TEST, PER_KEY[var]["test"], seed)
        if len(pairs) < C.N_TEST:
            pairs = SK._pairs(F, Q, C.N_TEST, 4 * PER_KEY[var]["test"], seed)
        plac = C._placebo(Q, C.N_TEST, seed + [2], 4 * PER_KEY[var]["test"])

        def snip(e):
            return SK.Snip(e.w, e.p, int(dom[e.w]), toks[e.w, e.p - C.SNIP + 1:e.p + 1])
        out[key] = [(snip(f), snip(q), st) for f, q, st in pairs]
        out["placebo:" + key] = [(snip(x), snip(y), st) for x, y, st in plac]
    return out


def _key(h):
    return h["condition"] + ("|" + ",".join(map(str, h["tokens"])) if h["tokens"] else "|class")


def build(a):
    P, man = _setup(a)
    from bonsai_interp.sweeplib import run as R
    sel = ct._read_gz(P["select"])[0]
    hyps = sel["hypotheses"]
    items = {_key(h): (h["condition"], h["tokens"]) for h in hyps}
    t0 = time.time()
    bat = battery(items, "select" if a.smoke else "test")
    print(f"batteries for {len(items)} item sets in {time.time() - t0:.0f} s", flush=True)
    tk = __import__("bonsai_interp.explore", fromlist=["tokenizer"]).tokenizer()
    rows = []
    for key in sorted(items):
        c, T = items[key]
        words = [tk.decode([t]) for t in T] if T else None
        lab = (f"compose: fires on {', '.join(repr(w) for w in words)} where {c} holds, not on the same tokens elsewhere" if T else
               f"compose: fires where {c} holds, whatever the token, not on tokens of the same frequency where it fails")
        pairs, plac = bat[key], bat["placebo:" + key]
        item = {"type": "items", "kind": KIND, "condition": key, "label": lab}
        if len(pairs) < C.N_TEST:
            rows.append({**item, "class": "untestable", "why": f"only {len(pairs)} held-out pairs"})
            continue
        rows.append({**item, "class": "tested", "fire": [R._text(f) for f, _, _ in pairs], "quiet": [R._text(q) for _, q, _ in pairs],
                     "quiet_kind": [st for _, _, st in pairs], "windows": [[f.w, f.p, q.w, q.p] for f, q, _ in pairs]})
        if len(plac) >= C.N_TEST:
            rows.append({"type": "items", "kind": KIND, "condition": "placebo:" + key, "label": "placebo", "class": "placebo",
                         "fire": [R._text(f) for f, _, _ in plac], "quiet": [R._text(q) for _, q, _ in plac],
                         "quiet_kind": ["placebo"] * len(plac), "windows": [[f.w, f.p, q.w, q.p] for f, q, _ in plac]})
    tested = {r["condition"] for r in rows if r["type"] == "items" and r["class"] == "tested"}
    for h in hyps:
        if _key(h) in tested:
            rows.append({"type": "test", "neuron": h["neuron"], "kind": KIND, "condition": _key(h),
                         "select": {k: h[k] for k in ("coverage", "cond_coverage", "lift", "events", "kind")}})
    sha = ct._write_gz(P["tests"], rows)
    counts = {"items": len(tested), "untestable": sum(r["type"] == "items" and r["class"] == "untestable" for r in rows),
              "tests": sum(r["type"] == "test" for r in rows)}
    pre = {"protocol": PROTOCOL, "protocol_commit": GR.registered() if not a.smoke else None,
           "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "tests_file": P["tests"].name, "tests_sha256": sha,
           "select_file": P["select"].name, "select_sha256": hashlib.sha256(gzip.open(P["select"], "rb").read()).hexdigest(),
           "counts": counts}
    P["prereg"].write_text(json.dumps(pre, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(counts), f"\nwrote {P['tests']} and {P['prereg']}: commit the manifest ALONE before scoring", flush=True)


def report(a):
    P = _paths(a.smoke) if a.smoke else _paths()
    _install()
    from bonsai_interp.graph.protocols import v2_stats
    res = ct._read_gz(P["results"])
    sel = {_key(h) + "@" + h["neuron"]: h for h in ct._read_gz(P["select"])[0]["hypotheses"]}
    pre = json.loads(P["prereg"].read_text(encoding="utf-8")) if P["prereg"].exists() else {}
    band = {L: f"{lo}-{hi}" for lo, hi in GR.BANDS for L in range(lo, hi + 1)}
    tab = {}
    rows = []
    for r in res:
        key = r["condition"].removeprefix("placebo:")
        h = sel[key + "@" + r["neuron"]]
        L = GR._neuron(r["neuron"])[0]
        fam = h["condition"].split(":")[0]
        for k in (("all",), ("kind", h["kind"]), ("family", fam), ("band", band[L])):
            e = tab.setdefault(" ".join(k), {"tests": 0, "held": 0, "placebo_tests": 0, "placebo_held": 0})
            e["placebo_tests" if r["placebo"] else "tests"] += 1
            e["placebo_held" if r["placebo"] else "held"] += r["outcome"] == "held"
        if not r["placebo"] and r["outcome"] == "held":
            st = v2_stats(r)
            rows.append({"neuron": r["neuron"], "condition": h["condition"], "tokens": h["tokens"], "kind": h["kind"], "label": r["label"],
                         "auroc": round(st[0], 3), "median_fire": round(st[1], 3),
                         "coverage": {"share": h["coverage"], "cond_share": h["cond_coverage"], "lift": h["lift"], "events": h["events"]}})
    ct._write_gz(P["coverage"], rows)
    poles = {r["neuron"] for r in rows}
    summ = {"protocol": PROTOCOL, "kind": KIND, "scored": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "toolchain_commit": ct._commit(), "prereg_commit": (GR.R._alone(P["prereg"]) or "")[:12] if not a.smoke else None,
            "tests": tab.get("all", {}).get("tests", 0), "held": len(rows), "poles_held": len(poles),
            "placebo_tests": tab.get("all", {}).get("placebo_tests", 0), "placebo_held": tab.get("all", {}).get("placebo_held", 0),
            "breakdown": tab, "poles_held_by_layer": np.bincount([GR._neuron(x)[0] for x in poles], minlength=36).tolist() if poles else [0] * 36,
            "results_file": P["results"].name, "results_sha256": hashlib.sha256(gzip.open(P["results"], "rb").read()).hexdigest(),
            "coverage_file": P["coverage"].name, "tests_sha256": pre.get("tests_sha256")}
    P["summary"].write_text(json.dumps(summ, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: summ[k] for k in ("tests", "held", "poles_held", "placebo_tests", "placebo_held", "breakdown")}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["protocol", "select", "build", "score", "report"])
    ap.add_argument("--smoke")
    ap.add_argument("--events", help="with --smoke: an event sample other than the registered one")
    ap.add_argument("--max", type=int, help="keep at most this many hypotheses (smoke)")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    if a.cmd == "score":
        _setup(a)
        return GR.score(a)
    {"protocol": protocol, "select": select, "build": build, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
