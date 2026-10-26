"""The grammar first pass (sweeplib@3): generic construction screens. No language model writes or judges anything.

  python -m bonsai_interp.grammar.parse select        # CPU, spaCy: parse the selection split's English windows
  python -m bonsai_interp.grammar.run protocol        # write the protocol manifest: commit it ALONE
  python -m bonsai_interp.grammar.run select          # GPU: every selection battery, all 884,736 poles at once
  python -m bonsai_interp.grammar.parse test          # CPU: parse the test split (only after the protocol is committed)
  python -m bonsai_interp.grammar.run build           # CPU: the held-out tests and their pre-registration manifest: commit
                                                      # the manifest ALONE
  python -m bonsai_interp.grammar.run score           # GPU: read the registered tests, judge with quick-8+8@2's rule
  python -m bonsai_interp.grammar.run report          # CPU: by construction, layer band and word-screen bin; coverage
  (select / build / score take --smoke DIR: no manifests, files in DIR; with --pilot, the full batteries)

A condition (cons.py) is a grammatical role of a snippet's last word, from the spaCy parse of its whole window. Its pairs
end on the same token (or, for the pos: variants, the same part of speech), so the quiet snippet is the same word in another
role. The reader, the pair sampler, the selection rule, the placebo pairs and tests, and the judge are the sweep library's
(sweeplib@1), reused by import: a 48-token snippet read at its last token after a neutral preamble, one read scoring every
pole of every layer.
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
from bonsai_interp.grammar import cons as C
from bonsai_interp.grammar import parse as PA
from bonsai_interp.sweeplib import run as R

ROOT = ct.ROOT
PROTOCOL = "sweeplib@3"
KIND = "grammar"
OUT = ROOT / "data" / "grammar"
MAN = ROOT / "graph" / "auto"
N, LAYERS = R.N, R.LAYERS
JUDGE = R.JUDGE
# sweeplib@1's selection rule loosened (the pilot on the selection split, 2026-10-25: placebo pairs selected 0 poles in every
# condition under the registered rule, a stricter one and this one; the registered rule selected 73 poles, this one 325). The
# held-out test applies the judge's bar, and placebo tests count its false holds.
SELECT = {"auroc": 0.80, "med_fire": 0.10, "quiet_hi": 0.30, "quiet_hi_share": 0.15, "win": 0.70, "mean_diff": 0.07}
SEPARATE = 0.85  # diagnostic, not a selection: poles whose selection pairs separate at AUROC >= this, at any magnitude
BANDS = [(0, 5), (6, 11), (12, 17), (18, 23), (24, 29), (30, 35)]
CODE = [Path(PA.__file__), Path(C.__file__), Path(__file__)]


def _paths(d=None):
    o, m = (Path(d), Path(d)) if d else (OUT, MAN)
    b = "bonsai1.sweeplib3"
    return {"protocol": m / f"{b}.protocol.json", "prereg": m / f"{b}.prereg.json", "summary": m / f"{b}.summary.json",
            "select": o / f"{b}.select.json.gz", "tests": o / f"{b}.tests.jsonl.gz", "results": o / f"{b}.results.jsonl.gz",
            "coverage": o / f"{b}.coverage.jsonl.gz", "report": o / f"{b}.report.json",
            "parse_select": OUT / "bonsai1.parse.select.npz", "parse_test": (Path(d) if d else OUT) / "bonsai1.parse.test.npz"}
    # (a smoke run's test parse lives in its own directory, so the pipeline can be checked before the test split is parsed)


def _code_sha():
    h = hashlib.sha256()
    for p in CODE:
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def _sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def registered():
    """The protocol manifest's first commit, when it was committed alone and is unchanged since (else None)."""
    return R._alone(_paths()["protocol"])


def _setup(a, need_prereg=False):
    if a.smoke:
        Path(a.smoke).mkdir(parents=True, exist_ok=True)
        return _paths(a.smoke), None
    P = _paths()
    if not P["protocol"].exists() or not registered():
        sys.exit(f"commit {P['protocol'].relative_to(ROOT)} on its own before running any stage (run `protocol` first)")
    man = json.loads(P["protocol"].read_text(encoding="utf-8"))
    if _code_sha() != man["code_sha256"]:
        sys.exit("the grammar screens' code changed since the protocol was registered: that is a new protocol version")
    if need_prereg and (not P["prereg"].exists() or not R._alone(P["prereg"])):
        sys.exit(f"commit {P['prereg'].relative_to(ROOT)} on its own before scoring (run `build` first)")
    OUT.mkdir(parents=True, exist_ok=True)
    return P, man


def protocol(a):
    for p in CODE:
        if subprocess.run(["git", "status", "--porcelain", "--", str(p)], cwd=ROOT, capture_output=True, text=True).stdout.strip():
            sys.exit(f"commit {p.relative_to(ROOT)} first: the manifest pins the committed code")
    P = _paths()
    man = {"protocol": PROTOCOL, "kind": KIND, "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "toolchain_commit": ct._commit(), "code_sha256": _code_sha(),
           "reused": "bonsai_interp/sweeplib (sweeplib@1): read_last, _pairs, selection rule and statistics, judge; unchanged",
           "parser": {"model": PA.MODEL, "domains": "edu, web (English)", "windows_per_split": PA.SAMPLE, "splits": PA.SPLITS,
                      "parse_select_sha256": _sha_file(P["parse_select"]) if P["parse_select"].exists() else None,
                      "test_split": "parsed only after this manifest is committed"},
           "layers": list(range(LAYERS)), "snippet": C.SNIP, "min_pos": C.MIN_POS,
           "pairs": {"select": C.N_SELECT, "min_select": C.MIN_SELECT, "test": C.N_TEST, "per_key": C.PER_KEY, "few_keys": C.FEW,
                     "cap": C.CAP},
           "last_token": "a whole alphabetic word that ends its spaCy token (or a possessive 's), at position >= 111",
           "conditions": {c: {"label": lab, "cue": C.CONDITIONS[c.removeprefix("pos:")][1],
                              "key": "part of speech" if c.startswith("pos:") else "token"} for c, lab in C.labels().items()},
           "selection": SELECT, "separating_diagnostic": f"poles with selection AUROC >= {SEPARATE} at any magnitude (not a selection)",
           "excluded": "weak poles (corpus max < 0.5)", "judge": JUDGE,
           "read": "last token, mean over the 3 PREAMBLES (selection: PREAMBLES[0])",
           "placebo": "per condition: selection on quiet-vs-quiet pairs (false selections) and an 8 + 8 quiet-vs-quiet test on the "
                      "selected poles (false holds)",
           "report": "held / failed / placebo by construction, by layer band, and by the pole's atlas bin (auto-token@2 word screen); "
                     "coverage of held conditions: the share of the pole's 32 strongest corpus activations (scan, English windows) "
                     "where the condition holds, against the condition's base rate"}
    P["protocol"].write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {P['protocol'].relative_to(ROOT)}: commit it ALONE before running any stage")


def _selected(st, ok):
    return ok & (st["auroc"] >= SELECT["auroc"]) & (st["med_fire"] >= SELECT["med_fire"]) & \
        (st["quiet_hi_share"] <= SELECT["quiet_hi_share"]) & (st["win"] >= SELECT["win"]) & (st["mean_diff"] >= SELECT["mean_diff"])


def select(a):
    P, man = _setup(a)
    from bonsai_interp.env import limit_memory
    from bonsai_interp.probe.textinv import PREAMBLES
    if not P["parse_select"].exists():
        sys.exit("parse the selection split first: python -m bonsai_interp.grammar.parse select")
    if man and man["parser"]["parse_select_sha256"] != _sha_file(P["parse_select"]):
        sys.exit("the selection parse differs from the one the protocol registered")
    conds = a.conds or list(C.labels())
    t0 = time.time()
    bat = C.battery(PA.load(P["parse_select"]), "select", conds)
    print(f"batteries built in {time.time() - t0:.0f} s", flush=True)
    ok = R._not_weak()
    limit_memory(0.8)
    model, tok, rec = ct._gpu_load()
    res = {"conditions": {}}
    for c in conds:
        pairs, plac = bat[c], bat["placebo:" + c]
        if a.smoke and not a.pilot:
            pairs, plac = pairs[:12], plac[:12]
        entry = {"pairs": len(pairs), "keys": len({int(f.ids[-1]) for f, _, _ in pairs}), "strata": {}}
        for _, _, st in pairs:
            entry["strata"][st] = entry["strata"].get(st, 0) + 1
        if len(pairs) < (4 if a.smoke and not a.pilot else C.MIN_SELECT):
            entry["why"] = f"only {len(pairs)} selection pairs (needs {C.MIN_SELECT})"
            res["conditions"][c] = entry
            print(f"{c}: untestable ({entry['why']})", flush=True)
            continue
        texts = [R._text(s) for f, q, _ in pairs for s in (f, q)] + [R._text(s) for f, q, _ in plac for s in (f, q)]
        x = R.read_last(model, tok, rec, texts, PREAMBLES[0], a.batch)
        m = 2 * len(pairs)
        st = R._stats(x, np.arange(0, m, 2), np.arange(1, m, 2))
        sel = _selected(st, ok)
        stp = R._stats(x, np.arange(m, len(x), 2), np.arange(m + 1, len(x), 2)) if len(plac) >= 4 else None
        del x
        if a.smoke and a.pilot:
            np.savez_compressed(Path(a.smoke) / f"stats.{c.replace(':', '_')}.npz", **{k: v.astype(np.float16) for k, v in st.items()},
                                **({"placebo_" + k: v.astype(np.float16) for k, v in stp.items()} if stp else {}))
        entry["placebo_pairs"] = len(plac)
        entry["placebo_selected"] = int(_selected(stp, ok).sum()) if stp else None
        entry["separating"] = int((ok & (st["auroc"] >= SEPARATE)).sum())
        entry["placebo_separating"] = int((ok & (stp["auroc"] >= SEPARATE)).sum()) if stp else None
        Ls, ps, ns = np.nonzero(sel)
        entry["selected"] = [{"neuron": R._pid(int(L), int(p), int(n)), **{k: round(float(v[L, p, n]), 3) for k, v in st.items()}}
                             for L, p, n in zip(Ls, ps, ns)]
        entry["by_layer"] = np.bincount(Ls, minlength=LAYERS).tolist()
        md = np.where(ok, st["mean_diff"], -np.inf).ravel()
        entry["best"] = [{"neuron": R._pid(*map(int, np.unravel_index(i, ok.shape))),
                          **{k: round(float(v.ravel()[i]), 3) for k, v in st.items()}} for i in np.argsort(-md)[:10]]
        res["conditions"][c] = entry
        print(f"{c}: {len(pairs)} pairs, {len(entry['selected'])} poles selected, placebo {entry['placebo_selected']} "
              f"({time.time() - t0:.0f} s)", flush=True)
    ct._gpu_idle("select")
    res["minutes"] = round((time.time() - t0) / 60, 1)
    ct._write_gz(P["select"], [res])
    print(f"wrote {P['select']} in {res['minutes']} min", flush=True)


def build(a):
    """Held-out 8 + 8 tests (and a placebo 8 + 8) for every condition with selected poles. CPU only: no neuron is read."""
    P, man = _setup(a)
    if not P["parse_test"].exists():
        sys.exit("parse the test split first: python -m bonsai_interp.grammar.parse test")
    sel = ct._read_gz(P["select"])[0]
    want = [c for c, e in sel["conditions"].items() if e.get("selected")]
    bat = C.battery(PA.load(P["parse_test"]), "test", want) if want else {}
    labs = C.labels()
    rows = []
    for c in want:
        pairs, plac = bat[c], bat.get("placebo:" + c, [])
        item = {"type": "items", "kind": KIND, "condition": c, "label": labs[c]}
        if len(pairs) < C.N_TEST:
            rows.append({**item, "class": "untestable", "why": f"only {len(pairs)} held-out pairs"})
            continue
        rows.append({**item, "class": "tested", "fire": [R._text(f) for f, _, _ in pairs], "quiet": [R._text(q) for _, q, _ in pairs],
                     "quiet_kind": [st for _, _, st in pairs], "windows": [[f.w, f.p, q.w, q.p] for f, q, _ in pairs]})
        if len(plac) >= C.N_TEST:
            rows.append({"type": "items", "kind": KIND, "condition": "placebo:" + c, "label": "placebo", "class": "placebo",
                         "fire": [R._text(f) for f, _, _ in plac], "quiet": [R._text(q) for _, q, _ in plac],
                         "quiet_kind": ["placebo"] * len(plac), "windows": [[f.w, f.p, q.w, q.p] for f, q, _ in plac]})
        for r in sel["conditions"][c]["selected"]:
            rows.append({"type": "test", "neuron": r["neuron"], "kind": KIND, "condition": c,
                         "select": {k: r[k] for k in ("auroc", "med_fire", "quiet_hi_share", "win", "mean_diff")}})
    sha = ct._write_gz(P["tests"], rows)
    counts = {"items": sum(r["type"] == "items" and r["class"] == "tested" for r in rows),
              "untestable": sum(r["type"] == "items" and r["class"] == "untestable" for r in rows),
              "tests": sum(r["type"] == "test" for r in rows)}
    pre = {"protocol": PROTOCOL, "protocol_commit": registered() if not a.smoke else None,
           "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "tests_file": P["tests"].name, "tests_sha256": sha,
           "select_file": P["select"].name, "select_sha256": hashlib.sha256(gzip.open(P["select"], "rb").read()).hexdigest(),
           "parse_test_sha256": _sha_file(P["parse_test"]), "counts": counts}
    P["prereg"].write_text(json.dumps(pre, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(counts), f"\nwrote {P['tests']} and {P['prereg']}: commit the manifest ALONE before scoring", flush=True)


def _neuron(r):
    L = int(r.split("/L")[1].split("/")[0])
    return L, 0 if r.endswith("+") else 1, int(r.split("/N")[1][:-1])


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
    items = {r["condition"]: r for r in rows if r["type"] == "items" and r["class"] in ("tested", "placebo")}
    tests = []
    for r in rows:
        if r["type"] == "test" and r["condition"] in items:
            tests.append((r, items[r["condition"]], False))
            if "placebo:" + r["condition"] in items:
                tests.append(({**r, "condition": "placebo:" + r["condition"]}, items["placebo:" + r["condition"]], True))
    want = {}
    for r, it, _ in tests:
        for t in it["fire"] + it["quiet"]:
            want.setdefault(t, set()).add(_neuron(r["neuron"]))
    by_top = {}
    for t, ps in want.items():
        by_top.setdefault(max(L for L, _, _ in ps), []).append(t)
    pm = R._pm()
    limit_memory(0.8)
    model, tok, rec = ct._gpu_load()
    t0, val = time.time(), {}
    print(f"scoring {len(tests)} tests on {len(want)} snippets x {len(PREAMBLES)} preambles", flush=True)
    for top in sorted(by_top):
        ts = sorted(by_top[top])
        for b in range(0, len(ts), 256):
            sl = ts[b:b + 256]
            acc = np.zeros((len(sl), top + 1, N), np.float32)
            for pre_text in PREAMBLES:
                acc += R.read_last(model, tok, rec, sl, pre_text, a.batch, top) / len(PREAMBLES)
            for i, t in enumerate(sl):
                for L, p, n in want[t]:
                    v = acc[i, L, n] / pm[L, 0, n] if p == 0 else -acc[i, L, n] / pm[L, 1, n]
                    val[(t, L, p, n)] = round(float(v), 4)
        print(f"  texts to layer {top}: {len(ts)} ({time.time() - t0:.0f} s)", flush=True)
    ct._gpu_idle("score")
    rule = PROTOCOLS[JUDGE]["rule"]
    out = []
    for r, it, plac in tests:
        L, p, n = _neuron(r["neuron"])
        per = [{"set": "fire", "text": t, "act": val[(t, L, p, n)]} for t in it["fire"]] + \
              [{"set": "quiet", "text": t, "act": val[(t, L, p, n)], "kind": k} for t, k in zip(it["quiet"], it["quiet_kind"])]
        out.append({"neuron": r["neuron"], "kind": KIND, "condition": r["condition"], "label": "placebo" if plac else it["label"],
                    "outcome": rule({"per_sentence": per}), "per_sentence": per, "select": r.get("select"), "placebo": plac,
                    "protocol": PROTOCOL})
    sha = ct._write_gz(P["results"], out)
    print(f"wrote {P['results']} ({sha[:12]}) in {(time.time() - t0) / 60:.1f} min; next: report", flush=True)


def _bins():
    """{(L, 'N123+'): atlas bin} from data/atlas (the auto-token@2 word screen's bin; layers without an atlas file: absent)."""
    from bonsai_interp import atlas
    out = {}
    for L in range(LAYERS):
        p = atlas.ATLAS / f"bonsai1_L{L}.arrow"
        if not p.exists():
            continue
        t = atlas._read(p).select(["neuron", "sign", "bin"]).to_pydict()
        for n, s, b in zip(t["neuron"], t["sign"], t["bin"]):
            out[(L, f"N{n}{s}")] = b or "none"
    return out


def coverage(held, procs=6):
    """For each held (pole, condition): the share of the pole's 32 strongest corpus activations (the scan's top examples) in
    English windows where the condition holds at that token, and the condition's base rate over all parsed words."""
    from bonsai_interp import explore
    s = explore.scan("bonsai1")
    dom = ct._domains()
    need = {}
    for nid, c in held:
        L, p, n = _neuron(nid)
        ex = [(int(w), int(q)) for w, q in zip(s["top_win"][L, p, :, n], s["top_pos"][L, p, :, n]) if dom[int(w)] in PA.DOMAINS]
        need[(nid, c)] = ex
    ws = sorted({w for ex in need.values() for w, _ in ex})
    if not ws:
        return {}, {}
    out, tabs = PA.parse(np.array(ws), procs)
    z = {**out, **{f"{k}_names": v for k, v in tabs.items()}, "windows": np.array(ws)}
    conds = sorted({c for _, c in held})
    hm = C.holds(z, conds)
    row = {w: i for i, w in enumerate(ws)}
    zs = PA.load(_paths()["parse_select"])
    words = C.holds(zs, conds)
    base_den = (zs["last"] & np.isin(zs["pos_names"][zs["pos"]], C.WORD_POS)).sum()
    base = {c: round(float(words[c].sum() / base_den), 4) for c in conds}
    cov = {}
    for (nid, c), ex in need.items():
        hit = [bool(hm[c][row[w], q]) for w, q in ex]
        cov[(nid, c)] = {"n_en": len(ex), "holds": int(sum(hit)), "share": round(sum(hit) / len(ex), 3) if ex else None,
                         "base": base[c], "lift": round(sum(hit) / len(ex) / base[c], 1) if ex and base[c] else None}
    return cov, base


def report(a):
    P = _paths(a.smoke) if a.smoke else _paths()
    from bonsai_interp.graph.protocols import v2_stats
    res = ct._read_gz(P["results"])
    sel = ct._read_gz(P["select"])[0]
    pre = json.loads(P["prereg"].read_text(encoding="utf-8")) if P["prereg"].exists() else {}
    bins = _bins()
    band = {L: f"{lo}-{hi}" for lo, hi in BANDS for L in range(lo, hi + 1)}
    by_cond, by_band, by_bin = {}, {}, {}
    held = []
    for r in res:
        c = r["condition"].removeprefix("placebo:")
        L, _, _ = _neuron(r["neuron"])
        e = by_cond.setdefault(c, {"selected": len(sel["conditions"][c].get("selected", [])),
                                   "placebo_selected": sel["conditions"][c].get("placebo_selected"),
                                   "tests": 0, "held": 0, "placebo_tests": 0, "placebo_held": 0})
        if r["placebo"]:
            e["placebo_tests"] += 1
            e["placebo_held"] += r["outcome"] == "held"
            continue
        e["tests"] += 1
        e["held"] += r["outcome"] == "held"
        b = by_band.setdefault(band[L], {"tests": 0, "held": 0})
        b["tests"] += 1
        b["held"] += r["outcome"] == "held"
        k = bins.get((L, r["neuron"].split("/")[-1]), "no atlas")
        bb = by_bin.setdefault(k, {"tests": 0, "held": 0, "held_poles": set()})
        bb["tests"] += 1
        if r["outcome"] == "held":
            bb["held"] += 1
            bb["held_poles"].add(r["neuron"])
            held.append((r["neuron"], r["condition"]))
    for c, e in sel["conditions"].items():
        by_cond.setdefault(c, {"selected": len(e.get("selected", [])), "placebo_selected": e.get("placebo_selected"), "tests": 0,
                               "held": 0, "placebo_tests": 0, "placebo_held": 0, **({"why": e["why"]} if "why" in e else {})})
    for v in by_bin.values():
        v["held_poles"] = len(v["held_poles"])
    cov, base = coverage(held, a.procs) if held else ({}, {})
    rows = []
    for r in res:
        if r["placebo"] or r["outcome"] != "held":
            continue
        st = v2_stats(r)
        L, _, _ = _neuron(r["neuron"])
        rows.append({"neuron": r["neuron"], "condition": r["condition"], "label": r["label"], "auroc": round(st[0], 3),
                     "median_fire": round(st[1], 3), "bin": bins.get((L, r["neuron"].split("/")[-1]), "no atlas"),
                     "coverage": cov.get((r["neuron"], r["condition"]))})
    ct._write_gz(P["coverage"], rows)
    poles = {r["neuron"] for r in rows}
    held_by_layer = np.bincount([_neuron(x)[0] for x in poles], minlength=LAYERS).tolist() if poles else [0] * LAYERS
    covs = [r["coverage"]["share"] for r in rows if r["coverage"] and r["coverage"]["share"] is not None]
    summ = {"protocol": PROTOCOL, "kind": KIND, "scored": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "toolchain_commit": ct._commit(), "prereg_commit": (R._alone(P["prereg"]) or "")[:12] if not a.smoke else None,
            "tests": sum(not r["placebo"] for r in res), "held": len(rows), "poles_held": len(poles),
            "placebo_tests": sum(r["placebo"] for r in res), "placebo_held": sum(r["placebo"] and r["outcome"] == "held" for r in res),
            "by_condition": by_cond, "by_layer_band": by_band, "by_atlas_bin": by_bin, "poles_held_by_layer": held_by_layer,
            "coverage": {"median_share_top32": round(float(np.median(covs)), 3) if covs else None, "n": len(covs), "base_rate": base},
            "results_file": P["results"].name, "results_sha256": hashlib.sha256(gzip.open(P["results"], "rb").read()).hexdigest(),
            "coverage_file": P["coverage"].name, "tests_sha256": pre.get("tests_sha256")}
    P["summary"].write_text(json.dumps(summ, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: summ[k] for k in ("tests", "held", "poles_held", "placebo_tests", "placebo_held", "by_layer_band",
                                           "by_atlas_bin", "coverage")}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["protocol", "select", "build", "score", "report"])
    ap.add_argument("--smoke", help="a scratch directory: no manifests")
    ap.add_argument("--pilot", action="store_true", help="with --smoke: the full batteries (an exploratory run on the selection split)")
    ap.add_argument("--conds", nargs="*", help="select only these conditions")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--procs", type=int, default=6, help="report: spaCy processes for the coverage parse")
    a = ap.parse_args()
    {"protocol": protocol, "select": select, "build": build, "score": score, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
