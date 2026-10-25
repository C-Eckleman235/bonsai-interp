"""Search everything the survey knows about poles: tested labels (graph/index.json), the sweep library's automatic tests and
the word screen's automatic tests (the atlas). CPU only, no model.

  python -m bonsai_interp.graph.play search "german"                  # any source, words in the label or context
  python -m bonsai_interp.graph.play search --kind lang --held        # every pole whose sweep-library lang test held
  python -m bonsai_interp.graph.play search --condition script/arabic --layers 30-35
  python -m bonsai_interp.graph.play search "year" --source survey    # only tested survey labels

Rows come in order of evidence: a held survey label, then a held sweep-library test, then a held word-screen test, then
the rest. Automatic results are scrutiny 0: hypotheses that passed a held-out 8 + 8, not survey labels.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RANK = {("survey", "held"): 0, ("survey", "partial"): 1, ("library", "held"): 1, ("screen", "held"): 2}


def _layers(spec):
    if not spec:
        return None
    out = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return out


def rows(text=None, kind=None, condition=None, held=False, layers=None, source=None):
    """Matching rows: (rank, source, pole id, status, what) dicts."""
    pat = re.compile(re.escape(text), re.I) if text else None
    out = []
    if source in (None, "survey") and not kind and not condition:
        ix = json.loads((ROOT / "graph" / "index.json").read_text(encoding="utf-8"))["neurons"]
        for nid, v in ix.items():
            if not nid.startswith("bonsai1/"):
                continue
            L = int(nid.split("/L")[1].split("/")[0])
            lab = v.get("label") or ""
            if (layers and L not in layers) or (held and v.get("status") != "held") or (pat and not pat.search(lab)):
                continue
            out.append({"source": "survey", "pole": nid, "layer": L, "status": v.get("status"), "what": lab})
    if source in (None, "library", "screen"):
        from bonsai_interp import atlas
        for L in sorted(layers or range(36)):
            t = atlas.load(L)
            if t is None:
                continue
            cols = t.column_names
            d = t.select([c for c in ("neuron", "sign", "bin", "context", "test", "lib_kind", "lib_condition", "lib_test", "lib_label",
                                      "lib_held") if c in cols]).to_pydict()
            for i, n in enumerate(d["neuron"]):
                nid = f"bonsai1/L{L}/N{n}{d['sign'][i]}"
                lk = d.get("lib_kind", [None] * len(d["neuron"]))[i]
                if source in (None, "library") and lk:
                    lc, lt = d["lib_condition"][i], d["lib_test"][i]
                    lab = d.get("lib_label", [None] * len(d["neuron"]))[i] or ""
                    held_under = d.get("lib_held", [[]] * len(d["neuron"]))[i] or []
                    ok = (not kind or lk == kind or any(h.startswith(kind + ":") for h in held_under)) and \
                         (not condition or lc == condition or any(h.endswith(":" + condition) for h in held_under)) and \
                         (not held or lt == "held") and (not pat or pat.search(f"{lk}:{lc} {lab}"))
                    if ok:
                        out.append({"source": "library", "pole": nid, "layer": L, "status": lt,
                                    "what": f"{lk}:{lc}" + (f" — {lab}" if lab else "") + (f" (also {', '.join(h for h in held_under if h != f'{lk}:{lc}')})"
                                                                                            if len(held_under) > 1 else "")})
                b = d.get("bin", [None] * len(d["neuron"]))[i]
                if source in (None, "screen") and b and not kind and not condition and b not in ("weak", "unexplained", "rare"):
                    ctx, tst = d["context"][i] or "", d["test"][i]
                    if (held and tst != "held") or (pat and not pat.search(f"{b} {ctx}")):
                        continue
                    out.append({"source": "screen", "pole": nid, "layer": L, "status": tst or "untested", "what": f"{b}" + (f": {ctx}" if ctx else "")})
    for r in out:
        r["rank"] = RANK.get((r["source"], r["status"]), 3)
    out.sort(key=lambda r: (r["rank"], r["layer"], r["pole"]))
    return out


def main(a):
    rs = rows(a.text, a.kind, a.condition, a.held, _layers(a.layers), a.source)
    if a.json:
        print(json.dumps(rs[:a.limit], ensure_ascii=False, indent=1))
        return
    by = {}
    for r in rs:
        by[(r["source"], r["status"])] = by.get((r["source"], r["status"]), 0) + 1
    print(f"{len(rs)} matches: " + ", ".join(f"{s} {st} {n}" for (s, st), n in sorted(by.items(), key=lambda kv: RANK.get(kv[0], 3))))
    for r in rs[:a.limit]:
        print(f"  {r['pole']:22s} {r['source']:8s} {str(r['status']):9s} {r['what'][:150]}")
    if len(rs) > a.limit:
        print(f"  ... {len(rs) - a.limit} more (--limit, --json)")


# ---------------------------------------------------------------- mission queues (the user, 2026-10-25: "missions are best spent on weird
# and ambiguous neurons. But there is some value to putting more scrutiny on held neurons to flesh out and clarify their definition.")

UNSETTLED = ("contested", "unresolved", "failed", "probed")


def queue(mode, layers=None):
    """Ranked mission candidates, deterministic from the atlas and graph/index.json.
    ambiguous: poles nothing explains yet, loudest first.
      1. Survey poles people could not settle (contested, unresolved, failed, probed).
      2. Unsurveyed eligible poles that fire often (strong count at or above the layer median), whose word screen found
         no word covering a tenth of their strong activations, and whose automatic tests did not hold.
    deepen: held labels that may be more than they say. First (tier 0) the 'partial' survey labels, whose measured coverage
      is under half of the pole. Then held survey or automatic labels whose top word covers under half of the pole's strong
      activations (the word is not the whole story), or that held under two or more sweep-library conditions.
    Score: strong activations x (1 - top word coverage), so a loud pole that no single word explains comes first."""
    from bonsai_interp import atlas
    ix = json.loads((ROOT / "graph" / "index.json").read_text(encoding="utf-8"))["neurons"]
    out = []
    for L in sorted(layers or range(36)):
        t = atlas.load(L)
        if t is None:
            continue
        d = t.select([c for c in ("neuron", "sign", "class", "strong", "hit_coverage", "hit_precision", "hit_tok", "bin", "test", "lib_test", "lib_held",
                                  "lib_kind", "lib_condition") if c in t.column_names]).to_pydict()
        strong = [s or 0 for s in d["strong"]]
        med = sorted(strong)[len(strong) // 2]
        get = lambda k, i, dflt=None: d[k][i] if k in d else dflt
        for i, n in enumerate(d["neuron"]):
            nid = f"bonsai1/L{L}/N{n}{d['sign'][i]}"
            rec = ix.get(nid) or {}
            st = rec.get("status")
            cov = (d["hit_coverage"][i] or [0])[0] if d["hit_coverage"][i] else 0.0
            score = strong[i] * (1 - cov)
            lib_held = get("lib_held", i, []) or []
            if mode == "ambiguous":
                if st in UNSETTLED:
                    out.append({"pole": nid, "layer": L, "tier": 1, "score": score, "why": f"survey {st}: {(rec.get('label') or '')[:90]}"})
                elif not st and d["class"][i] == "eligible" and strong[i] >= med and cov < 0.10 and d["test"][i] != "held" \
                        and get("lib_test", i) != "held":
                    out.append({"pole": nid, "layer": L, "tier": 2, "score": score,
                                "why": f"{strong[i]:,} strong activations, top word covers {cov:.0%}, screen {d['bin'][i] or '-'}"
                                       + (f", library {get('lib_kind', i)}:{get('lib_condition', i)} failed" if get("lib_test", i) == "failed" else "")})
            else:
                if st == "partial":  # measured: the held label covers under half of the pole (coverage.ASSIGN)
                    c_ = rec.get("coverage")
                    out.append({"pole": nid, "layer": L, "tier": 0, "score": score,
                                "why": f"partial: the held label covers {c_:.0%} of the pole; " + (rec.get("label") or "")[:90]})
                    continue
                held_by = "survey" if st == "held" else "screen" if d["test"][i] == "held" else "library" if get("lib_test", i) == "held" else None
                if not held_by:
                    continue
                reasons = []
                if cov < 0.5:
                    reasons.append(f"top word covers {cov:.0%} of {strong[i]:,} strong activations")
                if len(lib_held) >= 2:
                    reasons.append("held under " + ", ".join(lib_held))
                if reasons:
                    what = (rec.get("label") or "")[:80] if held_by == "survey" else (d["bin"][i] if held_by == "screen" else f"{get('lib_kind', i)}:{get('lib_condition', i)}")
                    out.append({"pole": nid, "layer": L, "tier": 1 if held_by == "survey" else 2, "score": score,
                                "why": f"{held_by} held ({what}); " + "; ".join(reasons)})
    out.sort(key=lambda r: (r["tier"], -r["score"], r["pole"]))
    return out


def main_queue(a):
    rs = queue("ambiguous" if a.ambiguous else "deepen", _layers(a.layers))
    if a.json:
        print(json.dumps(rs[:a.limit], ensure_ascii=False, indent=1))
        return
    tiers = {}
    for r in rs:
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
    print(f"{len(rs)} {'ambiguous' if a.ambiguous else 'deepen'} candidates (" + ", ".join(f"tier {t} {n}" for t, n in sorted(tiers.items())) + ")")
    for r in rs[:a.limit]:
        print(f"  {r['pole']:22s} t{r['tier']} {r['score']:>9,.0f}  {r['why'][:150]}")
