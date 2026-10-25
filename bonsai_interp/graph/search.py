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
RANK = {("survey", "held"): 0, ("library", "held"): 1, ("screen", "held"): 2}


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
