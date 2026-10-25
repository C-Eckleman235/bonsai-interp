"""Check a selection pass against what the survey already knows (CPU only).

  python -m bonsai_interp.sweeplib.validate <select.json.gz>

For each kind it prints three things:
  - the number of poles selected and the placebo selections, per condition;
  - recall: how many of the survey's held labels of that kind the kind selects, with the labels found by a strict pattern
    on the label text, plus the zoo's copy neurons;
  - enrichment: the selection rate among those held poles against the rate among all non-weak poles.
"""
import gzip
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATTERNS = {
    "copy": r"\bcop(y|ies|ying)\b|induction|repeat(ed|s)? (word|token|name)|earlier in the (text|passage|context)|seen before",
    "topic": r"\b(medic|clinic|legal|court|sport|financ|stock market|cook|recipe|religio|church|militar|war\b|music|biolog|physic|chemi|politic|election)",
    "lang": r"\b(french|spanish|german|portuguese|italian|dutch|cyrillic|russian|chinese|japanese|korean|hangul|arabic|non-english|foreign language)",
    "struct": r"string literal|inside (a )?string|comment|indent|nest(ed|ing)|inside parenthes|quotation|list item|bullet|heading",
    "number": r"\byears?\b|\bdigit|decimal|price|currency|dollar",
    "predict": r"predict|next token|about to|will be followed|followed by",
}
ZOO_COPY = ["bonsai1/L23/N11310+", "bonsai1/L24/N1091+", "bonsai1/L18/N624+", "bonsai1/L19/N4757+", "bonsai1/L26/N9961+",
            "bonsai1/L13/N6073-", "bonsai1/L14/N8520-", "bonsai1/L15/N9881-", "bonsai1/L16/N5346+", "bonsai1/L17/N9491+"]


def main(path):
    sel = json.loads(gzip.open(path, "rb").read())
    ix = json.load(open(ROOT / "graph" / "index.json", encoding="utf-8"))["neurons"]
    held = {k: v for k, v in ix.items() if v.get("status") == "held" and k.startswith("bonsai1/")}
    known = {kind: {k for k, v in held.items() if re.search(p, (v.get("label") or "").lower())} for kind, p in PATTERNS.items()}
    known["copy"] |= set(ZOO_COPY)
    by_kind = {}
    for kind, conds in sel["kinds"].items():
        s = set()
        print(f"\n== {kind}")
        for c, e in conds.items():
            n = len(e.get("selected", []))
            s |= {r["neuron"] for r in e.get("selected", [])}
            print(f"  {c:22s} pairs {e['pairs']:3d}  selected {n:6d}  placebo {e.get('placebo_selected')}  {e.get('why', '')}")
        by_kind[kind] = s
    by_kind["predict"] = {r["neuron"] for cs in sel.get("predict", {}).values() for r in cs}
    from bonsai_interp.graph import which
    n_ok = int((which.load_index()["pole_max"] >= 0.5).sum())
    print("\nkind      selected   known held   known selected   rate among known   rate among all")
    for kind, s in by_kind.items():
        k = known.get(kind, set())
        hit = k & s
        print(f"{kind:9s} {len(s):8d}   {len(k):10d}   {len(hit):14d}   {len(hit) / max(len(k), 1):16.1%}   {len(s) / n_ok:14.2%}")
        for h in sorted(hit)[:6]:
            print(f"            {h}  {held.get(h, {}).get('label', '(zoo)')[:90]}")
    print("\nzoo copy neurons:", {z: [c for c, e in sel["kinds"].get("copy", {}).items() if z in {r["neuron"] for r in e.get("selected", [])}]
                                  for z in ZOO_COPY})


if __name__ == "__main__":
    main(sys.argv[1])
