"""Turn the auto screens into the survey map's screen layer.

  python site/screens.py      data/auto/bonsai1_L<L>.screen.json.gz  ->  site/screens/bins.json + site/screens/L<L>.json

The screens are hypotheses from the auto tier (scrutiny level 0): which bin a pole falls in, the words that fire it
most often relative to their frequency, and its strongest context feature. They are bulky, so they live in the
`auto-screens` release rather than in git; the Pages build downloads them and runs this. The map loads bins.json
(one character per pole) when someone colours by screen, and a layer's details when someone opens a block or a
neuron in it. Labels and status still come only from tested contributions.
"""
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "screens"
N_N = 12288
# map groups, in the categorical palette's order; each pole's bin folds into one
GROUPS = [("spelling", ["spelling"]), ("word", ["lexical"]), ("meaning", ["sense", "concept"]), ("context", ["collocation", "context", "predictive"]),
          ("format", ["format"]), ("topic", ["topic"]), ("copy", ["copy"]), ("other", ["dense", "positional", "unexplained", "rare"])]
GROUP_OF = {b: i for i, (_, bins) in enumerate(GROUPS) for b in bins}
OTHER = len(GROUPS) - 1


def load(path):
    with (gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(path, encoding="utf-8")) as f:
        return json.load(f)


def main(src=ROOT / "data" / "auto"):
    files = sorted(Path(src).glob("bonsai1_L*.screen.json*"))
    if not files:
        print(f"no screens in {src}: nothing to build")
        return
    OUT.mkdir(exist_ok=True)
    layers, counts, protocols = {}, {}, set()
    for p in files:
        s = load(p)
        L = int(s["layer"])
        protocols.add(s.get("protocol", "?"))
        chars, detail, c = ["."] * (2 * N_N), {}, [0] * len(GROUPS)
        for key, r in s["poles"].items():
            if r.get("bin") == "weak" or (r.get("bin") in ("unexplained", "rare", None) and not r.get("hits")):
                continue  # no word hit: nothing to show (the export lists every pole)
            n, pole = int(key[1:-1]), key[-1]
            g = GROUP_OF.get(r.get("bin"), OTHER)
            held = r.get("test") == "held"
            chars[2 * n + (pole == "-")] = (chr(65 + g) if held else chr(97 + g))
            c[g] += 1
            detail[key] = [r.get("bin"), [[h.get("token"), round(h.get("lift", 0), 1), round(h.get("coverage", 0), 2)] for h in r.get("hits", [])[:3]],
                           (r.get("context") or "")[:160], r.get("test", "untested"), r.get("auroc")]
        layers[str(L)] = "".join(chars)
        counts[str(L)] = c
        (OUT / f"L{L}.json").write_text(json.dumps(detail, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    bins = {"protocol": sorted(protocols), "groups": [g for g, _ in GROUPS], "layers": layers, "counts": counts}
    (OUT / "bins.json").write_text(json.dumps(bins, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote site/screens/: {len(layers)} layers, {sum(map(sum, counts.values())):,} screened poles")


if __name__ == "__main__":
    main(*sys.argv[1:2])
