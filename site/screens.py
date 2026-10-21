"""Turn the atlas into the survey map's screen layer.

  python site/screens.py     data/atlas/bonsai1_L<L>.arrow + vocab.arrow  ->  site/screens/bins.json, L<L>.json, vocab.json

The atlas (bonsai_interp/atlas.py) holds every pole's corpus word hits and, for screened layers, its auto-token@2 bin,
context rule and auto test. Hypotheses at scrutiny 0, never labels. The Pages build downloads it from the `atlas`
release and runs this.
- bins.json: one character per pole (index 2n for +, 2n+1 for -): '.' nothing, a-i a map group, A-I the same group
  with a held auto test. A pole in a layer that hasn't been screened yet, but that has a stage 1 trigger, is
  "word hit".
- L<L>.json: columnar per layer: pole indices, then word ids / lift / coverage / trigger flags (5 per pole, -1 padded),
  plus bin, context, test and auroc for screened poles. Loaded when someone opens a block or a neuron in that layer.
- vocab.json: id -> text for every word id used, loaded once.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
OUT = HERE / "screens"
N_N = 12288
TOP = 5
# map groups, in the categorical palette's order; each pole's bin folds into one
GROUPS = [("spelling", ["spelling"]), ("word", ["lexical"]), ("meaning", ["sense", "concept"]), ("context", ["collocation", "context", "predictive"]),
          ("format", ["format"]), ("topic", ["topic"]), ("copy", ["copy"]), ("other", ["dense", "positional", "unexplained", "rare"]),
          ("word hit", [])]
GROUP_OF = {b: i for i, (_, bins) in enumerate(GROUPS) for b in bins}
OTHER, HIT = 7, 8


def main(src=ROOT / "data" / "atlas"):
    from bonsai_interp.atlas import _read
    src = Path(src)
    files = sorted(src.glob("bonsai1_L*.arrow"), key=lambda p: int(p.name.split("_L")[1].split(".")[0]))
    if not files:
        print(f"no atlas in {src}: nothing to build")
        return
    vt = _read(src / "vocab.arrow")
    vocab = dict(zip(vt["tok"].to_pylist(), vt["text"].to_pylist()))
    OUT.mkdir(exist_ok=True)
    layers, counts, used = {}, {}, set()
    for p in files:
        t = _read(p).to_pydict()
        L = int(t["layer"][0])
        chars, c = ["."] * (2 * N_N), [0] * len(GROUPS)
        cols = {"pole": [], "w": [], "lift": [], "cov": [], "trig": [], "bin": [], "context": [], "test": [], "auroc": []}
        for i, pole in enumerate(t["pole"]):
            words, b = t["hit_tok"][i], t["bin"][i]
            trig = t["hit_trigger"][i]
            if b == "weak" or (b in ("unexplained", "rare") and not any(trig)):
                b = None  # nothing specific to show beyond the words
            if b is None and not any(trig) and not words:
                continue
            if b is not None:
                g = GROUP_OF.get(b, OTHER)
                chars[pole] = chr(65 + g) if t["test"][i] == "held" else chr(97 + g)
                c[g] += 1
            elif any(trig):
                chars[pole] = chr(97 + HIT)
                c[HIT] += 1
            pad = TOP - len(words)
            cols["pole"].append(pole)
            cols["w"] += list(words) + [-1] * pad
            cols["lift"] += [round(x) for x in t["hit_lift"][i]] + [0] * pad
            cols["cov"] += [round(100 * x, 1) for x in t["hit_coverage"][i]] + [0] * pad
            cols["trig"] += [int(x) for x in trig] + [0] * pad
            cols["bin"].append(t["bin"][i])
            cols["context"].append(t["context"][i])
            cols["test"].append(t["test"][i])
            cols["auroc"].append(None if t["auroc"][i] is None else round(t["auroc"][i], 3))
            used.update(words)
        if not any(x is not None for x in cols["bin"]):  # an unscreened layer: drop the empty screen columns
            for k in ("bin", "context", "test", "auroc"):
                cols.pop(k)
        layers[str(L)] = "".join(chars)
        counts[str(L)] = c
        (OUT / f"L{L}.json").write_text(json.dumps(cols, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    (OUT / "vocab.json").write_text(json.dumps({str(k): vocab.get(k, "") for k in sorted(used)}, ensure_ascii=False, separators=(",", ":")) + "\n",
                                    encoding="utf-8", newline="\n")
    bins = {"protocol": ["auto-token@2"], "groups": [g for g, _ in GROUPS], "layers": layers, "counts": counts,
            "screened": sorted(int(L) for L, cc in counts.items() if sum(cc[:HIT]) > 0)}
    (OUT / "bins.json").write_text(json.dumps(bins, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote site/screens/: {len(layers)} layers ({len(bins['screened'])} screened), "
          f"{sum(sum(cc[:HIT]) for cc in counts.values()):,} screened poles, {sum(cc[HIT] for cc in counts.values()):,} word-hit poles")


if __name__ == "__main__":
    main(*sys.argv[1:2])
