"""sweeplib@4: part of speech and morphology screens (posmorph.py), run on the grammar pass's engine (run.py, unchanged).

  python -m bonsai_interp.grammar.run4 protocol | select | build | score | report   (same stages and flags as run.py)

The spaCy parses of the selection and test splits are the grammar pass's (data/grammar/bonsai1.parse.{select,test}.npz).
Only the conditions differ: posmorph.labels / battery / holds replace cons.labels / battery / holds.
"""
import json
import sys
from pathlib import Path

from bonsai_interp.grammar import cons as C
from bonsai_interp.grammar import posmorph as PM
from bonsai_interp.grammar import run as GR

PROTOCOL, KIND, BASE = "sweeplib@4", "pos", "bonsai1.sweeplib4"


def _install():
    paths3 = GR._paths

    def paths(d=None):
        P = paths3(d)
        for k in ("protocol", "prereg", "summary", "select", "tests", "results", "coverage", "report"):
            P[k] = P[k].with_name(P[k].name.replace("bonsai1.sweeplib3", BASE))
        return P
    GR._paths, GR.PROTOCOL, GR.KIND = paths, PROTOCOL, KIND
    GR.CODE = list(GR.CODE) + [Path(PM.__file__), Path(__file__)]
    C.labels, C.battery, C.holds = PM.labels, PM.battery, PM.holds
    C.CONDITIONS = {c: (lab, "the word itself") for c, lab in PM.labels().items()}
    C.PER_KEY, C.CAP, C.FEW = PM.PER_KEY, PM.CAP, ()


def main():
    _install()
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    GR.main()
    if cmd == "protocol":  # say how the pairs are keyed (the grammar manifest's text assumes token or part-of-speech keys)
        p = GR._paths()["protocol"]
        man = json.loads(p.read_text(encoding="utf-8"))
        man["about"] = PM.__doc__
        for c, v in man["conditions"].items():
            v["key"] = "token" if c.startswith("same:") else f"frequency band ({PM.BANDS} log-spaced bands)"
            v.pop("cue", None)
        p.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
