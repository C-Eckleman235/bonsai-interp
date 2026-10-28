"""compose@6: compose@3's cover (round3.py, unchanged) with anticipation conditions (conds4.py): a condition that holds at
the next token or the one after it, which the model has not seen when it reads the read token. Same event sample
(pinned by hash), no new GPU sampling pass.

  python -m bonsai_interp.compose.round6 protocol | select | build | score | report   (same stages and flags as round3.py)

Poles compose@3, compose@4 or compose@5 tested are left out, as well as every assignment-grade pole. The library is
compose@4's (format and shifted conditions) plus next1: and next2:; compose@5's far-context conditions are not
repeated. The aim is deep layers, where prediction-type poles concentrate (the census: 34% in L30-35), and a check of
whether mid-layer poles anticipate rather than recognize.
"""
import json
import sys
from pathlib import Path

from bonsai_interp.compose import conds4 as CD4
from bonsai_interp.compose import round3 as R3
from bonsai_interp.compose import round5 as R5


def _install():
    assigned = R3._assigned
    R3.PROTOCOL, R3.BASE = "compose@6", "bonsai1.compose6"
    R3.CD = CD4
    R3.CODE = list(R3.CODE) + [Path(CD4.__file__), Path(CD4.C2.__file__), Path(R5.__file__), Path(__file__)]
    R3._assigned = lambda: assigned() | R5._tested(3, 4, 5)


def main():
    _install()
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    R3.main()
    if cmd == "protocol":
        p = R3._paths()["protocol"]
        man = json.loads(p.read_text(encoding="utf-8"))
        man["about6"] = __doc__
        man["excluded"] = "assignment-grade poles (atlas, after compose@5) and every pole compose@3, @4 or @5 tested"
        p.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
