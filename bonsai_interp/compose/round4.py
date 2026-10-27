"""compose@4: compose@3's cover (round3.py, unchanged) with a wider condition library (conds2.py): text-format conditions
and every compose@3 condition shifted to the previous token and the one before it. The event sample is compose@3's
(pinned by hash), so no new GPU sampling pass is needed.

  python -m bonsai_interp.compose.round4 protocol | select | build | score | report   (same stages and flags as round3.py)

Two differences beyond the library:
  1. Poles compose@3 tested are left out, as well as every assignment-grade pole (which now includes compose@3's holds).
     A re-test could repeat compose@3's pairs exactly (an item set's seed is its key).
  2. The aim is the layers every round has found hardest (12-17): mid-layer poles whose strong activations spread over
     many read tokens and whose read-token conditions cover little (the 2026-10-27 diagnostic: at L14 and L16 only 7-14%
     of poles have a read-token condition covering half of them; shifted and format conditions roughly double that).
"""
import json
import sys
from pathlib import Path

from bonsai_interp.compose import conds2 as CD2
from bonsai_interp.compose import round3 as R3


def _tested3():
    """Poles compose@3 tested (from its registered tests file)."""
    from bonsai_interp.auto import context_tier as ct
    p = R3.OUT / "bonsai1.compose3.tests.jsonl.gz"
    if not p.exists():
        sys.exit(f"compose@4 needs compose@3's tests file ({p.name}, release compose3-data)")
    out = set()
    for r in ct._read_gz(p):
        if r.get("type") == "test":
            L = int(r["neuron"].split("/L")[1].split("/")[0])
            out.add((L, int(r["neuron"].split("/N")[1][:-1]), r["neuron"][-1]))
    return out


def _install():
    assigned3 = R3._assigned
    R3.PROTOCOL, R3.BASE = "compose@4", "bonsai1.compose4"
    R3.CD = CD2
    R3.CODE = list(R3.CODE) + [Path(CD2.__file__), Path(__file__)]
    R3._assigned = lambda: assigned3() | _tested3()


def main():
    _install()
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    R3.main()
    if cmd == "protocol":
        p = R3._paths()["protocol"]
        man = json.loads(p.read_text(encoding="utf-8"))
        man["about4"] = __doc__
        man["excluded"] = "assignment-grade poles (atlas, after compose@3) and every pole compose@3 tested"
        p.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
