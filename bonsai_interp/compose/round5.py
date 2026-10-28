"""compose@5: compose@3's cover (round3.py, unchanged) with far-context conditions (conds3.py): a condition held in the
8 tokens before, or unusually often in the 32 tokens before. Same event sample (pinned by hash), no new GPU sampling pass.

  python -m bonsai_interp.compose.round5 protocol | select | build | score | report   (same stages and flags as round3.py)

Poles compose@3 or compose@4 tested are left out, as well as every assignment-grade pole: a re-test could repeat an
earlier round's pairs exactly (an item set's seed is its key). The aim is layers 12-17, where local conditions stalled.
"""
import json
import sys
from pathlib import Path

from bonsai_interp.compose import conds3 as CD3
from bonsai_interp.compose import round3 as R3


def _tested(*rounds):
    """Poles the given earlier cover rounds tested (from their registered tests files)."""
    from bonsai_interp.auto import context_tier as ct
    out = set()
    for k in rounds:
        p = R3.OUT / f"bonsai1.compose{k}.tests.jsonl.gz"
        if not p.exists():
            sys.exit(f"compose@5 needs compose@{k}'s tests file ({p.name}, release compose{k}-data)")
        for r in ct._read_gz(p):
            if r.get("type") == "test":
                L = int(r["neuron"].split("/L")[1].split("/")[0])
                out.add((L, int(r["neuron"].split("/N")[1][:-1]), r["neuron"][-1]))
    return out


def _install():
    assigned = R3._assigned
    R3.PROTOCOL, R3.BASE = "compose@5", "bonsai1.compose5"
    R3.CD = CD3
    R3.CODE = list(R3.CODE) + [Path(CD3.__file__), Path(CD3.C2.__file__), Path(__file__)]
    R3._assigned = lambda: assigned() | _tested(3, 4)


def main():
    _install()
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    R3.main()
    if cmd == "protocol":
        p = R3._paths()["protocol"]
        man = json.loads(p.read_text(encoding="utf-8"))
        man["about5"] = __doc__
        man["dense_thresholds"] = CD3.thresholds()
        man["excluded"] = "assignment-grade poles (atlas, after compose@4) and every pole compose@3 or compose@4 tested"
        p.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
