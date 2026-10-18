"""Which techniques pay off? Aggregates the "methods" blocks of all contributions.

  python -m bonsai_interp.graph.meta

Per screen and per agent: how many contributions used it, how often their tests held, the median
minutes spent, and the mean scrutiny of the poles they touched. Correlation, not causation: deep
screens are used on hard neurons.
"""
import json
import statistics
from collections import defaultdict

from bonsai_interp.graph.contrib import ROOT, load_all


def main():
    idx = json.loads((ROOT / "graph" / "index.json").read_text(encoding="utf-8"))["neurons"]
    by_screen, by_agent = defaultdict(list), defaultdict(list)
    n_with = 0
    for _, c in load_all():
        m = c.get("methods")
        if not m and c["id"].startswith("core-"):  # legacy core work predates the block: infer it
            prot = {t.get("protocol", "") for t in c.get("tests", [])}
            m = {"agent": "claude-code opus-5.5 (core, inferred)" if c.get("contributor") != "neuron-cleanroom-blind-agent" else "claude-code opus-5.5 (blind room)",
                 "screens": sorted({"look"} | ({"quick-test"} if any(x.startswith("quick") for x in prot) else set())
                                   | ({"held-out", "bank-solo"} if any(x.startswith("held-out") for x in prot) else set())
                                   | ({"blind-room"} if any(x.startswith("cleanroom") for x in prot) else set())
                                   | ({"step-out"} if c["kind"] == "map" else set()))}
        if not m:
            continue
        n_with += 1
        held = [t.get("survives") for t in c.get("tests", []) if t.get("protocol", "").startswith("quick")]
        row = {"held": held[-1] if held else None, "minutes": m.get("minutes"),
               "scrutiny": (idx.get(c["neuron"], {}).get("scrutiny") or {}).get("level")}
        for s in set(m.get("screens", [])) | set(m.get("detected", [])):
            by_screen[s].append(row)
        by_agent[m.get("agent", "?")].append(row)
    print(f"{n_with} contributions with a methods block\n")
    for title, groups in (("screen", by_screen), ("agent", by_agent)):
        print(f"{title:<42}{'uses':>6}{'held':>8}{'median min':>12}{'mean scrutiny':>15}")
        for k, rows in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            h = [r["held"] for r in rows if r["held"] is not None]
            mins = [r["minutes"] for r in rows if r["minutes"] is not None]
            sc = [r["scrutiny"] for r in rows if r["scrutiny"] is not None]
            print(f"{k:<42}{len(rows):>6}{(f'{sum(h)}/{len(h)}' if h else '-'):>8}"
                  f"{(statistics.median(mins) if mins else '-'):>12}{(f'{statistics.mean(sc):.1f}' if sc else '-'):>15}")
        print()


if __name__ == "__main__":
    main()
