"""A readable connectome of the 'nerve' neuron's neighbourhood (L9/N3688, Bonsai-8B 1-bit).

Only the nodes that carry the story are drawn; the remaining crawled neurons are summarised.
Edge weights come from probes/bonsai1_L9_N3688_pos/web_nodes.json and neighbors.json; test
badges from the quick-check files.

  python -m mono.figures.connectome_figure  ->  reports/figures/nerve_connectome.{png,pdf}
"""
import json

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from mono.model import ROOT

matplotlib.use("Agg")
P = ROOT / "probes" / "bonsai1_L9_N3688_pos"
OUT = ROOT / "reports" / "figures"

INK, INK2, INK3, GRID, BG = "#14191a", "#4a5351", "#7d8784", "#e6e9e7", "#ffffff"
TEAL, TEAL_SOFT = "#1f6b5c", "#e2eeea"
BLUE, BLUE_SOFT = "#2a78d6", "#e8f1fc"
ORANGE, ORANGE_SOFT = "#d9531e", "#fdeee6"
VIOLET, VIOLET_SOFT = "#4a3aa7", "#eeecf8"
SLATE_SOFT = "#f1f3f2"
SANS, SERIF, MONO = "IBM Plex Sans", "Georgia", "Cascadia Mono"
GOOD, BAD = "#1f7a3a", "#b3261e"

plt.rcParams.update({"font.family": SANS, "font.size": 9, "text.color": INK, "pdf.fonttype": 42})


def load(name):
    return json.loads((P / name).read_text(encoding="utf-8"))


def tests():
    q = {n["unit"]: n for n in load("web_quick_claude.json")["neighbours"]}
    for n in load("quick.json")["neighbours"]:
        q.setdefault(n["unit"], n)
    return q


def node(ax, x, y, name, unit, kind, badge=None, w=15.5, h=5.6):
    fc, ec, lw = {"probed": (TEAL_SOFT, TEAL, 2.4), "input": (BG, INK, 1.2), "feeder": (SLATE_SOFT, "#b9c0bd", 0.9),
                  "gate": (ORANGE_SOFT, ORANGE, 1.4), "reader": (BLUE_SOFT, BLUE, 1.3), "family": (VIOLET_SOFT, VIOLET, 1.4),
                  "faint": (BG, INK3, 0.9)}[kind]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=lw,
                                ls=(0, (3, 2.5)) if kind == "faint" else "-", zorder=3))
    big = kind == "probed"
    ax.text(x, y + (0.9 if unit else 0), name, ha="center", va="center", zorder=4, fontsize=11 if big else 9,
            fontweight="semibold" if kind in ("probed", "input", "gate", "reader", "family") else "normal",
            color=INK if kind != "feeder" else INK2)
    if unit:
        ax.text(x, y - 1.55, unit, ha="center", va="center", zorder=4, fontsize=7.4 if not big else 8.2, family=MONO, color=INK3)
    if badge is not None:
        ok, txt = badge
        ax.text(x + w / 2 - 0.4, y + h / 2 + 0.25, ("✓ " if ok else "× ") + txt, ha="right", va="bottom", fontsize=6.8,
                color=GOOD if ok else BAD, fontweight="semibold", zorder=4)


def edge(ax, a, b, w=1.2, color=INK, label=None, style="-|>", ls="-", rad=0.0, lpos=0.5, loff=(0, 0.9), lcolor=None,
         shrinkA=8.5, shrinkB=8.5, fs=8, alpha=1.0, z=1):
    arr = FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=10, lw=w, color=color, ls=ls, alpha=alpha,
                          connectionstyle=f"arc3,rad={rad}", shrinkA=shrinkA, shrinkB=shrinkB, zorder=z)
    ax.add_patch(arr)
    if label:
        mx, my = a[0] + (b[0] - a[0]) * lpos + loff[0], a[1] + (b[1] - a[1]) * lpos + loff[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=fs, color=lcolor or color, zorder=6,
                bbox=dict(fc=BG, ec="none", pad=0.4))


def main():
    q = tests()
    badge = lambda u: (q[u]["survives"], f"{q[u]['n_fire']}/8") if u in q else None
    fig = plt.figure(figsize=(15, 9.6), dpi=300, facecolor=BG)
    ax = fig.add_axes([0.02, 0.06, 0.96, 0.80])
    ax.set_xlim(0, 120), ax.set_ylim(0, 66)
    ax.axis("off")

    fig.text(0.03, 0.965, "The neighbourhood of a ‘nerve’ neuron", family=SERIF, fontsize=22, fontweight="bold", va="top")
    fig.text(0.03, 0.925, "Bonsai-8B (1-bit), layer 9, neuron 3688. Two input routes and a pain gate shape it; its output spreads thinly; "
             "beside it sits a family of separate emotional-tension neurons that share one reader.", fontsize=10.5, color=INK2, va="top")
    fig.text(0.03, 0.898, "Every node's label was written blind from its top examples and checked with 8 sentences that should fire it "
             "and 8 near misses, committed before running (badge: held / failed, sentences fired of 8).", fontsize=8.6, color=INK3, va="top")

    # band labels and column headers
    for y, t in ((60.5, "ANATOMY & MOTOR"), (38.5, "THE WORD ‘NERVE’"), (21.5, "CONTEXT GATE")):
        ax.text(1, y, t, fontsize=7.8, color=INK3, fontweight="semibold", va="center", family=SANS)
    ax.text(20, 64.6, "text features · layers 0–7", ha="center", fontsize=8, color=INK3)
    ax.text(44, 64.6, "detectors · layers 3–8", ha="center", fontsize=8, color=INK3)
    ax.text(66, 64.6, "layer 9", ha="center", fontsize=8, color=TEAL, fontweight="semibold")
    ax.text(100, 64.6, "readers · layers 10–20", ha="center", fontsize=8, color=INK3)
    for x in (32, 55, 78):
        ax.plot([x, x], [16, 63.5], color=GRID, lw=0.9, zorder=0)

    # lateral family band
    ax.add_patch(FancyBboxPatch((38.5, 0.8), 55, 13.6, boxstyle="round,pad=0,rounding_size=2", fc="#f7f6fc", ec="#d9d4f0", lw=1.0, zorder=0))
    ax.text(39.8, 12.7, "LATERAL FAMILY", fontsize=7.8, color=VIOLET, fontweight="semibold", va="center")
    ax.text(51.2, 12.7, "co-fire with ‘nerve’ (r ≈ 0.6), yet separate jobs: zeroing one moves the others by ≤ 0.03",
            fontsize=7.3, color=VIOLET, va="center")

    # ---- nodes
    N = {}

    def put(key, x, y, name, unit, kind, b=True, **kw):
        N[key] = (x, y)
        node(ax, x, y, name, unit, kind, badge(unit) if (b and unit) else None, **kw)

    put("motor0", 20, 58, "‘motor’", "L0/N873", "feeder")
    put("motor2", 20, 50.5, "‘motor’, ‘-otor’", "L2/N3473", "feeder")
    put("vic", 20, 43, "‘vic’ (visceral…)", "L4/N7884", "feeder")
    put("serve", 20, 34, "‘serve’ (-erv-)", "L0/N1684", "feeder")
    put("arab", 20, 26.5, "Arabic word pieces", "L4/N5828 · L4/N5727", "feeder", b=False)
    put("pain_ml", 20, 19, "pain in other languages", "L6/N2055", "feeder")
    put("token", 44, 34, "spelling ‘nerv / nerf’", "L3/N6493", "input")
    put("concept", 44, 26.5, "nerve concept (Arabic-heavy)", "L8/N7756", "input")
    put("pain", 44, 19, "pain detector", "L8/N3858", "gate")
    put("nerve", 66, 30.6, "psychological ‘nerve’", "L9/N3688", "probed", w=19, h=9.6)
    ax.text(66, 27.0, "+ strain, composure  ·  − bodily sensation", ha="center", fontsize=7.4, color=TEAL, zorder=5)
    put("motorR", 100, 52, "motor ↔ sensory", "L10/N6901", "reader")
    put("others", 100, 37, "4 more readers", "labels did not hold", "faint", b=False)
    put("hub", 100, 22, "emotional hub", "L14/N9470", "reader")
    put("disap", 46.5, 6.4, "‘disappointed’", "L1/N4037", "family", w=12)
    put("suspense", 59.5, 6.4, "suspense", "L7/N9782", "family", w=12)
    put("anticip", 72.5, 6.4, "can’t wait", "L9/N1061", "family", w=12)
    put("overwh", 85.5, 6.4, "overwhelmed", "L12/N3688", "family", w=12)

    # ---- inputs
    edge(ax, N["serve"], N["token"], 1.0, INK3, "7%", fs=7.2)
    edge(ax, N["arab"], N["concept"], 1.3, INK3, "12%", fs=7.2)
    edge(ax, N["pain_ml"], N["pain"], 1.3, INK3, "13%", fs=7.2)
    edge(ax, N["token"], (56.5, 32.6), 4.2, INK, "74%", shrinkB=0, lpos=0.86, loff=(0, 1.6))
    edge(ax, N["concept"], (56.5, 29.2), 3.6, INK, "61%", shrinkB=0, lpos=0.86, loff=(0, -1.7))
    edge(ax, N["pain"], (60.5, 25.8), 2.2, ORANGE, None, style="-[", shrinkB=0, rad=0.12)
    ax.text(57.6, 21.2, "closes both routes\nwhen pain comes first", fontsize=7.4, color=ORANGE, ha="center", va="center",
            bbox=dict(fc=BG, ec="none", pad=0.3), zorder=6)
    ax.text(44, 39.4, "‘lost his nerve’ also closes this route", fontsize=7.2, color=INK2, ha="center", style="italic")

    # ---- readers
    edge(ax, (75.5, 33.6), N["motorR"], 1.8, BLUE, "+0.10", shrinkA=0, rad=-0.1, lpos=0.5, loff=(-1.5, 0.8))
    edge(ax, (75.5, 31.0), N["others"], 1.0, BLUE, None, shrinkA=0, ls=(0, (3, 2)), alpha=0.7)
    edge(ax, (75.5, 28.2), N["hub"], 1.6, BLUE, "+0.09", shrinkA=0, rad=0.08, lpos=0.5, loff=(0, 1.0))
    for k in ("motor0", "motor2", "vic"):
        edge(ax, N[k], N["motorR"], 0.9, "#9aa3a0", None, rad=-0.04 if k != "motor0" else 0.0)
    ax.text(62, 47.2, "other inputs of this reader (5–10% each)", fontsize=7.2, color=INK3, ha="center",
            bbox=dict(fc=BG, ec="none", pad=0.2), zorder=6)

    # ---- the family and the hub
    edge(ax, (46.5, 9.6), N["hub"], 1.3, VIOLET, None, shrinkA=0, rad=-0.3)
    ax.text(62.0, 16.6, "‘disappointed’ feeds the hub (7%)", fontsize=7.2, color=VIOLET, ha="center",
            bbox=dict(fc=BG, ec="none", pad=0.2), zorder=6)
    for k in ("suspense", "anticip", "overwh"):
        edge(ax, N[k], N["hub"], 1.0, VIOLET, None, ls=(0, (4, 3)), rad=-0.18, alpha=0.85)
    ax.text(108.5, 9.0, "suspense, can’t-wait and\noverwhelmed passages\ndrive the hub without\n‘nerve’ (inverse test)",
            fontsize=7.2, color=VIOLET, ha="center", va="center", zorder=6)

    # remaining crawl
    fig.text(0.03, 0.018, "Not drawn: 30 more crawled neurons (mostly spelling and token pieces such as diarr-, trait, -ate, p-value digits, "
             "Albany; citations; code). 38 of 48 labels held overall.\nInput %: share of the target's activation lost when that one neuron "
             "is zeroed. Reader numbers: change in the reader (fraction of its range) when ‘nerve’ is boosted.",
             fontsize=7.6, color=INK3, va="bottom", linespacing=1.5)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "nerve_connectome.png", dpi=300, facecolor=BG)
    fig.savefig(OUT / "nerve_connectome.pdf", facecolor=BG)
    print("wrote", OUT / "nerve_connectome.png")


if __name__ == "__main__":
    main()
