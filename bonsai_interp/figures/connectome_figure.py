"""A readable connectome of the 'nerve' neuron's neighbourhood (L9/N3688, Bonsai-8B 1-bit).

Only the nodes that carry the story are drawn; the remaining crawled neurons are summarised.
Edge weights come from probes/bonsai1_L9_N3688_pos/web_nodes.json and neighbors.json; test
badges from the quick-check files.

  python -m bonsai_interp.figures.connectome_figure  ->  reports/figures/nerve_connectome.{png,pdf}
"""
import json

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from bonsai_interp.model import ROOT

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
    """Latest quick-check result per unit: later labelling rounds override earlier ones."""
    q = {n["unit"]: n for n in load("quick.json")["neighbours"]}
    for f in ("web_quick_claude.json", "relabel_quick.json", "relabel2_quick.json", "lateral_out_quick.json"):
        if (P / f).exists():
            q.update({n["unit"]: n for n in load(f)["neighbours"]})
    return q


def node(ax, x, y, name, unit, kind, badge=None, w=15.5, h=5.6, fs=None):
    fc, ec, lw = {"probed": (TEAL_SOFT, TEAL, 2.4), "input": (BG, INK, 1.2), "feeder": (SLATE_SOFT, "#b9c0bd", 0.9),
                  "gate": (ORANGE_SOFT, ORANGE, 1.4), "reader": (BLUE_SOFT, BLUE, 1.3), "family": (VIOLET_SOFT, VIOLET, 1.4),
                  "faint": (BG, INK3, 0.9)}[kind]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=lw,
                                ls=(0, (3, 2.5)) if kind == "faint" else "-", zorder=3))
    big = kind == "probed"
    two = "\n" in name
    ax.text(x, y + (0.9 if unit else 0) + (0.5 if two else 0), name, ha="center", va="center", zorder=4, fontsize=fs or (11 if big else 9), linespacing=1.1,
            fontweight="semibold" if kind in ("probed", "input", "gate", "reader", "family") else "normal",
            color=INK if kind != "feeder" else INK2)
    if unit:
        ax.text(x, y - (2.3 if two else 1.55), unit, ha="center", va="center", zorder=4, fontsize=7.4 if not big else 8.2, family=MONO, color=INK3)
    if badge is not None:
        ok, txt = badge
        ax.text(x + w / 2 - 0.4, y + h / 2 + 0.25, ("✓ " if ok is True else "× " if ok is False else "") + txt, ha="right", va="bottom",
                fontsize=6.8, color=GOOD if ok is True else BAD if ok is False else INK3, fontweight="semibold", zorder=4)


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
    lat_w = {(e["from"], e["to"]): e["w"] for e in load("lateral_out_nodes.json")["edges"]}
    h4 = load("h4_results.json")
    fig = plt.figure(figsize=(16, 13.2), dpi=300, facecolor=BG)
    ax = fig.add_axes([0.02, 0.055, 0.96, 0.835])
    ax.set_xlim(0, 124), ax.set_ylim(0, 97)
    ax.axis("off")

    fig.text(0.03, 0.975, "The neighbourhood of a ‘nerve’ neuron", family=SERIF, fontsize=23, fontweight="bold", va="top")
    fig.text(0.03, 0.942, "Bonsai-8B (1-bit), layer 9, neuron 3688. Two input routes and a pain gate shape it, and its output spreads thinly. "
             "It responds to the word ‘nerve’ first, and more weakly to ‘strong’ and to words for being hit or hurt.", fontsize=10.5, color=INK2, va="top")
    fig.text(0.03, 0.922, "Beside it sits a family of separate emotional-tension neurons that share one reader; each of them has its own crisp inputs (bottom).",
             fontsize=10.5, color=INK2, va="top")
    fig.text(0.03, 0.902, "Every node's label was written blind and checked with 8 sentences that should fire it and 8 near misses, committed before running "
             "(badge: held / failed, sentences fired of 8; failed labels got up to two more rounds).", fontsize=8.6, color=INK3, va="top")

    # ---- upper half: the nerve circuit
    for y, t in ((90.5, "ANATOMY & MOTOR"), (66.5, "THE WORD ‘NERVE’"), (50.5, "CONTEXT GATE")):
        ax.text(1, y, t, fontsize=7.8, color=INK3, fontweight="semibold", va="center")
    ax.text(20, 95.2, "text features · layers 0–7", ha="center", fontsize=8, color=INK3)
    ax.text(44, 95.2, "detectors · layers 3–8", ha="center", fontsize=8, color=INK3)
    ax.text(68, 95.2, "layer 9", ha="center", fontsize=8, color=TEAL, fontweight="semibold")
    ax.text(102, 95.2, "readers · layers 10–20", ha="center", fontsize=8, color=INK3)
    for x in (32, 55, 82):
        ax.plot([x, x], [41.5, 94], color=GRID, lw=0.9, zorder=0)

    N = {}

    def put(key, x, y, name, unit, kind, b=True, **kw):
        N[key] = (x, y)
        node(ax, x, y, name, unit, kind, b if isinstance(b, tuple) else (badge(unit) if (b and unit) else None), **kw)

    put("motor0", 20, 88, "‘motor’", "L0/N873", "feeder")
    put("motor2", 20, 80.5, "‘motor’, ‘-otor’", "L2/N3473", "feeder")
    put("vic", 20, 73, "‘vic’ (visceral…)", "L4/N7884", "feeder")
    put("serve", 20, 64, "‘serve’ (-erv-)", "L0/N1684", "feeder")
    put("arab", 20, 56.5, "Arabic word pieces", "L4/N5828 · L4/N5727", "feeder", b=False)
    put("pain_ml", 20, 49, "pain in other languages", "L6/N2055", "feeder")
    put("token", 44, 64, "spelling ‘nerv / nerf’", "L3/N6493", "input")
    put("concept", 44, 56.5, "nerve concept (Arabic-heavy)", "L8/N7756", "input")
    put("pain", 44, 49, "pain detector", "L8/N3858", "gate")
    put("nerve", 68, 62, "the word ‘nerve’", "L9/N3688", "probed", w=21, h=10)
    ax.text(68, 58.2, "peak: psychological sense (strain, composure)", ha="center", fontsize=7.2, color=TEAL, zorder=5)
    put("motorR", 102, 89, "brain & nerve prose?", "L10/N6901", "reader", b=(False, "4/8 after 3 rounds"))
    put("bio", 102, 79.5, "biology textbook terms", "L10/N5553 (− pole)", "reader")
    put("dense", 102, 69.5, "3 dense / diffuse readers", "L13/N3792 · L19/N2291 · L20/N5963", "faint", b=(None, "on for 5–70% of all text"), w=19)
    put("hub", 102, 57, "tension hub (dense)", "L14/N9470", "reader", b=(None, "dense: no single label"))

    edge(ax, N["serve"], N["token"], 1.0, INK3, "7%", fs=7.2)
    edge(ax, N["arab"], N["concept"], 1.3, INK3, "12%", fs=7.2)
    edge(ax, N["pain_ml"], N["pain"], 1.3, INK3, "13%", fs=7.2)
    edge(ax, N["token"], (57.5, 64.2), 4.2, INK, "74%", shrinkB=0, lpos=0.84, loff=(0, 1.6))
    edge(ax, N["concept"], (57.5, 60.4), 3.6, INK, "61%", shrinkB=0, lpos=0.84, loff=(0, -1.7))
    edge(ax, N["pain"], (61.5, 57.0), 2.2, ORANGE, None, style="-[", shrinkB=0, rad=0.12)
    ax.text(44, 44.0, "closes both routes when pain comes first", fontsize=7.4, color=ORANGE, ha="center", va="center", zorder=6)
    ax.text(44, 69.4, "‘lost his nerve’ also closes this route", fontsize=7.2, color=INK2, ha="center", style="italic")

    edge(ax, (78.5, 65.5), N["motorR"], 1.8, BLUE, "+0.10", shrinkA=0, rad=-0.1, loff=(-1.5, 0.9))
    edge(ax, (78.5, 64.0), N["bio"], 1.4, BLUE, "−0.08", shrinkA=0, rad=-0.05, loff=(-0.5, 0.9))
    edge(ax, (78.5, 62.0), N["dense"], 1.0, BLUE, None, shrinkA=0, ls=(0, (3, 2)), alpha=0.7)
    edge(ax, (78.5, 59.5), N["hub"], 1.6, BLUE, "+0.09", shrinkA=0, rad=0.08, loff=(0, -1.2))
    for k in ("motor0", "motor2", "vic"):
        edge(ax, N[k], N["motorR"], 0.9, "#9aa3a0", None, rad=-0.04 if k != "motor0" else 0.0)
    ax.text(62, 85.4, "other inputs of this reader (5–10% each)", fontsize=7.2, color=INK3, ha="center",
            bbox=dict(fc=BG, ec="none", pad=0.2), zorder=6)

    # what fires it: graded responses (H4 test), drawn under the probed node
    ia = ax.inset_axes([64.5, 41.3, 15.5, 11.2], transform=ax.transData)
    ref = sum(r["act"] for r in h4["reference"]) / len(h4["reference"])
    B = h4["branches"]
    rows = [("psychological ‘nerve’", ref, TEAL, None), ("‘strong’, strength", B["strength"]["mean_fire"], BLUE, B["strength"]["auroc"]),
            ("hit, wounded, injured", B["impact_injury"]["mean_fire"], ORANGE, B["impact_injury"]["auroc"]),
            ("courage, no ‘nerve’", B["mental_fortitude"]["mean_fire"], "#b7bdba", B["mental_fortitude"]["auroc"]),
            ("itch, tingle, pulse", B["reaction_sensation"]["mean_fire"], "#b7bdba", B["reaction_sensation"]["auroc"])]
    for i, (lab, v, c, au) in enumerate(rows):
        ia.barh(-i, v, height=0.62, color=c, zorder=2)
        ia.text(-0.015, -i, lab, ha="right", va="center", fontsize=7.2, color=INK if c != "#b7bdba" else INK3)
        ia.text(v + 0.012, -i, f"{v:.2f}" + (f"  (AUROC {au:.2f})" if au is not None else ""), ha="left", va="center", fontsize=6.6, color=INK2)
    ia.set_xlim(0, 0.62), ia.set_ylim(-4.6, 1.15)
    ia.axis("off")
    ia.text(-0.19, 0.78, "WHAT FIRES IT · mean activation (fraction of max), 8 sentences each", fontsize=6.9, color=INK3, fontweight="semibold", va="center")

    # ---- lower half: the lateral family and its inputs
    ax.add_patch(FancyBboxPatch((1.5, 0.6), 121, 37.2, boxstyle="round,pad=0,rounding_size=2", fc="#f7f6fc", ec="#d9d4f0", lw=1.0, zorder=0))
    ax.text(3, 35.9, "LATERAL FAMILY", fontsize=7.8, color=VIOLET, fontweight="semibold", va="center")
    ax.text(14.2, 35.9, "co-fire with ‘nerve’ (r ≈ 0.6), yet separate jobs: zeroing one moves the others by ≤ 0.03. "
            "Below each, the inputs that drive it most, each labelled and tested blind.", fontsize=7.4, color=VIOLET, va="center")
    fam = {"suspense": (17, "suspense", "L7/N9782"), "anticip": (47, "can’t wait", "L9/N1061"),
           "overwh": (77, "overwhelmed", "L12/N3688"), "disap": (107, "‘disappointed’", "L1/N4037")}
    for k, (x, nm, u) in fam.items():
        put(k, x, 26.5, nm, u, "family", w=14)
    ins = {"suspense": [("‘life span’", "L1/N1313"), ("‘watch’ TV, films", "L0/N947"), ("unclear filler", "L4/N7913")],
           "anticip": [("‘eager to’, ‘can’t\nwait to’ (the ‘to’)", "L8/N3632"), ("‘can’t stand’,\n‘can’t bear’", "L7/N10645"), ("‘19th’ century", "L7/N2810")],
           "overwh": [("‘answer to’,\n‘accountable to’", "L8/N8229"), ("reader-need hook\nin ad copy", "L11/N8779"), ("2nd piece of a\ntechnical term", "L6/N2446")],
           "disap": [("‘disconnect’", "L0/N10546"), ("noun after ‘the’", "L0/N55"), ("‘app’ in code", "L0/N10323")]}
    for k, lst in ins.items():
        fx = fam[k][0]
        lst = sorted(lst, key=lambda t: -lat_w[(t[1], fam[k][2])])
        for i, (nm, u) in enumerate(lst):
            x, y = fx + (i - 1) * 9.9, 9.6
            ok = q.get(u, {}).get("survives", True)
            node(ax, x, y, nm, u, "input" if ok else "faint", badge(u), w=8.9, h=7.2, fs=7.2)
            wv = lat_w[(u, fam[k][2])]
            edge(ax, (x, y + 3.6), (fx + (i - 1) * 3.2, 23.7), 0.8 + 9 * wv, INK if ok else INK3, f"{100 * wv:.0f}%", shrinkA=0, shrinkB=0,
                 fs=6.8, lpos=0.45, loff=(0, 0), alpha=1 if ok else 0.6)

    # the family meets the hub
    edge(ax, (107, 29.3), N["hub"], 1.6, VIOLET, None, shrinkA=0, rad=0.25)
    ax.text(118.3, 44.5, "‘disappointed’ feeds\nthe hub (7%); suspense,\ncan’t-wait and over-\nwhelmed passages drive\nit without ‘nerve’\n(inverse test)",
            fontsize=7.0, color=VIOLET, ha="center", va="center", zorder=6, linespacing=1.35)

    fig.text(0.03, 0.012, "Not drawn: 30 more crawled neurons (mostly spelling and token pieces such as diarr-, trait, -ate, p-value digits, Albany; citations; code). "
             "52 of 60 labels held overall; dense readers and two neurons still unresolved after three rounds count as failed.\n"
             "Input %: share of the target's activation lost when that one neuron is zeroed. Reader numbers: change in the reader (fraction of its range) when ‘nerve’ is boosted. "
             "‘What fires it’: no sentence below the first row contains a ‘nerve’ word.",
             fontsize=7.6, color=INK3, va="bottom", linespacing=1.5)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "nerve_connectome.png", dpi=300, facecolor=BG)
    fig.savefig(OUT / "nerve_connectome.pdf", facecolor=BG)
    print("wrote", OUT / "nerve_connectome.png")


if __name__ == "__main__":
    main()
