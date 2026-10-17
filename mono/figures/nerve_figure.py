"""Publication figure for the 'nerve' neuron (L9/N3688, Bonsai-8B 1-bit), drawn from the probe's
result files so every mark is measured data.

  python -m mono.figures.nerve_figure   ->  reports/figures/nerve_neuron.{png,pdf}
"""
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from mono.model import ROOT

matplotlib.use("Agg")
P = ROOT / "probes"
NERVE = P / "bonsai1_L9_N3688_pos"
OUT = ROOT / "reports" / "figures"

INK, INK2, INK3 = "#14191a", "#4a5351", "#7d8784"
RULE, GRID, BG = "#d9dedb", "#eceeec", "#ffffff"
ACCENT, ACCENT_SOFT = "#1f6b5c", "#e2eeea"
BLUE, ORANGE, AQUA, RED, MID = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#d9d7d1"
SANS, SERIF, MONO = "IBM Plex Sans", "Georgia", "Cascadia Mono"
DIVERGE = LinearSegmentedColormap.from_list("pole", [RED, MID, BLUE])

plt.rcParams.update({"font.family": SANS, "font.size": 8.5, "text.color": INK, "axes.labelcolor": INK2,
                     "axes.edgecolor": RULE, "xtick.color": INK3, "ytick.color": INK3, "axes.linewidth": 0.8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "svg.fonttype": "none", "pdf.fonttype": 42})


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def panel_label(ax, letter, title, sub=None, x=-0.02, y=1.06):
    kw = dict(xy=(x, y), xycoords="axes fraction", textcoords="offset points", annotation_clip=False)
    ax.annotate(letter, xytext=(0, 10), fontsize=15, fontweight="bold", family=SERIF, va="bottom", color=INK, **kw)
    ax.annotate(title, xytext=(17, 11.5), fontsize=10.5, fontweight="semibold", va="bottom", color=INK, **kw)
    if sub:
        ax.annotate(sub, xytext=(17, 1), fontsize=7.6, va="bottom", color=INK3, **kw)


def strip(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]), ax.set_yticks([])


# ----------------------------------------------------------------------------- A. iris
def iris(ax, solo):
    key = solo["holdout"]["hypothesis"]
    pts = []
    for r in solo["rows"]:
        k = r["kind"]
        sec = ("on" if r.get(key, 0) >= 0.5 else "opposite" if r["rel_neg"] >= 0.8 else "unrelated") if k == "top" \
            else "unrelated" if k == "corpus" else k
        pts.append(dict(sec=sec, rel=r.get(key, 0), act=r["act"], opp=r["opp"], text=r["text"]))
    for r in solo["holdout"]["rows"]:
        pts.append(dict(sec="held-out", rel=0.95 if r["predict"] == "fire" else 0.15, act=r["act"], opp=r["opp"], text=r["text"], ho=True))
    order = ["on", "held-out", "near", "spelling", "opposite", "unrelated"]
    names = {"on": "on-concept", "held-out": "held-out\n(pre-registered)", "near": "related meaning",
             "spelling": "look-alikes", "opposite": "opposite\nconcept", "unrelated": "unrelated\n& corpus"}
    secs = [s for s in order if any(p["sec"] == s for p in pts)]
    R1, R = 0.32, 1.0
    rad = lambda rel: R1 + (1 - np.clip(rel, 0, 1)) * (R - R1)
    span = 2 * np.pi / len(secs)
    for rel, ls in ((1, "-"), (0.5, (0, (2, 3))), (0, "-")):
        ax.add_patch(plt.Circle((0, 0), rad(rel), fill=False, ec=GRID, lw=0.9, ls=ls, zorder=0))
    ax.text(0, rad(0.5) + 0.02, "relevance 0.5", ha="center", va="bottom", fontsize=6.5, color=INK3)
    ax.text(0, rad(0) + 0.02, "unrelated", ha="center", va="bottom", fontsize=6.5, color=INK3)
    for si, s in enumerate(secs):
        a0 = np.pi / 2 - si * span
        ax.plot([0.2 * np.cos(a0), 1.06 * np.cos(a0)], [0.2 * np.sin(a0), 1.06 * np.sin(a0)], color=GRID, lw=0.9, zorder=0)
        am = a0 - span / 2
        ha = "center" if abs(np.cos(am)) < 0.3 else ("left" if np.cos(am) > 0 else "right")
        ax.text(1.14 * np.cos(am), 1.14 * np.sin(am), names[s], ha=ha, va="center", fontsize=7.6, color=INK2)
        mine = sorted([p for p in pts if p["sec"] == s], key=lambda p: -p["rel"])
        for i, p in enumerate(mine):
            a = a0 - span * (0.08 + 0.84 * (i / max(1, len(mine) - 1)))
            rr = rad(p["rel"]) + ((i % 5) - 2) * 0.022
            p["x"], p["y"] = rr * np.cos(a), rr * np.sin(a)
    pts.sort(key=lambda p: abs(p["act"] - p["opp"]))
    v = np.clip([p["act"] - p["opp"] for p in pts], -1, 1)
    c = DIVERGE(0.5 + 0.5 * np.sign(v) * np.sqrt(np.abs(v)))
    size = 9 + 60 * np.clip([max(p["act"], p["opp"]) for p in pts], 0, 1)
    ax.scatter([p["x"] for p in pts], [p["y"] for p in pts], s=size, c=c, edgecolors=[INK2 if p.get("ho") else "white" for p in pts],
               linewidths=[0.7 if p.get("ho") else 0.5 for p in pts], zorder=3)
    ax.add_patch(plt.Circle((0, 0), 0.17, fc=ACCENT_SOFT, ec=ACCENT, lw=1.6, zorder=4))
    ax.text(0, 0.03, "L9/N3688", ha="center", va="center", fontsize=6.8, fontweight="semibold", family=MONO, zorder=5)
    ax.text(0, -0.06, "+ pole", ha="center", va="center", fontsize=7, color=INK3, zorder=5)
    callouts = [("nerve-racking experience", "'a nerve-racking experience'", (0.42, 1.43), "left"),
                ("Nerven blank", "'lagen ihre Nerven blank' (de)", (0.42, 1.30), "left"),
                ("lost his nerve at", "'he lost his nerve' (gated)", (0.42, 1.17), "left"),
                ("pinched nerve", "'pinched nerve …'", (1.0, 0.95), "left"),
                ("Pain receptors", "'pain receptors …'", (-1.12, -0.32), "right"),
                ("Nerf darts", "'Nerf darts'", (-1.0, -1.25), "right"),
                ("Emperor Nero", "'Emperor Nero'", (0.05, -1.40), "left")]
    used = set()
    for needle, label, (tx, ty), ha in callouts:
        p = next((q for q in pts if needle in q["text"] and id(q) not in used), None)
        if not p:
            continue
        used.add(id(p))
        val = f"+ pole {p['act']:.2f}" if p["act"] >= p["opp"] else f"− pole {p['opp']:.2f}"
        ax.annotate(f"{label}   {val}", (p["x"], p["y"]), (tx, ty), fontsize=6.6, color=INK2, ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color=INK3, lw=0.5, shrinkA=1, shrinkB=3,
                                    connectionstyle="arc3,rad=0.0"), zorder=6)
    ax.set_xlim(-1.62, 1.62), ax.set_ylim(-1.52, 1.5)
    ax.set_aspect("equal")
    strip(ax)
    # colour key
    cax = ax.inset_axes([0.14, -0.005, 0.72, 0.02])
    cax.imshow(np.linspace(0, 1, 256)[None], aspect="auto", cmap=DIVERGE, extent=(-1, 1, 0, 1))
    cax.set_yticks([]), cax.set_xticks([-1, 0, 1], ["drives opposite pole\n(bodily sensation)", "neither", "drives probed pole\n(psychological 'nerve')"])
    cax.tick_params(length=0, labelsize=6.8, pad=2)
    for s in cax.spines.values():
        s.set_visible(False)


# ----------------------------------------------------------------------------- B. circuit
def box(ax, x, y, w, h, title, sub, kind="unit", badge=None):
    fc, ec, lw = {"unit": (BG, INK, 1.0), "probed": (ACCENT_SOFT, ACCENT, 2.0), "faint": (BG, INK3, 0.9),
                  "text": ("#f6f7f5", "#f6f7f5", 0)}[kind]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.006,rounding_size=0.012", fc=fc, ec=ec, lw=lw,
                                ls=(0, (4, 3)) if kind == "faint" else "-", zorder=2))
    ax.text(x, y + h * 0.16, title, ha="center", va="center", fontsize=8.6 if kind != "text" else 8, zorder=3,
            family=MONO if kind in ("unit", "probed", "faint") else SANS, fontweight="semibold" if kind == "probed" else "normal")
    ax.text(x, y - h * 0.2, sub, ha="center", va="center", fontsize=7, color=INK2, zorder=3)
    if badge:
        ok = badge.startswith("✓")
        ax.text(x + w / 2 - 0.004, y + h / 2 + 0.002, badge, ha="right", va="bottom", fontsize=6.4, zorder=4,
                color="#1f7a3a" if ok else "#b3261e", fontweight="semibold")


def arrow(ax, p, q, label=None, lw=1.5, color=INK, style="-|>", ls="-", lpos=0.5, loff=(0, 0.018), lcolor=None, fs=7.2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, lw=lw, color=color, ls=ls, zorder=1,
                                 shrinkA=0, shrinkB=0))
    if label:
        mx, my = p[0] + (q[0] - p[0]) * lpos + loff[0], p[1] + (q[1] - p[1]) * lpos + loff[1]
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=fs, color=lcolor or color,
                bbox=dict(fc=BG, ec="none", pad=0.6), zorder=5)


def circuit(ax):
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    strip(ax)
    W, H = 0.155, 0.105
    # text inputs
    box(ax, 0.075, 0.80, 0.13, 0.12, "Latin-script text", "nerve-racking · Nerven\nles nerfs à vif", "text")
    box(ax, 0.075, 0.50, 0.13, 0.12, "Arabic / German text", "'asab (nerve) ·\nNervensystem", "text")
    box(ax, 0.075, 0.18, 0.13, 0.12, "Preceding context", "pain receptors ·\nSchmerzen", "text")
    # units
    box(ax, 0.29, 0.80, W, H, "L3/N6493", "spelling 'nerv / nerf'", badge="✓ 8/8")
    box(ax, 0.29, 0.50, W, H, "L8/N7756", "nerve concept (Arabic-heavy)", badge="✓ 8/8")
    box(ax, 0.29, 0.18, W, H, "L8/N3858", "pain detector", badge="✓ 7/8")
    box(ax, 0.565, 0.60, 0.19, 0.16, "L9/N3688", "+  psychological strain, composure\n−  bodily sensation (pain, numbness)", "probed")
    box(ax, 0.86, 0.80, W, H, "L10/N6901", "motor ↔ sensory unit", badge="× 4/8 (relabelled)")
    box(ax, 0.86, 0.60, W, H, "L14/N9470", "emotional distress")
    box(ax, 0.86, 0.40, W, H, "many readers", "each < 0.06 of range", "faint")
    # edges in
    for y in (0.80, 0.50, 0.18):
        arrow(ax, (0.142, y), (0.29 - W / 2 - 0.004, y), lw=1.0, color=INK3)
    arrow(ax, (0.29 + W / 2, 0.80), (0.565 - 0.095, 0.66), "74% of activation", lw=3.2, lpos=0.45, loff=(0, 0.02))
    arrow(ax, (0.29 + W / 2, 0.50), (0.565 - 0.095, 0.58), "61%", lw=2.6, lpos=0.5, loff=(0, 0.012))
    # gate: pain closes both routes (flat-ended)
    arrow(ax, (0.29 + W / 2, 0.18), (0.535, 0.517), None, lw=1.8, color=ORANGE, style="-[", ls="-")
    ax.text(0.47, 0.30, "gate: pain context before 'nerve'\ncloses both routes\n(e.g. 0.72 → 0.03; pre-registered,\n9 of 11 predictions held)",
            fontsize=6.9, color=ORANGE, ha="left", va="center", bbox=dict(fc=BG, ec="none", pad=0.8))
    ax.text(0.29, 0.715, "'lost his nerve' and 'had the nerve to'\nclose this route on their own", fontsize=6.6, color=INK2,
            ha="center", va="top", style="italic")
    # edges out
    arrow(ax, (0.565 + 0.095, 0.64), (0.86 - W / 2 - 0.004, 0.80), "+0.12", lw=1.6, lpos=0.55, loff=(0, 0.01))
    arrow(ax, (0.565 + 0.095, 0.60), (0.86 - W / 2 - 0.004, 0.60), "+0.06", lw=1.1, lpos=0.5, loff=(0, 0.01))
    arrow(ax, (0.565 + 0.095, 0.56), (0.86 - W / 2 - 0.004, 0.40), None, lw=0.9, color=INK3, ls=(0, (3, 3)))
    ax.text(0.5, 0.025, "Input arrows: share of the probed pole's activation lost when that one neuron is zeroed.  Output arrows: change in the "
            "reader (fraction of its range) when the probed neuron is zeroed.\nBadges: pre-registered 8-sentence quick checks of each neighbour's "
            "label (fired / 8).  A third, minor route ('strong') uses none of these inputs.", fontsize=6.6, color=INK3, ha="center", va="bottom")


# ----------------------------------------------------------------------------- C. gate test
def gates(ax, gate):
    names = {"A_open_latin": "Open constructions", "B_closed_latin": "Closed constructions",
             "C_concept_route": "Arabic (concept route)", "D_controls_latin_anatomical": "Anatomical controls"}
    rows, y = [], 0
    heads = []
    for k, rs in gate["rows"].items():
        rs = [r for r in rs if r["scored"]]
        heads.append((y, names[k]))
        y += 1
        for r in rs:
            rows.append((y, r))
            y += 1
        y += 0.4
    for yy, r in rows:
        ax.plot([r["nerve_pain"], r["nerve_neutral"]], [yy, yy], color="#b9bfbc", lw=1.2, zorder=1, solid_capstyle="round")
        ax.scatter([r["nerve_neutral"]], [yy], s=22, color=BLUE, zorder=3, edgecolors="white", linewidths=0.5)
        ax.scatter([r["nerve_pain"]], [yy], s=22, color=ORANGE, zorder=3, edgecolors="white", linewidths=0.5)
        t = r["text"]
        if not t.isascii():
            t = {"يتكون": "Arabic: peripheral nervous system …", "أصيب": "Arabic: optic nerve damaged …",
                 "كانت": "Arabic: my nerves were tense …", "فقد": "Arabic: he lost his nerves …"}.get(t.split()[0], t[:38])
        ax.text(-0.03, yy, (t[:44] + "…") if len(t) > 45 else t, ha="right", va="center", fontsize=6.4, color=INK2)
    for yy, h in heads:
        ax.text(-0.03, yy, h, ha="right", va="center", fontsize=7.4, fontweight="semibold", color=INK)
    ax.axvline(0.3, color=INK3, lw=0.8, ls=(0, (3, 3)), zorder=0)
    ax.text(0.31, y - 0.3, "fires (≥ 0.3)", fontsize=6.4, color=INK3, va="bottom")
    ax.set_ylim(y, -1.2)
    ax.set_xlim(0, 1.2)
    ax.set_yticks([])
    ax.set_xlabel("probed pole on the sentence (fraction of corpus max)", fontsize=7.4)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.scatter([], [], s=22, color=BLUE, label="after a neutral sentence")
    ax.scatter([], [], s=22, color=ORANGE, label="after a pain sentence")
    ax.legend(loc="lower right", fontsize=6.8, frameon=False, handletextpad=0.3, borderaxespad=0.2)


# ----------------------------------------------------------------------------- D. induction
def induce(ax, ind):
    sw = ind["sweep"]
    cs = [s["c"] for s in sw]
    x = np.arange(len(cs))
    ax.plot(x, [max(s["p_concept"], 1e-5) for s in sw], color=BLUE, lw=1.8, marker="o", ms=3.5, label="P(label words: nerve, nervous …)")
    ax.plot(x, [max(s["p_opposite"], 1e-5) for s in sw], color=ORANGE, lw=1.8, marker="o", ms=3.5, label="P(opposite words: pain, numb …)")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 4)
    ax.set_xticks(x, [f"{c:+d}" if c else "0" for c in cs])
    ax.set_xlabel("added to the neuron at every position (× its corpus max)", fontsize=7.4)
    ax.set_ylabel("next-token probability", fontsize=7.4)
    ax.grid(axis="y", color=GRID, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.axvspan(-0.5, cs.index(0) - 0.5, color="#fdf1ec", zorder=0)
    ax.axvspan(cs.index(0) + 0.5, len(cs) - 0.5, color="#edf4fc", zorder=0)
    gain = {s["c"]: s["top_gain"] for s in sw}
    stop = {"with", "during", "such", "over", "also", "the", "by", "was", "me", "you", "we", "your", "true", "wrong"}
    words = lambda c, k: [t.strip() for t in gain[c] if t.strip().isalpha() and t.strip().lower() not in stop][:k]
    ax.text(-0.3, 0.6, "push toward − pole:\n" + ", ".join(words(-10, 3)) + "\n" + ", ".join(words(-40, 3)),
            fontsize=6.5, color="#a8471e", va="top", linespacing=1.3)
    ax.text(len(cs) - 0.7, 0.6, "push toward + pole:\n" + ", ".join(words(10, 3)) + "\n" + ", ".join(words(10, 6)[3:]),
            fontsize=6.5, color="#1c5cab", va="top", ha="right", linespacing=1.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=6.4, frameon=False, ncol=1, handlelength=1.4)


# ----------------------------------------------------------------------------- E. inverse
def inverse(ax, de, gm, det, gmt):
    lv = [0.25, 0.5, 0.75]
    band = [0.05, 0.05, 0.05]
    xs = np.linspace(0.12, 0.88, 50)
    ax.fill_between(xs, xs - 0.05, xs + 0.05, color=ACCENT_SOFT, zorder=0, label="target ± 0.05")
    ax.plot([0.1, 0.9], [0.1, 0.9], color=INK3, lw=0.8, ls=(0, (3, 3)), zorder=1)
    for solver, res, mk, col, dx in (("SciPy DE", de, "s", AQUA, -0.012), ("globalMOO", gm, "o", BLUE, 0.012)):
        ys = [res["goals"][f"level_{int(l * 100)}"]["y"]["act"] for l in lv]
        ev = [res["goals"][f"level_{int(l * 100)}"]["evals"] for l in lv]
        ax.scatter(np.array(lv) + dx, ys, marker=mk, s=34, color=col, zorder=3, edgecolors="white", linewidths=0.6, label=solver)
    for res, col, dx, mk in ((det, AQUA, -0.012, "s"), (gmt, BLUE, 0.012, "o")):
        ys = [res["goals"][f"tight_{int(l * 100)}"]["y"]["act"] for l in lv]
        ax.scatter(np.array(lv) + dx, ys, marker=mk, s=34, facecolors="none", edgecolors=col, linewidths=1.2, zorder=3)
    tot_g = gm["gmoo"]["doe_evals"] + sum(g["evals"] for g in gm["goals"].values())
    tot_d = sum(g["evals"] for g in de["goals"].values())
    ax.text(0.04, 0.96, f"model evaluations for the full\n8-goal battery:\nglobalMOO {tot_g:,}  ·  SciPy DE {tot_d:,}",
            transform=ax.transAxes, fontsize=6.8, color=INK2, va="top")
    ax.text(0.97, 0.04, "filled: ±5% goals   hollow: ±1.5% goals\nfrom a neutral start (tight)", transform=ax.transAxes,
            fontsize=6.3, color=INK3, ha="right", va="bottom")
    ax.set_xlim(0.1, 0.9), ax.set_ylim(0.1, 0.9)
    ax.set_xticks(lv), ax.set_yticks(lv)
    ax.set_xlabel("requested activation (fraction of corpus max)", fontsize=7.4)
    ax.set_ylabel("achieved by the found passage", fontsize=7.4)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.36), fontsize=6.6, frameon=False, handletextpad=0.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ----------------------------------------------------------------------------- F. calibration
def calibration(ax, sols):
    names = ["nerve\nL9/N3688", "genre\nL10/N5749", "unclear\nL22/N11024"]
    fire = [s["holdout"]["fire_rate_predicted_fire"] for s in sols]
    opp = [s["opposite_pole"]["strong_on"] for s in sols]
    x = np.arange(3)
    w = 0.36
    b1 = ax.bar(x - w / 2 - 0.01, fire, w, color=BLUE, label="new label-written sentences that fire it")
    b2 = ax.bar(x + w / 2 + 0.01, opp, w, color=AQUA, label="opposite pole fires on its own label")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.0%}", ha="center", fontsize=6.8, color=INK2)
    verdicts = ["one context-gated\naxis", "two clean,\nunrelated poles", "no single\ncoherent job"]
    for xi, v, col in zip(x, verdicts, ["#1f7a3a", "#1f7a3a", "#b3261e"]):
        ax.text(xi, 1.08, v, ha="center", va="bottom", fontsize=6.6, color=col, fontweight="semibold")
    ax.set_xticks(x, names, fontsize=7)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.5, 1], ["0", "50%", "100%"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=6.5, frameon=False, ncol=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def main():
    solo = load(NERVE / "solo.json")
    fig = plt.figure(figsize=(17, 11.2), dpi=300, facecolor=BG)
    fig.text(0.035, 0.972, "Reading one neuron three ways", family=SERIF, fontsize=22, fontweight="bold", va="top")
    fig.text(0.035, 0.937, "Layer 9, neuron 3688 of Prism ML's 1-bit Bonsai-8B is a context-gated axis for the word 'nerve': "
             "psychological strain and composure at one pole, bodily sensation at the other.", fontsize=10.5, color=INK2, va="top")
    fig.text(0.035, 0.918, "Forward: what text fires it.   Inverse: solvers compose natural passages that hit requested activations.   "
             "Induce: push it inside the model.   Every number below is measured; predictions were committed to git before each test.",
             fontsize=8.6, color=INK3, va="top")

    axA = fig.add_axes([0.02, 0.475, 0.36, 0.385])
    iris(axA, solo)
    panel_label(axA, "A", "Every probe sentence around the neuron", "angle: sentence type · radius: relevance to the label · "
                "colour: which pole it drives · size: strength", x=0.0, y=1.0)

    axB = fig.add_axes([0.40, 0.475, 0.585, 0.385])
    circuit(axB)
    panel_label(axB, "B", "Its circuit: two input routes, a pain gate, a thin fan-out",
                "causal effects from zeroing single neurons on 12 branch examples; labels checked by pre-registered tests", x=0.0, y=1.0)

    axC = fig.add_axes([0.155, 0.075, 0.21, 0.29])
    gates(axC, load(NERVE / "gate_test_results.json"))
    panel_label(axC, "C", "Pre-registered gate test", "same sentence after a neutral vs a pain sentence", x=-0.6, y=1.05)

    axD = fig.add_axes([0.41, 0.075, 0.21, 0.29])
    induce(axD, load(NERVE / "induce.json"))
    gen = {g["c"]: g["continuation"].strip().split(".")[0] for g in load(NERVE / "induce.json")["generations"] if g["prompt"].startswith("Before")}
    panel_label(axD, "D", "Induce: push the neuron, read the model",
                f"“Before the big presentation, I felt …”  0: “{gen[0]}”  ·  −5×: “{gen[-5]}”", x=-0.13, y=1.05)

    axE = fig.add_axes([0.665, 0.075, 0.15, 0.29])
    inverse(axE, load(NERVE / "invert_de.json"), load(NERVE / "invert_gmoo.json"),
            load(NERVE / "invert_de_tight.json"), load(NERVE / "invert_gmoo_tight.json"))
    panel_label(axE, "E", "Inverse round trip", "ask for a level, get a passage", x=-0.2, y=1.05)

    axF = fig.add_axes([0.855, 0.12, 0.13, 0.245])
    calibration(axF, [load(P / d / "solo.json") for d in ("bonsai1_L9_N3688_pos", "bonsai1_L10_N5749_pos", "bonsai1_L22_N11024_pos")])
    panel_label(axF, "F", "Calibration", "same probe, three neurons", x=-0.2, y=1.12)

    fig.text(0.035, 0.012, "Bonsai-8B (1-bit weights, Qwen3-8B architecture), run exactly from unpacked weights; neuron = one channel of "
             "silu(gate)·up; activations are fractions of each pole's maximum over an 8M-token corpus, read after neutral preambles. "
             "Inverse solvers: globalMOO (one model per neuron) and SciPy differential evolution over a bank of natural sentences.",
             fontsize=6.6, color=INK3, va="bottom")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "nerve_neuron.png", dpi=300, facecolor=BG)
    fig.savefig(OUT / "nerve_neuron.pdf", facecolor=BG)
    print("wrote", OUT / "nerve_neuron.png")


if __name__ == "__main__":
    main()
