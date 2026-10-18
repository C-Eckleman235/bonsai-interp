"""Static images for the LessWrong version of the announcement (the web page draws these live).

  python announce/figures.py  ->  announce/img/survey_map.png, announce/img/token_bars.png
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NIGHT, DIM, INK, PLUS, MINUS = "#0f1433", "#8f98c9", "#e8ecff", "#c27a12", "#138a7c"


def survey_map():
    idx = json.loads((ROOT / "graph" / "index.json").read_text(encoding="utf-8"))["neurons"]
    fig = plt.figure(figsize=(12, 7.6), dpi=200, facecolor=NIGHT)
    ax = fig.add_axes([0.06, 0.08, 0.92, 0.84], facecolor=NIGHT)
    rng = np.random.default_rng(7)
    for L in range(36):
        n = rng.uniform(0, 12288, 160)
        y = L + np.where(rng.random(160) < 0.5, 0.3, 0.7)
        ax.scatter(n, y, s=0.6, c=DIM, alpha=0.35, linewidths=0)
    for nid, r in idx.items():
        _, L, n = nid.split("/")
        L, pole, n = int(L[1:]), n[-1], int(n[1:-1])
        k = r.get("scrutiny", {}).get("level", 1)
        y = L + (0.3 if pole == "+" else 0.7)
        c = PLUS if pole == "+" else MINUS
        ax.scatter([n], [y], s=(6 + 9 * k) ** 2 / 6, c=c, alpha=0.18, linewidths=0)
        ax.scatter([n], [y], s=6 + 6 * k, c=c, linewidths=0)
    ax.set_xlim(-100, 12388), ax.set_ylim(36, 0)
    ax.set_yticks(range(0, 36, 4), [f"L{L}" for L in range(0, 36, 4)], color=DIM, fontsize=9, family="monospace")
    ax.set_xticks([0, 4096, 8192, 12287], ["0", "4,096", "8,192", "12,287"], color=DIM, fontsize=9, family="monospace")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.text(0.06, 0.95, "Bonsai-8B: 36 layers × 12,288 MLP neurons. Lit: the 66 poles surveyed so far (amber +, teal −; bigger halo = more scrutiny)",
             color=INK, fontsize=10.5, family="sans-serif")
    fig.savefig(HERE / "img" / "survey_map.png", facecolor=NIGHT)
    plt.close(fig)


def token_bars():
    t = json.loads((ROOT / "probes" / "bonsai1_L9_N3688_pos" / "token_table.json").read_text(encoding="utf-8"))["rows"][:8]
    fam = lambda s: PLUS if "strong" in s else "#2d3fc6" if "nerv" in s else "#b4441f" if any(w in s for w in ("injur", "hit", "wound")) else "#7a8c88"
    fig, ax = plt.subplots(figsize=(8.5, 3.9), dpi=200)
    y = np.arange(len(t))[::-1]
    ax.barh(y, [r["strong"] for r in t], color=[fam(r["token"]) for r in t], height=0.62)
    for yi, r in zip(y, t):
        ax.text(r["strong"] + 1.2, yi, f"{r['strong']}  ({round(100 * r['share'])}% of its uses)", va="center", fontsize=9, color="#435753", family="monospace")
    ax.set_yticks(y, [repr(r["token"]).replace("'", '"') for r in t], family="monospace", fontsize=10)
    ax.set_xlim(0, 118)
    ax.set_xlabel("strong firings (≥ 0.3 of max) in 6.1M corpus tokens", fontsize=9, color="#435753")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("L9/N3688+: which tokens actually fire it", loc="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "img" / "token_bars.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    survey_map()
    token_bars()
    print("wrote announce/img/survey_map.png and token_bars.png")
