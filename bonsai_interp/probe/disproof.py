"""Stage 10 (disproof): try to show the probed pole is NOT monosemantic (plan: disproof_plan.json).

  d1          every strong token (>= 0.3) in the band scan: does a nerve morpheme sit within 3 tokens?
              unexplained ones -> d1_unexplained.json for blind annotation
  d2_request  30 random contexts per activation band, shuffled across bands (annotator cannot see
              the band) -> d2_request.json
  d2_score    join the blind annotations back to bands -> d2_results.json
  d3          large natural leak search with SciPy DE over sentences without any nerve morpheme

  python -m bonsai_interp.probe.disproof d1|d2_request|d2_score|d3 9/3688 +
"""
import re
import sys

import numpy as np

from bonsai_interp import explore
from bonsai_interp.probe import bands as B
from bonsai_interp.probe.workspace import Workspace

MORPH = re.compile(r"nerv|nerf|нерв|神经|神經|신경|عصب|اعصاب|أعصاب|nervi", re.I)
D2_BANDS = [(0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5)]


def d1(ws, tag="web"):
    d = B.load_bands(tag)
    a = d[f"L{ws.L}_N{ws.n}"].astype(np.float32)
    pmax = float(explore.scan(ws.model)["top_val"][ws.L, 0, 0, ws.n])
    f = a / pmax
    toks, tk = explore.corpus(), explore.tokenizer()
    w, p = np.where(f >= 0.3)
    expl, unexpl = 0, []
    for wi, pi in zip(w, p):
        ids = toks[wi]
        near = tk.decode(ids[max(0, pi - 3):pi + 4])
        if MORPH.search(near):
            expl += 1
        else:
            ctx = tk.decode(ids[max(0, pi - 60):pi]) + "⟦" + tk.decode(ids[pi:pi + 1]) + "⟧" + tk.decode(ids[pi + 1:pi + 13])
            unexpl.append({"frac": round(float(f[wi, pi]), 3), "window": int(wi), "text": ctx.replace("\n", "⏎")})
    res = {"strong_tokens": int(len(w)), "with_morpheme": int(expl), "share_with_morpheme": expl / max(1, len(w)),
           "unexplained": sorted(unexpl, key=lambda r: -r["frac"])}
    ws.save("d1_results.json", res)
    print(f"strong tokens {len(w)}; with a nerve morpheme within 3 tokens: {expl} ({res['share_with_morpheme']:.1%}); unexplained {len(unexpl)}")
    for r in res["unexplained"][:25]:
        print(f"  {r['frac']:.2f} | {r['text'][-110:]}")


def d2_request(ws, tag="web", k=30):
    s = B.sample(tag, ws.L, ws.n, "+", k=k, bands=D2_BANDS)
    items = [{"band": b, **e} for b, v in s.items() if not b.startswith("_") for e in v["examples"]]
    rng = np.random.default_rng(1)
    order = rng.permutation(len(items))
    blind = [{"id": int(i), "text": items[j]["text"]} for i, j in enumerate(order)]
    key = [{"id": int(i), "band": items[j]["band"], "frac": items[j]["frac"]} for i, j in enumerate(order)]
    ws.save("d2_request.json", {"instructions": "see agent prompt", "items": blind})
    ws.save("d2_key.json", {"key": key, "band_counts": {b: v["count"] for b, v in s.items() if not b.startswith("_")},
                            "total_tokens": s["_total_tokens"]})
    print(f"{len(blind)} contexts across {len(D2_BANDS)} bands; counts: " + str({b: v['count'] for b, v in s.items() if not b.startswith('_')}))


def d2_score(ws):
    key = {k["id"]: k for k in ws.load("d2_key.json")["key"]}
    ann = ws.load("d2_annotations.json")["annotations"]
    bands = {}
    for a in ann:
        b = key[a["id"]]["band"]
        bands.setdefault(b, []).append(a)
    out = {}
    for b, rows in sorted(bands.items()):
        cats = {c: sum(r["category"] == c for r in rows) / len(rows) for c in "ABCDE"}
        themes = {}
        for r in rows:
            if r["category"] == "D" and r.get("theme"):
                themes[r["theme"]] = themes.get(r["theme"], 0) + 1
        out[b] = {"n": len(rows), "share": cats, "D_themes": dict(sorted(themes.items(), key=lambda t: -t[1]))}
        print(f"{b}: n={len(rows)} " + " ".join(f"{c}={v:.0%}" for c, v in cats.items()) + f" | D themes {out[b]['D_themes']}")
    ws.save("d2_results.json", out)


def d3(ws, n_corpus=500):
    from bonsai_interp.inverse import solvers as S
    from bonsai_interp.inverse.problem import Goal
    from bonsai_interp.probe import forward, textinv
    from bonsai_interp.probe.run import setup
    texts = [r["text"] for r in forward.corpus_sentences(n_corpus, seed=11)]
    for b in range(3):
        for nb in ws.load(f"web_tests_claude_{b}.json")["neighbours"]:
            texts += nb["fire"]
    for nb in ws.load("web_tests_bonsai.json")["neighbours"]:
        texts += nb["fire"]
    texts = [t for t in dict.fromkeys(texts) if not MORPH.search(t)]
    model, tok, rec, ev = setup(ws)
    solo = ev(texts)
    rows = [{"text": t, "act": r["act"], "opp": r["opp"], "peak_token": r["peak_token"], "rel_pos": 0.0, "rel_neg": 0.0, "kind": "leak-bank"}
            for t, r in zip(texts, solo)]
    best_solo = sorted(rows, key=lambda r: -r["act"])[:20]
    man = textinv.SentenceManifold({"sentences": rows}, rows, "rel_pos")
    p = textinv.build_problem(ev, man)
    r = S.differential_evolution(p, [Goal("act", "max")], maxiter=20, popsize=8)
    res = {"bank_size": len(rows), "best_single_sentences": best_solo,
           "de_max": {"evals": r.n_evals, **textinv.describe(p, r.x, r.y)},
           "share_solo_ge_0.3": float(np.mean([x["act"] >= 0.3 for x in rows]))}
    ws.save("d3_results.json", res)
    print(f"bank {len(rows)} sentences without a nerve morpheme; solo >= 0.3: {res['share_solo_ge_0.3']:.1%}")
    for x in best_solo[:12]:
        print(f"  solo {x['act']:.2f} peak={x['peak_token']!r:<12} {x['text'][:100]}")
    print(f"DE max over passages: {r.y[0]:.2f} ({r.n_evals} evals) | {res['de_max']['text'][:200]} | peak {res['de_max']['peak_token']!r}")


if __name__ == "__main__":
    L, n = map(int, sys.argv[2].split("/"))
    ws = Workspace(L, n, sys.argv[3] if len(sys.argv) > 3 else "+")
    {"d1": d1, "d2_request": d2_request, "d2_score": d2_score, "d3": d3}[sys.argv[1]](ws)
