"""Exploratory: does the pole fire where 'nerve' is a likely NEXT token?

For corpus windows, compare the neuron's activation at each position (from the band scan) with
the model's own probability that the next token is a nerve token (' nerve', ' nerves', ' Nerve', ...).
Positions whose own token is a nerve token are excluded, so this asks about anticipation, not detection.

  python -m bonsai_interp.probe.predictive 9/3688 +
"""
import json
import sys

import numpy as np
import torch

from bonsai_interp import explore
from bonsai_interp.env import DEVICE
from bonsai_interp.model import load
from bonsai_interp.probe import bands as B
from bonsai_interp.probe.workspace import Workspace

NERVE_WORDS = [" nerve", " nerves", " Nerve", " Nerves", " nerv", " Nerven", " nerf", " nerfs", "nerve", "nerves"]


@torch.no_grad()
def run(ws, n_rand=240, seed=0):
    d = B.load_bands("web")
    a = d[f"L{ws.L}_N{ws.n}"].astype(np.float32)
    pmax = float(explore.scan(ws.model)["top_val"][ws.L, 0, 0, ws.n])
    f = a / pmax
    toks = explore.corpus()[:a.shape[0]]
    model, tok, rec = load(ws.model)
    # whole-word nerve tokens only: a word that splits into pieces would make its first piece count
    nerve_ids = sorted({ids[0] for w in NERVE_WORDS for ids in [tok(w, add_special_tokens=False).input_ids] if len(ids) == 1})
    print("nerve tokens:", [tok.decode([i]) for i in nerve_ids])
    strong_w = sorted(set(np.where(f >= 0.3)[0].tolist()))
    rng = np.random.default_rng(seed)
    wins = sorted(set(strong_w) | set(rng.choice(len(toks), n_rand, replace=False).tolist()))
    P, A, T = [], [], []
    for s in range(0, len(wins), 4):
        w = wins[s:s + 4]
        x = torch.from_numpy(toks[w].astype(np.int64)).to(DEVICE)
        lp = torch.log_softmax(model(x).logits.float(), -1)
        p = lp[..., nerve_ids].exp().sum(-1).cpu().numpy()  # P(next token is a nerve token) at each position
        P.append(p), A.append(f[w]), T.append(toks[w])
    P, A, T = np.concatenate(P).ravel(), np.concatenate(A).ravel(), np.concatenate(T).ravel()
    own = np.isin(T, nerve_ids)
    keep = ~own
    p, act = P[keep], A[keep]
    rank = lambda x: np.argsort(np.argsort(x))
    rho = float(np.corrcoef(rank(p), rank(act))[0, 1])
    hi = p >= np.quantile(p, 0.999)
    res = {"positions": int(keep.sum()), "spearman_act_vs_p_next_nerve": rho,
           "mean_act_top_0.1pct_p": float(act[hi].mean()), "mean_act_all": float(act.mean()),
           "p_next_nerve_when_act_ge_0.3": float(np.median(p[act >= 0.3])) if (act >= 0.3).any() else None,
           "p_next_nerve_overall_median": float(np.median(p)),
           "share_strong_with_p_ge_1pct": float(np.mean(p[act >= 0.3] >= 0.01)) if (act >= 0.3).any() else None}
    # examples: highest p_next_nerve positions and their activations
    tk = explore.tokenizer()
    idx = np.where(keep)[0]
    top = idx[np.argsort(-P[keep])[:15]]
    res["top_anticipation"] = [{"p_next_nerve": float(P[i]), "act": float(A[i]),
                                "context": tk.decode(T[max(0, i - 12):i + 1]).replace("\n", "⏎")} for i in top]
    ws.save("predictive.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "top_anticipation"}, indent=1))
    for r in res["top_anticipation"]:
        print(f"  P(next=nerve) {r['p_next_nerve']:.2f}  act {r['act']:+.2f} | …{r['context'][-70:]}")


if __name__ == "__main__":
    L, n = map(int, sys.argv[1].split("/"))
    run(Workspace(L, n, sys.argv[2] if len(sys.argv) > 2 else "+"))
