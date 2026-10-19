"""Why did the model say that? The neuron poles behind a prediction, with the map's label for each.

  python -m bonsai_interp.graph.play why "The capital of France is"            the model's own next token
  python -m bonsai_interp.graph.play why "The capital of France is" --target " Paris"
      [--k 20] [--check 8] [--out why.json]

1. Attribution: every MLP neuron at every position gets a gain of 1; the gradient of the target token's log-probability
   with respect to that gain is its first-order contribution (a * d logp / d a). Positions where the neuron is positive
   count for its + pole, negative ones for its − pole, so each pole gets its own score. Positive = it pushes the model
   toward the target token.
2. Check: the --check strongest poles are zeroed one at a time (at the positions where they hold that sign) and the
   change in the target's log-probability is measured. A pole whose zeroing barely moves the answer has redundant
   backups; attribution is a first-order guide, the zeroing is what happened.
3. The map: each pole's status and label from graph/index.json. Strong unlabelled poles are the obvious next missions.
"""
import json
import sys
from pathlib import Path

import numpy as np

from bonsai_interp.graph.contrib import ROOT


def _index():
    p = ROOT / "graph" / "index.json"
    return json.loads(p.read_text(encoding="utf-8"))["neurons"] if p.exists() else {}


def explain(model, tok, rec, prompt, target=None, k=20, check=8):
    import torch
    from bonsai_interp.env import DEVICE
    ids = torch.tensor(tok(prompt).input_ids, device=DEVICE)[None]
    T, N, nl = ids.shape[1], model.config.intermediate_size, model.config.num_hidden_layers

    def logprobs(masks=None):
        rec.interventions = {l: (lambda a, m=m: a * m.to(a.dtype)) for l, m in (masks or {}).items()}
        rec.mode, rec.layers = "act", set(range(nl))
        try:
            out = model(ids).logits[0, -1].float().log_softmax(-1)
        finally:
            rec.interventions, rec.mode = {}, None
        acts = {l: rec.store.pop(l)[0].detach().float() for l in range(nl)}
        rec.clear()
        return out, acts

    with torch.no_grad():
        lp, _ = logprobs()
    tgt = int(lp.argmax()) if target is None else tok(target, add_special_tokens=False).input_ids[0]
    target_text = tok.decode([tgt])
    masks = {l: torch.ones(1, T, N, device=DEVICE, requires_grad=True) for l in range(nl)}
    with torch.enable_grad():
        lp_g, acts = logprobs(masks)
        lp_g[tgt].backward()
    rows = []
    for l in range(nl):
        g = masks[l].grad[0].float()            # (T, N): a * d logp / d a, per position
        a = acts[l]
        for pole, sel in (("+", a > 0), ("-", a < 0)):
            contrib = torch.where(sel, g, torch.zeros_like(g))
            score = contrib.sum(0)              # (N,)
            top = torch.topk(score.abs(), k).indices
            for n in top.tolist():
                t = int(contrib[:, n].abs().argmax())
                rows.append({"neuron": f"bonsai1/L{l}/N{n}{pole}", "L": l, "n": n, "pole": pole,
                             "attribution": float(score[n]), "position": t, "token": tok.decode([int(ids[0, t])])})
    # two lists: what pushes toward the answer and what pushes away (late layers are full of single-token suppressors,
    # so one list ranked by size would hide the supporters)
    support = sorted((r for r in rows if r["attribution"] > 0), key=lambda r: -r["attribution"])[: (k + 1) // 2]
    oppose = sorted((r for r in rows if r["attribution"] < 0), key=lambda r: r["attribution"])[: k // 2]
    for r in support:
        r["side"] = "toward"
    for r in oppose:
        r["side"] = "away"
    rows = support + oppose
    # check the strongest of each side by zeroing them where they hold that sign
    base = float(lp[tgt])
    for r in support[: (check + 1) // 2] + oppose[: check // 2]:
        a = acts[r["L"]][:, r["n"]]
        keep = ~((a > 0) if r["pole"] == "+" else (a < 0))
        m = torch.ones(1, T, N, device=DEVICE)
        m[0, :, r["n"]] = keep.float()
        with torch.no_grad():
            lp2, _ = logprobs({r["L"]: m})
        r["ablation_dlogp"] = float(lp2[tgt]) - base
    idx = _index()
    for r in rows:
        rec_ = idx.get(r["neuron"])
        r["status"] = rec_["status"] if rec_ else None
        r["label"] = (rec_.get("short") or rec_.get("label")) if rec_ else None
    return {"prompt": prompt, "target": target_text, "target_id": tgt, "p": float(np.exp(base)),
            "top_prediction": tok.decode([int(lp.argmax())]), "poles": rows}


def render(res):
    out = [f'prompt: {res["prompt"]!r}  ->  target {res["target"]!r} (p = {res["p"]:.2f}; the model\'s top pick is {res["top_prediction"]!r})']
    for side, title in (("toward", f'PUSHING TOWARD {res["target"]!r}'), ("away", f'PUSHING AWAY FROM {res["target"]!r}')):
        out += ["", title, f'{"pole":<22} {"attrib":>7} {"zeroed":>8}  {"at token":<14} map']
        for r in (r for r in res["poles"] if r.get("side") == side):
            z = f'{r["ablation_dlogp"]:+.2f}' if "ablation_dlogp" in r else "     -"
            m = f'{r["status"]}: {(r["label"] or "")[:60]}' if r["status"] else "UNLABELLED"
            out.append(f'{r["neuron"]:<22} {r["attribution"]:+7.2f} {z:>8}  {r["token"]!r:<14} {m}')
    todo = [r for r in res["poles"] if not r["status"]][:5]
    out += ["", "Attribution: + pushes toward the target, − away. 'zeroed' is the measured change in the target's log-probability",
            "when that pole is zeroed (negative = the answer gets less likely without it)."]
    if todo:
        out += ["", "Unlabelled poles that matter here (missions that would explain this prediction):"]
        out += [f'  python -m bonsai_interp.graph.play brief interpret --neuron {r["neuron"]} --handle <you>' for r in todo]
    return "\n".join(out)


def main(a):
    from bonsai_interp.env import limit_memory
    from bonsai_interp.model import load
    limit_memory(0.8)
    model, tok, rec = load("bonsai1")
    res = explain(model, tok, rec, a.prompt, a.target, a.k, a.check)
    print(render(res))
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {a.out}")
