"""Stage 3 (forward-induce): push the neuron and watch what the model does.

  direct     the neuron's write direction read through the final norm and unembedding:
             which tokens it promotes / suppresses directly (weights only)
  sweep      add c x (pole's corpus max) to the neuron at every position of neutral
             prompts, c from negative (toward the opposite pole) to positive; measure
             the next-token probability of bank.json's concept_tokens and
             opposite_tokens, and KL from the unsteered model
  generate   greedy continuations at a few c, for an LLM judge (induce_judgement.json)
"""
import numpy as np
import torch

from bonsai_interp.probe.run import setup
from bonsai_interp.probe.workspace import Workspace

# Neutral default prompts. A bank may set "induce_prompts"; the opposite pole's words come from
# bank["opposite_tokens"]. (The first L9/N3688 run used topic-tailored prompts; they are recorded
# in its induce.json.)
PROMPTS = [
    "The thing I remember most about that day is",
    "After lunch we walked down to the",
    "The most important thing to know about this is",
    "He looked at the results and said",
]
SWEEP = [-40, -20, -10, -5, -2, 0, 2, 5, 10, 20, 40]
GEN_LEVELS = [-10, -5, 0, 5, 10]


def token_ids(tok, words):
    ids = []
    for w in words:
        t = tok(w, add_special_tokens=False).input_ids
        if t:
            ids.append(t[0])
    return sorted(set(ids))


@torch.no_grad()
def run(ws: Workspace, gen_tokens=32):
    model, tok, rec, ev = setup(ws)
    bank = ws.require("bank.json", "write bank.json first")
    fwd = ws.load("forward.json")
    pmax = abs(fwd["max"][ws.pole])
    sign = 1.0 if ws.pole == "+" else -1.0
    L, n = ws.L, ws.n

    # direct logit effect of the neuron's output direction
    d = model.model.layers[L].mlp.down_proj.weight()[:, n].float()
    logits = (model.lm_head.weight.float() @ (model.model.norm.weight.float() * d)) * sign
    top, bot = logits.topk(20), (-logits).topk(20)
    direct = {"promotes": [tok.decode([i]) for i in top.indices.tolist()],
              "suppresses": [tok.decode([i]) for i in bot.indices.tolist()]}

    concept = token_ids(tok, bank.get("concept_tokens", []))
    opposite = token_ids(tok, bank.get("opposite_tokens", []))
    prompts = bank.get("induce_prompts", PROMPTS)

    def steer(c):
        rec.interventions = {} if c == 0 else {L: (lambda a, c=c: _add(a, n, sign * c * pmax))}

    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    last = enc.attention_mask.sum(1) - 1
    sweep, base = [], None
    for c in SWEEP:
        steer(c)
        lg = model(**enc).logits.float()
        rec.interventions = {}
        lp = torch.log_softmax(lg[torch.arange(len(prompts)), last], -1)
        if c == 0:
            base = lp
        sweep.append({"c": c, "p_concept": lp[:, concept].exp().sum(-1).mean().item(),
                      "p_opposite": lp[:, opposite].exp().sum(-1).mean().item(),
                      "top_next": [tok.decode([i]) for i in lp.mean(0).topk(8).indices.tolist()]})
    for s, c in zip(sweep, SWEEP):
        steer(c)
        lp = torch.log_softmax(model(**enc).logits.float()[torch.arange(len(prompts)), last], -1)
        rec.interventions = {}
        s["kl_from_base"] = (lp.exp() * (lp - base)).sum(-1).mean().item()
        # tokens whose probability rises most (mean over prompts), among those that end up >= 1e-4
        gain = (lp - base).mean(0)
        gain[lp.exp().mean(0) < 1e-4] = -1e9
        s["top_gain"] = [tok.decode([i]) for i in gain.topk(10).indices.tolist()]

    gens = []
    tok.padding_side = "left"
    for c in GEN_LEVELS:
        steer(c)
        e = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        out = model.generate(**e, max_new_tokens=gen_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        rec.interventions = {}
        for prompt, o in zip(prompts, out):
            gens.append({"c": c, "prompt": prompt,
                         "continuation": tok.decode(o[e.input_ids.shape[1]:], skip_special_tokens=True)})
    tok.padding_side = "right"
    res = {"neuron": ws.id, "pole_max": pmax, "direct": direct,
           "concept_tokens": [tok.decode([i]) for i in concept], "opposite_tokens": [tok.decode([i]) for i in opposite],
           "sweep": sweep, "generations": gens}
    ws.save("induce.json", res)
    print("direct promotes:", direct["promotes"][:12])
    print("direct suppresses:", direct["suppresses"][:12])
    for s in sweep:
        print(f"c={s['c']:>4}  p(concept)={s['p_concept']:.4f}  p(opposite)={s['p_opposite']:.4f}  KL={s['kl_from_base']:.3f}  gains: {s['top_gain'][:8]}")
    for g in gens:
        print(f"[c={g['c']:>3}] {g['prompt']} ⟶ {g['continuation']!r}")
    return res


def _add(a, n, delta):
    a = a.clone()
    a[..., n] += delta
    return a
