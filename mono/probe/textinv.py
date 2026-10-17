"""Stage 2 (invert): a text-level inverse problem over a natural-sentence manifold.

Every candidate input is real, fluent text: a passage of SLOTS sentences drawn from
the neuron's bank (LLM-written on-concept / near-miss / look-alike / opposite /
unrelated sentences, plus rated random corpus sentences). Each slot is placed by two
continuous coordinates, so the input space is smooth enough for a learned inverse:

  u in [0,1]  concept coordinate: rank of the sentence's relevance to the pole label
  v in [0,1]  strength coordinate: rank of the sentence's own (solo) activation among
              sentences of similar relevance (a band of +-BAND around u)

Outputs (fractions of the pole's corpus max, so 1.0 = the strongest of 8M tokens):
  act       max activation of the probed pole over every token of the passage
  opp       max activation of the opposite pole
  dose      max label relevance of the chosen sentences (how on-concept the passage is)
  dose_mean mean label relevance
  dose_opp  max relevance to the opposite pole's label

The goal battery asks the inverse the questions that bear on monosemanticity:
graded targets (does the neuron have a dose-response the label predicts?), "fire
without the concept" (leakage, the essay's most-distant-concept test made adversarial),
"concept present but silent" (does the label over-claim?), and the near-miss ceiling.
"""
import time

import numpy as np
import torch

from mono.inverse.problem import Goal, InverseProblem, Var
from mono.model import StopForward

SLOTS = 3
BAND = 0.08
# Neutral context placed before every probe text. Neurons were characterized on tokens deep
# inside 1024-token windows; at the first few positions of a sequence many are damped, so a
# bare sentence confounds "what the text says" with "where the token sits". Activation is read
# only on the probe text's own tokens and averaged over the preambles.
PREAMBLES = [
    "The museum reopened its east wing last spring after a two-year renovation. Visitors can now see the "
    "restored ceiling panels and a small exhibit about the building's original architects.\n\n",
    "Our office is closed on public holidays, and orders placed over the weekend ship on Monday. "
    "If you have questions about an invoice, please include the order number in your message.\n\n",
    "Rainfall totals for the region were close to the seasonal average. Farmers in the northern valleys "
    "reported a good harvest, while coastal towns saw slightly cooler temperatures than usual.\n\n",
]

GOALS = {
    "level_25": [Goal("act", "~=", 0.25, tol=0.2)],
    "level_50": [Goal("act", "~=", 0.50, tol=0.1)],
    "level_75": [Goal("act", "~=", 0.75, tol=0.067)],
    "max": [Goal("act", "max")],
    "tight_25": [Goal("act", "~=", 0.25, tol=0.06)],
    "tight_50": [Goal("act", "~=", 0.50, tol=0.03)],
    "tight_75": [Goal("act", "~=", 0.75, tol=0.02)],
    "leak": [Goal("act", ">=", 0.30), Goal("dose_broad", "<=", 0.20)],
    "offshoot": [Goal("act", ">=", 0.30), Goal("dose", "<=", 0.20)],
    "silent_on_concept": [Goal("dose", ">=", 0.90), Goal("act", "<=", 0.10)],
    "near_miss_ceiling": [Goal("dose", "<=", 0.60), Goal("act", "max")],
}
QUESTIONS = {
    "level_25": "Text that drives the pole to 25% of its corpus max",
    "level_50": "… to 50%",
    "level_75": "… to 75%",
    "max": "The strongest natural text in the bank's span",
    "tight_25": "25% ± 0.015, starting from the neutral centre (exercises the learned inverse)",
    "tight_50": "50% ± 0.015, from the centre",
    "tight_75": "75% ± 0.015, from the centre",
    "leak": "Fires at ≥30% with nothing related to the broad label present (broad relevance ≤ 0.2)",
    "offshoot": "Fires at ≥30% without the refined sense (refined relevance ≤ 0.2): an offshoot of the main job",
    "silent_on_concept": "Concept clearly present (relevance ≥ 0.9) but the pole stays ≤ 10%",
    "near_miss_ceiling": "The strongest text that is at most related (relevance ≤ 0.6)",
}


class TextEvaluator:
    """Batched max-over-tokens activation of one neuron (both poles), stopping at its layer."""

    def __init__(self, model, tok, rec, L, n, pole_max, opp_max, sign=1.0, batch=16, preambles=PREAMBLES):
        self.model, self.tok, self.rec = model, tok, rec
        self.preambles = preambles or [""]
        self.pre_len = [len(tok(p).input_ids) if p else 0 for p in self.preambles]
        self.L, self.n, self.batch = L, n, batch
        self.pole_max, self.opp_max, self.sign = pole_max, opp_max, sign
        self.cache = {}
        self.forward_passes = 0

    @torch.no_grad()
    def _run(self, texts, pre_len):
        enc = self.tok(texts, return_tensors="pt", padding=True).to("cuda")
        rec = self.rec
        rec.mode, rec.layers, rec.stop_at = "act", {self.L}, self.L
        try:
            self.model.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        except StopForward:
            pass
        finally:
            rec.mode, rec.stop_at = None, None
        a = rec.store.pop(self.L)[:, :, self.n].float() * self.sign
        rec.clear()
        mask = enc.attention_mask.bool()
        mask[:, :pre_len] = False  # read the probe text's tokens only
        pos = torch.where(mask, a, torch.full_like(a, -1e9)).max(1)
        neg = torch.where(mask, -a, torch.full_like(a, -1e9)).max(1)
        self.forward_passes += 1
        out = []
        for i in range(len(texts)):
            ids = enc.input_ids[i]
            out.append({"act": pos.values[i].item() / self.pole_max, "opp": neg.values[i].item() / self.opp_max,
                        "peak_token": self.tok.decode(ids[pos.indices[i]]),
                        "peak_char": len(self.tok.decode(ids[pre_len: pos.indices[i] + 1]))})
        return out

    def __call__(self, texts):
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for s in range(0, len(todo), self.batch):
            chunk = todo[s:s + self.batch]
            per_pre = [self._run([p + t for t in chunk], k) for p, k in zip(self.preambles, self.pre_len)]
            for i, t in enumerate(chunk):
                rs = [r[i] for r in per_pre]
                self.cache[t] = {"act": float(np.mean([r["act"] for r in rs])), "opp": float(np.mean([r["opp"] for r in rs])),
                                 "act_spread": float(np.ptp([r["act"] for r in rs])),
                                 "peak_token": rs[0]["peak_token"], "peak_char": rs[0]["peak_char"]}
        return [self.cache[t] for t in texts]


class SentenceManifold:
    """Maps (u, v) per slot to bank sentences, and passages to texts and doses."""

    def __init__(self, bank, solo, key="rel_pos"):
        self.s, self.key = bank["sentences"], key
        rel = np.array([x.get(key, 0.0) for x in self.s])
        # concept coordinate: relevance rank (ties broken by solo activation, so u is monotone-ish in both)
        act = np.array([r["act"] for r in solo])
        order = np.lexsort((act, rel))
        self.q = np.empty(len(rel))
        self.q[order] = np.linspace(0, 1, len(rel))
        self.rel, self.rel_neg, self.solo_act = rel, np.array([x["rel_neg"] for x in self.s]), act
        self.rel_h1 = np.array([x["rel_pos"] for x in self.s])  # broad label: the morpheme in any sense

    def pick(self, u, v):
        band = np.where(np.abs(self.q - u) <= BAND)[0]
        if len(band) == 0:
            band = np.array([int(np.argmin(np.abs(self.q - u)))])
        band = band[np.argsort(self.solo_act[band])]
        return int(band[int(round(v * (len(band) - 1)))])

    def passage(self, x):
        idx = [self.pick(x[2 * k], x[2 * k + 1]) for k in range(SLOTS)]
        return idx, " ".join(self.s[i]["text"] for i in idx)


def build_problem(evaluator: TextEvaluator, manifold: SentenceManifold):
    vars_ = []
    for k in range(SLOTS):
        vars_ += [Var(f"s{k + 1}_concept", 0.0, 1.0), Var(f"s{k + 1}_strength", 0.0, 1.0)]
    log = []

    def evaluate_batch(X):
        passages = [manifold.passage(x) for x in X]
        res = evaluator([t for _, t in passages])
        Y = []
        for (idx, text), r in zip(passages, res):
            rel = manifold.rel[idx]
            y = [r["act"], r["opp"], rel.max(), rel.mean(), manifold.rel_neg[idx].max(), manifold.rel_h1[idx].max()]
            log.append({"idx": idx, "text": text, "y": y, "peak_token": r["peak_token"], "peak_char": r["peak_char"]})
            Y.append(y)
        return np.array(Y)

    p = InverseProblem("text-inverse", vars_, ["act", "opp", "dose", "dose_mean", "dose_opp", "dose_broad"], evaluate_batch)
    p.log, p.manifold = log, manifold
    p.scales = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    return p


def describe(p: InverseProblem, x, y):
    idx, text = p.manifold.passage(x)
    r = next((e for e in reversed(p.log) if e["text"] == text), None)
    return {"x": [round(float(v), 4) for v in x], "y": dict(zip(p.outputs, [round(float(v), 4) for v in y])),
            "text": text, "sentences": [{"i": i, **{k: p.manifold.s[i][k] for k in ("text", "kind", "branch")},
                                         "rel": p.manifold.s[i].get(p.manifold.key, 0.0),
                                         "solo_act": round(float(p.manifold.solo_act[i]), 4)} for i in idx],
            "peak_token": r and r["peak_token"], "peak_char": r and r["peak_char"]}


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


# --------------------------------------------------------------------------- circuit-level inverse

class CircuitEvaluator:
    """Max-over-text-tokens activation of several neurons at once (each as a fraction of its
    corpus pole max), after the same neutral preambles. nodes: [(name, L, n, sign, pole_max)]."""

    def __init__(self, model, tok, rec, nodes, batch=16, preambles=PREAMBLES):
        self.model, self.tok, self.rec, self.nodes, self.batch = model, tok, rec, nodes, batch
        self.preambles = preambles
        self.pre_len = [len(tok(p).input_ids) for p in preambles]
        self.layers = sorted({L for _, L, *_ in nodes})
        self.cache, self.forward_passes = {}, 0

    @torch.no_grad()
    def _run(self, texts, pre_len):
        enc = self.tok(texts, return_tensors="pt", padding=True).to("cuda")
        rec = self.rec
        rec.mode, rec.layers, rec.stop_at = "act", set(self.layers), self.layers[-1]
        try:
            self.model.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        except StopForward:
            pass
        finally:
            rec.mode, rec.stop_at = None, None
        store = {L: rec.store.pop(L) for L in self.layers}
        rec.clear()
        mask = enc.attention_mask.bool()
        mask[:, :pre_len] = False
        self.forward_passes += 1
        out = [dict() for _ in texts]
        for name, L, n, sign, pmax in self.nodes:
            a = store[L][:, :, n].float() * sign
            v = torch.where(mask, a, torch.full_like(a, -1e9)).max(1).values / pmax
            for i in range(len(texts)):
                out[i][name] = v[i].item()
        return out

    def __call__(self, texts):
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for s in range(0, len(todo), self.batch):
            chunk = todo[s:s + self.batch]
            per_pre = [self._run([p + t for t in chunk], k) for p, k in zip(self.preambles, self.pre_len)]
            for i, t in enumerate(chunk):
                self.cache[t] = {name: float(np.mean([r[i][name] for r in per_pre])) for name, *_ in self.nodes}
        return [self.cache[t] for t in texts]


def build_circuit_problem(evaluator: CircuitEvaluator, manifold: SentenceManifold):
    names = [name for name, *_ in evaluator.nodes]
    vars_ = []
    for k in range(SLOTS):
        vars_ += [Var(f"s{k + 1}_concept", 0.0, 1.0), Var(f"s{k + 1}_strength", 0.0, 1.0)]
    log = []

    def evaluate_batch(X):
        passages = [manifold.passage(x) for x in X]
        res = evaluator([t for _, t in passages])
        Y = []
        for (idx, text), r in zip(passages, res):
            y = [r[nm] for nm in names] + [manifold.rel[idx].max(), manifold.rel_h1[idx].max()]
            log.append({"idx": idx, "text": text, "y": y})
            Y.append(y)
        return np.array(Y)

    p = InverseProblem("circuit-inverse", vars_, names + ["dose", "dose_broad"], evaluate_batch)
    p.log, p.manifold = log, manifold
    p.scales = np.full(len(p.outputs), 0.1)
    return p


def describe_circuit(p, x, y):
    idx, text = p.manifold.passage(x)
    return {"x": [round(float(v), 4) for v in x], "y": dict(zip(p.outputs, [round(float(v), 4) for v in y])),
            "text": text, "sentences": [{"i": i, **{k: p.manifold.s[i][k] for k in ("text", "kind", "branch")},
                                         "rel": p.manifold.s[i].get(p.manifold.key, 0.0)} for i in idx]}
