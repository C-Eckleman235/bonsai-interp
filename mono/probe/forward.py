"""Stage 1 (forward): corpus evidence for a pole, plus the request for an LLM-built bank.

Writes
  forward.json        top examples for both poles + regime stats
  bank_request.json   what the bank author must produce (label, sentences, ratings),
                      including random corpus sentences that must be rated too
"""
import re

import numpy as np

from mono import explore
from mono.probe.workspace import Workspace

SPLIT = re.compile(r"(?<=[.!?。！？])\s+|\n+")


def corpus_sentences(k=40, seed=0, exclude_windows=()):
    rng = np.random.default_rng(seed)
    toks, tok = explore.corpus(), explore.tokenizer()
    out, seen = [], set()
    while len(out) < k:
        w = int(rng.integers(len(toks)))
        if w in exclude_windows:
            continue
        sents = [s.strip() for s in SPLIT.split(tok.decode(toks[w])) if 40 <= len(s.strip()) <= 220]
        if not sents:
            continue
        s = sents[int(rng.integers(len(sents)))]
        if s not in seen and "<|endoftext|>" not in s:
            seen.add(s)
            out.append({"text": s, "source": f"corpus window {w}"})
    return out


def top_sentence(text):
    """The sentence containing the ⟦peak⟧ token, with the brackets removed."""
    i = text.find("⟦")
    left = max(text.rfind(c, 0, i) for c in ".!?\n⏎。") + 1
    right_c = [j for j in (text.find(c, i) for c in ".!?⏎。") if j != -1]
    right = min(right_c) + 1 if right_c else len(text)
    return text[left:right].replace("⟦", "").replace("⟧", "").replace("⏎", " ").strip()


def run(ws: Workspace, k_top=16, k_corpus=40):
    s = explore.scan(ws.model)
    poles = {}
    for name, idx in (("+", 0), ("-", 1)):
        poles[name] = [{"act": a, "regime": r, "window": w, "pos": p, "text": t}
                       for a, r, w, p, t in explore.examples(ws.model, ws.L, ws.n, idx, k_top, 64, 12)]
    fwd = {"neuron": ws.id, "L": ws.L, "n": ws.n, "pole": ws.pole, "model": ws.model,
           "max": {"+": float(s["top_val"][ws.L, 0, 0, ws.n]), "-": float(s["top_val"][ws.L, 1, 0, ws.n])},
           "regimes": explore.regime_stats(ws.model, ws.L, ws.n), "examples": poles}
    ws.save("forward.json", fwd)
    used = {e["window"] for p in poles.values() for e in p}
    ws.save("bank_request.json", {
        "neuron": ws.id,
        "instructions": (
            "Write bank.json for this neuron. 1) 'label': one-line description of the + pole and of the - pole, "
            "from forward.json examples. 2) 'branches': named sub-senses seen in the examples. 3) 'sentences': "
            "~60 natural, self-contained sentences, each {text, rel_pos, rel_neg, branch, kind}: rel_pos / rel_neg "
            "= 0..1 semantic relevance to the + / - pole label (rate the meaning, not the spelling). kinds: "
            "'on' (the concept, across senses/languages), 'near' (related concepts without the key word), "
            "'spelling' (look-alike spellings with unrelated meaning), 'opposite' (the - pole concept), "
            "'unrelated'. 4) Rate every 'corpus' sentence below the same way and include it with kind 'corpus'. "
            "5) Include the 'top' sentences (from the pole's own top examples) with kind 'top'."),
        "top_sentences": {p: [top_sentence(e["text"]) for e in poles[p][:8]] for p in poles},
        "corpus": corpus_sentences(k_corpus, exclude_windows=used),
    })
    return fwd


if __name__ == "__main__":
    import sys
    L, n = map(int, sys.argv[1].split("/"))
    ws = Workspace(L, n, sys.argv[2] if len(sys.argv) > 2 else "+")
    run(ws)
    print("wrote", ws.path("forward.json"), "and", ws.path("bank_request.json"))
