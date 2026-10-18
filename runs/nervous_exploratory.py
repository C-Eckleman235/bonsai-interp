"""Exploratory: how does the 'nerve' circuit treat the word 'nervous', and is there a dedicated 'nervous' neuron?"""
import sys
sys.path.insert(0, r"~\Documents\GitHub\monosemanticity")
import numpy as np
import torch
from mono import explore
from mono.model import load, StopForward
from mono.probe import textinv

NERVOUS = [
    "I felt nervous before the job interview.", "She was nervous about the test results.",
    "He gave a nervous laugh and looked away.", "The crowd grew nervous as the storm approached.",
    "Investors were nervous ahead of the announcement.", "My dog gets nervous during thunderstorms.",
    "Her nervousness showed in her shaking hands.", "Er war vor der Prüfung sehr nervös.",
    "Il était nerveux avant le concert.", "Estaba muy nervioso antes del partido.", "Он очень нервничал перед экзаменом.",
]
CONTROL = [s.replace("nervousness", "anxiety").replace("nervous", "anxious") for s in NERVOUS[:7]] + [
    "Er war vor der Prüfung sehr ängstlich.", "Il était anxieux avant le concert.", "Estaba muy ansioso antes del partido.",
    "Он очень волновался перед экзаменом."]
SYSTEM = ["The nervous system controls every movement.", "Diseases of the nervous system are hard to treat."]
NERVE = ["Waiting for the results was nerve-racking.", "She finally worked up the nerve to ask."]

scan = explore.scan("bonsai1")
nodes = [("nerve", 9, 3688), ("token", 3, 6493), ("concept", 8, 7756), ("suspense", 7, 9782), ("anticip", 9, 1061), ("overwhelm", 12, 3688)]
model, tok, rec = load("bonsai1")
tok.padding_side = "right"
ev = textinv.CircuitEvaluator(model, tok, rec, [(n, L, i, 1.0, float(scan["top_val"][L, 0, 0, i])) for n, L, i in nodes])
print("tokenization:", {w: tok.tokenize(w) for w in (" nervous", " nervousness", " nerve", " nervös", " nerveux", " nervioso")})
for name, group in (("nervous", NERVOUS), ("anxious (control)", CONTROL), ("nervous system", SYSTEM), ("nerve", NERVE)):
    res = ev(group)
    print(f"\n{name}: " + "  ".join(f"{k}={np.mean([r[k] for r in res]):.2f}" for k in res[0]))
    if name == "nervous":
        for s, r in zip(group, res):
            print(f"   nerve {r['nerve']:.2f} token {r['token']:.2f} concept {r['concept']:.2f} overwhelm {r['overwhelm']:.2f} | {s}")

# is there a dedicated 'nervous' neuron? per-sentence max activation, nervous vs anxious control
@torch.no_grad()
def maxacts(texts, layers):
    pre = textinv.PREAMBLES[0]; k = len(tok(pre).input_ids)
    enc = tok([pre + t for t in texts], return_tensors="pt", padding=True).to("cuda")
    rec.mode, rec.layers, rec.stop_at = "act", set(layers), max(layers)
    try:
        model.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
    except StopForward:
        pass
    rec.mode, rec.stop_at = None, None
    m = enc.attention_mask.bool().clone(); m[:, :k] = False
    out = {l: torch.where(m[..., None], rec.store.pop(l).float(), torch.tensor(-1e9, device="cuda")).max(1).values.cpu().numpy() for l in layers}
    rec.clear()
    return out
layers = list(range(1, 15))
A, B = maxacts(NERVOUS, layers), maxacts(CONTROL, layers)
cands = []
for l in layers:
    pm = np.maximum(scan["top_val"][l, 0, 0], 1e-3)
    d = (A[l].mean(0) - B[l].mean(0)) / pm
    sel = (A[l] / pm >= 0.3).mean(0)  # fraction of nervous sentences where it fires strongly
    for j in np.argsort(-d)[:5]:
        cands.append((d[j], sel[j], l, j))
cands.sort(reverse=True)
print("\nneurons most specific to 'nervous' over 'anxious' (diff as fraction of corpus max, fire rate on nervous sentences):")
for d, sel, l, j in cands[:8]:
    ex = [t[t.find("⟦"):t.find("⟧") + 1] for *_, t in explore.examples("bonsai1", l, int(j), 0, 6, 0, 0)]
    print(f"  L{l}/N{j}: diff {d:.2f}, fires on {sel:.0%} of nervous sentences | top tokens {ex}")
