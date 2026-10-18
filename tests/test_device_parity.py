"""Device parity: known neurons must read the same on CUDA, Apple Silicon (MPS) and CPU.

The reference values in device_parity_reference.json were recorded on the reference machine
(RTX 5080, CUDA, bfloat16). Run this on yours to check that the toolchain computes the same thing:

    uv run python tests/test_device_parity.py                       # the device play doctor picks
    BONSAI_DEVICE=cpu uv run python tests/test_device_parity.py     # force a device
    uv run python tests/test_device_parity.py --record              # CUDA only: rewrite the reference

pytest also collects it. It loads the model once (about 4 GB) and needs the weights and data/scan_bonsai1.npz.
Checks: four neuron readings (layers 9-23) as fractions of their corpus max, the full-model next-token
distribution, and the backward pass through the 1-bit layers (attribution, as in inverse/neuron.py).
"""
import functools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REF = Path(__file__).with_name("device_parity_reference.json")

READINGS = [  # (layer, neuron, pole, sentence)
    (9, 3688, "+", "Waiting for the verdict was the most nerve-racking hour of my life."),
    (12, 5397, "-", "Call me at 555-0142 after 6pm; the door code is 7F3A, then 1984."),
    (20, 12208, "-", "Annual Report\nThe committee met twice this year to review the budget."),
    (23, 6305, "+", "After the storm passed, we walked down to the river to look at the damage."),
]
NEXT_TOKEN = "The capital of France is Paris, and the capital of Germany is"
ATTRIBUTION = (9, 3688, READINGS[0][3])  # L9/N3688+ at its peak token in the nerve sentence

# Tolerances: bfloat16 matmuls accumulate in a different order on each device, so readings move by
# about 1% of a neuron's max; anything structural (wrong weights, wrong rotary, wrong mask) moves far more.
ACT_TOL = 0.03      # |act - ref| in fractions of the corpus max
LOGPROB_TOL = 0.15  # next-token log-probabilities
ATTR_REL_TOL = 0.05  # clean activation at the attribution site
ATTR_CORR = 0.95    # correlation of the reference's top attributions


@functools.cache
def _model():
    from bonsai_interp import explore
    from bonsai_interp.device import limit_memory
    from bonsai_interp.model import load
    limit_memory(0.8)
    model, tok, rec = load("bonsai1")
    tok.padding_side = "right"
    return model, tok, rec, explore.scan("bonsai1")


def measure():
    """Everything the test compares, on this machine's device."""
    import numpy as np
    import torch
    from bonsai_interp.device import DEVICE, describe
    from bonsai_interp.inverse.neuron import NeuronSite
    from bonsai_interp.model import StopForward
    from bonsai_interp.probe import textinv
    model, tok, rec, s = _model()
    out = {"device": describe(), "readings": [], "next_token": {}, "attribution": {}}
    for L, n, pole, text in READINGS:
        p = 0 if pole == "+" else 1
        ev = textinv.TextEvaluator(model, tok, rec, L, n, float(s["top_val"][L, p, 0, n]), float(s["top_val"][L, 1 - p, 0, n]),
                                   1.0 if pole == "+" else -1.0)
        r = ev([text])[0]
        out["readings"].append({"neuron": f"L{L}/N{n}{pole}", "text": text, "act": round(r["act"], 4),
                                "opp": round(r["opp"], 4), "peak_token": r["peak_token"]})
    with torch.no_grad():
        ids = tok(NEXT_TOKEN, return_tensors="pt").input_ids.to(DEVICE)
        lp = torch.log_softmax(model(ids).logits[0, -1].float(), -1)
        top = lp.topk(5)
    out["next_token"] = {"prompt": NEXT_TOKEN, "top": [[tok.decode(int(i)), round(float(v), 4)] for v, i in zip(top.values, top.indices)]}
    L, n, text = ATTRIBUTION
    ids = tok(text).input_ids
    rec.mode, rec.layers, rec.stop_at = "act", {L}, L
    try:
        with torch.no_grad():
            model.model(torch.tensor([ids], device=DEVICE))
    except StopForward:
        pass
    finally:
        rec.mode, rec.stop_at = None, None
    pos = int(rec.store.pop(L)[0, :, n].float().argmax())
    rec.clear()
    attr, clean = NeuronSite(model, rec, ids, L, n, pos).attribution()
    flat = np.concatenate([attr[l] for l in range(L)])
    top = np.argsort(-np.abs(flat))[:50]
    N = model.config.intermediate_size
    out["attribution"] = {"site": f"L{L}/N{n} at token {pos} ({tok.decode(ids[pos])!r})", "pos": pos, "clean": round(clean, 4),
                          "top": [[f"L{i // N}/N{i % N}", round(float(flat[i]), 5)] for i in top]}
    return out


def compare(got, ref):
    """List of human-readable failures (empty when the devices agree)."""
    bad = []
    for g, r in zip(got["readings"], ref["readings"]):
        if abs(g["act"] - r["act"]) > ACT_TOL or abs(g["opp"] - r["opp"]) > ACT_TOL:
            bad.append(f"{r['neuron']}: act {g['act']:.3f} / opp {g['opp']:.3f}, reference {r['act']:.3f} / {r['opp']:.3f}")
        if g["peak_token"] != r["peak_token"] and abs(g["act"] - r["act"]) > ACT_TOL / 3:
            bad.append(f"{r['neuron']}: peak on {g['peak_token']!r}, reference {r['peak_token']!r}")
    gt, rt = got["next_token"]["top"], ref["next_token"]["top"]
    if gt[0][0] != rt[0][0]:
        bad.append(f"next token {gt[0][0]!r}, reference {rt[0][0]!r}")
    rlp = dict(rt)
    for t, v in gt[:3]:
        if t in rlp and abs(v - rlp[t]) > LOGPROB_TOL:
            bad.append(f"log p({t!r}) {v:.3f}, reference {rlp[t]:.3f}")
    ga, ra = got["attribution"], ref["attribution"]
    if ga["pos"] != ra["pos"]:
        bad.append(f"attribution site at token {ga['pos']}, reference {ra['pos']}")
    elif abs(ga["clean"] - ra["clean"]) > ATTR_REL_TOL * abs(ra["clean"]):
        bad.append(f"attribution clean activation {ga['clean']:.3f}, reference {ra['clean']:.3f}")
    gmap = dict(ga["top"])
    import numpy as np
    rv = np.array([v for _, v in ra["top"]])
    gv = np.array([gmap.get(u, np.nan) for u, _ in ra["top"]])
    if np.isnan(gv).sum() > 10:
        bad.append(f"attribution: {int(np.isnan(gv).sum())} of the reference's top 50 upstream units are missing from this device's top 50")
    else:
        # units missing from this device's top 50 are small here; compare on the shared ones
        ok = ~np.isnan(gv)
        c = float(np.corrcoef(rv[ok], gv[ok])[0, 1])
        if c < ATTR_CORR:
            bad.append(f"attribution correlation with the reference {c:.3f} < {ATTR_CORR}")
    return bad


def test_device_parity():
    from bonsai_interp.device import describe
    got = measure()
    bad = compare(got, json.loads(REF.read_text(encoding="utf-8")))
    assert not bad, f"{describe()} disagrees with the CUDA reference:\n  " + "\n  ".join(bad)


def main():
    from bonsai_interp.device import DEVICE
    got = measure()
    if "--record" in sys.argv:
        if DEVICE.type != "cuda":
            sys.exit("record the reference on CUDA (the reference machine), not " + DEVICE.type)
        REF.write_text(json.dumps(got, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {REF}")
        return
    ref = json.loads(REF.read_text(encoding="utf-8"))
    print(f"device: {got['device']}   reference: {ref['device']}")
    for g, r in zip(got["readings"], ref["readings"]):
        print(f"  {r['neuron']:<12} act {g['act']:.3f} (ref {r['act']:.3f})  opp {g['opp']:.3f} (ref {r['opp']:.3f})  peak {g['peak_token']!r}")
    print(f"  next token   {got['next_token']['top'][0]} (ref {ref['next_token']['top'][0]})")
    print(f"  attribution  clean {got['attribution']['clean']:.3f} (ref {ref['attribution']['clean']:.3f}), "
          f"top unit {got['attribution']['top'][0]} (ref {ref['attribution']['top'][0]})")
    bad = compare(got, ref)
    print("PARITY OK" if not bad else "PARITY FAILED:\n  " + "\n  ".join(bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
