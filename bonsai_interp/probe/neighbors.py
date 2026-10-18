"""Stage 4 (fan out): the probed pole's causal neighbourhood, per branch.

Upstream  attribution of the pole's activation to every earlier MLP neuron on examples of
          each branch, then a causal check: zero each top candidate alone and measure how
          much of the pole's activation is lost.
Context   for a sense-flip pair (concept alone vs. after a context that silences it), the
          upstream units that push the pole down only in the flipped passage, checked by
          zeroing them in the flipped passage and measuring how much activation returns.
Downstream zero the probed neuron and measure how much each later neuron (next LAYERS_DOWN
          layers) changes at and just after the peak token, relative to its corpus max.

Writes neighbors.json. Every passage is preceded by a neutral preamble, as in the probe.
"""
import json

import numpy as np
import torch

from bonsai_interp import explore
from bonsai_interp.device import DEVICE
from bonsai_interp.inverse.neuron import NeuronSite
from bonsai_interp.model import StopForward
from bonsai_interp.probe.run import setup
from bonsai_interp.probe.textinv import PREAMBLES
from bonsai_interp.probe.workspace import Workspace

LAYERS_DOWN = 12
N_CAND = 40


def ids_and_peak(model, tok, rec, text, L, n, sign, find=None):
    """Token ids of preamble+text and the position of the pole's peak inside the text
    (or of the first token whose decoded string contains `find`)."""
    pre = tok(PREAMBLES[0]).input_ids
    ids = tok(PREAMBLES[0] + text).input_ids
    with torch.no_grad():
        rec.mode, rec.layers, rec.stop_at = "act", {L}, L
        try:
            model.model(torch.tensor([ids], device=DEVICE))
        except StopForward:
            pass
        finally:
            rec.mode, rec.stop_at = None, None
        a = rec.store.pop(L)[0, :, n].float() * sign
        rec.clear()
    if find:
        cand = [i for i in range(len(pre), len(ids)) if find in tok.decode([ids[i]])]
        pos = cand[0]
    else:
        pos = len(pre) + int(a[len(pre):].argmax())
    return np.array(ids), pos, float(a[pos])


def single_ablation(site, units, chunk=20):
    """Fraction of the clean activation kept when each unit alone is zeroed."""
    N = site.model.config.intermediate_size
    out = []
    with torch.no_grad():
        clean = site.run({}).item()
        for s in range(0, len(units), chunk):
            part = units[s:s + chunk]
            B = len(part)
            masks = {}
            for b, (l, m) in enumerate(part):
                masks.setdefault(l, torch.ones(B, N, device=DEVICE))[b, m] = 0.0
            out += (site.run(masks, batch=B) / clean).tolist() if clean else [1.0] * B
    return out


def upstream(model, rec, examples, L, n, sign):
    """examples: list of (branch, ids, pos). Returns ranked candidate list with per-branch drops."""
    attrs, sites = [], []
    for branch, ids, pos in examples:
        site = NeuronSite(model, rec, ids, L, n, pos)
        attr, y = site.attribution()
        flat = np.concatenate([attr[l] for l in range(L)]) * sign / max(abs(y), 1e-6)
        attrs.append(flat)
        sites.append((branch, site))
    mean = np.mean(attrs, 0)
    N = model.config.intermediate_size
    cand = [(int(i // N), int(i % N)) for i in np.argsort(-mean)[:N_CAND]]
    drops = {u: {} for u in cand}
    for branch, site in sites:
        kept = single_ablation(site, cand)
        for u, k in zip(cand, kept):
            drops[u].setdefault(branch, []).append(1 - k)
    rows = []
    for u in cand:
        per = {b: float(np.mean(v)) for b, v in drops[u].items()}
        rows.append({"unit": f"L{u[0]}/N{u[1]}", "L": u[0], "n": u[1], "attr": float(mean[u[0] * N + u[1]]),
                     "drop_mean": float(np.mean([x for v in drops[u].values() for x in v])), "drop_by_branch": per})
    rows.sort(key=lambda r: -r["drop_mean"])
    return rows


def disambiguators(model, rec, flip_ids, flip_pos, alone_ids, alone_pos, L, n, sign, k=80):
    N = model.config.intermediate_size
    s_f = NeuronSite(model, rec, flip_ids, L, n, flip_pos)
    s_a = NeuronSite(model, rec, alone_ids, L, n, alone_pos)
    a_f, y_f = s_f.attribution()
    a_a, y_a = s_a.attribution()
    f = np.concatenate([a_f[l] for l in range(L)]) * sign
    a = np.concatenate([a_a[l] for l in range(L)]) * sign
    # most negative in the flipped passage, relative to the same unit's role when the concept is alone
    score = f - np.minimum(a, 0)
    cand = [(int(i // N), int(i % N)) for i in np.argsort(score)[:k]]
    kept = single_ablation(s_f, cand)
    # restore: how far zeroing the unit moves the flipped activation back toward the pole,
    # as a fraction of the gap between the flipped and the alone activation
    gap = (y_a - y_f) * sign
    rows = [{"unit": f"L{l}/N{m}", "L": l, "n": m, "attr_flipped": float(f[l * N + m]), "attr_alone": float(a[l * N + m]),
             "restore": float((kp * y_f - y_f) * sign / gap)} for (l, m), kp in zip(cand, kept)]
    rows.sort(key=lambda r: -r["restore"])
    # group test: zero the top-k restorers together
    with torch.no_grad():
        group = {}
        for k in (5, 10, 20, 40):
            units = [(r["L"], r["n"]) for r in rows[:k]]
            masks = {}
            for l, m in units:
                masks.setdefault(l, torch.ones(1, N, device=DEVICE))[0, m] = 0.0
            new = s_f.run(masks).item()
            group[k] = float((new - y_f) * sign / gap)
    return {"act_alone": y_a * sign, "act_flipped": y_f * sign, "units": rows, "group_restore": group}


def downstream(model, rec, examples, L, n, scan, window=3):
    """Mean |change| of each later neuron (relative to its corpus max) when the probed neuron is zeroed."""
    layers = list(range(L + 1, min(L + 1 + LAYERS_DOWN, model.config.num_hidden_layers)))
    acc = {l: [] for l in layers}
    for branch, ids, pos in examples:
        x = torch.tensor(np.asarray(ids[: pos + window])[None], device=DEVICE)
        res = []
        for ablate in (False, True):
            rec.mode, rec.layers, rec.stop_at = "act", set(layers), layers[-1]
            rec.interventions = {L: (lambda a: _zero(a, n))} if ablate else {}
            with torch.no_grad():
                try:
                    model.model(x)
                except StopForward:
                    pass
            rec.mode, rec.stop_at, rec.interventions = None, None, {}
            res.append({l: rec.store.pop(l)[0, pos: pos + window].float() for l in layers})
            rec.clear()
        for l in layers:
            d = (res[0][l] - res[1][l])  # clean - ablated: what the probed neuron contributes
            # signed change at the position of largest |change|, normalised by the pole it moves toward
            idx = d.abs().argmax(0)
            dd = d.gather(0, idx[None])[0].cpu().numpy()
            pmax = np.where(dd >= 0, scan["top_val"][l, 0, 0], -scan["top_val"][l, 1, 0]).astype(np.float64)
            acc[l].append(dd / np.maximum(np.abs(pmax), 1e-6))
    rows = []
    for l in layers:
        m = np.mean(acc[l], 0)
        for j in np.argsort(-np.abs(m))[:8]:
            rows.append({"unit": f"L{l}/N{int(j)}", "L": l, "n": int(j), "effect": float(m[j]),
                         "pole": "+" if m[j] > 0 else "-"})
    rows.sort(key=lambda r: -abs(r["effect"]))
    return rows


def _zero(a, n):
    a = a.clone()
    a[..., n] = 0
    return a


def run(ws: Workspace, parts=("up", "flip", "down")):
    model, tok, rec, ev = setup(ws)
    sign = 1.0 if ws.pole == "+" else -1.0
    spec = ws.require("neighbors_request.json", "list branch example texts in neighbors_request.json")
    examples, used = [], []
    for branch, items in spec["branches"].items():
        for it in items:
            text, find = (it, None) if isinstance(it, str) else (it["text"], it.get("find"))
            ids, pos, act = ids_and_peak(model, tok, rec, text, ws.L, ws.n, sign, find)
            used.append({"branch": branch, "text": text, "act_raw": act, "token": tok.decode([ids[pos]])})
            examples.append((branch, ids, pos))
    print(f"{len(examples)} examples")
    up = upstream(model, rec, examples, ws.L, ws.n, sign)
    print("upstream top:", [(r["unit"], round(r["drop_mean"], 3)) for r in up[:10]])
    fl = spec["flip"]
    fi, fp, _ = ids_and_peak(model, tok, rec, fl["flipped"], ws.L, ws.n, sign, fl["find"])
    ai, ap, _ = ids_and_peak(model, tok, rec, fl["alone"], ws.L, ws.n, sign, fl["find"])
    dis = disambiguators(model, rec, fi, fp, ai, ap, ws.L, ws.n, sign)
    print(f"flip: alone {dis['act_alone']:.2f} -> flipped {dis['act_flipped']:.2f};",
          "top restore:", [(r["unit"], round(r["restore"], 3)) for r in dis["units"][:8]], "group:", dis["group_restore"])
    scan = explore.scan(ws.model)
    down = downstream(model, rec, examples, ws.L, ws.n, scan)
    print("downstream top:", [(r["unit"], round(r["effect"], 3)) for r in down[:10]])
    ws.save("neighbors.json", {"examples": used, "upstream": up, "disambiguators": dis, "downstream": down})


if __name__ == "__main__":
    import sys
    L, n = map(int, sys.argv[1].split("/"))
    run(Workspace(L, n, sys.argv[2] if len(sys.argv) > 2 else "+"))
