"""Feeders by position: which upstream poles, at which distance back, drive a pole's strong activations.

The survey's `play step` attribution (bonsai_interp/inverse/neuron.py) gives each upstream neuron one number, summed over
positions. That hides attention. Here every upstream neuron gets one gain per position, so the attribution of a target
activation at position t splits into terms (layer l, neuron m, position s ≤ t):

    contribution(l, m, s) = a_lms · ∂y/∂a_lms

The offset t − s says how the signal reached the target:
    0      same token: the residual stream at the target's own position (no attention needed)
    1      previous token (a previous-token head, a bigram)
    2-4    local window (phrase-level attention)
    5-16   clause / sentence
    17+    far back (induction, copying, topic)
So a feeder is (pole, offset band), and a composed label can say "A here and B a token before" or "A anywhere earlier and
B here". The pole of a feeder is the sign of its activation at that position (+ when a > 0). Contributions are signed:
positive terms push the target pole up.

GPU; every forward and backward goes through context_tier's background gpu_turn tick.
"""
import numpy as np

BANDS = [(0, 0), (1, 1), (2, 4), (5, 16), (17, 10_000)]
BAND_NAMES = ["same token", "previous token", "2-4 back", "5-16 back", "17+ back"]
N = 12288


def band_of(off):
    for i, (a, b) in enumerate(BANDS):
        if a <= off <= b:
            return i
    return len(BANDS) - 1


def site_contributions(model, rec, ids, L, n, pos, sign, max_ctx=256, keep=4096):
    """Signed contributions of every (layer < L, neuron, position) to the target pole at `pos`, normalised by the target
    activation. Returns (y, rows) with rows = [(contribution, layer, neuron, offset, feeder sign)] for the `keep` largest
    |contribution| terms, and the total positive mass (for shares)."""
    import torch
    from bonsai_interp.env import DEVICE
    from bonsai_interp.model import StopForward
    start = max(0, pos + 1 - max_ctx)
    x = torch.as_tensor(ids[start:pos + 1], device=DEVICE).long()[None]
    T, p = x.shape[1], pos - start
    gains = {l: torch.ones(1, T, N, device=DEVICE, requires_grad=True) for l in range(L)}
    acts = {}

    def hook(l):
        def f(a):
            acts[l] = a.detach()[0].float()
            return a * gains[l].to(a.dtype)
        return f
    rec.interventions = {l: hook(l) for l in range(L)}
    rec.mode, rec.layers, rec.stop_at, rec.keep_grad = "act", {L}, L, True
    try:
        with torch.enable_grad():
            try:
                model.model(x)
            except StopForward:
                pass
            y = rec.store.pop(L)[0, p, n].float() * sign
            y.backward()
    finally:
        rec.interventions, rec.stop_at, rec.mode, rec.keep_grad = {}, None, None, False
        rec.clear()
    yv = float(y.detach())
    if abs(yv) < 1e-6:
        return yv, [], 0.0
    C = torch.stack([gains[l].grad[0] for l in range(L)]) / yv  # (L, T, N): a · ∂y/∂a per term, as a share of y
    S = torch.stack([acts[l] for l in range(L)])  # activations, for the feeder's pole
    flat = C.reshape(-1)
    pos_mass = float(flat.clamp_min(0).sum())
    k = min(keep, flat.numel())
    v, idx = flat.abs().topk(k)
    idx = idx.cpu().numpy()
    lay, rem = np.divmod(idx, T * N)
    tt, mm = np.divmod(rem, N)
    vals = flat[torch.as_tensor(idx, device=flat.device)].cpu().numpy()
    sg = S.reshape(-1)[torch.as_tensor(idx, device=flat.device)].cpu().numpy()
    rows = [(float(vals[i]), int(lay[i]), int(mm[i]), int(p - tt[i]), "+" if sg[i] > 0 else "-") for i in range(k)]
    del gains, C, S
    return yv, rows, pos_mass


def pole_feeders(rows_per_site, pos_masses):
    """Aggregate several sites (examples) of one pole: {(layer, neuron, sign, band): mean share of the target}."""
    agg = {}
    for rows in rows_per_site:
        for c, l, m, off, s in rows:
            key = (l, m, s, band_of(off))
            agg[key] = agg.get(key, 0.0) + c / len(rows_per_site)
    return agg


def ablate(model, rec, ids, L, n, pos, sign, terms, max_ctx=256):
    """Target activation (sign-adjusted) with the given (layer, neuron, offset) terms zeroed at their positions."""
    import torch
    from bonsai_interp.env import DEVICE
    from bonsai_interp.model import StopForward
    start = max(0, pos + 1 - max_ctx)
    x = torch.as_tensor(ids[start:pos + 1], device=DEVICE).long()[None]
    T, p = x.shape[1], pos - start
    by_layer = {}
    for l, m, off in terms:
        by_layer.setdefault(l, []).append((p - off, m))

    def hook(l):
        def f(a):
            g = torch.ones_like(a)
            for t, m in by_layer[l]:
                g[0, t, m] = 0
            return a * g
        return f
    rec.interventions = {l: hook(l) for l in by_layer}
    rec.mode, rec.layers, rec.stop_at = "act", {L}, L
    try:
        with torch.no_grad():
            try:
                model.model(x)
            except StopForward:
                pass
            y = float(rec.store.pop(L)[0, p, n]) * sign
    finally:
        rec.interventions, rec.stop_at, rec.mode = {}, None, None
        rec.clear()
    return y


def occlude(model, rec, ids, L, n, pos, sign, offsets=32, filler=None, max_ctx=256):
    """Target activation (sign-adjusted) when the token `k` positions back is replaced by a filler token, for k = 0 ..
    offsets - 1, in one batch: [y_k]. The drop at each offset is how much the pole needs that token, wherever attention
    carried it from."""
    import torch
    from bonsai_interp.env import DEVICE
    from bonsai_interp.model import StopForward
    start = max(0, pos + 1 - max_ctx)
    x = torch.as_tensor(ids[start:pos + 1], device=DEVICE).long()[None]
    T, p = x.shape[1], pos - start
    K = min(offsets, p)
    X = x.repeat(K, 1)
    for k in range(K):
        X[k, p - k] = filler
    rec.mode, rec.layers, rec.stop_at = "act", {L}, L
    try:
        with torch.no_grad():
            try:
                model.model(X)
            except StopForward:
                pass
            ys = (rec.store.pop(L)[:, p, n].float() * sign).cpu().numpy()
    finally:
        rec.stop_at, rec.mode = None, None
        rec.clear()
    return ys
