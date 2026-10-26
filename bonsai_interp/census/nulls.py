"""Random-direction nulls for the weight statistics, on the CPU from the GGUF (no GPU, no model load).

    python -m bonsai_interp.census.nulls        # writes data/census/bonsai1.nulls.npz

Two nulls per layer:
- isotropic: Gaussian unit directions in the residual space;
- sign-random: what a neuron with random weights would write in this 1-bit model: the layer's real per-(row, 128-group)
  scales with random ±1 signs. In Bonsai every down_proj column is ±scale[row, group], so the output norm is a
  property of the 128-neuron group, and this null keeps that structure.
For each: the direct-effect spike of both signs (as predict.direct_effect), freq_r and null_share (as weights.Unembed).
"""
import numpy as np
import torch

from bonsai_interp.census.run import NAME, OUT, _counts
from bonsai_interp.census.weights import GAP, Unembed

LAYERS = tuple(range(36))
R = 1024


class _Shim:
    """Just enough of the model for weights.Unembed: lm_head.weight and model.norm.weight, float32 on the CPU."""

    def __init__(self, by):
        from bonsai_interp import gguf_weights as gw
        self.lm_head = torch.nn.Module()
        self.lm_head.weight = gw.dense(by["lm_head.weight"]).float()
        self.model = torch.nn.Module()
        self.model.norm = torch.nn.Module()
        self.model.norm.weight = gw.dense(by["model.norm.weight"]).float()


def spike(WU, g, cols, topk=20, chunk=8192):
    """(2, n) spike of each column (d, n), as predict.direct_effect: the topk-th token's z-score over the vocabulary."""
    cols = cols * g[:, None]
    n = cols.shape[1]
    s1, s2 = torch.zeros(n), torch.zeros(n)
    sm, sq = torch.zeros(n), torch.zeros(n)
    top = [torch.full((topk, n), -float("inf")), torch.full((topk, n), -float("inf"))]
    V = WU.shape[0]
    for r0 in range(0, V, chunk):
        E = WU[r0:r0 + chunk] @ cols
        sm += E.sum(0)
        sq += (E * E).sum(0)
        for p, s in ((0, 1.0), (1, -1.0)):
            top[p] = torch.cat([top[p], s * E]).topk(topk, dim=0).values
    mu = sm / V
    sd = (sq / V - mu * mu).clamp_min(1e-18).sqrt() * np.sqrt(V / (V - 1))
    return torch.stack([(top[0][-1] - mu) / sd, (top[1][-1] + mu) / sd]).numpy()


def main():
    from bonsai_interp import gguf_weights as gw
    from bonsai_interp.model import MODELS
    torch.set_num_threads(8)
    by = gw.by_name(MODELS[NAME])
    m = _Shim(by)
    U = Unembed(m, _counts(m.lm_head.weight.shape[0]), chunk=8192)
    WU, g = m.lm_head.weight, m.model.norm.weight
    d = WU.shape[1]
    rng = np.random.default_rng(7)
    out = {"layers": np.array(LAYERS), "eig": U.eig}
    iso = torch.from_numpy(rng.standard_normal((d, R)).astype(np.float32))
    iso /= iso.norm(dim=0, keepdim=True)
    out["iso_spike"] = spike(WU, g, iso)
    out.update({f"iso_{k}": v.numpy() for k, v in U.stats(iso).items()})
    for L in LAYERS:
        _, scale, out_f, in_f = gw._blocks(by[f"model.layers.{L}.mlp.down_proj.weight"])  # scale (d, N/128)
        grp = rng.integers(0, scale.shape[1], R)
        cols = torch.from_numpy(scale[:, grp].astype(np.float32) * rng.choice([-1.0, 1.0], (d, R)).astype(np.float32))
        out[f"sgn{L}_spike"] = spike(WU, g, cols)
        out.update({f"sgn{L}_{k}": v.numpy() for k, v in U.stats(cols).items()})
        print(f"L{L}: sign-random spike median {np.median(out[f'sgn{L}_spike']):.2f} q99 {np.quantile(out[f'sgn{L}_spike'], .99):.2f}; "
              f"|freq_r| q99.9 {np.quantile(np.abs(out[f'sgn{L}_freq_r']), .999):.3f}; null_share q99.9 {np.quantile(out[f'sgn{L}_null_share'], .999):.4f}",
              flush=True)
    print(f"isotropic: spike median {np.median(out['iso_spike']):.2f} q99 {np.quantile(out['iso_spike'], .99):.2f}; "
          f"|freq_r| q99.9 {np.quantile(np.abs(out['iso_freq_r']), .999):.3f}; null_share median {np.median(out['iso_null_share']):.4f}")
    out["k_gap"] = U.k_gap
    np.savez_compressed(OUT / f"{NAME}.nulls.npz", **out)
    for L in LAYERS:  # add the gap null share to weights files written before weights.py computed it
        p = OUT / f"{NAME}_L{L}.weights.npz"
        w = dict(np.load(p))
        if "null_gap_share" not in w:
            Wd = gw.dense(by[f"model.layers.{L}.mlp.down_proj.weight"]).float()
            w["null_gap_share"] = ((U.Qg.T @ Wd).pow(2).sum(0) / Wd.pow(2).sum(0)).numpy().astype(np.float32)
            np.savez_compressed(p, **w)
    print(f"null space: {U.k_gap} directions below {GAP} x the median singular value")


if __name__ == "__main__":
    main()
