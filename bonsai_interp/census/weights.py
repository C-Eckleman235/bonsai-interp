"""Per-neuron output statistics from the weights alone (the unigram counts are the only corpus input).

Let W = W_U · diag(g) be the effective unembedding (g the final-norm gain), centred over the vocabulary (a uniform logit
shift does not change the softmax), and w = W_down[:, n] the neuron's output direction (the − pole writes −w).
- out_norm = ||w||.
- null_gap_share = ||Q^T w||² / ||w||², Q the eigenvectors of W^T W whose singular values lie below GAP x the median
  (the effective null space; 2 directions in Bonsai-8B, then a jump to 0.27 x median). Entropy neurons (Stolfo et al.
  2024) write a large norm into this space: they change the residual's RMS, so they rescale every logit (the
  confidence) without changing which token is preferred. A random direction scores k / d (0.0005 for k = 2).
  null_share is the same with a fixed NULL_K = 40 bottom directions (random ≈ 0.01), a softer near-null measure.
- logit_gain = std over the vocabulary of W w, per unit ||w||: low for null-space writers.
- freq_r = Pearson correlation over the vocabulary (tokens seen in the corpus) between W w and log unigram count, for
  the + pole (the − pole's is −freq_r). Token-frequency neurons move the output toward or away from the unigram prior.
- the direct logit effect itself (top tokens and spike) is sweeplib's predict.direct_effect.
Each statistic also gets a random-direction null (N_RAND isotropic unit directions).
"""
import numpy as np
import torch

NULL_K = 40
GAP = 0.1
N_RAND = 4096


class Unembed:
    """The centred Gram matrices of the effective unembedding, its null space and the log-frequency covector."""

    def __init__(self, model, counts, chunk=16384):
        WU = model.lm_head.weight
        g = model.model.norm.weight.float()
        V, d = WU.shape
        dev = WU.device
        cnt = np.zeros(V, np.float64)
        cnt[:min(V, len(counts))] = counts[:V]
        seen = cnt > 0
        f = np.zeros(V, np.float64)
        f[seen] = np.log(cnt[seen])
        f[seen] -= f[seen].mean()
        f_t, m_t = torch.from_numpy(f).float().to(dev), torch.from_numpy(seen).to(dev)
        gram, gm = torch.zeros(d, d, device=dev), torch.zeros(d, d, device=dev)
        mu, mum, u = torch.zeros(d, device=dev), torch.zeros(d, device=dev), torch.zeros(d, device=dev)
        for r0 in range(0, V, chunk):
            W = WU[r0:r0 + chunk].float() * g
            gram += W.T @ W
            mu += W.sum(0)
            Wm = W[m_t[r0:r0 + chunk]]
            gm += Wm.T @ Wm
            mum += Wm.sum(0)
            u += W.T @ f_t[r0:r0 + chunk]
        nm = int(seen.sum())
        C = (gram - torch.outer(mu, mu) / V).double().cpu()
        self.Cm = (gm - torch.outer(mum, mum) / nm).to(dev)  # centred over the seen tokens
        ev, Q = torch.linalg.eigh(C)
        self.eig = ev.numpy()  # ascending
        self.Q = Q[:, :NULL_K].float().to(dev)
        sv = np.sqrt(np.maximum(self.eig, 0))
        self.k_gap = max(1, int((sv < GAP * np.median(sv)).sum()))
        self.Qg = Q[:, :self.k_gap].float().to(dev)
        self.C = C.float().to(dev)
        self.u, self.fnorm, self.n_seen, self.V = u, float(np.sqrt((f[seen] ** 2).sum())), nm, V

    def stats(self, w):
        """w (d, n) float32 columns -> dict of (n,) float32 arrays."""
        nrm = w.norm(dim=0)
        null = (self.Q.T @ w).pow(2).sum(0) / nrm.pow(2)
        gap = (self.Qg.T @ w).pow(2).sum(0) / nrm.pow(2)
        var_all = (w * (self.C @ w)).sum(0).clamp_min(0)
        var_m = (w * (self.Cm @ w)).sum(0).clamp_min(1e-30)
        return {"out_norm": nrm, "null_share": null, "null_gap_share": gap, "logit_gain": (var_all / self.V).sqrt() / nrm,
                "freq_r": (self.u @ w) / (self.fnorm * var_m.sqrt())}

    def random_null(self, d, seed=0):
        r = torch.from_numpy(np.random.default_rng(seed).standard_normal((d, N_RAND)).astype(np.float32)).to(self.Q.device)
        r /= r.norm(dim=0, keepdim=True)
        return {k: v.cpu().numpy() for k, v in self.stats(r).items()}


def layer_stats(model, U, L):
    """The weight statistics and the direct logit effect of every neuron in layer L."""
    from bonsai_interp.sweeplib import predict
    Wd = model.model.layers[L].mlp.down_proj.weight().float()  # (d, N)
    out = {k: v.cpu().numpy().astype(np.float32) for k, v in U.stats(Wd).items()}
    del Wd
    S, spike = predict.direct_effect(model, L)
    out["dla_top"], out["dla_spike"] = S, spike.astype(np.float16)
    return out
