"""Steer lens (after lyra's neuron-explorer): push a direction into the residual stream at the last position of a few
short neutral contexts and read the next-token distribution.

The push is added after layer L's MLP (the same as adding it to down_proj's output) at the last position only, so every
earlier position is unchanged. The prefix's keys and values are therefore computed once per context, and a push costs
one single-token pass through layers L+1..35 plus the unembedding. Many pushes run as one batch.

Dose: alpha x ||h_L|| (the residual norm at the last position after layer L, per context) along the unit direction.
alpha is calibrated per layer so that random unit directions reach a fixed median KL (CALIB_KL).
"""
import numpy as np
import torch
import torch.nn.functional as F

CONTEXTS = (
    "The meeting was held on a Tuesday, and",
    "She picked up the book and",
    "In the next section, we describe the",
    "He said that he would",
)
CALIB_KL = 0.1          # target median KL(p0 || p) of random unit directions, nats
ALPHA_GRID = (0.03, 0.06, 0.12, 0.25, 0.5, 1.0, 2.0, 4.0)
TOPK = 10


def _rot(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), -1)


class Lens:
    def __init__(self, model, tok, contexts=CONTEXTS):
        self.m, self.layers = model, model.model.layers
        cfg = model.config
        self.H, self.Hkv, self.hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.G, self.scale = self.H // self.Hkv, self.hd ** -0.5
        self.nL = len(self.layers)
        self.dev, self.dt = model.lm_head.weight.device, model.lm_head.weight.dtype
        self.ids = [tok(c)["input_ids"] for c in contexts]
        with torch.no_grad():
            self._prefix()

    def _rope(self, pos):
        x = torch.zeros(1, dtype=self.dt, device=self.dev)
        cos, sin = self.m.model.rotary_emb(x, torch.tensor([pos], device=self.dev))
        return cos[0], sin[0]  # (T, hd)

    def _qkv(self, a, x, cos, sin):
        n = x.shape[0]
        q = a.q_norm(a.q_proj(x).view(n, self.H, self.hd))
        k = a.k_norm(a.k_proj(x).view(n, self.Hkv, self.hd))
        v = a.v_proj(x).view(n, self.Hkv, self.hd)
        c, s = cos[:, None], sin[:, None]
        return q * c + _rot(q) * s, k * c + _rot(k) * s, v

    def _prefix(self):
        """Keys/values of every prefix (all tokens but the last) by layer, then the unpushed last-token pass: the residual
        after each layer (h) and before each MLP (mid), and the baseline log-probs."""
        self.K, self.V, self.cs, self.h, self.mid, self.lp0 = [], [], [], [], [], []
        for ids in self.ids:
            P = len(ids) - 1
            x = self.m.model.embed_tokens(torch.tensor(ids[:-1], device=self.dev))
            cos, sin = self._rope(list(range(P)))
            Ks, Vs = [], []
            for layer in self.layers:
                a = layer.self_attn
                q, k, v = self._qkv(a, layer.input_layernorm(x), cos, sin)
                o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1).repeat_interleave(self.G, 0)[None],
                                                   v.transpose(0, 1).repeat_interleave(self.G, 0)[None], is_causal=True, scale=self.scale)
                x = x + a.o_proj(o[0].transpose(0, 1).reshape(P, -1))
                x = x + layer.mlp(layer.post_attention_layernorm(x))
                Ks.append(k.transpose(0, 1).contiguous())  # (Hkv, P, hd)
                Vs.append(v.transpose(0, 1).contiguous())
            self.K.append(Ks)
            self.V.append(Vs)
            self.cs.append(self._rope([P]))
            c = len(self.cs) - 1
            h = self.m.model.embed_tokens(torch.tensor(ids[-1:], device=self.dev))
            hs, mids = [], []
            for l in range(self.nL):
                h, mid = self._layer(l, h, c)
                hs.append(h[0])
                mids.append(mid[0])
            self.h.append(torch.stack(hs))      # (nL, d)
            self.mid.append(torch.stack(mids))  # (nL, d)
            self.lp0.append(self._logp(h)[0])
        self.hnorm = torch.stack([h.float().norm(dim=-1) for h in self.h])  # (C, nL)
        self.p0 = [lp.exp() for lp in self.lp0]

    def _layer(self, l, h, c):
        layer = self.layers[l]
        a = layer.self_attn
        n = h.shape[0]
        q, k, v = self._qkv(a, layer.input_layernorm(h), *self.cs[c])
        K, V = self.K[c][l], self.V[c][l]
        qg = q.view(n, self.Hkv, self.G, self.hd)
        s = torch.cat([torch.einsum("bkgd,kpd->bkgp", qg, K), torch.einsum("bkgd,bkd->bkg", qg, k)[..., None]], -1) * self.scale
        w = torch.softmax(s.float(), -1).to(h.dtype)
        o = torch.einsum("bkgp,kpd->bkgd", w[..., :-1], V) + w[..., -1:] * v[:, :, None]
        h = h + a.o_proj(o.reshape(n, -1))
        return h + layer.mlp(layer.post_attention_layernorm(h)), h

    def _logp(self, h):
        return torch.log_softmax(self.m.lm_head(self.m.model.norm(h)).float(), -1)

    def run_from(self, l0, h, c):
        for l in range(l0, self.nL):
            h, _ = self._layer(l, h, c)
        return self._logp(h)

    @torch.no_grad()
    def check(self, tok=None):
        """Max |log-prob| difference between this lens's unpushed last-token pass and the full HF forward, per context."""
        out = []
        for c, ids in enumerate(self.ids):
            ref = torch.log_softmax(self.m(torch.tensor([ids], device=self.dev)).logits[0, -1].float(), -1)
            top = ref.topk(50).indices
            out.append(float((ref[top] - self.lp0[c][top]).abs().max()))
        return out

    @torch.no_grad()
    def push(self, L, dirs, alpha, topk=TOPK):
        """dirs (n, d) unit directions. Each is pushed +alpha and -alpha (times ||h_L|| per context) after layer L.
        Returns float32 kl (2, n) (mean over contexts of KL(p0 || p)), int32 top ids (2, n, topk) and float32 top
        Δlog-probs (2, n, topk) of the context-mean Δlog-prob."""
        n = dirs.shape[0]
        d = dirs.to(self.dev, torch.float32)
        kl = torch.zeros(2 * n, device=self.dev)
        acc = None
        for c in range(len(self.ids)):
            h0 = self.h[c][L].float()
            delta = (alpha * float(self.hnorm[c, L])) * d
            rows = torch.cat([h0 + delta, h0 - delta]).to(self.dt)
            lp = self.run_from(L + 1, rows, c) if L + 1 < self.nL else self._logp(rows)
            lp -= self.lp0[c]  # Δlog-prob
            kl -= lp @ self.p0[c]
            acc = lp if acc is None else acc.add_(lp)
            del lp
        acc /= len(self.ids)
        v, i = acc.topk(topk, dim=-1)
        kl /= len(self.ids)
        return (kl.view(2, n).cpu().numpy(), i.view(2, n, topk).int().cpu().numpy(), v.view(2, n, topk).cpu().numpy())

    def calibrate(self, L, gen, n=64, target=CALIB_KL):
        """alpha at which the median KL of n random unit directions (both signs) is `target`, interpolated log-log on
        ALPHA_GRID; returns (alpha, [(alpha, median KL)])."""
        r = torch.from_numpy(gen.standard_normal((n, self.h[0].shape[-1])).astype(np.float32))
        r /= r.norm(dim=-1, keepdim=True)
        curve = [(a, float(np.median(self.push(L, r, a, topk=1)[0]))) for a in ALPHA_GRID]
        xs = np.log([a for a, _ in curve])
        ys = np.maximum.accumulate(np.log([max(k, 1e-12) for _, k in curve]))
        lt = np.log(target)
        if lt <= ys[0]:
            alpha = float(np.exp(xs[0] + (lt - ys[0]) / max(ys[1] - ys[0], 1e-6) * (xs[1] - xs[0])))
        elif lt >= ys[-1]:
            alpha = float(np.exp(xs[-1]))
        else:
            j = int(np.searchsorted(ys, lt))  # KL rises with alpha
            alpha = float(np.exp(xs[j - 1] + (lt - ys[j - 1]) / (ys[j] - ys[j - 1]) * (xs[j] - xs[j - 1])))
        return alpha, curve
