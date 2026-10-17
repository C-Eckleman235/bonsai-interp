"""L0/L1 backward maps: exact, data-free (weights only) inverses of one neuron.

For target neuron n in layer L, gate and up are linear in the normalized residual:
    g = (γ_L ⊙ w_g[n]) · x / rms(x)        u = (γ_L ⊙ w_u[n]) · x / rms(x)
The residual is a sum of writes; an upstream MLP neuron m in layer l < L writes
a_m · d_m (column m of down_proj). Its direct "virtual weight" into the target is
    V_g[l, m] = (γ_L ⊙ w_g[n]) · d_m
so on the direct path g = Σ V_g a / rms. Embedding rows give the same for tokens.

Inverse problems solved exactly here:
  * which upstream neurons/tokens can push each of g and u, and how hard
    (|V| times the neuron's observed activation range)
  * min-support set (scipy milp): the fewest upstream neurons whose direct writes,
    within observed ranges, move (g, u) from baseline into a target pole regime.
"""
import json

import numpy as np
import torch
from safetensors import safe_open
from scipy.optimize import Bounds, LinearConstraint, milp

from mono.model import MODELS


class Weights:
    """Lazy access to raw tensors of one model (fp32, on `device`)."""

    def __init__(self, name="bonsai1", device="cpu"):
        self.path = MODELS[name]
        self.idx = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        self.device = device

    def get(self, tensor_name, rows=None):
        with safe_open(str(self.path / self.idx[tensor_name]), "pt", device="cpu") as h:
            if rows is None:
                t = h.get_tensor(tensor_name)
            else:
                t = h.get_slice(tensor_name)[rows]
        return t.float().to(self.device)


def read_vectors(W: Weights, L: int, n: int):
    """(γ_L ⊙ w_g[n], γ_L ⊙ w_u[n]) — the neuron's two read directions."""
    gamma = W.get(f"model.layers.{L}.post_attention_layernorm.weight")
    wg = W.get(f"model.layers.{L}.mlp.gate_proj.weight", slice(n, n + 1))[0]
    wu = W.get(f"model.layers.{L}.mlp.up_proj.weight", slice(n, n + 1))[0]
    return gamma * wg, gamma * wu


def virtual_weights(W: Weights, L: int, n: int):
    """V[l] : (2, 12288) direct effect of each upstream neuron's activation on (g, u)·rms."""
    rg, ru = read_vectors(W, L, n)
    R = torch.stack([rg, ru])  # (2, 4096)
    return {l: (R @ W.get(f"model.layers.{l}.mlp.down_proj.weight")).cpu().numpy() for l in range(L)}


def token_weights(W: Weights, L: int, n: int):
    """Direct effect of each token embedding on (g, u)·rms: (2, vocab)."""
    rg, ru = read_vectors(W, L, n)
    E = W.get("model.embed_tokens.weight")
    return (torch.stack([rg, ru]) @ E.T).cpu().numpy()


def min_support(V, lo, hi, need_g, need_u, rms, max_candidates=3000, time_limit=60):
    """Fewest upstream neurons whose direct writes add >= need_g to g and >= need_u to u.

    V: (2, M) virtual weights (all upstream neurons flattened); lo/hi: (M,) observed
    activation range of each upstream neuron; rms: residual rms at the target layer.
    Signs of need_* select the regime: e.g. +x+ pole: need_g > 0, need_u > 0;
    echo (-x-): need_g < 0, need_u < 0 (pass the constraint as <= by negating).
    Returns (chosen indices, activations) or (None, None) if infeasible.
    """
    sg, su = np.sign(need_g) or 1.0, np.sign(need_u) or 1.0
    Vg, Vu = sg * V[0] / rms, su * V[1] / rms  # want Vg·a >= |need_g|, Vu·a >= |need_u|
    # best achievable push of each neuron toward both goals, to pick candidates
    reach = np.maximum(np.maximum(Vg * hi, Vg * lo), 0) / max(abs(need_g), 1e-9) + \
            np.maximum(np.maximum(Vu * hi, Vu * lo), 0) / max(abs(need_u), 1e-9)
    cand = np.argsort(-reach)[:max_candidates]
    k = len(cand)
    # variables: a (k, continuous in [lo, hi]) then z (k, binary)
    c = np.concatenate([np.zeros(k), np.ones(k)])
    A_goal = np.zeros((2, 2 * k))
    A_goal[0, :k], A_goal[1, :k] = Vg[cand], Vu[cand]
    # link: a - hi z <= 0 and a - lo z >= 0  (a can be nonzero only if z = 1)
    eye = np.eye(k)
    A_up = np.hstack([eye, -np.diag(hi[cand])])
    A_dn = np.hstack([eye, -np.diag(lo[cand])])
    cons = [LinearConstraint(A_goal, [abs(need_g), abs(need_u)], np.inf),
            LinearConstraint(A_up, -np.inf, 0), LinearConstraint(A_dn, 0, np.inf)]
    lb = np.concatenate([np.minimum(lo[cand], 0), np.zeros(k)])
    ub = np.concatenate([np.maximum(hi[cand], 0), np.ones(k)])
    res = milp(c, constraints=cons, bounds=Bounds(lb, ub),
               integrality=np.concatenate([np.zeros(k), np.ones(k)]),
               options={"time_limit": time_limit, "disp": False})
    if res.x is None:
        return None, None
    a, z = res.x[:k], res.x[k:] > 0.5
    return cand[z], a[z]
