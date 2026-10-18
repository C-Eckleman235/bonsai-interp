"""L2 backward maps: in-context causal antecedents of one neuron on a real input.

A NeuronSite is (model, input ids, target layer L, neuron n, position p). Upstream
units are earlier-layer MLP neurons; each gets a gain θ multiplying its activation
at every position (θ=1 clean, θ=0 zero-ablated). The forward pass stops right after
layer L, so one evaluation costs ~L/36 of a forward pass and θ-batches run as one
batch. The gain map θ -> target activation is differentiable, so the same problem
serves least_squares / L-BFGS-B (torch_fn) and black-box solvers (evaluate_batch).
"""
import numpy as np
import torch

from bonsai_interp.env import DEVICE
from bonsai_interp.inverse.problem import InverseProblem, Var
from bonsai_interp.model import StopForward


class NeuronSite:
    """max_ctx caps the tokens kept before the target position (backward passes through
    deep layers on 1k tokens do not fit beside the model on a 16 GB card)."""

    def __init__(self, model, rec, ids, L, n, pos, max_ctx=384):
        self.model, self.rec = model, rec
        start = max(0, pos + 1 - max_ctx)
        self.ids = torch.as_tensor(ids[start: pos + 1], device=DEVICE).long()[None]  # nothing after p matters
        self.L, self.n, self.pos = L, n, pos - start

    def run(self, masks: dict, batch: int = 1):
        """Target activation for each batch row; masks[l] is (batch, 12288) gains."""
        rec = self.rec
        rec.interventions = {l: (lambda a, m=m: a * m[:, None, :].to(a.dtype)) for l, m in masks.items()}
        rec.mode, rec.layers, rec.stop_at, rec.keep_grad = "act", {self.L}, self.L, torch.is_grad_enabled()
        try:
            self.model.model(self.ids.expand(batch, -1))
        except StopForward:
            pass
        finally:
            rec.interventions, rec.stop_at, rec.mode, rec.keep_grad = {}, None, None, False
        out = rec.store.pop(self.L)[:, self.pos, self.n]
        rec.clear()
        return out.float()

    def clean(self):
        with torch.no_grad():
            return self.run({}).item()

    def attribution(self):
        """d(target)/d(gain) at θ=1 for every upstream neuron: Σ_t a_t · ∂target/∂a_t."""
        masks = {l: torch.ones(1, self.model.config.intermediate_size, device=DEVICE, requires_grad=True)
                 for l in range(self.L)}
        with torch.enable_grad():
            y = self.run(masks)
            y.sum().backward()
        return {l: m.grad[0].float().cpu().numpy() for l, m in masks.items()}, y.item()


def circuit_problem(site: NeuronSite, attr: dict, k: int = 24, gain_hi: float = 1.0, chunk: int = 16):
    """Gains on the k upstream neurons with the largest |attribution|.

    Outputs: 'act' (target activation), 'kept' (Σθ, number of units effectively kept),
    'frac' (act / clean act).
    """
    flat = [(abs(v), l, m) for l, a in attr.items() for m, v in enumerate(a)]
    flat.sort(reverse=True)
    units = [(l, m) for _, l, m in flat[:k]]
    clean = site.clean()
    N = site.model.config.intermediate_size

    def masks_for(theta):  # theta: (B, k) tensor
        B = theta.shape[0]
        masks = {}
        for j, (l, m) in enumerate(units):
            if l not in masks:
                masks[l] = torch.ones(B, N, device=DEVICE, dtype=theta.dtype)
            masks[l] = masks[l].index_put((torch.arange(B, device=DEVICE), torch.full((B,), m, device=DEVICE)),
                                          theta[:, j])
        return masks

    def outputs(theta, act):
        kept = theta.sum(-1)
        return torch.stack([act, kept, act / clean], -1)

    def evaluate_batch(X):
        res = []
        with torch.no_grad():
            for s in range(0, len(X), chunk):
                th = torch.tensor(X[s:s + chunk], dtype=torch.float32, device=DEVICE)
                res.append(outputs(th, site.run(masks_for(th), batch=len(th))).cpu().numpy())
        return np.concatenate(res)

    def torch_fn(theta):
        with torch.enable_grad():
            th = theta[None]
            return outputs(th, site.run(masks_for(th)))[0]

    vars_ = [Var(f"L{l}N{m}", 0.0, gain_hi) for l, m in units]
    p = InverseProblem(f"circuit L{site.L}N{site.n}@{site.pos}", vars_, ["act", "kept", "frac"],
                       evaluate_batch, torch_fn)
    p.units, p.clean = units, clean
    return p
