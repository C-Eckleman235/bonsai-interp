"""Memory-compact loading of the 8B precision ladder with MLP-neuron hooks.

Bonsai (1-bit) and Ternary-Bonsai store every linear weight as code * scale,
code in {-1, +1} or {-1, 0, +1}, one fp16 scale per (row, 128-column group).
PackedLinear stores the codes at 2 bits/weight and is exact for both.
Qwen3-8B (full precision) is stored as int8 with a per-group absmax scale,
which is near-lossless and lets it fit beside the display on a 16 GB card.

A "neuron" is one channel of the SwiGLU product silu(gate) * up, the unit the
essay studies. Its two poles are the positive and negative sides of that
product; the gate/up signs separate the +x+ and -x- ("echo") regimes.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent


def _model_dir(env, default):
    """Weights already on this machine can be used in place: set e.g. BONSAI1_PATH=<folder> in the
    environment or in .env, instead of downloading a second copy into models/."""
    import os
    v = os.environ.get(env)
    if not v and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
            if line.startswith(env + "="):
                v = line.split("=", 1)[1].strip().strip('"')
    return Path(v) if v else ROOT / "models" / default


MODELS = {
    "bonsai1": _model_dir("BONSAI1_PATH", "Bonsai-8B-unpacked"),
    "ternary": _model_dir("TERNARY_PATH", "Ternary-Bonsai-8B-unpacked"),
    "qwen": _model_dir("QWEN_PATH", "Qwen3-8B"),
}
GROUP = 128
COMPUTE_DTYPE = torch.bfloat16


class _DequantLinear(torch.autograd.Function):
    """x @ W.T where W is dequantized on the fly. Backward re-dequantizes W instead of
    letting autograd keep every layer's full bf16 weight alive (that is the whole model)."""

    @staticmethod
    def forward(ctx, x, mod):
        ctx.mod = mod
        return F.linear(x, mod.weight())

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out @ ctx.mod.weight(), None


class PackedLinear(nn.Module):
    """Exact 2-bit storage of a ternary/binary group-scaled weight."""

    def __init__(self, w: torch.Tensor):
        super().__init__()
        out_f, in_f = w.shape
        g = w.float().reshape(out_f, in_f // GROUP, GROUP)
        scale = g.abs().amax(-1)
        codes = torch.where(scale[..., None] > 0, g / scale[..., None].clamp_min(1e-30), 0).round()
        if not torch.equal(codes * scale[..., None], g):
            raise ValueError("weight is not group-ternary; use Int8Linear")
        c = (codes.to(torch.uint8) + 1).reshape(out_f, in_f // 4, 4)  # {0,1,2}
        self.register_buffer("packed", c[..., 0] | c[..., 1] << 2 | c[..., 2] << 4 | c[..., 3] << 6)
        self.register_buffer("scale", scale.to(torch.float16))
        self.in_features, self.out_features = in_f, out_f

    def weight(self) -> torch.Tensor:
        p = self.packed
        c = torch.stack([p & 3, p >> 2 & 3, p >> 4 & 3, p >> 6], -1).to(COMPUTE_DTYPE) - 1
        c = c.reshape(self.out_features, -1, GROUP) * self.scale[..., None].to(COMPUTE_DTYPE)
        return c.reshape(self.out_features, self.in_features)

    def forward(self, x):
        return _DequantLinear.apply(x, self)


class Int8Linear(nn.Module):
    """int8 weight, fp16 absmax scale per (row, 128-group). Near-lossless."""

    def __init__(self, w: torch.Tensor):
        super().__init__()
        out_f, in_f = w.shape
        g = w.float().reshape(out_f, in_f // GROUP, GROUP)
        scale = g.abs().amax(-1).clamp_min(1e-12) / 127
        self.register_buffer("q", (g / scale[..., None]).round().to(torch.int8))
        self.register_buffer("scale", scale.to(torch.float16))
        self.in_features, self.out_features = in_f, out_f

    def weight(self) -> torch.Tensor:
        w = self.q.to(COMPUTE_DTYPE) * self.scale[..., None].to(COMPUTE_DTYPE)
        return w.reshape(self.out_features, self.in_features)

    def forward(self, x):
        return _DequantLinear.apply(x, self)


class StopForward(Exception):
    """Raised after the recorder's stop_at layer so passes end early."""


class NeuronRecorder:
    """Collects MLP activations from hooked layers during a forward pass.

    mode: None (off), "act" (signed product only), "full" (gate, up, product).
    Interventions: per-layer callables act -> act applied before down_proj.
    """

    def __init__(self):
        self.mode = None
        self.layers = None  # set of layer indices to record, None = all
        self.store = {}
        self.interventions = {}
        self.stop_at = None  # layer index after whose MLP the forward pass stops
        self.keep_grad = False  # store "act" tensors undetached (for inverse problems)

    def clear(self):
        self.store = {}


class HookedMLP(nn.Module):
    def __init__(self, mlp, layer_idx: int, rec: NeuronRecorder):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = mlp.gate_proj, mlp.up_proj, mlp.down_proj
        self.layer_idx, self.rec = layer_idx, rec

    def forward(self, x):
        g = self.gate_proj(x)
        u = self.up_proj(x)
        a = F.silu(g) * u
        rec = self.rec
        if rec.mode and (rec.layers is None or self.layer_idx in rec.layers):
            if rec.mode == "full":
                rec.store[self.layer_idx] = (g.detach(), u.detach(), a.detach())
            else:
                rec.store[self.layer_idx] = a if rec.keep_grad else a.detach()
        fn = rec.interventions.get(self.layer_idx)
        if fn is not None:
            a = fn(a)
        if rec.stop_at == self.layer_idx:
            raise StopForward
        return self.down_proj(a)


def _read_all(path: Path):
    idx = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    by_file = {}
    for name, f in idx.items():
        by_file.setdefault(f, []).append(name)
    for f, names in by_file.items():
        with safe_open(str(path / f), "pt", device="cuda") as h:
            for n in names:
                yield n, h.get_tensor(n)


def load(name: str = "bonsai1"):
    """Returns (model, tokenizer, recorder). The model lives on cuda."""
    path = MODELS[name]
    cfg = AutoConfig.from_pretrained(path)
    cfg.dtype = COMPUTE_DTYPE
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=COMPUTE_DTYPE)
    lin_cls = Int8Linear if name == "qwen" else PackedLinear

    modules = dict(model.named_modules())
    for n, t in _read_all(path):
        mod_name, _, attr = n.rpartition(".")
        parent = modules[mod_name]
        if isinstance(parent, nn.Linear) and attr == "weight" and ".layers." in n:
            pname, _, child = mod_name.rpartition(".")
            setattr(modules[pname], child, lin_cls(t))
        else:
            parent._parameters[attr] = nn.Parameter(t.to(COMPUTE_DTYPE), requires_grad=False)
        del t
    if cfg.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight

    # rotary buffers were created on meta; rebuild them on cuda
    with torch.device("cuda"):
        model.model.rotary_emb = type(model.model.rotary_emb)(cfg)

    rec = NeuronRecorder()
    for i, layer in enumerate(model.model.layers):
        layer.mlp = HookedMLP(layer.mlp, i, rec)
    model.eval().requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(path)
    return model, tok, rec
