"""Read Bonsai-8B from PrismML's 1.16 GB Q1_0 GGUF instead of the 16 GB unpacked safetensors.

A Q1_0 block is 128 weights: one fp16 scale and 128 sign bits (bit 1 -> +scale, bit 0 -> -scale).
That is exactly the group-binary form PackedLinear stores, so the 2-bit codes are built straight
from the bits, with no float copy of the weight. Checked against the unpacked safetensors: every
linear weight, the embeddings and the output head are bit-identical, and the norms are identical
after rounding through float16 (how the unpacked release stores them). On CUDA the two loads give
bit-identical activations and logits.

The folder also needs the small config and tokenizer files from prism-ml/Bonsai-8B-unpacked
(scripts/download_models.sh fetches both).
"""
import re
from pathlib import Path

import numpy as np
import torch

# GGUF (llama.cpp qwen3) tensor names -> Hugging Face Qwen3 names
_TOP = {"token_embd.weight": "model.embed_tokens.weight", "output.weight": "lm_head.weight",
        "output_norm.weight": "model.norm.weight"}
_BLK = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
        "attn_output": "self_attn.o_proj", "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
        "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
        "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}


def find(path: Path):
    """The Q1_0 GGUF in a model folder, or None."""
    hits = sorted(Path(path).glob("*Q1_0*.gguf")) or sorted(Path(path).glob("*.gguf"))
    return hits[0] if hits else None


def hf_name(name: str) -> str:
    if name in _TOP:
        return _TOP[name]
    m = re.fullmatch(r"blk\.(\d+)\.(\w+)\.weight", name)
    if not m or m.group(2) not in _BLK:
        raise KeyError(f"unexpected GGUF tensor {name}")
    return f"model.layers.{m.group(1)}.{_BLK[m.group(2)]}.weight"


def _blocks(t):
    raw = np.asarray(t.data).view(np.uint8).reshape(-1, 18)
    out_f, in_f = int(t.shape[1]), int(t.shape[0])  # GGUF lists the fastest dimension first
    scale = raw[:, :2].copy().view(np.float16).reshape(out_f, in_f // 128)
    bits = np.unpackbits(raw[:, 2:], axis=1, bitorder="little").reshape(out_f, in_f)
    return bits, scale, out_f, in_f


def packed_parts(t):
    """(packed uint8 [out, in/4], fp16 scale [out, in/128], out_f, in_f) in PackedLinear's layout:
    codes {-1, +1} stored as {0, 2}, four per byte, little end first."""
    bits, scale, out_f, in_f = _blocks(t)
    c = (bits.reshape(out_f, in_f // 4, 4) * 2).astype(np.uint8)
    packed = c[..., 0] | c[..., 1] << 2 | c[..., 2] << 4 | c[..., 3] << 6
    return torch.from_numpy(packed), torch.from_numpy(scale.copy()), out_f, in_f


def dense(t) -> torch.Tensor:
    """The full float32 weight (embeddings, norms)."""
    if t.tensor_type.name == "Q1_0":
        bits, scale, out_f, in_f = _blocks(t)
        w = (bits.astype(np.float32) * 2 - 1).reshape(out_f, in_f // 128, 128) * scale.astype(np.float32)[..., None]
        return torch.from_numpy(w.reshape(out_f, in_f))
    if t.tensor_type.name in ("F32", "F16", "BF16"):
        # the unpacked release (the reference every result was measured on) stores the norms as float16; the GGUF's
        # float32 copies differ in the last bits, so round through float16 to reproduce the reference exactly
        w = np.asarray(t.data).astype(np.float16).astype(np.float32)
        return torch.from_numpy(w.reshape([int(x) for x in reversed(t.shape)]))
    raise ValueError(f"{t.name}: tensor type {t.tensor_type.name} is not supported (only the Q1_0 Bonsai GGUF is)")


def tensors(path: Path):
    """Yields (hf_name, gguf tensor) for every tensor in the folder's GGUF."""
    from gguf import GGUFReader
    for t in GGUFReader(str(find(path))).tensors:
        yield hf_name(t.name), t
