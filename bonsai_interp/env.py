"""The one device setting for the whole toolchain: CUDA, Apple Silicon (PyTorch MPS) or CPU.

Pick it with BONSAI_DEVICE=cuda|mps|cpu (environment or .env); otherwise the first available of
cuda > mps > cpu is used. Every module puts tensors on DEVICE and computes in DTYPE.

DTYPE is bfloat16, as on the reference CUDA machine. MPS has bfloat16 only on macOS 14+; where a
bfloat16 matmul fails we fall back to float32 (float16 can overflow Qwen3's large residual
activations) and warn. BONSAI_DTYPE=bfloat16|float16|float32 overrides the choice.
"""
import os
import warnings
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # an op MPS lacks runs on the CPU instead of failing

import torch

ROOT = Path(__file__).resolve().parent.parent


def _setting(name):
    v = os.environ.get(name)
    if not v and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
            if line.startswith(name + "="):
                v = line.split("=", 1)[1].strip().strip('"')
    return (v or "").strip().lower() or None


def _pick_device():
    want = _setting("BONSAI_DEVICE")
    if want:
        if want.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"BONSAI_DEVICE={want} but torch sees no CUDA GPU")
        if want == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("BONSAI_DEVICE=mps but torch.backends.mps is not available (Apple Silicon + macOS 12.3+ needed)")
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pick_dtype(dev):
    want = _setting("BONSAI_DTYPE")
    if want:
        return getattr(torch, want)
    if dev.type == "mps":
        try:
            a = torch.ones(4, 4, dtype=torch.bfloat16, device=dev)
            (a @ a).sum().item()
        except Exception as e:  # older macOS: no bfloat16 on MPS
            warnings.warn(f"bfloat16 is not supported on this MPS device ({type(e).__name__}); computing in float32 "
                          "(slightly different numerics from the CUDA reference; update to macOS 14+ for bfloat16)")
            return torch.float32
    return torch.bfloat16


DEVICE = _pick_device()
DTYPE = _pick_dtype(DEVICE)


def limit_memory(frac=0.8):
    """Fail loudly instead of spilling into shared memory. CUDA only; a no-op elsewhere."""
    if DEVICE.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(frac)


def empty_cache():
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()


def synchronize():
    """Wait for queued kernels (for timing)."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif DEVICE.type == "mps":
        torch.mps.synchronize()


def describe():
    """One line for logs and `play doctor`, e.g. 'cuda: NVIDIA GeForce RTX 5080, 16 GB (bfloat16)'."""
    dt = str(DTYPE).removeprefix("torch.")
    if DEVICE.type == "cuda":
        p = torch.cuda.get_device_properties(DEVICE)
        return f"cuda: {p.name}, {p.total_memory / 2**30:.0f} GB ({dt})"
    if DEVICE.type == "mps":
        import platform
        import subprocess
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        mem = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip()
        mem = f", {int(mem) / 2**30:.0f} GB unified" if mem.isdigit() else ""
        return f"mps: {chip or 'Apple Silicon'}{mem}, macOS {platform.mac_ver()[0]} ({dt})"
    return f"cpu ({dt}; slow, for checks only)"


def memory_gb():
    """Memory the model can use on DEVICE, in GB (GPU memory, unified memory on a Mac, RAM on CPU)."""
    if DEVICE.type == "cuda":
        return torch.cuda.get_device_properties(DEVICE).total_memory / 2**30
    if DEVICE.type == "mps":
        return torch.mps.recommended_max_memory() / 2**30
    try:
        import psutil
        return psutil.virtual_memory().total / 2**30
    except ImportError:
        return float("nan")
