"""What this machine offers, detected in one place: OS, compute device, dtype, memory, agent CLIs.

Device: BONSAI_DEVICE=cuda|mps|cpu (environment or .env); otherwise the first available of
cuda > mps > cpu. On a CUDA machine nothing changes from the original CUDA-only toolchain:
same device, bfloat16, same memory cap. Every module puts tensors on DEVICE and computes in DTYPE.

DTYPE is bfloat16 where the device supports it (any recent NVIDIA GPU, the CPU, MPS on macOS 14+).
Otherwise it falls back to float32 with a warning (float16 can overflow Qwen3's large residual
activations). BONSAI_DTYPE=bfloat16|float16|float32 overrides the choice.

`python -m bonsai_interp.env` prints the capability report that `play doctor` shows.
"""
import os
import platform
import shutil
import subprocess
import warnings
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # an op MPS lacks runs on the CPU instead of failing

import torch

ROOT = Path(__file__).resolve().parent.parent
OS = {"Windows": "windows", "Darwin": "macos", "Linux": "linux"}.get(platform.system(), platform.system().lower())


def _setting(name):
    v = os.environ.get(name)
    if not v and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
            if line.startswith(name + "="):
                v = line.split("=", 1)[1].strip().strip('"')
    return (v or "").strip().lower() or None


def _cuda_ok():
    # CUDA_VISIBLE_DEVICES="" leaves is_available() True with no device to use
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def _pick_device():
    want = _setting("BONSAI_DEVICE")
    if want:
        if want.startswith("cuda") and not _cuda_ok():
            raise RuntimeError(f"BONSAI_DEVICE={want} but torch sees no CUDA GPU")
        if want == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("BONSAI_DEVICE=mps but torch.backends.mps is not available (Apple Silicon + macOS 14 needed)")
        return torch.device(want)
    if _cuda_ok():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pick_dtype(dev):
    want = _setting("BONSAI_DTYPE")
    if want:
        return getattr(torch, want)
    if dev.type == "cuda" and not torch.cuda.is_bf16_supported():
        warnings.warn(f"{torch.cuda.get_device_name(dev)} has no bfloat16; computing in float32 (slower, slightly different numerics)")
        return torch.float32
    if dev.type == "mps":
        try:
            a = torch.ones(4, 4, dtype=torch.bfloat16, device=dev)
            (a @ a).sum().item()
        except Exception as e:  # macOS before 14: no bfloat16 on MPS
            warnings.warn(f"bfloat16 is not supported on this MPS device ({type(e).__name__}); computing in float32 "
                          "(slightly different numerics from the CUDA reference; update to macOS 14+ for bfloat16)")
            return torch.float32
    return torch.bfloat16


DEVICE = _pick_device()
DTYPE = _pick_dtype(DEVICE)


def limit_memory(frac=0.8):
    """Fail loudly instead of spilling into shared memory. CUDA only; a no-op elsewhere.

    BONSAI_GPU_FRACTION (environment or .env) lowers the cap for every process of a checkout, so jobs that share
    the GPU (a stream and a batch run) each stay inside a budget: the caps must add up to less than the card."""
    budget = _setting("BONSAI_GPU_FRACTION")
    if budget:
        frac = min(frac, float(budget))
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


def _sysctl(key):
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def ram_gb():
    """System RAM (unified memory on a Mac), in GB; nan when unknown."""
    if OS == "macos":
        v = _sysctl("hw.memsize")
        return int(v) / 2**30 if v.isdigit() else float("nan")
    try:
        import psutil
        return psutil.virtual_memory().total / 2**30
    except ImportError:
        return float("nan")


def memory_gb():
    """Memory the model can use on DEVICE, in GB: GPU memory, unified memory on a Mac, RAM on the CPU."""
    if DEVICE.type == "cuda":
        return torch.cuda.get_device_properties(DEVICE).total_memory / 2**30
    return ram_gb()


def describe():
    """One line, e.g. 'cuda: NVIDIA GeForce RTX 5080, 16 GB (bfloat16)'."""
    dt = str(DTYPE).removeprefix("torch.")
    if DEVICE.type == "cuda":
        p = torch.cuda.get_device_properties(DEVICE)
        return f"cuda: {p.name}, {p.total_memory / 2**30:.0f} GB ({dt})"
    if DEVICE.type == "mps":
        return f"mps: {_sysctl('machdep.cpu.brand_string') or 'Apple Silicon'}, {ram_gb():.0f} GB unified ({dt})"
    return f"cpu ({dt}; slow, for checks only)"


def agent_clis():
    """Agent CLIs on PATH (blind clean-room missions need one): name -> path or None."""
    return {name: shutil.which(name) for name in ("claude", "codex")}


def report():
    """The capability report: what this machine can run and how."""
    os_ver = {"macos": platform.mac_ver()[0], "windows": platform.version()}.get(OS, platform.release())
    return {"os": f"{OS} {os_ver} ({platform.machine()})", "device": DEVICE.type, "dtype": str(DTYPE).removeprefix("torch."),
            "accelerator": describe(), "memory_gb": round(memory_gb(), 1), "ram_gb": round(ram_gb(), 1),
            "device_forced": _setting("BONSAI_DEVICE") is not None, "torch": torch.__version__, "agent_clis": agent_clis()}


def report_lines():
    r = report()
    clis = ", ".join(f"{k} ({'found' if v else 'missing'})" for k, v in r["agent_clis"].items())
    return [f"OS           {r['os']}",
            f"device       {r['accelerator']}" + (" [BONSAI_DEVICE]" if r["device_forced"] else " [auto: cuda > mps > cpu]"),
            f"memory       {r['memory_gb']:.0f} GB on the device, {r['ram_gb']:.0f} GB RAM",
            f"torch        {r['torch']}",
            f"agent CLIs   {clis}"]


if __name__ == "__main__":
    print("\n".join(report_lines()))
