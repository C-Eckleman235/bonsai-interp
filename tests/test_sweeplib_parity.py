"""The sweep library's cached reader (the preamble's keys and values reused) against the full read. GPU.

  PYTHONPATH=. python tests/test_sweeplib_parity.py     -> PARITY OK when cached vs full is no worse than full vs full

Reads in a 1-bit model move with the batch shape: the same texts read in full at batch 16 and at batch 4 differ by up to
about 0.1 of a pole's max on a few of 35 million (text, pole) cells (2026-10-24: max 0.116, 99.9th percentile 0.006). So
the check compares the cached read with that floor rather than with zero. Errors are fractions of the larger of the pole
max and the reading, over non-weak poles.
"""
import numpy as np

from bonsai_interp.auto import context_tier as ct
from bonsai_interp.probe.textinv import PREAMBLES
from bonsai_interp.sweeplib import kinds as K
from bonsai_interp.sweeplib import run as R


def _err(a, b, pm, ok):
    x = np.abs(a - b)[:, :, None, :] / np.maximum(pm[None], np.abs(b)[:, :, None, :])
    x = np.where(ok[None], x, 0)
    return float(x.max()), float(np.quantile(x, 0.999)), float(np.quantile(x, 0.9999))


def main():
    bat = K.battery("number", "select")
    texts = [R._text(s) for f, q, _ in bat["num/year"][:20] for s in (f, q)]
    model, tok, rec = ct._gpu_load()
    c16 = R.read_last(model, tok, rec, texts, PREAMBLES[1], 16, cached=True)
    f16 = R.read_last(model, tok, rec, texts, PREAMBLES[1], 16, cached=False)
    f4 = R.read_last(model, tok, rec, texts, PREAMBLES[1], 4, cached=False)
    ct._gpu_idle("parity")
    pm, ok = R._pm(), R._not_weak()
    floor, got = _err(f16, f4, pm, ok), _err(c16, f16, pm, ok)
    for name, e in (("full b16 vs full b4 (floor)", floor), ("cached b16 vs full b16", got)):
        print(f"{name:28s} max {e[0]:.4f}  99.9% {e[1]:.4f}  99.99% {e[2]:.4f}")
    good = got[1] <= 1.25 * floor[1] + 1e-4 and got[2] <= 1.25 * floor[2] + 1e-4 and got[0] <= 2 * floor[0] + 0.02
    print("PARITY OK" if good else "PARITY FAILED")


if __name__ == "__main__":
    main()
