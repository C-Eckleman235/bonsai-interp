"""Run a context_tier stage from an older checkout (registered code that predates gpu_turn) with GPU turn-taking.

  cd <checkout at a registration commit>
  PYTHONPATH=. python <this repo>/scripts/auto2_turn_run.py score --layer 4 --batch 16

The registered module is not touched (its sha256 still matches its manifest): this wrapper patches the model loader so
every forward pass first takes a gpu_turn tick, exits cleanly at a batch boundary when this repo's data/auto2/STOP exists,
and releases the GPU at exit. gpu_turn is imported from this repo, so the older checkout does not need it.
"""
import atexit
import os
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gpu_turn", HERE / "bonsai_interp" / "gpu_turn.py")
gpu_turn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu_turn)
STOP = Path(os.environ.get("AUTO2_STOP", HERE / "data" / "auto2" / "STOP"))  # override for tests only
YIELD = {"seconds": 0.0, "ticks": 0}


def tick(*_):
    if STOP.exists():
        gpu_turn.release()
        sys.exit("STOP file found: stopping at a batch boundary (nothing written)")
    YIELD["seconds"] += gpu_turn.background_tick("auto2")
    YIELD["ticks"] += 1


def report():
    gpu_turn.release()
    print(f"gpu_turn: yielded {YIELD['seconds']:.1f} s over {YIELD['ticks']} batches", flush=True)


atexit.register(report)

import bonsai_interp.model as model_mod  # noqa: E402

_load = model_mod.load


def load(name="bonsai1"):
    tick()
    model, tok, rec = _load(name)
    model.model.register_forward_pre_hook(tick)
    return model, tok, rec


model_mod.load = load
from bonsai_interp.auto import context_tier  # noqa: E402  (imports load inside its functions, so it gets the patched one)

sys.argv = ["context_tier"] + sys.argv[1:]
context_tier.main()
