"""One directory per probed neuron pole: probes/L{L}_N{n}_{pos|neg}/.

Stages write JSON files. Stages that need an LLM (label, bank ratings, judging)
write a *_request.json describing exactly what is needed; whoever fills it in
(Claude in a session, or an API call) writes the matching JSON file, and the
next stage picks it up.
"""
import json
from pathlib import Path

from bonsai_interp.model import ROOT


def _plain(o):
    """json.dumps fallback for values solvers hand back: numpy scalars and arrays, sets, paths."""
    if hasattr(o, "item") and getattr(o, "shape", None) == ():  # numpy scalar (int64, float32, bool_)
        return o.item()
    if hasattr(o, "tolist"):  # numpy array
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    if isinstance(o, Path):
        return o.as_posix()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


class Workspace:
    def __init__(self, L: int, n: int, pole: str = "+", model: str = "bonsai1"):
        self.L, self.n, self.pole, self.model = L, n, pole, model
        self.pole_idx = 0 if pole == "+" else 1
        self.dir = ROOT / "probes" / f"{model}_L{L}_N{n}_{'pos' if pole == '+' else 'neg'}"
        self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def id(self):
        return f"L{self.L}/N{self.n}{self.pole}"

    def path(self, name):
        return self.dir / name

    def has(self, name):
        return self.path(name).exists()

    def load(self, name):
        return json.loads(self.path(name).read_text(encoding="utf-8"))

    def save(self, name, obj):
        # serialise first, so a bad value cannot leave a half-written file behind
        text = json.dumps(obj, ensure_ascii=False, indent=1, default=_plain)
        self.path(name).write_text(text, encoding="utf-8")
        return self.path(name)

    def require(self, name, how):
        if not self.has(name):
            raise SystemExit(f"[{self.id}] missing {self.path(name)}\n  -> {how}")
        return self.load(name)
