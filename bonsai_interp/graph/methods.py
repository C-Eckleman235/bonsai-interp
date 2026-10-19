"""Which agents, screens and toolchains produced a contribution, so the survey can learn which
techniques pay off (`python -m bonsai_interp.graph.meta`).

Every new contribution carries a "methods" block:

  {"agent": "claude-code opus-5.5" | "codex gpt-6.1" | "human" | ...,
   "screens": ["look", "bands", "probe", "quick-test", ...],   declared by the contributor, plus
   "detected": [...],                                          what play run could see on disk
   "toolchain_commit": "...", "harness": "neuron-cleanroom@..." (blind runs),
   "minutes": 37}                                              pre-registration commit -> results

SCREENS is the shared vocabulary. Unknown screen names are allowed (new tools arrive by PR) but
reported, so the vocabulary can be extended in the same PR that adds the tool.
"""
import subprocess
from pathlib import Path

SCREENS = {
    "look": "top corpus examples (play look / explore)",
    "bands": "full activation distribution and band samples (probe.bands)",
    "probe": "exploratory sentence scoring, nothing recorded (play probe)",
    "card": "neuron card: top examples, density, regimes, logit effect, flags (play card)",
    "which": "reverse search: which poles fire on a text or token (play which)",
    "quick-test": "pre-registered 8 + 8 sentence test (quick-8+8)",
    "second-round": "a further labelling round after a failed test",
    "step-out": "strongest feeders by attribution + single-unit zeroing (play step), edges recorded",
    "step-down": "readers: later poles that lose activation in this pole's contexts when it is zeroed (play step --down), edges recorded",
    "bank-solo": "sentence bank with competing hypotheses, dose-response (probe.run solo)",
    "held-out": "pre-registered held-out set (held-out protocol)",
    "inverse-de": "text-level inverse with SciPy differential evolution (probe.run invert --solver de)",
    "inverse-gmoo": "text-level inverse with a globalMOO learned model (probe.run invert --solver gmoo)",
    "induce": "steering the neuron and reading output effects (probe.run induce)",
    "neighbors": "upstream / disambiguator / downstream map (probe.neighbors)",
    "disproof": "corpus-wide attempts to break the label (frequency tables, band annotation, leak search)",
    "blind-room": "investigation by a blind agent in a clean room (neuron-cleanroom)",
    "atlas": "word x frame x language fingerprint",
}


def detect(root: Path, L: int, n: int, pole: str) -> list:
    """Screens whose traces for this neuron exist in this checkout (probe files, band scans)."""
    found = []
    d = root / "probes" / f"bonsai1_L{L}_N{n}_{'pos' if pole == '+' else 'neg'}"
    files = {p.name for p in d.glob("*")} if d.exists() else set()
    for screen, names in (("bank-solo", {"bank.json", "solo.json"}), ("held-out", {"holdout.json"}),
                          ("inverse-de", {"invert_de.json"}), ("inverse-gmoo", {"invert_gmoo.json"}),
                          ("induce", {"induce.json"}), ("neighbors", {"neighbors.json"}), ("look", {"forward.json"})):
        if files & names:
            found.append(screen)
    import zipfile
    for npz in (root / "data").glob("bands_*.npz"):
        try:
            if f"L{L}_N{n}.npy" in zipfile.ZipFile(npz).namelist():
                found.append("bands")
                break
        except zipfile.BadZipFile:
            pass
    return sorted(set(found))


def minutes_between(root: Path, tests_rel: str) -> int | None:
    out = subprocess.run(["git", "log", "--diff-filter=A", "--format=%ct", "--", tests_rel], cwd=root, capture_output=True, text=True).stdout.split()
    if not out:
        return None
    import time
    return round((time.time() - int(out[-1])) / 60)
