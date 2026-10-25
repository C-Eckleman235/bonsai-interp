"""Contributions: immutable, append-only files that are the only way information enters the graph.

Layout (one folder per neuron pole):

  graph/contributions/<model>/L{L}_N{n}_{pos|neg}/<id>.tests.json   pre-registration (committed FIRST)
  graph/contributions/<model>/L{L}_N{n}_{pos|neg}/<id>.json         results (committed in a LATER commit)

`<id>` is `YYYY-MM-DD-<github handle>-<4+ chars>`, unique forever. Contributions are never edited
or deleted; a correction is a new contribution. Neuron records under graph/neurons/ are derived
from contributions by `fold` and are never edited by hand.

Kinds: interpret (label a pole), qc (re-test a label), replicate (blind clean-room run), map (edges only),
withhold (the pole's trigger is sensitive content: a masked description instead of a live test).
"""
import json
import re
from pathlib import Path

from bonsai_interp.graph.protocols import LEGACY, PROTOCOLS

ROOT = Path(__file__).resolve().parents[2]
C = ROOT / "graph" / "contributions"
KINDS = {"interpret", "qc", "replicate", "map", "withhold", "coverage"}
WITHHELD_REASONS = {"slur", "sexual", "personal data", "violence", "other"}
QC_VERDICTS = {"confirm", "dispute", "refine"}
ID = re.compile(r"^\d{4}-\d{2}-\d{2}-[A-Za-z0-9-]+-[A-Za-z0-9]{4,}$|^core-[a-z0-9-]+$")
NEURON = re.compile(r"^(?P<model>[a-z0-9_]+)/L(?P<L>\d+)/N(?P<n>\d+)(?P<pole>[+-])$")
LOCAL_PATH = re.compile(r"(?i)\b[a-z]:[\\/]+users[\\/]|/home/[a-z]|/Users/[A-Za-z]")
MAX_FILE = 64 * 1024
METHODS_REQUIRED_FROM = "2026-10-18T18:30"  # contributions created earlier predate the methods block
MAX_NOTES = 4000
MAX_EXAMPLE = 400


def folder_for(neuron):
    m = NEURON.match(neuron)
    return C / m["model"] / f"L{m['L']}_N{m['n']}_{'pos' if m['pole'] == '+' else 'neg'}"


def load_all():
    """[(path, contribution)] for every results file, in deterministic order."""
    out = []
    for p in sorted(C.glob("*/*/*.json")):
        if p.name.endswith(".tests.json"):
            continue
        out.append((p, json.loads(p.read_text(encoding="utf-8"))))
    return out


def check(path: Path, c: dict):
    """Problems with one results file (empty list = fine)."""
    errs = []
    say = errs.append
    rel = path.relative_to(ROOT).as_posix()
    if path.stat().st_size > MAX_FILE:
        say(f"{rel}: larger than {MAX_FILE // 1024} KB; move bulky material to the library and list it under 'artifacts'")
    if LOCAL_PATH.search(path.read_text(encoding="utf-8")):
        say(f"{rel}: contains a local filesystem path")
    for k in ("id", "kind", "neuron", "created", "contributor"):
        if not c.get(k):
            say(f"{rel}: needs '{k}'")
    if errs:
        return errs
    if not ID.match(c["id"]) or path.stem != c["id"]:
        say(f"{rel}: id must match the file name and look like YYYY-MM-DD-<handle>-<suffix>")
    if c["kind"] not in KINDS:
        say(f"{rel}: kind must be one of {sorted(KINDS)}")
    if not NEURON.match(c["neuron"]) or folder_for(c["neuron"]) != path.parent:
        say(f"{rel}: neuron {c['neuron']!r} does not match its folder")
    if not c["id"].startswith("core-") and c["kind"] in ("interpret", "qc", "replicate") and c["created"] >= METHODS_REQUIRED_FROM:
        mt = c.get("methods") or {}
        if not mt.get("agent") or not isinstance(mt.get("screens"), list) or not mt.get("screens"):
            say(f"{rel}: needs a 'methods' block with 'agent' and a non-empty 'screens' list (see bonsai_interp/graph/methods.py)")
    if c["kind"] in ("interpret", "qc") and not c.get("tests"):
        say(f"{rel}: an {c['kind']} contribution needs at least one pre-registered test")
    if c["kind"] == "interpret" and not c.get("label"):
        say(f"{rel}: an interpret contribution needs a 'label'")
    if c["kind"] == "withhold":
        from bonsai_interp.sensitive import mask
        if not c.get("label"):
            say(f"{rel}: a withhold contribution needs a masked 'label' (what kind of trigger, without the content)")
        elif mask(c["label"]) != c["label"] or mask(c.get("short") or "") != (c.get("short") or ""):
            say(f"{rel}: a withhold label must not contain the sensitive word itself; describe it masked")
        if (c.get("withheld") or {}).get("reason") not in WITHHELD_REASONS:
            say(f"{rel}: withheld.reason must be one of {sorted(WITHHELD_REASONS)}")
        if not (c.get("methods") or {}).get("agent"):
            say(f"{rel}: needs a 'methods' block with 'agent'")
    if c["kind"] == "coverage":
        marks, sample = c.get("marks"), c.get("sample")
        if not c.get("label_contribution"):
            say(f"{rel}: a coverage round needs 'label_contribution' (the id of the held label it measures)")
        if not isinstance(sample, list) or not sample or not isinstance(marks, list) or len(marks) != len(sample)                 or not all(isinstance(m, bool) for m in marks):
            say(f"{rel}: a coverage round needs 'sample' (the drawn activations) and one true/false 'marks' entry per sample")
        elif abs(c.get("coverage", -1) - sum(marks) / len(marks)) > 1e-3:
            say(f"{rel}: 'coverage' must equal the share of marks that are true")
        if not (c.get("methods") or {}).get("agent"):
            say(f"{rel}: needs a 'methods' block with 'agent'")
    if c["kind"] == "qc":
        q = c.get("qc") or {}
        if q.get("verdict") not in QC_VERDICTS:
            say(f"{rel}: qc.verdict must be one of {sorted(QC_VERDICTS)}")
        if q.get("verdict") == "refine" and not c.get("label"):
            say(f"{rel}: a 'refine' QC must propose a new 'label'")
    for i, t in enumerate(c.get("tests", [])):
        proto = LEGACY.get(t.get("protocol"), t.get("protocol"))
        if proto not in PROTOCOLS:
            say(f"{rel}: tests[{i}] protocol {t.get('protocol')!r} unknown (known: {sorted(PROTOCOLS)})")
        tf = t.get("tests_file")
        if not tf:
            say(f"{rel}: tests[{i}] needs 'tests_file' (the pre-registration)")
        elif not (ROOT / tf).exists():
            say(f"{rel}: tests[{i}].tests_file does not exist: {tf}")
    if len(c.get("notes", "")) > MAX_NOTES:
        say(f"{rel}: notes longer than {MAX_NOTES} characters; put long write-ups in the library")
    for e in c.get("examples", []):
        if len(e) > MAX_EXAMPLE:
            say(f"{rel}: corpus examples are capped at {MAX_EXAMPLE} characters each")
            break
    for a in c.get("artifacts", []):
        if not (a.get("sha256") and a.get("url") and a.get("name")):
            say(f"{rel}: each artifact needs name, url and sha256")
    for e in c.get("edges", []):
        if not (NEURON.match(e.get("from", "")) and NEURON.match(e.get("to", "")) and e.get("kind") and "weight" in e):
            say(f"{rel}: each edge needs from, to (neuron ids), kind and weight")
    return errs


def check_tests_file(path: Path):
    errs, rel = [], path.relative_to(ROOT).as_posix()
    if path.stat().st_size > MAX_FILE:
        errs.append(f"{rel}: larger than {MAX_FILE // 1024} KB")
    try:
        t = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"{rel}: invalid JSON ({e})"]
    if LOCAL_PATH.search(path.read_text(encoding="utf-8")):
        errs.append(f"{rel}: contains a local filesystem path")
    if not (t.get("neighbours") or t.get("sentences") or t.get("tests")):
        errs.append(f"{rel}: a tests file needs 'neighbours' (quick tests), 'sentences' (held-out) or 'tests'")
    return errs
