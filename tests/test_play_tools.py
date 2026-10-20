"""Model-free checks of three contributor tools: seconds, no GPU, no weights.

- `play brief step|stepin --depth quick` drops the band scan; the default briefs keep their old shape
- `play new` / `play draw` record the optional `selection` field
- `scripts/colab_results.py` rebuilds a results file byte for byte, ignores CRLF, refuses a tampered row, and
  copies the same optional keys as `play run` (so the two cannot drift apart silently)

  uv run python tests/test_play_tools.py      prints PLAY TOOLS OK
"""
import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bonsai_interp.graph import play  # noqa: E402

spec = importlib.util.spec_from_file_location("colab_results", ROOT / "scripts" / "colab_results.py")
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)


def brief(*args):
    out = subprocess.run([sys.executable, "-m", "bonsai_interp.graph.play", "brief", *args, "--handle", "t"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return out


def test_step_depth_quick():
    for kind in ("step", "stepin"):
        default = brief(kind, "--neuron", "bonsai1/L10/N8330+")
        quick = brief(kind, "--neuron", "bonsai1/L10/N8330+", "--depth", "quick")
        assert "bands scan" in default and "bands sample" in default, kind
        assert "bands scan" not in quick and "bands sample" not in quick and "Depth: QUICK" in quick, kind
    assert "bands scan" not in brief("interpret", "--neuron", "bonsai1/L10/N8330+")  # interpret stays quick by default


def test_selection_field():
    with tempfile.TemporaryDirectory() as d:
        real = play.folder_for
        play.folder_for = lambda nid: Path(d) / real(nid).name
        try:
            base = dict(handle="t", kind="interpret", verdict="confirm")
            p = play._new_one(argparse.Namespace(**base, selection="concept search: chess"), "bonsai1/L3/N3+")
            assert json.loads(p.read_text(encoding="utf-8"))["selection"] == "concept search: chess"
            p = play._new_one(argparse.Namespace(**base), "bonsai1/L3/N4+")
            assert "selection" not in json.loads(p.read_text(encoding="utf-8"))
        finally:
            play.folder_for = real
    src = (ROOT / "bonsai_interp" / "graph" / "play.py").read_text(encoding="utf-8")
    assert 'a.selection = out, "interpret", "confirm", "random draw"' in src  # play draw writes it


def test_colab_results_round_trip():
    # the optional keys colab_results copies must be the ones play run copies, in the same order
    src = (ROOT / "bonsai_interp" / "graph" / "play.py").read_text(encoding="utf-8")
    run_keys = re.search(r'for k in \(("short"[^)]*)\):', src).group(1)
    assert tuple(k.strip().strip('"') for k in run_keys.split(",")) == cr.OPTIONAL, "play run and colab_results disagree"

    t = {"contribution": "2026-01-01-t-0000", "contributor": "t", "kind": "interpret", "neuron": "bonsai1/L3/N3+",
         "protocol": "quick-8+8@3", "neighbours": [{"unit": "L3/N3", "pole": "+", "label": "a label", "fire": [], "quiet": []}],
         "short": "s", "roles": ["feeds bonsai1/L4/N4+"], "selection": "random draw"}
    tests = [{"protocol": "quick-8+8@3", "per_sentence": [{"set": "fire", "i": 0, "act": 0.5, "peak": " x"}], "outcome": "held"}]
    methods = {"agent": "t", "screens": ["card"], "detected": [], "toolchain_commit": "abc1234", "minutes": 1, "batch_size": 1}
    rel_tests = "graph/contributions/bonsai1/L3_N3_pos/2026-01-01-t-0000.tests.json"
    with tempfile.TemporaryDirectory() as d:
        cr.ROOT = Path(d)
        (cr.ROOT / rel_tests).parent.mkdir(parents=True)
        (cr.ROOT / rel_tests).write_text(json.dumps(t), encoding="utf-8")
        data = cr.build(t, rel_tests, "2026-01-01T00:00:00+00:00", tests, methods)
        rel = rel_tests.replace(".tests.json", ".json")
        assert cr.digest(data) == cr.digest(data.replace(b"\n", b"\r\n"))  # CRLF checkouts hash the same
        row = [rel, cr.digest(data), "2026-01-01T00:00:00+00:00", tests, methods]
        rows = Path(d) / "rows.jsonl"
        rows.write_text(json.dumps(row) + "\n", encoding="utf-8")
        try:
            cr.rebuild(argparse.Namespace(rows=str(rows)))
        except SystemExit as e:
            assert e.code == 0
        assert (cr.ROOT / rel).read_bytes() == data
        (cr.ROOT / rel).unlink()
        row[3] = [{**tests[0], "outcome": "failed"}]  # a mistyped row must not be written
        rows.write_text(json.dumps(row) + "\n", encoding="utf-8")
        try:
            cr.rebuild(argparse.Namespace(rows=str(rows)))
        except SystemExit as e:
            assert e.code == 1
        assert not (cr.ROOT / rel).exists()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("PLAY TOOLS OK")
