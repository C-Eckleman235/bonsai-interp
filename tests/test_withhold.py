"""Sensitive triggers: the denylist mask, the `withhold` contribution check and how the fold shows a withheld pole.

  uv run python -m pytest tests/test_withhold.py      (no model, no GPU)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bonsai_interp import sensitive  # noqa: E402
from bonsai_interp.graph import contrib, fold  # noqa: E402

# the denylist is hashes only; a test word is added for the duration of the tests
WORD = "zzqxslurtestword"


def setup_module():
    sensitive.HASHES = sensitive.HASHES | {sensitive._h(WORD)}


def test_mask_words_and_tokens():
    assert sensitive.mask(f"the {WORD} piece") == "the [masked] piece"
    assert sensitive.mask(f"{WORD.upper()}!") == "[masked]!"
    assert sensitive.mask_token(" " + WORD) == " [masked]"
    assert sensitive.mask_token(" late") == " late"
    assert sensitive.masked({"a": [f"x {WORD}"], "b": 3}) == {"a": ["x [masked]"], "b": 3}
    assert "zzqx" not in Path(sensitive.__file__).read_text(encoding="utf-8")  # the module carries hashes, not words


def _withhold(tmp, label, reason="slur"):
    d = tmp / "graph" / "contributions" / "bonsai1" / "L1_N7_pos"
    d.mkdir(parents=True, exist_ok=True)
    c = {"id": "2026-10-22-tester-abcd1234", "kind": "withhold", "neuron": "bonsai1/L1/N7+", "contributor": "tester",
         "created": "2026-10-22T20:00:00+00:00", "label": label, "short": "slur token (masked)",
         "withheld": {"reason": reason}, "methods": {"agent": "test", "screens": ["card"]}}
    p = d / f"{c['id']}.json"
    p.write_text(__import__("json").dumps(c), encoding="utf-8")
    return p, c


def test_check_rejects_unmasked_label(tmp_path, monkeypatch):
    monkeypatch.setattr(contrib, "ROOT", tmp_path)
    monkeypatch.setattr(contrib, "C", tmp_path / "graph" / "contributions")
    p, c = _withhold(tmp_path, f"the token ' {WORD}' in any context")
    assert any("must not contain the sensitive word" in e for e in contrib.check(p, c))
    p, c = _withhold(tmp_path, "a slur token (masked), any context", reason="rude")
    assert any("withheld.reason" in e for e in contrib.check(p, c))
    p, c = _withhold(tmp_path, "a slur token (masked), any context")
    assert contrib.check(p, c) == []


def test_fold_withheld_wins_and_masks_history():
    held = {"id": "2026-10-19-a-0001", "kind": "interpret", "neuron": "bonsai1/L1/N7+", "contributor": "a",
            "created": "2026-10-19T00:00:00+00:00", "label": f"the word {WORD} and friends", "short": f"{WORD}",
            "tests": [{"protocol": "quick-8+8@1", "n_fire": 8, "n_false_pos": 0, "tests_file": "x"}]}
    wh = {"id": "2026-10-22-b-0002", "kind": "withhold", "neuron": "bonsai1/L1/N7+", "contributor": "b",
          "created": "2026-10-22T00:00:00+00:00", "label": "a slur token (masked)", "short": "slur (masked)",
          "withheld": {"reason": "slur"}}
    r = fold.fold_neuron("bonsai1/L1/N7+", [held, wh])
    assert r["status"] == "withheld" and r["label"] == "a slur token (masked)" and r["short"] == "slur (masked)"
    assert r["withheld"] == {"reason": "slur", "contribution": "2026-10-22-b-0002"}
    assert all(WORD not in h[0] for h in fold._history(r))
