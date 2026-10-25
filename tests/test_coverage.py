"""Coverage: a held label is the pole's assignment only if it covers at least half of the pole's strong activations; below
that the pole is 'partial'. Also the coverage contribution check and the quoted-word trigger estimate.

  uv run python -m pytest tests/test_coverage.py      (no model, no GPU)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bonsai_interp.graph import contrib, coverage, fold  # noqa: E402

NID = "bonsai1/L3/N5+"


def _held(cid="2026-10-25-tester-aaaa1111"):
    t = {"protocol": "quick-8+8@2", "tests_file": "x.tests.json",
         "per_sentence": [{"set": "fire", "act": 0.6}] * 8 + [{"set": "quiet", "act": 0.0}] * 8}
    return {"id": cid, "kind": "interpret", "neuron": NID, "contributor": "tester", "created": "2026-10-25T10:00:00+00:00",
            "label": "the word 'zebra'", "tests": [t]}


def _round(marks, label_contribution="2026-10-25-tester-aaaa1111", cid="2026-10-25-tester-bbbb2222"):
    return {"id": cid, "kind": "coverage", "neuron": NID, "contributor": "tester", "created": "2026-10-25T11:00:00+00:00",
            "label": "the word 'zebra'", "label_contribution": label_contribution, "protocol": "coverage-16@1",
            "sample": [{"window": 16500 + i, "position": 80, "act": 0.4, "text": "x⟦y⟧"} for i in range(len(marks))],
            "marks": marks, "coverage": sum(marks) / len(marks), "methods": {"agent": "test", "screens": ["coverage-round"]}}


def test_held_without_coverage_stays_held(monkeypatch):
    monkeypatch.setattr(fold, "measured", lambda: {})
    r = fold.fold_neuron(NID, [_held()])
    assert r["status"] == "held" and r["coverage"] is None


def test_round_below_half_is_partial(monkeypatch):
    monkeypatch.setattr(fold, "measured", lambda: {})
    r = fold.fold_neuron(NID, [_held(), _round([True] * 7 + [False] * 9)])
    assert r["status"] == "partial" and r["coverage"]["value"] == 7 / 16


def test_round_at_half_is_held(monkeypatch):
    monkeypatch.setattr(fold, "measured", lambda: {})
    r = fold.fold_neuron(NID, [_held(), _round([True] * 8 + [False] * 8)])
    assert r["status"] == "held"


def test_round_on_an_older_label_does_not_count(monkeypatch):
    monkeypatch.setattr(fold, "measured", lambda: {})
    r = fold.fold_neuron(NID, [_held(), _round([False] * 16, label_contribution="2026-10-01-tester-old00000")])
    assert r["status"] == "held" and r["coverage"] is None


def test_declared_token_measure_wins_over_a_round(monkeypatch):
    monkeypatch.setattr(fold, "measured", lambda: {NID: {"label_contribution": "2026-10-25-tester-aaaa1111", "coverage": 0.8,
                                                         "strong": 500, "method": "declared trigger"}})
    r = fold.fold_neuron(NID, [_held(), _round([False] * 16)])
    assert r["status"] == "held" and r["coverage"]["value"] == 0.8


def test_estimate_yields_to_a_round(monkeypatch):
    monkeypatch.setattr(fold, "measured", lambda: {NID: {"label_contribution": "2026-10-25-tester-aaaa1111", "coverage": 0.02,
                                                         "strong": 500, "method": "quoted terms of a word-shaped label"}})
    assert fold.fold_neuron(NID, [_held()])["status"] == "partial"
    assert fold.fold_neuron(NID, [_held(), _round([True] * 12 + [False] * 4)])["status"] == "held"


def test_coverage_contribution_check(tmp_path, monkeypatch):
    monkeypatch.setattr(contrib, "ROOT", tmp_path)
    monkeypatch.setattr(contrib, "C", tmp_path / "graph" / "contributions")
    c = _round([True] * 4 + [False] * 12)
    d = contrib.folder_for(NID)
    d.mkdir(parents=True)
    p = d / f"{c['id']}.json"
    p.write_text(__import__("json").dumps(c), encoding="utf-8")
    assert contrib.check(p, c) == []
    bad = {**c, "coverage": 0.9}
    assert any("share of marks" in e for e in contrib.check(p, bad))
    assert any("one true/false" in e for e in contrib.check(p, {**c, "marks": c["marks"][:3]}))


def test_legacy_trigger_only_for_word_shaped_labels():
    assert coverage.legacy_trigger("the word 'cobra' / ' Cobra' (snake)", None)["words"] == ["Cobra", "cobra"]
    assert coverage.legacy_trigger("words that express fondness ('fan', 'love')", "meaning") is None
    assert coverage.legacy_trigger("the word ' so' joining clauses ('so', ',', 'Well, that')", None)["words"] == ["so"]
