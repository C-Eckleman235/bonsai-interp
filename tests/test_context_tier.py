"""Model-free checks of the auto-token@2 machinery (bonsai_interp.auto.context_tier): seconds, no GPU, no weights.

  uv run python tests/test_context_tier.py      prints CONTEXT TIER OK
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bonsai_interp.auto import context_tier as ct  # noqa: E402


def planted(n=240, seed=0):
    """Occurrences whose activation depends on one feature, with the others random."""
    rng = np.random.default_rng(seed)
    f = {k: rng.choice(["a", "b", "c"], n).astype(object) for k in ct.FEATURES + ct.TABLE_ONLY}
    return f, rng


def test_word_classes():
    for s, c in [(" very", "intensifier"), (" the", "determiner"), ("\n", "newline"), (".", "sentence-end"), (" Paris", "capitalized"),
                 (" running", "verb-ing"), ("ing", "subword"), ("(", "open-bracket"), (" 3", "number"), (" 東京", "non-latin")]:
        assert ct._word_class(s) == c, (s, ct._word_class(s), c)


def test_context_rule():
    f, rng = planted()
    n = len(f["line_pos"])
    f["prev_class"] = rng.choice(["intensifier", "determiner", "word"], n).astype(object)
    frac = np.where(f["prev_class"] == "intensifier", 0.6, 0.02) + rng.normal(0, 0.03, n)
    res, rule = ct.fit(f, frac)
    assert res["bin"] == "context" and rule["conditions"][0] == ["prev_class", "intensifier"], res


def test_format_and_positional():
    f, rng = planted(seed=1)
    f["line_pos"] = rng.choice(["line-start", "16+"], len(f["line_pos"])).astype(object)
    frac = np.where(f["line_pos"] == "line-start", 0.5, 0.0) + rng.normal(0, 0.02, len(f["line_pos"]))
    assert ct.fit(f, frac)[0]["bin"] == "format"
    f["win_pos"] = f["line_pos"].copy()
    f["line_pos"] = rng.choice(["a", "b"], len(f["line_pos"])).astype(object)
    assert ct.fit(f, frac)[0]["bin"] == "positional"


def test_lexical_unexplained_rare():
    f, rng = planted(seed=2)
    n = len(f["line_pos"])
    assert ct.fit(f, 0.4 + rng.normal(0, 0.05, n))[0]["bin"] == "lexical"
    assert ct.fit(f, np.where(rng.random(n) < 0.1, 0.5, 0.0))[0]["bin"] == "unexplained"
    small = {k: v[:10] for k, v in f.items()}
    assert ct.fit(small, np.full(10, 0.5))[0]["bin"] == "rare"


def test_table_only_features_are_never_a_rule():
    for feat in ct.TABLE_ONLY:  # the following token (non-causal) and the corpus domain (not in a snippet)
        f, rng = planted(seed=3)
        f[feat] = rng.choice(["x", "y"], len(f["line_pos"])).astype(object)
        res, rule = ct.fit(f, np.where(f[feat] == "x", 0.6, 0.0))
        assert not rule or all(c[0] not in ct.TABLE_ONLY for c in rule["conditions"])
        assert res["features"][feat]["eta2"] > 0.9  # still tabulated


def test_rule_families_pick_the_bin():
    assert ct.bin_of({"conditions": [["tok_repeat", "new"]]}) == "context"  # first appearance is not copying
    for feat, v, b in [("tok_repeat", "repeat-near", "copy"), ("bigram_repeat", "bigram-repeat", "copy"), ("next_tok", "t5", "predictive"),
                       ("pmi", "pmi 5+", "collocation"), ("ctx_mean", "0.1+", "topic"), ("win_pos", "512+", "positional"),
                       ("prev_content", "w:eager", "context"), ("line_kind", "list-item", "format")]:
        assert ct.bin_of({"conditions": [[feat, v]]}) == b, feat
    assert ct.bin_of({"conditions": [["line_kind", "x"], ["tok_repeat", "repeat-far"]]}) == "copy"
    assert ct.bin_of({"conditions": [["prev_tok", "x"], ["ctx_mean", "y"]]}) == "topic"  # untestable wins
    assert set(ct.TESTED).isdisjoint(ct.UNTESTABLE)


def test_copy_rule_from_planted_repetition():
    f, rng = planted(seed=5)
    f["tok_repeat"] = rng.choice(["new", "repeat-near"], len(f["line_pos"])).astype(object)
    res, rule = ct.fit(f, np.where(f["tok_repeat"] == "repeat-near", 0.5, 0.0) + rng.normal(0, 0.02, len(f["line_pos"])))
    assert res["bin"] == "copy", res


def test_edit_distance_one():
    assert ct._edit1("rain", "rein") and ct._edit1("rain", "brain") and ct._edit1("train", "rain")
    assert not ct._edit1("rain", "rain") and not ct._edit1("rain", "tangent")


def test_sparse_counter():
    import torch
    acc = ct._Acc()
    rng = np.random.default_rng(4)
    keys = rng.integers(0, 50, 5000)
    for chunk in np.array_split(keys, 70):  # forces several merges
        acc.add(torch.from_numpy(chunk))
    acc.merge()
    u, c = np.unique(keys, return_counts=True)
    assert np.array_equal(acc.k, u) and np.array_equal(acc.c, c)


def test_splits_disjoint():
    assert ct.SELECT[1] <= ct.SCREEN[0] and ct.SCREEN[1] <= ct.TEST[0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("CONTEXT TIER OK")
