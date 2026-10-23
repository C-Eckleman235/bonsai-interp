"""The spot-replication verdict: exact, device drift, or a mismatch (no model needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bonsai_interp.graph.replicate import verdict  # noqa: E402


def test_verdicts():
    near = [0.002] * 16
    assert verdict(near, 16, True) == "ok"
    assert verdict(near[:15] + [0.101], 16, True) == "drift"          # PR #677: one sentence, outcome reproduced
    assert verdict(near[:12] + [0.053, 0.017, 0.013, 0.010], 16, True) == "drift"  # PR #746
    assert verdict(near[:15] + [0.101], 16, False) == "fail"         # outcome changed
    assert verdict(near[:13] + [0.05, 0.05, 0.05], 16, True) == "fail"  # three sentences off
    assert verdict(near[:15] + [0.2], 16, True) == "fail"            # too large
    assert verdict(near[:15], 16, True) == "fail"                    # a sentence missing
