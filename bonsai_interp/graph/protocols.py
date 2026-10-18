"""Versioned test protocols: how a test's numbers become an outcome.

A contribution names the protocol of each test ("quick-8+8@1"). Old contributions keep the rule
they were made under; a new pipeline adds a new version here instead of changing an old one.
Tests under a protocol this code does not know are kept and shown, but do not change a status.
"""

def _quick_8_8_v1(t):
    return "held" if t["n_fire"] >= 5 and t["n_false_pos"] <= 1 else "failed"


def _quick_8_8_v2(t):
    """Same 8 + 8 test as @1, judged on separation instead of a hard count at exactly 0.30.

    held when all three hold:
      - the fire sentences rank above the near misses: AUROC >= 0.90 (at most 6 of the 64 fire/near-miss
        pairs inverted), so the label separates what it claims from what it doesn't;
      - the neuron responds for real: the median fire sentence reaches >= 0.20 of the pole's corpus max;
      - near misses stay clean: at most 1 reaches 0.30.
    @1 failed labels on rounding noise (0.295 and 0.298 against a 0.30 bar); @2 asks the questions the
    bar was a proxy for. Needs per-sentence activations, which `play run` records."""
    fire = [r["act"] for r in t["per_sentence"] if r["set"] == "fire"]
    quiet = [r["act"] for r in t["per_sentence"] if r["set"] == "quiet"]
    if not fire or not quiet:
        return "invalid"
    wins = sum((f > q) + 0.5 * (f == q) for f in fire for q in quiet)
    auroc = wins / (len(fire) * len(quiet))
    med = sorted(fire)[len(fire) // 2] if len(fire) % 2 else sum(sorted(fire)[len(fire) // 2 - 1: len(fire) // 2 + 1]) / 2
    return "held" if auroc >= 0.90 and med >= 0.20 and sum(q >= 0.30 for q in quiet) <= 1 else "failed"


def _held_out_v1(t):
    # a held-out set passes when its predicted-fire sentences fire at >= 0.5 and its predicted-quiet ones at <= 0.2
    return "held" if t.get("fire_rate_predicted_fire", 0) >= 0.5 and t.get("fire_rate_predicted_quiet", 1) <= 0.2 else "failed"


def _cleanroom_v1(t):
    # a blind replication "agrees" when the reviewer marks the blind label as matching; outcome given explicitly
    return t.get("outcome", "recorded")


PROTOCOLS = {
    "quick-8+8@1": {"rule": _quick_8_8_v1, "required": ["n_fire", "n_false_pos", "n_written"],
                    "about": "8 sentences that should fire and 8 near misses; held if >= 5 fire at >= 0.3 of the pole's corpus max and <= 1 near miss does"},
    "quick-8+8@2": {"rule": _quick_8_8_v2, "required": ["per_sentence"],
                    "about": "8 + 8 test judged on separation: AUROC >= 0.90, median fire >= 0.20 of max, at most 1 near miss >= 0.30"},
    "held-out@1": {"rule": _held_out_v1, "required": ["fire_rate_predicted_fire", "fire_rate_predicted_quiet"],
                   "about": "pre-registered held-out sentences with fire/quiet predictions, scored by fire rate at >= 0.3"},
    "cleanroom@1": {"rule": _cleanroom_v1, "required": ["outcome"],
                    "about": "a blind clean-room run (neuron-cleanroom); outcome agrees / partly / disagrees, judged by the reviewer"},
}
LEGACY = {"quick-8+8": "quick-8+8@1", "held-out": "held-out@1"}


def outcome(test):
    p = PROTOCOLS.get(LEGACY.get(test.get("protocol") or test.get("type"), test.get("protocol") or test.get("type")))
    if not p:
        return "recorded"
    missing = [k for k in p["required"] if k not in test]
    return "invalid" if missing else p["rule"](test)
