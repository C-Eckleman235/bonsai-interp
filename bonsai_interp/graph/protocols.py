"""Versioned test protocols: how a test's numbers become an outcome.

A contribution names the protocol of each test ("quick-8+8@1"). Old contributions keep the rule
they were made under; a new pipeline adds a new version here instead of changing an old one.
Tests under a protocol this code does not know are kept and shown, but do not change a status.
"""

def _quick_8_8_v1(t):
    return "held" if t["n_fire"] >= 5 and t["n_false_pos"] <= 1 else "failed"


def _held_out_v1(t):
    # a held-out set passes when its predicted-fire sentences fire at >= 0.5 and its predicted-quiet ones at <= 0.2
    return "held" if t.get("fire_rate_predicted_fire", 0) >= 0.5 and t.get("fire_rate_predicted_quiet", 1) <= 0.2 else "failed"


def _cleanroom_v1(t):
    # a blind replication "agrees" when the reviewer marks the blind label as matching; outcome given explicitly
    return t.get("outcome", "recorded")


PROTOCOLS = {
    "quick-8+8@1": {"rule": _quick_8_8_v1, "required": ["n_fire", "n_false_pos", "n_written"],
                    "about": "8 sentences that should fire and 8 near misses; held if >= 5 fire at >= 0.3 of the pole's corpus max and <= 1 near miss does"},
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
