# Composition: explaining a pole by the units that feed it

This is the recursive step of the deterministic harness (docs/HARNESS.md): a pole is characterized by
already-characterized units at the token offsets that drive its peaks. A round that succeeds adds new characterized
units, and the next round, one depth band up, can compose from them.

Code: `bonsai_interp/compose/`
- `feeders.py`: position-resolved attribution, ablation and token occlusion
- `pilot.py`, `pilot2.py`
- `match.py`: upstream units covering a token class
- `round.py`: the registered round

Protocol: compose@1 (graph/auto/bonsai1.compose1.*). Data: the `compose-data` release.

## Why tokens at offsets (pilot, FINDINGS 2026-10-26)

Zeroing a pole's 10 most attributed upstream neurons removes about half of a peak. Replacing a single token removes far
more:
- the read token: about 99%;
- the previous token: 47–87%, depending on depth;
- tokens 10–31 back: about 1%.

So a pole's peak is mostly a local function of its read token and the one before it. Each pole's top 32 peaks span
22–24 distinct read tokens, which makes them classes, not fixed bigrams. Attention enters as "which earlier positions
the pole needs", which occlusion measures directly.

## Round 1 (compose@1, layers 6–11)

**Hypotheses** (CPU, every residual eligible pole of layers 6–11):
- For offsets 0 and 1, take the tokens at the pole's 32 peaks and cover them with a greedy union of up to 3 upstream
  units. A unit qualifies when its focus is at least 0.3. Each layer's sparse token tables are matched against every
  target in one product.
- Units must already be **characterized**: a held survey label, or a held word-screen or sweep-library test. This is
  the sudoku rule.
- **pair:** both offsets covered on ≥ 60% of the peaks, and both together on ≥ 50%.
- **class:** the read token covered on ≥ 70% of the peaks, with at least 3 distinct tokens.

This gave 29,105 hypotheses (15,145 pair, 13,960 class); the best 6,000 were kept.

**Tests** (held-out windows 16,400 on, 8 + 8, quick-8+8@2 at the last token over 3 preambles):
- **pair:** fire when class A follows class B. The quiet half is A without B (the same read word); the other half is B
  without A (a read word of the same word class).
- **class:** fire on the class's words; quiet on the same word class outside it.
- **Classes:** A is the upstream units' trigger words that the target itself fires on (selection windows). B is the
  previous-offset units' trigger words.
- **Placebo:** every 10th hypothesis also gets an 8 + 8 test of quiet against quiet (199 had enough held-out items).

| | |
|---|---|
| Protocol | 6c3cc6c0 |
| Tests pre-registered | d03a6f39 |
| Tests built | 3,940 (2,060 hypotheses had too few held-out items) |

**Results:**

| | tests | held | held with coverage ≥ 50% |
|---|---:|---:|---:|
| pair | 1,690 | 405 (24%) | 270 |
| class | 2,250 | 101 (4.5%) | 64 |
| **all** | 3,940 | **506 poles** | **334** |
| placebo | 199 | **0** | |

- **By layer** (held): L6 263, L7 113, L8 90, L9 18, L10 13, L11 9.
- **Coverage** of held compositions (quartiles): 0.45 / 0.61 / 0.81. Coverage is the share of the target's strong
  activations on its class-A tokens. It is exact for class labels and an upper bound for pairs.

Examples (pole, coverage):

| pole | coverage | composition |
|---|---:|---|
| L8/N11199− | 0.72 | ` (` right after ` =` or ` return` (a call or tuple being built) |
| L6/N6274+ | 0.61 | ` day`, ` days`, `日` right after a digit |
| L6/N12084+ | 0.60 | ` le`, ` me`, ` les` right after Spanish ` se` |
| L6/N2951+ | 0.83 | ` of` right after ` ahead` |
| L7/N7774− | 0.58 | ` mean` right after `I` |
| L6/N10015+ | 0.72 | ` in` right after fill / filled / filling |
| L11/N6932− | 0.66 | ` raise` right after a code indent |
| L8/N1934+ | 0.53 | Arabic relative pronouns (التي, الذي) |

## Reading

- **The snowball turns, but slowly with depth.** Composing from known word units explains a quarter of the pair
  hypotheses it tests. 334 new assignment-grade poles are concentrated in layers 6–8, where the word units it builds
  on are densest.
- **Deeper layers need deeper units.** Hold rates fall from L9, where the inputs are less often word units. A second
  round, composing layers 12 and up from round 1's own holds, is the recursive test.
- **Class unions mostly fail** (4.5%): a union of a few word units rarely is the whole story of a pole.
- **The gaps are informative.** A hypothesis whose second offset has no covering unit (the "making verb" example in
  FINDINGS) names a unit kind to add.

Next: round 2 (layers 12–17) with round 1's holds as units, the attention distance sweep on held pairs, and per-round
coverage gain as the stopping rule.
