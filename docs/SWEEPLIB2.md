# sweeplib@2: twin batteries seeded by word hits

This is the first deterministic pass that builds on an earlier one. The word screen (auto-token@2) found which words a
pole's strong activations fall on. For the 300 commonest seed words (279,441 seed pole-words), a twin battery then asks
which part of the word is the trigger, one axis at a time. These axes are the Twitch loop's most-used method, the
"one-swap twin", made automatic.

Code: `bonsai_interp/sweeplib2/` (`twins.py` builds the batteries, `run.py` runs the stages). Protocol: sweeplib@2
(graph/auto/bonsai1.sweeplib2.*, with amendment 1: a speed-only reorder of the coverage pass). Data: the
`sweeplib2-data` release.

## The axes (claims)

Each claim is a fire set against quiet sets that change one thing. Invariant claims say the twin fires too, against a
control word in the same slot and form. Selective claims say only the seed's own form or context fires.

| claim | holds when the pole fires on… |
|---|---|
| form/any-case | the word in any case (word, Word, WORD), not another word in that slot |
| form/this-case | only this case of the word |
| form/no-space | the word with or without its leading space |
| form/space-only | only the space-led token |
| piece/inside | the letters also as a split piece inside a longer word |
| piece/own | the word only as its own token |
| infl/family | the word and its inflections (-s, -ed, -ing) |
| infl/this-form | only this form |
| pos/initial, pos/mid | the word only at a sentence start, or only mid-sentence |
| slot/comment, slot/string, slot/code, slot/prose | the word only in a code comment, a string, code, or prose |

**Selection and test** reuse sweeplib@1's machinery:
- selection pairs come from windows 0–15,999 and tests from 16,400 on;
- reads are at the last token;
- the judge is quick-8+8@2 over 3 preambles;
- every claim has a placebo test on the same poles.

## Results (2026-10-26)

| | |
|---|---|
| Protocol | 019937f2 |
| Tests pre-registered | 41c258b1 |
| Amendment | 7b0f0d9f |
| Read time | 39 min on the GPU |

| axis | tests | held | placebo tests held |
|---|---:|---:|---:|
| form (case, leading space) | 24,774 | 13,301 | 2 |
| piece (own token vs inside a word) | 8,453 | 3,960 | 13 |
| infl (inflection family) | 2,396 | 1,281 | 0 |
| slot (comment, string, code, prose) | 3,466 | 1,196 | 2 |
| pos (sentence start vs mid) | 1,184 | 426 | 0 |
| **all** | **40,273** | **20,164 (on 10,715 poles)** | **20 of 40,273 (0.05%)** |

- **By claim.** The invariant claims dominate: any case (5,414 held), with or without the space (6,453), own token
  (3,394), inflection family (946). Most word-detector poles are case- and spacing-blind and fire on the whole word,
  not its letters inside other words.
- **By depth** (held / tests):

  | layers | held / tests |
  |---|---:|
  | 0–5 | 8,466 / 15,335 |
  | 6–11 | 1,241 / 3,079 |
  | 12–17 | 379 / 906 |
  | 18–23 | 3,216 / 5,719 |
  | 24–29 | 3,926 / 7,857 |
  | 30–35 | 2,936 / 7,377 |

- **Coverage of held claims.** The median is 0.22, and 4,570 held claims cover at least half of their pole's strong
  activations. Token-set claims are measured exactly; context claims are estimated, with the seed's token coverage as
  the upper bound.
- **Pass 2 sharpening pass 1.** 9,825 poles whose word-screen result was partial (the word covers under half the pole)
  gained a held twin claim. For **1,378** of them the sharper claim reaches assignment-grade.

**FINDINGS paragraph.** Twin batteries seeded by the word screen's hits are pass 2 building on pass 1. On 300 seed words
they tested 40,273 (pole, claim) pairs that each vary one axis of the word (case, leading space, own token vs piece,
inflection, sentence position, code slot). 20,164 held on 10,715 poles, against 20 of 40,273 placebo tests (0.05%). Most
word poles are case- and spacing-blind and fire on the whole word, not its letters inside other words. 9,825 poles with
a partial word result gained a sharper held claim, and 1,378 of those now cover at least half of their pole.
