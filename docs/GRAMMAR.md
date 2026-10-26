# Grammar first pass: generic construction screens (sweeplib@3)

The grammar pass asks, for every pole at once, whether it tracks a **grammatical role** of the word it is reading:
the object of a preposition, a subject, a word after an intensifier, a passive participle, and so on. It is the sweep
library's method (docs/SWEEPLIB.md) applied to a dependency parse. No language model writes or judges anything.

Code: `bonsai_interp/grammar/` (parse.py, cons.py, run.py). It imports the sweep library's reader, pair sampler,
statistics and judge unchanged. Protocol: `sweeplib@3` (graph/auto/bonsai1.sweeplib3.protocol.json).

## Method

1. **Parse.** spaCy `en_core_web_sm` (the optional `grammar` extra: `uv sync --extra grammar`) parses every English
   window (domains edu and web) of the selection split (windows 0–15,999: 11,166 windows) and, only after the protocol
   was committed, of the test split (16,400 onwards). Each 1,024-token window is parsed whole. Every model token gets the
   attributes of the spaCy word that holds its first non-space byte (POS, tag, dependency label, the model position of its
   head word's last piece, whether it ends the word, whether its sentence ends with "?"). Stored in
   `data/grammar/bonsai1.parse.<split>.npz`.
2. **Conditions** (cons.py). A condition is a predicate on the parse at a 48-token snippet's last token. The last token
   must be a whole alphabetic word (or a possessive `'s`). The quiet snippet comes from another window and ends on the
   **same token** in a different grammatical role, so a pole that only knows the word cannot separate the pair. Quiet
   snippets are balanced over strata (the quiet word's role: subject, object, modifier, clause, function word, …).

   | condition | fire | quiet (same word) | cue |
   |---|---|---|---|
   | pobj | object of a preposition | other roles | left |
   | prep | preposition linking two content words | not a preposition (particle, infinitival *to*, …) | right |
   | verb_dobj | verb with a direct object | the verb with no object | right |
   | nsubj | subject | other roles | right |
   | dobj | direct object | other roles | left |
   | intensified | adjective/adverb after *very, so, too, really, …* | no modifier | left |
   | amod | adjective modifying a noun | predicative or other | right |
   | amod_head | noun after an adjective | bare noun | left |
   | advmod_verb | adverb modifying a verb | adverb modifying an adjective/adverb | right |
   | det_noun | noun with a determiner | bare noun | left |
   | poss | possessor (*her book*, *John's*) | other roles | right |
   | poss_s | possessive `'s` | `'s` = *is* / *has* | right |
   | conj2 | second conjunct of *X and/or Y* | other roles | left |
   | aux_verb | verb after an auxiliary | no auxiliary | left |
   | negated | verb after *not / never* | not negated | left |
   | than | the compared term after *than* | other roles | left |
   | relcl_start | relative pronoun (*the book which*) | the same word as question word, determiner, complementizer | left |
   | particle | phrasal-verb particle (*give up*) | the same word as preposition or adverb | right |
   | passive | passive participle (*was taken*) | perfect (*has taken*) or other participle | left |
   | wh_question | question word in a direct question | the same word elsewhere | right |

   "cue" says where the evidence is. *Left*: in the words already read. *Right*: mostly after the last token, so a pole
   selected there anticipates the construction.

   15 conditions also have a **word-class variant**, `pos:<condition>`. Its pair key is the last word's part of speech
   instead of the token, so a pole selected there follows the role whatever the word.
3. **Selection** on 64 pairs (selection split), with placebo pairs (two quiet snippets with the same key) run through the
   same rule. One all-layer read of each battery scores all 884,736 poles.
4. **Test.** Each selected pole gets its condition's held-out 8 + 8 (test split), registered before any of it is read, and
   a placebo 8 + 8 (quiet against quiet). Judged by quick-8+8@2 at the last token over the 3 preambles, as sweeplib@1.
5. **Report.** Held / failed / placebo by construction, by layer band, and by the pole's earlier word-screen bin (the
   atlas `bin` column). For held conditions, **coverage**: the share of the pole's 32 strongest corpus activations (scan
   top examples, English windows) where the condition holds at that token, against the condition's base rate over all
   parsed words.

## Pilot (selection split, 2026-10-25)

Every condition and variant was read on its 64 selection pairs and 64 placebo pairs (43 min of wall time beside the live
stream, about 20 min of it reading). Poles selected / placebo selections:

| rule | selected | placebo |
|---|---:|---:|
| sweeplib@1's rule | 73 | 0 |
| stricter (AUROC ≥ 0.90, median ≥ 0.20, mean diff ≥ 0.15) | 12 | 0 |
| looser (AUROC ≥ 0.80, median ≥ 0.10, win ≥ 0.70, mean diff ≥ 0.07) | 325 | 0 |

Placebo pairs selected nothing under any rule, so the looser rule was registered. The held-out test keeps the judge's
bar, and placebo tests count false holds.

The pilot already showed the main result: **the core argument roles are not carried by single poles at the last token.**
Subject, direct object, object of a preposition, preposition, determiner + noun, possessor, *than*, relative pronoun and
question word selected 0 poles even under the loose rule. Even without a magnitude bar, almost no pole separates them at
AUROC ≥ 0.85 (pobj 0, dobj 0, nsubj 1, prep 0, poss 0 of 884,736). The selections came from locally marked
constructions: possessive `'s` vs the contraction (179), passive participles (18, word-class 21), negation (12,
word-class 26), intensifiers (5, word-class 20) and adjectives in the word-class variant (24).

## Results: sweeplib@3 (2026-10-26)

| | |
|---|---|
| Protocol commit | 1b6a2d6e |
| Tests pre-registered | 184d4ed7 |
| Artifacts | the `sweeplib3-data` release |

326 tests were built from the selected (pole, condition) pairs: 19 conditions with test items, each with a placebo test
on the same poles.

| | tests | held |
|---|---:|---:|
| all conditions | 326 | **30 poles** |
| possessive 's (vs the contraction of is/has) | 179 | 22 |
| negation, word-class variant | 26 | 3 |
| X and Y (second conjunct) | 3 | 1 |
| passive, intensified, modified adjective (word class) | 84 | 1 each |
| placebo tests | 326 | **0** |

- **By depth.** 21 of the 30 holds are in layers 0–5, 6 in layers 18–23 and 3 in layers 30–35. None are in layers
  6–17.
- **By earlier word-screen bin.** 16 holds are on poles the word screen called dense and 9 on unexplained ones. The
  grammar pass adds labels where the word screen had none.
- **Coverage.** The held conditions cover a median 3% of their poles' top activations. So they are true but partial
  facts (sub-triggers), not assignments.

**Reading.** Generic grammatical roles are not a first-pass explainer of single neurons in this model. Read at the last
token, the core argument roles (subject, object, preposition, determiner, possessor) separate essentially no pole from
the same word in another role. Only locally marked constructions do. The possessive 's is the clear case: an early
layer can tell "John's" from "John's going" by the next-to-local context. Grammar is better used as an ingredient of
composition (a role at an offset) than as a screen of its own.

**FINDINGS paragraph.** A grammar first pass over all 884,736 poles (20 spaCy roles, plus word-class variants, the same
word in another role as the near miss) held for 30 poles, against 0 of 326 placebo tests. 22 of the holds were the
possessive 's against its contraction. The core argument roles (subject, object, preposition object, determiner,
possessor) selected no pole at all. Held conditions cover a median 3% of their poles, so grammar marks sub-triggers, not
assignments, at the single-neuron level.

