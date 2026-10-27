# sweeplib@4: part of speech and morphology

The model is a model of language, so word classes are a natural axis to sort its neurons by: are there "noun-like",
"verb-like" or "adjective-like" poles? sweeplib@4 asks that of every pole at once, with the grammar pass's engine
(docs/GRAMMAR.md) and the same spaCy parses.

Code: `bonsai_interp/grammar/posmorph.py` (conditions and batteries), `bonsai_interp/grammar/run4.py` (stages).
Protocol: sweeplib@4 (graph/auto/bonsai1.sweeplib4.*). Data: the `sweeplib4-data` release.

## The conditions (32)

Each is read at a snippet's last token, which ends a whole alphabetic word.

| family | holds when the last word… | quiet snippet | pairs keyed by |
|---|---|---|---|
| `class:X` (12) | is part of speech X (NOUN, PROPN, VERB, ADJ, ADV, PRON, DET, ADP, AUX, CCONJ, SCONJ, NUM) | a word of another class | frequency band |
| `same:X` (12) | is used as X | the same token used as another class | token |
| `morph:M` (8) | has the feature: plural, past, third_sg, gerund, participle, comparative, superlative, proper (noun) | the same class without it | frequency band |

Frequency bands are 12 log-spaced bands of the token's corpus count. They stop a class pole from passing merely by
preferring common or rare words. The reader, selection rule (sweeplib@3's), placebo pairs and tests, and judge
(quick-8+8@2 over 3 preambles) are unchanged.

## Results (2026-10-27)

| | |
|---|---|
| Protocol | 575d174a |
| Tests pre-registered | 2fe5ded2 |
| Results | 25a86b4a |
| Selection time | 16 min on the GPU |
| Scoring time | 0.5 min on the GPU |

| family | selected (= tests) | held | placebo tests held |
|---|---:|---:|---:|
| class | 122 | 5 | 0 |
| same | 1 | 0 | 0 |
| morph | 190 | 21 | 0 |
| **all** | **313** | **26 poles** | **0 of 313** |

- **By condition:**
  - superlative 10 of 105;
  - proper noun 3 of 8;
  - comparative 3 of 19;
  - gerund 3 of 16;
  - third person singular 2 of 12;
  - number 2 of 74;
  - verb, auxiliary and subordinating conjunction 1 each.
- **By depth:** 20 of the 26 are in layers 0–5. The rest:
  - superlative poles at L18, L19, L22 and L24;
  - a number pole at L24;
  - a superlative pole at L6.
- **Coverage:** 11 of the 26 cover at least half of their pole's 32 strongest activations, so they are
  assignment-grade. Among them:
  - L0/N5888+ (superlative) and L0/N11965− (comparative), each at 100%;
  - L3/N11719− (auxiliary verb), 100%;
  - L18/N11306+ (superlative), 62% against a 0.3% base rate.
- **`same:` selects almost nothing.** Of the 12 "same word, other class" conditions, only one pole was selected (and
  it failed). With the word held fixed, no pole separates nouns from verbs at the selection bar.

## Reading

Whole-class poles are rare. Where they exist, they sit early and track surface morphology: the *-est* and *-er*
forms, *-ing*, and *-s* on verbs. Those classes are marked in the letters, so an early layer can read them off the
token. Abstract classes (noun, adjective, adverb, pronoun) selected few poles and held almost none. Once the frequency
band is matched, the model does not dedicate single poles to "noun" or "verb".

Two readings fit, and this pass cannot tell them apart:
- the class is carried by many poles, each partial (a population code);
- it is carried in combination with the word, a conjunction that a class-keyed battery cannot see.

Composition round 3 (docs/COMPOSE.md) tests the second reading directly. It asks whether a pole's own strong tokens
fire only in a given class or role.

**FINDINGS paragraph.** sweeplib@4 screened all 884,736 poles for part of speech and morphology:
- 12 word classes, each against other classes of the same word frequency;
- the same word in two classes;
- 8 morphological features.

313 poles were selected and 26 held (0 of 313 placebo tests), 11 of them assignment-grade. The holds are morphological
and early: superlative, comparative, gerund and third person singular, 20 of them in layers 0–5. A few superlative
poles appear at L18–24. No pole held as a pure noun, adjective, adverb or pronoun detector once the frequency band was
matched. With the word held fixed, no pole was selected for the class it is used in.
