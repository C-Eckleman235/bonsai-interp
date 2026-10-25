# Toward a deterministic characterization harness

The goal is a harness that takes a new model and runs until every MLP pole has either a tested label or a recorded
reason why none fits, with no language model in the loop. Search and steering then run on the result. This document
covers three things:
- what the passes share today;
- how each pass should seed the next;
- the kinds to add, including ideas absorbed from similar projects and from the Twitch loop.

## What passes share today (2026-10-25)

| from → to | shared? | how |
|---|---|---|
| scan → word screen (auto-token@2) | yes | the word screen's stage 1 reads the scan's pole max and its dense, positional and weak flags |
| word screen stage 1 → stage 2 → tests | yes | word hits seed the context screen, and the fitted rule seeds the test |
| word screen → sweep library (sweeplib@1) | **no** | the library screened every pole from scratch. It only uses the word screen at display time (a library result colours a pole the screen left unexplained) |
| scan + weights → predict | yes | top examples and the direct logit effect select the candidates |
| sweep library → word screen (a second round) | **no** | nothing has been re-run with what the library learned |
| survey labels (people, Twitch) → screens | **by hand only** | the copy battery was rewritten after reading the zoo's copy labels, and the validator checks recall against held labels |
| screens → Twitch agent | partly | `play card` showed the word screen. With #964 it shows the library and adds `play search`. The stream's own loop does not read either yet |

So the passes are mostly side by side, not stacked.

## The loop: each pass seeds the next

```
  weights census ──► scan ──► word screen ──► twin batteries ──► kind batteries ──► composition ──► residual mining ─┐
     (no text)       (top-k,   (word hits,     (seeded by each     (pole-agnostic:    (labels flow     (features shared │
                     density)   context rule)   word hit: case,    copy, lang,        along weight     by the top        │
                                                split, slot,       number, syntax,    links: B reads   examples of the   │
                                                sense, tense)      genre, predict)    what A writes)   still-unexplained)│
     ▲                                                                                                                  │
     └──────────── round 2: new conditions and kinds, re-registered as a new protocol version ◄────────────────────────┘
```

Rules that keep it deterministic and honest:
- **Pre-registration per round.** Each round is a protocol version. Its selection rule and tests are committed before
  anything is read on the test split.
- **Placebos everywhere.** Every condition has a quiet-vs-quiet placebo, and every pole's test has a placebo test. In
  sweeplib@1 placebos selected 0 poles and held 0 of 333 tests, so the rules can afford to be loose at selection.
- **A pole leaves the residual** when a test of it holds. Later rounds spend their GPU time on what is left, but a held
  pole can still be retested by a sharper kind.
- **Coverage is the stopping rule.** Each round reports the share of non-weak poles with a held label, by layer. The
  harness stops when a round adds less than a set amount of coverage.

## The kinds to add, in order of expected yield

1. **Twin batteries seeded by word hits.**
   - What it is: the Twitch playbook's most-used method (the "one-swap twin", 69 uses) made deterministic. For each held
     or high-precision word hit, generate the same word across fixed axes, then let the test decide which axis the
     pole cares about:
     - case and leading space;
     - whole token vs split piece;
     - sentence-initial vs mid-sentence;
     - code slot (string, comment, docstring, identifier, prose);
     - inflection family;
     - sense (when a parser can tell senses apart).
   - Why first: it leverages pass 1 directly. In a random 50 of the Twitch loop's 339 held labels, about 20 were "the word X
     in any case and sense" or "only where the letters form their own piece", and these axes decide exactly that.
2. **A weights-only census** (no corpus), run first on a new model:
   - direct logit effect (predict and suppression sets, as sweeplib@1's predict);
   - entropy neurons: a large output norm in the unembedding's effective null space;
   - token-frequency neurons: output aligned with the unigram log-frequency direction;
   - a steer lens: push each neuron ± and record the next-token Δlog-prob and KL on a few contexts;
   - a weight connection matrix between layers, P[B, A] = (W_gate_B·v_A)(W_up_B·v_A), giving each pole's strongest
     direct readers and writers.

   This is minutes of GPU time per layer. It gives every pole an output-side hypothesis before any text is read.
3. **A syntax kind**, from a deterministic parser (spaCy part of speech and dependency labels). It covers conditions
   such as "the same word in a different grammatical role": `so` as purpose vs degree, the particle of a separated
   phrasal verb, the predicate after a dummy "it". In a random 50 of the stream's held labels, 8 were of this shape.
4. **Genre and boilerplate kinds**, from regex document classifiers: licence headers, citations (scripture, legal,
   academic), chemical formulas, standards codes, and code languages. The stream found several such poles by hand.
5. **Lookahead by class.** Extend predict from token sets to classes: "the next token is a digit, a year, a unit, a
   closing bracket". Build the tests from the model's own next-token class probability, as predict does.
6. **Composition.** For a still-unexplained pole, take the labels of its strongest weight-linked upstream poles.
   Propose the conjunction or difference of their conditions (for example "German" and "number" give "numbers in
   German text"), and test it with the batteries that already exist. This is how a second full cycle learns what the
   first could not.
7. **Residual mining.** For poles still unexplained, run the feature extractor (word class, script, language, line
   kind, code state, parser labels, upstream labels) over their top and middle examples against matched controls. Turn
   the best-separating conjunction into a candidate condition for the next round. auto-token@2's context screen does
   this for one word at a time; residual mining does it for the pole.

## What to absorb from similar work

- **lyra's neuron-explorer** ([github.com/lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer),
  Qwen3.5-4B). It characterizes the *output* side: every neuron is pushed in steered rollouts and ranked against
  random-direction controls. It also has a steer lens covering all layers, at 111 s per layer. Lessons we can take:
  - Doses in units of the residual norm, calibrated per layer (their first layers broke at a dose that was gentle
    elsewhere).
  - A random-direction null, the counterpart of our placebo.
  - Weight-only connection links, precomputed once.
  - A depth story to compare against ours: topics early, discourse in the middle, lexical families late.
  - Steering "recipes" (for example `L19N7839−0.7 and L17N9209+0.5`) as the user-facing way to influence behaviour.
    That is what search over a characterized atlas should feed.
- **Universal neurons** (Gurnee et al., [arXiv 2401.12181](https://arxiv.org/pdf/2401.12181)) and **confidence
  regulation neurons** (Stolfo et al., [arXiv 2406.16254](https://arxiv.org/pdf/2406.16254)). These give a taxonomy of
  neuron families with weight-based signatures: prediction, suppression, entropy, token frequency, attention
  deactivation, position, previous token. Each family is a census kind.
- **LLM-written explanations** (Bills et al.; [openai/automated-interpretability](https://github.com/openai/automated-interpretability),
  archived 2026; [Neuronpedia](https://www.neuronpedia.org/)). These are the non-deterministic baseline the harness
  replaces. Their simulate-and-score idea maps onto our held-out 8 + 8.

## From atlas to search and steering

Once most poles have a held condition, the atlas is an index in both directions:
- **condition → poles.** `play search --kind lang --condition lang/de --held` lists German-text gates.
- **pole → conditions.** `play card` shows a pole's held conditions.

Steering a behaviour then means choosing the poles whose held conditions name it and pushing them with calibrated
doses. A steer-lens check confirms what the push writes before the push is used.

## For a new model

The harness needs four adapters:
1. an MLP hook that records and can replace activations;
2. the down-projection columns, the final norm and the unembedding;
3. a tokenizer with byte-level decoding;
4. a corpus with held-out splits.

Everything else, from the census through the scan, screens, batteries, placebos, judging and atlas, is
model-agnostic. Costs scale with depth by the measured model: ms per scored sequence = 0.309 + 0.713·(L+1). The
reasons the 8B sweep was slow are recorded in docs/SWEEP_LEARNINGS.md, so a re-sweep can avoid them.
