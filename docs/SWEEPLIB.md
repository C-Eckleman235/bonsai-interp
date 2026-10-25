# The sweep library

The sweep library is a set of deterministic screens. Each screen looks for one *kind* of neuron, and no language model
writes or judges any part of it. It runs after the word screen (auto-token@2) and targets the poles that a word hit
does not explain: induction and copying, topic context, language and script gates, position in code or document
structure, number roles, and next-token prediction. Every screen is pre-registered, judged by the same rule as a human
test (quick-8+8@2), and checked against a placebo.

Code: `bonsai_interp/sweeplib/` (kinds.py, predict.py, run.py, validate.py). Protocol: `sweeplib@1`
(graph/auto/bonsai1.sweeplib1.protocol.json).

## How a battery screen works

A **kind** is a family of **conditions**. A condition is a predicate on a 48-token corpus snippet, evaluated at the
snippet's last token. For example, `copy/piece/code` holds when the last token is a word-internal piece whose bigram
already occurred earlier in the snippet.

1. **Battery.** The code samples about 650,000 snippet ends per split (edu, web, multilingual, code) and classifies each
   one. A condition's battery is a set of minimal pairs:
   - a *fire* snippet, where the condition holds;
   - a *quiet* snippet from a different window that **ends on the same token** (the pair key), where it does not.

   Quiet snippets come in named strata, for example English vs other Latin-script languages for `lang/fr`, and the
   sampler balances the strata. Selection pairs come from windows 0–15,999. Test snippets come from window 16,400
   onwards.
2. **One read for every pole.** A battery does not depend on the pole. The reader runs each snippet once, after a
   neutral preamble, and records all 36 layers × 12,288 neurons at the last token. That one pass scores all 884,736
   poles. A condition costs about 20 s of GPU time.
3. **Selection** (pre-registered; on 64 pairs):
   - AUROC ≥ 0.85;
   - median fire ≥ 0.15 of the pole's corpus max;
   - at most 15% of quiet snippets ≥ 0.30;
   - the fire snippet beats its own quiet in ≥ 75% of pairs;
   - mean paired difference ≥ 0.10.

   Weak poles (corpus max < 0.5) are excluded.
4. **Placebo.** The same rule runs on placebo pairs: two quiet snippets with the same key. The count it selects
   estimates false selections.
5. **Test.** Each selected pole gets its condition's 8 + 8 test from held-out windows, written and committed before any
   of it is read. Each tested condition also gets a placebo test (8 quiet against 8 quiet) on the same poles, which
   counts false holds. Judging follows auto-token@2: the read is at the last token, averaged over the 3 preambles, and
   the rule is quick-8+8@2 (AUROC ≥ 0.90, median fire ≥ 0.20, at most 1 near miss ≥ 0.30).

## predict: an inverse analysis of the output side

Prediction neurons are found from the weights, not from text:

- `W_U · (final-norm gain ⊙ W_down[:, n]) · sign` ranks the tokens each pole promotes.
- A pole is a candidate when at least 8 of its 32 strongest corpus activations are followed by one of its 20 most
  promoted tokens, at 10 times or more the base rate. The tested set is the promoted tokens actually seen there.
- The model's own next-token probabilities build the test. Fire snippets are those where the model puts ≥ 0.5 of its
  probability on the set. Quiet snippets are those where it puts ≤ 0.05 on the set, ending on the same word. So the
  near miss differs only in what the model expects next. No neuron is read before registration: building the test
  reads only logits.

## Running it

```bash
python -m bonsai_interp.sweeplib.run protocol      # then commit the manifest ALONE
python -m bonsai_interp.sweeplib.run select        # GPU, ~30 min beside the stream
python -m bonsai_interp.sweeplib.run build         # GPU (logits only); then commit the prereg manifest ALONE
python -m bonsai_interp.sweeplib.run score         # GPU
python -m bonsai_interp.sweeplib.validate data/sweeplib/bonsai1.sweeplib1.select.json.gz
```

`--smoke DIR` runs a small version with no manifests. `--smoke DIR --pilot` runs the full batteries and saves every
pole's statistics, for calibrating a rule offline before registering it. Every model call goes through gpu_turn at
background priority, and `data/auto2/STOP` stops a stage between batches.

## Adding a kind

1. Subclass `Kind` in kinds.py. Give it `name`, `about` and `labels` (one label per condition, the claim a held test
   makes). Write `classify(snip, vocab)`, which returns `{condition: "fire" | quiet stratum}`, and optionally
   `key(...)`, the pair key (by default, the last token).
2. Make the quiet side a near miss. It keeps the last token and changes only the claimed property. Add strata when
   "not X" has several kinds, such as English vs other languages.
3. Pilot it on the selection split (`--smoke DIR --pilot --kinds yours`). Check the placebo counts and the known held
   labels of that kind (validate.py). Then register a new protocol version. Changing the code changes the code
   sha256, so a new kind is always a new protocol, never an edit of a registered one.

## What the pilot showed (selection split, 2026-10-24)

- **Placebos.** Placebo pairs selected 0 poles in every condition, under the registered rule and under three stricter
  ones. So false selections are not the limit; effect size is.
- **Last-token contexts are weak.** Most poles barely respond to abstract context at the last token.
  - Topic-without-the-word and most structure conditions selected 0–1 poles even under the loose rule.
  - Under the registered rule, these selected the most poles: year 66, currency 40, Arabic script 25, German 19,
    Hangul 17, first digit of a number 10, Cyrillic 10.
- **The first copy battery asked the wrong question.** It tested "the word repeats earlier", and the zoo's copy neurons
  sit at AUROC 0.5 on that. Their held labels describe the *later pieces of a rare multi-piece string* in a second
  copy, so the kind was rewritten around word-internal pieces and repeated trigrams, which selected 17–28 poles per
  condition with 0 placebo.
- **The zoo's copy neurons are still not selected.** A probe built to match one zoo neuron's label also reads 0.035 on
  that neuron, so these neurons need longer or more specific contexts than a 48-token snippet gives. Zoo recall is a
  weak gate for this kind.
- **Batch-shape noise.** Reads in this 1-bit model move with the batch shape: up to 0.12 of a pole's max on a few of 35
  million cells (99.9th percentile 0.006). The cached reader is checked against that floor
  (tests/test_sweeplib_parity.py), not against zero.
