# Contributing to the Bonsai interp graph

The graph is only worth something if its labels can fail. Every contribution follows one rule:

> **Write the test before you see its result, and commit it first.**

CI enforces the commit order. Reviewers and the [merge agent](MERGE_AGENT.md) enforce the spirit
of it: tests that could embarrass the label, not tests chosen to flatter it. What may be
committed, and what belongs in the library, is set out in [DATA_POLICY.md](DATA_POLICY.md).

The easiest way in is the **game page** (`site/index.html`). It draws a random unclaimed neuron
or one that needs QC, opens the claim, and gives you the exact commands and the right PR shape.
The steps below are what it walks you through.

## Setup (once)

```bash
git clone https://github.com/moridinamael/bonsai-interp && cd bonsai-interp
uv sync                          # Python 3.11-3.13; torch with CUDA 12.8
bash scripts/download_models.sh  # Bonsai-8B weights (about 16 GB unpacked)
bash scripts/fetch_data.sh       # corpus + full neuron scan (1.4 GB)
```

You need a GPU with at least 16 GB of memory. globalMOO is optional: put `GMOO_API_KEY` in
`.env`.

## Shape 1: interpret a neuron pole

1. **Claim it.** Open a claim issue (the game does this), so two people don't work on the same pole.
2. **Look at it.**
   - `python -m bonsai_interp.graph.play look --neuron bonsai1/L12/N5397-` shows the top corpus
     examples.
   - For weak and middle activations, use `bonsai_interp.probe.bands`.
3. **Pre-register.** Run
   `python -m bonsai_interp.graph.play new interpret --neuron bonsai1/L12/N5397- --handle <you>`.
   - Fill in your label, 8 sentences that should fire the pole, and 8 near misses that keep the
     topic but drop the trigger.
   - **Commit that file alone.**
4. **Run.** `python -m bonsai_interp.graph.play run <the .tests.json>` writes `<id>.json` with
   the results. Then run `python -m bonsai_interp.graph.fold`.
5. **Commit the results in a second commit.** Open a PR with `?template=interpret.md`.

A failed label is a contribution too, and it is recorded as such. If a pole resists three honest
rounds it becomes `unresolved`. If it is on for most text, set `"unit_kind": "dense / positional"`.

## Shape 2: QC an existing neuron pole

1. Pick a pole, preferably a `held` one with few confirmations. The game suggests these.
2. Run `python -m bonsai_interp.graph.play new qc --neuron bonsai1/L9/N3688+ --handle <you> --verdict confirm|dispute|refine`.
   - Write **new** sentences. Don't reuse the original test.
   - `confirm` re-tests the label as it stands.
   - `dispute` is a test designed to break it.
   - `refine` proposes a better label and tests that.
3. Commit, run, fold and commit again, as above. Open a PR with `?template=qc.md`.

A `confirm` whose test fails counts as a dispute. When disputes outnumber confirmations since
the label last changed, the pole becomes `contested`.

## Stepping out one neuron

`python -m bonsai_interp.graph.play step --neuron bonsai1/L12/N5397-` lists the neurons that feed
a pole most strongly. Interpret one of them as above, and in its tests file:
- set `"roles": ["feeds bonsai1/L12/N5397-"]`
- list the edge in your results, with `from`, `to`, `kind: "feeds"`, `weight` (the share of
  activation lost when the feeder is zeroed) and `method`.

## Blind replication

Run the [neuron-cleanroom](https://github.com/moridinamael/neuron-cleanroom) harness on a pole
and add a `replicate` contribution. Its tests file is your pre-registered prediction of what the
blind agent will find. The full run record lives in the clean-room repo, or in the library.

## Good practice

These lessons were learned the hard way and are recorded in `FINDINGS.md`:

- **Read text after a neutral preamble.** `TextEvaluator` does this for you, on the text's own
  tokens.
  - One caveat from the blind replication: the office/invoice preamble damps some physical
    vocabulary.
- **Look beyond the top 32 examples.**
  - Check which tokens fire the pole most *often* across the corpus, not just most strongly.
  - Check the weak bands.
  - The "nerve" neuron's commonest strong trigger turned out to be " strong".
- **Separate the meaning from the word.** Test the concept without its usual word, the word
  without the concept, and other languages.
- **Record who wrote each test sentence.** Model-written test sentences fail in their own ways;
  pass `--author bonsai` to `play run` if the model wrote them.
- **Use globalMOO sparingly.** Use one learned-inverse model per carefully chosen problem, and
  SciPy for bulk work.
