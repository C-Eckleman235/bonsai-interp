# Contributing to the Bonsai interp graph

The graph is only worth something if its labels can fail. Every contribution follows one rule:

> **Write the test before you see its result, and commit it first.**

CI enforces the commit order (`bonsai_interp.graph.check_prereg`). Reviewers enforce the spirit
of it: tests that could embarrass the label, not tests chosen to flatter it.

## Kinds of contribution

1. **Label a new neuron (quick depth).** Give one pole a label and an 8 + 8 quick test.
2. **Relabel or challenge a neuron.** Propose a new label with its own pre-registered test, or a
   test designed to break an existing label.
3. **Full probe.** Run the forward, bank, held-out, inverse, induce and neighbours stages on one
   pole. See `bonsai_interp/probe/README.md`.
4. **Blind replication.** Run the [neuron-cleanroom](https://github.com/moridinamael/neuron-cleanroom)
   harness on a neuron and add its report under `runs/`.
5. **Tooling.** Improvements to the toolchain. Normal code review applies.

## Workflow for a quick label

1. Look at the neuron:
   - `python -m bonsai_interp.explore --model bonsai1 --neuron L/n --pole +`
   - the activation bands, using `bonsai_interp.probe.bands`
2. Create `probes/bonsai1_L{L}_N{n}_{pos|neg}/quick_tests.json`. It needs your label, 8
   sentences that should fire the pole, and 8 near misses that keep the topic but should not
   fire it:

   ```json
   {"neighbours": [{"unit": "L12/N345", "pole": "+", "label": "...", "fire": ["..."], "quiet": ["..."]}]}
   ```

3. **Commit it.** This commit is your pre-registration.
4. Run `python -m bonsai_interp.probe.quick L/n +`, which writes `quick.json`, and commit the
   result in a **separate, later** commit.
5. Add or update `graph/neurons/bonsai1/L{L}_N{n}_{pos|neg}.json` as described in
   [graph/SCHEMA.md](graph/SCHEMA.md).
   - Fill `prereg_commit` with the hash from step 3.
   - Set `status` from the result: `held` means at least 5 of 8 fire at 0.3 or more of the
     pole's corpus max, and at most 1 near miss does.
6. Run `python -m bonsai_interp.graph.validate`, then open a PR.

A failed label is a contribution too: record it with `status: "failed"`. If a neuron resists
three honest rounds, mark it `unresolved`. If it is on for most text, say so with
`kind: "dense / positional"` and `status: "dense"`.

## How CI decides which file is a test and which is a result

- **Test files:** `*tests*.json`, `holdout*.json`, `*plan*.json`, `*_test.json`, `PREREG*.md`,
  `*request*.json`.
- **Result files:** `quick*.json`, `*_quick*.json`, `*results*.json`, `*_out.json`, `solo.json`,
  `invert_*.json`.

Each result file a PR adds under `probes/` must have a test file in the same folder that was
committed earlier. That test file can be on `main` already, or in an earlier commit of the PR.
Do not squash your pre-registration commit into the results commit. Merge PRs with a merge
commit, not a squash, so the order survives on `main`.

## Good practice

These lessons are learned the hard way in `FINDINGS.md`:

- **Read text after a neutral preamble.** The first positions of a sequence are damped. The
  `TextEvaluator` does this for you, on the text's own tokens.
  - One caveat from the blind replication: the office/invoice preamble lowers some physical
    vocabulary, so check that it doesn't affect your branch.
- **Look beyond the top 32 examples.**
  - Check which tokens fire the pole most often across the corpus, not just most strongly.
  - Check the weak and middle bands.
  - The "nerve" neuron's commonest strong trigger turned out to be " strong".
- **Separate the meaning from the word.** Test the concept without its usual word, the word
  without the concept, and other languages.
- **Test sentences written by the model itself fail in a different way.** Bonsai-written near
  misses often still fire, so record which author wrote each test.
- **Use globalMOO sparingly.** Use one learned-inverse model per carefully chosen problem, and
  SciPy for bulk work.

## Review checklist (for maintainers)

- [ ] The tests were committed before the results (CI is green), and the tests are adversarial,
      not flattering.
- [ ] The label says what fires the pole, at the level the tests support (word, sense or topic).
- [ ] The graph record validates, and its `evidence` points at the probe files.
- [ ] Any claim beyond the tests is marked as a hypothesis.
