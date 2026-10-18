# Instructions for AI agents (Claude Code, Codex, …)

You are helping a person contribute to the **Bonsai Neuron Survey**, an open, test-backed map of
what each neuron in Bonsai-8B does. Bonsai-8B is a 1-bit language model with 36 layers and
12,288 MLP neurons per layer, and each neuron has a + and a − pole. The person usually pasted a
prompt from the survey page. Your job is to do the work end to end on their machine and stop
only for things only they can do.

Run every Python command through the project environment: `uv run python -m …`, or
`.venv/Scripts/python` on Windows, `.venv/bin/python` elsewhere.

## 1. Get ready (once per machine)

1. Install the environment: `uv sync`. You need Python 3.11–3.13 and uv. If uv is missing,
   install it from https://docs.astral.sh/uv/.
2. Download the model: `bash scripts/download_models.sh` gets Bonsai-8B only (about 16 GB of
   disk), which is all you need. Ask before using that much disk.
3. Fetch the data: `bash scripts/fetch_data.sh`. This gets the corpus and the full neuron scan
   (1.4 GB) and needs `gh` to be logged in.
4. Run `python -m bonsai_interp.graph.play doctor` and fix whatever it reports, until it says
   **READY**.
5. **Ask the person** only for things you cannot do yourself:
   - logging in to GitHub (`gh auth login`)
   - their GitHub handle
   - a globalMOO key, which is optional and goes in `.env` as `GMOO_API_KEY=…`
   - permission to use the disk space
6. For blind clean-room missions, also clone https://github.com/moridinamael/neuron-cleanroom
   next to this repository. You also need the `claude` CLI.

When everything passes, tell the person "Ready to survey". Report the GPU and anything optional
that is missing.

## 2. Play a mission

The survey page gives the person a one-line prompt containing a mission, such as `interpret
bonsai1/L12/N5397-`. Turn it into a full brief and follow that:

```
python -m bonsai_interp.graph.play brief <interpret|qc|step|blind> --neuron <id> --handle <github handle> [--gmoo]
```

The brief is generated from this repository, so it always matches the current pipeline. Follow
it in order.

## 3. Rules you must not break

1. **Pre-register.** Write the test (label, 8 sentences that should fire the pole, and 8 near
   misses) and **commit it on its own before running anything that scores it**. Then run, fold
   and commit the results in a **second** commit. Never amend or squash those two commits
   together, and never edit a committed `.tests.json`. CI checks all of this.
2. **Append-only.** Never edit or delete anyone's files under `graph/contributions/`, including
   your own earlier ones. A correction is a new contribution.
3. **Records are derived.** Never edit `graph/neurons/`, `graph/edges.json` or `graph/index.json`
   by hand. Run `python -m bonsai_interp.graph.fold`.
4. **Data policy** ([DATA_POLICY.md](DATA_POLICY.md)):
   - Only the two contribution files go in git, at most 64 KB each.
   - Banks, logs and transcripts go to the library and are referenced by hash.
   - Never commit tensors, weights, `.env`, keys or local paths.
5. **globalMOO.** Use at most one globalMOO model per neuron, and only if the person supplied a
   key. Use SciPy for everything else.
6. **Claim before you work.** Run
   `gh issue create --repo moridinamael/bonsai-interp --title "claim: <id>" --label claim --body "<mission>"`.
   If an open claim already exists for that pole, stop and ask for another neuron.
7. **Report honestly.** A failed label is a valid contribution. Do not rewrite the test after
   seeing the result. If you want a better label, that is a new round with new sentences.

## 4. Doing the science well

- **Look widely before you label.** Read the top examples with `play look`. Then look at weak
  and middle activations and at which tokens fire the pole most *often* (`bonsai_interp.probe.bands`).
  The flagship example: the "nerve" neuron's commonest strong trigger was " strong".
- **Label at the level your test can support:**
  - a word ("the token ' disconnect'")
  - a context ("'to' after an eagerness predicate")
  - a topic or meaning

  Test the meaning without the word, and the word without the meaning, before you claim a
  meaning.
- **Near misses carry the test.** They keep the topic, length and style, and drop only the
  claimed trigger. Easy near misses (unrelated filler) make a label look cleaner than it is.
- **Neurons can be dense.** Some are on for most text, and an 8 + 8 test cannot pass for them.
  Say so with `"unit_kind": "dense / positional"` instead of forcing a label.
- **Write your own sentences.** If the model wrote them, run with `--author bonsai`, because
  model-written near misses misfire more often.

## 5. Finish

Open the PR with the right shape:

```
gh pr create --repo moridinamael/bonsai-interp --template interpret.md   # or qc.md
```

Fill in the template. List the two commit hashes in order and include the one-line result.
Then tell the person what you found in two or three sentences, and give them the PR link.
