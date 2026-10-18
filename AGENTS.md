# Instructions for AI agents (Claude Code, Codex, …)

You are helping a person contribute to the **Bonsai Neuron Survey**, an open, test-backed map of
what each neuron in Bonsai-8B does. Bonsai-8B is a 1-bit language model with 36 layers and
12,288 MLP neurons per layer, and each neuron has a + and a − pole. The person usually pasted a
prompt from the survey page. Your job is to do the work end to end on their machine and stop
only for things only they can do.

Run every Python command through the project environment: `uv run python -m …`, or
`.venv/Scripts/python` on Windows, `.venv/bin/python` elsewhere.

## 1. Get ready (once per machine)

You need about 18 GB of disk in total: 16 GB of weights and 1.4 GB of data. The scripts are
bash; on Windows, Git Bash (which comes with Git for Windows) runs them.

1. Clone https://github.com/moridinamael/neuron-cleanroom next to this repository; blind
   clean-room missions need it. Ask the person for their GitHub handle now, since every
   mission uses it.
2. Install the environment: `uv sync`. You need Python 3.11–3.13 and uv. If uv is missing,
   install it from https://docs.astral.sh/uv/.
3. Get the model weights (Bonsai-8B only):
   - If they are **already on this machine**, put `BONSAI1_PATH=<that folder>` in `.env`
     instead of downloading. The folder is the one that contains `config.json` and the
     `model-0000x-of-00004.safetensors` shards.
   - Otherwise run `bash scripts/download_models.sh` (about 16 GB). Ask before using that much
     disk. A warning about unauthenticated Hugging Face requests is harmless.
4. Fetch the data: `bash scripts/fetch_data.sh`. It downloads the corpus (120 MB), the full
   neuron scan (1.1 GB) and a band scan (0.2 GB), and needs `gh` to be logged in.
5. Run `python -m bonsai_interp.graph.play doctor` and fix whatever it reports, until it says
   **READY**. Doctor checks every weight shard, then loads the model and reads one known neuron.
6. **Ask the person** only for things you cannot do yourself:
   - logging in to GitHub (`gh auth login`)
   - a globalMOO key, which is optional and goes in `.env` as `GMOO_API_KEY=…`
   - permission to use the disk space
   - for blind missions, the `claude` CLI or `codex` CLI

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
