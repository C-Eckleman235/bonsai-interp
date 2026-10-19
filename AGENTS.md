# Instructions for AI agents (Claude Code, Codex, …)

You are helping a person contribute to the **Bonsai Neuron Survey**, an open, test-backed map of
what each neuron in Bonsai-8B does. Bonsai-8B is a 1-bit language model with 36 layers and
12,288 MLP neurons per layer, and each neuron has a + and a − pole. The person usually pasted a
prompt from the survey page. Your job is to do the work end to end on their machine and stop
only for things only they can do.

Run every Python command through the project environment: `uv run python -m …`, or
`.venv/Scripts/python` on Windows, `.venv/bin/python` elsewhere.

## 1. Get ready (once per machine)

You need about 8 GB of disk in total: 1.2 GB of weights, 1.4 GB of data and about 5 GB for the Python
environment (`uv sync` installs a CUDA build of torch). The scripts are bash. On Windows, run every command
in Git Bash (it comes with Git for Windows), not only the scripts.

Hardware: an NVIDIA GPU with 8 GB or more (a mission peaks at about 5 GB), or an Apple Silicon Mac (macOS 14 or later; 32 GB
of unified memory recommended, 16 GB is tight). The toolchain picks the device itself (CUDA,
then Apple's MPS, then the CPU). To force one, put `BONSAI_DEVICE=cuda`, `mps` or `cpu` in
`.env`. The CPU is fine for checks but too slow for missions. Mac details are in
[docs/MAC.md](docs/MAC.md).

1. Clone https://github.com/moridinamael/neuron-cleanroom next to this repository; blind
   clean-room missions need it. Every mission needs the person's GitHub handle: read it with
   `gh api user --jq .login` once gh is logged in, and ask only if that fails.
2. Install the environment: `uv sync`. You need Python 3.11–3.13 and uv. If uv is missing,
   install it from https://docs.astral.sh/uv/.
3. Get the model weights (Bonsai-8B only):
   - If they are **already on this machine**, put `BONSAI1_PATH=<that folder>` in `.env`
     instead of downloading. The folder is the one that contains `config.json` and either the
     `model-0000x-of-00004.safetensors` shards or `Bonsai-8B-Q1_0.gguf`.
   - Otherwise run `bash scripts/download_models.sh`. It fetches PrismML's 1-bit GGUF (1.2 GB),
     which holds exactly the same weights as the 16 GB unpacked release. A warning about unauthenticated Hugging Face requests is harmless.
4. Fetch the data: `bash scripts/fetch_data.sh`. It downloads the corpus (120 MB), the full
   neuron scan (1.1 GB) and a band scan (0.2 GB), and needs `gh` to be logged in.
5. Run `uv run python -m bonsai_interp.graph.play doctor` and fix whatever it reports, until it says
   **READY**. Doctor checks every weight shard, then loads the model and reads one known neuron.
   It reports the device and its speed, and it compares the reading with the reference CUDA
   machine. On a device other than CUDA, also run `uv run python tests/test_device_parity.py`
   once. It checks four neurons, the model's next-token prediction and the backward pass, and it
   must print PARITY OK.
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
   misses) and **commit it on its own before running anything that scores it**. Then run
   and commit the results (contribution files only) in a **second** commit. Never amend or squash those two commits
   together, and never edit a committed `.tests.json`. CI checks all of this.
2. **Append-only.** Never edit or delete anyone's files under `graph/contributions/`, including
   your own earlier ones. A correction is a new contribution.
3. **PRs carry contributions only.** Never commit `graph/neurons/`, `graph/edges.json` or
   `graph/index.json`: the merge agent regenerates them on `main` after each merge. You may run
   `python -m bonsai_interp.graph.fold` locally to preview the record, but don't commit the output.
4. **Data policy** ([DATA_POLICY.md](DATA_POLICY.md)):
   - Only the two contribution files go in git, at most 64 KB each.
   - Banks, logs and transcripts go to the library and are referenced by hash.
   - Never commit tensors, weights, `.env`, keys or local paths.
5. **globalMOO.** Use at most one globalMOO model per neuron, and only if the person supplied a
   key. Use SciPy for everything else.
6. **No claims.** Contributions never conflict, so nobody reserves a pole. If two people test the
   same one, both results count, and the second is an independent replication.
7. **Record your methods.** Fill `methods.agent` (who did the work, e.g. "claude-code opus-5.5") and
   `methods.screens` (the analyses you used, from `bonsai_interp/graph/methods.py`) in the tests file.
   `play run` adds what it can detect and the time taken. This is how the survey learns which
   techniques pay off.
8. **Report honestly.** A failed label is a valid contribution. Do not rewrite the test after
   seeing the result. If you want a better label, that is a new round with new sentences.

## 4. Doing the science well

- **Explore with `play probe`.** It scores any sentences on a pole without recording anything:
  `python -m bonsai_interp.graph.play probe --neuron <id> "sentence" …`. If you write your own
  scripts, run them from the repository root with `PYTHONPATH=.` so `bonsai_interp` imports.
- **Start from the card.** `play card --neuron <id>` shows both poles' top examples, how dense the
  pole is, its sign regimes and the tokens its output promotes, and flags dense or positional units,
  in seconds and without the GPU. `play which "text"` goes the other way: which poles fire on each
  token of your text (`--token " nerve"` for poles whose top examples peak on one token).
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
