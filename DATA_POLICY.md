# What goes in the repo, what goes in the library, and what goes nowhere

The repo must stay small enough to clone in seconds while the graph grows to thousands of
neurons. A quick-tested neuron costs about 16 KB here. A fully probed neuron produces 1–5 MB of
files, almost all of it solver logs and generated HTML. So git holds the **evidence and the
recipe**; everything bulky lives in the **library** and is referenced by hash.

## 1. In git: the evidence and the recipe

Each contribution is at most two files and **64 KB per file**. A PR adds at most **2 MB under
`graph/`** (enough for a sweep of about 350 poles), and CI enforces both limits. The per-file cap is what keeps bulk out of git;
the per-PR cap was 256 KB until 2026-10-26 and was raised so that large pre-registered sweeps need not be split after scoring.

| What | Why it is in git |
|---|---|
| The pre-registration (`<id>.tests.json`): label, the 8 + 8 test sentences, predictions | It *is* the test. Its commit date is the proof it came first |
| The results (`<id>.json`): counts, per-sentence activation and peak token (3 decimals), outcome | Small, and what reviewers read |
| Edges found while stepping out (`from`, `to`, `kind`, `weight`, `method`) | They are the graph |
| The recipe: toolchain commit, command, seed | Anything here can be recomputed from it |
| Up to 8 corpus examples, at most 400 characters each | Enough to see the job; corpus text stays under its own licence |
| Notes, at most 4,000 characters | Longer write-ups go in the library |

Neuron records (`graph/neurons/`), `graph/edges.json` and `graph/index.json` are **derived** by
`python -m bonsai_interp.graph.fold`. Nobody edits them by hand.

## 1b. The atlas: per-pole machine evidence, in releases

The auto tier's per-pole statistics are one Arrow IPC table per layer (zstd): word hits, auto screens and auto tests,
keyed by pole. They are published in the `atlas` release, not in git, and downloaded with
`bash scripts/fetch_data.sh atlas`. They are corpus statistics and automatic tests at scrutiny 0, and never change a
pole's status. See [docs/SCALING.md](docs/SCALING.md) for how the formats are meant to grow to all 884,736 poles.

## 2. In the library: bulky, optional, referenced by hash

Put large or numerous artefacts in the library and list them in your contribution's `artifacts`:

```json
"artifacts": [{"name": "bank.json", "url": "https://huggingface.co/datasets/<org>/bonsai-interp-library/resolve/main/<id>/bank.json",
               "sha256": "…", "bytes": 174612}]
```

Examples:
- sentence banks larger than the test sets
- inverse-solver evaluation logs
- band samples
- steering generations
- clean-room transcripts and room bundles
- HTML dossiers and figures

The library is a Hugging Face dataset repository: versioned, built for large files, and free.
The merge agent checks that each URL resolves and that its hash matches. A missing library file
never invalidates a contribution, because git already holds the evidence.

## 3. Nowhere

- **Activation vectors, tensors, scans or caches.** Recompute them from the recipe. If one is
  essential to a claim, upload it to the library and explain why.
- **Model weights, or copies of the corpus.** They are on Hugging Face and in the `data-v1`
  release.
- **Secrets** (`.env`, API keys, tokens), **local paths**, and personal data about anyone.
- **Long copied text**: corpus excerpts over 400 characters, or third-party documents.

## Why this split

- **Reproducible:** every number in git has a recipe. Anyone with the model can regenerate the
  bulky parts, so storing them is a convenience, not a requirement.
- **Reviewable:** a PR is a few kilobytes of JSON that a person or an agent can read in full.
- **Bounded:** at about 16 KB per quick-tested neuron, 10,000 neurons is about 150 MB of git.
  The library grows without slowing anyone's clone.
