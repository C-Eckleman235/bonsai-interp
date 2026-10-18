# Merge agent

This is the procedure for the agent (or person) who merges contribution PRs. It is written so that
merging is **mechanical, idempotent and non-destructive**: the agent never decides what a neuron
"is". It checks that a contribution is well formed and honest about its order, then lets `fold`
derive the record.

## For each PR

1. **Classify it** from the title or template: `interpret`, `qc`, `replicate`, `map`, or tooling.
   Tooling PRs go to a human reviewer, so stop here for those.
2. **Run the checks.** CI runs them too.
   ```bash
   python -m bonsai_interp.graph.check_prereg --base origin/main   # append-only, tests-before-results, size caps
   python -m bonsai_interp.graph.validate                          # schemas, protocols, derived files up to date
   ```
   If either fails, comment with the exact error lines and the fix. Do not edit the
   contributor's files.
3. **Check the changed paths.** A contribution PR may add only:
   - `graph/contributions/<model>/<pole folder>/<id>.tests.json` and `<id>.json`, one pair per
     contribution (a `map` contribution may be a single `<id>.json`)
   - the regenerated `graph/neurons/…`, `graph/edges.json` and `graph/index.json`

   Anything else (code, probes, data) means the PR should be split.
4. **Check the content.** These are judgement calls, but they are bounded:
   - **Sentences.** The 8 near misses keep the topic and drop the claimed trigger. If they
     are unrelated filler, ask for real near misses. Near misses that are too easy make a
     label look cleaner than it is.
   - **QC independence.** A QC's sentences are new and do not copy the original test.
   - **Label level.** The label claims no more than the test shows. A label about a meaning
     that was tested only with one word should be rephrased as a word-level label.
   - **Artefacts.** Each URL resolves and its sha256 matches. If one does not, ask for a fix;
     it does not block the merge.
   - **Recipe.** It names a toolchain commit that exists on `main`.
5. **PRs carry contributions only.** `check_prereg` rejects derived files
   (`graph/neurons/`, `edges.json`, `index.json`) in a PR. Ask the contributor to drop them.
6. **Merge with a merge commit, never a squash.** Then, on `main`, run
   `python -m bonsai_interp.graph.fold`, commit the regenerated files as "fold after #<n>", and check
   them with `python -m bonsai_interp.graph.validate --strict`. If that check fails, stop merging and
   alert the owner (circuit breaker). Squashing would destroy the
   tests-before-results order that CI checked.
7. **Close the claim issue.** Add the labels `held`, `failed`, `contested` or `refined` from the
   folded status. Post a one-line summary, for example: "L12/N5397- held 6/8 (0 FP); 2 QCs
   pending".

## Conflicts between PRs

PRs never conflict: each contribution has its own `<id>`, and derived files only ever change on
`main` in the merge agent's fold commits. Fold is deterministic, so the order of merges does not
change the result.

## Idempotence, concretely

- **The same contribution submitted twice** has the same `<id>`. The second PR adds nothing new,
  so close it as a duplicate.
- **A contributor re-runs their test** and gets different numbers (a toolchain change, say).
  That is a new contribution with a new `<id>`; the old one stays. The record shows both
  results, and status follows the latest.
- **A wrong contribution** is never deleted. Ask for a QC `dispute`, or for a new interpret
  round that supersedes it. If a contribution is malicious or broken, revert the merge commit;
  that is the only removal path, and a human must approve it.

## When the pipeline changes

New test kinds are new protocol versions in `bonsai_interp/graph/protocols.py`, for example
`quick-8+8@2`, never edits to an old one. A new field in a contribution is allowed if
`validate` passes. `fold` copies unknown scalar fields only if they are added to `SCALARS`, so
a new field is invisible until a maintainer opts it in. Old contributions are never migrated
in place. If records need a new shape, change `fold`; the contributions stay as they were.
