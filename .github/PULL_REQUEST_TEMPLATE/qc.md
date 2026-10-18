<!-- PR shape 2 of 2: QC an existing neuron pole. Title: "qc: bonsai1/L9/N3688+ confirm|dispute|refine" -->

## QC: `bonsai1/L__/N____±`

- **Claim issue:** #<!-- optional for QC -->
- **Label under test:** <!-- copied from graph/neurons/…; for refine, also give your new label -->
- **Intended verdict:** <!-- confirm (re-test as is) / dispute (a test designed to break it) / refine (a better label, with its own test) -->

## Pre-registration

| | commit |
|---|---|
| `<id>.tests.json` (your own 8 fire + 8 near misses, written without reusing the original sentences) | <!-- hash, FIRST commit --> |
| `<id>.json` (results from `play run`) | <!-- hash, LATER commit --> |

## Result

<!-- the line printed by `play run`. A confirm whose test fails counts as a dispute automatically. -->

## What you looked at

<!-- e.g. "weak bands 0.1–0.3", "other languages", "the word without the meaning" -->

## Checklist

- [ ] New sentences, not copies of the original test.
- [ ] Tests were committed before results (do not squash those commits).
- [ ] Only new files under `graph/contributions/…` plus refreshed derived files; no edits to anyone's contribution.
- [ ] `python -m bonsai_interp.graph.validate` passes.
