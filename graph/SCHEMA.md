# Graph schema

```
graph/
  contributions/<model>/L{L}_N{n}_{pos|neg}/
      <id>.tests.json     pre-registration, committed FIRST       (contributors write this)
      <id>.json           results, committed in a LATER commit    (written by `play run`)
  neurons/<model>/L{L}_N{n}_{pos|neg}.json     derived record     (written by `fold`, never by hand)
  edges.json                                   derived edge list
  index.json                                   derived summary for the site and the game
```

The model is `bonsai1` (Bonsai-8B, 1-bit): 36 layers, 12,288 neurons per layer, and two poles
per neuron.

## Contributions are append-only

A contribution is never edited, renamed or deleted after it is merged; CI rejects any PR that
does. A correction is a new contribution. Every record is recomputed from its neuron's
contributions in `(created, id)` order. This makes a merge **idempotent and non-destructive**:
merging the same contribution twice changes nothing, and new information never overwrites old.

`<id>` is `YYYY-MM-DD-<github handle>-<random suffix>`. Core-team history uses `core-…` ids.

### `<id>.tests.json` (the pre-registration)

| Field | Meaning |
|---|---|
| `contribution`, `contributor`, `kind`, `neuron`, `protocol` | Who is testing which pole, and under which test protocol (e.g. `quick-8+8@1`) |
| `neighbours` | `[{"unit": "L12/N5397", "pole": "-", "label": …, "fire": [8], "quiet": [8]}]` |
| `qc` | For QC: `{"verdict": "confirm", "dispute" or "refine", "label_under_test": …}` |
| `predictions`, `short`, `unit_kind`, `confidence`, `notes`, `roles`, `via` | Optional. They are copied into the results |

### `<id>.json` (the results)

| Field | Meaning |
|---|---|
| `id`, `kind`, `neuron`, `contributor`, `created` | `kind` is one of `interpret`, `qc`, `replicate` (a blind clean-room run) or `map` (edges only) |
| `label` | For `interpret` and `qc` refine: the label that was tested |
| `qc` | `{"verdict": "confirm" or "dispute" or "refine", "label_under_test": …}` |
| `tests` | `[{protocol, tests_file, prereg_commit, author, …numbers…, per_sentence}]`. `counts_for_status: false` excludes a test from the status (used for model-written test sentences) |
| `recipe` | `{"toolchain_commit": …, "command": …}` |
| `edges` | `[{from, to, kind, weight, method}]` with full neuron ids |
| `artifacts` | `[{name, url, sha256, bytes}]` in the library |
| `short`, `unit_kind`, `confidence`, `depth`, `roles`, `via`, `examples`, `evidence`, `notes`, `verdict`, `summary`, `offshoots`, `lineage` | Optional |

## Protocols are versioned

`bonsai_interp/graph/protocols.py` maps each protocol to its pass rule:

- `quick-8+8@1`: held if at least 5 of 8 sentences fire at 0.3 or more of the pole's corpus max,
  and at most 1 near miss does.
- `held-out@1`
- `cleanroom@1`

A new pipeline adds a new version, such as `quick-8+8@2`. Old contributions keep the rule they
were made under. A test under a protocol the code doesn't know is kept and shown, but doesn't
change the status.

## How a record is folded

The record keeps the earlier fields: `id`, `label`, `short`, `kind`, `status`, `depth`,
`tests`, `label_history`, `examples`, `evidence`, `roles` and `contributors`. It adds
`contributions`, `qc` counts and `replications`.

| status | when |
|---|---|
| `unlabelled` | no contribution has tested a label yet |
| `held` | the latest round of the current label held |
| `contested` | the label held, but since it last changed, more QC tests have failed than held |
| `failed` | no label has held yet; one or two failed rounds |
| `unresolved` | no label has held after three failed rounds |
| `dense` | a contribution set `unit_kind` to a dense kind |
| `probed` | a full probe (`depth: full`) |
| `replicated` | only blind replications so far |

More rules:

- A failed round after a held label goes into `label_history` and does not replace the label. To
  challenge a held label, use a QC `dispute`, or a QC `refine` with a better label.
- `replicate` contributions never change the label.
- For list fields, the record holds the union of all contributions, in order. For scalar fields,
  the latest contribution wins.

## Commands

```bash
python -m bonsai_interp.graph.play new interpret --neuron bonsai1/L12/N5397- --handle you
python -m bonsai_interp.graph.play run graph/contributions/bonsai1/L12_N5397_neg/<id>.tests.json
python -m bonsai_interp.graph.fold            # refresh the derived files
python -m bonsai_interp.graph.validate        # what CI runs
python -m bonsai_interp.graph.check_prereg --base origin/main
```
