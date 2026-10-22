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
| `selection` | Optional, copied into the results. How the pole was chosen: `random draw` (written by `play draw`), `step-out from <id>`, `step-in from <id>`, `concept search: <concept>` or `picked` (`play new --selection …`). Statistics about the share of clean poles should use random draws only; absent means not recorded |

### `<id>.json` (the results)

| Field | Meaning |
|---|---|
| `id`, `kind`, `neuron`, `contributor`, `created` | `kind` is one of `interpret`, `qc`, `replicate` (a blind clean-room run), `map` (edges only) or `withhold` (a sensitive trigger, below) |
| `label` | For `interpret` and `qc` refine: the label that was tested |
| `qc` | `{"verdict": "confirm" or "dispute" or "refine", "label_under_test": …}` |
| `tests` | `[{protocol, tests_file, prereg_commit, author, …numbers…, per_sentence}]`. `counts_for_status: false` excludes a test from the status (used for model-written test sentences) |
| `recipe` | `{"toolchain_commit": …, "command": …}` |
| `edges` | `[{from, to, kind, weight, method}]` with full neuron ids |
| `artifacts` | `[{name, url, sha256, bytes}]` in the library |
| `short`, `unit_kind`, `confidence`, `depth`, `roles`, `via`, `selection`, `examples`, `evidence`, `notes`, `verdict`, `summary`, `offshoots`, `lineage` | Optional |

## Protocols are versioned

`bonsai_interp/graph/protocols.py` maps each protocol to its pass rule:

- `quick-8+8@1`: held if at least 5 of 8 sentences fire at 0.3 or more of the pole's corpus max,
  and at most 1 near miss does.
- `quick-8+8@2`: held if the fire sentences rank above the near misses (AUROC >= 0.90), the median
  fire sentence reaches 0.20 of the pole's corpus max, and at most 1 near miss reaches 0.30. It needs
  the per-sentence activations that `play run` records.
- `quick-8+8@3` (the default for new tests): @2's rule on sentences whose results the author did not
  already know. `play probe` logs every sentence it scores (`.probe-log/`, local); `play run` records
  in `probed_before` how many of the 16 were scored on this neuron before the registration commit.
  More than 2 makes the outcome `exploratory`: recorded, but neither a hold nor a fail, so a pole whose
  only tests are exploratory shows as `probed`. Why: pre-registration stops editing a test after its
  result, but not choosing sentences whose results are already known.
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
| `withheld` | a `withhold` contribution: the trigger is sensitive content, recorded with a masked description (takes precedence over every other status) |
| `probed` | a full probe (`depth: full`) |
| `replicated` | only blind replications so far |

More rules:

- A failed round after a held label goes into `label_history` and does not replace the label. To
  challenge a held label, use a QC `dispute`, or a QC `refine` with a better label.
- `replicate` contributions never change the label.
- For list fields, the record holds the union of all contributions, in order. For scalar fields,
  the latest contribution wins.

## Sensitive triggers: `withhold`

Some poles fire on slurs or other content that should not be shown on the public map or the Twitch stream, or
tested live. Such a pole gets a `withhold` contribution instead of an `interpret` round:

```
uv run python -m bonsai_interp.graph.play withhold --neuron bonsai1/L1/N362+ --handle <you> --reason slur     --label "a slur token (masked), any context" --short "slur token (masked)" --agent "claude-code opus-5.5" --screens card atlas
```

- `label` (and `short`) describe the kind of trigger without the content; the check rejects a label that contains a
  word on the denylist (`bonsai_interp/sensitive.py`, stored as hashes).
- `withheld.reason` is one of `slur`, `sexual`, `personal data`, `violence`, `other`.
- There is no test to pre-register, so it is one file in one commit. Unmasked evidence, if any, goes to the library and
  is listed under `artifacts` by hash, never in git.
- The fold makes the pole `withheld`, above every other status, shows only the withhold's masked label and short, and
  masks the pole's earlier labels in `history`. Edges may point at a withheld pole like any other.

Separately, everything the map and the atlas release display (labels, word hits, screen rules, card examples) passes
through the same denylist mask; `graph/index.json` and the records stay the raw data.

## Read API

The derived files are the survey's public, read-only API. There is no server and no account: read
them from a clone, or over HTTPS from the `main` branch (once the repository is public):

```
https://raw.githubusercontent.com/moridinamael/bonsai-interp/main/graph/index.json
https://raw.githubusercontent.com/moridinamael/bonsai-interp/main/graph/neurons/bonsai1/L20_N12208_neg.json
https://raw.githubusercontent.com/moridinamael/bonsai-interp/main/graph/edges.json
```

They change only when the merge agent folds a merged contribution, so a commit hash pins a snapshot.
Neuron ids are always `<model>/L<layer>/N<index><pole>`, e.g. `bonsai1/L20/N12208-`; record file names
are `L<layer>_N<index>_<pos|neg>.json`.

### `graph/index.json` (one object, about 1 KB per surveyed pole)

| Field | Type | Meaning |
|---|---|---|
| `models` | `{model: {layers, neurons_per_layer}}` | The models the ids refer to (only `bonsai1`) |
| `neurons` | `{id: entry}` | One entry per pole that has at least one contribution; poles not listed are unsurveyed |
| `neurons[id].status` | string | One of the statuses in the table above |
| `neurons[id].label`, `.short` | string or null | Current label (see "How a record is folded") and its short form |
| `neurons[id].kind` | string or null | `unit_kind`, e.g. `dense / positional`, `token/lexical` |
| `neurons[id].depth` | string | `quick` or `full` |
| `neurons[id].qc` | `{confirm, dispute, refine}` | QC counts by what the QC test showed (a confirm whose test failed counts as a dispute; exploratory QCs count for neither) |
| `neurons[id].n_contrib` | int | Number of contributions |
| `neurons[id].replicated` | bool | Has a blind clean-room replication |
| `neurons[id].contributors` | `[handle]` | In order of first contribution |
| `neurons[id].t` | `[[protocol, outcome, n_fire, n_false_pos, author, contribution id]]` | Every test, oldest first. `outcome` is `held`, `failed`, `invalid` (fields missing for its protocol), `recorded` (a protocol the code does not score) or, for `cleanroom@1`, the blind verdict in words |
| `neurons[id].history` | `[[label, outcome]]` | The last six labels tried |
| `neurons[id].scrutiny` | `{level, name, test_rounds, sentences, contributors, qc}` | How much evidence there is, 0 `unexamined` to 5 `replicated` |
| `edges` | `[[from id, to id, kind]]` | Every edge, without weights (those are in `edges.json`) |
| `contributors` | `{handle: points}` | The scoreboard, highest first |

### `graph/neurons/<model>/<pole>.json` (the full record)

Every field listed under "How a record is folded": `id`, `model`, `layer`, `index`, `pole`, `label`,
`short`, `kind`, `status`, `depth`, `tests` (each with its `contribution` and `outcome`), `label_history`
(`[contribution id, label, outcome]`), `examples`, `evidence`, `roles`, `contributors`, `contributions`,
`qc`, `replications` and `scrutiny`, plus the latest value of each optional scalar (`verdict`, `summary`,
`offshoots`, `lineage`). A record's `contributions` list names the files that produced it, so every
number can be traced to a pre-registered test.

### `graph/edges.json`

`{"edges": [{from, to, kind, weight, method, contribution, …}]}`, sorted, with the latest measurement of
each `(from, to, kind, method)`. `kind` is `feeds`, `reads`, `gates`, `co-fires` or an older free-text
kind such as `feeds lateral`; for `feeds`, `weight` is the share of the target's activation lost when
the source is zeroed.

### Not part of the API

- `site/cards.json` and the cards embedded in the survey page are label-free summaries of the corpus
  scan and the weights (top examples, density, regimes, direct logit effect) for surveyed neurons.
  Their compact format may change with the page; for any neuron, run
  `python -m bonsai_interp.graph.play card --neuron <id> --out card.json`.
- `data/token_index_bonsai1.npz` (token -> poles, used by `play which --token`) is a local artifact
  rebuilt from the scan on first use; it is never committed.

### Versioning

Field names above are stable. Changes are **additive**: new fields may appear at any level, and
readers should ignore fields they don't know. Status values and test outcomes may gain new values;
treat an unknown one as "not held". A change that renames or removes a field, or changes a field's
meaning, adds a top-level `"schema"` integer to `index.json` (its absence means schema 1) and is
announced in this file. Protocols are versioned separately (see above), so a test's numbers are always
read under the rule it was made under.

## Commands

```bash
python -m bonsai_interp.graph.play new interpret --neuron bonsai1/L12/N5397- --handle you
python -m bonsai_interp.graph.play run graph/contributions/bonsai1/L12_N5397_neg/<id>.tests.json
python -m bonsai_interp.graph.fold            # preview the derived files locally (the merge agent commits them on main)
python -m bonsai_interp.graph.validate        # what CI runs
python -m bonsai_interp.graph.check_prereg --base origin/main
python -m bonsai_interp.graph.play card --neuron bonsai1/L9/N3688+   # any neuron: examples, density, logit effect
python -m bonsai_interp.graph.play which "some text"                 # reverse search; or --token " nerve"
```
