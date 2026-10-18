# Graph schema

## Neuron records

Each record lives at `graph/neurons/<model>/L{layer}_N{index}_{pos|neg}.json`, one file per
neuron pole. `<model>` is `bonsai1` (Bonsai-8B, 1-bit), with 36 layers and 12,288 neurons per
layer.

| Field | Type | Meaning |
|---|---|---|
| `id` | string | `bonsai1/L9/N3688+` |
| `model`, `layer`, `index`, `pole` | str, int, int, `+`/`-` | must match `id` and the file name |
| `label` | string | what fires this pole, at the level the tests support |
| `short` | string | a short name for graphs |
| `kind` | string or null | e.g. `sparse feature`, `lexical`, `contextual`, `dense / positional`, `incoherent` |
| `status` | enum | `unlabelled`, `held`, `failed`, `unresolved` (3+ honest rounds), `dense`, `probed` (full probe; see `verdict`) |
| `depth` | enum | `quick`, `full`, `blind replication only` |
| `tests` | list | the pre-registered tests, oldest first (see below) |
| `label_history` | list of `[stage, label]` | every label tried, including the ones rejected |
| `verdict`, `summary`, `offshoots` | optional | for full probes |
| `replications` | optional list | blind or independent replications, each with its own label and verdict |
| `examples` | list of strings | a few top corpus contexts, with the peak token in ⟦ ⟧ |
| `roles` | list | how the neuron entered the graph (input, reader, co-fires, feeds lateral, …) |
| `evidence` | list of paths | the probe files and pages behind the record |
| `contributors` | list | who added or changed the record |

## Tests

| Field | Meaning |
|---|---|
| `type` | `quick-8+8` or `held-out` (other types welcome; describe them in the PR) |
| `author` | who wrote the test sentences: `claude`, `bonsai`, a GitHub handle, … |
| `tests_file`, `result_file` | repo paths. Both must exist |
| `prereg_commit` | the commit that first added `tests_file`. It must come before the result |
| `label_tested` | the label as it stood when the test was written |
| `n_fire`, `n_false_pos`, `n_written`, `survives` | for `quick-8+8`. `survives` is true when `n_fire ≥ 5` and `n_false_pos ≤ 1` |
| `accuracy`, `auroc`, `fire_rate_predicted_fire`, … | for `held-out` |

`status: "held"` requires that the latest non-Bonsai quick test survived.

## Edges

`graph/edges.json` has the form `{"edges": [...]}`, and each edge has these fields:

| Field | Meaning |
|---|---|
| `from`, `to` | neuron ids. Both must have records |
| `kind` | `feeds`, `reads`, `gates`, `co-fires`, `feeds reader`, `feeds lateral` |
| `weight` | for `feeds`: the share of the target's activation lost when the source alone is zeroed. For `reads`: the change in the reader (as a fraction of its range) when the source is boosted. For `gates`: a signed effect |
| `method`, `source` | how the edge was measured, and where the files are |

Run `python -m bonsai_interp.graph.validate` to check everything.
`python -m bonsai_interp.graph.export` regenerates the records for the core team's probes from
`probes/`. Contributors edit records by hand.
