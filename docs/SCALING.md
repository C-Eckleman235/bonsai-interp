# Scaling to a full interpretation

A full interpretation of Bonsai-8B covers 884,736 poles (36 layers × 12,288 neurons × 2 signs). The formats below are
chosen now so that the survey still fits in a clone, a browser and a CI job when most poles carry evidence.

## Where each kind of evidence lives

| evidence | volume at full scale | home | format |
|---|---|---|---|
| Machine evidence per pole: word hits, auto screens, auto tests, later screens | 885k rows × a few dozen columns | the `atlas` release (`data/atlas/`) | **one Arrow IPC table per layer, zstd**, keyed by `pole` = 2·neuron + (1 for −); words as token ids, plus `vocab.arrow` |
| Tested contributions: pre-registrations and results | about 16 KB per contribution | git, `graph/contributions/` | JSON, two files per contribution: the audit trail that pre-registration needs |
| Derived records: status, label, scrutiny | 1 row per surveyed pole | git, `graph/index.json` and `graph/neurons/` (folded) | JSON today; per-layer shards before it passes about 10 MB (see below) |
| Bulky artefacts: banks, logs, transcripts, stage tables | unbounded | releases or the HF library, referenced by sha256 | whatever the tool writes |
| The map's view | lazy, one layer at a time | GitHub Pages | per-layer columnar JSON derived from the atlas, plus one vocab file |

## Why these formats

- **Columnar and typed:**
  - A layer of 24,576 poles with 5 word hits each is about 1.8 MB, and the whole model about 65 MB. The same as
    one JSON file per pole would be gigabytes.
  - Adding a screen means adding columns, not files.
- **Readable everywhere:**
  - Arrow IPC is read by pyarrow (and pandas), polars, DuckDB, Hugging Face datasets, and Arrow JS in a browser:
    `ipc.open_file(path).read_all()`.
  - Parquet stores the same columns. The reference machine's application-control policy blocks pyarrow's Parquet
    and Feather modules, so we write Arrow IPC and convert on CI if a Parquet copy is ever needed, e.g. for the
    Hugging Face dataset viewer.
- **Token ids, not strings:** words are 4-byte ids. `vocab.arrow` has their text once, and shows byte fragments as
  `<0xE5>`, so readers need no tokenizer.
- **Joinable:** the atlas and the graph share the key (layer, neuron, sign). A full report is one join, not a crawl
  over files.

## Getting the atlas

```
bash scripts/fetch_data.sh atlas                    # -> data/atlas/, about 70 MB
python -m bonsai_interp.graph.play card --neuron bonsai1/L20/N70+    # word hits and the auto screen appear on the card
```

```python
from bonsai_interp import atlas
df = atlas.load(4).to_pandas()    # one row per pole of layer 4
vocab = atlas.vocab()             # token id -> text
```

## Next steps before the graph outgrows git

1. **Shard the map index by layer.** `graph/index.json` is embedded in the page and grows about 0.75 KB per surveyed
   pole. At 10k poles that's about 7.5 MB, which is acceptable. Past that, emit `index/L<L>.json` and load layers
   lazily, as the screen layer already does.
2. **Atlas columns for the tested record.** At each fold, write status, label and test statistics per pole into the
   atlas as well, so one table answers "what do we know about pole X" for machines and people alike.
3. **Pack old contributions** if git passes about 1 GB. Move closed rounds' result files into an archived release
   table, keyed by contribution id and sha256, and keep the pre-registration files in git, because they carry the
   commit-time proof. That decision belongs to the maintainers when the time comes; nothing needs it yet (5 MB today).
4. **Edges:** `graph/edges.json` becomes a per-layer edge table (from, to, kind, weight, method) in the same Arrow
   format once it passes about 50k edges.
