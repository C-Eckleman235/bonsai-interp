"""The atlas: one compact, typed table per layer with everything known about each of its 24,576 poles.

  python -m bonsai_interp.atlas build [--auto2 data/auto2] [--auto data/auto]   ->  data/atlas/bonsai1_L<L>.arrow
                                                                                  + data/atlas/vocab.arrow

A full interpretation of the model is 884,736 poles. Gzipped JSON or a file per pole does not scale to that, so the
atlas is an Arrow IPC file with zstd compression: columnar, typed and compressed, and read directly by pyarrow (and so
pandas), polars, DuckDB, Hugging Face datasets and Arrow JS in a browser. Parquet is the same columnar data, but the
reference machine's application-control policy blocks pyarrow's Parquet module, so a Parquet copy, if wanted for a
hosted viewer, is converted on CI. Words are stored as token ids. vocab.arrow maps an id to its text, so readers need
no tokenizer.

    import pyarrow.ipc as ipc; t = ipc.open_file('data/atlas/bonsai1_L4.arrow').read_all(); df = t.to_pandas()

One row per pole, keyed by `pole` = 2 * neuron + (1 for the - pole):
  layer, neuron, sign             int8, int16, "+" / "-"
  class                           stage 1 eligibility: eligible | weak | dense | positional
  strong                          strong activations (>= 0.3 of the pole max) in the selection windows (0..7999)
  hit_tok, hit_n, hit_precision,  the pole's top words by count: token id, strong count, P(strong | token),
  hit_lift, hit_coverage,         lift over the pole's base rate, share of the pole's strong activations,
  hit_trigger                     and whether stage 1 screened that word (lists, at most 5)
  bin, context, test, auroc       the auto-token@2 screen and auto test, when the layer has been screened
Scrutiny 0 throughout: corpus statistics and automatic tests, never a label. Labels and statuses from tested
contributions stay in graph/ (git), keyed the same way, and can be joined on (layer, neuron, sign).
Published in the `atlas` release; `bash scripts/fetch_data.sh atlas` downloads it.
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

from bonsai_interp.sensitive import mask, mask_token

ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / "data" / "atlas"
N = 12288
TOP = 5


def _write(t, path):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    with pa.OSFile(str(path), "wb") as f, ipc.new_file(f, t.schema, options=ipc.IpcWriteOptions(compression="zstd")) as w:
        w.write_table(t)


def _read(path):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    with pa.OSFile(str(path), "rb") as f:  # read into memory: no memory map, so the file is never held open
        return ipc.open_file(f).read_all()


def _layer_of(p):
    return int(p.name.split("_L")[1].split(".")[0])


def _schema():
    import pyarrow as pa
    lst = lambda t: pa.list_(t)
    return pa.schema([("pole", pa.int32()), ("layer", pa.int8()), ("neuron", pa.int16()), ("sign", pa.dictionary(pa.int8(), pa.string())),
                      ("class", pa.dictionary(pa.int8(), pa.string())), ("strong", pa.int32()),
                      ("hit_tok", lst(pa.int32())), ("hit_n", lst(pa.int32())), ("hit_precision", lst(pa.float32())),
                      ("hit_lift", lst(pa.float32())), ("hit_coverage", lst(pa.float32())), ("hit_trigger", lst(pa.bool_())),
                      ("bin", pa.dictionary(pa.int8(), pa.string())), ("context", pa.string()),
                      ("test", pa.dictionary(pa.int8(), pa.string())), ("auroc", pa.float32())])


def build_layer(hits_path, screen_path=None, out_dir=ATLAS):
    import pyarrow as pa
    L = _layer_of(hits_path)
    screen = {}
    if screen_path and screen_path.exists():
        with gzip.open(screen_path, "rt", encoding="utf-8") as f:
            screen = json.load(f)["poles"]
    cols = {k: [] for k in _schema().names}
    with gzip.open(hits_path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = r["neuron"].split("/")[-1]                     # "N123+"
            n, sign = int(key[1:-1]), key[-1]
            trig = {t["token"] for t in r.get("triggers", [])}
            w = (r.get("words") or [])[:TOP]
            s = screen.get(key, {})
            cols["pole"].append(2 * n + (sign == "-"))
            cols["layer"].append(L)
            cols["neuron"].append(n)
            cols["sign"].append(sign)
            cols["class"].append(r["class"])
            cols["strong"].append(r.get("strong"))
            cols["hit_tok"].append([int(x[0]) for x in w])
            cols["hit_n"].append([int(x[1]) for x in w])
            cols["hit_precision"].append([float(x[2]) for x in w])
            cols["hit_lift"].append([float(x[3]) for x in w])
            cols["hit_coverage"].append([float(x[4]) for x in w])
            cols["hit_trigger"].append([int(x[0]) in trig for x in w])
            cols["bin"].append(s.get("bin"))
            cols["context"].append(mask(s.get("context")))  # sensitive words never reach the public atlas
            cols["test"].append(s.get("test"))
            cols["auroc"].append(s.get("auroc"))
    schema = _schema()
    arrays = []
    for field in schema:
        a = pa.array(cols[field.name], type=field.type.value_type if pa.types.is_dictionary(field.type) else field.type)
        arrays.append(a.dictionary_encode() if pa.types.is_dictionary(field.type) else a)
    t = pa.Table.from_arrays(arrays, schema=schema).sort_by("pole")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"bonsai1_L{L}.arrow"
    _write(t, out)
    return out, t


def build_vocab(ids, out_dir=ATLAS):
    """id -> text for every token the atlas mentions (readers need no tokenizer)."""
    import pyarrow as pa
    from bonsai_interp import explore
    from bonsai_interp.auto.context_tier import _bytes_to_unicode
    tok = explore.tokenizer()
    b2u = {u: b for b, u in _bytes_to_unicode().items()}

    def text(i):  # a piece of a multi-byte character decodes to U+FFFD; show its bytes instead (<0xE5>)
        s = tok.decode([i])
        if "�" not in s:
            return s
        piece = tok.convert_ids_to_tokens(i) or ""
        try:
            raw = bytes(b2u[c] for c in piece)
        except KeyError:
            return s
        out, k = [], 0
        while k < len(raw):
            for j in range(min(len(raw), k + 4), k, -1):
                try:
                    out.append(raw[k:j].decode("utf-8"))
                    k = j
                    break
                except UnicodeDecodeError:
                    continue
            else:
                out.append(f"<0x{raw[k]:02X}>")
                k += 1
        return "".join(out)

    ids = sorted(ids)
    t = pa.table({"tok": pa.array(ids, pa.int32()), "text": pa.array([mask_token(text(i)) for i in ids], pa.string())})
    out = out_dir / "vocab.arrow"
    _write(t, out)
    return out


def load(L, root=ROOT):
    """One layer's atlas as a pyarrow Table, or None when data/atlas has no file for it."""
    p = root / "data" / "atlas" / f"bonsai1_L{L}.arrow"
    return _read(p) if p.exists() else None


def vocab(root=ROOT):
    p = root / "data" / "atlas" / "vocab.arrow"
    if not p.exists():
        return {}
    t = _read(p)
    return dict(zip(t["tok"].to_pylist(), t["text"].to_pylist()))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the atlas from the auto-token@2 stage 1 tables and screen exports")
    b.add_argument("--auto2", default=str(ROOT / "data" / "auto2"), help="where bonsai1_L*.hits.jsonl.gz live")
    b.add_argument("--auto", default=str(ROOT / "data" / "auto"), help="where bonsai1_L*.screen.json.gz live")
    a = ap.parse_args()
    files = sorted(Path(a.auto2).glob("bonsai1_L*.hits.jsonl.gz"), key=_layer_of)
    if not files:
        sys.exit(f"no stage 1 tables in {a.auto2}")
    ids = set()
    for f in files:
        L = _layer_of(f)
        out, t = build_layer(f, Path(a.auto) / f"bonsai1_L{L}.screen.json.gz")
        for row in t["hit_tok"].to_pylist():
            ids.update(row)
        screened = t["bin"].drop_null()
        print(f"L{L}: {t.num_rows} poles, {sum(1 for x in t['hit_tok'].to_pylist() if x)} with word hits, "
              f"{len(screened)} screened -> {out.relative_to(ROOT)} ({out.stat().st_size // 1024} KB)")
    v = build_vocab(ids)
    print(f"vocab: {len(ids)} tokens -> {v.relative_to(ROOT)} ({v.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
