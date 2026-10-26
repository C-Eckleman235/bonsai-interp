"""Parse English corpus windows with spaCy and map the parse onto model tokens (CPU only).

  python -m bonsai_interp.grammar.parse select     # -> data/grammar/bonsai1.parse.select.npz
  python -m bonsai_interp.grammar.parse test       # -> data/grammar/bonsai1.parse.test.npz
  python -m bonsai_interp.grammar.parse windows W1 W2 ... --out FILE   # any windows (coverage of the scan's top examples)

Needs the optional `grammar` extra (spaCy and en_core_web_sm); nothing else in the package imports spaCy.

Each window (1,024 model tokens) is decoded to text, parsed whole, and every model token gets the attributes of the spaCy
token that holds its first non-space byte. Per model token (arrays of shape (windows, 1024)):
  pos, tag, dep   codes into the POS / TAG / DEP tables stored in the file (0 = no spaCy token, e.g. whitespace)
  head            the model position of the head word's last piece (its own position for a root; -1 for none)
  last            the token ends its spaCy word (the word is complete when it is read)
  qsent           the token's sentence ends with "?"
A window whose bytes are not valid UTF-8 (a character cut at the window's edge) is kept but left unparsed (all zeros).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "grammar"
SPLITS = {"select": (0, 16000), "test": (16400, None)}  # auto-token@2's splits (as the sweep library)
DOMAINS = (0, 1)  # edu, web: the English domains
SAMPLE = 12000  # windows per split (at most: every English window of the selection split)
MODEL = "en_core_web_sm"


def sample_windows(split):
    from bonsai_interp.auto import context_tier as ct
    dom = ct._domains()
    lo, hi = SPLITS[split]
    hi = len(dom) if hi is None else hi
    wins = np.nonzero(np.isin(dom[lo:hi], DOMAINS))[0] + lo
    rng = np.random.default_rng([31, 0 if split == "select" else 1])
    return np.sort(rng.choice(wins, min(SAMPLE, len(wins)), replace=False))


def _nlp():
    try:
        import spacy
    except ImportError:
        sys.exit("spaCy is not installed: `uv sync --extra grammar` (or uv pip install spacy and the en_core_web_sm wheel)")
    return spacy.load(MODEL, disable=["ner", "lemmatizer"])


def _texts(ws):
    from bonsai_interp import explore
    from bonsai_interp.auto import context_tier as ct
    raw, toks = ct.vocab()["raw"], explore.corpus()
    for w in ws:
        parts = [raw[int(i)] for i in toks[w]]
        b = b"".join(parts)
        try:
            text = b.decode("utf-8")
        except UnicodeDecodeError:
            yield w, None, None, None
            continue
        ob = np.cumsum([0] + [len(p) for p in parts])  # model token byte starts (and the end)
        cb = np.cumsum([0] + [len(ch.encode("utf-8")) for ch in text])  # char -> byte
        ws_ = np.frombuffer(b, np.uint8)
        nonws = ~np.isin(ws_, list(b" \t\n\r\x0b\x0c"))
        yield w, text, ob, (cb, nonws)


def parse(ws, n_process=1, batch=16):
    """{pos, tag, dep, head, last, qsent} arrays (len(ws), 1024) and the code tables."""
    nlp = _nlp()
    T = 1024
    tabs = {"pos": [""], "tag": [""], "dep": [""]}
    idx = {k: {"": 0} for k in tabs}

    def code(k, v):
        if v not in idx[k]:
            idx[k][v] = len(tabs[k])
            tabs[k].append(v)
        return idx[k][v]

    out = {k: np.zeros((len(ws), T), np.uint8) for k in ("pos", "tag", "dep")}
    out["head"] = np.full((len(ws), T), -1, np.int16)
    out["last"] = np.zeros((len(ws), T), bool)
    out["qsent"] = np.zeros((len(ws), T), bool)
    meta = list(_texts(ws))
    good = [i for i, m in enumerate(meta) if m[1] is not None]
    t0 = time.time()
    docs = nlp.pipe((meta[i][1] for i in good), batch_size=batch, n_process=n_process)
    for k, (i, doc) in enumerate(zip(good, docs)):
        _, text, ob, (cb, nonws) = meta[i]
        S = np.array([cb[t.idx] for t in doc])
        E = np.array([cb[t.idx + len(t.text)] for t in doc])
        # first non-space byte of each model token (or its start when it is all whitespace)
        nxt = np.full(len(nonws) + 1, len(nonws))
        pos_nw = np.nonzero(nonws)[0]
        nxt[pos_nw] = pos_nw
        nxt = np.minimum.accumulate(nxt[::-1])[::-1]
        f = np.minimum(nxt[ob[:-1]], ob[1:] - 1)
        allws = nxt[ob[:-1]] >= ob[1:]
        sid = np.searchsorted(S, f, "right") - 1
        ok = (sid >= 0) & ~allws
        ok &= f < E[np.clip(sid, 0, None)]
        sid = np.where(ok, sid, -1)
        lastpiece = np.zeros(len(doc), np.int64) - 1  # spaCy token -> model position of its last piece
        for j in np.nonzero(ok)[0]:
            lastpiece[sid[j]] = j
        qs = np.zeros(len(doc), bool)
        for s in doc.sents:
            q = s.text.rstrip().endswith("?")
            qs[s.start:s.end] = q
        pos = np.array([code("pos", t.pos_) for t in doc], np.uint8)
        tag = np.array([code("tag", t.tag_) for t in doc], np.uint8)
        dep = np.array([code("dep", t.dep_) for t in doc], np.uint8)
        hd = np.array([t.head.i for t in doc])
        j = np.nonzero(ok)[0]
        s_ = sid[j]
        sp = np.array([t.pos_ != "SPACE" for t in doc])
        j, s_ = j[sp[s_]], s_[sp[s_]]
        out["pos"][i, j], out["tag"][i, j], out["dep"][i, j] = pos[s_], tag[s_], dep[s_]
        out["head"][i, j] = lastpiece[hd[s_]]
        out["last"][i, j] = ob[1:][j] >= E[s_]
        out["qsent"][i, j] = qs[s_]
        if (k + 1) % 250 == 0:
            print(f"  parsed {k + 1}/{len(good)} windows ({time.time() - t0:.0f} s)", flush=True)
    return out, {k: np.array(v) for k, v in tabs.items()}


def save(path, ws, out, tabs):
    import spacy
    np.savez_compressed(path, windows=np.asarray(ws, np.int32), **out, **{f"{k}_names": v for k, v in tabs.items()},
                        spacy=np.array(f"{MODEL} {spacy.util.get_package_version(MODEL)} spacy {spacy.__version__}"))


def load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["select", "test", "windows"])
    ap.add_argument("ws", nargs="*", type=int)
    ap.add_argument("--out")
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.what == "test" and not a.out:
        from bonsai_interp.grammar import run
        if not run.registered():
            sys.exit("the test split is parsed only after the protocol manifest is committed on its own (run.py protocol)")
    ws =sample_windows(a.what) if a.what != "windows" else np.array(sorted(set(a.ws)))
    path = Path(a.out) if a.out else OUT / f"bonsai1.parse.{a.what}.npz"
    t0 = time.time()
    out, tabs = parse(ws, a.procs)
    save(path, ws, out, tabs)
    print(f"wrote {path}: {len(ws)} windows in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
