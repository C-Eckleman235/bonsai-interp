"""Build a tokenized, mixed-domain corpus of fixed-length windows.

The essay stresses broad data (some neurons only become legible with it) and
long context (short windows make long-context neurons look polysemantic), so
documents are packed into 1024-token windows from several domains.

Usage: python -m mono.corpus --tokens 30_000_000
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from mono.model import MODELS, ROOT

SOURCES = [  # (name, hf path, config, text field, share)
    ("edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", "text", 0.45),
    ("web", "HuggingFaceFW/fineweb", "sample-10BT", "text", 0.25),
    ("multi", "HuggingFaceFW/fineweb-2", None, "text", 0.15),
    ("code", "codeparrot/codeparrot-clean-valid", None, "content", 0.15),
]
MULTI_LANGS = ["deu_Latn", "fra_Latn", "spa_Latn", "rus_Cyrl", "cmn_Hani", "jpn_Jpan", "arb_Arab", "kor_Hang"]
WINDOW = 1024


def stream(path, config, field):
    if path.endswith("fineweb-2"):
        its = [load_dataset(path, lang, split="train", streaming=True) for lang in MULTI_LANGS]
        for rows in zip(*its):
            for r in rows:
                yield r[field]
    else:
        ds = load_dataset(path, config, split="train", streaming=True)
        for r in ds:
            yield r[field]


def build(total_tokens: int, out: Path):
    tok = AutoTokenizer.from_pretrained(MODELS["bonsai1"])  # all three share the Qwen3 tokenizer
    eos = tok.convert_tokens_to_ids("<|endoftext|>")
    windows, domains = [], []
    for d_i, (name, path, config, field, share) in enumerate(SOURCES):
        budget = int(total_tokens * share)
        buf, got = [], 0
        for text in stream(path, config, field):
            buf.extend(tok(text).input_ids + [eos])
            while len(buf) >= WINDOW:
                windows.append(np.array(buf[:WINDOW], dtype=np.uint32))
                domains.append(d_i)
                buf = buf[WINDOW:]
                got += WINDOW
            if got >= budget:
                break
        print(f"{name}: {got:,} tokens", flush=True)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(windows))
    np.savez(out, tokens=np.stack(windows)[order], domain=np.array(domains, dtype=np.uint8)[order],
             domain_names=np.array([s[0] for s in SOURCES]))
    print("saved", out, len(windows), "windows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=30_000_000)
    ap.add_argument("--out", default=str(ROOT / "data" / "corpus.npz"))
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    build(a.tokens, Path(a.out))
