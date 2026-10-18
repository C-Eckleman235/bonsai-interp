"""The auto tier: label and test every token-detector pole in a layer, with no LLM and no person.

  python -m bonsai_interp.auto.token_tier prereg --layer 12   CPU: hypotheses + tests -> data/auto/, manifest -> graph/auto/
  (commit the manifest ON ITS OWN: it pins the tests by sha256 before anything is scored)
  python -m bonsai_interp.auto.token_tier score --layer 12    GPU: one pass for the whole layer -> data/auto/ and a summary

Rule (protocol auto-token@1, fixed before any scoring):
1. Hypothesis. A pole is a token-detector candidate when one token (case and leading space folded) is the
   peak token of at least half of its 32 top scan examples, its corpus max is at least MIN_MAX, and the card
   would not flag it dense or positional. Every other pole is classified (dense, positional, weak, context)
   and left for people and agents: that is the point of the tier.
2. Test. Fire: 8 held-out corpus snippets (windows the scan never saw) of up to 24 tokens that END on the
   token. Near misses: the same 8 snippets with only that token replaced by a different word of the same
   shape (leading space, capitalisation, letters) and similar frequency, never a form of the same word.
   So each pair keeps the topic, length and style and drops only the trigger.
3. Judge with the quick-8+8@2 rule: AUROC >= 0.90, median fire >= 0.20, at most 1 near miss >= 0.30,
   reading each snippet after the same three neutral preambles as every other test.
A held result means "this pole responds to this token, in context, more than to a matched other word".
It is scrutiny level 0 ("auto"): a starting point that a person or agent can confirm, refine or dispute.
"""
import argparse
import datetime as dt
import gzip
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "auto"
MANIFESTS = ROOT / "graph" / "auto"
SCAN_WINDOWS = 8000  # the scan read windows 0..7999; everything after is held out
COVERAGE, MIN_MAX, SNIP, N = 0.5, 0.5, 24, 8
PROTOCOL = "auto-token@1"


def _key(s):
    k = s.strip().lower()
    return k if k else repr(s)


def _shape(s):
    w = s.strip()
    return (s[:1] == " ", w[:1].isupper(), w.isalpha())


def _commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def _paths(L):
    return OUT / f"bonsai1_L{L}.tests.jsonl.gz", MANIFESTS / f"bonsai1_L{L}.prereg.json", OUT / f"bonsai1_L{L}.results.jsonl.gz"


def prereg(a):
    from bonsai_interp import explore
    from bonsai_interp.graph import card
    L = a.layer
    s, toks, tok = explore.scan("bonsai1"), explore.corpus(), explore.tokenizer()
    ti = np.load(ROOT / "data" / "token_index_bonsai1.npz")
    vocab = len(tok)
    text = {i: tok.decode([i]) for i in range(vocab)}
    keys = {i: _key(t) for i, t in text.items()}
    by_key = {}
    for i, k in keys.items():
        by_key.setdefault(k, []).append(i)
    held = toks[SCAN_WINDOWS:]  # (windows, 1024)
    freq = np.bincount(held.ravel(), minlength=vocab)
    flat = held.ravel()
    order = np.argsort(flat, kind="stable")  # positions grouped by token id
    starts = np.searchsorted(flat[order], np.arange(vocab + 1))
    # substitutes: frequent single-token words, grouped by shape, sorted by frequency
    cands = [i for i in np.argsort(-freq)[:20000] if freq[i] >= 50 and text[i].strip().isalpha() and len(text[i].strip()) >= 3]
    by_shape = {}
    for i in cands:
        by_shape.setdefault(_shape(text[i]), []).append(i)
    # per shape: ids sorted by log frequency, so the closest-frequency words are a searchsorted away
    shape_ids = {sh: np.array(sorted(ids, key=lambda i: freq[i])) for sh, ids in by_shape.items()}
    shape_lf = {sh: np.log(freq[ids] + 1.0) for sh, ids in shape_ids.items()}

    def occurrences(ids, rng):
        pos = np.concatenate([order[starts[i]:starts[i + 1]] for i in ids]) if ids else np.array([], dtype=np.int64)
        w, p = pos // held.shape[1], pos % held.shape[1]
        ok = p >= 4
        w, p = w[ok], p[ok]
        if len(w) == 0:
            return []
        pick, seen = [], set()
        for j in rng.permutation(len(w)):
            if w[j] in seen:
                continue
            seen.add(w[j]); pick.append((int(w[j]), int(p[j])))
            if len(pick) == N:
                break
        return pick

    def substitute(t_id, k, rng):
        sh = _shape(text[t_id])
        if sh not in shape_ids:
            return None
        ids, lf = shape_ids[sh], shape_lf[sh]
        c = int(np.searchsorted(lf, np.log(freq[t_id] + 1.0)))
        window = ids[max(0, c - 30): c + 30]
        near = [int(i) for i in window if keys[i] != k and keys[i][:4] != k[:4] and k[:4] not in keys[i]]
        return near[rng.integers(len(near))] if near else None

    rows, counts = [], {}
    pid = lambda p, n: f"bonsai1/L{L}/N{n}{'+' if p == 0 else '-'}"
    for p in (0, 1):
        for n in range(12288):
            mx = float(ti["pole_max"][L, p, n])
            st = {"density": {"0.1": float(ti["dens10"][L, p, n])}, "peak_positions": {"first16": float(ti["early"][L, p, n]), "commonest_share": 0.0}}
            pos = s["top_pos"][L, p, :, n]
            vals, c = np.unique(pos, return_counts=True)
            st["peak_positions"]["commonest_share"] = float(c.max() / len(pos))
            peaks = toks[s["top_win"][L, p, :, n], s["top_pos"][L, p, :, n]]
            pk = [keys[int(t)] for t in peaks]
            top, cnt = max(((k, pk.count(k)) for k in set(pk)), key=lambda x: x[1])
            if mx < MIN_MAX:
                cls = "weak"
            elif st["density"]["0.1"] >= card.DENSE_10:
                cls = "dense"
            elif st["peak_positions"]["first16"] >= card.POSITIONAL or st["peak_positions"]["commonest_share"] >= card.POSITIONAL:
                cls = "positional"
            elif cnt / len(pk) < COVERAGE:
                cls = "context"
            else:
                cls = "token"
            counts[cls] = counts.get(cls, 0) + 1
            row = {"neuron": pid(p, n), "class": cls, "pole_max": round(mx, 4), "top_key": top, "coverage": round(cnt / len(pk), 3)}
            if cls == "token":
                rng = np.random.default_rng([L, p, n])
                ids = by_key[top]
                occ = occurrences(ids, rng)
                fire, quiet = [], []
                for w, q in occ:
                    sub = substitute(int(held[w, q]), top, rng)
                    if sub is None:
                        continue
                    ctx = held[w, max(0, q - SNIP + 1): q + 1].astype(np.int64)
                    fire.append(tok.decode(ctx))
                    quiet.append(tok.decode(np.concatenate([ctx[:-1], [sub]])))
                if len(fire) < N:
                    row["class"] = "token-untested"
                    counts["token"] -= 1
                    counts["token-untested"] = counts.get("token-untested", 0) + 1
                else:
                    shown = text[max(ids, key=lambda i: freq[i])]
                    row.update({"label": f"the token {shown!r} (auto: peak token of {cnt}/32 top corpus examples)", "fire": fire, "quiet": quiet})
            rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    tests_p, man_p, _ = _paths(L)
    blob = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    with gzip.open(tests_p, "wb", mtime=0) as f:
        f.write(blob)
    man = {"protocol": PROTOCOL, "layer": L, "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "toolchain_commit": _commit(), "tests_file": tests_p.relative_to(ROOT).as_posix(),
           "tests_sha256": hashlib.sha256(blob).hexdigest(), "classes": counts,
           "rule": {"coverage": COVERAGE, "min_max": MIN_MAX, "snippet_tokens": SNIP, "n": N, "held_out_windows": f"{SCAN_WINDOWS}..",
                    "judge": "quick-8+8@2 (AUROC >= 0.90, median fire >= 0.20, <= 1 near miss >= 0.30)"}}
    man_p.write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(counts))
    print(f"wrote {tests_p.relative_to(ROOT)} and {man_p.relative_to(ROOT)}\n"
          f"Commit the manifest ALONE before scoring: git add {man_p.relative_to(ROOT).as_posix()} && git commit -m \"Pre-register auto tier, layer {L}\"")


def score(a):
    import torch
    from bonsai_interp import explore
    from bonsai_interp.env import DEVICE, limit_memory
    from bonsai_interp.graph.protocols import PROTOCOLS
    from bonsai_interp.model import StopForward, load
    from bonsai_interp.probe.textinv import PREAMBLES
    L = a.layer
    tests_p, man_p, res_p = _paths(L)
    man = json.loads(man_p.read_text(encoding="utf-8"))
    blob = gzip.open(tests_p, "rb").read()
    if hashlib.sha256(blob).hexdigest() != man["tests_sha256"]:
        sys.exit("the tests file does not match its pre-registered sha256")
    committed = subprocess.run(["git", "log", "--format=%H", "--", man_p.relative_to(ROOT).as_posix()], cwd=ROOT, capture_output=True, text=True).stdout.split()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", man_p.relative_to(ROOT).as_posix()], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if not committed or dirty:
        sys.exit(f"commit {man_p.relative_to(ROOT)} on its own before scoring (pre-registration)")
    rows = [json.loads(x) for x in blob.decode("utf-8").splitlines()]
    todo = [r for r in rows if "fire" in r]
    texts = sorted({t for r in todo for t in r["fire"] + r["quiet"]})
    print(f"layer {L}: {len(todo)} token poles, {len(texts)} distinct snippets x {len(PREAMBLES)} preambles")
    limit_memory(0.8)
    model, tok, rec = load("bonsai1")
    tok.padding_side = "right"
    s = explore.scan("bonsai1")
    need = {}  # text -> [(row index, sign)]
    for i, r in enumerate(todo):
        for t in r["fire"] + r["quiet"]:
            need.setdefault(t, []).append(i)
    pmax = torch.tensor(s["top_val"][L, 0, 0, :], device=DEVICE).float()
    nmax = torch.tensor(s["top_val"][L, 1, 0, :], device=DEVICE).float()
    want ={t: sorted({(todo[i]["neuron"]) for i in need[t]}) for t in texts}
    t0, B = time.time(), a.batch
    sums = {}
    for pre in PREAMBLES:
        k = len(tok(pre).input_ids)
        for b in range(0, len(texts), B):
            chunk = texts[b:b + B]
            enc = tok([pre + t for t in chunk], return_tensors="pt", padding=True).to(DEVICE)
            rec.mode, rec.layers, rec.stop_at = "act", {L}, L
            try:
                with torch.no_grad():
                    model.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
            except StopForward:
                pass
            finally:
                rec.mode, rec.stop_at = None, None
            x = rec.store.pop(L).float()
            rec.clear()
            mask = enc.attention_mask.bool()
            mask[:, :k] = False
            m = mask[..., None]
            pos = torch.where(m, x, torch.full_like(x, -1e9)).amax(1) / pmax
            neg = torch.where(m, -x, torch.full_like(x, -1e9)).amax(1) / nmax
            pos, neg = pos.cpu().numpy(), neg.cpu().numpy()
            for j, t in enumerate(chunk):
                for nid in want[t]:
                    n = int(nid.split("/N")[1][:-1])
                    v = pos[j, n] if nid.endswith("+") else neg[j, n]
                    sums[(t, nid)] = sums.get((t, nid), 0.0) + float(v) / len(PREAMBLES)
            if b // B % 50 == 0:
                done = (PREAMBLES.index(pre) * len(texts) + b + len(chunk)) / (len(PREAMBLES) * len(texts))
                print(f"  {done:.0%}  {time.time() - t0:.0f} s", end="\r", flush=True)
    print()
    rule = PROTOCOLS["quick-8+8@2"]["rule"]
    out, tally = [], {}
    for r in rows:
        if "fire" in r:
            per = [{"set": "fire", "text": t, "act": round(sums[(t, r["neuron"])], 4)} for t in r["fire"]] + \
                  [{"set": "quiet", "text": t, "act": round(sums[(t, r["neuron"])], 4)} for t in r["quiet"]]
            res = rule({"per_sentence": per})
            fire = sorted(p["act"] for p in per if p["set"] == "fire")
            r = {**r, "per_sentence": per, "outcome": res, "median_fire": round(float(np.median(fire)), 3)}
            tally[res] = tally.get(res, 0) + 1
        out.append(r)
    with gzip.open(res_p, "wb", mtime=0) as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out).encode("utf-8"))
    summ = {"layer": L, "protocol": PROTOCOL, "scored": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "toolchain_commit": _commit(), "prereg_commit": committed[-1][:12], "classes": man["classes"], "token_outcomes": tally,
            "minutes": round((time.time() - t0) / 60, 1), "results_file": res_p.relative_to(ROOT).as_posix(),
            "results_sha256": hashlib.sha256(gzip.open(res_p, "rb").read()).hexdigest()}
    (MANIFESTS / f"bonsai1_L{L}.summary.json").write_text(json.dumps(summ, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summ, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prereg", "score"):
        p = sub.add_parser(name)
        p.add_argument("--layer", type=int, required=True)
        p.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    {"prereg": prereg, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
