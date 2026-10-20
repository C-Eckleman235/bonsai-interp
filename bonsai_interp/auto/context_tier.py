"""The auto tier, version 2 (protocol auto-token@2): word hits by frequency, a context screen, dossiers.

auto-token@1 labelled a pole by the commonest peak token of its 32 top examples and swapped that token out;
at layer 12 only 34 of 644 such poles held, because most poles respond to their peak token only in particular
contexts. This tier asks the corpus first and the top examples never:

  python -m bonsai_interp.auto.context_tier protocol --layer 20   CPU: write the rule manifest (commit it ALONE, before any run)
  python -m bonsai_interp.auto.context_tier hits     --layer 20   GPU, stage 1: strong activations by token over windows 0..7999
  python -m bonsai_interp.auto.context_tier screen   --layer 20   GPU, stage 2: context model + minimal pairs on windows 8000..15999
  python -m bonsai_interp.auto.context_tier prereg   --layer 20   CPU: auto tests from windows 16400.. (commit the manifest ALONE)
  python -m bonsai_interp.auto.context_tier score    --layer 20   GPU: judge the auto tests
  python -m bonsai_interp.auto.context_tier export   --layer 20   CPU: the survey map's screen layer (site/screens.py)
  python -m bonsai_interp.auto.context_tier backup   --layer 20   hash data/auto2 files into the summary, upload them to releases
  python -m bonsai_interp.auto.context_tier refit    --layer 20   CPU: refit stage 2 from its per-occurrence store (new features)
  python -m bonsai_interp.auto.context_tier protocol-all|hits-all|screen-all|prereg-all|score-all|export-all --layers 0,1,2
      the same stages for many layers under ONE registration: one forward pass per batch reads every layer (hits, screen)
  python -m bonsai_interp.auto.context_tier dossier  --neuron bonsai1/L20/N123+ [--links]   stage 3: input for an agent
  (--smoke DIR runs any stage without the manifest checks and writes to DIR: for CPU tests only)
  Measured on an RTX 5080 at layer 20: hits 23 min (6,000 tok/s, batch 8, 9.6 GB uncapped), screen 28 min (8.3 GB at
  batch 8), score 19 min (1,185 tests, batch 16, 5.8 GB). Under BONSAI_GPU_FRACTION=0.35 (5.6 GB) the model takes
  about 3 GB, and attention over 1,024-token windows needs --batch 2 for hits and screen; score fits at --batch 16.

Corpus windows are split three ways, so no stage reads the windows a later stage is judged on:
  0..7999      selection (the scan read these too): stage 1
  8000..15999  screening: stage 2 fits the context model and runs the minimal pairs here
  16400..      testing: read only by `prereg` (texts, never activations) and `score`, after the tests are pinned
  (16000..16399 are unused: the CPU smoke test of this code scored snippets from them at layer 20)

Rule (protocol auto-token@2; every number lives in RULE and is pinned by the protocol manifest):
1. Word hits. A pole is eligible when its corpus max >= min_max and the card would flag it neither dense nor
   positional. Over the selection windows, from window position min_pos on (earlier positions are a window-start
   artifact, FINDINGS 2026-10-18), an activation is strong when it reaches `strong` of the pole's corpus max. Per (pole, token) and (pole, previous token, token): n strong, precision = P(strong | token), lift =
   precision / the pole's base rate, coverage = the share of the pole's strong activations the token explains.
   A word hit has n >= min_strong, coverage >= min_coverage and lift >= min_lift. A pole's triggers are its
   word hits, plus the token of any bigram hit whose token is not itself a word hit, at most triggers_per_pole,
   by coverage.
2. Context screen. For each (pole, trigger): up to occ_per_trigger occurrences of the trigger in the screening
   windows, scored in their own context. Features (families in brackets; all read only what precedes the trigger
   within a test snippet's reach, except the predictive ones):
     [context]      previous token and its word class; the nearest content word 1-3 tokens back; where the token
                    sits in its word (start, 2nd, 3rd+ piece); digit position in a number; script and language
                    of the preceding text
     [format]       line position, sentence position, code-like text, line kind (list item, numbered, table row,
                    '#' heading or comment, indented, markup, key: value, plain), inside quotes
     [copy]         the same token or the same (previous, token) pair earlier in the snippet
     [predictive]   the next token and its class: the model cannot see it, but a pole can fire because it predicts
                    it, and a test can select snippets by their continuation
     [collocation]  PMI of (previous token, token) over the selection windows
     [topic]        the pole's own mean activation over the preceding snippet (a topic gate, not a token)
     [positional]   window position
     (the corpus domain is tabulated only: a snippet does not carry which corpus a window came from)
   A rule of one or two (feature == value) conditions is fitted by information gain on strong. Bins:
     rare         fewer than min_occ occurrences
     a family's bin  the rule has precision >= rule_precision, recall >= rule_recall and beats the rate outside it by
                  rule_margin; its families decide the bin: positional or topic first (no text-only test can
                  select them), then copy (repetition values only), predictive, collocation, context, format
     lexical      no such rule, and strong on >= lexical_rate of all occurrences
     unexplained  none of the above
   (a rule that splits the occurrences wins over a high overall rate: it is the more specific claim)
   Minimal pairs on screening snippets check the bin: lexical, swap the trigger for a same-shape word, a look-alike
   spelling (rain -> brain: 'spelling' when it keeps firing) and an inflection (take -> takes: 'lexeme'); context
   on the previous token, swap that token out of (and into) the rule's context. Recorded, not a gate.
3. Auto test (the protocol's own test), for the lexical, context, format, copy, predictive and collocation bins.
   Snippets of up to `snippet` tokens from the testing windows that END on the trigger. Fire: 8 occurrences where
   the rule holds (any, for lexical). Near misses: lexical, the same 8 with only the trigger swapped (as
   auto-token@1); the others, 4 occurrences of the trigger where the rule fails (the word without the context)
   and 4 fire snippets with the trigger swapped (the context without the word), or 8 rule-fails when no
   substitute exists. Each snippet is read at its last token (the trigger, or its swap), averaged over the three
   standard preambles, and judged with the quick-8+8@2 thresholds. Scrutiny 0, like auto-token@1.
4. Dossier (stage 3): the card, the stage 1 and 2 tables, screening examples and, with --links, the strongest
   upstream feeders and downstream readers (play step / step --down). Nothing in it is scored on a test
   sentence: an agent labels the pole from it and then writes a FRESH quick-8+8@3 test.
"""
import argparse
import copy
import datetime as dt
import gzip
import hashlib
import json
import subprocess
import sys
import time
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "auto2"
MANIFESTS = ROOT / "graph" / "auto"
PROTOCOL = "auto-token@2"
N = 12288
SELECT, SCREEN, TEST = (0, 8000), (8000, 16000), (16400, None)  # 16000..16399 were read by the CPU smoke test: never test windows
RULE = {
    "strong": 0.3, "min_max": 0.5, "min_pos": 64,
    "min_strong": 8, "min_coverage": 0.05, "min_lift": 10.0, "triggers_per_pole": 3,
    "occ_per_trigger": 256, "min_occ": 24, "min_support": 8,
    "lexical_rate": 0.5, "rule_precision": 0.5, "rule_recall": 0.3, "rule_margin": 0.3, "second_gain": 0.15,
    "template_n": 8, "template_ctx": 32, "template_move": 0.15,
    "snippet": 48, "n": 8, "n_rule_fail": 4,
    "tests_per_pole": 0,  # 0: test every testable hit (up to triggers_per_pole); k: only each pole's k best (see TEST_PRIORITY)
    "judge": "quick-8+8@2 thresholds (AUROC >= 0.90, median fire >= 0.20, <= 1 near miss >= 0.30) on the activation at the "
             "snippet's last token, averaged over the three standard preambles",
}
# feature families; a rule's families decide its bin (see bin_of)
CONTEXT = ("prev_tok", "prev_class", "prev_content", "piece", "num_pos", "script", "lang")
FORMAT = ("line_pos", "sent_pos", "codeish", "line_kind", "in_quotes")
COPY = ("tok_repeat", "bigram_repeat")
PREDICTIVE = ("next_tok", "next_class")
COLLOCATION = ("pmi",)
POSITIONAL = ("win_pos",)
TOPIC = ("ctx_mean",)  # the pole's own mean activation over the preceding snippet: per hit, added in screen
FEATURES = CONTEXT + FORMAT + COPY + PREDICTIVE + COLLOCATION + POSITIONAL
TABLE_ONLY = ("domain",)  # tabulated, never in a rule (see the docstring)
UNTESTABLE = ("positional", "topic")  # bins whose rule a text-only test cannot select
TESTED = ("lexical", "context", "format", "copy", "predictive", "collocation")
MODULE = Path(__file__).resolve()


# ---------------------------------------------------------------- files, manifests, guards

def _commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def _code_sha():
    return hashlib.sha256(MODULE.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _paths(L, out=OUT, man=MANIFESTS):
    b = f"bonsai1_L{L}"
    return {"protocol": man / f"{b}.auto2.protocol.json", "prereg": man / f"{b}.auto2.prereg.json", "summary": man / f"{b}.auto2.summary.json",
            "hits": out / f"{b}.hits.npz", "table": out / f"{b}.hits.jsonl.gz", "screen": out / f"{b}.context.jsonl.gz",
            "tests": out / f"{b}.auto2.tests.jsonl.gz", "results": out / f"{b}.auto2.results.jsonl.gz", "dossiers": out / "dossiers",
            "occ": out / f"{b}.occ.npz", "refit": out / f"{b}.context.refit.jsonl.gz", "amend": man / f"{b}.auto2.amendments.json"}


def _committed_alone_clean(p):
    rel = p.relative_to(ROOT).as_posix()
    log = subprocess.run(["git", "log", "--format=%H", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout.split()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return (log[-1] if log else None), bool(dirty)


def _setup(a):
    """Output paths for this run. Real runs need the committed protocol manifest and this file unchanged since it."""
    if a.smoke:
        d = Path(a.smoke)
        d.mkdir(parents=True, exist_ok=True)
        return _paths(a.layer, d, d), {"rule": RULE}
    P = _paths(a.layer)
    if not P["protocol"].exists():
        sys.exit(f"no protocol manifest: run `protocol --layer {a.layer}` and commit {P['protocol'].relative_to(ROOT)} on its own first")
    first, dirty = _committed_alone_clean(P["protocol"])
    if not first or dirty:
        sys.exit(f"commit {P['protocol'].relative_to(ROOT)} on its own before running any stage (pre-registration)")
    man = json.loads(P["protocol"].read_text(encoding="utf-8"))
    if _code_sha() not in _allowed_shas({"amend": P["amend"]}, man):
        sys.exit(f"{MODULE.name} changed since the protocol was registered and the change is not a committed speed amendment "
                 f"(amend --layer {a.layer}): that is a new protocol version, not this one")
    OUT.mkdir(parents=True, exist_ok=True)
    return P, man


def _write_gz(p, rows):
    blob = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    with gzip.GzipFile(p, "wb", mtime=0) as f:
        f.write(blob)
    return hashlib.sha256(blob).hexdigest()


def _read_gz(p):
    return [json.loads(x) for x in gzip.open(p, "rb").read().decode("utf-8").splitlines()]


def _windows(span, a):
    from bonsai_interp import explore
    toks = explore.corpus()
    lo, hi = span[0], span[1] if span[1] is not None else len(toks)
    if getattr(a, "windows", None):
        hi = min(hi, lo + a.windows)
    t = getattr(a, "tokens", None) or toks.shape[1]
    return toks[lo:hi, :t], _domains()[lo:hi], lo


@lru_cache(None)
def _domains():
    return np.load(ROOT / "data" / "corpus.npz")["domain"]


def pid(p, n, L):
    return f"bonsai1/L{L}/N{n}{'+' if p == 0 else '-'}"


# ---------------------------------------------------------------- vocabulary tables (byte-exact, no per-occurrence decode)

def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


_CLOSED = [  # first match wins
    ("negation", "not never no n't nothing none nobody"),
    ("intensifier", "very so too quite really extremely highly most more less rather pretty fairly incredibly particularly "
                    "especially absolutely completely totally truly deeply"),
    ("determiner", "the a an this that these those my your his her its our their some any every each another"),
    ("preposition", "of in on at to for with from by about into over under between through after before without within "
                    "during against among across upon"),
    ("conjunction", "and or but nor yet because while although though if than whether"),
    ("pronoun", "i you he she it we they me him us them who whom which what someone something"),
    ("auxiliary", "is are was were be been being am has have had do does did will would can could should may might must shall "
                  "'s 're 've 'll 'd"),
]
CLOSED = {w: c for c, ws in _CLOSED for w in ws.split()}
PHRASE = {"negation": "a negation", "intensifier": "an intensifier (very, so, too, ...)", "determiner": "a determiner (the, a, this, ...)",
          "preposition": "a preposition", "conjunction": "a conjunction", "pronoun": "a pronoun", "auxiliary": "an auxiliary verb",
          "newline": "a line break", "comma": "a comma", "sentence-end": "a full stop or other sentence end", "open-bracket": "an opening bracket",
          "close-bracket": "a closing bracket", "quote": "a quotation mark", "colon": "a colon or semicolon", "symbol": "a code symbol",
          "number": "a number", "capitalized": "a capitalised word", "subword": "a word fragment", "adverb-ly": "an -ly adverb",
          "verb-ing": "an -ing form", "verb-ed": "an -ed form", "noun-suffix": "a -tion/-ment/-ness noun",
          "adjective-suffix": "an adjective (-ous, -ful, -ive, -al, ...)", "word": "an ordinary word", "space": "whitespace",
          "non-latin": "a non-Latin-script word", "byte": "a byte fragment"}
SCRIPTS = {"LATIN": "latin", "CYRILLIC": "cyrillic", "ARABIC": "arabic", "GREEK": "greek", "HEBREW": "hebrew", "DEVANAGARI": "devanagari",
           "CJK": "cjk", "HIRAGANA": "cjk", "KATAKANA": "cjk", "HANGUL": "hangul", "THAI": "thai"}
CODE_CHARS = set("{}()[];=<>_/\\|#$@*&^%~`")


def _script_of(ch):
    try:
        return SCRIPTS.get(unicodedata.name(ch).split(" ")[0], "other")
    except ValueError:
        return "other"


def _word_class(s):
    if "\n" in s:
        return "newline"
    w = s.strip()
    if not w:
        return "space"
    if "�" in s:
        return "byte"
    if all(unicodedata.category(c)[0] in "PS" for c in w):
        if w in "([{" or w[-1] in "([{":
            return "open-bracket"
        if w[0] in ")]}":
            return "close-bracket"
        if w[0] in "\"'“”‘’`«»":
            return "quote"
        if w == ",":
            return "comma"
        if w[-1] in ".!?。！？":
            return "sentence-end"
        if w[-1] in ":;":
            return "colon"
        return "symbol"
    if any(c.isalpha() and _script_of(c) != "latin" for c in w):
        return "non-latin"
    if any(c.isdigit() for c in w) and not any(c.isalpha() for c in w):
        return "number"
    if not s.startswith(" ") and w.isalpha():
        return "subword"
    lw = w.lower()
    if lw in CLOSED:
        return CLOSED[lw]
    if w[:1].isupper():
        return "capitalized"
    for suf, c in (("ly", "adverb-ly"), ("ing", "verb-ing"), ("ed", "verb-ed")):
        if lw.endswith(suf) and len(lw) > len(suf) + 2:
            return c
    if lw.endswith(("tion", "sion", "ment", "ness", "ity", "ism", "ance", "ence")):
        return "noun-suffix"
    if lw.endswith(("ous", "ful", "ive", "able", "ible", "al", "ic", "less", "ish")) and len(lw) > 4:
        return "adjective-suffix"
    return "word"


@lru_cache(None)
def vocab():
    """Per token id: decoded text, raw bytes, word class, and newline / sentence-end flags."""
    from bonsai_interp import explore
    tok = explore.tokenizer()
    V = len(tok)
    b2u = {u: b for b, u in _bytes_to_unicode().items()}
    pieces = tok.convert_ids_to_tokens(list(range(V)))
    raw = []
    for p in pieces:
        try:
            raw.append(bytes(b2u[c] for c in p))
        except (KeyError, TypeError):
            raw.append((p or "").encode("utf-8"))
    text = [r.decode("utf-8", errors="replace") for r in raw]
    cls = np.array([_word_class(t) for t in text], dtype=object)
    nl = np.array(["\n" in t for t in text])
    send = np.array([t.strip()[-1:] in tuple(".!?。！？") if t.strip() else False for t in text])
    lead = np.array([t[:1] in (" ", "\t") for t in text])
    an0 = np.array([t.lstrip()[:1].isalnum() for t in text])
    an1 = np.array([t[-1:].isalnum() for t in text])
    digit = np.array([t.strip().isdigit() for t in text])
    return {"V": V, "text": text, "raw": raw, "cls": cls, "nl": nl, "send": send, "lead": lead, "an0": an0, "an1": an1, "digit": digit}


@lru_cache(None)
def bigram_table():
    """Unigram and bigram counts over the selection windows, for the PMI feature."""
    from bonsai_interp import explore
    t = explore.corpus()[SELECT[0]:SELECT[1]].astype(np.int64)
    V = vocab()["V"]
    uni = np.bincount(t.ravel(), minlength=V)
    keys, cnt = np.unique((t[:, :-1] * V + t[:, 1:]).ravel(), return_counts=True)
    return {"uni": uni, "keys": keys, "cnt": cnt, "n": t.size, "V": V}


def _pmi(prev, tok, bt):
    k = prev * bt["V"] + tok
    i = int(np.searchsorted(bt["keys"], k))
    c = int(bt["cnt"][i]) if i < len(bt["keys"]) and bt["keys"][i] == k else 0
    if c < 3:
        return "rare-pair"
    v = np.log(c * bt["n"] / (bt["uni"][prev] * bt["uni"][tok]))
    return "pmi<0" if v < 0 else "pmi 0-2" if v < 2 else "pmi 2-5" if v < 5 else "pmi 5+"


_LANG = {"en": "the and of to is in that it was for with as on are this be",
         "es": "el la los las de que y en un una por con para es del se",
         "fr": "le la les de des et est un une que du en pour dans qui pas",
         "de": "der die das und ist nicht ein eine zu den mit von sich auf für",
         "pt": "o a os as de que e do da em um uma para com não é",
         "it": "il la di che e un una per non è del della con sono gli",
         "nl": "de het een en van is dat niet op te zijn voor met"}
LANG = {k: set(v.split()) for k, v in _LANG.items()}
CONTENT = {"word", "capitalized", "adverb-ly", "verb-ing", "verb-ed", "noun-suffix", "adjective-suffix", "non-latin"}


def _line_kind(prefix):
    """What kind of line the token is on, from the line's start up to the token (None: the line began before the snippet)."""
    import re
    if prefix is None:
        return "long-line"
    for kind, pat in (("list-item", r"\s*[-*•]\s"), ("numbered", r"\s*\d+[.)]\s"), ("heading-or-comment", r"#{1,6}\s"),
                      ("indented", r"(\t| {4})"), ("markup", r"\s*<"), ("key-value", r"\s*[\w.\-]+\s*[:=]\s")):
        if re.match(pat, prefix):
            return kind
    return "table-row" if "|" in prefix else "plain"


def features(ids, p, dom, vb, bt=None):
    """Context features of the token at position p of a window (ids). All but next_tok / next_class read only what
    precedes it, and only as far back as a test snippet reaches (RULE["snippet"] tokens including the trigger)."""
    lo = max(0, p - RULE["snippet"] + 1)
    tok = int(ids[p])
    prev = int(ids[p - 1]) if p > 0 else -1
    f = {"prev_tok": f"t{prev}" if prev >= 0 else "none", "prev_class": vb["cls"][prev] if prev >= 0 else "none"}
    # the nearest preceding content word within 3 tokens, joined from its pieces
    f["prev_content"] = "none"
    for k in range(1, 4):
        j = p - k
        if j < lo:
            break
        c = vb["cls"][ids[j]]
        if c in CONTENT or (c == "subword" and vb["an1"][ids[j]]):
            a_ = j
            while a_ > lo and not vb["lead"][ids[a_]] and vb["an0"][ids[a_]] and vb["an1"][ids[a_ - 1]]:
                a_ -= 1
            w = b"".join(vb["raw"][int(t)] for t in ids[a_:j + 1]).decode("utf-8", errors="ignore").strip().lower()
            if w:
                f["prev_content"] = f"w:{w}"
            break
    # where the token sits in its word, from the left (whether the word goes on is not visible yet)
    if not vb["an0"][tok]:
        f["piece"] = "non-word"
    elif vb["lead"][tok] or prev < 0 or not vb["an1"][prev]:
        f["piece"] = "word-start"
    else:
        k = 1
        while p - k - 1 >= lo and not vb["lead"][ids[p - k]] and vb["an0"][ids[p - k]] and vb["an1"][ids[p - k - 1]]:
            k += 1
        f["piece"] = "piece-2" if k == 1 else "piece-3+"
    k = 0
    while k < 6 and p - k >= lo and vb["digit"][ids[p - k]]:
        k += 1
    f["num_pos"] = "not-digit" if k == 0 else f"digit-{k}" if k < 5 else "digit-5+"
    k = 0
    while k < 16 and p - 1 - k >= 0 and not vb["nl"][ids[p - 1 - k]]:
        k += 1
    f["line_pos"] = "line-start" if k == 0 else "1-3" if k <= 3 else "4-15" if k < 16 else "16+"
    k = 0
    while k < 5 and p - 1 - k >= 0 and not (vb["nl"][ids[p - 1 - k]] or vb["send"][ids[p - 1 - k]]):
        k += 1
    f["sent_pos"] = "first" if k == 0 else "2-5" if k < 5 else "6+"
    f["win_pos"] = "64-255" if p < 256 else "256-511" if p < 512 else "512+"
    before = np.asarray(ids[lo:p])
    pre = b"".join(vb["raw"][int(t)] for t in before).decode("utf-8", errors="ignore")
    sc = {}
    for ch in pre:
        if ch.isalpha():
            s_ = _script_of(ch)
            sc[s_] = sc.get(s_, 0) + 1
    f["script"] = max(sc, key=sc.get) if sc else "none"
    words = b"".join(vb["raw"][int(t)] for t in ids[max(lo, p - 20):p]).decode("utf-8", errors="ignore").lower().split()
    score = {k_: sum(w in v for w in words) for k_, v in LANG.items()}
    best = max(score, key=score.get)
    unique_best = score[best] >= 2 and list(score.values()).count(score[best]) == 1
    f["lang"] = (best if unique_best else "unknown") if f["script"] == "latin" else f["script"]
    ctx = pre + vb["text"][tok]
    sym = sum(ch in CODE_CHARS for ch in ctx)
    f["codeish"] = "code-like" if ctx and (sym / len(ctx) >= 0.06 or "\n    " in ctx or "\t" in ctx) else "prose-like"
    cut = pre.rfind("\n")
    line = pre[cut + 1:] if cut >= 0 else (pre if lo == 0 else None)
    f["line_kind"] = _line_kind(line)
    q = line if line is not None else pre
    f["in_quotes"] = "in-quotes" if (q.count('"') % 2 == 1 or q.count("“") > q.count("”")) else "no"
    seen = np.flatnonzero(before == tok)
    f["tok_repeat"] = "new" if not len(seen) else "repeat-near" if p - (lo + int(seen[-1])) <= 16 else "repeat-far"
    pair = len(before) > 1 and prev >= 0 and bool(np.any((before[:-1] == prev) & (before[1:] == tok)))
    f["bigram_repeat"] = "bigram-repeat" if pair else "no"
    nxt = int(ids[p + 1]) if p + 1 < len(ids) else -1
    f["next_tok"] = f"t{nxt}" if nxt >= 0 else "none"
    f["next_class"] = vb["cls"][nxt] if nxt >= 0 else "none"
    f["pmi"] = _pmi(prev, tok, bt) if bt is not None and prev >= 0 else "none"
    f["domain"] = ["edu", "web", "multi", "code"][int(dom)] if dom is not None else "none"
    return f


def value_text(feat, v, vb=None):
    if feat in ("prev_tok", "next_tok") and v.startswith("t"):
        return repr((vb or vocab())["text"][int(v[1:])])
    if feat == "prev_content" and v.startswith("w:"):
        return repr(v[2:])
    return v


def condition_phrase(feat, v, vb):
    if feat == "prev_tok":
        return f"right after {value_text(feat, v, vb)}"
    if feat == "prev_class":
        return f"right after {PHRASE.get(v, v)}"
    if feat == "line_pos":
        return "at the start of a line" if v == "line-start" else f"{v} tokens into a line"
    if feat == "sent_pos":
        return "as the first word of a sentence" if v == "first" else f"{v} tokens into a sentence"
    if feat == "win_pos":
        return f"within window positions {v}"
    if feat == "script":
        return f"in {v}-script text"
    if feat == "domain":
        return {"multi": "in non-English (multilingual) text", "code": "in source code"}.get(v, f"in {v} text")
    if feat == "codeish":
        return "in code-like text" if v == "code-like" else "in prose"
    if feat == "prev_content":
        return f"shortly after the word {value_text(feat, v, vb)}"
    if feat == "piece":
        return {"word-start": "at the start of a word", "piece-2": "as the second piece of a word",
                "piece-3+": "as the third or later piece of a word"}.get(v, f"as a {v} token")
    if feat == "num_pos":
        return f"as digit {v.split('-', 1)[1]} of a number" if v.startswith("digit") else "outside a number"
    if feat == "lang":
        return {"en": "in English text", "es": "in Spanish text", "fr": "in French text", "de": "in German text",
                "pt": "in Portuguese text", "it": "in Italian text", "nl": "in Dutch text"}.get(v, f"in {v} text")
    if feat == "line_kind":
        return {"list-item": "in a bulleted list item", "numbered": "in a numbered list item", "table-row": "in a table row",
                "heading-or-comment": "in a '#' heading or comment", "indented": "on an indented line", "markup": "in a markup line",
                "key-value": "in a key: value line", "long-line": "deep in a long line", "plain": "in a plain line"}.get(v, v)
    if feat == "in_quotes":
        return "inside a quotation or string" if v == "in-quotes" else "outside quotes"
    if feat == "tok_repeat":
        return {"repeat-near": "when the same token appeared within the last 16 tokens",
                "repeat-far": "when the same token appeared 17-47 tokens back",
                "new": "on its first appearance in the snippet"}.get(v, v)
    if feat == "bigram_repeat":
        return "repeating a token pair seen earlier in the snippet" if v == "bigram-repeat" else "not repeating an earlier pair"
    if feat == "next_tok":
        return f"right before {value_text(feat, v, vb)}"
    if feat == "next_class":
        return f"right before {PHRASE.get(v, v)}"
    if feat == "pmi":
        return {"pmi 5+": "in a fixed pair with the previous token (PMI >= 5)", "pmi 2-5": "in a common pair with the previous token",
                "pmi 0-2": "in a loose pair with the previous token", "pmi<0": "after an unusual previous token (PMI < 0)",
                "rare-pair": "after a previous token it rarely follows"}.get(v, v)
    if feat == "ctx_mean":
        return f"when the pole is already active over the preceding text (mean {v})"
    return f"{feat} = {v}"


# ---------------------------------------------------------------- model helpers

def _forward(model, rec, x, L):
    import torch
    from bonsai_interp.model import StopForward
    rec.mode, rec.layers, rec.stop_at = "act", {L}, L
    try:
        with torch.no_grad():
            model.model(input_ids=x)
    except StopForward:
        pass
    finally:
        rec.mode, rec.stop_at = None, None
    a = rec.store.pop(L)
    rec.clear()
    return a


def _pole_max(L):
    from bonsai_interp import explore
    s = explore.scan("bonsai1")
    return s["top_val"][L, 0, 0].astype(np.float64), s["top_val"][L, 1, 0].astype(np.float64)  # both positive magnitudes


def eligibility(L):
    """Per pole (2, N): 'eligible', 'weak', 'dense' or 'positional' (the card's flags, as auto-token@1)."""
    from bonsai_interp import explore
    from bonsai_interp.graph import card, which
    ti, s = which.load_index(), explore.scan("bonsai1")
    out = np.full((2, N), "eligible", dtype=object)
    for p in (0, 1):
        pos = s["top_pos"][L, p]  # (32, N)
        share = np.array([np.unique(pos[:, n], return_counts=True)[1].max() / pos.shape[0] for n in range(N)])
        early = ti["early"][L, p].astype(np.float64)
        out[p][(early >= card.POSITIONAL) | (share >= card.POSITIONAL)] = "positional"
        out[p][ti["dens10"][L, p] >= card.DENSE_10] = "dense"
        out[p][ti["pole_max"][L, p] < RULE["min_max"]] = "weak"
    return out


class _Acc:
    """Sparse counter of int64 keys. Each batch's keys are uniqued on the device; every `flush` batches those are merged
    on the device (counts summed per key) and moved to the CPU once; the CPU merges its parts now and then. Exact counts."""

    def __init__(self, flush=16):
        self.k, self.c, self.parts = np.zeros(0, np.int64), np.zeros(0, np.int64), []
        self.dk, self.dc, self.flush, self.n = [], [], flush, 0

    def add(self, keys):
        import torch
        if keys.numel():
            u, c = torch.unique(keys, return_counts=True)
            self.dk.append(u)
            self.dc.append(c)
        self.n += 1
        if self.n % self.flush == 0:
            self._to_cpu()

    def _to_cpu(self):
        import torch
        if not self.dk:
            return
        k, c = torch.cat(self.dk), torch.cat(self.dc)
        self.dk, self.dc = [], []
        u, inv = torch.unique(k, return_inverse=True)
        sums = torch.zeros(len(u), dtype=c.dtype, device=c.device).scatter_add_(0, inv, c)
        self.parts.append((u.cpu().numpy(), sums.cpu().numpy()))
        if len(self.parts) >= 32:
            self._merge_parts()

    def _merge_parts(self):
        if not self.parts:
            return
        k = np.concatenate([self.k] + [p[0] for p in self.parts])
        c = np.concatenate([self.c] + [p[1] for p in self.parts])
        self.parts = []
        if not len(k):
            self.k, self.c = k, c
            return
        o = np.argsort(k, kind="stable")
        k, c = k[o], c[o]
        first = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
        self.k, self.c = k[first], np.add.reduceat(c, first)

    def merge(self):
        self._to_cpu()
        self._merge_parts()


# ---------------------------------------------------------------- stage 1: word hits by frequency

class _HitCounter:
    """Stage 1's counters for one layer: strong activations by (token, pole) and (bigram, pole), thresholded on the device."""

    def __init__(self, L, V, device):
        import torch
        self.L, self.V = L, V
        self.elig = eligibility(L)
        pmax, nmax = _pole_max(L)
        thr = np.stack([pmax, nmax]) * RULE["strong"]
        thr[self.elig != "eligible"] = np.inf  # only eligible poles are counted (dense poles would swamp the counters)
        self.thr = torch.tensor(thr, device=device, dtype=torch.float32)
        self.uni, self.bi = _Acc(), _Acc()
        self.total = torch.zeros(2 * N, dtype=torch.int64, device=device)

    def add(self, act, x):
        import torch
        mp, NN, V = RULE["min_pos"], 2 * N, self.V
        act = act[:, mp:]  # window starts are an artifact (FINDINGS 2026-10-18): not counted
        for p in (0, 1):
            v = act.float() if p == 0 else -act.float()
            b, t, n = (v >= self.thr[p]).nonzero(as_tuple=True)
            del v
            pole = p * N + n
            tk = x[b, t + mp]
            self.uni.add(tk * NN + pole)
            self.bi.add((x[b, t + mp - 1] * V + tk) * NN + pole)
            self.total += torch.bincount(pole, minlength=NN)

    def save(self, P, toks, minutes):
        mp, NN, V = RULE["min_pos"], 2 * N, self.V
        uni, bi = self.uni, self.bi
        uni.merge()
        bi.merge()
        keep = bi.c >= 3  # a bigram seen < 3 times strong can never be a hit (min_strong) and is noise in the tables
        flat = toks.astype(np.int64)
        freq = np.bincount(flat[:, mp:].ravel(), minlength=V)
        bg, bgc = np.unique((flat[:, mp - 1:-1] * V + flat[:, mp:]).ravel(), return_counts=True)
        bk = bi.k[keep] // NN
        bfreq = bgc[np.clip(np.searchsorted(bg, bk), 0, len(bg) - 1)]
        np.savez_compressed(P["hits"], layer=self.L, windows=len(toks), tokens_per_window=toks.shape[1], n_tokens=flat[:, mp:].size,
                            uni_keys=uni.k, uni_counts=uni.c.astype(np.int32), bi_keys=bi.k[keep], bi_counts=bi.c[keep].astype(np.int32),
                            bi_freq=bfreq.astype(np.int64), freq=freq, total=self.total.cpu().numpy(), elig=(self.elig == "eligible"),
                            minutes=minutes)
        print(f"layer {self.L}: {len(uni.k):,} (token, pole) and {int(keep.sum()):,} (bigram, pole) pairs -> {P['hits']}")


def _progress(done, total, t0, unit="windows", tokens=None):
    el = time.time() - t0
    rate = f"  {done * tokens / el:,.0f} tok/s" if tokens else ""
    print(f"  {done}/{total} {unit}{rate}  eta {el / max(done, 1) * (total - done) / 60:.1f} min", flush=True)


def _stop_requested():
    """data/auto2/STOP ends a GPU loop at the next batch boundary (nothing of the stage is written). Stopping this way leaves
    the GPU idle; killing a CUDA process mid-kernel is a likely trigger of a display-driver reset (FINDINGS 2026-10-20)."""
    if (OUT / "STOP").exists():
        _gpu_idle("stop")
        sys.exit("STOP file found: stopping before the pass finished (nothing written)")


YIELD = {"seconds": 0.0, "ticks": 0}


def _tick(*_):
    """Before every forward pass: honour STOP, then take the GPU turn (gpu_turn yields to a foreground such as the stream).
    Before yielding, the allocator's cached blocks are returned, so the foreground can use what this process is not holding."""
    from bonsai_interp import gpu_turn
    _stop_requested()
    if gpu_turn._live("want"):
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    YIELD["seconds"] += gpu_turn.background_tick("auto2")
    YIELD["ticks"] += 1


def _gpu_idle(label=""):
    """Give the GPU up for a CPU stretch or at exit (the next forward pass takes it back)."""
    from bonsai_interp import gpu_turn
    gpu_turn.release()
    if label and label != "stop":
        print(f"gpu_turn: {label}: yielded {YIELD['seconds']:.1f} s over {YIELD['ticks']} batches so far", flush=True)


def _gpu_load():
    """The model, with a gpu_turn tick before every forward pass (one per batch in every stage)."""
    from bonsai_interp.model import load
    _tick()
    model, tok, rec = load("bonsai1")
    model.model.register_forward_pre_hook(_tick)
    return model, tok, rec


def hits(a):
    import torch
    from bonsai_interp.env import DEVICE, limit_memory
    P, man = _setup(a)
    L = a.layer
    toks, _, _ = _windows(SELECT, a)
    vb = vocab()
    counter = _HitCounter(L, vb["V"], DEVICE)
    limit_memory(0.8)
    model, tok, rec = _gpu_load()
    t0, B = time.time(), a.batch
    print(f"stage 1, layer {L}: {len(toks)} windows x {toks.shape[1]} tokens, {int((counter.elig == 'eligible').sum())} eligible poles", flush=True)
    for b0 in range(0, len(toks), B):
        _stop_requested()
        x = torch.from_numpy(toks[b0:b0 + B].astype(np.int64)).to(DEVICE)
        counter.add(_forward(model, rec, x, L), x)
        done = b0 + len(x)
        if (b0 // B) % 25 == 0 or done == len(toks):
            _progress(done, len(toks), t0, tokens=toks.shape[1])
    _gpu_idle(f"stage 1, layer {L}")
    counter.save(P, toks, (time.time() - t0) / 60)
    print(f"stage 1 pass {(time.time() - t0) / 60:.1f} min")
    rank(a, P, counter.elig)


def rank(a, P=None, elig=None, L=None):
    """The ranked word-hit table per pole and the triggers stage 2 screens (CPU, from the stage 1 counts)."""
    if P is None:
        P, _ = _setup(a)
    L = a.layer if L is None else L
    d = np.load(P["hits"])
    vb = vocab()
    V, NN = vb["V"], 2 * N
    if elig is None:
        elig = eligibility(L)
    ntok, total, freq = int(d["n_tokens"]), d["total"], d["freq"]
    pmax, nmax = _pole_max(L)
    uk, uc = d["uni_keys"], d["uni_counts"]
    bk, bc, bf = d["bi_keys"], d["bi_counts"], d["bi_freq"]

    def groups(keys):
        pole = keys % NN
        o = np.argsort(pole, kind="stable")
        starts = np.searchsorted(pole[o], np.arange(NN + 1))
        return o, starts

    uo, us = groups(uk)
    bo, bs = groups(bk)
    rows, counts = [], {}
    R = RULE
    for pp in range(NN):
        p, n = divmod(pp, N)
        cls = elig[p, n]
        row = {"neuron": pid(p, n, L), "class": cls, "pole_max": round(float(pmax[n] if p == 0 else nmax[n]), 4)}
        if cls == "eligible":
            tot = int(total[pp])
            base = tot / ntok
            row["strong"] = tot
            sel = uo[us[pp]:us[pp + 1]]
            t_ids, cnt = uk[sel] // NN, uc[sel].astype(np.int64)
            prec = cnt / np.maximum(freq[t_ids], 1)
            lift = prec / base if base else np.zeros_like(prec)
            cov = cnt / max(tot, 1)
            o = np.argsort(-cnt, kind="stable")[:12]
            row["words"] = [[int(t_ids[i]), int(cnt[i]), round(float(prec[i]), 4), round(float(lift[i]), 1), round(float(cov[i]), 4)] for i in o]
            ok = (cnt >= R["min_strong"]) & (cov >= R["min_coverage"]) & (lift >= R["min_lift"])
            trig = [{"token": int(t_ids[i]), "via": "word", "n": int(cnt[i]), "precision": round(float(prec[i]), 4),
                     "lift": round(float(lift[i]), 1), "coverage": round(float(cov[i]), 4)}
                    for i in np.flatnonzero(ok)[np.argsort(-cov[ok], kind="stable")]]
            sel = bo[bs[pp]:bs[pp + 1]]
            bkey, bcnt, bfr = bk[sel] // NN, bc[sel].astype(np.int64), bf[sel]
            bprev, btok = bkey // V, bkey % V
            bprec = bcnt / np.maximum(bfr, 1)
            blift = bprec / base if base else np.zeros_like(bprec)
            bcov = bcnt / max(tot, 1)
            o = np.argsort(-bcnt, kind="stable")[:12]
            row["bigrams"] = [[int(bprev[i]), int(btok[i]), int(bcnt[i]), round(float(bprec[i]), 4), round(float(blift[i]), 1),
                               round(float(bcov[i]), 4)] for i in o]
            ok = (bcnt >= R["min_strong"]) & (bcov >= R["min_coverage"]) & (blift >= R["min_lift"])
            have = {t["token"] for t in trig}
            for i in np.flatnonzero(ok)[np.argsort(-bcov[ok], kind="stable")]:
                if int(btok[i]) not in have:
                    have.add(int(btok[i]))
                    trig.append({"token": int(btok[i]), "via": "bigram", "prev": int(bprev[i]), "n": int(bcnt[i]),
                                 "precision": round(float(bprec[i]), 4), "lift": round(float(blift[i]), 1), "coverage": round(float(bcov[i]), 4)})
            row["triggers"] = sorted(trig, key=lambda t: -t["coverage"])[:R["triggers_per_pole"]]
            cls = "hit" if row["triggers"] else "no-hit"
        counts[cls] = counts.get(cls, 0) + 1
        rows.append(row)
    _write_gz(P["table"], rows)
    n_trig = sum(len(r.get("triggers", [])) for r in rows)
    print(f"stage 1 table: {json.dumps(counts)}; {n_trig} (pole, trigger) hits -> {P['table']}")
    return rows


# ---------------------------------------------------------------- stage 2: context screen + minimal pairs

def _entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def _best_condition(enc, y, mask, exclude=()):
    """Best (feature, value) by information gain on y within mask: (gain, feat, value index, support, positives)."""
    best = None
    yb = y[mask]
    n = len(yb)
    if n == 0:
        return None
    h0 = _entropy(yb.mean()) * n
    for f, (u, inv) in enc.items():
        if f in exclude:
            continue
        iv = inv[mask]
        cnt = np.bincount(iv, minlength=len(u))
        pos = np.bincount(iv, weights=yb, minlength=len(u))
        for j in np.flatnonzero(cnt >= RULE["min_support"]):
            if u[j] in ("other", "none"):
                continue
            c, k = cnt[j], pos[j]
            if c == n:
                continue
            h = _entropy(k / c) * c + _entropy((yb.sum() - k) / (n - c)) * (n - c)
            g = (h0 - h) / n
            if k / c > (yb.sum() - k) / (n - c) and (best is None or g > best[0]):
                best = (g, f, j, int(c), int(k))
    return best


def fit(feats, frac):
    """The context model of one (pole, trigger) on its screening occurrences.
    feats: {feature: array of values}; frac: activation at the trigger as a fraction of the pole's corpus max."""
    R = RULE
    y = (frac >= R["strong"]).astype(np.float64)
    n = len(y)
    out = {"n_occ": n, "strong_rate": round(float(y.mean()), 3) if n else 0.0, "median_frac": round(float(np.median(frac)), 3) if n else 0.0}
    if n < R["min_occ"]:
        out["bin"] = "rare"
        return out, None
    enc, table = {}, {}
    tot_ss = float(((frac - frac.mean()) ** 2).sum()) or 1.0
    for f, vals in feats.items():
        vals = np.asarray(vals, dtype=object).astype(str)
        u, inv = np.unique(vals, return_inverse=True)
        cnt = np.bincount(inv)
        vals = np.where(cnt[inv] >= R["min_support"], vals, "other")  # rare values pooled, so eta2 is not inflated
        u, inv = np.unique(vals, return_inverse=True)
        cnt = np.bincount(inv, minlength=len(u))
        mean = np.bincount(inv, weights=frac, minlength=len(u)) / np.maximum(cnt, 1)
        rate = np.bincount(inv, weights=y, minlength=len(u)) / np.maximum(cnt, 1)
        eta2 = float((cnt * (mean - frac.mean()) ** 2).sum() / tot_ss)
        top = np.argsort(-mean)[:4]
        table[f] = {"eta2": round(eta2, 3), "top": [[str(u[j]), int(cnt[j]), round(float(mean[j]), 3), round(float(rate[j]), 3)] for j in top]}
        if f not in TABLE_ONLY:
            enc[f] = (u, inv)
    out["features"] = table
    all_mask = np.ones(n, bool)
    c1 = _best_condition(enc, y, all_mask)
    rule = None
    if c1:
        conds = [(c1[1], str(enc[c1[1]][0][c1[2]]))]
        m = enc[c1[1]][1] == c1[2]
        c2 = _best_condition(enc, y, m, exclude=(c1[1],))
        if c2:
            m2 = m & (enc[c2[1]][1] == c2[2])
            prec1, prec2 = y[m].mean(), y[m2].mean()
            if prec2 - prec1 >= R["second_gain"] and y[m2].sum() >= 0.5 * y[m].sum():
                conds.append((c2[1], str(enc[c2[1]][0][c2[2]])))
                m = m2
        prec = float(y[m].mean())
        rec_ = float(y[m].sum() / max(y.sum(), 1))
        out_rate = float(y[~m].mean()) if (~m).any() else 0.0
        rule = {"conditions": [list(c) for c in conds], "support": int(m.sum()), "precision": round(prec, 3), "recall": round(rec_, 3),
                "rate_out": round(out_rate, 3)}
        out["rule"] = rule
    if rule and rule["precision"] >= R["rule_precision"] and rule["recall"] >= R["rule_recall"] \
            and rule["precision"] - rule["rate_out"] >= R["rule_margin"]:  # the more specific claim first
        out["bin"] = bin_of(rule)
    elif out["strong_rate"] >= R["lexical_rate"]:
        out["bin"] = "lexical"
    else:
        out["bin"] = "unexplained"
    return out, rule


def bin_of(rule):
    """A rule's bin from its features' families: an untestable family first, then the most specific claim."""
    # a copy rule claims repetition: "first appearance" or "no repeated pair" is a context claim, not a copy one
    fams = {c[0] if c[0] not in COPY or c[1] in ("repeat-near", "repeat-far", "bigram-repeat") else "prev_tok" for c in rule["conditions"]}
    for b, members in (("positional", POSITIONAL), ("topic", TOPIC), ("copy", COPY), ("predictive", PREDICTIVE),
                       ("collocation", COLLOCATION), ("context", CONTEXT)):
        if fams & set(members):
            return b
    return "format"


def ctx_bucket(v):
    return "<0.02" if v < 0.02 else "0.02-0.05" if v < 0.05 else "0.05-0.1" if v < 0.1 else "0.1+"


def rule_mask(feats, rule):
    m = np.ones(len(next(iter(feats.values()))), bool)
    for f, v in rule["conditions"]:
        m &= np.asarray(feats[f], dtype=object).astype(str) == v
    return m


class Substitutes:
    """Same-shape words of similar frequency (the auto-token@1 swap), from non-test windows only."""

    def __init__(self, freq, vb, seed=0):
        self.freq, self.vb = freq, vb
        text = vb["text"]
        cands = [i for i in np.argsort(-freq)[:20000] if freq[i] >= 50 and text[i].strip().isalpha() and len(text[i].strip()) >= 3]
        by = {}
        for i in cands:
            by.setdefault(self.shape(i), []).append(i)
        self.ids = {sh: np.array(sorted(v, key=lambda i: freq[i])) for sh, v in by.items()}
        self.lf = {sh: np.log(freq[v] + 1.0) for sh, v in self.ids.items()}

    def shape(self, i):
        s = self.vb["text"][i]
        w = s.strip()
        return (s[:1] == " ", w[:1].isupper(), w.isalpha())

    def __call__(self, t, rng):
        sh = self.shape(t)
        if sh not in self.ids or not self.vb["text"][t].strip().isalpha():
            return None
        k = self.vb["text"][t].strip().lower()
        ids, lf = self.ids[sh], self.lf[sh]
        c = int(np.searchsorted(lf, np.log(self.freq[t] + 1.0)))
        near = [int(i) for i in ids[max(0, c - 30): c + 30]
                if (w := self.vb["text"][i].strip().lower()) != k and w[:4] != k[:4] and k[:4] not in w]
        return near[rng.integers(len(near))] if near else None

    def _related(self, t, want):
        """The commonest same-shape frequent word that is a look-alike (shares the spelling, other word) or an inflection."""
        sh = self.shape(t)
        k = self.vb["text"][t].strip().lower()
        if sh not in self.ids or len(k) < 3 or not k.isalpha():
            return None
        infl = {k + "s", k + "es", k + "ed", k + "d", k + "ing", k[:-1] + "ing", k + "er", k + "ly"}
        best = None
        for i in self.ids[sh]:
            w = self.vb["text"][int(i)].strip().lower()
            if w == k:
                continue
            is_infl = w in infl or k in {w + "s", w + "es", w + "ed", w + "d", w + "ing", w[:-1] + "ing", w + "er", w + "ly"}
            if want == "inflection":
                ok = is_infl
            else:
                ok = not is_infl and ((k in w or w in k) and len(w) >= 3 or _edit1(k, w))
            if ok and (best is None or self.freq[i] > self.freq[best]):
                best = int(i)
        return best

    def lookalike(self, t):
        return self._related(t, "lookalike")

    def inflection(self, t):
        return self._related(t, "inflection")


def _edit1(a, b):
    """Edit distance exactly 1 (one substitution, insertion or deletion)."""
    if abs(len(a) - len(b)) > 1 or a == b:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    if len(a) > len(b):
        a, b = b, a
    return any(b[:i] + b[i + 1:] == a for i in range(len(b)))


def _occurrences(toks, tokens, M, L, min_pos=1):
    """{token: (windows, positions)}: up to M occurrences of each token in toks (seeded, spread over windows)."""
    flat = toks.ravel()
    T = toks.shape[1]
    want = np.zeros(int(flat.max()) + 2 if flat.size else 1, bool)
    tokens = [t for t in tokens if t < len(want)]
    want[tokens] = True
    idx = np.flatnonzero(want[flat])
    idx = idx[(idx % T) >= min_pos]
    o = np.argsort(flat[idx], kind="stable")
    idx = idx[o]
    st = np.searchsorted(flat[idx], np.arange(len(want) + 1))
    out = {}
    for t in tokens:
        pos = idx[st[t]:st[t + 1]]
        if len(pos) > M:
            pos = np.sort(np.random.default_rng([L, t]).choice(pos, M, replace=False))
        out[t] = (pos // T, pos % T)
    return out


def _hitlist(P, a):
    table = _read_gz(P["table"])
    hitlist = [(r["neuron"], t) for r in table for t in r.get("triggers", [])]
    if getattr(a, "limit_hits", None):  # smoke tests only
        keep = np.random.default_rng(0).permutation(len(hitlist))[:a.limit_hits]
        hitlist = [hitlist[i] for i in sorted(keep)]
    return hitlist


class _Pairs:
    """Stage 2's reads for one layer: one (occurrence, hit) pair per value, hit-major in occurrence order, with the
    activation at the trigger (frac) and the pole's mean activation over the preceding snippet (ctx_mean) filled in."""

    def __init__(self, L, hitlist, occ):
        pmax, nmax = _pole_max(L)
        pw, pp, pn, ps = [], [], [], []
        for nid, t in hitlist:
            w, p = occ.get(t["token"], (np.zeros(0, int), np.zeros(0, int)))
            n, sgn = int(nid.split("/N")[1][:-1]), 1.0 if nid.endswith("+") else -1.0
            pw.append(w); pp.append(p); pn.append(np.full(len(w), n)); ps.append(np.full(len(w), sgn))
        cat = lambda x, t: np.concatenate(x).astype(t) if x else np.zeros(0, t)
        self.L, self.pw, self.pp, self.pn, self.ps = L, cat(pw, np.int64), cat(pp, np.int64), cat(pn, np.int64), cat(ps, np.float64)
        self.mx = np.where(self.ps > 0, pmax[self.pn], nmax[self.pn])
        self.frac = np.zeros(len(self.pw), np.float32)
        self.ctxm = np.zeros(len(self.pw), np.float32)
        self.order = np.argsort(self.pw, kind="stable")
        self.pw_sorted = self.pw[self.order]

    def select(self, ws):
        """Pairs whose window is in ws (a sorted array of windows)."""
        lo, hi = np.searchsorted(self.pw_sorted, [ws[0], ws[-1] + 1])
        sel = self.order[lo:hi]
        return sel[np.isin(self.pw[sel], ws)]

    def read(self, act, ws, sel):
        import torch
        if not len(sel):
            return
        dev = act.device
        back = torch.arange(1, RULE["snippet"], device=dev)
        row = torch.from_numpy(np.searchsorted(ws, self.pw[sel])).to(dev)
        pos_t, n_t = torch.from_numpy(self.pp[sel]).to(dev), torch.from_numpy(self.pn[sel]).to(dev)
        v = act[row, pos_t, n_t].float().cpu().numpy()
        self.frac[sel] = v * self.ps[sel] / self.mx[sel]
        win = act[row[:, None], (pos_t[:, None] - back).clamp_min(0), n_t[:, None]].float()
        scale = torch.from_numpy(self.ps[sel] / self.mx[sel]).to(dev).float()[:, None]
        self.ctxm[sel] = (win * scale).clamp_min(0).mean(1).cpu().numpy()


def _finish_screen(P, L, hitlist, occ, pairs, toks, dom, w0, vb):
    """Store, features, fit and bins for one layer (minimal pairs come after, batched across layers when many)."""
    save_store(P["occ"], hitlist, occ, pairs.frac, pairs.ctxm, w0)
    return screen_rows(hitlist, occ, pairs.frac, pairs.ctxm, toks, dom, w0, P, vb, L)


def _write_screen(P, L, rows, t0):
    _write_gz(P["screen"], rows)
    tally = {}
    for r in rows:
        tally[r["bin"]] = tally.get(r["bin"], 0) + 1
    print(f"stage 2 bins, layer {L}: {json.dumps(tally)}; minutes {(time.time() - t0) / 60:.1f} -> {P['screen']}")


def screen(a):
    import torch
    from bonsai_interp.env import DEVICE, limit_memory
    P, man = _setup(a)
    L, R = a.layer, RULE
    vb = vocab()
    hitlist = _hitlist(P, a)
    toks, dom, w0 = _windows(SCREEN, a)
    print(f"stage 2, layer {L}: {len(hitlist)} hits, {len(toks)} screening windows (from {w0})", flush=True)
    occ = _occurrences(toks, sorted({t["token"] for _, t in hitlist}), R["occ_per_trigger"], L, min_pos=R["min_pos"])
    pairs = _Pairs(L, hitlist, occ)
    need = np.unique(pairs.pw)
    limit_memory(0.8)
    model, tok, rec = _gpu_load()
    t0, B = time.time(), a.batch
    for i in range(0, len(need), B):
        _stop_requested()
        ws = need[i:i + B]
        sel = pairs.select(ws)
        T = int(pairs.pp[sel].max()) + 1
        x = torch.from_numpy(toks[ws, :T].astype(np.int64)).to(DEVICE)
        pairs.read(_forward(model, rec, x, L), ws, sel)
        if (i // B) % 25 == 0 or i + B >= len(need):
            _progress(min(i + B, len(need)), len(need), t0)
    t_pass = time.time() - t0
    _gpu_idle(f"stage 2 pass, layer {L}")
    rows, plans = _finish_screen(P, L, hitlist, occ, pairs, toks, dom, w0, vb)
    print(f"stage 2 pass {t_pass / 60:.1f} min, fit {(time.time() - t0 - t_pass) / 60:.1f} min", flush=True)
    _templates(model, rec, {L: (rows, plans)}, a.batch * 32)
    _gpu_idle(f"stage 2 minimal pairs, layer {L}")
    _write_screen(P, L, rows, t0)


def save_store(path, hitlist, occ, frac, ctxm, w0):
    """Stage 2's per-occurrence activations, so a new context feature is a CPU-only refit (see `refit`).
    Occurrences are stored once per trigger; frac and ctx_mean once per (hit, occurrence), hit-major in occurrence order."""
    trig = np.array(sorted(occ), dtype=np.int64)
    counts = np.array([len(occ[t][0]) for t in trig], dtype=np.int64)
    np.savez_compressed(path, trigger=trig, occ_start=np.r_[0, np.cumsum(counts)],
                        window=np.concatenate([occ[t][0] + w0 for t in trig]).astype(np.int32) if len(trig) else np.zeros(0, np.int32),
                        position=np.concatenate([occ[t][1] for t in trig]).astype(np.int16) if len(trig) else np.zeros(0, np.int16),
                        hit_neuron=np.array([nid for nid, _ in hitlist]), hit_trigger=np.array([t["token"] for _, t in hitlist], dtype=np.int64),
                        hit_stage1=np.array([json.dumps(t) for _, t in hitlist]),
                        frac=frac.astype(np.float16), ctx_mean=ctxm.astype(np.float16))


def load_store(path):
    """(hitlist, occ, frac, ctxm) as screen had them, from the store (occ windows relative to the screening split)."""
    d = np.load(path)
    occ = {}
    for i, t in enumerate(d["trigger"]):
        a, b = d["occ_start"][i], d["occ_start"][i + 1]
        occ[int(t)] = (d["window"][a:b].astype(np.int64) - SCREEN[0], d["position"][a:b].astype(np.int64))
    hitlist = [(str(n), json.loads(str(j))) for n, j in zip(d["hit_neuron"], d["hit_stage1"])]
    return hitlist, occ, d["frac"].astype(np.float32), d["ctx_mean"].astype(np.float32)


def screen_rows(hitlist, occ, frac, ctxm, toks, dom, w0, P, vb, L):
    """Features, the fitted rule and the bin for every hit, from activations already read (stage 2 or the store)."""
    R = RULE
    counts = np.array([len(occ.get(t["token"], ([], []))[0]) for _, t in hitlist], dtype=np.int64)
    hst = np.r_[0, np.cumsum(counts)]  # pairs are hit-major, in occurrence order
    feats, bt = {}, bigram_table()
    for t, (w, p) in occ.items():
        fs = [features(toks[wi], int(pi), dom[wi], vb, bt) for wi, pi in zip(w, p)]
        feats[t] = {f: np.array([x[f] for x in fs], dtype=object) for f in FEATURES + TABLE_ONLY}
    rows, plans = [], []
    freq = np.bincount(np.asarray(toks, np.int64).ravel(), minlength=vb["V"])
    subs = Substitutes(freq + np.load(P["hits"])["freq"], vb) if P["hits"].exists() else Substitutes(freq, vb)
    for h, (nid, t) in enumerate(hitlist):
        sel = np.arange(hst[h], hst[h + 1])
        fr = frac[sel].astype(np.float64)
        fts = dict(feats.get(t["token"], {f: np.zeros(0, object) for f in FEATURES + TABLE_ONLY}))
        fts["ctx_mean"] = np.array([ctx_bucket(v) for v in ctxm[sel]], dtype=object)
        res, rule = fit(fts, fr)
        w, p = occ.get(t["token"], (np.zeros(0, int), np.zeros(0, int)))
        o = np.argsort(-fr)
        weak = [j for j in o[::-1] if fr[j] < R["strong"]][:3]
        row = {"neuron": nid, "trigger": t["token"], "trigger_text": vb["text"][t["token"]], "via": t["via"], "stage1": t, **res,
               "examples": {"strong": [[int(w[j]) + w0, int(p[j]), round(float(fr[j]), 3)] for j in o[:3] if fr[j] >= R["strong"]],
                            "weak": [[int(w[j]) + w0, int(p[j]), round(float(fr[j]), 3)] for j in weak]}}
        rows.append(row)
        plans.append(_template_plan(row, rule, fts, fr, w, p, toks, subs, L))
    return rows, plans


def refit(a):
    """CPU only: refit stage 2 from the per-occurrence store (after a feature change), without the minimal pairs.
    Writes .context.refit.jsonl.gz next to the registered stage 2 file, which it never touches."""
    P, _ = _setup(a) if a.smoke else (_paths(a.layer), None)
    if not P["occ"].exists():
        sys.exit(f"no store {P['occ'].name}: stage 2 of this layer ran before the store existed; re-run `screen` to make one")
    vb = vocab()
    hitlist, occ, frac, ctxm = load_store(P["occ"])
    toks, dom, w0 = _windows(SCREEN, a)
    rows, _ = screen_rows(hitlist, occ, frac, ctxm, toks, dom, w0, P, vb, a.layer)
    for r in rows:
        r["template"] = {"check": "n/a (refit: minimal pairs need the GPU)"}
    _write_gz(P["refit"], rows)
    tally = {}
    for r in rows:
        tally[r["bin"]] = tally.get(r["bin"], 0) + 1
    print(f"refit bins: {json.dumps(tally)} -> {P['refit']}")


def backup(a):
    """Hash every data/auto2 file of a layer into its summary and upload them (and the map file) to GitHub releases."""
    P = _paths(a.layer)
    if not P["summary"].exists():
        sys.exit("score the layer first: the summary records the artifacts")
    names = ("hits", "table", "screen", "tests", "results", "occ")
    files = [P[k] for k in names if P[k].exists()]
    summ = json.loads(P["summary"].read_text(encoding="utf-8"))
    summ["artifacts"] = [{"name": f.name, "sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "bytes": f.stat().st_size,
                          "release": "auto2-data"} for f in files]
    screen_map = ROOT / "data" / "auto" / f"bonsai1_L{a.layer}.screen.json.gz"
    if screen_map.exists():
        summ["artifacts"].append({"name": screen_map.name, "sha256": hashlib.sha256(screen_map.read_bytes()).hexdigest(),
                                  "bytes": screen_map.stat().st_size, "release": "auto-screens"})
    P["summary"].write_text(json.dumps(summ, indent=1) + "\n", encoding="utf-8", newline="\n")
    if a.no_upload:
        print(f"recorded {len(summ['artifacts'])} artifacts in {P['summary'].name} (no upload)")
        return
    repo = "moridinamael/bonsai-interp"
    for rel, notes, fs in (("auto2-data", "auto-token@2 data per layer (hits, stage 2 tables, per-occurrence store, tests, results); "
                            "sha256 in graph/auto/bonsai1_L<L>.auto2.summary.json", files),
                           ("auto-screens", "per-layer screen files for the survey map (site/screens.py)", [screen_map] if screen_map.exists() else [])):
        if not fs:
            continue
        if subprocess.run(["gh", "release", "view", rel, "--repo", repo], capture_output=True).returncode != 0:
            subprocess.run(["gh", "release", "create", rel, "--repo", repo, "--title", rel, "--notes", notes], check=True)
        subprocess.run(["gh", "release", "upload", rel, *map(str, fs), "--repo", repo, "--clobber"], check=True)
        print(f"uploaded {len(fs)} files to release {rel}")
    print(f"recorded {len(summ['artifacts'])} artifacts in {P['summary'].relative_to(ROOT)}: commit it on its own")


def _template_plan(row, rule, fts, fr, w, p, toks, subs, L):
    """Minimal pairs for the bin the corpus model suggests: [(kind, original ids, edited ids)]."""
    R, C = RULE, RULE["template_ctx"]
    if row["bin"] not in ("lexical", "context"):
        return None
    rng = np.random.default_rng([L, row["trigger"], int(row["neuron"].split("/N")[1][:-1])])
    ok = p >= C - 1
    snip = lambda j: toks[w[j], p[j] - C + 1: p[j] + 1].astype(np.int64)
    pairs = []
    if row["bin"] == "lexical":
        sub = subs(row["trigger"], rng)
        if sub is None:
            return None
        # a random same-shape word (is it this word at all?), a look-alike spelling (rain -> brain: is it the letters?)
        # and an inflection (take -> takes: is it the lexeme?)
        swaps = [("swap-trigger", sub), ("look-alike", subs.lookalike(row["trigger"])), ("inflection", subs.inflection(row["trigger"]))]
        for j in rng.permutation(np.flatnonzero(ok))[:R["template_n"]]:
            s = snip(j)
            for kind, rep_ in swaps:
                if rep_ is not None:
                    e = s.copy()
                    e[-1] = rep_
                    pairs.append((kind, s, e))
        row["swaps"] = {k: subs.vb["text"][v] for k, v in swaps if v is not None}
        return pairs
    prev = [c for c in rule["conditions"] if c[0] in ("prev_tok", "prev_class")]
    if not prev:
        return None
    f, v = prev[0]
    rest = {"conditions": [c for c in rule["conditions"] if c != prev[0]]}
    vals = np.asarray(fts[f], dtype=object).astype(str)
    m_rest = rule_mask(fts, rest) if rest["conditions"] else np.ones(len(fr), bool)
    in_m, out_m = m_rest & (vals == v) & ok, m_rest & (vals != v) & ok
    prev_ids = np.array([int(toks[w[j], p[j] - 1]) for j in range(len(w))]) if len(w) else np.zeros(0, int)

    def commonest(mask):
        if not mask.any():
            return None
        u, c = np.unique(prev_ids[mask], return_counts=True)
        return int(u[np.argmax(c)])
    into = int(v[1:]) if f == "prev_tok" else commonest(in_m & (fr >= R["strong"]))
    away = commonest(out_m & (fr < R["strong"]))
    if into is None or away is None:
        return None
    for kind, mask, rep in (("leave-context", in_m, away), ("enter-context", out_m, into)):
        for j in rng.permutation(np.flatnonzero(mask))[:R["template_n"]]:
            s = snip(j)
            e = s.copy()
            e[-2] = rep
            pairs.append((kind, s, e))
    return pairs


def _templates(model, rec, by_layer, B):
    """Score every minimal pair at the trigger (last) position and record the check in each row.
    by_layer: {L: (rows, plans)}. One forward per batch, to the deepest layer in it, reading each layer in a hook."""
    import torch
    from bonsai_interp.env import DEVICE
    from bonsai_interp.model import StopForward
    jobs = []  # (layer, row index, kind, which, ids, n, sign)
    for L in sorted(by_layer):
        rows, plans = by_layer[L]
        for i, (r, pl) in enumerate(zip(rows, plans)):
            if not pl:
                r["template"] = {"check": "n/a"}
                continue
            n, sgn = int(r["neuron"].split("/N")[1][:-1]), 1.0 if r["neuron"].endswith("+") else -1.0
            for kind, s_, e in pl:
                jobs += [(L, i, kind, 0, s_, n, sgn), (L, i, kind, 1, e, n, sgn)]
    pm = {L: _pole_max(L) for L in by_layer}
    vals = np.zeros(len(jobs))
    for b in range(0, len(jobs), B):
        _stop_requested()
        ch = jobs[b:b + B]
        x = torch.from_numpy(np.stack([j[4] for j in ch])).to(DEVICE)
        layers = sorted({j[0] for j in ch})
        got = {}

        def grab(L):
            def f(act):
                got[L] = act[:, -1].float().cpu().numpy()
                return act
            return f
        rec.mode, rec.interventions, rec.stop_at = None, {L: grab(L) for L in layers}, layers[-1]
        try:
            with torch.no_grad():
                model.model(input_ids=x)
        except StopForward:
            pass
        finally:
            rec.interventions, rec.stop_at = {}, None
        for k, j in enumerate(ch):
            pmax, nmax = pm[j[0]]
            vals[b + k] = got[j[0]][k, j[5]] * j[6] / (pmax[j[5]] if j[6] > 0 else nmax[j[5]])
    res = {}
    for (L, i, kind, which, *_), v in zip(jobs, vals):
        res.setdefault((L, i), {}).setdefault(kind, [[], []])[which].append(float(v))
    mv = RULE["template_move"]
    for (L, i), kinds in res.items():
        t = {}
        for kind, (orig, edit) in kinds.items():
            d = np.array(edit) - np.array(orig)
            t[kind] = {"n": len(d), "orig_median": round(float(np.median(orig)), 3), "edit_median": round(float(np.median(edit)), 3),
                       "median_change": round(float(np.median(d)), 3)}
        if "swap-trigger" in t:
            ok = t["swap-trigger"]["median_change"] <= -mv
            t["check"] = "confirmed" if ok else "not confirmed"
            if "look-alike" in t:  # the word matters, but a look-alike spelling fires too: letters, not the word
                t["spelling"] = bool(ok and t["look-alike"]["median_change"] > -mv)
            if "inflection" in t:  # the word matters and its inflection fires too: the lexeme, not the token
                t["lexeme"] = bool(ok and t["inflection"]["median_change"] > -mv)
        else:
            down = "leave-context" in t and t["leave-context"]["median_change"] <= -mv
            up = "enter-context" in t and t["enter-context"]["median_change"] >= mv
            t["check"] = "confirmed" if down and up else "partial" if down or up else "not confirmed"
        by_layer[L][0][i]["template"] = t


# ---------------------------------------------------------------- auto tests (pre-registered) and scoring

# the order in which a capped run keeps a pole's hits: layer 20's held rates per bin (lexical 97%, copy 67%, context 49%,
# format 49%, predictive 47%, collocation 40%), then stage 1 coverage
TEST_PRIORITY = ["lexical", "copy", "context", "format", "predictive", "collocation"]


def cap_tests(rows, k):
    """At most k testable hits per pole (k = 0: all), by TEST_PRIORITY then coverage; the rest stay screened, untested."""
    if not k:
        return rows
    by = {}
    for r in rows:
        by.setdefault(r["neuron"], []).append(r)
    keep = set()
    for lst in by.values():
        for r in sorted(lst, key=lambda r: (TEST_PRIORITY.index(r["bin"]), -r["stage1"]["coverage"]))[:k]:
            keep.add(id(r))
    return [r for r in rows if id(r) in keep]


def label(row, vb):
    t = repr(vb["text"][row["trigger"]])
    if row["bin"] == "lexical":
        return f"the token {t} (auto: lexical; strong on {row['strong_rate']:.0%} of screening occurrences)"
    cond = " and ".join(condition_phrase(f, v, vb) for f, v in row["rule"]["conditions"])
    return f"the token {t} {cond} (auto: {row['bin']}; strong on {row['rule']['precision']:.0%} of such screening occurrences)"


def prereg(a):
    """Auto tests from the testing windows (texts only, nothing scored), pinned by sha256 in a manifest."""
    P, man = _setup(a)
    L = a.layer
    counts, sha = _prereg_layer(L, P, a)
    pre = {"protocol": PROTOCOL, "layer": L, "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "toolchain_commit": _commit(), "protocol_manifest": P["protocol"].relative_to(ROOT).as_posix() if not a.smoke else None,
           "tests_file": P["tests"].relative_to(ROOT).as_posix() if not a.smoke else str(P["tests"]), "tests_sha256": sha,
           "test_windows": f"{TEST[0]}..", "counts": counts}
    P["prereg"].write_text(json.dumps(pre, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(counts))
    print(f"wrote {P['tests']} and {P['prereg']}\nCommit the manifest ALONE before scoring.")


def _prereg_layer(L, P, a):
    """One layer's auto tests (texts only), written to its tests file: (counts, sha256 of the file's content)."""
    from bonsai_interp import explore
    R = RULE
    vb, tok = vocab(), explore.tokenizer()
    rows = _read_gz(P["screen"])
    toks, dom, w0 = _windows((16000, TEST[0]) if a.smoke else TEST, a)  # smoke tests stay out of the testing windows
    todo = cap_tests([r for r in rows if r["bin"] in TESTED], R.get("tests_per_pole", 0))
    occ = _occurrences(toks, sorted({r["trigger"] for r in todo}), 4096, L, min_pos=max(R["snippet"], R["min_pos"]))
    d = np.load(P["hits"]) if P["hits"].exists() else None
    subs = Substitutes(d["freq"] if d is not None else np.bincount(toks.ravel().astype(np.int64), minlength=vb["V"]), vb)
    feats_cache = {}
    out, counts = [], {}
    for r in rows:
        counts[r["bin"]] = counts.get(r["bin"], 0) + 1
    for r in todo:
        n = int(r["neuron"].split("/N")[1][:-1])
        rng = np.random.default_rng([L, r["trigger"], n, 2])
        w, p = occ.get(r["trigger"], (np.zeros(0, int), np.zeros(0, int)))
        if r["trigger"] not in feats_cache:
            fs = [features(toks[wi], int(pi), dom[wi], vb, bigram_table()) for wi, pi in zip(w, p)]
            feats_cache[r["trigger"]] = {f: np.array([x[f] for x in fs], dtype=object) for f in FEATURES}
        fts = feats_cache[r["trigger"]]
        m = rule_mask(fts, r["rule"]) if r["bin"] != "lexical" and len(w) else np.ones(len(w), bool)

        def pick(mask, k):
            got, seen = [], set()
            for j in rng.permutation(np.flatnonzero(mask)):
                if w[j] not in seen:
                    seen.add(w[j]); got.append(j)
                if len(got) == k:
                    break
            return got
        snip = lambda j: toks[w[j], p[j] - R["snippet"] + 1: p[j] + 1].astype(np.int64)
        fire_j = pick(m, R["n"])
        sub = subs(r["trigger"], rng)
        fire, quiet, kinds = [tok.decode(snip(j)) for j in fire_j], [], []
        swap = lambda j: tok.decode(np.concatenate([snip(j)[:-1], [sub]]))
        if r["bin"] == "lexical":
            if sub is not None:
                quiet, kinds = [swap(j) for j in fire_j], ["swap-trigger"] * len(fire_j)
        else:
            k_fail = R["n_rule_fail"] if sub is not None else R["n"]
            fail_j = pick(~m, k_fail)
            quiet = [tok.decode(snip(j)) for j in fail_j]
            kinds = ["rule-fails"] * len(fail_j)
            if sub is not None:
                quiet += [swap(j) for j in fire_j[:R["n"] - k_fail]]
                kinds += ["swap-trigger"] * (R["n"] - k_fail)
        base = {"neuron": r["neuron"], "trigger": r["trigger"], "bin": r["bin"], "template": r.get("template", {}).get("check")}
        if len(fire) < R["n"] or len(quiet) < R["n"]:
            counts["untestable"] = counts.get("untestable", 0) + 1
            out.append({**base, "class": "untestable", "why": f"{len(fire)} fire, {len(quiet)} near misses available"})
            continue
        counts["tested"] = counts.get("tested", 0) + 1
        out.append({**base, "class": "tested", "label": label(r, vb), "fire": fire, "quiet": quiet, "quiet_kind": kinds})
    return counts, _write_gz(P["tests"], out)


def score(a):
    from bonsai_interp.env import limit_memory
    P, man = _setup(a)
    L = a.layer
    pre = json.loads(P["prereg"].read_text(encoding="utf-8"))
    first = None
    if not a.smoke:
        first, dirty = _committed_alone_clean(P["prereg"])
        if not first or dirty:
            sys.exit(f"commit {P['prereg'].relative_to(ROOT)} on its own before scoring (pre-registration)")
    limit_memory(0.8)
    model, tok, rec = _gpu_load()
    _score_layer(model, tok, rec, L, P, pre["tests_sha256"], pre["counts"], first, a.batch, a.smoke)


def _score_layer(model, tok, rec, L, P, tests_sha, counts, prereg_commit, B, smoke):
    """Judge one layer's pinned auto tests: results file and summary."""
    import torch
    from bonsai_interp.env import DEVICE
    from bonsai_interp.graph.protocols import PROTOCOLS
    from bonsai_interp.model import StopForward
    from bonsai_interp.probe.textinv import PREAMBLES
    from transformers import DynamicCache
    blob = gzip.open(P["tests"], "rb").read()
    if hashlib.sha256(blob).hexdigest() != tests_sha:
        sys.exit(f"layer {L}: the tests file does not match its pre-registered sha256")
    rows = [json.loads(x) for x in blob.decode("utf-8").splitlines()]
    todo = [r for r in rows if r["class"] == "tested"]
    texts = sorted({t for r in todo for t in r["fire"] + r["quiet"]})
    want = {}
    for r in todo:
        for t in r["fire"] + r["quiet"]:
            want.setdefault(t, set()).add(r["neuron"])
    pmax, nmax = _pole_max(L)
    tok.padding_side = "right"
    t0, sums, y0 = time.time(), {}, YIELD["seconds"]
    # checkpoint: partial sums and the next batch to run, so a stop (STOP, a reset, a crash) loses minutes, not the layer
    ck = P["results"].with_name(f"bonsai1_L{L}.score.ckpt")
    resume, n_done = (-1, -1, -1), 0
    if ck.exists():
        import pickle
        st = pickle.loads(ck.read_bytes())
        if st["tests_sha"] == tests_sha and st["B"] == B:
            sums, resume = st["sums"], tuple(st["pos"])
            print(f"resuming layer {L} from its checkpoint at {resume} ({len(sums)} partial sums)", flush=True)

    def save(pos):
        import pickle
        tmp = ck.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps({"tests_sha": tests_sha, "B": B, "pos": pos, "sums": sums}))
        tmp.replace(ck)
    print(f"scoring layer {L}: {len(todo)} tests, {len(texts)} snippets x {len(PREAMBLES)} preambles", flush=True)
    def add(chunk, at):
        for j, t in enumerate(chunk):
            for nid in want[t]:
                n = int(nid.split("/N")[1][:-1])
                v = at[j, n] / pmax[n] if nid.endswith("+") else -at[j, n] / nmax[n]
                sums[(t, nid)] = sums.get((t, nid), 0.0) + float(v) / len(PREAMBLES)

    def run(ids, mask, cache=None):
        rec.mode, rec.layers, rec.stop_at = "act", {L}, L
        try:
            with torch.no_grad():
                model.model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=cache is not None)
        except StopForward:
            pass
        finally:
            rec.mode, rec.stop_at = None, None
        x = rec.store.pop(L)
        rec.clear()
        return x

    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    n_total, t_last = len(PREAMBLES) * -(-len(texts) // B), [time.time()]

    def progress():
        if time.time() - t_last[0] >= 60:
            t_last[0] = time.time()
            el = time.time() - t0
            print(f"  layer {L}: {n_done}/{n_total} batches this run, {n_done * B / max(el, 1e-9):.1f} snippets/s, "
                  f"yielded {YIELD['seconds'] - y0:.0f} s of {el:.0f} s", flush=True)
    for pi, pre_text in enumerate(PREAMBLES):
        if pi < resume[0]:
            continue
        # the preamble's keys and values, computed once to this layer and reused for every snippet: the same token inputs
        # as encoding preamble + snippet together; snippets whose joined encoding splits differently take the full path
        pre_ids = tok(pre_text).input_ids
        P_ = len(pre_ids)
        cache = DynamicCache()
        rec.mode, rec.stop_at = None, L
        try:
            with torch.no_grad():
                model.model(input_ids=torch.tensor([pre_ids], device=DEVICE), past_key_values=cache, use_cache=True)
        except StopForward:
            pass
        finally:
            rec.stop_at = None
        joined = {t: tok(pre_text + t).input_ids for t in texts}
        cached = sorted((t for t in texts if joined[t][:P_] == pre_ids), key=lambda t: len(joined[t]))
        full = [t for t in texts if joined[t][:P_] != pre_ids]
        for b in range(0, len(cached), B):
            if (pi, 0, b) < resume:
                continue
            try:
                _stop_requested()
                chunk = cached[b:b + B]
                suf = [joined[t][P_:] for t in chunk]
                T = max(len(x) for x in suf)
                ids = torch.tensor([x + [pad] * (T - len(x)) for x in suf], device=DEVICE)
                m = torch.tensor([[1] * (P_ + len(x)) + [0] * (T - len(x)) for x in suf], device=DEVICE)
                c = copy.deepcopy(cache)
                c.batch_repeat_interleave(len(chunk))
                x = run(ids, m, c)
                last = torch.tensor([len(x_) - 1 for x_ in suf], device=x.device)  # read at the snippet's last token
                add(chunk, x[torch.arange(len(chunk), device=x.device), last].float().cpu().numpy())
            except SystemExit:  # STOP before or during this batch: nothing of it was added
                save((pi, 0, b))
                raise
            n_done += 1
            progress()
            if n_done % 300 == 0:
                save((pi, 0, b + B))
        for b in range(0, len(full), B):
            if (pi, 1, b) < resume:
                continue
            try:
                _stop_requested()
                chunk = full[b:b + B]
                enc = tok([pre_text + t for t in chunk], return_tensors="pt", padding=True).to(DEVICE)
                x = run(enc.input_ids, enc.attention_mask)
                # read at the snippet's last token (the trigger or its swap): the claim is about that position, and a max
                # over a long snippet would let a near miss fire elsewhere (on another place where the rule holds)
                last = enc.attention_mask.sum(1) - 1
                add(chunk, x[torch.arange(len(chunk), device=x.device), last].float().cpu().numpy())
            except SystemExit:
                save((pi, 1, b))
                raise
            n_done += 1
            progress()
            if n_done % 300 == 0:
                save((pi, 1, b + B))
    rule = PROTOCOLS["quick-8+8@2"]["rule"]
    out, tally, by_bin = [], {}, {}
    for r in rows:
        if r["class"] == "tested":
            per = [{"set": "fire", "text": t, "act": round(sums[(t, r["neuron"])], 4)} for t in r["fire"]] + \
                  [{"set": "quiet", "text": t, "act": round(sums[(t, r["neuron"])], 4), "kind": k} for t, k in zip(r["quiet"], r["quiet_kind"])]
            res = rule({"per_sentence": per})
            r = {**r, "per_sentence": per, "outcome": res, "median_fire": round(float(np.median([q["act"] for q in per if q["set"] == "fire"])), 3)}
            tally[res] = tally.get(res, 0) + 1
            key = f"{r['bin']}/{r['template'] or 'n/a'}"
            by_bin.setdefault(key, {}).setdefault(res, 0)
            by_bin[key][res] += 1
        out.append(r)
    _gpu_idle(f"score, layer {L}")
    sha = _write_gz(P["results"], out)
    ck.unlink(missing_ok=True)
    summ = {"layer": L, "protocol": PROTOCOL, "scored": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "toolchain_commit": _commit(), "prereg_commit": prereg_commit[:12] if prereg_commit else None, "counts": counts, "outcomes": tally,
            "outcomes_by_bin_and_template": by_bin, "poles_held": len({r["neuron"] for r in out if r.get("outcome") == "held"}),
            "minutes": round((time.time() - t0) / 60, 1), "results_file": P["results"].relative_to(ROOT).as_posix() if not smoke else str(P["results"]),
            "results_sha256": sha, "yield_seconds": round(YIELD["seconds"] - y0, 1)}
    P["summary"].write_text(json.dumps(summ, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: summ[k] for k in ("layer", "outcomes", "poles_held", "minutes", "yield_seconds")}))
    return summ


# ---------------------------------------------------------------- the map's screen layer (site/screens.py)

BIN_ORDER = ["spelling", "lexical", "copy", "predictive", "collocation", "context", "format", "topic", "positional", "unexplained", "rare"]


def _map_bin(r):
    """The bin shown on the map: a lexical hit whose look-alike swap keeps firing is spelling, not a word."""
    return "spelling" if r["bin"] == "lexical" and (r.get("template") or {}).get("spelling") else r["bin"]


def export(a):
    """One record per pole for the survey map: data/auto/bonsai1_L<L>.screen.json.gz (scrutiny 0, never a label).
    A pole's bin is that of its best hit (a held test first, then the order of BIN_ORDER); poles without a word hit
    are 'unexplained' (no token explains a twentieth of their strong activations), and ineligible ones keep their class."""
    P, _ = _setup(a)
    _export_layer(a.layer, P, a.smoke)


def _export_layer(L, P, smoke):
    vb = vocab()
    table = _read_gz(P["table"])
    screened = {}
    for r in _read_gz(P["screen"]) if P["screen"].exists() else []:
        screened.setdefault(r["neuron"], []).append(r)
    tested = {}
    if P["results"].exists():
        from bonsai_interp.graph.protocols import v2_stats
        for r in _read_gz(P["results"]):
            if r.get("outcome"):
                tested[(r["neuron"], r["trigger"])] = (r["outcome"], round(v2_stats(r)[0], 3))
    poles = {}
    for t in table:
        key = t["neuron"].split("/")[-1]
        rec = {"bin": t["class"] if t["class"] in ("dense", "positional", "weak") else "unexplained",
               "hits": [{"token": vb["text"][h["token"]], "lift": h["lift"], "coverage": h["coverage"], "precision": h["precision"]}
                        for h in t.get("triggers", [])]}
        best = None
        for r in screened.get(t["neuron"], []):
            out, auroc = tested.get((r["neuron"], r["trigger"]), (None, None))
            rank_ = (out != "held", BIN_ORDER.index(_map_bin(r)))
            if best is None or rank_ < best[0]:
                best = (rank_, r, out, auroc)
        if best:
            _, r, out, auroc = best
            rec["bin"] = _map_bin(r)
            if "rule" in r and r["bin"] not in ("lexical", "unexplained", "rare"):
                rec["context"] = (f"{vb['text'][r['trigger']]!r} " + " and ".join(condition_phrase(f, v, vb) for f, v in r["rule"]["conditions"])
                                  + f" (strong on {r['rule']['precision']:.0%} there, {r['rule']['rate_out']:.0%} elsewhere)")
            elif r["bin"] == "lexical":
                rec["context"] = f"{vb['text'][r['trigger']]!r} in any context (strong on {r['strong_rate']:.0%} of occurrences)"
            rec["test"] = out or "untested"
            if auroc is not None:
                rec["auroc"] = auroc
        poles[key] = rec
    dst = (P["hits"].parent if smoke else ROOT / "data" / "auto") / f"bonsai1_L{L}.screen.json.gz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    first = None if smoke or not P["prereg"].exists() else _committed_alone_clean(P["prereg"])[0]
    blob = json.dumps({"layer": L, "protocol": PROTOCOL, "prereg_commit": first[:12] if first else None, "poles": poles},
                      ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with gzip.GzipFile(dst, "wb", mtime=0) as f:
        f.write(blob)
    tally = {}
    for r in poles.values():
        tally[r["bin"]] = tally.get(r["bin"], 0) + 1
    if P["summary"].exists():  # the summary in git pins the screen file, which lives in a release
        summ = json.loads(P["summary"].read_text(encoding="utf-8"))
        summ.update({"screen_file": dst.name, "screen_sha256": hashlib.sha256(blob).hexdigest(), "screen_bins": tally})
        P["summary"].write_text(json.dumps(summ, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{json.dumps(tally)} -> {dst} ({dst.stat().st_size / 1e6:.1f} MB; publish it in the auto-screens release, not in git)")


# ---------------------------------------------------------------- all-layer passes (one registration for many layers)
# hits-all and screen-all read every layer in ONE forward pass per batch (hooks threshold or gather each layer and keep
# nothing else); score-all loads the model once and runs each layer's tests to that layer. Per-layer outputs are the same
# files as the single-layer stages write. One protocol manifest lists the layers; one prereg manifest pins every tests file.

def _multi_paths(out=OUT, man=MANIFESTS):
    return {"protocol": man / "bonsai1_multi.auto2.protocol.json", "prereg": man / "bonsai1_multi.auto2.prereg.json",
            "amend": man / "bonsai1_multi.auto2.amendments.json"}


def _layers(a):
    layers = sorted({int(x) for x in a.layers.split(",") if x.strip()})
    if 12 in layers:
        sys.exit("layer 12 was used to look: it cannot be registered")
    return layers


def _setup_multi(a, need_prereg=False):
    """Per-layer paths for an all-layer pass, with the guard: the multi protocol manifest committed alone and clean,
    this file unchanged since, the layers registered there (and, to score, the prereg manifest committed alone too)."""
    layers = _layers(a)
    if a.smoke:
        d = Path(a.smoke)
        d.mkdir(parents=True, exist_ok=True)
        M = _multi_paths(d, d)
        Ps = {L: {**_paths(L, d, d), "protocol": M["protocol"], "prereg": M["prereg"]} for L in layers}
        return layers, Ps, M, {"rule": RULE, "layers": layers}
    M = _multi_paths()
    if not M["protocol"].exists():
        sys.exit(f"no protocol manifest: run `protocol-all --layers ...` and commit {M['protocol'].relative_to(ROOT)} on its own first")
    first, dirty = _committed_alone_clean(M["protocol"])
    if not first or dirty:
        sys.exit(f"commit {M['protocol'].relative_to(ROOT)} on its own before running any stage (pre-registration)")
    man = json.loads(M["protocol"].read_text(encoding="utf-8"))
    if _code_sha() not in _allowed_shas(M, man):
        sys.exit(f"{MODULE.name} changed since the protocol was registered and the change is not a committed speed amendment "
                 f"(amend-all): that is a new protocol version, not this one")
    if set(layers) - set(man["layers"]):
        sys.exit(f"layers {sorted(set(layers) - set(man['layers']))} are not in {M['protocol'].name}")
    if need_prereg:
        first, dirty = _committed_alone_clean(M["prereg"])
        if not first or dirty:
            sys.exit(f"commit {M['prereg'].relative_to(ROOT)} on its own before scoring (pre-registration)")
    OUT.mkdir(parents=True, exist_ok=True)
    Ps = {L: {**_paths(L), "protocol": M["protocol"], "prereg": M["prereg"]} for L in layers}
    return layers, Ps, M, man


def _allowed_shas(M, man):
    """The registered code, plus speed-only amendments (same outcomes on the same inputs, parity checked), each added to
    the amendments file and committed alone. A change that alters results is a new protocol, never an amendment."""
    ok = {man["code_sha256"]}
    if M["amend"].exists():
        first, dirty = _committed_alone_clean(M["amend"])
        if first and not dirty:
            ok |= {x["code_sha256"] for x in json.loads(M["amend"].read_text(encoding="utf-8"))["amendments"]}
    return ok


def amend_all(a):
    """Record the current code as a speed-only amendment of the multi-layer protocol, or with `amend --layer L` of one
    layer's protocol (then commit the file ALONE)."""
    M = _multi_paths() if a.layer is None else {"amend": _paths(a.layer)["amend"]}
    rel = MODULE.relative_to(ROOT).as_posix()
    if subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip():
        sys.exit(f"commit {rel} first: the amendment pins the committed code by sha256")
    if not a.note or not a.parity:
        sys.exit("give --note (what changed, and why it is speed-only) and --parity (the check that outcomes are unchanged)")
    doc = json.loads(M["amend"].read_text(encoding="utf-8")) if M["amend"].exists() else {"protocol": PROTOCOL, "amendments": []}
    doc["amendments"].append({"code_sha256": _code_sha(), "toolchain_commit": _commit(), "note": a.note, "parity": a.parity,
                              "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
    M["amend"].write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {M['amend'].relative_to(ROOT)}: commit it ALONE before the next stage")


def protocol_all(a):
    """The rule for many layers at once, pinned before any of them runs: commit this file ALONE, then run the passes."""
    layers = _layers(a)
    M = _multi_paths()
    rel = MODULE.relative_to(ROOT).as_posix()
    if subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip():
        sys.exit(f"commit {rel} first: the manifest pins the committed code by sha256")
    done = [L for L in layers if _paths(L)["protocol"].exists()]
    if done:
        sys.exit(f"layers {done} already have their own protocol manifests")
    man = {"protocol": PROTOCOL, "layers": layers, "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "toolchain_commit": _commit(), "code": rel, "code_sha256": _code_sha(),
           "windows": {"selection": f"{SELECT[0]}..{SELECT[1] - 1}", "screening": f"{SCREEN[0]}..{SCREEN[1] - 1}", "testing": f"{TEST[0]}.."},
           "rule": RULE, "features": list(FEATURES), "table_only": list(TABLE_ONLY),
           "bins": {"context": list(CONTEXT), "format": list(FORMAT), "copy": list(COPY), "predictive": list(PREDICTIVE),
                    "collocation": list(COLLOCATION), "topic": list(TOPIC), "positional": list(POSITIONAL)},
           "tested_bins": list(TESTED),
           "passes": "hits-all and screen-all read every layer in one forward pass per batch; score-all runs each layer's tests "
                     "to that layer; the outputs per layer are the files the single-layer stages write",
           "expectations": [
               "held rates per bin are similar to layer 20's (lexical about 95%, context, format and predictive about 50%)",
               "lexical and spelling hits are commoner in early layers, copy and topic hits in late ones",
               "most word hits are unexplained at every layer"],
           "note": "expectations are recorded before the run and reported either way; they are not gates"}
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    M["protocol"].write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {M['protocol'].relative_to(ROOT)} for {len(layers)} layers: commit it ALONE before running any pass")


def _hooked_pass(model, rec, x, hooks, top):
    """One forward to layer `top`, calling hooks[L](act) at every hooked layer (the activations are not stored)."""
    import torch
    from bonsai_interp.model import StopForward
    rec.mode, rec.interventions, rec.stop_at = None, {L: (lambda act, f=f: (f(act), act)[1]) for L, f in hooks.items()}, top
    try:
        with torch.no_grad():
            model.model(input_ids=x)
    except StopForward:
        pass
    finally:
        rec.interventions, rec.stop_at = {}, None


def hits_all(a):
    import torch
    from bonsai_interp.env import DEVICE, limit_memory
    layers, Ps, M, man = _setup_multi(a)
    toks, _, _ = _windows(SELECT, a)
    vb = vocab()
    counters = {L: _HitCounter(L, vb["V"], DEVICE) for L in layers}
    limit_memory(0.8)
    model, tok, rec = _gpu_load()
    t0, B = time.time(), a.batch
    print(f"stage 1, layers {layers}: {len(toks)} windows x {toks.shape[1]} tokens, one pass to layer {layers[-1]}", flush=True)
    for b0 in range(0, len(toks), B):
        _stop_requested()
        x = torch.from_numpy(toks[b0:b0 + B].astype(np.int64)).to(DEVICE)
        _hooked_pass(model, rec, x, {L: (lambda act, c=c: c.add(act, x)) for L, c in counters.items()}, layers[-1])
        done = b0 + len(x)
        if (b0 // B) % 100 == 0 or done == len(toks):
            _progress(done, len(toks), t0, tokens=toks.shape[1])
    minutes = (time.time() - t0) / 60
    _gpu_idle("stage 1, all layers")
    print(f"stage 1 pass {minutes:.1f} min for {len(layers)} layers", flush=True)
    for L, c in counters.items():
        c.save(Ps[L], toks, minutes)
        rank(a, Ps[L], c.elig, L=L)


def screen_all(a):
    import torch
    from bonsai_interp.env import DEVICE, limit_memory
    layers, Ps, M, man = _setup_multi(a)
    R, vb = RULE, vocab()
    toks, dom, w0 = _windows(SCREEN, a)
    hl, occ, pairs = {}, {}, {}
    for L in layers:
        hl[L] = _hitlist(Ps[L], a)
        occ[L] = _occurrences(toks, sorted({t["token"] for _, t in hl[L]}), R["occ_per_trigger"], L, min_pos=R["min_pos"])
        pairs[L] = _Pairs(L, hl[L], occ[L])
    need = np.unique(np.concatenate([pairs[L].pw for L in layers])) if layers else np.zeros(0, np.int64)
    print(f"stage 2, layers {layers}: {sum(len(hl[L]) for L in layers)} hits, {len(need)} screening windows needed (from {w0})", flush=True)
    limit_memory(0.8)
    model, tok, rec = _gpu_load()
    t0, B = time.time(), a.batch
    for i in range(0, len(need), B):
        _stop_requested()
        ws = need[i:i + B]
        sel = {L: pairs[L].select(ws) for L in layers}
        live = [L for L in layers if len(sel[L])]
        if not live:
            continue
        T = max(int(pairs[L].pp[sel[L]].max()) for L in live) + 1
        x = torch.from_numpy(toks[ws, :T].astype(np.int64)).to(DEVICE)
        _hooked_pass(model, rec, x, {L: (lambda act, L=L: pairs[L].read(act, ws, sel[L])) for L in live}, live[-1])
        if (i // B) % 100 == 0 or i + B >= len(need):
            _progress(min(i + B, len(need)), len(need), t0)
    t_pass = time.time() - t0
    _gpu_idle("stage 2 pass, all layers")
    print(f"stage 2 pass {t_pass / 60:.1f} min for {len(layers)} layers", flush=True)
    done = {}
    for L in layers:
        done[L] = _finish_screen(Ps[L], L, hl[L], occ[L], pairs[L], toks, dom, w0, vb)
    print(f"fit {(time.time() - t0 - t_pass) / 60:.1f} min", flush=True)
    t1 = time.time()
    _stop_requested()
    _templates(model, rec, done, a.batch * 32)
    _gpu_idle("minimal pairs, all layers")
    print(f"minimal pairs {(time.time() - t1) / 60:.1f} min", flush=True)
    for L in layers:
        _write_screen(Ps[L], L, done[L][0], t0)


def prereg_all(a):
    layers, Ps, M, man = _setup_multi(a)
    tests = {}
    for L in layers:
        counts, sha = _prereg_layer(L, Ps[L], a)
        tests[str(L)] = {"tests_file": Ps[L]["tests"].relative_to(ROOT).as_posix() if not a.smoke else str(Ps[L]["tests"]),
                         "tests_sha256": sha, "counts": counts}
        print(f"layer {L}: {json.dumps(counts)}", flush=True)
    pre = {"protocol": PROTOCOL, "layers": layers, "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "toolchain_commit": _commit(), "protocol_manifest": M["protocol"].relative_to(ROOT).as_posix() if not a.smoke else None,
           "test_windows": f"{TEST[0]}..", "tests": tests}
    M["prereg"].write_text(json.dumps(pre, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {M['prereg']}: commit it ALONE before scoring")


def score_all(a):
    from bonsai_interp.env import limit_memory
    layers, Ps, M, man = _setup_multi(a, need_prereg=True)
    pre = json.loads(M["prereg"].read_text(encoding="utf-8"))
    first = None if a.smoke else _committed_alone_clean(M["prereg"])[0]
    limit_memory(0.8)
    model, tok, rec = _gpu_load()
    for L in layers:
        if Ps[L]["summary"].exists() and not a.rescore:
            print(f"layer {L} already scored ({Ps[L]['summary'].name}): skipped", flush=True)
            continue
        t = pre["tests"][str(L)]
        _score_layer(model, tok, rec, L, Ps[L], t["tests_sha256"], t["counts"], first, a.batch, a.smoke)


def export_all(a):
    layers, Ps, M, man = _setup_multi(a)
    for L in layers:
        if Ps[L]["summary"].exists() or a.smoke:
            _export_layer(L, Ps[L], a.smoke)


# ---------------------------------------------------------------- stage 3: dossiers

def _context(w, p, before=40, after=8):
    from bonsai_interp import explore
    toks, tok = explore.corpus(), explore.tokenizer()
    ids = toks[w]
    return (tok.decode(ids[max(0, p - before):p]) + "⟦" + tok.decode(ids[p:p + 1]) + "⟧" + tok.decode(ids[p + 1:p + 1 + after])).replace("\n", "⏎")


def dossier(neuron, P, vb, links=False, model=None, max_bytes=64000):
    """Everything an agent needs to label one pole, from selection and screening windows only (never a test window)."""
    from bonsai_interp.graph import card
    name, L, n, pole = card.parse(neuron)
    nid = f"bonsai1/L{L}/N{n}{pole}"
    t1 = next((r for r in _read_gz(P["table"]) if r["neuron"] == nid), None)
    t2 = [r for r in _read_gz(P["screen"]) if r["neuron"] == nid] if P["screen"].exists() else []
    show = lambda t: vb["text"][t]
    d = {"neuron": nid, "protocol": PROTOCOL, "about": "input for labelling; NOT a test. Write a fresh quick-8+8@3 test after reading it. "
         "Fractions are of the pole's corpus max; stage 1 read windows 0..7999, stage 2 windows 8000..15999.",
         "card": card.card(nid, k=6)}
    if t1:
        d["stage1"] = {"class": t1["class"], "strong": t1.get("strong"),
                       "words": [{"token": show(t), "n": c, "precision": pr, "lift": li, "coverage": co} for t, c, pr, li, co in t1.get("words", [])],
                       "bigrams": [{"prev": show(a_), "token": show(t), "n": c, "precision": pr, "lift": li, "coverage": co}
                                   for a_, t, c, pr, li, co in t1.get("bigrams", [])],
                       "triggers": [{**t, "token": show(t["token"]), **({"prev": show(t["prev"])} if "prev" in t else {})} for t in t1.get("triggers", [])]}
    d["stage2"] = []
    for r in t2:
        e = {k: r[k] for k in ("trigger_text", "via", "n_occ", "strong_rate", "median_frac", "bin") if k in r}
        if "rule" in r:
            e["rule"] = {**r["rule"], "reads": " and ".join(condition_phrase(f, v, vb) for f, v in r["rule"]["conditions"])}
        if "features" in r:
            e["features"] = {f: {"eta2": x["eta2"], "top": [[value_text(f, v, vb), c, m, s] for v, c, m, s in x["top"]]}
                             for f, x in sorted(r["features"].items(), key=lambda kv: -kv[1]["eta2"])}
        e["template"] = r.get("template")
        e["examples"] = {k: [{"frac": fr, "text": _context(w, p)} for w, p, fr in v] for k, v in r.get("examples", {}).items()}
        d["stage2"].append(e)
    if links and model is not None:
        from bonsai_interp.probe.quick import one_back, one_forward
        rec = model[2]
        d["upstream"] = one_back(model[0], rec, L, n, 0 if pole == "+" else 1, k_show=6)
        d["downstream"] = one_forward(model[0], rec, L, n, 0 if pole == "+" else 1, k_show=6)
        rec_idx = ROOT / "graph" / "index.json"
        idx = json.loads(rec_idx.read_text(encoding="utf-8"))["neurons"] if rec_idx.exists() else {}
        for side in ("upstream", "downstream"):
            for u in d[side]:
                for sfx in ("+", "-"):
                    r_ = idx.get(f"bonsai1/{u['unit'].rstrip('+-')}{sfx}")
                    if r_:
                        u.setdefault("map", {})[sfx] = f"{r_['status']}: {r_.get('short') or r_.get('label')}"
    else:
        d["links"] = "not computed (run with --links on a GPU: play step / step --down)"
    blob = json.dumps(d, ensure_ascii=False)
    for k in (6, 4, 2, 1):  # trim card examples, then stage 2 examples, to stay under the 64 KB data-policy cap
        if len(blob.encode("utf-8")) <= max_bytes:
            break
        for p_ in d["card"]["poles"].values():
            p_["examples"] = p_["examples"][:k]
        for e in d["stage2"]:
            e["examples"] = {kk: v[:max(1, k // 2)] for kk, v in e["examples"].items()}
            for f in e.get("features", {}).values():
                f["top"] = f["top"][:max(2, k // 2)]
        blob = json.dumps(d, ensure_ascii=False)
    return d


def dossier_main(a):
    P, _ = _setup(a) if a.smoke else (_paths(a.layer), None)
    vb = vocab()
    if a.neuron:
        ids = a.neuron
    else:
        rows = _read_gz(P["screen"])
        ids = list(dict.fromkeys(r["neuron"] for r in rows if r["bin"] != "rare"))[:a.top]
    model = None
    if a.links:
        from bonsai_interp.env import limit_memory
        limit_memory(0.8)
        model = _gpu_load()
    P["dossiers"].mkdir(parents=True, exist_ok=True)
    for nid in ids:
        d = dossier(nid, P, vb, links=a.links, model=model)
        f = P["dossiers"] / (nid.replace("bonsai1/", "bonsai1_").replace("/", "_").replace("+", "_pos").replace("-", "_neg") + ".json")
        f.write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {f} ({f.stat().st_size / 1024:.1f} KB)")


# ---------------------------------------------------------------- the protocol manifest

def protocol(a):
    """The rule, pinned before stage 1 runs on this layer: commit this file ALONE, then run the stages."""
    P = _paths(a.layer)
    rel = MODULE.relative_to(ROOT).as_posix()
    if subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip():
        sys.exit(f"commit {rel} first: the manifest pins the committed code by sha256")
    man = {"protocol": PROTOCOL, "layer": a.layer, "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "toolchain_commit": _commit(), "code": rel, "code_sha256": _code_sha(),
           "windows": {"selection": f"{SELECT[0]}..{SELECT[1] - 1}", "screening": f"{SCREEN[0]}..{SCREEN[1] - 1}", "testing": f"{TEST[0]}.."},
           "rule": RULE, "features": list(FEATURES), "table_only": list(TABLE_ONLY), "bins": {"context": list(CONTEXT), "format": list(FORMAT), "copy": list(COPY), "predictive": list(PREDICTIVE),
                    "collocation": list(COLLOCATION), "topic": list(TOPIC), "positional": list(POSITIONAL)},
           "tested_bins": list(TESTED),
           "expectations": [
               "more poles at this layer get a held auto label than auto-token@1 found at layer 12 (34 of 24,576)",
               "lexical hits hold at a higher rate than context hits, and context hits at a higher rate than format hits",
               "among context and lexical hits, those whose minimal pairs confirm the bin hold at a higher rate than the rest",
               "most eligible poles have at least one word hit, and most hits land in context or unexplained, not lexical"],
           "note": "expectations are recorded before the run and reported either way; they are not gates"}
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    P["protocol"].write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {P['protocol'].relative_to(ROOT)}\nCommit it ALONE before running any stage: git add {P['protocol'].relative_to(ROOT).as_posix()} "
          f"&& git commit -m \"Pre-register auto-token@2, layer {a.layer}\"")


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("protocol", "amend", "hits", "rank", "screen", "refit", "prereg", "score", "export", "backup", "dossier",
                 "protocol-all", "amend-all", "hits-all", "screen-all", "prereg-all", "score-all", "export-all"):
        p = sub.add_parser(name)
        if name.endswith("-all"):
            p.add_argument("--layers", required=True, help="comma-separated layers, e.g. 0,1,2,3")
            p.add_argument("--rescore", action="store_true", help="score-all: score layers that already have a summary again")
            p.set_defaults(layer=None)
        else:
            p.add_argument("--layer", type=int, required=True)
        if name in ("amend", "amend-all"):
            p.add_argument("--note", help="what changed and why it is speed-only")
            p.add_argument("--parity", help="the parity check (same outcomes on the same inputs)")
        p.add_argument("--batch", type=int, default=8)
        p.add_argument("--smoke", help="run without the manifest checks, writing everything to this folder (CPU tests)")
        p.add_argument("--windows", type=int, help="read at most this many windows of each split (smoke tests)")
        p.add_argument("--tokens", type=int, help="read only the first this-many tokens of each window (smoke tests)")
        p.add_argument("--threads", type=int, help="torch CPU threads")
        p.add_argument("--rule", nargs="*", default=[], help="smoke only: override RULE entries, e.g. min_strong=3")
        p.add_argument("--limit-hits", type=int, help="smoke only: screen a random subset of this many hits")
        if name == "dossier":
            p.add_argument("--neuron", nargs="*")
            p.add_argument("--top", type=int, default=8, help="without --neuron: dossiers for the first N screened poles")
            p.add_argument("--links", action="store_true", help="add upstream feeders and downstream readers (loads the model)")
        if name == "backup":
            p.add_argument("--no-upload", action="store_true", help="only record the hashes in the summary")
    a = ap.parse_args()
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)
    if not a.smoke and (a.windows or a.tokens or a.rule or a.limit_hits):
        sys.exit("--windows, --tokens, --rule and --limit-hits are for --smoke runs only: a registered run reads the whole split")
    for kv in a.rule:
        k, v = kv.split("=", 1)
        RULE[k] = type(RULE[k])(v)
    {"protocol": protocol, "hits": hits, "rank": rank, "screen": screen, "prereg": prereg, "score": score, "export": export, "dossier": dossier_main, "refit": refit, "backup": backup,
     "protocol-all": protocol_all, "amend-all": amend_all, "amend": amend_all, "hits-all": hits_all, "screen-all": screen_all, "prereg-all": prereg_all, "score-all": score_all,
     "export-all": export_all}[a.cmd](a)


if __name__ == "__main__":
    main()
