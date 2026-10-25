"""The sweep library's contrast batteries (CPU only, deterministic).

A kind is a family of conditions. A condition is a predicate on a 48-token corpus snippet, judged at its last token. A
condition's battery is a set of minimal pairs drawn from different windows: a snippet where the condition holds ("fire")
and one that ends on the same token (the pair key) where it does not ("quiet"). Quiet snippets come in named strata, such
as English vs another language, and the sampler balances them. The battery does not depend on the pole, so one read of its
snippets scores every pole of every layer at once.

Splits are those of auto-token@2:
  - selection pairs come from windows 0 to 15,999;
  - test snippets come from window 16,400 onwards.
No test snippet is read before its test is registered. Positions below 64 are never used.
"""
import re
from functools import lru_cache

import numpy as np

from bonsai_interp.auto import context_tier as ct

SNIP = 48
MIN_POS = 64
SPLITS = {"select": (0, 16000), "test": (16400, None)}
PER_DOMAIN = (100_000, 100_000, 300_000, 150_000)  # sampled snippet ends per corpus domain (edu, web, multi, code) and split
N_SELECT, MIN_SELECT, N_TEST = 64, 32, 8
PER_KEY = {"select": 8, "test": 1}  # pairs per pair key (test: 2 when there are too few keys for 1)
WORDISH = ct.CONTENT | {"subword"}
SHARED = {"comma", "sentence-end", "colon", "open-bracket", "close-bracket", "number", "newline", "space", "quote", "symbol"}
FUNC = {"determiner", "preposition", "conjunction", "comma", "sentence-end", "auxiliary", "pronoun"}


class Snip:
    __slots__ = ("w", "p", "dom", "ids", "_text", "_head")

    def __init__(self, w, p, dom, ids):
        self.w, self.p, self.dom, self.ids = w, p, dom, ids
        self._text = self._head = None

    def text(self):
        """The whole snippet, decoded."""
        if self._text is None:
            self._text = _dec(self.ids)
        return self._text

    def head(self):
        """Everything before the last token, decoded (the context the last token is read in)."""
        if self._head is None:
            self._head = _dec(self.ids[:-1])
        return self._head


def _dec(ids):
    raw = ct.vocab()["raw"]
    return b"".join(raw[int(i)] for i in ids).decode("utf-8", errors="replace")


@lru_cache(None)
def snippets(split):
    """PER_DOMAIN snippet ends per domain, uniform over the split's windows and positions; seeded, so always the same."""
    from bonsai_interp import explore
    toks, dom = explore.corpus(), ct._domains()
    lo, hi = SPLITS[split]
    hi = len(toks) if hi is None else hi
    out = []
    for d in range(4):
        wins = np.nonzero(dom[lo:hi] == d)[0] + lo
        rng = np.random.default_rng([7, d, 0 if split == "select" else 1])
        w = rng.choice(wins, PER_DOMAIN[d])
        p = rng.integers(MIN_POS + SNIP - 1, toks.shape[1], PER_DOMAIN[d])
        for wi, pi in sorted(set(zip(w.tolist(), p.tolist()))):
            out.append(Snip(wi, pi, d, toks[wi, pi - SNIP + 1:pi + 1]))  # a view of the corpus, not a copy
    return out


# ---------------------------------------------------------------- kinds

class Kind:
    name, about = "", ""
    labels = {}  # condition -> the label a held test gets

    def classify(self, s, vb):
        """{condition: "fire" | quiet stratum name} for the conditions this snippet bears on (absent: no bearing)."""
        raise NotImplementedError

    def key(self, s, cond, vb):
        """Pair key: a fire and a quiet snippet are paired only when their keys are equal (default: the last token)."""
        return int(s.ids[-1])


class Copy(Kind):
    name = "copy"
    about = "induction and copying: the passage is re-typing something it already contains"
    labels = {f"{c}/{d}": t.format(d=d) for d in ("prose", "code") for c, t in (
        ("repeat", "copy/induction in {d}: fires on a word that already appeared earlier in the passage, not on the same word seen for the first time"),
        ("piece", "copy in {d}: fires on a later piece of a multi-piece word or name already seen earlier in the passage (the second mention), "
                  "not on the same piece in a first mention"),
        ("trigram", "copy in {d}: fires deep inside a repeated string, where the last three tokens repeat an earlier three-token sequence, "
                    "not on the same token where nothing repeats"))}

    def classify(self, s, vb):
        ids, f = s.ids, int(s.ids[-1])
        d = "code" if s.dom == 3 else "prose" if s.dom in (0, 1) else None
        if d is None or not vb["text"][f].strip():
            return {}
        out, prev = {}, int(ids[-2])
        seen = bool((ids[:-2] == f).any())
        first = not seen and ids[-2] != f
        if vb["cls"][f] in WORDISH and len(vb["text"][f].strip()) >= 3:
            out[f"repeat/{d}"] = "fire" if seen else "first time" if first else None
        # a word-internal piece: no leading space, and the previous token ends in a letter or digit (the same word goes on)
        if vb["an0"][f] and not vb["lead"][f] and vb["an1"][prev]:
            bi = bool(((ids[:-3] == prev) & (ids[1:-2] == f)).any())
            out[f"piece/{d}"] = "fire" if bi else "first mention" if first else None
        tri = bool(((ids[:-5] == ids[-3]) & (ids[1:-4] == ids[-2]) & (ids[2:-3] == f)).any())
        out[f"trigram/{d}"] = "fire" if tri else "not repeated" if first else None
        return {k: v for k, v in out.items() if v is not None}


_TOPICS = {
    "medicine": "patient patients doctor doctors hospital hospitals disease diseases treatment treatments symptom symptoms clinical surgery "
                "drug drugs medical diagnosis cancer therapy infection infections nurse nurses vaccine medication chronic",
    "law": "court courts judge judges law laws legal attorney attorneys lawyer lawyers plaintiff defendant statute trial jury lawsuit "
           "contract appeal verdict prosecutor",
    "sport": "game games team teams season player players coach league score scored goal goals championship match tournament playoff "
             "football basketball baseball soccer",
    "finance": "market markets stock stocks investor investors investment bank banks interest rates revenue profit earnings shares "
               "financial economy dollar loans debt fund",
    "food": "recipe recipes cook cooking bake sauce flour butter sugar salt oven chicken cheese garlic onion dish meal flavor taste kitchen "
            "ingredients",
    "computing": "software code computer computers data server user users file files program programming network database app "
                 "application linux browser algorithm",
    "religion": "god church faith jesus christ prayer bible lord holy spirit worship religious christian pastor gospel heaven sin temple "
                "scripture",
    "war": "war army military soldiers soldier troops battle forces enemy attack weapons navy combat invasion regiment commander",
    "music": "music song songs album band singer guitar concert musical musicians lyrics piano tour recorded rock jazz vocals",
    "biology": "cells cell species protein proteins gene genes dna organisms evolution bacteria tissue enzyme plants animals molecular genetic",
    "physics": "energy particles particle atoms atom electron electrons quantum molecules chemical reaction physics magnetic velocity mass "
               "wave",
    "politics": "election elections vote voters party government president minister senate congress political policy campaign candidate "
                "parliament democrats republicans",
}
TOPICS = {k: set(v.split()) for k, v in _TOPICS.items()}


class Topic(Kind):
    name = "topic"
    about = "topic context: the passage is about a subject, read at a neutral function word that names nothing"
    labels = {f"topic/{t}": f"topic: fires on ordinary words inside a passage about {t}, not on the same words in English text on other subjects"
              for t in TOPICS}

    def classify(self, s, vb):
        if s.dom not in (0, 1) or vb["cls"][int(s.ids[-1])] not in FUNC:
            return {}
        words = re.findall(r"[a-z]+", _dec(s.ids[:-3]).lower())
        if sum(w in ct.LANG["en"] for w in words) < 6:
            return {}
        hits = {t: sum(w in lex for w in words) for t, lex in TOPICS.items()}
        distinct = {t: len({w for w in words if w in lex}) for t, lex in TOPICS.items()}
        out = {}
        for t in TOPICS:
            rest = max(v for k, v in hits.items() if k != t)
            if hits[t] >= 3 and distinct[t] >= 2 and hits[t] >= 2 * rest:
                out[f"topic/{t}"] = "fire"
            elif hits[t] == 0:
                out[f"topic/{t}"] = "other topic" if rest >= 3 else "general"
        return out


LANGS = ("es", "fr", "de", "pt", "it", "nl")
SCRIPT_NAMES = ("cyrillic", "cjk", "arabic", "greek", "hebrew", "devanagari", "hangul", "thai")


def _lang(text):
    words = re.findall(r"\w+", text.lower())
    c = sorted(((sum(w in st for w in words), k) for k, st in ct.LANG.items()), reverse=True)
    return c[0][1] if c[0][0] >= 5 and c[0][0] >= 2 * c[1][0] else None


def _script(text):
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < 20:
        return None
    sc = {}
    for ch in letters:
        k = ct._script_of(ch)
        sc[k] = sc.get(k, 0) + 1
    k, v = max(sc.items(), key=lambda kv: kv[1])
    return k if v >= 0.6 * len(letters) else None


class Lang(Kind):
    name = "lang"
    about = "language and script gates: the passage is in one language or script, read at punctuation or digits every language shares"
    labels = {**{f"lang/{x}": f"language: fires inside {x} text, not inside English or other Latin-script languages" for x in LANGS},
              "lang/non-en": "language: fires inside non-English Latin-script text (es/fr/de/pt/it/nl), not inside English",
              **{f"script/{x}": f"script: fires inside {x}-script text, not inside Latin-script or other-script text" for x in SCRIPT_NAMES}}

    def classify(self, s, vb):
        if vb["cls"][int(s.ids[-1])] not in SHARED:
            return {}
        t = s.text()
        sc = _script(t)
        if sc is None:
            return {}
        out = {}
        if sc == "latin":
            lg = _lang(t)
            if lg is not None:
                for x in LANGS:
                    out[f"lang/{x}"] = "fire" if lg == x else "en" if lg == "en" else "other latin"
                out["lang/non-en"] = "en" if lg == "en" else "fire"
        for x in SCRIPT_NAMES:
            out[f"script/{x}"] = "fire" if sc == x else "latin" if sc == "latin" else "other script"
        return out


def _code_state(line):
    """(inside a string, inside a comment) at the end of one line of code: a small scanner (", ', #, //, backslash)."""
    q, i = None, 0
    while i < len(line):
        ch = line[i]
        if q:
            if ch == "\\":
                i += 1
            elif ch == q:
                q = None
        elif ch in "\"'":
            q = ch
        elif ch == "#" or line.startswith("//", i):
            return False, True
        i += 1
    return q is not None, False


def _depth(text, opens="([{", closes=")]}"):
    """Bracket depth at the end of the text and whether it ever went below zero (then it is unknown)."""
    d, low = 0, 0
    for ch in text:
        if ch in opens:
            d += 1
        elif ch in closes:
            d -= 1
            low = min(low, d)
    return d, low < 0


class Struct(Kind):
    name = "struct"
    about = "structure: where in the code or document the current word sits (string, comment, nesting, indentation, list, heading)"
    labels = {"code/in-string": "structure: fires on a word inside a string literal in code, not on the same word in code outside strings",
              "code/in-comment": "structure: fires on a word inside a code comment, not on the same word in code",
              "code/nested": "structure: fires on a word nested two or more brackets deep in code, not on the same word at depth 0",
              "code/indented": "structure: fires on a word on a code line indented 8 or more spaces, not on the same word on an unindented line",
              "prose/in-parens": "structure: fires on a word inside parentheses in prose, not on the same word outside them",
              "prose/in-quote": "structure: fires on a word inside a quotation in prose, not on the same word in unquoted text",
              "prose/list-item": "structure: fires on a word in a list item (a line starting with -, * or 1.), not on the same word in ordinary lines",
              "prose/heading": "structure: fires on a word in a markdown heading (a line starting with #), not on the same word in body text"}

    def classify(self, s, vb):
        f = int(s.ids[-1])
        if vb["cls"][f] not in WORDISH or s.dom == 2:
            return {}
        head = s.head()
        line = head.rsplit("\n", 1)[1] if "\n" in head else None
        out = {}
        if s.dom == 3:
            if line is not None:
                st, cm = _code_state(line)
                if st and not cm:
                    out["code/in-string"] = "fire"
                if cm:
                    out["code/in-comment"] = "fire"
                if not st and not cm:
                    out["code/in-string"] = out["code/in-comment"] = "code"
                ind = len(line) - len(line.lstrip(" \t")) + 3 * line[:len(line) - len(line.lstrip(" \t"))].count("\t")
                if line.strip():
                    out["code/indented"] = "fire" if ind >= 8 else "unindented" if ind == 0 else None
            d, unknown = _depth(head)
            if not unknown:
                out["code/nested"] = "fire" if d >= 2 else "depth 0" if d == 0 else None
        else:
            d, unknown = _depth(head, "(", ")")
            if not unknown:
                out["prose/in-parens"] = "fire" if d >= 1 else "outside"
            dq, lo, rc = head.count('"'), head.count("“"), head.count("”")
            if dq % 2 == 1 or lo > rc:
                out["prose/in-quote"] = "fire"
            elif dq == 0 and lo == 0 and rc == 0:
                out["prose/in-quote"] = "no quote"
            if line is not None and line.strip():
                out["prose/list-item"] = "fire" if re.match(r"\s*([-*•]|\d+[.)])\s", line) else "ordinary line"
                out["prose/heading"] = "fire" if re.match(r"#{1,6} ", line) else "body line"
        return {k: v for k, v in out.items() if v is not None}


class Number(Kind):
    name = "number"
    about = "numbers: which digit of a number this is, and what kind of number (year, decimal, price)"
    labels = {**{f"num/digit{k}": f"number: fires on the {['first', 'second', 'third', 'fourth'][k - 1]} digit of a number, "
                                  f"not on the same digit elsewhere in a number" for k in (1, 2, 3, 4)},
              "num/year": "number: fires on the last digit of a year (1500-2099), not on the fourth digit of other numbers",
              "num/decimal": "number: fires on a digit after a decimal point, not on the same digit in a whole number",
              "num/currency": "number: fires on a digit of an amount after a currency sign ($, €, £, ¥), not on the same digit in other numbers"}

    @staticmethod
    def _run(s, vb):
        ids, k = s.ids, 0
        while k < len(ids) and vb["digit"][int(ids[-1 - k])] and len(vb["text"][int(ids[-1 - k])].strip()) == 1:
            k += 1
        prev = vb["text"][int(ids[-1 - k])] if k < len(ids) else None
        prev2 = int(ids[-2 - k]) if k + 1 < len(ids) else None
        return k, prev, prev2

    def classify(self, s, vb):
        f = int(s.ids[-1])
        if not (vb["digit"][f] and len(vb["text"][f].strip()) == 1):
            return {}
        k, prev, prev2 = self._run(s, vb)
        if prev is None:
            return {}
        out = {f"num/digit{j}": "fire" if k == j else f"digit {min(k, 6)}" for j in (1, 2, 3, 4)}
        digits = "".join(vb["text"][int(i)].strip() for i in s.ids[len(s.ids) - k:])
        if k == 4:
            year = 1500 <= int(digits) <= 2099 and prev.strip() not in (".", ",", "$", "€", "£")
            out["num/year"] = "fire" if year else "other 4-digit"
        p = prev.strip()
        after_digit = prev2 is not None and vb["digit"][prev2]
        if p == "." and after_digit:
            out["num/decimal"] = "fire"
        elif p not in (".", ","):
            out["num/decimal"] = "whole number"
        if p[-1:] in ("$", "€", "£", "¥"):
            out["num/currency"] = "fire"
        elif p not in (".", ",", ":", "/", "-") and not after_digit:
            out["num/currency"] = "no currency sign"
        return out

    def key(self, s, cond, vb):
        if cond.startswith("num/digit") or cond == "num/year":
            return int(s.ids[-1])
        return int(s.ids[-1]), min(self._run(s, vb)[0], 3)  # same digit at the same place in its number


KINDS = {k.name: k for k in (Copy(), Topic(), Lang(), Struct(), Number())}


# ---------------------------------------------------------------- pairs

def _pairs(F, Q, n, per_key, seed):
    """Up to n (fire, quiet, stratum) pairs. Keys are taken in a seeded order, per_key pairs per key per round. Within a
    key the least-used stratum goes first. No window is used twice."""
    rng = np.random.default_rng(seed)
    keys = sorted(set(F) & set(Q), key=repr)
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(F[k])
        for st in Q[k].values():
            rng.shuffle(st)
    used, out, uses = set(), [], {}
    for _ in range(per_key):
        for k in keys:
            if len(out) >= n:
                return out
            f = next((s for s in F[k] if s.w not in used), None)
            if f is None:
                continue
            for st in sorted(Q[k], key=lambda x: (uses.get(x, 0), x)):
                q = next((s for s in Q[k][st] if s.w not in used and s.w != f.w), None)
                if q is not None:
                    used |= {f.w, q.w}
                    uses[st] = uses.get(st, 0) + 1
                    out.append((f, q, st))
                    break
    return out


def _placebo(Q, n, seed):
    """Up to n pairs of two quiet snippets with the same key (the condition fails in both): the null a selection is held to."""
    rng = np.random.default_rng(seed)
    keys = sorted(Q, key=repr)
    rng.shuffle(keys)
    used, out = set(), []
    for _ in range(PER_KEY["select"]):
        for k in keys:
            pool = [s for st in sorted(Q[k]) for s in Q[k][st] if s.w not in used]
            if len(pool) < 2 or len(out) >= n:
                continue
            a, b = (pool[i] for i in rng.choice(len(pool), 2, replace=False))
            if a.w != b.w:
                used |= {a.w, b.w}
                out.append((a, b, "placebo"))
    return out


@lru_cache(None)
def battery(kind, split):
    """{condition: [(fire Snip, quiet Snip, stratum)]} for one kind and split (selection: N_SELECT pairs, test: N_TEST).
    Both splits also have "placebo:<condition>": as many pairs of two quiet snippets, read and selected exactly as the
    real pairs are, so the number of poles a placebo selects estimates the false selections of its condition."""
    K, vb = KINDS[kind], ct.vocab()
    F, Q = {}, {}
    for s in snippets(split):
        for c, v in K.classify(s, vb).items():
            k = K.key(s, c, vb)
            if v == "fire":
                F.setdefault(c, {}).setdefault(k, []).append(s)
            else:
                Q.setdefault(c, {}).setdefault(k, {}).setdefault(v, []).append(s)
    out = {}
    for c in K.labels:
        seed = [11, sum(map(ord, c)), 0 if split == "select" else 1]
        if split == "select":
            out[c] = _pairs(F.get(c, {}), Q.get(c, {}), N_SELECT, PER_KEY["select"], seed)
            out["placebo:" + c] = _placebo(Q.get(c, {}), len(out[c]), seed + [2])
        else:
            out[c] = _pairs(F.get(c, {}), Q.get(c, {}), N_TEST, PER_KEY["test"], seed)
            if len(out[c]) < N_TEST:  # few distinct keys: allow a second pair per key
                out[c] = _pairs(F.get(c, {}), Q.get(c, {}), N_TEST, 2, seed)
            out["placebo:" + c] = _placebo(Q.get(c, {}), N_TEST, seed + [2])
    return out
