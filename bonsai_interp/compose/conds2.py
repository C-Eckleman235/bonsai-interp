"""compose@4's condition library: compose@3's (conds.py, unchanged) plus text-format conditions and every condition shifted
to the previous token and the one before it. CPU only. The same interface as conds.py (names, at, population, _parse,
_lex, _english, _code), so compose@3's engine runs on it.

  fmt:line_start      the first token of a line            fmt:sent_start   the first word after . ! or ?
  fmt:digit           a token of digits                    fmt:after_digit  right after one
  fmt:capitalized     a capitalized word (Word)            fmt:allcaps      an all-capitals word (WORD)
  fmt:punct           a punctuation-only token             fmt:after_punct  right after one
  fmt:in_parens       inside ( ) or [ ] opened on the same line
  fmt:in_quote        inside a quotation opened on the same line (straight quotes by parity, curly quotes by pairs)
  fmt:url             in a URL (from http or www to the next whitespace)
  fmt:list_line       on a line that opens with -, *, • or a number      fmt:header_line  on a line that opens with #
  prev1:C, prev2:C    compose@3's condition C holds at the token 1 or 2 back (domain gates excluded)
Format conditions apply in both domains; a shifted condition applies in its own condition's domain.
"""
from functools import lru_cache

import numpy as np

from bonsai_interp.auto import context_tier as ct
from bonsai_interp.compose import conds as C3

ROOT = ct.ROOT
_parse, _lex = C3._parse, C3._lex
FMT = ("fmt:line_start", "fmt:sent_start", "fmt:digit", "fmt:after_digit", "fmt:capitalized", "fmt:allcaps", "fmt:punct",
       "fmt:after_punct", "fmt:in_parens", "fmt:in_quote", "fmt:url", "fmt:list_line", "fmt:header_line")
SHIFTS = (1, 2)


def names():
    base = C3.names()
    return base + list(FMT) + [f"prev{k}:{n}" for k in SHIFTS for n in base if not n.startswith("domain:")]


@lru_cache(None)
def _tok():
    """Per token id: the attributes the format conditions read."""
    text = [t for t in ct.vocab()["text"]]
    s = [t.strip() for t in text]
    a = {
        "nl": np.array(["\n" in t for t in text]),
        "digit": np.array([x.isdigit() for x in s]),
        "cap": np.array([x[:1].isupper() and x[1:].islower() and x.isalpha() for x in s]),
        "allcaps": np.array([len(x) > 1 and x.isalpha() and x.isupper() for x in s]),
        "punct": np.array([len(x) > 0 and not any(ch.isalnum() for ch in x) for x in s]),
        "end": np.array([x.endswith((".", "!", "?")) and not any(ch.isalnum() for ch in x) for x in s]),
        "open": np.array([x.count("(") + x.count("[") for x in s], np.int32),
        "close": np.array([x.count(")") + x.count("]") for x in s], np.int32),
        "dq": np.array([x.count('"') for x in s], np.int32),
        "cq_open": np.array([x.count("“") for x in s], np.int32),
        "cq_close": np.array([x.count("”") for x in s], np.int32),
        "url": np.array(["http" in x or x.startswith("www") for x in s]),
        "ws_start": np.array([t[:1].isspace() for t in text]),
        "list": np.array([x in ("-", "*", "•") or (x[:-1].isdigit() and x.endswith(".")) or x.isdigit() for x in s]),
        "header": np.array([x.startswith("#") and set(x) <= {"#"} for x in s]),
    }
    return a


def _line_scan(x, a):
    """Line-scoped running state per position: bracket depth, straight-quote parity, curly-quote depth, the first token of
    the line, and whether a URL is open."""
    W, T = x.shape
    nl = a["nl"][x]
    depth = np.zeros((W, T), np.int32)
    dq = np.zeros((W, T), np.int32)
    cq = np.zeros((W, T), np.int32)
    first = np.zeros((W, T), np.int64)
    url = np.zeros((W, T), bool)
    d = np.zeros(W, np.int32)
    q = np.zeros(W, np.int32)
    c = np.zeros(W, np.int32)
    f = x[:, 0].copy()
    u = np.zeros(W, bool)
    newline = np.ones(W, bool)
    for t in range(T):
        tok = x[:, t]
        f = np.where(newline, tok, f)
        u = a["url"][tok] | (u & ~a["ws_start"][tok] & ~newline)  # a URL runs until a token that starts with whitespace
        d = np.maximum(d + a["open"][tok] - a["close"][tok], 0)
        q = q + a["dq"][tok]
        c = np.maximum(c + a["cq_open"][tok] - a["cq_close"][tok], 0)
        depth[:, t], dq[:, t], cq[:, t], first[:, t], url[:, t] = d, q, c, f, u
        newline = nl[:, t]
        d = np.where(newline, 0, d)
        q = np.where(newline, 0, q)
        c = np.where(newline, 0, c)
    return depth, dq, cq, first, url, nl


def fmt(ids):
    """{fmt name: (W, T) bool} on token ids."""
    a = _tok()
    x = ids
    depth, dq, cq, first, url, nl = _line_scan(x, a)
    prev_nl = np.pad(nl[:, :-1], ((0, 0), (1, 0)), constant_values=True)
    punct = a["punct"][x]
    digit = a["digit"][x]

    def after(m):
        return np.pad(m[:, :-1], ((0, 0), (1, 0)))
    return {
        "fmt:line_start": prev_nl & ~nl,
        "fmt:sent_start": after(a["end"][x]) & ~punct,
        "fmt:digit": digit,
        "fmt:after_digit": after(digit),
        "fmt:capitalized": a["cap"][x],
        "fmt:allcaps": a["allcaps"][x],
        "fmt:punct": punct,
        "fmt:after_punct": after(punct) & ~punct,
        "fmt:in_parens": depth > 0,
        "fmt:in_quote": ((dq % 2) == 1) | (cq > 0),
        "fmt:url": url,
        "fmt:list_line": a["list"][first],
        "fmt:header_line": a["header"][first],
    }


def _shift(M):
    out = {}
    for k in SHIFTS:
        for n, m in M.items():
            if not n.startswith("domain:"):
                out[f"prev{k}:{n}"] = np.pad(m[:, :-k], ((0, 0), (k, 0)))
    return out


def _english(z, ids, quiet=False):
    if quiet:
        return C3._english(z, ids, quiet=True)  # natural complements: compose@3's (none for the new conditions)
    M = C3._english(z, ids)
    return {**M, **fmt(ids), **_shift(M)}


def _code(z, ids, quiet=False):
    if quiet:
        return C3._code(z, ids, quiet=True)
    M = C3._code(z, ids)
    return {**M, **fmt(ids), **_shift(M)}


def at(wp):
    """(len(wp), len(names())) bool, as conds.at."""
    from bonsai_interp import explore
    toks = explore.corpus()
    nm = names()
    col = {n: i for i, n in enumerate(nm)}
    wp = np.asarray(wp, np.int64)
    w, p = wp // 1024, wp % 1024
    out = np.zeros((len(wp), len(nm)), bool)
    for z, fn in ((_parse(), _english), (_lex(), _code)):
        row = np.full(toks.shape[0], -1, np.int64)
        row[z["windows"]] = np.arange(len(z["windows"]))
        r = row[w]
        sel = np.nonzero(r >= 0)[0]
        if not len(sel):
            continue
        for n, m in fn(z, toks[z["windows"]].astype(np.int64)).items():
            out[sel, col[n]] = m[r[sel], p[sel]]
    return out


def population(min_pos):
    """As conds.population, over this library."""
    from bonsai_interp import explore
    toks = explore.corpus()
    nm = names()
    col = {n: i for i, n in enumerate(nm)}
    V = ct.vocab()["V"]
    hist = np.zeros((len(nm), V), np.int64)
    alltok = np.zeros(V, np.int64)
    n = 0
    for z, fn in ((_parse(), _english), (_lex(), _code)):
        ids = toks[z["windows"]].astype(np.int64)
        alltok += np.bincount(ids[:, min_pos:].ravel(), minlength=V)
        for k, m in fn(z, ids).items():
            hist[col[k]] += np.bincount(ids[:, min_pos:][m[:, min_pos:]], minlength=V)
        n += len(z["windows"]) * (1024 - min_pos)
    return hist, n, alltok
