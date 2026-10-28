"""compose@5's condition library: compose@4's (conds2.py, unchanged) plus far-context conditions. CPU only; the same
interface as conds.py, so compose@3's engine runs on it.

For each of compose@4's position conditions C (the read-token and format conditions, not the shifted or domain ones):
  win8:C      C held at one of the 8 tokens before the read token (not at the read token itself)
  dense32:C   C held at unusually many of the 32 tokens before the read token: at least the count that only a tenth of
              positions reach (the 90th percentile over the selection windows; at least 2)
These read context beyond the token pair: register and genre states such as a past-tense narrative, a conversational
passage full of pronouns, or a bracketed aside. They fit what the 2026-10-28 diagnostic found: at layers 14-16 the
conditions that best cover poles are of this kind (a pronoun in the last 8 tokens, past tense in the last 32). All
stay within the 48 tokens a test snippet shows.
"""
from functools import lru_cache

import numpy as np

from bonsai_interp.auto import context_tier as ct
from bonsai_interp.compose import conds2 as C2

ROOT = ct.ROOT
_parse, _lex = C2._parse, C2._lex
NEAR, FAR, DENSE_Q = 8, 32, 0.9


def _base_names():
    return [n for n in C2.names() if not n.startswith(("prev", "domain:"))]


def names():
    b = _base_names()
    return C2.names() + [f"win{NEAR}:{n}" for n in b] + [f"dense{FAR}:{n}" for n in b]


def _prior_count(m, W):
    """(W, T): how many of the W tokens before each position have m (the position itself excluded)."""
    c = np.cumsum(m, axis=1, dtype=np.int32)
    return np.pad(c[:, :-1], ((0, 0), (1, 0))) - np.pad(c[:, :-(W + 1)], ((0, 0), (W + 1, 0)))


@lru_cache(None)
def thresholds():
    """{condition: the dense32 count threshold}, from the selection windows (positions >= 64), both domains together."""
    from bonsai_interp import explore
    toks = explore.corpus()
    counts = {}
    for z, fn in ((_parse(), C2._english), (_lex(), C2._code)):
        M = fn(z, toks[z["windows"]].astype(np.int64))
        for n in _base_names():
            if n in M:
                pc = _prior_count(M[n], FAR)[:, 64:].ravel()
                counts.setdefault(n, []).append(np.bincount(pc, minlength=FAR + 1)[:FAR + 1])
    out = {}
    for n, hs in counts.items():
        h = np.sum(hs, axis=0)
        cdf = np.cumsum(h) / h.sum()
        out[n] = max(2, int(np.searchsorted(cdf, DENSE_Q) + 1))
    return out


def _context(M):
    th = thresholds()
    out = {}
    for n in _base_names():
        if n not in M:
            continue
        out[f"win{NEAR}:{n}"] = _prior_count(M[n], NEAR) > 0
        out[f"dense{FAR}:{n}"] = _prior_count(M[n], FAR) >= th[n]
    return out


def _english(z, ids, quiet=False):
    if quiet:
        return C2._english(z, ids, quiet=True)
    M = C2._english(z, ids)
    return {**M, **_context(M)}


def _code(z, ids, quiet=False):
    if quiet:
        return C2._code(z, ids, quiet=True)
    M = C2._code(z, ids)
    return {**M, **_context(M)}


def at(wp):
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
