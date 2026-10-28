"""compose@6's condition library: compose@4's (conds2.py, unchanged) plus anticipation conditions: what comes next.

  next1:C, next2:C   compose@3's condition C holds at the token right after the read token, or the one after that

The read snippet ends at the read token, so the model never sees the next token: a pole that fires before C is
anticipating it from the context. The census (docs/CENSUS.md) found prediction-type poles rising with depth (0% in
L0-11 to 34% in L30-35). For mid layers, this asks whether poles track what is about to come rather than what is here.
Test pairs end on the same read token, one followed by C in the corpus and one not.
"""
import numpy as np

from bonsai_interp.auto import context_tier as ct
from bonsai_interp.compose import conds as C3
from bonsai_interp.compose import conds2 as C2

ROOT = ct.ROOT
_parse, _lex = C2._parse, C2._lex
AHEAD = (1, 2)


def names():
    base = [n for n in C3.names() if not n.startswith("domain:")]
    return C2.names() + [f"next{k}:{n}" for k in AHEAD for n in base]


def _ahead(M):
    out = {}
    for k in AHEAD:
        for n, m in M.items():
            if not n.startswith("domain:"):
                out[f"next{k}:{n}"] = np.pad(m[:, k:], ((0, 0), (0, k)))
    return out


def _english(z, ids, quiet=False):
    if quiet:
        return C2._english(z, ids, quiet=True)
    M = C2._english(z, ids)
    return {**M, **_ahead(C3._english(z, ids))}


def _code(z, ids, quiet=False):
    if quiet:
        return C2._code(z, ids, quiet=True)
    M = C2._code(z, ids)
    return {**M, **_ahead(C3._code(z, ids))}


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
