"""The condition library compose@3 covers poles with: every grammatical, part-of-speech, morphological and code-syntax
condition of the deterministic passes, plus the two domain gates, evaluated at any (window, position) of the parsed
(English) and lexed (code) selection windows. CPU only.

  domain:english, domain:code         the window's domain (context units: a pole gated by language or code)
  class:X, morph:M                    sweeplib@4's part of speech and morphology (posmorph.py) at the token
  gram:C                              sweeplib@3's grammatical roles (cons.py) at the token
  dep:G                               the token's dependency group (cons.DEPGRP: subject, object, ...)
  qsent                               the token's sentence is a question
  role:R                              sweeplib@5's code syntax roles (codesyn/roles.py) at the token
A condition is false where its parse does not apply (an English condition in a code window and the reverse).
"""
from functools import lru_cache

import numpy as np

from bonsai_interp.auto import context_tier as ct

ROOT = ct.ROOT


@lru_cache(None)
def _parse():
    from bonsai_interp.grammar import parse as PA
    return PA.load(ROOT / "data" / "grammar" / "bonsai1.parse.select.npz")


@lru_cache(None)
def _lex():
    from bonsai_interp.codesyn import lex as X
    return X.load(ROOT / "data" / "codesyn" / "bonsai1.lex.select.npz")


def names():
    from bonsai_interp.codesyn import lex as X
    from bonsai_interp.grammar import cons as C
    from bonsai_interp.grammar import posmorph as PM
    out = ["domain:english", "domain:code"]
    out += [f"class:{x}" for x in PM.CLASSES] + [f"morph:{m}" for m in PM.MORPH] + ["morph:proper"]
    out += [f"gram:{c}" for c in C.CONDITIONS]
    out += [f"dep:{g}" for g in sorted(set(C.DEPGRP.values()))] + ["qsent"]
    out += [f"role:{r}" for r in X.ROLES]
    return out


def _english(z, ids, quiet=False):
    """{name: (W, T) bool} for the English conditions on a parse z. quiet: instead, each condition's natural complement
    where its pass defines one (sweeplib@3/4: the same class without the feature, the same word in another role), else
    None."""
    from bonsai_interp.grammar import cons as C
    from bonsai_interp.grammar import posmorph as PM
    pm, _ = PM.masks(z, ids)
    g, _ = C.masks(z, ids)
    if quiet:
        out = {k: (v[1] != None) for k, v in pm.items() if k.startswith(("class:", "morph:"))}  # noqa: E711 (elementwise)
        out.update({f"gram:{k}": (v[1] != None) for k, v in g.items()})  # noqa: E711
        return out
    out = {k: v[0] for k, v in pm.items() if k.startswith(("class:", "morph:"))}
    out.update({f"gram:{k}": v[0] for k, v in g.items()})
    dep = z["dep_names"][z["dep"]]
    grp = np.vectorize(lambda d: C.DEPGRP.get(d, "other"), otypes=[object])(dep)
    for gname in sorted(set(C.DEPGRP.values())):
        out[f"dep:{gname}"] = (grp == gname) & z["last"]
    out["qsent"] = z["qsent"] & z["last"]
    out["domain:english"] = np.ones(ids.shape, bool)
    return out


def _code(z, ids, quiet=False):
    from bonsai_interp.codesyn import roles as RO
    m = RO.masks(z, ids)
    if quiet:  # sweeplib@5's quiet side: the same kind of token outside the role (code roles: code tokens only)
        return {f"role:{r}": (v[1] != None) for r, v in m.items()}  # noqa: E711
    out = {f"role:{r}": v[0] for r, v in m.items()}
    out["domain:code"] = np.ones(ids.shape, bool)
    return out


def at(wp):
    """(len(wp), len(names())) bool: every condition at each (window, position) of wp (int array of window * 1024 + pos).
    Positions outside the parsed and lexed windows get all False."""
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
        M = fn(z, toks[z["windows"]].astype(np.int64))
        for n, m in M.items():
            out[sel, col[n]] = m[r[sel], p[sel]]
    return out


def base_rates(min_pos):
    """Per condition: its share of all positions >= min_pos over the parsed and lexed windows (the sample's population)."""
    from bonsai_interp import explore
    toks = explore.corpus()
    nm = names()
    tot = np.zeros(len(nm))
    n = 0
    for z, fn in ((_parse(), _english), (_lex(), _code)):
        M = fn(z, toks[z["windows"]].astype(np.int64))
        for k, m in M.items():
            tot[nm.index(k)] += m[:, min_pos:].sum()
        n += len(z["windows"]) * (1024 - min_pos)
    return tot / n
