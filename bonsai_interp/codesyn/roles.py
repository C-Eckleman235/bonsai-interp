"""Code syntax roles as minimal-pair batteries (sweeplib@5; CPU only, deterministic, from the Pygments lex in lex.py).

The model is also a model of programming languages, whose syntax is explicit: an if statement's condition, a call's
arguments, a list literal, a dictionary's keys. Each role (lex.ROLES) is a condition on a snippet's last token, which may be
any non-whitespace token of a Python window. Two families:

  role:R   the same token in role R against that token elsewhere ('x' in an if condition vs 'x' outside one); quiet snippets
           are spread over the token's other contexts (strata: the innermost context, lex.CTX). For a code role both
           tokens are code (neither is in a string, docstring or comment); for string, docstring and comment the quiet token
           is the same token in code or another of the three.
  any:R    role R whatever the token, against tokens of the same coarse type (name, keyword, number, ...) and corpus-frequency
           band outside R. Not for roles that fix the token type (strings, comments, docstrings, def and class names).

Both snippets must show the token's whole logical line: its start is at most SNIP - 2 tokens back, so the role is decidable
from the 48 tokens the model reads.
"""
import numpy as np

from bonsai_interp.auto import context_tier as ct
from bonsai_interp.codesyn import lex as X
from bonsai_interp.grammar import cons as C
from bonsai_interp.grammar import posmorph as PM
from bonsai_interp.sweeplib import kinds as SK

ABOUT = {
    "cond": "the condition of an if, elif or while statement",
    "for_target": "the loop variable of a for statement (between 'for' and 'in')",
    "for_iter": "the iterable of a for statement (between 'in' and ':')",
    "return": "a return statement's expression",
    "import": "an import statement",
    "def_name": "the name of a function being defined (after 'def')",
    "def_sig": "a function definition's parameter list",
    "class_name": "the name of a class being defined (after 'class')",
    "decorator": "a decorator line",
    "call_arg": "a function call's argument list (innermost brackets)",
    "kwarg_name": "a keyword argument's name in a call ('b' in 'f(b=2)')",
    "subscript": "a subscript or slice ('x[i]' at 'i')",
    "list": "a list literal or list comprehension (innermost brackets)",
    "dict_key": "a dictionary literal's key",
    "dict_value": "a dictionary literal's value",
    "comprehension": "the 'for ... in ...' part of a comprehension",
    "assign_lhs": "the target of an assignment (left of a top-level '=')",
    "assign_rhs": "the value of an assignment (right of a top-level '=')",
    "string": "a string literal (not a docstring)",
    "docstring": "a docstring",
    "comment": "a comment",
    "nested": "two or more levels of brackets deep, not one",
}
LITERAL = ("string", "docstring", "comment")
NO_ANY = LITERAL + ("def_name", "class_name")
PER_KEY = {"tok": {"select": 8, "test": 1}, "any": {"select": 16, "test": 2}}
CAP = {"tok": 12, "any": 400}


def labels():
    out = {f"role:{r}": f"code syntax: fires on a token in {ABOUT[r]}, not on the same token elsewhere in code" for r in X.ROLES}
    out.update({f"any:{r}": f"code syntax (any token): fires on {ABOUT[r]}, whatever the token, not on tokens of the same type "
                            "and frequency elsewhere" for r in X.ROLES if r not in NO_ANY})
    return out


def masks(z, ids):
    """{role: (fire (W, T) bool, quiet (W, T) object array of strata or None)}: no position cutoff (the battery applies it)."""
    W, T = ids.shape
    ok = z["ok"] & (z["start"] >= 0) & (np.arange(T)[None, :] - z["start"] <= C.SNIP - 2)
    ctx = z["ctx_names"][z["ctx"]].astype(object)
    lit = (z["roles"] & np.uint32(X.BIT["string"] | X.BIT["docstring"] | X.BIT["comment"])) > 0
    out = {}
    for r in X.ROLES:
        has = (z["roles"] & np.uint32(X.BIT[r])) > 0
        if r not in LITERAL:  # a code role: both sides are code tokens (not a string, docstring or comment)
            has = has & ~lit
        q = np.full((W, T), None, dtype=object)
        qm = ok & ~has & (~lit if r not in LITERAL else True)
        if r == "nested":
            qm &= ~np.isin(ctx, ("top", "string", "comment"))  # one level deep: inside a bracket, not two
        q[qm] = ctx[qm]
        out[r] = (ok & has, q)
    return out


def battery(z, split, conds=None):
    """{condition: [(fire Snip, quiet Snip, stratum)], "placebo:<condition>": [...]}."""
    from bonsai_interp import explore
    toks, dom = explore.corpus(), ct._domains()
    win = z["windows"]
    ids = toks[win].astype(np.int64)
    M = masks(z, ids)
    okpos = np.zeros(ids.shape, bool)
    okpos[:, C.MIN_POS + C.SNIP - 1:] = True
    band = PM._band()
    out = {}
    for name in (conds or labels()):
        var, r = name.split(":")
        var = "tok" if var == "role" else "any"
        fire, quiet = M[r]
        fire = fire & okpos
        qm = (quiet != None) & okpos  # noqa: E711 (elementwise)
        fw, fp = np.nonzero(fire)
        qw, qp = np.nonzero(qm)
        fk, qk = ids[fw, fp], ids[qw, qp]
        if var == "any":
            fk = z["kind"][fw, fp].astype(np.int64) * 16 + band[fk]
            qk = z["kind"][qw, qp].astype(np.int64) * 16 + band[qk]
        seed = [17, sum(map(ord, name)), 0 if split == "select" else 1]
        rng = np.random.default_rng(seed + [5])
        F = C._cap(rng, fk.tolist(), win[fw], fp, CAP[var])
        Q = C._cap(rng, qk.tolist(), win[qw], qp, CAP[var], quiet[qw, qp].tolist())
        n, pk = (C.N_SELECT, PER_KEY[var]["select"]) if split == "select" else (C.N_TEST, PER_KEY[var]["test"])
        pairs = SK._pairs(F, Q, n, pk, seed)
        if split == "test" and len(pairs) < C.N_TEST:
            pairs = SK._pairs(F, Q, n, 2 * pk, seed)
        plac = C._placebo(Q, len(pairs) if split == "select" else C.N_TEST, seed + [2], PER_KEY[var]["select"])

        def snip(e):
            return SK.Snip(e.w, e.p, int(dom[e.w]), toks[e.w, e.p - C.SNIP + 1:e.p + 1])
        out[name] = [(snip(f), snip(q), st) for f, q, st in pairs]
        out["placebo:" + name] = [(snip(a), snip(b), st) for a, b, st in plac]
    return out


def holds(z, conds):
    """{condition: (W, T) bool} where each role holds (coverage; the visibility rule applies, no position cutoff)."""
    from bonsai_interp import explore
    ids = explore.corpus()[z["windows"]].astype(np.int64)
    M = masks(z, ids)
    return {c: M[c.split(":")[1]][0] for c in conds}
