"""Lex the corpus's Python code windows with Pygments and map syntactic roles onto model tokens (CPU only, deterministic).

  python -m bonsai_interp.codesyn.lex select     # -> data/codesyn/bonsai1.lex.select.npz
  python -m bonsai_interp.codesyn.lex test       # -> data/codesyn/bonsai1.lex.test.npz (only after the protocol is committed)

A window (1,024 model tokens) of the code domain counts as Python when at least 3 of its lines open with a Python statement
keyword (def, class, import, from, if, elif, for, while, return, with, try, except) and they outnumber the lines that end in
';', '{' or '}'. Its text is lexed whole with Pygments' PythonLexer; a bracket stack and a per-statement state machine then
give every Pygments token a set of roles (ROLES), its innermost context (CTX) and the start of its logical line. Every model
token takes the attributes of the Pygments token that holds its first non-space byte. Per model token, arrays (windows, 1024):
  roles    uint32 bit mask over ROLES
  ctx      code into CTX: the innermost context (a string, a comment, or the innermost bracket's kind; 'top' outside brackets)
  kind     code into KINDS: the Pygments token type, coarsened
  start    model position of the token's logical line start (where the statement holding it begins; -1 unknown)
  ok       the token is lexed Python: a non-whitespace model token after the window's first line break
A window that opens inside a triple-quoted string is lexed with that quote prepended (an odd count of one triple quote).
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "codesyn"
SPLITS = {"select": (0, 16000), "test": (16400, None)}  # the sweep library's splits
DOMAIN = 3  # code
ROLES = ("cond", "for_target", "for_iter", "return", "import", "def_name", "def_sig", "class_name", "decorator", "call_arg",
         "kwarg_name", "subscript", "list", "dict_key", "dict_value", "comprehension", "assign_lhs", "assign_rhs", "string",
         "docstring", "comment", "nested")
BIT = {r: 1 << i for i, r in enumerate(ROLES)}
CTX = ("", "top", "string", "comment", "call", "def_sig", "class_bases", "subscript", "list", "paren", "brace")
KINDS = ("", "name", "builtin", "function", "class", "keyword", "operator", "punct", "number", "string", "comment", "other")
HEADS = {"if": "cond", "elif": "cond", "while": "cond", "for": "for", "return": "return", "import": "import", "from": "import",
         "def": "def", "class": "class"}
COMPOUND = ("if", "elif", "while", "for", "def", "class", "else", "try", "except", "finally", "with")
PY_LINE = re.compile(r"^\s*(def|class|import|from|if|elif|for|while|return|with|try|except)\b")
C_LINE = re.compile(r"[;{}]\s*$")


def is_python(text):
    lines = text.split("\n")
    py = sum(bool(PY_LINE.match(x)) for x in lines)
    return py >= 3 and py > sum(bool(C_LINE.search(x)) for x in lines)


def _kind(t):
    from pygments.token import Comment, Keyword, Name, Number, Operator, Punctuation, String
    if t in Comment:
        return "comment"
    if t in String:
        return "string"
    if t in Number:
        return "number"
    if t in Keyword or t in Operator.Word:
        return "keyword"
    if t in Name.Builtin or t in Name.Builtin.Pseudo:
        return "builtin"
    if t in Name.Function:
        return "function"
    if t in Name.Class:
        return "class"
    if t in Name:
        return "name"
    if t in Operator:
        return "operator"
    if t in Punctuation:
        return "punct"
    return "other"


def roles(text):
    """Per Pygments token: (start char, end char, roles mask, ctx, kind, logical line start char), for a Python text."""
    from pygments.lexers import PythonLexer
    from pygments.token import Comment, Keyword, Name, Operator, Punctuation, String, Text
    toks = list(PythonLexer(stripnl=False, ensurenl=False).get_tokens_unprocessed(text))
    out = []
    stack = []  # brackets: dict(ch, kind, item=[token indices of the current item], colon=bool, comp=bool)
    line_start, head, stage, prev_sig, sig_since_line, first = None, None, 0, None, 0, None
    lines = []  # per logical line: (indices of depth-0 tokens, index of the first depth-0 '=' or None, head)
    cur, eq = [], None
    CLOSE = {")": "(", "]": "[", "}": "{"}

    def end_line():
        nonlocal cur, eq, line_start, head, stage, sig_since_line
        lines.append((cur, eq, head))
        cur, eq, line_start, head, stage, sig_since_line = [], None, None, None, 0, 0

    for i, (c0, t, v) in enumerate(toks):
        if t in Text and v.strip() == "":
            if "\n" in v and not stack and not (prev_sig is not None and toks[prev_sig][2] == "\\"):
                end_line()
            out.append(None)
            continue
        if line_start is None:
            line_start = c0
        m = 0
        ctx = "top"
        is_open, is_close = t in Punctuation and v in "([{", t in Punctuation and v in ")]}"
        if is_close and stack and stack[-1]["ch"] == CLOSE[v]:
            ctx = stack[-1]["kind"]  # a closer belongs to its bracket
        elif stack:
            ctx = stack[-1]["kind"]
        if t in String:
            m |= BIT["docstring"] if t in String.Doc else BIT["string"]
            ctx = "string" if t not in String.Interpol else ctx
        elif t in Comment:
            m |= BIT["comment"]
            ctx = "comment"
        depth = len(stack)
        if depth >= 2:
            m |= BIT["nested"]
        if stack:
            b = stack[-1]
            m |= {"call": BIT["call_arg"], "def_sig": BIT["def_sig"], "subscript": BIT["subscript"], "list": BIT["list"]}.get(b["kind"], 0)
            if b["comp"]:
                m |= BIT["comprehension"]
        # the statement's head keyword (the first significant token of a logical line or after a compound header's ':')
        if t in Comment:
            pass
        elif sig_since_line == 0:
            if t in Keyword and v in HEADS:
                head = HEADS[v]
            elif t in Name.Decorator or (t in Operator and v == "@"):
                head = "decorator"
            else:
                head = None
            stage = 0
        if not (t in Comment):
            sig_since_line += 1
        if head == "cond" and sig_since_line > 1 and not stack and t in Punctuation and v == ":":
            head, sig_since_line = None, 0
        elif head == "cond" and sig_since_line > 1:
            m |= BIT["cond"]
        elif head == "for" and sig_since_line > 1:
            if not stack and t in Operator.Word and v == "in" and stage == 0:
                stage = 1
            elif not stack and t in Punctuation and v == ":":
                head, sig_since_line = None, 0
            else:
                m |= BIT["for_target"] if stage == 0 else BIT["for_iter"]
        elif head == "return" and sig_since_line > 1:
            m |= BIT["return"]
        elif head == "import":
            m |= BIT["import"]
        elif head == "decorator":
            m |= BIT["decorator"]
        elif head == "def":
            if t in Name.Function:
                m |= BIT["def_name"]
            elif not stack and t in Punctuation and v == ":" and stage == 0:
                head, sig_since_line = None, 0
        elif head == "class":
            if t in Name.Class:
                m |= BIT["class_name"]
            elif not stack and t in Punctuation and v == ":":
                head, sig_since_line = None, 0
        if sig_since_line == 1:
            first = v if t in Keyword else None
        elif head is None and first in COMPOUND and not stack and t in Punctuation and v == ":":
            sig_since_line = 0  # 'else:', 'try:', 'with x as y:': the rest of the line is a new statement
        split = sig_since_line == 0 and not (t in Comment)  # this token ended a compound header
        # brackets
        if stack and t in Keyword and v == "for" and stack[-1]["kind"] in ("list", "paren", "brace"):
            stack[-1]["comp"] = True
            m |= BIT["comprehension"]
        if stack and stack[-1]["kind"] == "call" and t in Name and i + 1 < len(toks) and toks[i + 1][2] == "=":
            m |= BIT["kwarg_name"]
        if stack and stack[-1]["kind"] == "brace" and not (is_close and stack[-1]["ch"] == "{"):
            b = stack[-1]
            if t in Punctuation and v == ":" and not b["colon"]:
                for j in b["item"]:
                    out[j][2] |= BIT["dict_key"]
                b["colon"] = True
            elif t in Punctuation and v == ",":
                b["item"], b["colon"] = [], False
            elif b["colon"]:
                m |= BIT["dict_value"]
            else:
                b["item"].append(i)
        if is_open:
            p = toks[prev_sig] if prev_sig is not None else None
            after_value = p is not None and (p[1] in Name or p[2] in (")", "]") or p[1] in String) and not (p[1] in Keyword)
            if v == "(":
                kind = ("def_sig" if p is not None and p[1] in Name.Function else
                        "class_bases" if p is not None and p[1] in Name.Class else "call" if after_value else "paren")
            elif v == "[":
                kind = "subscript" if after_value else "list"
            else:
                kind = "brace"
            stack.append({"ch": v, "kind": kind, "item": [], "colon": False, "comp": False})
        elif is_close and stack and stack[-1]["ch"] == CLOSE[v]:
            stack.pop()
        elif is_close:
            stack = []  # unbalanced: start over
        if not stack and t in Operator and v == "=" and eq is None and head is None:
            eq = i
        if split:
            lines.append((cur, eq, None))
            cur, eq = [], None
        else:
            cur.append(i)
        out.append([c0, c0 + len(v), m, ctx, _kind(t), line_start])
        prev_sig = i
    end_line()
    for idx, eq_, hd in lines:
        if eq_ is None or hd is not None:
            continue
        for j in idx:
            if out[j] is None or j == eq_:
                continue
            out[j][2] |= BIT["assign_lhs"] if j < eq_ else BIT["assign_rhs"]
    return [x for x in out if x is not None]


def _prep(text):
    for q in ('"""', "'''"):
        if text.count(q) % 2 == 1:
            return q + text, len(q)
    return text, 0


def lex(ws):
    """{roles, ctx, kind, start, ok} arrays (len(ws), 1024) and the windows' Python flags."""
    from bonsai_interp import explore
    from bonsai_interp.auto import context_tier as ct
    raw, toks = ct.vocab()["raw"], explore.corpus()
    T = 1024
    out = {"roles": np.zeros((len(ws), T), np.uint32), "ctx": np.zeros((len(ws), T), np.uint8),
           "kind": np.zeros((len(ws), T), np.uint8), "start": np.full((len(ws), T), -1, np.int16),
           "ok": np.zeros((len(ws), T), bool)}
    py = np.zeros(len(ws), bool)
    ci, ki = {c: i for i, c in enumerate(CTX)}, {k: i for i, k in enumerate(KINDS)}
    t0 = time.time()
    for r, w in enumerate(ws):
        parts = [raw[int(i)] for i in toks[w]]
        b = b"".join(parts)
        try:
            text = b.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if not is_python(text):
            continue
        py[r] = True
        src, shift = _prep(text)
        tt = roles(src)
        if not tt:
            continue
        ob = np.cumsum([0] + [len(p) for p in parts])
        cb = np.cumsum([0] + [len(ch.encode("utf-8")) for ch in text])  # char -> byte (of the window's own text)

        def byte(c):
            return cb[min(max(c - shift, 0), len(text))]
        S = np.array([byte(x[0]) for x in tt])
        E = np.array([byte(x[1]) for x in tt])
        bb = np.frombuffer(b, np.uint8)
        nonws = ~np.isin(bb, list(b" \t\n\r\x0b\x0c"))
        nxt = np.full(len(nonws) + 1, len(nonws))
        pn = np.nonzero(nonws)[0]
        nxt[pn] = pn
        nxt = np.minimum.accumulate(nxt[::-1])[::-1]
        f = np.minimum(nxt[ob[:-1]], ob[1:] - 1)
        allws = nxt[ob[:-1]] >= ob[1:]
        sid = np.searchsorted(S, f, "right") - 1
        ok = (sid >= 0) & ~allws
        ok &= f < E[np.clip(sid, 0, None)]
        nl = b.find(b"\n")
        ok &= ob[:-1] > nl if nl >= 0 else False
        j = np.nonzero(ok)[0]
        s_ = sid[j]
        out["roles"][r, j] = [tt[k][2] for k in s_]
        out["ctx"][r, j] = [ci[tt[k][3]] for k in s_]
        out["kind"][r, j] = [ki[tt[k][4]] for k in s_]
        ls = np.array([byte(tt[k][5]) for k in s_])
        out["start"][r, j] = np.searchsorted(ob, ls, "right") - 1
        out["ok"][r, j] = True
        if (r + 1) % 250 == 0:
            print(f"  lexed {r + 1}/{len(ws)} windows ({time.time() - t0:.0f} s)", flush=True)
    return out, py


def sample_windows(split):
    from bonsai_interp.auto import context_tier as ct
    dom = ct._domains()
    lo, hi = SPLITS[split]
    hi = len(dom) if hi is None else hi
    return np.nonzero(dom[lo:hi] == DOMAIN)[0] + lo  # every code window of the split


def save(path, ws, out, py):
    import pygments
    np.savez_compressed(path, windows=np.asarray(ws, np.int32), python=py, **out, roles_names=np.array(ROLES),
                        ctx_names=np.array(CTX), kind_names=np.array(KINDS), pygments=np.array(f"pygments {pygments.__version__}"))


def load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["select", "test"])
    ap.add_argument("--out")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.what == "test" and not a.out:
        from bonsai_interp.codesyn import run5
        run5._install()
        from bonsai_interp.grammar import run as GR
        if not GR.registered():
            sys.exit("the test split is lexed only after the protocol manifest is committed on its own (run5 protocol)")
    ws = sample_windows(a.what)
    path = Path(a.out) if a.out else OUT / f"bonsai1.lex.{a.what}.npz"
    t0 = time.time()
    out, py = lex(ws)
    save(path, ws, out, py)
    print(f"wrote {path}: {len(ws)} code windows, {int(py.sum())} Python, in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
