"""sweeplib@5: code syntax roles (roles.py on the Pygments lex, lex.py), run on the grammar pass's engine (grammar/run.py).

  python -m bonsai_interp.codesyn.lex select         # CPU: lex the selection split's code windows
  python -m bonsai_interp.codesyn.run5 protocol      # write the manifest: commit it ALONE
  python -m bonsai_interp.codesyn.run5 select        # GPU
  python -m bonsai_interp.codesyn.lex test           # CPU: only after the protocol is committed
  python -m bonsai_interp.codesyn.run5 build         # CPU: commit the prereg manifest ALONE
  python -m bonsai_interp.codesyn.run5 score         # GPU
  python -m bonsai_interp.codesyn.run5 report        # CPU: coverage over the scan's top examples in Python windows

The reader, pair sampler, selection rule (sweeplib@3's), placebo pairs and tests, and judge (quick-8+8@2) are unchanged. Only
the parse (a Pygments lex instead of spaCy), the conditions and the coverage's windows (code, not English) differ.
"""
import json
import sys
from pathlib import Path

import numpy as np

from bonsai_interp.codesyn import lex as X
from bonsai_interp.codesyn import roles as RO
from bonsai_interp.grammar import cons as C
from bonsai_interp.grammar import parse as PA
from bonsai_interp.grammar import posmorph as PM
from bonsai_interp.grammar import run as GR

PROTOCOL, KIND, BASE = "sweeplib@5", "code", "bonsai1.sweeplib5"
OUT = GR.ROOT / "data" / "codesyn"


def coverage(held, procs=6):
    """grammar/run.py's coverage for code: the share of each held pole's 32 strongest corpus activations, in Python windows,
    where its role holds; the base rate is the role's share of the lexed selection split's tokens."""
    from bonsai_interp import explore
    from bonsai_interp.auto import context_tier as ct
    s = explore.scan("bonsai1")
    dom = ct._domains()
    need = {}
    for nid, c in held:
        L, p, n = GR._neuron(nid)
        need[(nid, c)] = [(int(w), int(q)) for w, q in zip(s["top_win"][L, p, :, n], s["top_pos"][L, p, :, n])
                          if dom[int(w)] == X.DOMAIN]
    ws = sorted({w for ex in need.values() for w, _ in ex})
    if not ws:
        return {}, {}
    out, py = X.lex(np.array(ws))
    z = {**out, "windows": np.array(ws), "ctx_names": np.array(X.CTX)}
    conds = sorted({c for _, c in held})
    hm = RO.holds(z, conds)
    row = {w: i for i, w in enumerate(ws)}
    zs = PA.load(GR._paths()["parse_select"])
    words = RO.holds(zs, conds)
    den = zs["ok"].sum()
    base = {c: round(float(words[c].sum() / den), 4) for c in conds}
    cov = {}
    for (nid, c), ex in need.items():
        ex = [(w, q) for w, q in ex if py[row[w]]]
        hit = [bool(hm[c][row[w], q]) for w, q in ex]
        cov[(nid, c)] = {"n_en": len(ex), "holds": int(sum(hit)), "share": round(sum(hit) / len(ex), 3) if ex else None,
                         "base": base[c], "lift": round(sum(hit) / len(ex) / base[c], 1) if ex and base[c] else None}
    return cov, base


def _install():
    paths3 = GR._paths

    def paths(d=None):
        P = paths3(d)
        for k in ("protocol", "prereg", "summary", "select", "tests", "results", "coverage", "report"):
            P[k] = P[k].with_name(P[k].name.replace("bonsai1.sweeplib3", BASE))
            if k not in ("protocol", "prereg", "summary") and not d:
                P[k] = OUT / P[k].name
        P["parse_select"] = OUT / "bonsai1.lex.select.npz"
        P["parse_test"] = (Path(d) if d else OUT) / "bonsai1.lex.test.npz"
        return P
    GR._paths, GR.PROTOCOL, GR.KIND, GR.OUT = paths, PROTOCOL, KIND, OUT
    GR.CODE = list(GR.CODE) + [Path(X.__file__), Path(RO.__file__), Path(PM.__file__), Path(__file__)]
    C.labels, C.battery, C.holds = RO.labels, RO.battery, RO.holds
    C.CONDITIONS = {c: (lab, "code") for c, lab in RO.labels().items()}
    C.PER_KEY, C.CAP, C.FEW = RO.PER_KEY, RO.CAP, ()
    GR.coverage = coverage
    PA.MODEL, PA.SAMPLE = "pygments PythonLexer with bracket and statement state (codesyn/lex.py)", "every code window"


def main():
    _install()
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    smoke = sys.argv[sys.argv.index("--smoke") + 1] if "--smoke" in sys.argv else None
    if cmd in ("select", "build") and not GR._paths(smoke)["parse_" + ("select" if cmd == "select" else "test")].exists():
        sys.exit(f"lex the split first: python -m bonsai_interp.codesyn.lex {'select' if cmd == 'select' else 'test'}")
    GR.main()
    if cmd == "protocol":
        p = GR._paths()["protocol"]
        man = json.loads(p.read_text(encoding="utf-8"))
        man["about"] = RO.__doc__
        man["parser"]["domains"] = "code, windows judged Python (lex.is_python)"
        man["last_token"] = "any non-whitespace token of a Python window after its first line break, at position >= 111, whose " \
                            f"logical line starts at most {C.SNIP - 2} tokens back"
        for c, v in man["conditions"].items():
            v["key"] = "token" if c.startswith("role:") else "coarse token type x corpus-frequency band"
            v.pop("cue", None)
        p.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
