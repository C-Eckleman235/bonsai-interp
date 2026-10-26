"""sweeplib@2's twin batteries (CPU only, deterministic): one-swap twins seeded by the word screen's hits.

The word screen (auto-token@2) found, for each pole, the words its strong activations fall on. A word hit says "this pole
fires on ' word'"; the Twitch loop's commonest next step is a one-swap twin that asks which part of that is the trigger:
the case, the leading space, the whole token, the sentence position, the slot in code, the word form. A twin battery
asks those questions of every pole that shares the seed word at once, so one read serves them all.

For a seed token such as ' word' (a whole-word token: a leading space and letters only), snippets of 48 corpus tokens are
read at their last token. The sets (n snippets each; selection from windows 0-15,999, tests from 16,400 on):

  base       the seed mid-sentence (after a word or a comma; from prose when there is enough)
  base2      a second, disjoint sample of the same pool (the context placebo)
  case       base with only the last token changed to another case of the word (' Word', ' WORD', ' word'; rotated)
  case_ctl   base with the last token changed to the control word in that same case
  ctl, ctl2  base with the last token changed to a control word: a same-shape word of similar frequency (auto-token@2's
             swap); ctl against ctl2 is the swap placebo
  infl       base with the last token changed to an inflection of the word (-s, -ed, -ing, or the base form; rotated)
  infl_ctl   base with the last token changed to a same-shape word of similar frequency, with the same ending when one exists
  space      the word's letters without the leading space, at a word start (after a bracket, quote, slash or line break)
  space_ctl  space with the last token changed to the control word's letters
  piece      the same letters as a split piece inside a longer word (the previous token ends in a letter)
  piece_ctl  piece with the last token changed to the control word's letters
  longer     a longer whole-word token that contains the letters (' word' -> ' password', ' words' is an inflection)
  initial    the seed at the start of a sentence (after . ! or ?), prose
  comment, string, code   the seed in code (corpus domain 3): in a comment or docstring, inside a string literal, elsewhere
  prose      the seed mid-sentence in English prose (corpus domains 0 and 1)

A claim is a fire set against one or more quiet sets (strata, round-robin), so each claim varies ONE axis:
  invariant claims ('inv'): the other form fires too, against a control word in the same slot and the same form;
  selective claims ('sel'): the seed's own form or context fires, the twin does not.
Each claim's placebo pairs set two snippets against each other where the axis does not change: base against base2 for
a selective claim, ctl against ctl2 for an invariant one.
"""
from functools import lru_cache

import numpy as np

from bonsai_interp.auto import context_tier as ct
from bonsai_interp.sweeplib import kinds as K

SNIP, MIN_POS, SPLITS = K.SNIP, K.MIN_POS, K.SPLITS
N_SELECT, MIN_SELECT, N_TEST = 16, 12, 8
CODE_CAP = 1500        # code-domain occurrences per seed whose slot is classified (a Python loop), sampled first
LONGER_MAX = 8         # longer tokens per seed
MIN_FREQ = 50          # an inflection or a longer token must occur this often in the selection windows (0-7,999)

# claim -> (axis, inv | sel, fire set, quiet strata, placebo)
CLAIMS = {
    "form/any-case": ("form", "inv", "case", ("case_ctl",), "swap"),
    "form/this-case": ("form", "sel", "base", ("case",), "ctx"),
    "form/no-space": ("form", "inv", "space", ("space_ctl",), "swap"),
    "form/space-only": ("form", "sel", "base", ("space",), "ctx"),
    "piece/inside": ("piece", "inv", "piece", ("piece_ctl",), "swap"),
    "piece/own": ("piece", "sel", "base", ("piece", "longer"), "ctx"),
    "pos/initial": ("pos", "sel", "initial", ("base",), "ctx"),
    "pos/mid": ("pos", "sel", "base", ("initial",), "ctx"),
    "slot/comment": ("slot", "sel", "comment", ("string", "code", "prose"), "ctx"),
    "slot/string": ("slot", "sel", "string", ("comment", "code", "prose"), "ctx"),
    "slot/code": ("slot", "sel", "code", ("comment", "string", "prose"), "ctx"),
    "slot/prose": ("slot", "sel", "prose", ("comment", "string", "code"), "ctx"),
    "infl/family": ("infl", "inv", "infl", ("infl_ctl",), "swap"),
    "infl/this-form": ("infl", "sel", "base", ("infl",), "ctx"),
}
PLACEBO = {"ctx": ("base", "base2"), "swap": ("ctl", "ctl2")}
SPLIT_CODE = {"select": 0, "test": 1}


def label(claim, w):
    """The claim a held test makes, in the survey's shapes."""
    s, c = w["text"], w["core"]
    q = lambda x: "'" + x + "'"
    forms = ", ".join(q(x) for _, x in w["case"])
    infl = ", ".join(q(x) for _, x, _ in w["infl"])
    return {
        "form/any-case": f"the word {q(c)} in any case: {q(s)} and {forms} all fire, not other words in the same slot",
        "form/this-case": f"{q(s)} only in this case: not {forms} in the same place",
        "form/no-space": f"the word {q(c)} with or without its leading space ({q(c)} at a word start after a bracket, quote or line break "
                         f"fires too), not other words there",
        "form/space-only": f"{q(s)} only with its leading space: not {q(c)} at a word start after a bracket, quote or line break",
        "piece/inside": f"the letters {q(c)}, also as a split piece inside a longer word, not other pieces there",
        "piece/own": f"{q(s)} only where it is its own token: not the same letters as a piece of a longer word or inside a longer token",
        "pos/initial": f"{q(s)} only at the start of a sentence, not mid-sentence",
        "pos/mid": f"{q(s)} only mid-sentence, not at the start of a sentence",
        "slot/comment": f"{q(s)} only in code comments (and docstrings), not in strings, code or prose",
        "slot/string": f"{q(s)} only inside string literals in code, not in comments, code or prose",
        "slot/code": f"{q(s)} only in code itself (identifiers, keywords), not in comments, strings or prose",
        "slot/prose": f"{q(s)} only in prose, not in code (comments, strings or code itself)",
        "infl/family": f"the word {q(c)} and its inflections ({infl}), not other words in the same slot",
        "infl/this-form": f"{q(s)} only in this form, not its inflections ({infl}) in the same place",
    }[claim]


# ---------------------------------------------------------------- vocabulary helpers

@lru_cache(None)
def _tables():
    vb = ct.vocab()
    text = vb["text"]
    last = [t[-1:] for t in text]
    return {"t2i": {t: i for i, t in enumerate(text)},
            "alpha_last": np.array([x.isalpha() for x in last]),
            "space_last": np.array([x in (" ", "\t") or x == "" for x in last]),
            "us_last": np.array([x == "_" for x in last]),
            "comma": np.array([t.strip() in (",", ";") for t in text]),
            "freq": np.load(ct.OUT / "bonsai1_L0.hits.npz")["freq"].astype(np.int64)}


@lru_cache(None)
def _subs():
    tb = _tables()
    return ct.Substitutes(tb["freq"], ct.vocab())


@lru_cache(None)
def _words_by_freq():
    """[(token, lower-case letters)] of whole-word tokens seen >= MIN_FREQ times, commonest first."""
    vb, f = ct.vocab(), _tables()["freq"]
    return [(int(i), vb["text"][int(i)][1:].lower()) for i in np.argsort(-f, kind="stable")
            if f[i] >= MIN_FREQ and is_seed_token(int(i), vb)]


def is_seed_token(t, vb=None):
    """A whole-word seed: a leading space and 2 or more letters, nothing else."""
    s = (vb or ct.vocab())["text"][t]
    return s[:1] == " " and s[1:].isalpha() and len(s) >= 3


def _case(c, kind):
    return {"lower": c.lower(), "cap": c[:1].upper() + c[1:].lower(), "upper": c.upper()}[kind]


def _inflect(b):
    """{-s, -ed, -ing} forms of a lower-case base."""
    cy = b.endswith("y") and len(b) > 2 and b[-2] not in "aeiou"
    return {"s": b[:-1] + "ies" if cy else b + "es" if b.endswith(("s", "x", "z", "ch", "sh")) else b + "s",
            "ed": b + "d" if b.endswith("e") else b[:-1] + "ied" if cy else b + "ed",
            "ing": b[:-1] + "ing" if b.endswith("e") and not b.endswith("ee") else b + "ing"}


def _bases(c):
    out = {c}
    for suf, add in (("ies", "y"), ("es", ""), ("s", ""), ("ied", "y"), ("ed", ""), ("ed", "e"), ("d", ""), ("ing", ""), ("ing", "e")):
        if c.endswith(suf) and len(c) - len(suf) + len(add) >= 3:
            out.add(c[:-len(suf)] + add)
    return out


def word_info(t):
    """The seed's forms: case variants, its space-less token, inflections and longer tokens (token ids, all deterministic)."""
    vb, tb = ct.vocab(), _tables()
    s = vb["text"][t]
    c = s[1:]
    case = [(k, " " + _case(c, k)) for k in ("lower", "cap", "upper") if _case(c, k) != c and len(c) >= 2]
    seen = set()
    case = [x for x in case if not (x[1] in seen or seen.add(x[1]))]
    infl = []
    if len(c) >= 3 and c.lower() not in ct.CLOSED:  # function words have no inflections (the + ing is not 'thing')
        lo = c.lower()
        fam = {}
        for b in sorted(_bases(lo)):
            fam.setdefault(b, "base")
            for k, f in _inflect(b).items():
                fam.setdefault(f, k)
        for f, k in sorted(fam.items()):
            if f == lo:
                continue
            f2 = _case(f, "cap") if c[:1].isupper() and not c.isupper() else f.upper() if c.isupper() and len(c) > 1 else f
            i = tb["t2i"].get(" " + f2)
            if i is not None and tb["freq"][i] >= MIN_FREQ:
                infl.append((k, " " + f2, int(i)))
    core_id = tb["t2i"].get(c)
    longer = []
    if len(c) >= 3:
        lo, fams = c.lower(), {x[1].strip().lower() for x in infl}
        for i, x in _words_by_freq():
            if len(x) >= len(c) + 2 and lo in x and x not in fams:
                longer.append(i)
                if len(longer) >= LONGER_MAX:
                    break
    return {"tok": int(t), "text": s, "core": c, "case": case, "infl": infl, "core_id": None if core_id is None else int(core_id),
            "longer": longer}


# ---------------------------------------------------------------- occurrences and slots

def _slot(ids, vb):
    """Where the last token of a code snippet sits, judged from its own line: 'comment' (after #, // or <!-- on the line),
    'string' (inside a string literal opened on the line) or 'code'; None when the snippet cannot tell (a docstring or block
    comment delimiter anywhere in it, or a quote that may close a string opened on an earlier line). ids: the tokens before it."""
    head = "".join(vb["text"][int(i)] for i in ids)
    if '"""' in head or "'''" in head or "/*" in head or "*/" in head:
        return None
    line = head.rsplit("\n", 1)[-1]
    q, i, before = None, 0, ""
    while i < len(line):
        ch = line[i]
        if q:
            if ch == "\\":
                i += 2
                continue
            if ch == q:
                q = None
        elif line.startswith("//", i) or line.startswith("<!--", i) or ch == "#" and line[i + 1:i + 2] in ("", " ", "#"):
            return "comment"
        elif ch in "\"'`":
            q, before = ch, line[:i].rstrip(" \t")[-1:]
        i += 1
    if q:
        return "string" if before in ("", "=", "(", ",", "[", "{", ":", "+", "%") or line[:line.rfind(q)].strip() == "" else None
    # code: the line shows code punctuation, or the token starts a statement after a line that does (else it may be prose
    # inside a docstring opened before the snippet)
    prev = head.rsplit("\n", 2)[-2] if head.count("\n") >= 1 else ""
    if any(c in line for c in "=(){}[];") or line.strip() == "" and any(c in prev for c in "=(){}[]:;"):
        return "code"
    return None


@lru_cache(None)
def _occ(split, tokens):
    """{token: (windows, positions)}: every occurrence of each token in the split, at positions where a 48-token snippet
    starts at MIN_POS or later."""
    from bonsai_interp import explore
    toks = explore.corpus()
    lo, hi = SPLITS[split]
    hi = len(toks) if hi is None else hi
    T = toks.shape[1]
    flat = toks[lo:hi].ravel()
    want = np.zeros(int(ct.vocab()["V"]) + 1, bool)
    want[list(tokens)] = True
    idx = np.flatnonzero(want[flat])
    idx = idx[(idx % T) >= MIN_POS + SNIP - 1]
    o = np.argsort(flat[idx], kind="stable")
    idx = idx[o]
    st = np.searchsorted(flat[idx], np.arange(len(want) + 1))
    return {t: (idx[st[t]:st[t + 1]] // T + lo, idx[st[t]:st[t + 1]] % T) for t in tokens}


def _pick(w, p, n, rng, avoid=()):
    """Up to n occurrences from distinct windows (seeded), none in `avoid`."""
    o = rng.permutation(len(w))
    out, used = [], set(avoid)
    for i in o:
        if int(w[i]) not in used:
            used.add(int(w[i]))
            out.append((int(w[i]), int(p[i])))
            if len(out) >= n:
                break
    return out


def _ctl_like(t, suffix, rng, avoid):
    """A same-shape word of similar frequency to token t (auto-token@2's swap), ending in `suffix` when one does."""
    sb = _subs()
    sh = sb.shape(t)
    if sh not in sb.ids:
        return None
    ids, lf = sb.ids[sh], sb.lf[sh]
    c = int(np.searchsorted(lf, np.log(sb.freq[t] + 1.0)))
    near = [int(i) for i in ids[max(0, c - 60): c + 60]
            if (x := sb.vb["text"][int(i)].strip().lower())[:4] not in avoid and not any(a in x for a in avoid)]
    pref = [i for i in near if suffix and sb.vb["text"][i].strip().lower().endswith(suffix)]
    pool = pref or near
    return pool[rng.integers(len(pool))] if pool else None


# ---------------------------------------------------------------- a word's battery

def battery(words, split, n):
    """{token: {"info": word_info, "sets": {set: [(window, position, replacement text or None)]}, "counts": {...}}}.
    counts: the natural share of each context among the seed's occurrences in the split (for coverage estimates)."""
    vb, tb = ct.vocab(), _tables()
    from bonsai_interp import explore
    toks, dom = explore.corpus(), ct._domains()
    infos = {t: word_info(t) for t in words}
    need = set(words)
    for w in infos.values():
        need |= set(w["longer"]) | ({w["core_id"]} if w["core_id"] is not None else set())
    occ = _occ(split, tuple(sorted(need)))
    sc = SPLIT_CODE[split]
    out = {}
    for t in words:
        info, sets, counts = infos[t], {}, {}
        rng = lambda k: np.random.default_rng([int(t), sc, k])
        w, p = occ[t]
        prev = toks[w, p - 1].astype(np.int64)
        d = dom[w]
        nl = vb["nl"][prev]
        mid = (vb["an1"][prev] | tb["comma"][prev]) & ~nl
        initial = vb["send"][prev] & ~nl & (d <= 2)
        counts["occurrences"] = int(len(w))
        counts["initial"], counts["mid"] = int(initial.sum()), int(mid.sum())
        pro = mid & (d <= 2)
        pool = pro if pro.sum() >= 3 * n else mid
        b = _pick(w[pool], p[pool], 2 * n, rng(1))
        sets["base"] = [(x, y, None) for x, y in b[:n]]
        sets["base2"] = [(x, y, None) for x, y in b[n:2 * n]]
        sets["initial"] = [(x, y, None) for x, y in _pick(w[initial], p[initial], n, rng(2))]
        pr = mid & (d <= 1)
        sets["prose"] = [(x, y, None) for x, y in _pick(w[pr], p[pr], n, rng(3))]
        # code slots: classify a sample of the code occurrences
        cd = np.flatnonzero(d == 3)
        if len(cd) > CODE_CAP:
            cd = np.sort(rng(4).choice(cd, CODE_CAP, replace=False))
        slots = {"comment": [], "string": [], "code": [], None: []}
        for i in cd:
            slots[_slot(toks[w[i], p[i] - SNIP + 1:p[i]], vb)].append(i)
        counts["code_unclear"] = len(slots.pop(None))
        counts["code_sampled"] = int(len(cd))
        counts["code_total"] = int((d == 3).sum())
        counts["prose_total"] = int((d <= 1).sum())
        for k, ix in slots.items():
            ix = np.array(ix, dtype=np.int64)
            counts[k] = int(len(ix))
            sets[k] = [(x, y, None) for x, y in _pick(w[ix], p[ix], n, rng(5 + len(k)))] if len(ix) else []
        # controls: per base snippet, two different same-shape words
        r = rng(9)
        avoid = {info["core"].lower()[:4]}
        u1, u2 = [], []
        for _ in sets["base"]:
            a = _subs()(t, r)
            bb = _subs()(t, r)
            for _ in range(5):
                if bb != a:
                    break
                bb = _subs()(t, r)
            u1.append(a)
            u2.append(bb if bb != a else None)
        ok = [i for i in range(len(sets["base"])) if u1[i] is not None and u2[i] is not None]
        base = [sets["base"][i] for i in ok]
        u1, u2 = [u1[i] for i in ok], [u2[i] for i in ok]
        sets["base"], sets["base2"] = base, sets["base2"][:len(base)]
        cores = [vb["text"][u].strip() for u in u1]
        sets["ctl"] = [(x, y, vb["text"][u]) for (x, y, _), u in zip(base, u1)]
        sets["ctl2"] = [(x, y, vb["text"][u]) for (x, y, _), u in zip(base, u2)]
        if info["case"]:
            ks = [info["case"][i % len(info["case"])] for i in range(len(base))]
            sets["case"] = [(x, y, v) for (x, y, _), (_, v) in zip(base, ks)]
            sets["case_ctl"] = [(x, y, " " + _case(u, k)) for (x, y, _), (k, _), u in zip(base, ks, cores)]
        if info["infl"]:
            fs = [info["infl"][i % len(info["infl"])] for i in range(len(base))]
            sets["infl"] = [(x, y, f) for (x, y, _), (_, f, _) in zip(base, fs)]
            ri = rng(10)
            ctl = [_ctl_like(fi, {"s": "s", "ed": "ed", "ing": "ing"}.get(k, ""), ri, avoid) for k, _, fi in fs]
            keep = [i for i, c in enumerate(ctl) if c is not None]
            sets["infl"] = [sets["infl"][i] for i in keep]
            sets["infl_ctl"] = [(base[i][0], base[i][1], vb["text"][ctl[i]]) for i in keep]
        ci = info["core_id"]
        if ci is not None:
            cw, cp = occ[ci]
            cprev = toks[cw, cp - 1].astype(np.int64)
            start = ~vb["an1"][cprev] & ~tb["space_last"][cprev] & ~tb["us_last"][cprev]
            # a split piece: the previous token ends in a letter (lower-case words only: a capitalised one there is camel case)
            piece = tb["alpha_last"][cprev] & info["core"].islower()
            counts["space"], counts["piece"] = int(start.sum()), int(piece.sum())
            sp = _pick(cw[start], cp[start], n, rng(11))
            pc = _pick(cw[piece], cp[piece], n, rng(12))
            low = info["core"].islower()
            sets["space"] = [(x, y, None) for x, y in sp]
            sets["space_ctl"] = [(x, y, cores[i % len(cores)]) for i, (x, y) in enumerate(sp)] if cores else []
            sets["piece"] = [(x, y, None) for x, y in pc]
            sets["piece_ctl"] = [(x, y, cores[i % len(cores)].lower() if low else cores[i % len(cores)]) for i, (x, y) in enumerate(pc)] \
                if cores else []
        if info["longer"]:
            lw = np.concatenate([occ[x][0] for x in info["longer"]])
            lp = np.concatenate([occ[x][1] for x in info["longer"]])
            counts["longer"] = int(len(lw))
            sets["longer"] = [(x, y, None) for x, y in _pick(lw, lp, n, rng(13))]
        out[t] = {"info": info, "sets": {k: v for k, v in sets.items() if v}, "counts": counts}
    return out


def pairs(sets, claim):
    """[(fire item index, quiet set, quiet item index, stratum)] for a claim: fire item i against the quiet strata in turn.
    Twins built from base (case, infl, the controls) are paired with their own base snippet."""
    _, _, f, qs, _ = CLAIMS[claim]
    F = sets.get(f, [])
    avail = [q for q in qs if sets.get(q)]
    if not F or not avail:
        return []
    nxt, out = {q: 0 for q in avail}, []
    for i in range(len(F)):
        for j in range(len(avail)):
            q = avail[(i + j) % len(avail)]
            if nxt[q] < len(sets[q]):
                out.append((i, q, nxt[q], q))
                nxt[q] += 1
                break
    return out


def placebo_pairs(sets, kind):
    a, b = PLACEBO[kind]
    m = min(len(sets.get(a, [])), len(sets.get(b, [])))
    return [(i, b, i, "placebo") for i in range(m)], a


def text(item):
    """The snippet text: the corpus snippet ending at (window, position), its last token replaced when the item says so."""
    from bonsai_interp import explore
    w, p, rep = item
    ids = explore.corpus()[w, p - SNIP + 1:p + 1]
    return K._dec(ids) if rep is None else K._dec(ids[:-1]) + rep
