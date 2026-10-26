"""Generic grammatical constructions as minimal-pair batteries (CPU only, deterministic; no spaCy at import).

A condition is a predicate on the spaCy parse (parse.py) at a snippet's last token, which must end its word. Its battery
pairs a snippet where the condition holds ("fire") with one from another window that ends on the same token where it does
not ("quiet": the same word in a different grammatical role). Every condition also has a word-class variant, "pos:<name>",
whose pair key is the last word's part of speech instead of the token: a pole selected there follows the role whatever the
word. Pairs, placebo pairs, splits and snippet length are the sweep library's (sweeplib/kinds.py), reused by import.
"""
from functools import lru_cache

import numpy as np

from bonsai_interp.auto import context_tier as ct
from bonsai_interp.sweeplib import kinds as SK

SNIP, MIN_POS = SK.SNIP, SK.MIN_POS
N_SELECT, MIN_SELECT, N_TEST = SK.N_SELECT, SK.MIN_SELECT, SK.N_TEST
PER_KEY = {"tok": {"select": 8, "test": 1}, "few": {"select": 32, "test": 4}, "pos": {"select": 64, "test": 8}}  # pairs per key
FEW = ("poss_s", "relcl_start", "wh_question")  # token-keyed conditions with only a handful of possible last words
CAP = {"tok": 12, "few": 200, "pos": 600}  # candidate snippets kept per key and stratum (seeded), so batteries stay small in memory
WORD_POS = ("NOUN", "PROPN", "PRON", "VERB", "AUX", "ADJ", "ADV", "ADP", "DET", "PART", "CCONJ", "SCONJ", "INTJ")
INTENS = {"very", "so", "too", "really", "extremely", "quite", "rather", "pretty", "incredibly", "highly", "fairly", "somewhat",
          "especially", "particularly", "absolutely", "totally", "completely", "remarkably", "terribly", "awfully", "exceptionally",
          "unusually", "truly", "super"}
NEG = {"not", "n't", "never", "no", "nt"}
DEPGRP = {**{d: "subject" for d in ("nsubj", "nsubjpass", "csubj", "csubjpass", "expl")},
          **{d: "object" for d in ("dobj", "iobj", "dative", "attr", "oprd")},
          "pobj": "prep-object",
          **{d: "modifier" for d in ("amod", "advmod", "nmod", "compound", "npadvmod", "nummod", "poss", "det", "predet", "quantmod",
                                     "appos", "acomp", "neg", "prt", "case")},
          **{d: "clause" for d in ("ROOT", "ccomp", "xcomp", "advcl", "relcl", "acl", "conj", "pcomp", "parataxis", "csubj")},
          **{d: "function" for d in ("prep", "aux", "auxpass", "mark", "cc", "agent", "preconj")}}

# condition -> (label, where the cue is: "left" = in the words before the last token, "right" = mostly after it, so a pole
# selected there anticipates the construction)
CONDITIONS = {
    "pobj": ("the object of a preposition ('in the house' at 'house')", "left"),
    "prep": ("a preposition linking two content words ('the edge of the table' at 'of')", "right"),
    "verb_dobj": ("a verb that takes a direct object ('she wrote a letter' at 'wrote')", "right"),
    "nsubj": ("the subject of a clause", "right"),
    "dobj": ("the direct object of a verb ('she wrote a letter' at 'letter')", "left"),
    "intensified": ("an adjective or adverb modified by an intensifier before it ('very old' at 'old')", "left"),
    "amod": ("an adjective modifying a noun ('an old book' at 'old')", "right"),
    "amod_head": ("a noun modified by an adjective before it ('an old book' at 'book')", "left"),
    "advmod_verb": ("an adverb modifying a verb, not an adjective or adverb ('ran quickly', not 'quickly forgotten fact')", "right"),
    "det_noun": ("a noun headed by a determiner ('the book' at 'book')", "left"),
    "poss": ("a possessor ('her book', 'John's book' at 'her', 'John')", "right"),
    "poss_s": ("the possessive 's ('John's book' at ''s'), not the contraction of is or has", "right"),
    "conj2": ("the second conjunct of 'X and Y' or 'X or Y' (at 'Y')", "left"),
    "aux_verb": ("a verb with an auxiliary before it ('has eaten', 'will go' at the verb)", "left"),
    "negated": ("a verb negated by a 'not' or 'never' before it ('did not go' at 'go')", "left"),
    "than": ("the compared term after 'than' ('bigger than a house' at 'house')", "left"),
    "relcl_start": ("a relative pronoun opening a relative clause ('the book which' at 'which')", "left"),
    "particle": ("the particle of a phrasal verb ('give up' at 'up')", "right"),
    "passive": ("a passive participle ('was taken' at 'taken'), not a perfect ('has taken')", "left"),
    "wh_question": ("a question word in a direct question ('Where did' in a sentence ending with '?')", "right"),
}
POS_VARIANT = ("pobj", "nsubj", "dobj", "poss", "amod_head", "det_noun", "conj2", "aux_verb", "negated", "verb_dobj", "intensified",
               "amod", "advmod_verb", "than", "passive")


def labels():
    out = {}
    for c, (lab, _) in CONDITIONS.items():
        out[c] = f"grammar: fires on {lab}, not on the same word in another grammatical role"
    for c in POS_VARIANT:
        out["pos:" + c] = f"grammar (any word): fires on {CONDITIONS[c][0]}, whatever the word, not on words of the same part of " \
                          "speech in other roles"
    return out


@lru_cache(None)
def _words():
    vb = ct.vocab()
    return np.array([t.strip().lower().replace("’", "'") for t in vb["text"]], dtype=object)


def masks(z, ids):
    """{condition: (fire (W, T) bool, quiet (W, T) object array of stratum names or None)} for a parse z (parse.load) and the
    windows' token ids (W, T). No position cutoff here (the battery applies it)."""
    W, T = ids.shape
    nm = {k: z[f"{k}_names"] for k in ("pos", "tag", "dep")}
    pos, tag, dep = (nm[k][z[k]] for k in ("pos", "tag", "dep"))  # (W, T) arrays of names
    H, last = z["head"].astype(np.int64), z["last"]
    word = last & np.isin(pos, WORD_POS)
    wd = _words()[ids]
    ar = np.arange(T)[None, :]
    hasH = H >= 0
    Hc = np.clip(H, 0, T - 1)
    rows = np.arange(W)[:, None]
    hpos, hdep, hwd = pos[rows, Hc], dep[rows, Hc], wd[rows, Hc]

    def child(M, left=False):
        """(W, T): the token has a child (a last piece) where M holds (left: the child comes before it)."""
        sel = M & last & hasH & (H != ar)
        if left:
            sel &= ar < H
        out = np.zeros((W, T), bool)
        w, j = np.nonzero(sel)
        out[w, H[w, j]] = True
        return out

    grp = np.vectorize(lambda d: DEPGRP.get(d, "other"), otypes=[object])(dep)
    noun = np.isin(pos, ("NOUN", "PROPN"))
    verb = pos == "VERB"
    content = np.isin(pos, ("NOUN", "PROPN", "VERB", "ADJ", "AUX"))
    has_pobj = child(dep == "pobj")
    has_dobj = child(dep == "dobj")
    has_obj = child(np.isin(dep, ("dobj", "ccomp", "xcomp", "dative", "iobj", "oprd", "attr")))
    has_amod = child(dep == "amod", left=True)
    has_nmod = child(np.isin(dep, ("amod", "compound", "nmod", "nummod")))
    has_det = child(dep == "det", left=True)
    has_poss = child(dep == "poss")
    has_aux = child(np.isin(dep, ("aux", "auxpass")), left=True)
    has_auxpass = child(dep == "auxpass") | child(dep == "nsubjpass")
    has_have = child((dep == "aux") & np.isin(wd, ("have", "has", "had", "'ve", "'d", "having")), left=True)
    has_neg = child((dep == "neg") | np.isin(wd, tuple(NEG)) & (dep == "advmod"), left=True)
    any_neg = child(dep == "neg")
    has_int = child((dep == "advmod") & np.isin(wd, tuple(INTENS)), left=True)
    has_adv = child(dep == "advmod")
    # the head has a cc child 'and' / 'or' between the head and this conjunct
    cc_and = (dep == "cc") & np.isin(wd, ("and", "or"))
    cc_pos = np.full((W, T), -1)
    w_, j_ = np.nonzero(cc_and & hasH)
    cc_pos[w_, H[w_, j_]] = j_
    hcc = cc_pos[rows, Hc]
    wh = np.isin(tag, ("WDT", "WP", "WP$", "WRB"))
    out = {}

    def put(c, fire, quiet, strata=None):
        q = np.full((W, T), None, dtype=object)
        qm = quiet & ~fire & word
        q[qm] = grp[qm] if strata is None else strata[qm] if isinstance(strata, np.ndarray) else strata
        out[c] = (fire & word, q)

    put("pobj", dep == "pobj", np.ones((W, T), bool))
    put("prep", (dep == "prep") & (pos == "ADP") & hasH & np.isin(hpos, ("NOUN", "PROPN", "VERB", "ADJ", "AUX")) & has_pobj,
        dep != "prep")
    put("verb_dobj", verb & has_dobj, verb & ~has_obj, "no object")
    put("nsubj", np.isin(dep, ("nsubj", "nsubjpass")), np.ones((W, T), bool))
    put("dobj", dep == "dobj", np.ones((W, T), bool))
    adjadv = np.isin(pos, ("ADJ", "ADV"))
    put("intensified", adjadv & has_int, adjadv & ~has_adv, "no modifier")
    put("amod", (dep == "amod") & (pos == "ADJ"), (pos == "ADJ") & (dep != "amod"))
    put("amod_head", noun & has_amod, noun & ~has_nmod, "no modifier")
    advv = (pos == "ADV") & (dep == "advmod")
    put("advmod_verb", advv & hasH & (hpos == "VERB"), advv & hasH & np.isin(hpos, ("ADJ", "ADV")), "modifies adjective or adverb")
    put("det_noun", noun & has_det, noun & ~has_det & ~has_poss, "bare noun")
    put("poss", dep == "poss", (dep != "poss") & (dep != "case"))
    ps_ = np.isin(wd, ("'s", "'"))
    put("poss_s", ps_ & (dep == "case"), ps_ & (dep != "case"))
    put("conj2", (dep == "conj") & hasH & (H < ar) & (hcc > H) & (hcc < ar), dep != "conj")
    put("aux_verb", verb & has_aux, verb & ~has_aux, "no auxiliary")
    vaux = np.isin(pos, ("VERB", "AUX"))
    put("negated", vaux & has_neg, vaux & ~any_neg, "not negated")
    put("than", hasH & (hwd == "than") & (H < ar), ~(hasH & (hwd == "than")))
    H2 = H[rows, Hc]  # the head's head: for a relative pronoun, the noun the clause modifies
    rel = np.isin(wd, ("that", "which", "who", "whom", "whose")) & _lower()[ids]
    put("relcl_start", rel & wh & hasH & (hdep == "relcl") & (H > ar) & (H2 >= 0) & (H2 < ar), rel & ~(hasH & (hdep == "relcl")))
    put("particle", dep == "prt", dep != "prt")
    vbn = tag == "VBN"
    put("passive", vbn & has_auxpass, vbn & ~has_auxpass, np.where(has_have, "perfect", "other participle"))
    put("wh_question", wh & z["qsent"], wh & ~z["qsent"], "not a question")
    del content
    return out, pos


class E:
    """A light candidate snippet end (window, position); made into a sweeplib Snip only when it is paired."""
    __slots__ = ("w", "p")

    def __init__(self, w, p):
        self.w, self.p = w, p


def _placebo(Q, n, seed, rounds):
    """sweeplib's placebo pairs (two quiet snippets with the same key) with a set number of rounds over the keys."""
    rng = np.random.default_rng(seed)
    keys = sorted(Q, key=repr)
    rng.shuffle(keys)
    used, out = set(), []
    for _ in range(rounds):
        for k in keys:
            pool = [s for st in sorted(Q[k]) for s in Q[k][st] if s.w not in used]
            if len(pool) < 2 or len(out) >= n:
                continue
            a, b = (pool[i] for i in rng.choice(len(pool), 2, replace=False))
            if a.w != b.w:
                used |= {a.w, b.w}
                out.append((a, b, "placebo"))
    return out


def _cap(rng, keys, ws, ps, cap, strata=None):
    """{key: [E]} or {key: {stratum: [E]}}, at most cap per key (and stratum), in a seeded random order."""
    n = len(keys)
    if not n:
        return {}
    order = rng.permutation(n)
    k = np.asarray(keys)[order]
    g = np.unique(k, return_inverse=True)[1].astype(np.int64)
    if strata is not None:
        s = [str(x) for x in np.asarray(strata, dtype=object)[order]]
        g = g * 64 + np.unique(s, return_inverse=True)[1]
    o = np.argsort(g, kind="stable")
    gs = g[o]
    start = np.r_[0, np.nonzero(np.diff(gs))[0] + 1]
    rank = np.arange(n) - np.repeat(start, np.diff(np.r_[start, n]))
    keep = np.sort(o[rank < cap])
    out = {}
    w, p = np.asarray(ws)[order], np.asarray(ps)[order]
    for i in keep:
        kk = k[i].item() if hasattr(k[i], "item") else k[i]
        if strata is None:
            out.setdefault(kk, []).append(E(int(w[i]), int(p[i])))
        else:
            out.setdefault(kk, {}).setdefault(s[i], []).append(E(int(w[i]), int(p[i])))
    return out


@lru_cache(None)
def _lower():
    return np.array([t.strip().islower() for t in ct.vocab()["text"]])


@lru_cache(None)
def _whole():
    """Per token id: a whole alphabetic word (or a possessive 's / ' token), and: the token ends in whitespace."""
    vb = ct.vocab()
    alpha = np.array([t.strip().isalpha() or t in ("'s", "’s", "'") for t in vb["text"]])
    endws = np.array([t[-1:].isspace() for t in vb["text"]])
    return alpha, endws


def battery(z, split, conds=None):
    """{condition: [(fire Snip, quiet Snip, stratum)], "placebo:<condition>": [...]} from a parse z of the split's windows."""
    from bonsai_interp import explore
    toks, dom = explore.corpus(), ct._domains()
    win = z["windows"]
    ids = toks[win].astype(np.int64)
    M, pos = masks(z, ids)
    okpos = np.zeros(ids.shape, bool)
    okpos[:, MIN_POS + SNIP - 1:] = True
    # the last token is a whole word: alphabetic, begun by a space or after whitespace (it also ends its word: masks)
    alpha, endws = _whole()
    prev = np.zeros(ids.shape, bool)
    prev[:, 1:] = endws[ids[:, :-1]]
    okpos &= alpha[ids] & (ct.vocab()["lead"][ids] | prev | np.isin(_words()[ids], ("'s", "'")))
    out = {}
    for name in (conds or labels()):
        var, c = ("pos", name[4:]) if name.startswith("pos:") else ("few" if name in FEW else "tok", name)
        fire, quiet = M[c]
        fire = fire & okpos
        qm = (quiet != None) & okpos  # noqa: E711 (elementwise)
        fw, fp = np.nonzero(fire)
        qw, qp = np.nonzero(qm)
        if var != "pos":
            fk, qk = ids[fw, fp], ids[qw, qp]
        else:
            fk, qk = pos[fw, fp], pos[qw, qp]
        seed = [11, sum(map(ord, name)), 0 if split == "select" else 1]
        rng = np.random.default_rng(seed + [5])
        F = _cap(rng, fk.tolist(), win[fw], fp, CAP[var])
        Q = _cap(rng, qk.tolist(), win[qw], qp, CAP[var], quiet[qw, qp].tolist())
        n, pk = (N_SELECT, PER_KEY[var]["select"]) if split == "select" else (N_TEST, PER_KEY[var]["test"])
        pairs = SK._pairs(F, Q, n, pk, seed)
        if split == "test" and len(pairs) < N_TEST:
            pairs = SK._pairs(F, Q, n, 2 * pk, seed)
        plac = _placebo(Q, len(pairs) if split == "select" else N_TEST, seed + [2], PER_KEY[var]["select"])

        def snip(e):
            return SK.Snip(e.w, e.p, int(dom[e.w]), toks[e.w, e.p - SNIP + 1:e.p + 1])
        out[name] = [(snip(f), snip(q), st) for f, q, st in pairs]
        out["placebo:" + name] = [(snip(a), snip(b), st) for a, b, st in plac]
    return out


def holds(z, conds):
    """{condition: (W, T) bool} where each condition's fire predicate holds (for coverage; no position cutoff)."""
    from bonsai_interp import explore
    ids = explore.corpus()[z["windows"]].astype(np.int64)
    M, _ = masks(z, ids)
    return {c: M[c[4:] if c.startswith("pos:") else c][0] for c in conds}
