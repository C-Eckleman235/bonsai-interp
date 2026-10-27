"""Part of speech and morphology as minimal-pair batteries (sweeplib@4; CPU only, deterministic, from the spaCy parse).

The organizing principle: the model is a model of language, so word classes should be a basic axis of its neurons ("noun-
like", "verb-like", "adjective-like"). Three families of condition, read at a snippet's last token, which must end a whole
alphabetic word (the grammar battery's rule):

  class:X   the last word's coarse part of speech is X (NOUN, PROPN, VERB, ADJ, ADV, PRON, DET, ADP, AUX, CCONJ, SCONJ,
            NUM), against words of other classes (strata: their class). Pairs share a word-frequency band, so a class
            pole cannot pass by preferring common or rare words.
  same:X    the same token as class X against that token in another class ('run' as a verb vs 'run' as a noun): the
            class carried by the context, not the word.
  morph:M   a morphological feature within one class, against the same class without it, paired by frequency band:
            plural (NNS/NNPS vs NN/NNP), past (VBD vs VBP/VBZ), third_sg (VBZ vs VBP), gerund (VBG vs other verb forms),
            participle (VBN vs other verb forms), comparative (JJR/RBR vs JJ/RB), superlative (JJS/RBS vs JJ/RB),
            proper (PROPN vs NOUN).

Frequency bands: 12 log-spaced bands of the token's corpus count (the word screen's unigram counts).
"""
from functools import lru_cache

import numpy as np

from bonsai_interp.auto import context_tier as ct
from bonsai_interp.grammar import cons as C
from bonsai_interp.sweeplib import kinds as SK

CLASSES = ("NOUN", "PROPN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "AUX", "CCONJ", "SCONJ", "NUM")
NAMES = {"NOUN": "a noun", "PROPN": "a proper noun", "VERB": "a verb", "ADJ": "an adjective", "ADV": "an adverb",
         "PRON": "a pronoun", "DET": "a determiner", "ADP": "a preposition", "AUX": "an auxiliary verb",
         "CCONJ": "a coordinating conjunction", "SCONJ": "a subordinating conjunction", "NUM": "a number word"}
MORPH = {  # name: (label, fire tags, quiet tags)
    "plural": ("a plural noun, not a singular one", ("NNS", "NNPS"), ("NN", "NNP")),
    "past": ("a past-tense verb, not a present-tense one", ("VBD",), ("VBP", "VBZ")),
    "third_sg": ("a third-person singular present verb (-s), not another present form", ("VBZ",), ("VBP",)),
    "gerund": ("an -ing verb form, not another verb form", ("VBG",), ("VB", "VBD", "VBN", "VBP", "VBZ")),
    "participle": ("a past participle, not another verb form", ("VBN",), ("VB", "VBD", "VBG", "VBP", "VBZ")),
    "comparative": ("a comparative adjective or adverb (-er, more), not the plain form", ("JJR", "RBR"), ("JJ", "RB")),
    "superlative": ("a superlative adjective or adverb (-est, most), not the plain form", ("JJS", "RBS"), ("JJ", "RB")),
}
BANDS = 12
PER_KEY = {"band": {"select": 16, "test": 2}, "tok": {"select": 8, "test": 1}}
CAP = {"band": 400, "tok": 12}


def labels():
    out = {f"class:{x}": f"part of speech: fires on {NAMES[x]}, whatever the word, not on words of other classes of the same "
                         "frequency" for x in CLASSES}
    out.update({f"same:{x}": f"part of speech in context: fires on a word used as {NAMES[x]}, not on the same word used as "
                             "another class" for x in CLASSES})
    out.update({f"morph:{m}": f"morphology: fires on {lab}, of the same frequency" for m, (lab, _, _) in MORPH.items()})
    out["morph:proper"] = "morphology: fires on a proper noun, not on a common noun of the same frequency"
    return out


@lru_cache(None)
def _band():
    """Per token id: its corpus frequency band (0..BANDS-1, log-spaced)."""
    from bonsai_interp import explore
    V = ct.vocab()["V"]
    cnt = np.bincount(explore.corpus().ravel().astype(np.int64), minlength=V).astype(np.float64)
    lg = np.log10(cnt + 1)
    return np.clip((lg / max(lg.max(), 1e-9) * BANDS).astype(np.int64), 0, BANDS - 1)


def masks(z, ids):
    """{condition: (fire (W, T) bool, quiet (W, T) object array of strata or None, key kind)} for a parse z and ids."""
    W, T = ids.shape
    nm = {k: z[f"{k}_names"] for k in ("pos", "tag")}
    pos, tag = nm["pos"][z["pos"]], nm["tag"][z["tag"]]
    word = z["last"] & np.isin(pos, CLASSES)
    out = {}
    for x in CLASSES:
        f = word & (pos == x)
        q = np.full((W, T), None, dtype=object)
        qm = word & (pos != x)
        q[qm] = pos[qm]
        out[f"class:{x}"] = (f, q, "band")
        out[f"same:{x}"] = (f, q, "tok")
    for m, (_, ft, qt) in MORPH.items():
        f = word & np.isin(tag, ft)
        q = np.full((W, T), None, dtype=object)
        qm = word & np.isin(tag, qt)
        q[qm] = tag[qm]
        out[f"morph:{m}"] = (f, q, "band")
    f = word & (pos == "PROPN")
    q = np.full((W, T), None, dtype=object)
    qm = word & (pos == "NOUN")
    q[qm] = "NOUN"
    out["morph:proper"] = (f, q, "band")
    return out, pos


def battery(z, split, conds=None):
    """{condition: [(fire Snip, quiet Snip, stratum)], "placebo:<condition>": [...]}; the grammar battery with keys by
    frequency band (class:, morph:) or by token (same:)."""
    from bonsai_interp import explore
    toks, dom = explore.corpus(), ct._domains()
    win = z["windows"]
    ids = toks[win].astype(np.int64)
    M, _ = masks(z, ids)
    okpos = np.zeros(ids.shape, bool)
    okpos[:, C.MIN_POS + C.SNIP - 1:] = True
    alpha, endws = C._whole()
    prev = np.zeros(ids.shape, bool)
    prev[:, 1:] = endws[ids[:, :-1]]
    okpos &= alpha[ids] & (ct.vocab()["lead"][ids] | prev)
    band = _band()
    out = {}
    for name in (conds or labels()):
        fire, quiet, var = M[name]
        fire = fire & okpos
        qm = (quiet != None) & okpos  # noqa: E711 (elementwise)
        fw, fp = np.nonzero(fire)
        qw, qp = np.nonzero(qm)
        fk, qk = ids[fw, fp], ids[qw, qp]
        if var == "band":
            fk, qk = band[fk], band[qk]
        seed = [13, sum(map(ord, name)), 0 if split == "select" else 1]
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
    """{condition: (W, T) bool} where each fire predicate holds (coverage)."""
    from bonsai_interp import explore
    ids = explore.corpus()[z["windows"]].astype(np.int64)
    M, _ = masks(z, ids)
    return {c: M[c][0] for c in conds}
