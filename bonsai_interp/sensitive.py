"""Masking for sensitive words on the public map, the atlas and the stream.

Some poles fire on slurs. The survey keeps the science (which pole, what kind of trigger, how often) but never shows
the word itself on the public map, the atlas release or the Twitch stream. The denylist is stored as truncated
SHA-256 hashes of the normalised word (NFKC, casefolded), so the repository does not carry the words either.

  mask("text")      -> the text with every denylisted word replaced by MASK
  mask_token(" x")  -> MASK (keeping a leading space) if the token's word is denylisted, else the token unchanged
  masked(obj)       -> a copy of a JSON-like object with every string masked

To add a word, append the first 16 hex digits of sha256(unicodedata.normalize("NFKC", word).casefold()) to HASHES.
A pole whose trigger is a denylisted word gets a `withhold` contribution (graph/SCHEMA.md) instead of a live test.
"""
import hashlib
import re
import unicodedata

MASK = "[masked]"
HASHES = frozenset({
    "044eb98b18769b88", "08a841e996781e9e", "0ce875d620076b53", "120f6e5b4ea32f65",
    "12e6274e4309293e", "158869a97379229b", "16ea09fc78ca83ca", "17a110e533284872",
    "17bde8b4064612e2", "1e02eec4f1095143", "22fc75e65a0e9d34", "268651b3ece98010",
    "333f7618092958c7", "33740da8a87dc7de", "341d56384afc0f47", "3b1e0d7c5dd45583",
    "402f7ebd98864afe", "45cd3e1bb472d8e2", "52c1d7686727b74e", "5b3ae48be122f7ed",
    "6d1833779e389477", "70e4043b678ff365", "7122c9a14ba2d20e", "82159dd02870c13e",
    "85580f904474c32c", "87d1690e74a1d827", "886d51e97ad7931d", "8f5083e3e5c7dc89",
    "92a9bf818e7e0002", "94c6d2e10a76c313", "98b52c4b6b7d1f48", "9915ba2d822280f2",
    "bb209d506bb8b178", "bd331fb1d24298f5", "c1cabb6f6e431f9c", "c3de533e9b7fe63b",
    "c74677aba35aef49", "cc02032349c833ac", "da0a6b1b9213d48f", "daec0c235d3135fc",
    "dc675e448132fd2a", "e793d67686f2824f", "e7b98c6aa5b944e0", "eef3bd091670c344",
    "f9d0d9b18ae9033a",
})
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def _h(word):
    return hashlib.sha256(unicodedata.normalize("NFKC", word).casefold().encode()).hexdigest()[:16]


def sensitive(word):
    """True when this single word is on the denylist."""
    return _h(word) in HASHES


def mask(text):
    if not text or not isinstance(text, str):
        return text
    return _WORD.sub(lambda m: MASK if sensitive(m.group(0)) else m.group(0), text)


def mask_token(tok):
    """A vocabulary entry: masked whole if its letters form a denylisted word (a leading space is kept)."""
    if not tok or not isinstance(tok, str):
        return tok
    core = "".join(_WORD.findall(tok))
    if core and sensitive(core):
        return (" " if tok[:1].isspace() else "") + MASK
    return mask(tok)


def masked(obj):
    if isinstance(obj, str):
        return mask(obj)
    if isinstance(obj, list):
        return [masked(x) for x in obj]
    if isinstance(obj, dict):
        return {k: masked(v) for k, v in obj.items()}
    return obj
