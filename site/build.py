"""Build the survey game page from the derived index.

  python site/build.py [--refresh-cards]
    -> site/survey.html   page body only (for embedding / the Artifact preview)
    -> site/index.html    complete document (for GitHub Pages)
    -> site/cards.json    neuron cards for every surveyed neuron (a cache: built from data/scan_bonsai1.npz and
                          the weights, so machines without them reuse it; --refresh-cards rebuilds every card)

Run after `python -m bonsai_interp.graph.fold`; the page embeds graph/index.json and the cards so it works offline.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


def build_cards(idx, refresh=False, logits=True):
    """{"L/n": compact card} for every neuron with at least one surveyed pole; new ones computed when the scan is here."""
    path = HERE / "cards.json"
    cache = {} if refresh or not path.exists() else json.loads(path.read_text(encoding="utf-8"))
    want = sorted({f"{int(L)}/{int(n)}" for L, n in (k.split("/L")[1].rstrip("+-").split("/N") for k in idx["neurons"])},
                  key=lambda u: tuple(map(int, u.split("/"))))
    todo = [u for u in want if u not in cache]
    if todo:
        if not (ROOT / "data" / "scan_bonsai1.npz").exists():
            print(f"note: {len(todo)} neurons have no card yet and data/scan_bonsai1.npz is missing (scripts/fetch_data.sh); building without them")
        else:
            from bonsai_interp.graph import card
            for i, u in enumerate(todo):
                cache[u] = card.compact(card.card(f"bonsai1/L{u.replace('/', '/N')}", logits=logits))
                print(f"  card {i + 1}/{len(todo)} L{u}", end="\r", flush=True)
            print()
    cards = {u: cache[u] for u in want if u in cache}
    path.write_text(json.dumps(cards, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    return cards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-cards", action="store_true")
    ap.add_argument("--no-logits", action="store_true", help="new cards without the logit effect (no model weights needed; the Pages build)")
    a = ap.parse_args()
    idx = json.loads((ROOT / "graph" / "index.json").read_text(encoding="utf-8"))
    cards = build_cards(idx, a.refresh_cards, logits=not a.no_logits)
    # U+FFFD (corpus text with broken characters) is written as a JSON escape, so the page source has none
    blob = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/").replace("�", "\\ufffd")
    body = (HERE / "template.html").read_text(encoding="utf-8").replace("__INDEX__", blob(idx)).replace("__CARDS__", blob(cards))
    (HERE / "survey.html").write_text(body, encoding="utf-8", newline="\n")
    doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
           "<style>:root{color-scheme:light dark}body{margin:0}[hidden]{display:none!important}img{max-width:100%}</style>\n"
           "</head>\n<body>\n" + body + "\n</body>\n</html>\n")
    (HERE / "index.html").write_text(doc, encoding="utf-8", newline="\n")
    print(f"wrote site/survey.html and site/index.html ({len(idx['neurons'])} poles, {len(cards)} neuron cards, "
          f"{len(doc.encode()) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
