"""Build the survey game page from the derived index.

  python site/build.py
    -> site/survey.html   page body only (for embedding / the Artifact preview)
    -> site/index.html    complete document (for GitHub Pages)

Run after `python -m bonsai_interp.graph.fold`; the page embeds graph/index.json so it works offline.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    idx = json.loads((ROOT / "graph" / "index.json").read_text(encoding="utf-8"))
    blob = json.dumps(idx, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    body = (HERE / "template.html").read_text(encoding="utf-8").replace("__INDEX__", blob)
    (HERE / "survey.html").write_text(body, encoding="utf-8", newline="\n")
    doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
           "<style>:root{color-scheme:light dark}body{margin:0}[hidden]{display:none!important}img{max-width:100%}</style>\n"
           "</head>\n<body>\n" + body + "\n</body>\n</html>\n")
    (HERE / "index.html").write_text(doc, encoding="utf-8", newline="\n")
    print(f"wrote site/survey.html and site/index.html ({len(idx['neurons'])} poles)")


if __name__ == "__main__":
    main()
