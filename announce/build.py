"""Build the launch announcement page from the derived graph index.

  python announce/build.py   ->  announce/index.html (page body; images in announce/img/)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    idx = json.loads((ROOT / "graph" / "index.json").read_text(encoding="utf-8"))["neurons"]
    pts = []
    for nid, r in idx.items():
        _, L, n = nid.split("/")
        pts.append({"L": int(L[1:]), "n": int(n[1:-1]), "pole": n[-1], "st": r["status"], "k": r.get("scrutiny", {}).get("level", 1), "s": r.get("short") or (r.get("label") or "")[:80]})
    blob = json.dumps(pts, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = (HERE / "template.html").read_text(encoding="utf-8").replace("__POINTS__", blob)
    (HERE / "index.html").write_text(page, encoding="utf-8", newline="\n")
    print(f"wrote announce/index.html with {len(pts)} surveyed poles")


if __name__ == "__main__":
    main()
