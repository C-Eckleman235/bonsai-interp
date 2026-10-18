"""Redact local paths (and the user name inside them) from run evidence before it is committed.

  python scripts/redact_paths.py <file or folder> ...      rewrites .json/.jsonl/.md/.txt files in place

C:/Users/<name>/…, C:\\Users\\<name>\\…, /c/Users/<name>/…, /home/<name>/… and /Users/<name>/… become ~/…,
Claude Code's project keys (C--Users-<name>-…) become C--Users-USER-…, and any other whole-word
occurrence of the user name (the owner column of `ls -l`) becomes USER. DATA_POLICY: no local paths.
"""
import re
import sys
from pathlib import Path

NAME = Path.home().name


def redact(text, name=NAME):
    n = re.escape(name)
    text = re.sub(rf"(?i)[a-z]:(?:\\\\|\\|/)+users(?:\\\\|\\|/)+{n}", "~", text)
    text = re.sub(rf"(?i)/[a-z]/users/{n}", "~", text)
    text = re.sub(rf"/(?:home|Users)/{n}", "~", text)
    text = re.sub(rf"(?i)([a-z])--users-{n}-", r"\1--Users-USER-", text)
    return re.sub(rf"\b{n}\b", "USER", text)  # e.g. the owner column of `ls -l`


def main():
    for arg in sys.argv[1:]:
        p = Path(arg)
        for f in ([p] if p.is_file() else [q for q in p.rglob("*") if q.suffix in (".json", ".jsonl", ".md", ".txt")]):
            s = f.read_text(encoding="utf-8")
            r = redact(s)
            if r != s:
                f.write_text(r, encoding="utf-8", newline="\n")
                print(f"redacted {f}")


if __name__ == "__main__":
    main()
