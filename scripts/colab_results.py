"""Move `play run` results from a GPU runtime (e.g. Google Colab) to the checkout that holds the git history,
without copying whole files by hand. See docs/COLAB.md, "Route B".

On the GPU runtime, after `play run`, from the repository root:
    python scripts/colab_results.py export [--commit <pre-registration commit, default HEAD>]
prints one compact JSON row per results file: its path, sha256, and the parts `play run` computed
(created, tests, methods). Everything else in a results file is copied from the committed tests file.

On the PC, with the same pre-registration commit checked out, save the rows to a file and run:
    python scripts/colab_results.py rebuild rows.jsonl
It rebuilds each results file exactly as `play run` writes it, and writes it ONLY if its sha256 matches the
runtime's. A mismatch means a row was copied wrongly (fix the copy, never a number) or the results format has
changed since this script was written (then copy the file itself and check its sha256).
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OPTIONAL = ("short", "unit_kind", "confidence", "notes", "roles", "via", "selection", "trigger")  # play run's order


def digest(b):
    """sha256 of the file with LF line endings (git on Windows may check files out with CRLF)."""
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


def results_for(commit):
    names = subprocess.run(["git", "show", "--name-only", "--format=", commit], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.split()
    return [n[:-len(".tests.json")] + ".json" for n in names if n.endswith(".tests.json")]


def export(a):
    for rel in results_for(a.commit):
        p = ROOT / rel
        if not p.exists():
            print(f"# missing {rel}: was it run?", file=sys.stderr)
            continue
        b = p.read_bytes()
        d = json.loads(b)
        print(json.dumps([rel, digest(b), d["created"], d["tests"], d["methods"]],
                         ensure_ascii=True, separators=(",", ":")))


def build(t, rel_tests, created, tests, methods):
    """The same dict `play run` writes (bonsai_interp/graph/play.py, run)."""
    c = {"id": t["contribution"], "kind": t["kind"], "neuron": t["neuron"], "contributor": t["contributor"],
         "created": created,
         "recipe": {"toolchain_commit": methods["toolchain_commit"], "command": f"python -m bonsai_interp.graph.play run {rel_tests}"},
         "label": t["neighbours"][0]["label"], "tests": tests, "methods": methods}
    if t["kind"] == "qc":
        c["qc"] = {"verdict": t["qc"]["verdict"], "label_under_test": t["qc"].get("label_under_test")}
        if t["qc"]["verdict"] != "refine":
            c.pop("label")
    for k in OPTIONAL:
        if t.get(k) is not None:
            c[k] = t[k]
    return (json.dumps(c, ensure_ascii=False, indent=1) + "\n").encode("utf-8")


def rebuild(a):
    ok = n = 0
    for line in Path(a.rows).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        n += 1
        rel, sha, created, tests, methods = json.loads(line)
        rel_tests = rel[:-len(".json")] + ".tests.json"
        t = json.loads((ROOT / rel_tests).read_text(encoding="utf-8"))
        data = build(t, rel_tests, created, tests, methods)
        if digest(data) == sha:
            (ROOT / rel).write_bytes(data)
            ok += 1
            print("MATCH", rel)
        else:
            print("DIFF ", rel, "(not written)")
    print(f"{ok}/{n} rebuilt byte for byte")
    sys.exit(0 if ok == n else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="on the GPU runtime: print one row per results file")
    e.add_argument("--commit", default="HEAD", help="the pre-registration commit whose tests files were run (default HEAD)")
    r = sub.add_parser("rebuild", help="on the PC: rebuild the results files from the rows, checking each sha256")
    r.add_argument("rows", help="a file with the exported rows, one per line")
    a = ap.parse_args()
    {"export": export, "rebuild": rebuild}[a.cmd](a)


if __name__ == "__main__":
    main()
