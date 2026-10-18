"""Pull-request checks that need git history (CI runs this on every PR):

  python -m bonsai_interp.graph.check_prereg --base origin/main

1. Append-only: a PR may add files under graph/contributions/ but never modify, rename or delete one.
2. Pre-registration order: every contribution results file (<id>.json) added by the PR names its
   pre-registration (tests_file); that file must be on the base branch already or have been added
   in an EARLIER commit of the PR, never in the same commit.
   For legacy probe folders (probes/), result files must likewise follow a test file in that folder.
3. Size: files added under graph/ are at most 64 KB each and 256 KB in total per PR (bulky material
   belongs in the library; see DATA_POLICY.md). Nothing may be added under data/ or models/.
"""
import argparse
import fnmatch
import json
import subprocess
import sys
from pathlib import PurePosixPath

TESTS = ["*.tests.json", "*tests*.json", "holdout*.json", "*plan*.json", "*_test.json", "PREREG*.md", "*request*.json"]
RESULTS = ["quick*.json", "*_quick*.json", "*results*.json", "*_out.json", "solo.json", "invert_*.json"]
MAX_FILE, MAX_TOTAL = 64 * 1024, 256 * 1024


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, encoding="utf-8", check=True).stdout


def role(name):
    if any(fnmatch.fnmatch(name, p) for p in TESTS):
        return "test"
    if any(fnmatch.fnmatch(name, p) for p in RESULTS):
        return "result"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--allow-derived", action="store_true", help="for the merge agent's own fold commits")
    a = ap.parse_args()
    commits = git("rev-list", "--reverse", f"{a.base}..HEAD").split()
    order = {c: i for i, c in enumerate(commits)}
    first, problems = {}, []
    for c in commits:
        for line in git("show", "--name-status", "--format=", c).splitlines():
            st, *paths = line.split("\t")
            derived = paths and paths[-1].startswith(("graph/neurons/", "graph/edges.json", "graph/index.json"))
            if paths and (st.startswith(("A", "C")) or (st.startswith("M") and derived)):  # a modified contribution must stay unrecorded here
                first.setdefault(paths[-1], order[c])
            for p in paths:
                if p.startswith("graph/contributions/") and (st.startswith(("M", "D", "R"))) and p not in first:
                    problems.append(f"{p}: contributions are append-only; it was {dict(M='modified', D='deleted', R='renamed').get(st[0])} in {c[:7]}")
    on_base = lambda p: subprocess.run(["git", "cat-file", "-e", f"{a.base}:{p}"], capture_output=True).returncode == 0
    total = 0
    for path, i in first.items():
        p = PurePosixPath(path)
        if path.startswith(("graph/neurons/", "graph/edges.json", "graph/index.json")) and not a.allow_derived:
            problems.append(f"{path}: derived file in a PR; drop it (git rm --cached) - the merge agent regenerates derived files on main")
        if path.startswith(("data/", "models/")):
            problems.append(f"{path}: nothing may be added under data/ or models/ (use the release or the library)")
        if path.startswith("graph/"):
            size = len(git("show", f"HEAD:{path}").encode("utf-8")) if path in git("ls-tree", "-r", "--name-only", "HEAD").split("\n") else 0
            total += size
            if size > MAX_FILE and path.startswith("graph/contributions/"):
                problems.append(f"{path}: {size // 1024} KB is over the {MAX_FILE // 1024} KB per-file cap")
        if path.startswith("graph/contributions/") and p.suffix == ".json" and not p.name.endswith(".tests.json"):
            c = json.loads(git("show", f"HEAD:{path}"))
            for t in c.get("tests", []):
                tf = t.get("tests_file", "")
                if on_base(tf) or (tf in first and first[tf] < i):
                    continue
                same = " (added in the same commit as the results)" if first.get(tf) == i else ""
                problems.append(f"{path}: pre-registration {tf!r} was not committed before the results{same}")
        elif path.startswith("probes/") and role(p.name) == "result":
            tests_here = [j for q, j in first.items() if PurePosixPath(q).parent == p.parent and role(PurePosixPath(q).name) == "test"]
            base_tests = [q for q in git("ls-tree", "--name-only", a.base, str(p.parent) + "/").split() if role(PurePosixPath(q).name) == "test"]
            if not base_tests and not any(j < i for j in tests_here):
                problems.append(f"{path}: no test file committed before it in {p.parent}")
    if total > MAX_TOTAL:
        problems.append(f"this PR adds {total // 1024} KB under graph/, over the {MAX_TOTAL // 1024} KB cap")
    for m in problems:
        print("PR CHECK", m)
    print(f"{len(commits)} commits checked, {len(problems)} problems")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
