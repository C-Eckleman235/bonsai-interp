"""Check that a pull request pre-registered its tests: every result file it adds under probes/
must sit next to a test file that was committed in an EARLIER commit (not the same one).

  python -m bonsai_interp.graph.check_prereg --base origin/main     (CI runs this on pull requests)

File roles are decided by name (see CONTRIBUTING.md):
  test files    *tests*.json, holdout*.json, *plan*.json, *_test.json, PREREG*.md, *request*.json
  result files  quick*.json, *_quick*.json, *results*.json, *_out.json, solo.json, invert_*.json
"""
import argparse
import fnmatch
import subprocess
import sys
from pathlib import PurePosixPath

TESTS = ["*tests*.json", "holdout*.json", "*plan*.json", "*_test.json", "PREREG*.md", "*request*.json"]
RESULTS = ["quick*.json", "*_quick*.json", "*results*.json", "*_out.json", "solo.json", "invert_*.json"]


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout


def role(name):
    if any(fnmatch.fnmatch(name, p) for p in TESTS):
        return "test"
    if any(fnmatch.fnmatch(name, p) for p in RESULTS):
        return "result"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="origin/main")
    a = ap.parse_args()
    commits = git("rev-list", "--reverse", f"{a.base}..HEAD").split()
    order = {c: i for i, c in enumerate(commits)}
    first = {}  # path -> index of the commit that added it in this PR
    for c in commits:
        for line in git("show", "--name-status", "--format=", c).splitlines():
            st, *paths = line.split("\t")
            if st.startswith(("A", "R", "C")) and paths:
                first.setdefault(paths[-1], order[c])
    problems = []
    for path, i in first.items():
        p = PurePosixPath(path)
        if not path.startswith("probes/") or role(p.name) != "result":
            continue
        tests_here = [(q, j) for q, j in first.items() if PurePosixPath(q).parent == p.parent and role(PurePosixPath(q).name) == "test"]
        on_base = [q for q in git("ls-tree", "--name-only", a.base, str(p.parent) + "/").split() if role(PurePosixPath(q).name) == "test"]
        if on_base or any(j < i for _, j in tests_here):
            continue
        same = [q for q, j in tests_here if j == i]
        problems.append(f"{path}: no test file committed before it" + (f" (tests added in the same commit: {same})" if same else ""))
    for m in problems:
        print("PREREG", m)
    print(f"{len(commits)} commits checked, {len(problems)} pre-registration problems")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
