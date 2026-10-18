"""Play sweep missions headless, as a player would: the survey page's sweep prompt, pasted into a fresh agent
in an ordinary player checkout. One sweep at a time (the GPU is shared with the display).

  python runs/playtest/sweep.py [--agent claude|codex] [--k 16] [--budget 8]
      -> runs/playtest/<date>/sweep-<n>-<agent>.jsonl   (transcript; its final message ends with a Friction list)

The player checkout is PLAYTEST_DIR/bonsai-interp (default ~/Documents/survey-playtest), set up by the bootstrap
playtest and refreshed here with git pull. Runs below normal priority so the desktop stays responsive.
"""
import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("PLAYTEST_DIR", Path.home() / "Documents" / "survey-playtest")) / "bonsai-interp"
HANDLE = "moridinamael"


def prompt(k):
    # the survey page's sweep prompt (site/template.html, sweepPrompt), with the handle filled in
    return (f"In my bonsai-interp checkout, first update it (git pull), then run:\n\nuv run python -m bonsai_interp.graph.play brief sweep --k {k} --handle {HANDLE}\n\n"
            "Then carry out the brief it prints, following AGENTS.md: draw 16 unsurveyed poles, label each from its card, pre-register all the tests in one commit, "
            "run them together, commit the results separately and open one pull request. Stop only for things I must do myself. "
            "When you're done, tell me how many held and give me the pull-request link."
            "\n\n---\nPlaytest note: the person is away. gh is logged in and the machine is ready (play doctor says READY). "
            "Do not ask questions; decide sensibly and say what you decided. Only one GPU job at a time: never run two model-loading commands at once. "
            "End your final message with a short list headed 'Friction' of anything confusing, broken or slow in the brief or the tools.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["claude", "codex"], default="claude")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--budget", type=float, default=8.0)
    a = ap.parse_args()
    out = HERE / dt.date.today().isoformat()
    out.mkdir(parents=True, exist_ok=True)
    n = 1 + len(list(out.glob("sweep-*.jsonl")))
    subprocess.run(["git", "checkout", "-q", "main"], cwd=WORK)
    subprocess.run(["git", "pull", "-q", "--ff-only"], cwd=WORK)
    if a.agent == "claude":
        cmd = ["claude", "-p", prompt(a.k), "--model", "claude-opus-5-5", "--setting-sources", "project", "--strict-mcp-config",
               "--allowedTools", "Bash,Read,Edit,Write,Glob,Grep", "--max-budget-usd", str(a.budget),
               "--output-format", "stream-json", "--verbose"]
    else:
        cmd = ["codex", "exec", "--json", "--sandbox", "danger-full-access", "-c", "approval_policy=never", prompt(a.k)]
    flags = 0x00004000 if sys.platform == "win32" else 0  # BELOW_NORMAL_PRIORITY_CLASS
    t = out / f"sweep-{n}-{a.agent}.jsonl"
    with open(t, "w", encoding="utf-8") as f, open(t.with_suffix(".stderr.txt"), "w", encoding="utf-8") as e:
        rc = subprocess.run(cmd, cwd=WORK, stdout=f, stderr=e, stdin=subprocess.DEVNULL, creationflags=flags).returncode
    print(f"sweep {n} ({a.agent}): exit {rc}; transcript {t}")


if __name__ == "__main__":
    main()
