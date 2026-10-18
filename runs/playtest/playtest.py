"""Playtest the survey as a brand-new player: a fresh headless Claude Code agent in an empty folder,
given only the prompts the survey page shows, plus a short note standing in for the person's answers.

  python runs/playtest/playtest.py boot      phase 1: the page's bootstrap prompt, until READY
  python runs/playtest/playtest.py mission "<mission prompt copied from the page>"

Each phase writes runs/playtest/<date>/<phase>.jsonl (the full stream-json transcript) for the friction review.
"""
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("PLAYTEST_DIR", Path.home() / "Documents" / "survey-playtest"))
OUT = HERE / dt.date.today().isoformat()
REPO, CLEAN = "moridinamael/bonsai-interp", "moridinamael/neuron-cleanroom"

BOOT = (f"Set me up to play the Bonsai Neuron Survey, an open, test-backed map of what each neuron in the 1-bit Bonsai-8B model does.\n\n"
        f"1. Ask me where to work (default: a folder called bonsai-survey in my home directory). Clone these two repositories side by side there:\n"
        f"   https://github.com/{REPO}\n   https://github.com/{CLEAN}\n"
        f"2. In bonsai-interp, read AGENTS.md and follow section 1, \"Get ready\".\n"
        f"3. Run: uv run python -m bonsai_interp.graph.play doctor\n   Fix every FAIL and run it again until it prints READY.\n"
        f"4. Ask me only for what you can't do yourself: logging in to GitHub (gh auth login), permission to use about 20 GB of disk, "
        f"and, optionally, a globalMOO API key for .env.\n"
        f"5. Finish by telling me \"Ready to survey\", with the GPU and the doctor summary. From then on I'll paste mission prompts from the survey page.")

ANSWERS = (f"\n\n---\nPlaytest note: the person is away and gave these answers in advance; use them instead of asking.\n"
           f"- Work folder: {WORK.as_posix()} (it already exists and is empty).\n"
           f"- Disk use: approved.\n- GitHub: gh is already logged in; my handle is moridinamael.\n"
           f"- globalMOO: no key; play without it.\n"
           f"- Keep a short list of anything confusing, broken or slow in these instructions and put it at the end of your final message under 'Friction'.")


def run(phase, prompt, resume=False):
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = ["claude", "-p", prompt + ANSWERS, "--model", "claude-opus-5-5", "--setting-sources", "project", "--strict-mcp-config",
           "--allowedTools", "Bash,Read,Edit,Write,Glob,Grep,Agent", "--output-format", "stream-json", "--verbose"]
    if resume:
        cmd.insert(1, "--continue")
    with open(OUT / f"{phase}.jsonl", "w", encoding="utf-8") as f, open(OUT / f"{phase}.stderr.txt", "w", encoding="utf-8") as e:
        rc = subprocess.run(cmd, cwd=WORK, stdout=f, stderr=e).returncode
    print(f"{phase}: exit {rc}; transcript {OUT / (phase + '.jsonl')}")


if __name__ == "__main__":
    WORK.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "boot":
        run("boot", BOOT)
    else:
        run("mission", sys.argv[2], resume=True)
