# Playtest 2026-10-18

A fresh headless Claude Code agent (Opus 5.5, no project memory) played from the survey page's own prompts in an empty folder (`playtest.py`).

| Phase | Result | Time / cost |
|---|---|---|
| Bootstrap prompt → READY | Cloned both repos, `uv sync`, downloaded weights and data, doctor READY with the end-to-end check passing | about 4 min |
| Mission: interpret bonsai1/L24/N5287+ | Claim issue #1, band scan, exploration, pre-registration `38c94fe`, results `0063063`, PR #2 | about 13 min, $1.43 |

The label it found: "comeback / resurgence". It fires on the meaning of something returning, often most strongly on the token before the return, and not on other senses of "back". The test held at 5/8 with 0 false positives; two fire sentences sat just under the bar, at 0.295 and 0.298.

## Friction, and what was done

**Boot**

| Friction | Fix |
|---|---|
| Doctor's weights check passed partway through the download | Doctor now checks every shard and the total size (ee4bd8d) |
| Sizes in the docs didn't match | Corrected: about 18 GB in total |
| Output garbled on Windows (�) | Fixed: UTF-8 output |
| Harmless torch warning printed on every command | Silenced |
| Docs didn't say bash is needed on Windows | Git Bash requirement documented |
| Setup order didn't match the bootstrap prompt | Order aligned |
| A stray file with a maintainer's local path in the repo | Removed and ignored |

**Mission**

| Friction | Fix |
|---|---|
| The repo had no `claim` label | Created, along with held, failed, contested and refined |
| No CI | Pending: the token lacks the workflow scope (go-public checklist) |
| `bands sample` couldn't be run from the command line | Added |
| Scratch scripts couldn't import the package | Documented: run with `PYTHONPATH=.`, or use the new `play probe` |
| The band scan was slow and showed no estimate | It now prints an ETA, and the brief uses 1,000 windows |
| No command for exploratory probes | Added `play probe`, which records nothing |
| The 0.3 bar is easy to miss narrowly | The brief now warns about it |
| The brief never said to create a branch | The brief now creates one |
| `gh pr create --template` needs an editor | The brief now uses `--body-file` |
| Windows encoding and torch warning (carried over from boot) | Fixed in the package |

All mission fixes are in eef023a.
