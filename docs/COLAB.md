# Playing without a GPU: Google Colab

No NVIDIA GPU with 8 GB, and no Apple Silicon Mac? A free Google Colab notebook with a **Tesla T4** (15 GB) runs every
mission. It is slower than the reference machine, but it gives the same readings (`tests/test_device_parity.py` prints
PARITY OK on it).

## What to expect from a free T4

Measured on 2026-10-20 (full forward pass, batch 8 × 256, bfloat16, the default):

| | T4 (free Colab) | Reference RTX 5080 |
|---|---|---|
| Throughput | ~177 tokens/s | ~3,600 tokens/s |
| Model load | ~34 s | ~15 s |
| Peak GPU memory | 5.6 GB | ~5.2 GB |

- **Cheap on a T4:** sweeps, quick interpret missions, QC. Most of the time is the model load, so batch your work:
  one `probe --plan` for many sentences or poles, and one `play run` over several tests files.
- **Expensive:** band scans (`probe.bands scan --windows 1000` is about 1M tokens, so over an hour), and blind
  clean-room missions (hours, and they need the `claude` CLI on the GPU machine). For step and stepin missions,
  `play brief step --depth quick` leaves out the band scan.
- Free runtimes are recycled when idle or after a few hours, which wipes `/content`. Push your branch before you
  leave, and re-run the setup cell when you come back.
- Sometimes Colab has no free GPU ("Failed to assign a backend"). Decline the CPU-only runtime and try again later.

**Why the T4 is ~20× slower, not ~4×:** it has tensor cores for float16 but not for bfloat16, which this toolchain
uses to match the reference. A plain 4096 × 4096 matmul on the T4 runs at 25 TFLOPS in float16 but 2.3 TFLOPS in bfloat16.
`BONSAI_DTYPE=float16` ran the model at ~939 tokens/s, and the parity test passed (all readings within 0.004 of the
reference), but contributions are compared against a bfloat16 reference and `bonsai_interp/env.py` warns that float16
can overflow, so **keep the default bfloat16 for contributions** unless the maintainers say otherwise.

## Setup (once per runtime)

1. Open a new notebook at [colab.research.google.com](https://colab.research.google.com), then choose
   *Runtime → Change runtime type → T4 GPU*.
2. Paste this into the first cell and run it (about 3 minutes; it is safe to re-run):

```bash
%%bash
set -e
cd /content
[ -d bonsai-interp ] || git clone -q https://github.com/moridinamael/bonsai-interp
[ -d neuron-cleanroom ] || git clone -q https://github.com/moridinamael/neuron-cleanroom
pip -q install uv
cd bonsai-interp
uv sync -q
[ -f models/Bonsai-8B/config.json ] || bash scripts/download_models.sh > /tmp/models.log 2>&1
# the data release is public, so curl works without logging in to gh
for f in corpus.npz scan_bonsai1.npz bands_web.npz; do
  [ -s data/$f ] || curl -sSL -o data/$f https://github.com/moridinamael/bonsai-interp/releases/download/data-v1/$f
done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo SETUP DONE
```

3. Check it: `uv run python tests/test_device_parity.py` must print **PARITY OK**.

## Route A: everything in Colab (simplest for a person)

Open Colab's **Terminal** (bottom bar), then:

```bash
cd /content/bonsai-interp
git config --global user.name "<your name or handle>"
git config --global user.email "<your GitHub noreply address>"
gh auth login          # pick GitHub.com, HTTPS, "Login with a web browser"; enter the code it shows
uv run python -m bonsai_interp.graph.play doctor     # should end with READY
```

Then play missions exactly as AGENTS.md describes, in that terminal (or with an agent running there). Fork the
repository and push your branches to the fork. Colab's disk is temporary, so push before the runtime is recycled.

## Route B: git on your own computer, scoring in Colab

This suits an AI agent on your computer that drives the notebook through a browser. The checkout, every commit and
every PR stay on your computer, and Colab is only a GPU. Colab does not need to be logged in to GitHub. (`play doctor`
there then reports one expected FAIL, "GitHub CLI logged in"; the parity test is the check that matters.)

1. **On your computer:** do everything that needs no GPU (`brief`, `card`, `look`, `new`, `fill`, `validate`), commit
   the pre-registration alone, and push the branch to your fork.
2. **In Colab:** probe and score on the GPU:
   ```bash
   cd /content/bonsai-interp
   git fetch -q https://github.com/<you>/bonsai-interp <branch> && git checkout -q -B <branch> FETCH_HEAD
   uv run python -m bonsai_interp.graph.play run <tests files>
   python scripts/colab_results.py export        # HEAD is the pre-registration commit
   ```
   Do the probing and the `play run` of one mission on the same runtime: `.probe-log/` lives there, and it is how
   `play run` counts sentences scored before the pre-registration.
3. **Back on your computer:** save the printed rows to a file (e.g. `.sweep/rows.jsonl`) and run
   ```bash
   python scripts/colab_results.py rebuild .sweep/rows.jsonl
   ```
   It rebuilds each results file from the committed tests file plus the row, and writes it **only if its sha256 matches
   Colab's** (line endings normalised). Then commit the results separately, `validate`, push, and open the PR.
   Mention in the PR that scoring ran on a Colab T4.

Notes for agents driving Colab through a browser:
- Typed text may not reach Colab's code editor. Setting a cell's text through the page's editor model and running it
  with Ctrl+Enter works, and cell outputs can be read back from the page.
- `%%bash` cells show their output only when they finish. Run long jobs in the background with a log
  (`nohup env PYTHONUNBUFFERED=1 uv run python -m … > /tmp/job.log 2>&1 &`) and poll `tail /tmp/job.log`.
- Never type a Google or GitHub password for the person; ask them to sign in.
