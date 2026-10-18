# bonsai-interp

An open, testable map of what individual neurons do in **Bonsai-8B**, PrismML's 1-bit LLM
(Apache-2.0, derived from Qwen3-8B). It also contains the toolchain used to build that map.

Every label in the graph comes with tests that were **committed before they were run**. Pull
requests that add or relabel neurons are checked for that ordering in CI.

## What is here

| Path | What it is |
|---|---|
| `bonsai_interp/` | The toolchain: model loading with exact 1-bit weights, corpus scan, forward probes, inverse solvers, steering, circuit tools, and the graph format |
| `graph/` | The interp graph. Append-only contributions (`graph/contributions/`) are folded into one derived record per neuron pole (`graph/neurons/`), plus `edges.json` and `index.json`. See [`graph/SCHEMA.md`](graph/SCHEMA.md) |
| `site/` | The Bonsai Neuron Survey: a map of all 442,368 neurons to zoom through, and the mission desk that hands out neurons to interpret or QC (`python site/build.py`) |
| `DATA_POLICY.md`, `MERGE_AGENT.md` | What may be committed and what belongs in the library, and the mechanical merge procedure |
| `probes/` | Full working files for each probed neuron: banks, pre-registered tests, results, dossiers |
| `runs/` | One-off studies, including the blind clean-room replication of L9/N3688 |
| `reports/` | HTML templates, the calibration report and the figures |
| `FINDINGS.md` | The lab notebook, in chronological order. Commit hashes cited there are pre-registrations |
| `RESEARCH_PLAN.md` | The original plan and the choice of tool per question |

## The method in one paragraph

A **neuron** is one SwiGLU channel, `silu(gate) · up`. Its positive and negative **poles** are
studied separately.

1. **Forward.** Read the top corpus examples and the full activation distribution.
2. **Label.** Write a label, then test it on new sentences and near misses written *before*
   evaluation.
3. **Inverse.** Ask a solver for the text that drives the pole to a target level. SciPy
   differential evolution is the local baseline. **globalMOO** trains a learned inverse once
   and then answers each target in a few evaluations.
4. **Induce.** Steer the neuron and see what the model writes.
5. **Map the circuit.** Find which neurons feed and read it, by single-unit ablation.
6. **Try to break the label.** Look at frequency-weighted triggers, the weak bands and leak
   searches. Run blind replications in a [clean room](https://github.com/moridinamael/neuron-cleanroom).

The worked example is **L9/N3688**, the "nerve" neuron:
`probes/bonsai1_L9_N3688_pos/dossier.html` and `web.html`.

## Quick start

```bash
uv sync                                   # Python 3.11-3.13; CUDA 12.8 torch wheels (PyPI torch with MPS on macOS)
bash scripts/download_models.sh           # Bonsai-8B into models/ (--all adds the ternary and Qwen3-8B ladder)
bash scripts/fetch_data.sh                # corpus.npz + scan_bonsai1.npz from the GitHub release into data/
uv run python -m bonsai_interp.explore --model bonsai1 --neuron 9/3688 --pole +
uv run python -m bonsai_interp.graph.validate
```

Hardware: one 16 GB NVIDIA GPU is enough (the reference setup is Windows with an RTX 5080). The
1-bit weights are stored packed and dequantized per layer. An Apple Silicon Mac also works (see
below), and the CPU works for checks. `bonsai_interp/env.py` detects the OS, device, dtype,
memory and agent CLIs in one place. `uv run python -m bonsai_interp.env` prints that report, and
`play doctor` prints it too. On a CUDA machine the defaults are unchanged: CUDA, bfloat16 and
the 0.8 memory cap. globalMOO is optional; it needs `GMOO_API_KEY` in `.env` (see
`.env.example`).

### Running on a Mac

The toolchain runs on Apple Silicon through PyTorch's MPS backend, with the same 2-bit packed
weights and the same bfloat16 arithmetic as on CUDA.

- **Requirements.** An M1 or later Mac with macOS 14 or later. The macOS torch wheels need
  macOS 14, and so does bfloat16 on MPS. We recommend 32 GB or more of unified memory. 16 GB
  works but is tight: one process holds about 8 GB (the 4.5 GB model, the 1.1 GB scan and the
  batches), next to macOS and the agent CLI. 8 GB is not enough. You also need about 18 GB of
  disk.
- **Install.** Use the same commands as above. `uv sync` takes torch from PyPI on macOS, and
  that build includes MPS. The scripts run in the stock bash or zsh.
- **Device.** The device is picked in this order: CUDA, then MPS, then the CPU. To force one,
  set `BONSAI_DEVICE=mps` (or `cuda` or `cpu`) in `.env` or in the environment. If you set it
  and that device is unavailable, the command stops with an error instead of falling back.
  `BONSAI_DTYPE=float32` overrides the arithmetic precision. An op that MPS lacks runs on the
  CPU (`PYTORCH_ENABLE_MPS_FALLBACK=1` is set for you).
- **Check parity.** Run `uv run python -m bonsai_interp.graph.play doctor`, then run
  `uv run python tests/test_device_parity.py`. The test compares four neuron readings, the
  next-token distribution and the backward pass (attribution through the 1-bit layers) with
  values recorded on the reference RTX 5080. It must print PARITY OK. If it does not, please
  open an issue with its output.
- **Speed.** Doctor prints the forward-pass throughput (batch 8 × 256) and how it compares
  with the reference machine. Measured so far: an RTX 5080 does 3,600 tokens/s and loads the
  model in 15 s. A 24-core Intel Core Ultra 9 CPU does 99 tokens/s, about 36 times slower. We
  have not measured a Mac yet. From GPU compute and memory bandwidth, and allowing for MPS
  being less efficient than CUDA, we expect missions to take about 4 to 6 times as long on an
  M-series Max or Ultra, about 10 times as long on a Pro, and about 20 times as long on a base
  chip. Only the model work scales by that factor; the agent's own reading and writing time
  does not. Blind rooms suit a Mac better than live play, because nobody waits on them. Please
  report what doctor prints on your Mac so we can replace these estimates.
- **Clean-room missions on macOS.** The Windows handling in neuron-cleanroom stays, behind
  platform checks. `build_room` links the weights and data into the room with junctions on
  Windows and with symlinks on macOS and Linux. Clearing read-only git files when a room is
  removed happens only on Windows. Codex gets its `[windows] sandbox = "unelevated"` setting
  only on Windows. Elsewhere Codex runs inside its own sandbox in `workspace-write` mode, which
  on macOS is Seatbelt. As on Windows, that sandbox blocks writes outside
  the room and blocks the network, but it does not block reads, so the audit is still the check
  on reads. We have not yet tested whether Metal is reachable inside the sandbox. Launch rooms
  with `BONSAI_DEVICE=mps` exported, so that a room which cannot see the GPU fails at once
  instead of running on the CPU for hours. Then run `play doctor` inside the room once.
- **A dedicated Mac.** A Mac mini that runs blind rooms keeps them off your display GPU. The
  rule of one GPU job at a time still applies, because unified memory is shared and two
  model-loading processes will compete for it.

To rebuild the data instead of downloading it, run `python -m bonsai_interp.corpus` and then
`python -m bonsai_interp.scan --model bonsai1` (about 1 GPU-hour for 8M tokens).

## Contributing

Add a neuron, relabel one, or contribute a blind replication. See
[CONTRIBUTING.md](CONTRIBUTING.md). The short version:

1. Draw a neuron on the survey page, or pick one to QC, and claim it.
2. `python -m bonsai_interp.graph.play new …` writes the pre-registration. Fill it in and commit it alone.
3. `python -m bonsai_interp.graph.play run …` writes the results; commit them separately (contribution files only).
4. Open a PR with the `interpret` or `qc` template. CI checks that contributions are append-only, that the tests came first, the size caps, and the derived files.

## Licences

- Code: MIT ([LICENSE](LICENSE)).
- Graph data, probe results, reports and figures: CC-BY-4.0 ([LICENSE-DATA](LICENSE-DATA)).
- The Bonsai models are Apache-2.0 (PrismML). The corpus is drawn from FineWeb, FineWeb-Edu,
  FineWeb-2 (ODC-By) and CodeParrot.

This work builds on Nick Cammarata's essay *Neural Networks Are Already Clean* (Goodfire, 2026).
The essay is not redistributed here.
