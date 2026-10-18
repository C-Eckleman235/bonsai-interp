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
uv sync                                   # Python 3.11-3.13, CUDA 12.8 wheels for torch
bash scripts/download_models.sh           # Bonsai-8B into models/ (--all adds the ternary and Qwen3-8B ladder)
bash scripts/fetch_data.sh                # corpus.npz + scan_bonsai1.npz from the GitHub release into data/
uv run python -m bonsai_interp.explore --model bonsai1 --neuron 9/3688 --pole +
uv run python -m bonsai_interp.graph.validate
```

Hardware: one 16 GB GPU is enough. The 1-bit weights are stored packed and dequantized per
layer. globalMOO is optional; it needs `GMOO_API_KEY` in `.env` (see `.env.example`).

To rebuild the data instead of downloading it, run `python -m bonsai_interp.corpus` and then
`python -m bonsai_interp.scan --model bonsai1` (about 1 GPU-hour for 8M tokens).

## Contributing

Add a neuron, relabel one, or contribute a blind replication. See
[CONTRIBUTING.md](CONTRIBUTING.md). The short version:

1. Draw a neuron on the survey page, or pick one to QC, and claim it.
2. `python -m bonsai_interp.graph.play new …` writes the pre-registration. Fill it in and commit it alone.
3. `python -m bonsai_interp.graph.play run …` writes the results. Run `python -m bonsai_interp.graph.fold`, then commit again.
4. Open a PR with the `interpret` or `qc` template. CI checks that contributions are append-only, that the tests came first, the size caps, and the derived files.

## Licences

- Code: MIT ([LICENSE](LICENSE)).
- Graph data, probe results, reports and figures: CC-BY-4.0 ([LICENSE-DATA](LICENSE-DATA)).
- The Bonsai models are Apache-2.0 (PrismML). The corpus is drawn from FineWeb, FineWeb-Edu,
  FineWeb-2 (ODC-By) and CodeParrot.

This work builds on Nick Cammarata's essay *Neural Networks Are Already Clean* (Goodfire, 2026).
The essay is not redistributed here.
