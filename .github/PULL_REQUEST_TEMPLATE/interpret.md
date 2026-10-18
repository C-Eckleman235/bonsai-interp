<!-- PR shape 1 of 2: INTERPRET one or more neuron poles. Title: "interpret: bonsai1/L12/N5397- <short label>",
     or for a sweep "sweep: 16 poles (<held> held)" with one line per pole under "Sweep" below. -->

## Interpret: `bonsai1/L__/N____±`

- **Role:** <!-- target (from the game's random draw) / peripheral: feeds or reads <neuron id> (stepping out) -->
- **Label:** <!-- one sentence: what fires this pole, at the level your test supports -->
- **Unit kind:** <!-- lexical / contextual / sparse feature / dense / positional / unclear -->

## Pre-registration

| | commit |
|---|---|
| `<id>.tests.json` (label + 8 fire + 8 near misses) | <!-- hash, FIRST commit --> |
| `<id>.json` (results from `play run`) | <!-- hash, LATER commit --> |

## Result

<!-- paste the line printed by `play run`, e.g. "L12/N5397-: fire 6/8, near misses firing 0/8 -> HELD" -->

## Methods used

- Agent: <!-- e.g. claude-code opus-5.5, codex gpt-6.1, human -->
- Screens: <!-- from bonsai_interp/graph/methods.py: look, bands, probe, quick-test, step-out, ... (also in the results file's "methods") -->
- Depth / time: <!-- quick / standard / deep; minutes -->

## Edges (if you stepped out)

<!-- from -> to, kind, weight, method; also listed in the results file -->

## Library artefacts (optional)

<!-- name, url, sha256: anything bulky (banks, logs, transcripts); see DATA_POLICY.md -->

## Checklist

- [ ] Exactly the two contribution files are added, under `graph/contributions/…`, plus refreshed derived files from `python -m bonsai_interp.graph.fold`.
- [ ] Tests were committed before results (do not squash those commits).
- [ ] Near misses keep the topic and drop the claimed trigger.
- [ ] Nothing bulky, no tensors, no local paths (DATA_POLICY.md).
- [ ] `python -m bonsai_interp.graph.validate` passes.

## Sweep (only for a sweep PR)

| pole | label | outcome |
|---|---|---|
| <!-- bonsai1/L__/N____± --> | <!-- short label --> | <!-- held / failed / dense --> |
