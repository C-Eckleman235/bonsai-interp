# Sweep learnings: what would make re-sweeping a new network faster

A running log from the auto-token@2 sweep of Bonsai-8B (from 2026-10-20). It runs on one RTX 5080 that also drives the
display and a live Twitch stream. After every run unit (a layer, an all-layer pass or a scoring chunk), we add:
- what it took
- the bottleneck
- the one change made before the next unit
- the lesson that transfers to a new model (for example Bonsai-27B)

After the sweep this becomes the source for a write-up on re-sweeping a network fast.

Entry format:

```
### <date> <unit>
- measured: minutes per stage, tokens/s, GPU utilisation, peak memory, tests/snippets
- bottleneck: …
- change before the next unit: … (parity check: …)
- gain of the previous change: … (measured)
- transfers: …
```

## Lessons so far (before the all-layer passes)

**Compute plan**
- **Share passes across layers.** One forward pass can record every layer at once, so per-layer corpus passes
  repeat the same work. auto-token@2 ran stage 1 and stage 2 once per layer: 34 passes × about 50 GPU min.
  All-layer passes cut that to about 2–3 h. Scoring cannot be shared (each layer's snippets differ), so it becomes
  the long pole.
- **Test counts scale with how lexical a layer is.** L20 produced 1,185 auto tests; L4 produced 10,088
  (158,857 snippets × 3 preambles, about 2.5 h of scoring). Budget scoring per layer from stage 2's bin counts.
  Decide a cap on tests per pole before registering, not after.
- **Measure utilisation, not just time.** L4's word-hit stage ran at 17–63% GPU utilisation. It was CPU-bound on
  a sparse counter; moving the merge onto the GPU fixed it. A cap that forces batch 2 on 1,024-token windows also
  points to the attention kernel: check which SDPA backend actually runs.

**One display GPU, a live stream and a batch**
- **Memory caps are necessary but not sufficient.**
  - What works: `BONSAI_GPU_FRACTION` per checkout makes an over-budget process fail with an OOM instead of
    spilling.
  - Size a cap from the load peak, not the resident size: a Bonsai-8B load peaks at about 5.2 GB (the vocabulary
    matrix lands on top of 4.0 GB), while it holds 4.1 GB loaded.
  - What failed: five NVIDIA driver resets (nvlddmkm 153) in two days.
    - Four came as a second process started or loaded the model while another was computing.
    - One came in steady running with two processes computing at once.
- **What makes sharing safe: turn-taking on compute, not just budgets on memory.**
  `bonsai_interp/gpu_turn.py`:
  - Both models stay loaded.
  - The foreground (the stream) takes the GPU, and the batch yields at its next batch boundary.
  - Only one process runs kernels at a time.
- **Load once.** Every subprocess that reloads the model is a reset risk and costs about 25 s. Serving commands
  from one resident model per session removed both. The stream's `play why` dropped from about 28 s to 3.3 s.
- **Never force-kill a GPU process mid-kernel.** A Stop-Process during scoring was followed 3 s later by another
  reset and a TDR. Stop at batch boundaries (a STOP file).
- **Verify a "release" from the allocator, not from the caller.** A probe reported that it had yielded while still
  holding 4.4 GB. A stale model reference meant a second copy loaded beside it.
- **A smaller helper model frees room.** The stream's chat screener on Bonsai-8B (1.9 GB) instead of Bonsai-27B
  (4.3 GB) returned 2.4 GB to the budget.

**Process**
- **One registration for many layers is as valid as one per layer,** and much faster: fewer commits and fewer
  waits. What matters is that the rule and the test sets are pinned before scoring.
- **Keep per-occurrence activations from the screen** (`occ.npz`). A new context filter is then a CPU refit
  instead of another GPU pass.
- **Merge with merge commits, never squash or rebase,** so the registration hashes cited in summaries stay valid.
