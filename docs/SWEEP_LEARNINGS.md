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
    - One (10:20:24) came with NO stream model on the card. Only the batch was computing, beside the
      stream's llama-server chat screen, the NVENC video encoder and a captured browser page.
    - Two resets on the first night had no batch at all, just the screen, NVENC and a stream model job.
    - So the common factor is heavy torch work beside *any* other GPU compute client, helper servers
      included, not two copies of the model.
- **What makes sharing safe: turn-taking on compute, not just budgets on memory.**
  `bonsai_interp/gpu_turn.py`:
  - Both models stay loaded.
  - The foreground (the stream) takes the GPU, and the batch yields at its next batch boundary.
  - Only one process runs kernels at a time.
  - Every compute client belongs in it, including helper servers such as a llama.cpp screen: wrap their
    calls from the client side if the server itself can't take part.
- **Load once.** Every subprocess that reloads the model is a reset risk and costs about 25 s. Serving commands
  from one resident model per session removed both. The stream's `play why` dropped from about 28 s to 3.3 s.
- **Never force-kill a GPU process mid-kernel.** A Stop-Process during scoring was followed 3 s later by another
  reset and a TDR. Stop at batch boundaries (a STOP file).
- **Verify a "release" from the allocator, not from the caller.** A probe reported that it had yielded while still
  holding 4.4 GB. A stale model reference meant a second copy loaded beside it.
- **A smaller helper model frees room.** The stream's chat screener on Bonsai-8B (1.9 GB) instead of Bonsai-27B
  (4.3 GB) returned 2.4 GB to the budget.

**Caps must be enforced where the model loads, not where callers remember them** (2026-10-20, 12:23 reset)
- About 20 modules loaded the model without calling `limit_memory()`. One was the stream's band scan, which
  grew to 9 GB against a 5.7 GB budget.
- An orphaned scan (left behind by a tool timeout) also kept its turn request alive for about 30 minutes. So
  the batch yielded 70% of its run while the card sat above 15 of 16.3 GB, and the driver reset.
- Fixes:
  - `model.load()` applies the cap itself (861621a).
  - Budget for at least 3 GB of free headroom.
  - Make long scoring resumable, so a stop costs minutes, not hours.
- **Transfers:** put resource limits and turn-taking in the one choke point every job passes through (the
  model loader), and treat any process a timeout leaves behind as a resource leak.

**Don't load what the job never reads** (2026-10-20)
- Bonsai-8B's untied lm_head expands to a dense 1.24 GB bf16 matrix. Activation screening never uses it, yet
  it was 40% of the batch's 3.05 GB of weights.
- `model.load(lm_head=False)` (fe17bd1) leaves it on the meta device: about 1.8 GB resident, with identical
  scores. That let the batch fit a 3.6 GB budget beside a live stream.
- **Transfers:** for activation-only sweeps, skip the unembedding, and also any layers deeper than the
  deepest one being read. On a larger-vocabulary or wider model (e.g. 27B), the saving grows with
  vocab × hidden.

**From the stream side** (the Twitch session, 2026-10-20)
- **A load gate checked at process start is too early.** Python imports take 10 s or more, so the gate
  passed before the batch had allocated. Check again immediately before the allocation, or hold a turn for
  the process's whole life.
- **A loader swapped in for an in-process command can pin the model.** It stayed bound in a module that was
  first imported during that command. Restore every loader binding after the run, and check the freed memory
  with the allocator's own count.
- **Keep secrets out of resident services.** The solver that needs an API key (globalMOO) stays a
  subprocess, which holds a GPU turn for its whole life. The long-lived service an agent can reach never
  holds the key.
- **Watchers must survive log rewrites.** A supervisor that rewrites its log on restart breaks a watcher that
  reads only new bytes. Re-read the whole file and take the last marker.
- **Turn-taking in practice:** with a background holding the GPU, a foreground `play why` waited less than
  one background batch, and the background yielded for the whole call (1.9 s).

**Process**
- **One registration for many layers is as valid as one per layer,** and much faster: fewer commits and fewer
  waits. What matters is that the rule and the test sets are pinned before scoring.
- **Keep per-occurrence activations from the screen** (`occ.npz`). A new context filter is then a CPU refit
  instead of another GPU pass.
- **Merge with merge commits, never squash or rebase,** so the registration hashes cited in summaries stay valid.

## Run log

### 2026-10-20 L4 score trial (beside the live stream, gpu_turn on)
- measured: 99 min of wall time for 25,054 of about 29,800 batches (84%), at batch 16, BelowNormal priority, the
  0.35 cap and L4's registered (uncached) scorer. 4,187 s (70%) was yielded to the stream's foreground turns, so
  about 29 GPU minutes did 84% of the work: a full L4 score is about 35 GPU minutes.
- bottleneck:
  - The stream held the GPU 70% of the time.
  - A driver event (nvlddmkm 153, 12:23:29 local) made the watcher create STOP. The scorer exited at the next batch
    boundary, 3 s later, with no kill and no follow-on reset.
  - Scoring writes nothing until a layer ends, so 29 GPU minutes were lost.
- change before the next unit: scoring checkpoints its partial sums every few hundred batches and resumes from
  them, so a stop loses minutes, not a layer (parity check: a stopped and resumed score must equal an uninterrupted
  one).
- gain of the previous change: the graceful stop worked as designed (STOP to exit in 3 s, the turn released).
- memory: the card held more than 15 of 16.3 GB for 70 of the 99 minutes. The reset came at about 15.5 GB, with
  turns on. Near-full memory on the display GPU is the leading suspect.
- further changes before the next unit:
  - Return the allocator's cached blocks (torch.cuda.empty_cache) before every yield, so the stream can use what
    the batch is not holding.
  - A progress line every minute (batches done / total, snippets/s, yield seconds). The trial printed nothing for
    99 minutes.
  - The cap gets sized from the benchmark instead of 0.35: the batch stays at or under about 4 GB, and the whole
    card at or under 13 GB.
- transfers:
  - On a shared display GPU, budget wall time by the foreground's duty cycle (here 70%), not by GPU minutes.
  - Budget memory to leave real headroom on the card (at least 3 GB): a turn scheme stops two processes computing at
    once, not the card filling up.
  - Stops are routine, so every long stage must checkpoint and report progress. Design both in from the start, not
    after a lost run.
