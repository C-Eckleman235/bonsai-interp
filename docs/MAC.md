# Running bonsai-interp on a Mac

The toolchain runs on Apple Silicon through PyTorch's MPS backend, with the same 2-bit packed
weights and the same bfloat16 arithmetic as on CUDA.

- **Requirements.** An M1 or later Mac with macOS 14 or later. The macOS torch wheels need
  macOS 14, and so does bfloat16 on MPS. We recommend 32 GB or more of unified memory. 16 GB
  works but is tight: one process holds about 8 GB (the 4.5 GB model, the 1.1 GB scan and the
  batches), next to macOS and the agent CLI. 8 GB is not enough. You also need about 8 GB of
  disk.
- **Install.** Use the commands in the README's [quick start](../README.md#quick-start). `uv sync` takes torch from PyPI on macOS, and
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
  model in 15 s. A 24-core Intel Core Ultra 9 CPU does 99 tokens/s, about 36 times slower. A
  base Apple M5 with 16 GB (macOS 27, torch 2.14.1) does 417 tokens/s and loads the model in
  9 s, about 9 times slower than the reference, and passes `tests/test_device_parity.py`
  (L9/N3688+ reads 0.495 against 0.497). On that M5, a card for 16 poles took 11 s, and
  `play run` for a 16-pole sweep took about 2.5 minutes including the model load. 16 GB was
  enough with one model-loading process at a time.
  That is about twice as fast as we estimated for a base chip, so the other estimates are
  probably pessimistic too: we expected about 4 to 6 times as long on an M-series Max or
  Ultra and about 10 times as long on a Pro, but have not measured those. Only the model work
  scales by that factor; the agent's own reading and writing time does not. Blind rooms suit a
  Mac better than live play, because nobody waits on them. Please report what doctor prints
  on your Mac so we can replace the remaining estimates.
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

Back to the [README](../README.md).
