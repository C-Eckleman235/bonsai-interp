# float16 for missions (GPUs without bfloat16)

**Why this matters.** Free Colab T4s have no bfloat16 tensor cores. The toolchain's default, bfloat16, runs about 5×
slower there than float16 would. `bonsai_interp/env.py` warned that float16 could overflow Qwen3's large residual
activations.

**Question.** With `BONSAI_DTYPE=float16`, do survey readings and test outcomes match the bfloat16 reference closely
enough for contributors to use it?

**Script:** `scripts/fp16_parity.py`. **Outputs:** `data/fp16/` (not in git).

## Method (RTX 5080, 2026-10-25)

1. **Re-scored tests.** A stratified sample of 150 registered quick-8+8 tests from graph/contributions was re-scored,
   half held and half failed, spread over layers 0–35. Scoring was exactly as `play run` does it: TextEvaluator,
   3 preambles, max over the text's tokens, fraction of the pole's max. That makes 2,400 sentence readings per dtype.
   Each dtype ran in its own process.
2. **Full windows.** 32 full 1,024-token corpus windows were read through all 36 layers in both dtypes: 28.3 million
   (token, neuron) cells. Overflow was checked at every residual, attention and MLP output.
3. **Reference noise.** bfloat16's own batch-shape noise is the yardstick: the same sentences read at another batch
   size.

## Results

| sentence readings (fraction of pole max) | max | 99.9% | 99% | mean |
|---|---:|---:|---:|---:|
| float16 vs bfloat16 | 0.049 | 0.016 | 0.007 | 0.0011 |
| bfloat16 vs itself at another batch size (noise floor) | 0.025 | 0.012 | 0.006 | 0.0010 |
| bfloat16 re-run vs the recorded values (3 d.p.) | 0.014 | 0.011 | 0.006 | 0.0009 |
| float16 vs the recorded values | 0.049 | 0.012 | 0.007 | 0.0011 |

- **Outcomes.** float16 and bfloat16 agree on all 150 tests (held or failed). Both agree with the recorded outcome on
  149. On the remaining one (L32/N8999−, AUROC 0.91 against a 0.90 bar), both dtypes give the same new outcome: device
  drift, not dtype.
- **Replication.** CI's replication verdict comparing float16 with bfloat16 is "ok" for 149 tests and "drift" for 1,
  which still passes. The peak token agrees on 98.8% of sentences.
- **No overflow.** No non-finite value appeared in either dtype. The largest activation anywhere is 38,528 (layer 35's
  residual, the same in both), against float16's limit of 65,504: a margin of about 1.7×.
- **Full windows: the caveat.** Over 28 million cells, the 99.9th percentile difference is 0.011 of a pole's max.
  Rare cells in layers 16–35 reach 0.19–0.57; it is under 0.04 in layers 0–15. Single cells far out in long contexts
  can move, so long-window scans and the corpus screens should stay in bfloat16.
- **Speed and memory here.** float16 scoring of the 150 tests took 247 s of GPU time and peaked at 4.6 GB. The T4
  speedup cannot be measured on this card; it is the motivation, from the hardware's tensor-core support.

## Recommendation

- **float16 is allowed for missions on GPUs without bfloat16** (`BONSAI_DTYPE=float16` in `.env`). Its differences are
  within about twice bfloat16's own batch-shape noise, test outcomes are unchanged, and nothing overflows.
- `play doctor` now says so on such GPUs (compute capability below 8.0, e.g. a T4).
- `play run` records the dtype in a contribution's `methods.dtype` when it is not bfloat16.
- Keep the corpus screens, the scan and long-window work in bfloat16, the reference dtype.

**FINDINGS paragraph (2026-10-25).** float16 matches bfloat16 for survey tests. 150 re-scored quick-8+8 tests agree
on every outcome. Sentence readings differ by at most 0.049 of a pole's max (99.9th percentile 0.016), against
bfloat16's own batch-shape noise of 0.025. No value overflows: the largest activation is 38,528 against float16's
limit of 65,504. Rare cells deep in long windows move by up to 0.57, so the scan and screens stay in bfloat16, but
T4 contributors can run missions in float16 (docs/FLOAT16.md).
