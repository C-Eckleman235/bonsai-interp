# The weights-only census

The census is stage one of the deterministic harness (docs/HARNESS.md). It gives every pole of Bonsai-8B (36 layers ×
12,288 neurons × 2 poles) an output-side description before any corpus text is read for activations. The only corpus
input is the unigram token counts. No language model writes or judges any part of it.

Code: `bonsai_interp/census/` (weights.py, connect.py, lens.py, run.py, nulls.py, summary.py). Data: `data/census/`
(release `census-data`), never in git.

```
python -m bonsai_interp.census.run all       # GPU, one batch-slot hold: weights, links, steer lens (about 30 min)
python -m bonsai_interp.census.nulls         # CPU, from the GGUF: random-direction nulls (about 10 min)
python -m bonsai_interp.census.summary       # CPU: per-pole table, flags, summary, validation
python -m bonsai_interp.census.summary pole bonsai1/L35/N123-   # one pole's entry, tokens decoded
```

## What it measures

Write W = W_U · diag(g) for the effective unembedding (g is the final-norm gain), centred over the vocabulary. Write
w = W_down[:, n] for a neuron's output column. The − pole writes −w.

1. **Direct logit effect.** The top 20 tokens of ±W w and the spike: how far the 20th token stands above the vocabulary
   mean, in standard deviations. This is sweeplib's `predict.direct_effect`, imported unchanged. The census output is
   identical to sweeplib@1's `effect.npz` (36 of 36 layers, every token id and spike).
2. **Entropy neurons** (Stolfo et al. 2024). The share of w in the unembedding's effective null space. That space is
   the eigenvectors of WᵀW below 0.1 × the median singular value. In Bonsai-8B it has exactly **2 directions**
   (0.007 and 0.009 × the median), then a gap to 0.27. A softer 40-direction share is also kept.
3. **Token-frequency neurons.** The Pearson correlation, over the tokens seen in the corpus, between W w and the log
   unigram count. Computed in closed form (u·w / ‖f‖ √(wᵀC w)), so no V × N matrix is formed.
4. **Steer lens** (after lyra's neuron-explorer):
   - The push is added at the last position after layer L's MLP, equivalent to adding it to down_proj's output.
   - It runs on 4 short neutral contexts (lens.py `CONTEXTS`).
   - Dose: α_L × ‖h_L‖, the residual norm at that position, along w/‖w‖, with + for the + pole and − for the − pole.
   - α_L is calibrated per layer so that 64 random unit directions reach a median KL(p0‖p) of 0.1 nats.
   - Recorded: the mean KL, the 10 tokens with the largest context-mean Δlog-prob, and the same for 256 random
     directions at the same dose.
   - Only the last position changes, so the prefix keys and values are computed once. A push is then one single-token
     pass through layers L+1..35, and 256 neurons × 2 signs × 4 contexts run as one batch.
5. **Weight connection matrix.**
   - P[B, A] = G·U, where G = W_gate_B (ĝ_B ⊙ v_A) / r_B and U = W_up_B (ĝ_B ⊙ v_A) / r_B. ĝ_B is B's MLP-input norm gain
     and r_B is the typical RMS of the residual there (from the lens contexts).
   - P is the part of B's product that A drives alone. It does not depend on A's sign, and its sign says which pole of
     B is pushed.
   - G and U are stored, so a reader can form the sign-aware first-order effect at an operating point:
     Δact_B ≈ (silu'(g0)·u0·G + silu(g0)·U)·a_A.
   - Kept: the 24 readers with the largest |P| per neuron, and the 24 writers with the largest +P (+ pole) and −P
     (− pole) per neuron.
   - Attention and the intermediate norms are ignored: these are direct residual-stream links only.

## Nulls

Every statistic is compared against random directions. Two kinds of null are used (`nulls.py`):
- **isotropic:** Gaussian unit vectors;
- **sign-random:** the layer's real per-(row, 128-group) scales with random ±1 signs. This is what a random neuron
  would write in this 1-bit model.

In Bonsai every down_proj column is ±scale[row, group], so a neuron's output norm is set by its 128-neuron group (the
coefficient of variation within a layer is about 1%). The sign-random null keeps that structure.

## Results (Bonsai-8B, 2026-10-26)

Flags per pole, out of 884,736. Each flag is set against the layer's sign-random null:

| flag | rule | poles |
|---|---|---:|
| predict | direct-logit spike above the null's 99.9th percentile, **and** the steer push writes those same tokens (agreement ≥ 0.3) | 107,486 (12%) |
| steer | the push moves the next-token distribution ≥ 3 standard deviations beyond random directions of the same dose | 88,172 (10%) |
| frequency | \|r\| with log unigram frequency ≥ max(0.2, the null's 99.9th percentile) | 550 |
| entropy | null-space share ≥ 2 × the null's 99.9th percentile | **0** |

By depth:

| layers | predict | steer | frequency |
|---|---:|---:|---:|
| 0–5 | 4 | 15,182 | 4 |
| 6–11 | 271 | 14,363 | 2 |
| 12–17 | 1,099 | 9,277 | 0 |
| 18–23 | 14,312 | 12,099 | 0 |
| 24–29 | 41,501 | 19,174 | 12 |
| 30–35 | 50,299 | 18,077 | 532 |

Findings:
- **Prediction neurons live late.** Output-side prediction appears from about layer 18 and reaches a third of the
  layer 30–35 poles. Early poles steer the model too, but not toward the tokens their weights promote directly.
- **No classic entropy neurons.** Bonsai's unembedding has only 2 near-null directions (singular values 0.007 and
  0.009 of the median), and no neuron writes into them beyond the null.
- **Frequency neurons** are a small late group: 532 of the 550 are in layers 30–35.

Validation: a flagged share for each known group, against random non-weak poles from the same layers.

| group | n | predict | random | steer | random |
|---|---:|---:|---:|---:|---:|
| sweeplib@1 prediction tests that held | 101 | 48% | 15% | 16% | 10% |
| sweeplib@1 prediction tests that failed | 1,292 | 28% | 13% | 27% | 10% |
| survey labels worded as prediction | 85 | 48% | 21% | 24% | 12% |
| survey labels worded as frequency | 45 | 20% | 15% | 4% | 11% |
| all held survey labels | 1,715 | 14% | 13% | 13% | 10% |
| survey dense / positional | 100 | 2% | 8% | 9% | 10% |

So the predict flag picks out known prediction neurons at about 2.3–3.2 times the base rate. The frequency flag does
not recover the survey's frequency-worded labels, which describe word frequency in the input rather than the output.

**FINDINGS paragraph.** A weights-only census of all 884,736 poles:
- **12% are prediction-type.** Their direct logit effect stands above a sign-random null, and pushing them writes the
  same tokens. They rise from 0% in layers 0–11 to 34% in layers 30–35.
- **10% steer the output strongly.**
- **550 are frequency-type,** almost all in the last six layers.
- **There are no entropy neurons.** The model's unembedding has only two near-null directions.
- **Validation:** the predict flag finds the sweep library's held prediction tests at 3.2× and survey prediction labels
  at 2.3× the base rate.

