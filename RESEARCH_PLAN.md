# Monosemanticity × globalMOO on Bonsai — Research Plan

Source: Nick Cammarata, *Neural Networks Are Already Clean* (Goodfire, 2026), `paper-contents.txt`.

## What the essay claims (the parts we can test)

- Native MLP neurons in modern LLMs are mostly clean. For Qwen3-32B, an estimated ~90% of neuron **poles** have one main job, ~7% have two related jobs, and 2–4% are not understood.
- A pole is the positive or negative side of the signed SwiGLU product (gate × up). The sign regimes (+×+ and −×−, the "echo") must be split before labelling.
- Apparent polysemanticity is mostly **recursive generalization**: a main job plus locally followable offshoots (iy → Miyazaki → diy → "do it yourself"). It is not superposition-style packing of unrelated concepts.
- Capacity constraints make neurons **broader, not packed**.
- The essay's validation tests:
  - **Blind maximum.** An agent writes a max-activating text from the explanation alone.
  - **New examples vs. near misses.** New examples fired strongly 76% of the time; near misses 11%.
  - **Most-distant concept.** Unrelated text fired strongly about 1% of the time.
  - **Per-bundle blind maximum under ablation.**
- **The gap we can fill:** the essay does not use optimization to search text inputs for LLMs at all. All of its LLM input search is one-shot agent generation. Quantization is never discussed.

## Why Bonsai is interesting

Bonsai is 1-bit or ternary weights, quantization-aware-trained from Qwen bases. That is a different capacity constraint from "fewer neurons": the neuron count is the same, but weight precision is far lower. The essay predicts broader neurons, not packing. Nobody has checked this.

| Model | Arch | Layers × MLP width | Neurons (poles ×2) | Status |
|---|---|---|---|---|
| Bonsai-27B Q1_0 (local) | qwen35 hybrid: 48 Gated-DeltaNet + 16 full-attn | 64 × 17408 | 1.11M | GGUF on disk |
| Bonsai-8B (1-bit) | qwen3 (same family as essay's Qwen3-32B) | 36 × 12288 | 442k | `prism-ml/Bonsai-8B-unpacked` on HF |
| Ternary-Bonsai-8B | qwen3 | 36 × 12288 | 442k | `prism-ml/Ternary-Bonsai-8B-unpacked` |
| Qwen3-8B (FP baseline) | qwen3 | 36 × 12288 | 442k | HF |

The 8B row may be a paired comparison: same architecture and same base, at bf16, ternary and 1-bit precision. If quantization-aware training (QAT) preserves neuron identity, we can compare the *same neuron* across precisions. Verify this with activation correlation.

## Instrumentation

- **Path:** use the HF "unpacked" safetensors, then a custom exact `PackedSignLinear` in PyTorch.
  - Weights are ±scale per 128-group, so this is lossless.
  - Footprint is ~1.2 GB for 8B and ~3.6 GB for 27B, so both fit in 16 GB of VRAM on the RTX 5080.
  - This gives forward hooks on every `act_fn(gate) * up` and gradients for baselines. No llama.cpp rebuild and no CUDA toolkit are needed.
- **Fallback for the local 27B GGUF:** dequantize Q1_0 directly with `gguf` into the same packed module.

## Experiments

**E0. Infrastructure.**
- Activation server with hooks, run over a corpus (FineWeb-Edu + code + multilingual, long context ≥2k tokens).
- Store per-pole top-k examples with gate and up signs.
- Use 20–50M tokens to start.

**E1. Replicate the audit on Bonsai.**
- Take 200 random neurons. For each one, get top positive and negative examples and have Claude agents write explanations.
- Score every neuron on the essay's protocol:
  - 5 new examples.
  - 5 near misses.
  - 5 most-distant-concept texts.
  - Strong = ≥30% of the corpus max.
- Report the share of poles with 1, 2 or N jobs, and the size of the echo regime.
- Compare against the essay's Qwen3-32B numbers.

**E2. Precision ladder (the new result).**
- Run E1 on the same neuron indices across Qwen3-8B, Ternary-Bonsai-8B and Bonsai-8B.
- Measure neuron identity preservation and breadth: how many distinct explanation clusters are in the top-k, and the selectivity curve.
- The essay predicts broadening without packing.

## Forwards and backwards: neurons as inverse problems

**Forward interpretability** (the essay's method) runs inputs, sees what fires, and explains the neuron.

**Backward interpretability** asks: given a target state of the neuron, what antecedents produce it? The antecedents can be inputs, upstream neurons, heads or residual directions.

An inverse model that learns this map "surgically" does three things:
1. It reaches the neuron's boundary in a few hundred targeted evaluations instead of 10⁸ corpus tokens. That is the speed claim.
2. Its solutions are causal, because they come from interventions, not correlations.
3. It enables a *round-trip test*: `forward(inverse(target)) ≈ target`, calibrated across target levels. That is the convincingness claim.

When the forward explanation and the backward explanation (the concepts in the inverse solutions) agree, the explanation is cross-validated from both directions.

### The antecedent chain of one SwiGLU neuron

The neuron is `a = silu(g)·u`, with `g = w_g·x̂` and `u = w_u·x̂`, where `x̂ = γ⊙x/rms(x)`. The residual `x` is the sum of upstream writes: earlier neurons `a_m·d_m`, attention heads, and the embedding. Each level of the chain has a different mathematical structure, so the right solver differs by level.

| Level | Antecedent space | Structure | Right tool | Why not the others |
|---|---|---|---|---|
| **L0 Local pre-image** | normalized residual x̂ ∈ R⁴⁰⁹⁶ | Two linear constraints (g\*, u\*); the target `a*` is a hyperbola in (g,u) with sign regimes | **Closed form**: min-norm solution plus null space. No optimizer needed | Anything iterative is wasted here |
| **L1 Direct upstream (weights only)** | activations of upstream neurons and heads, bounded by observed ranges | Linear given rms: `g = Σ V_nm a_m`, with virtual weights V | **`scipy.optimize.lsq_linear`** (bounded), **`nnls`**, **`milp`** (minimum-support set via binary indicators and big-M from corpus ranges) | Exact and data-free, so black-box search adds nothing |
| **L2 In-context causal (interventions)** | gains θ∈[0,1]ᵏ on k upstream bundles or neurons, on real inputs | Smooth, differentiable, cheap (~50 ms per eval). Objectives conflict: fidelity vs sparsity vs KL | Single target: **torch autograd + `scipy.optimize.minimize` (L-BFGS-B)** or **`least_squares`** (hit an exact activation vector on several neurons). Pareto front or non-smooth sparsity (L0 counts): **globalMOO** with k ≤ ~30 bundles | Gradients are free here. globalMOO earns its place on the multi-objective and inverse-target queries |
| **L3 Residual / soft-input** | coefficients over a k-dim concept basis (upstream neuron write directions or PCA directions) | Continuous and differentiable, but off-manifold risk | Many dims: **torch Adam**. Few dims (k ≤ 30) with an inverse map over many targets: **globalMOO**. Check with `scipy.optimize.least_squares` | High-dim black-box is a poor fit for globalMOO |
| **L4 Text / tokens** | tokens, or a structured generator of semantic slots (topic, language, cue word, cue position, register, distractor, context length) | Discrete and non-differentiable; each eval may include a judge or embedding call | Interpretable factor space: **globalMOO** (categorical/int inputs, multi-output, inverse targets, sample-efficient learned surrogate). Baselines: **`scipy.optimize.differential_evolution`** (with `integrality`) and `dual_annealing`. Raw tokens: **GCG / EPO** (gradient-guided discrete). Essay baseline: LLM-agent blind maximum | Gradients only reach tokens via GCG approximations. Factor space is where an *interpretable* inverse lives |

**Rule of thumb:**
- If the map is linear, solve it exactly with SciPy.
- If it is smooth with one target, use gradients with SciPy local solvers.
- If it is multi-objective, needs inverse targets, is categorical, or is a black box with expensive evals, use globalMOO.
- On each problem, use SciPy's global black-box solvers as the head-to-head baseline for globalMOO.

### Budget protocol: screen locally, promote few to globalMOO

- **Tier 1, bulk screening (local only).**
  - The forward scan.
  - L1 linear and MILP maps.
  - L2 circuit inversions with `least_squares` / L-BFGS-B / `differential_evolution` across ~200 neurons.
  - Cheap and unlimited. Produces the screening metrics and picks the candidates.
- **Tier 2, careful globalMOO runs (a few neurons, one at a time).**
  - Choose 2–4 neurons that the screen flags as most informative. For example:
    - one clean neuron;
    - one with visible recursive generalization;
    - one dominated by the echo regime;
    - one that changes most across the precision ladder.
  - For each, build one globalMOO model on one well-posed problem, train it once and query it many times (target levels, regimes, constraints).
  - Before the run, write down the hypothesis it tests. Afterwards, record what it showed.

### Experiments, by level

**E3. Weights-only backward map (L0–L1).**
- For each neuron, solve the L1 inverse: the minimum upstream set (MILP) that can drive each pole to 30%, 60% and 100% of its corpus max within observed upstream ranges.
- Compare that set's labels with the forward explanation.
- This is a data-free, seconds-per-neuron backward explanation. Does it already predict the forward one?

**E4. Causal circuit inversion (L2).** This is the optimizer version of the essay's per-bundle blind maximum.
- Run on the neuron's top corpus examples.
- Cluster attribution into bundles, then find the Pareto front of {pole retained, bundles kept, next-token KL}.
- Use globalMOO for the front and L-BFGS-B for single scalarized points, and compare the solutions and the evaluations each method needed.

**E5. Inverse response mapping and recursive generalization (L4).**
- Train globalMOO on the slot generator. Query inverse targets at 25%, 50%, 75% and 100% of max for each pole, and for the echo regime.
- Each target returns the factor settings that achieve it. The neuron's branches (iy → Miyazaki → diy) become explicit regions of factor space.
- Report round-trip fidelity: achieved vs requested activation.

**E6. Adversarial monosemanticity (L4, multi-objective).** This is the essay's "most-distant concept" test turned into a search.
- Objectives:
  - Pole ≥30% of max.
  - Maximize semantic distance from the forward explanation.
  - Naturalness (perplexity).
  - Opposite pole quiet.
- Solvers: globalMOO vs `differential_evolution` vs GCG/EPO vs the agent blind maximum.
- An empty "far & strong" region is evidence of monosemanticity. A populated one gives candidate packing.

**E7. Steering inverse design (L2/L3, stretch).**
- Find gains on k labelled neurons that hit a behavioural target vector, such as logits of target tokens or a judged topic.
- This tests whether labels compose.

### Metrics for "faster, cleaner, more convincing"

- **Evals-to-explanation:** model forward passes needed by the backward method vs tokens scanned by the forward method to reach the same explanation quality, scored on the essay's new-example and near-miss test.
- **Round-trip fidelity:** the curve of achieved vs requested activation, per neuron and pole.
- **Forward/backward agreement:** a blinded judge decides whether the two explanations name the same job.
- **Per-solver efficiency** on identical problems: the best objective reached vs the number of evaluations.

## Deliverables

- `bonsai_interp/` pipeline.
- A neuron-paper browser (artifact).
- A write-up comparing against the essay's numbers.
