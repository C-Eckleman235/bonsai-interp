# Findings log

## 2026-10-17: Neuron identity survives quantization-aware training (mid layers)

**Setup** (`python -m mono.identity --windows 12`)
- 12,288 tokens from the corpus tail, run through all three models.
- Neuron = SwiGLU product channel.
- Metric 1: same-index Pearson r.
- Metric 2: share of neurons whose best-correlated neuron in the other model is the same index.

| pair | L1 | L5 | L9 | L13 | L17 | L21 | L25 | L29 | L33 | L35 |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B ~ Ternary, best match = same idx | 43% | 99.1% | 99.9% | 99.8% | 99.4% | 99.6% | 99.5% | 98.7% | 97.6% | 78.5% |
| Qwen3-8B ~ 1-bit | 40% | 94.6% | 98.7% | 97.2% | 95.3% | 94.7% | 92.0% | 84.2% | 80.7% | 47.2% |
| Ternary ~ 1-bit | 43% | 99.7% | 99.9% | 99.7% | 98.3% | 99.4% | 99.2% | 96.6% | 96.8% | 82.5% |

**Reading the table**
- Bonsai's quantization-aware training (QAT) keeps the base model's neuron basis. In layers 5–33, the same index is the same neuron, so the precision ladder is a paired comparison.
- Same-index Pearson r is only moderate (median 0.25–0.54). It is dominated by the bulk of weak activations, and the tail metric is the one that matters for the essay's audit; it is still to be measured.
- Identity erodes in late layers, most at 1 bit. At L35, 1-bit vs FP gives 47%. This is where the essay's "say vs think" / token-suppression neurons live.
- L1 matches poorly at every precision pair. This may be an artifact: early neurons are token-specific and sparse, and 12k tokens may be too few. Re-check with more tokens and a tail-overlap metric.

**Perplexity** on a 1,024-token essay passage: Qwen3-8B 28.7, Ternary 32.7, 1-bit 36.7.

## 2026-10-17: Solver behaviour on a toy SwiGLU neuron (8 inputs)

The toy is `a = silu(w_g·x)·(w_u·x)`, with x ∈ [-1,1]⁸. All solvers ran against the same `InverseProblem`.

| goal | globalMOO (after 181-pt DOE) | differential_evolution | least_squares + autograd |
|---|---|---|---|
| a ≈ 1.0 ±1% (inside DOE range) | ✔ 5 evals | ✔ 704 | ✗ 0.99, just outside the band (7) |
| a ≈ 3.0 ±1% (beyond DOE max 1.84) | ✗ stalled at 1.84 | ✔ 704 | ✔ 7 |
| a ≈ -0.2 with g ≥ 0.5 (leak regime) | ✔ 1 (already in DOE) | ✔ 704 | ✔ 11 |
| maximize a | 1.92 (60) | 3.53 (704) | n/a |
| a ≈ 1.0 & ‖x‖₁ ≤ 2 | ✗ 0.61 | ✗ 0.84 | ✗ 0.88 (may be infeasible) |

**Reading the table**
- globalMOO is a learned surrogate. It interpolates inverse targets inside its trained envelope in a handful of evals, and once trained, each query is cheap ("train once, query many"). It does not extrapolate past the DOE range.
- So for *maximization* or out-of-envelope targets, use gradient methods or DE, or re-train globalMOO on a zoomed box around the best region.
- For *mapping a neuron's response surface at many target levels*, globalMOO is the right tool.

**Integration bug, fixed:** globalMOO PERCENT bands around an initial value of 0 collapse to exactly 0 and pin that output. Untargeted outputs now use absolute VALUE bands.

## 2026-10-17: Pilot, 12 random Bonsai-8B (1-bit) neurons, read forwards and backwards

**Forward reading** (`runs/pilot_interps.json`)
- Corpus: 8M tokens. Read the top 10 examples per pole for 12 random neurons from layers 8–28, 24 poles in all.
- Result: 19 of 24 poles show one main job, 1 shows two related jobs, and 4 are not yet understood.
- Many jobs hold across languages:
  - "nerve" in English, German, Korean and Arabic;
  - the afterlife ("heaven", "Paradise", الدنيا والآخرة);
  - "best" in competitions.
- The two poles are usually related (afterlife / God's presence; payment terms / debt collection). L10/N5749 is the exception: genre (+) vs refuelling at gas stations (−).
- No echo (−×−) regime appears among strong activations for any of the 12 neurons.
- Caveat: this is a first reading by Claude, not blind-tested.

**Backward screen** (`data/screen_bonsai1.jsonl`, local only, no globalMOO)
- Method: gains on each pole's top 24 upstream neurons by attribution, using its top example, with context capped at 384 tokens. The cap keeps 91–103% of the full-window activation.
- **Round trip.** Whenever the target is reachable, `least_squares` lands within about 0.04 of the requested fraction in ≤25 evaluations. Targets below the zero-all-24 floor are unreachable by construction.
- **Concentration varies widely.** Zeroing the top 24 leaves anywhere from 0.00 (printf pole) to about 1.0 (several poles whose input comes through attention or is spread out).
- **Keep-curves.** Keep only the top k upstream neurons and zero the rest:
  - L9/N3688+ ("nerve"): 0.37 at k=7, jumping to 0.98 at k=8. One unit (L7/N384, N-word subword completion) is pivotal.
  - L13/N10734+: 0.41 at k=1 and 0.78 at k=2.
- **Antecedent chains** (`runs/pilot_antecedents.json`):
  - "genre" = the token "genre" + the subword "gener-" + typology + literary forms;
  - printf = `{}` placeholders + `%s` specifiers + `/%`;
  - the "nerve" pole is fed by an Arabic nerves/neural neuron.
- **Lesson 1: sample each branch.** The top example of the "best" pole is its "all … in common" offshoot, so the backward screen described the offshoot. Run the inverse per forward cluster.
- **Lesson 2: the route-sharing metric is too strict.** It compares raw attribution vectors over 442k units at cos > 0.5, and nearly every pole came out as 8 separate routes. Group into labelled bundles before comparing.

**Infrastructure fixes**
- Autograd kept every dequantized weight alive, which is the whole model in bf16. A custom backward now re-dequantizes instead.
- Screens set a GPU memory cap, so they fail instead of spilling into shared memory.

## 2026-10-17: Neuron probe (forward + inverse + induce) on L9/N3688, Bonsai-8B 1-bit

Tool: `python -m mono.probe.run {forward|solo|invert|induce} 9/3688 +`. All artifacts are in `probes/bonsai1_L9_N3688_pos/`.

**Forward**
- Top corpus examples suggested H1: "the morpheme *nerve* across languages".
- A bank of 98 natural sentences, rated by meaning before evaluation, showed that H1 over-claims: anatomical "nerve" is weak.
- H2: *nerve(s) as psychological strain or composure*.

**Position confound (methodology)**
- Bare sentences put the key token at sequence positions 1–10, where this neuron is damped.
- The tool now evaluates every text after neutral preambles (mean of 3), reading only the text's tokens.
- This changed conclusions, so it is now the default.

**Pre-registered held-out test of H2** (24 sentences written before evaluation, with fire/quiet predictions)
- Accuracy 0.79, AUROC 0.92.
- 7 of 12 predicted-fire sentences fired (≥30% of corpus max); 0 of 12 predicted-quiet fired.
- Misses: French "les nerfs", Korean "신경을 쓰다", "nerves of steel", "nervous energy", "nerves were shot". All were weaker than predicted, never false positives.

**Text-level inverse** (3-sentence passages on the bank manifold; inputs are concept and strength coordinates per slot)
- Graded targets: 25 / 50 / 75% met by both solvers. Context can raise a sentence from 0.17 alone to 1.08.
- Leak, i.e. firing with nothing nerve-related: only differential evolution found one, 0.40 on " strong" ("A strong breeze…" after a violent sentence). This is plausibly the fortitude branch.
- Offshoot: anatomical "nerve" fires 0.33–0.60 in non-initial positions (神经, нерва). The Korean name 신경민 fires 0.51, a token-identity offshoot.
- Sense flip: "She finally worked up the nerve" scores 0.72 alone and 0.08 after a pain-receptor sentence, while the opposite pole jumps to 1.56. The neuron resolves the sense of "nerve" from context.

**globalMOO vs SciPy differential_evolution** (one globalMOO model, trial 17552; plan pre-registered in `gmoo_plan.json`)

| | globalMOO | differential_evolution |
|---|---|---|
| Model evaluations for the 8-goal battery | 268 (161 shared design points + queries) | 4,992 |
| Goals met | 6/8 (max 0.84; missed leak and near-miss) | 8/8 (max 0.97) |
| Tight ±0.015 targets from a neutral start | 25% in 17 evals; 50% and 75% within 0.02–0.03 after 41 evals | all three at 624 evals each |

- Caveat: 5 of globalMOO's goals were met by its best design point. The tight-target queries are the fair test of its learned inverse.

**Forward-induce** (add c × corpus max to the neuron at every position)
- c = −2 to −40: gains pain / heat / warmth, then stimulus / receptor / reflex. "I felt" continues "the pain" instead of "nervous".
- c = +2 to +10: gains calmly / inspired / strongly / powerful / proud.
- Above +20: text degrades. Concept tokens never rise, which fits a layer-9 detector.

**Verdict:** one context-resolved axis, psychological strain/composure (+) versus bodily sensation (−), with local offshoots (anatomical sense, the 신경 token in names, "strong"). No unrelated concept was found by the forward, inverse or induce views.

## 2026-10-17: Fan-out from L9/N3688+ ("nerve"): neighbourhood and circuit inverse

`python -m mono.probe.neighbors 9/3688 +` uses 12 branch examples (6 psychological, 4 anatomical, 1 name, 1 "strong"), all read after a neutral preamble.

**Upstream (single-unit zero ablation; share of the pole's activation lost)**
- **Narrow fan-in, two dominant inputs:**
  - L3/N6493, 74%: the token "nerve" (nerve, Nerven, ner-ve). Loses 95% on anatomical examples and 77% on psychological ones.
  - L8/N7756, 61%: the nerve/neural concept, with Arabic عصب top examples.
- **Minor inputs:**
  - L7/N1548 (veins), L5/N5628 (enceph-), L3/N6185 and L0/N11629 ("neuro-" prefix, mostly the name branch), L0/N1684 ("serve", the "-erve" spelling), L6/N1460 (numbness/tingling).
- The "strong" branch uses none of these.

**Sense-flip disambiguators** ("worked up the nerve" alone vs after a pain-receptor sentence; nerve goes from +6.5 to −9.4)
- L8/N3858, a pain detector: zeroing it alone restores 25% of the gap.
- L6/N2055, pain across languages (Schmerzen, doler, وجع).
- Zeroing the top 40 candidates together restores 65%. The rest presumably comes through attention.

**Downstream (zero the nerve neuron, change in later neurons relative to their corpus max)**
- **Wide, weak fan-out.**
  - Psychological branch: L10/N6901 (nervous system / neuroanatomy) moves +0.115; L14/N9470 (emotional disappointment) +0.06; the rest are under 0.06.
  - Anatomical branch: different readers, all under 0.05.

**Circuit inverse** (5 outputs: nerve, token, concept, pain, reader; same natural-passage manifold; `circuit.json`)

| goal | DE (624 evals each) | globalMOO (161 design points + ≤41 per goal) |
|---|---|---|
| bypass: nerve ≥ .3, token ≤ .1 | ✔ 0.61 via the concept route (Arabic "nervous system") | ✗ 0.29 |
| gated: token ≥ .5, nerve ≤ .1 | ✔ with pain 1.04 (sense flip) | ✔ in 10 evals, **pain 0.02**: "He lost his nerve" |
| override: nerve ≥ .5, pain ≥ .5 | ✔ 0.61 via the concept route next to "chronic pain" | ✗ 0.28 |
| concept without token | ✔ (same passage as bypass) | ✗ near miss (concept 0.50, nerve 0.29) |
| reader with nerve quiet | ✗ best 0.40 | ✗ |
| clean max (pain ≤ .1) | 1.06 | 0.37 |

**Reading (hypothesis-grade)**
- Two parallel input routes. The token route (Latin script) is sense-gated: pain context closes it, and so does the "lost his nerve" construction (counterexample found by globalMOO).
- The concept route (non-Latin scripts) looks sense-agnostic.
- The pole's psychological specificity is a property of the token route and its gates.
- Solver note: on this 5-output, rare-combination problem DE beat globalMOO on 5 of 6 goals. globalMOO's one success produced the most informative counterexample.

## 2026-10-17: Pre-registered gate/route test for L9/N3688+ (`gate_test.json`, committed in 9613a67 before the run)

Each sentence was read after [preamble] + context, under neutral and pain contexts (2 contexts × 3 preambles each). Nodes were measured on the sentence's own tokens. Result: **9 of 11 predictions pass.**

| prediction | result |
|---|---|
| A open Latin constructions fire (≥ .3) | ✔ 7/8 |
| A pain context halves them (ratio ≤ .5) | ✔ 6/8 (e.g. "Ich brauchte starke Nerven" 0.72 → 0.03) |
| A pain unit ≥ .3 on the sentence's tokens | ✗ 0/8. The pain unit fires on the context sentence, not on the test sentence. The gate is carried forward, presumably by attention. |
| B "lost his nerve" / "had the nerve to": token ≥ .4 | ✔ 6/6 |
| B …while the pole stays < .3 | ✔ 6/6; mean 0.19 vs 0.60 for set A |
| C Arabic concept unit ≥ .3 | ✔ 4/4 (token unit 0.00) |
| C Arabic fires in either sense | ✔ 3/4 |
| C Arabic not gated by pain (ratio ≥ .5) | ✗ **0/4**. Ratios 0.07–0.37: pain closes the concept route too. |
| D Latin anatomical controls: token present, pole weak | ✔ 4/4, ✔ 4/4 |

**Exploratory (unscored)**
- "kept his nerve" fires (0.41); "lost his nerve" does not (0.14).
- "What a nerve he has" fires (0.61); "had the nerve to" does not (0.14–0.25).
- Korean 신경 fires ~0.31–0.37 in both senses. Chinese 神经 does not fire (0.02–0.04).

**Correction to the earlier circuit reading**
- The "override" example (Arabic beside "chronic pain" at 0.61) had the pain sentence *after* the Arabic one. A causal model cannot be gated by later text, so that example never tested the gate.
- Revised story: both input routes are gated by *preceding* physical-sensation context. The token route is additionally construction-sensitive: open for "worked up / kept / mustered the nerve", "nerve-racking", "Nerven blank"; closed for "lost his nerve" and "had the nerve to".

## 2026-10-17: Probe calibration on three neurons, and neighbour mining for L9/N3688

**Protocol**
- Every neuron got the same probe.
- Banks for the two calibration neurons were written by separate agents that saw only corpus examples.
- At most one label refinement is allowed before the held-out test. The L22 refinement was done by an agent that saw the bank results.
- Every held-out set and globalMOO plan was committed before it was run (commits 3977f9d, 103f75a, 973fd76).

| measure | L9/N3688 nerve | L10/N5749 genre | L22/N11024 unclear |
|---|---|---|---|
| held-out fire rate (label-written text fires ≥ .3) | 58% | **75%** | **33%** |
| held-out false positives | 0% | 0% | 0% |
| held-out AUROC | 0.92 | 1.00 | 0.92 (after refinement) |
| opposite pole, on-label fire rate | 82% | 89% | **0%** |
| cross-talk (opposite pole on the pole's concept) | 0.21 (related axis) | 0.005 (unrelated) | 0.02 |
| leak found by the DE inverse | 0.40 ("strong") | 0.29 (none) | 0.23 (none) |
| induction | coherent both ways | coherent both ways (fuel; genre / literary / tradition) | incoherent (relief, glad / on, still), ~6× weaker KL |
| globalMOO 25 / 50 / 75% | ✓✓✓ | ✓✓✓ in 7–10 evals | ✓✗✗ |

**Reading**
- The probe's verdicts differ as intended: one context-gated axis, two clean unrelated poles, and no single coherent job.
- No single number does it. AUROC is too lenient: a refined label can rank sentences without making the neuron fire. Low leakage is not cleanliness: the unclear neuron barely fires on anything.
- The discriminating evidence is the held-out fire rate (the essay's own new-example test), opposite-pole coherence, induction coherence and strength, and cross-talk (related vs unrelated poles).
- The unclear neuron's refined label did find real structure. Hypothetical money outcomes ("would have roughly $200,000") reach 0.4–1.0.

**Neighbour mining, L9/N3688** (`mono/probe/mine.py`)
- Method: 115 top corpus sentences of 12 causal neighbours (6 inputs, 2 gates, 4 readers), rated for relevance before evaluation (commit 3977f9d).
- 6 fired the pole off-label:
  - 3 Arabic anatomical sentences, the known concept-route offshoot;
  - "nerve poison";
  - a symptom list peaking on "tingling", a small physical-sensation leak;
  - one unexplained (0.33).
- None of the 40 sentences from downstream readers fired it.
- No unrelated alternative reading was found. The mining confirms the known offshoots.

**New views**
- The iris plot: every probe sentence placed around the neuron by type (angle) and label relevance (radius), coloured by which pole it drives. It is in every dossier and on the calibration page (`reports/calibration.html`).

## 2026-10-17: Quick checks of nerve's four key neighbours (`mono/probe/quick.py`; tests committed in d56f984)

Each neighbour got: weights-only direct token drivers; "one layer back" (attribution on its own top corpus examples, then zeroing each candidate feeder); and 8 fire / 8 quiet pre-registered sentences.

| neighbour | label | fire / FP | one layer back |
|---|---|---|---|
| L3/N6493 + (main input) | token "nerve", Latin script | ✔ 8/8, 1 FP ("Nerf" 0.39) | Direct tokens: ` nerve`, ` nerves`, then Serv / erved / reserved. Feeders are layer-0 spelling units (serve, reserved, -erv-). **A spelling detector for nerv/nerf.** |
| L8/N7756 + (second input) | nerve/nervous-system concept, Arabic-heavy | ✔ 8/8, 0 FP (Arabic 0.9–1.2, German 0.67, English 0.37–0.49) | Arabic subword units (بي, ب) and the "veins" unit |
| L8/N3858 + (pain gate) | pain, word and concept | ✔ 7/8, 0 FP ("pain au chocolat" 0.11) | Multilingual pain unit L6/N2055 (Schmerzen, doler), then numbness |
| L10/N6901 + (strongest reader) | nervous system / neuroanatomy | ✗ 4/8 | Fires on motor commands / motor control / speech production; silent on hippocampus, thalamus, dopamine. Feeders: ` motor` detectors at L0 and L2. **Post hoc (untested): a motor ↔ sensory axis.** Its − pole is taste and smell receptors. |

Reading: the circuit's inputs and gate are solid. The token route is spelling-level. The reader was mislabelled, and it does not respond to psychological "nerve" ("my nerves were on edge" 0.02).

## 2026-10-17: The neighbourhood web around L9/N3688 (`mono/probe/web.py`, `lateral.py`; tests committed in b273a0c, plan in bfac101)

**Crawl:** 48 neurons.
- 6 readers: kept if at least 2 of 3 methods agree (zero-ablation, a 2×-max boost, weights-only cosine).
- 24 siblings: other main inputs of those readers.
- 10 co-firing on nerve's sentences. Sanity check: the top one is nerve's own concept input, r = 0.69.
- 8 second-hop upstream feeders.

**Testing:**
- Every node got a blind label from 3 Claude agents (top examples only), and two pre-registered 8 + 8 sentence tests from the same label: one written by Claude, one by Bonsai-8B itself.
- A label holds with at least 5 of 8 firing and at most 1 false positive.

**What held**
- Claude-written tests held **38 of 48** labels.
- Upstream feeders and siblings are mostly crisp token or spelling detectors: motor, serve, diarr-, trait, -ate, vic, Albany, p-value digits.
- **All 6 readers failed.** Nerve's downstream is diffuse at this level of analysis.

**Lateral family of psychological "nerve"** (all held their labels)
- suspense L7/N9782; eager anticipation L9/N1061; overwhelmed L12/N3688; "disappointed" L1/N4037.
- Each fires on its own sentences (0.6–0.9) and barely on the others' or on nerve's.
- Zeroing any one moves the others by at most 0.03: parallel jobs, not a chain.
- Their shared reader L14/N9470 reaches 0.30–0.34 without nerve, via anticipation and suspense. This **falsified** the pre-registered prediction; it is a family hub.

**Lateral-cluster inverse** (6 outputs)
- DE met 6 of 6 goals at about 600 evals each.
- globalMOO met 5 of 6 at **2–10 evals** each (one model, trial 17556; a crashed first run was resumed on the same trained model).
- Answers: every relative fires without nerve; nerve fires with all of them quiet (0.60); mixed passages co-activate them.

**Two authors**
- Bonsai's sentences are more predictable to Bonsai (median perplexity 40 vs 57), yet they fire the targets less (66% vs 80%) and misfire more as near misses (27% vs 3%).
- Excluding the 146 sentences with leaked prompt wording: 69% vs 80% fire, 15% vs 3% false positives.
- Bonsai's tests held 18 of 48 labels, failing mostly on false positives. The model can produce a concept but cannot reliably write around it.
- Self-generated text gets no special treatment from these neurons.

**globalMOO usage to date:** 8 models (3 toy, nerve, nerve circuit, genre, unclear, nerve lateral cluster).

## 2026-10-17: "nervous" vs the word "nerve" (exploratory, not pre-registered; `runs/nervous_exploratory.py`)

- English " nervous" is a single token, unlike " nerve". **The nerve neuron barely responds to it.**
  - 0.01–0.03 on "felt nervous", "nervous about", "investors were nervous".
  - "a nervous laugh" 0.24; "nervousness" 0.15.
  - Same meaning, different surface form. "nerve-racking" and "worked up the nerve" give 0.6+.
- Forms tokenized with a nerv- piece do fire it: German "nervös" 0.68, Russian "нервничал" 0.49, French "nerveux" 0.44, Spanish "nervioso" 0.17.
- Control: swapping in "anxious" gives 0.02. The suspense and anticipation units respond about equally to nervous and anxious (0.18–0.26).
- Contrasting "nervous" with "anxious" over layers 1–14 finds only spelling-level units: L2/N8177 "Nor", L4/N7189 "-ve / erv", L1/N9531 "na-", plus the known token and concept inputs. No dedicated semantic "nervous ≠ anxious" unit turned up. Expected, since the two are near-synonyms.

**Reading:** L9/N3688 is not a "nervousness" neuron. It responds to the word *nerve* (and nerv- forms) when used in its psychological sense. The token route makes it surface-bound: the meaning of English "nervous" is carried elsewhere, alongside "anxious". This is a third sharpening of the label, after H1 (any "nerve" morpheme) and H2 (psychological sense).
