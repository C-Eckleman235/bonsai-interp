# Bonsai-8B (1-bit) L17/N9491: report

Clean-room investigation by claude-code opus-5.5, 2026-10-18, about 75 minutes of wall-clock time.
I made no globalMOO calls (0 of 1 used). The SciPy text inverse was not run, because constructed text barely moves either pole (see "Least sure").

## Bottom line

| pole | final label | verdict | confidence |
|---|---|---|---|
| **+** | **Near-certain continuation in new text.** Fires where the next token is almost fixed by earlier context. Most often this is the next piece of a rare word, name or code identifier already used in the document (copying from context), or the completion of an established pattern. | **One main job with offshoots.** The main job is copying a token sequence seen earlier. Offshoots are fixed-phrase completions and predictable determiners. It is very sparse, and low entropy alone is far from sufficient. | 0.45 |
| **−** | **No label survived.** Best description: tokens inside the second or contrasted member of a comparison, parallel or concession ("compared to the UK's 5⟦.⟧1%", "not to say that her art is ⟦not⟧"). This is a correlate only: deleting the contrast words does not reduce the activation. | **Unclear / several jobs** | 0.25 (that the contrast reading is the main job) |

The + pole is unusual. Its strong activations need long context (usually more than 128 to 256 tokens). It almost never passes 0.2 of max on short constructed text, even when that text is on-label.

## Evidence, + pole

1. **Top examples** (`play look`, 32 examples): the peaks look unrelated on the surface. Examples: "handled⟦ the⟧ situation", "D⟦rey⟧fus", "Garelick (20⟦0⟧8)", "⟦ else⟧:" in nested Python, "traditional⟦ U⟧.S.", "Società⟦ Cooper⟧ativa", "cssize_re⟦li⟧ability". The pole is very sparse: 548 tokens fall in 0.3 to 0.5 and 38 tokens reach ≥0.5, out of 4.1M scanned (band scan `data/bands_n9491.npz`, windows 0 to 3999).
2. **Context dependence** (`notes/context_length_ablation.txt`): for the top 16 examples, keeping only 8 to 64 tokens of left context leaves most of them below 0.1. They need 256 to 1024 tokens to reach full strength.
3. **Occlusion** (`notes/occlusion_pos.txt`): removing the earlier mention of the same name kills or halves the peak. "2/3 Goat" fell 95%, "Società Cooperativa" 99%, "Ellis and Garelick" 70%, "Dreyfus" 48%.
4. **Repetition statistics** (band scan): at + ≥0.45, 62% of peaks have a (token, next-token) bigram that already occurred earlier in the window. The baseline is 21%. Tokens at the start of a repeat and inside a repeat are 3 to 4x enriched for + ≥0.3.
5. **Predictability**, the strongest single result:
   - In-scan, median next-token entropy falls steadily with activation: random 1.97 nats, +0.1–0.2 1.66, +0.2–0.3 0.95, +≥0.3 0.31. Median p(actual next token) goes 0.12, 0.32, 0.64, 0.93.
   - **Pre-registered held-out check** (commit `0178781`; windows 6000–6599, results in `ee2582a`): random 1.86, +0.1–0.2 1.37, +0.2–0.3 0.63, +≥0.3 0.82 (n=92). The direction replicates, but my pre-registered thresholds were only partly met: P1 (<0.6) failed narrowly, and P2 (monotone) failed in the top band.
   - Selectivity (P3, held): only 0.035% of entropy<0.3 tokens reach +≥0.3. "Low entropy" is necessary-ish but nowhere near sufficient.
6. **Blind labelling** (sub-agent, 40 band examples ≥0.3, `notes/blind_pos.txt`): unprompted, it gave "highly predictable continuation, mostly copying a word or pattern already in context". That covers 72% of examples under the broad reading and 52% for strict copying.
7. **Verbatim repetition does *not* drive it** (`notes/verbatim_repeat.txt`): repeating a 250-token chunk raised the mean max from 0.13 to 0.15 (24 windows). That is why the final label says "in new text". This result is hard to square with "copying", and it is the main reason confidence is below 0.5.
8. **Pre-registered tests:**
   - Round 1, "long-range re-mention of an entity" (pre-registered at `85b3f15`, file `043a`): **FAILED**, 0/8 fire, 1/8 near misses, AUROC 0.41. The fire-set peaks did land on pieces of the re-mentioned names, but only at 0.09–0.13.
   - Round 2, the final label (`9e322f7`, file `ebb6`): **HELD under quick-8+8@2**. Median fire was 0.224 and AUROC 0.94, with 0 near misses ≥0.3; only 2/8 fire items reached 0.3, so it would fail under @1. I had predicted a FAIL. Code with repeated long identifiers scored highest (cssize_reliability 0.65). A first registration of this test (`84843ff`, file `4fdb`) could not run, because `play run` reads the `<` characters in the code as placeholders. I withdrew it in `7182bbe` and re-registered it with `<` rewritten as a reversed `>`.
   - Caveat: the round-2 near misses were structurally different (lists of distinct names) as well as copy-free.
9. **Causal minimal pairs** (pre-registered at `636e331`, results in `8056ea9`): I replaced every re-use of the repeated name or identifier with a fresh rare name. Activation dropped in **8/8** passages, from a mean of 0.257 to 0.154 (−40%; the bar was 40%). The mean of per-passage drops was 32%, so B1 holds only at the margin.
10. **Steering** (`notes/steering.txt`, ±5 and ±10× pole max at every position): the effects are small. At +c, greedy continuations repeat their own earlier phrases slightly more ("I remember … I remember", "I think we can do … I think we can do"). Mean next-token entropy rose slightly, from 2.68 to 2.82; it did not fall. Direct logit effects are uninformative at layer 17.

## Evidence, − pole

1. **Top examples:** many are the second member of a comparison or concession. Examples: "compared with the national average gain of 2⟦.⟧9", "the UK's 5⟦.⟧1%", "which is not to say that her art is⟦ not⟧", "They're like us–as a family,⟦ we⟧", "American⟦ politicians⟧ would fare badly", "much as the photon is⟦ the⟧". Others are code: "⟦ =⟧" in an else-branch, awk "⟦print⟧". The pole is denser than + (1,274 tokens ≥0.3 in the scan).
2. **Exploratory probes** (`notes/exploratory_probes.txt`): paraphrases of the top examples reached 0.24–0.51, and the matched controls without a contrast 0.01–0.05. But many contrast sentences stay quiet ("whereas the old model…" 0.03, "red team / blue team" 0.05).
3. **Corpus correlate:** a contrast-cue word appears in the previous 30 tokens for 16% of random tokens, 39% of −≥0.3 tokens and 44% of −≥0.45 tokens. Held out: 0.16 against 0.37, so P4 held. Next-token entropy at − peaks is near baseline (2.27 against 1.86, P5 held), so the − pole is not the mirror of the + pole's predictability.
4. **Blind labelling** (`notes/blind_neg.txt`): "token inside the second or contrasted member of a parallel or contrastive structure". That covers 42% clearly and 70% generously. The labeller said no single label covers more than 60%, and noted a third cluster of "verb/modal after the subject".
5. **Pre-registered tests:**
   - Round 1 (`88e7c38`, file `3f18`): **FAILED**, 1/8 fire. Separation was good (AUROC 0.95), but median fire was 0.194, just under 0.20.
   - Round 2 (`6e735c6`, file `6639`), with longer minimal pairs and a broader contrast label: **FAILED**, median 0.142 and AUROC 0.79.
6. **Causal test** (pre-registered at `636e331`): I deleted the contrast-cue words before the peak in 36 corpus contexts that had a cue. The mean drop was 0.125, against **0.145 for deleting random words**, so A1 and A2 failed. Half of the 72 strong contexts contain no cue at all. The contrast reading is a correlate, not the mechanism. This is why I give the − pole no label.

## Rejected labels

| label | evidence against |
|---|---|
| + "long-range re-mention of an entity introduced earlier" | Test 043a: 0/8, AUROC 0.41. Short re-mention probes stayed below 0.1. |
| + "induction / verbatim copying of an earlier span" | Repeating 250-token chunks verbatim does not raise the pole (0.13 → 0.15). |
| + "token is the model's confident next prediction" (pure low entropy) | Only 0.035% of entropy<0.3 tokens fire. Steering + raises entropy slightly instead of lowering it. |
| − "counterpart slot of a comparison/concession" | Test 3f18 failed on amplitude (median 0.194). Cue deletion has no effect beyond control. |
| − "second, contrasted member of a contrast/parallel structure" | Test 6639 failed (median 0.142, AUROC 0.79). Cue deletion has no effect beyond control. |

## Neighbourhood (`play step --k 4`, `notes/step_feeders.txt`)

All feeders are weak (single-unit drops of 0.02–0.04), so the input is distributed.

| feeder | → pole | drop | label (from `play look`) | test |
|---|---|---|---|---|
| L15/N8076+ | + | 0.04 | sub-word pieces inside code identifiers and capitalised defined terms | **FAILED** 0/8 (`808bccb`). It does not fire on short identifier lines. |
| L3/N2114− | + | 0.04 | arsenic / arsenide word pieces | **HELD** 8/8, 0 false positives (`9c52739`) |
| L9/N1314+ | − | 0.04 | vampires and vampire fiction | **FAILED** under @2: 8/8 fire, but 2 near misses fired ("Sookie", "Bram Stoker"). My near misses were really on-concept (vampire-fiction names), so this is closer to a bad test than a bad label. |
| L14/N7100+ | − | 0.03 | onset of an analogy vehicle ("like a⟦ strong person⟧", "similar to⟦ a⟧", "imagine⟦ a⟧") | **HELD** under @2: 4/8 at ≥0.3, AUROC 1.00 (`03d2ef1`) |

Edges are recorded in `1ec1bd4`. The L14/N7100 "analogy" feeder into the − pole is consistent with the comparison correlate. But the cue-deletion result says the − pole does not depend on those words directly.

## Monosemanticity verdicts

- **+ pole: one main job with offshoots (confidence 0.45).** One computational theme, "the next token is fixed by earlier context, usually by copying a rare token sequence", explains most strong examples. It has dose-response with entropy, held-out replication of the direction, a held 8+8 test and an 8/8 causal minimal-pair result. Offshoots are fixed-phrase completions ("point of view", "each other") and predictable determiners ("handled the situation"). Against a "monosemantic" verdict: it fires on only a tiny subset of such positions, it ignores verbatim repetition, and what selects that subset is unknown.
- **− pole: unclear / several jobs (confidence 0.25 in any single reading).** The comparison/contrast tendency is real as a correlation. But it covers about half of the strong examples, fails two 8+8 tests, and fails the causal cue-deletion test. The rest is code (else-branches, assignment after a parallel branch) and generic verb/modal slots after a subject.

## What I would test next

- **+:** what separates the rare firing copy positions from the ~99.9% of low-entropy positions that don't fire? Candidates are distance to the earlier occurrence, multi-token versus single-token names, and copying a span that occurred exactly once before versus many times. A matched corpus regression over these features would answer it.
- **+:** an attention-head attribution. Which L≤16 heads (induction-like or previous-token) carry the signal? That would explain why verbatim copies are ignored.
- **−:** use the inverse tool (DE, or the single globalMOO model) on a bank of long passages rather than short sentences, to find what actually maximises the pole. Also run the cue-deletion test with replacement words instead of deletion.
- Re-test the vampire feeder with near misses that are truly off-concept.

## Least sure

- Short-text amplitudes are tiny for both poles (usually below 0.2), so the 8+8 protocol mostly measures separation, not strong firing. The single + test that held is at the margin (median 0.224) and depends on code items.
- B1 (−40%) sits exactly at its threshold, and the round-2 + near misses differ in structure as well as in copying.
- For the − pole, I may have missed a cleaner structural description (for example "the slot after a re-instantiated syntactic frame"), which could explain both the comparison and the code cases.

## Commits

Pre-registrations: `88e7c38` (− r1), `85b3f15` (+ r1), `808bccb` `9c52739` `077fd83` `03d2ef1` (feeders), `0178781` (held-out), `6e735c6` (− r2), `84843ff` (+ r2, withdrawn in `7182bbe`), `9e322f7` (+ r2), `636e331` (ablations).
Results: `8fc31b3`, `ee2582a`, `8c888fa`, `8056ea9`; edges `1ec1bd4`.
