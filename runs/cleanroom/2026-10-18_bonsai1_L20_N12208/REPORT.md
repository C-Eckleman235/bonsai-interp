# Bonsai-8B (1-bit) — L20/N12208, both poles

Pole maxima: + 20.0, − 13.56 (raw `silu(gate)·up`). Every activation below is a fraction of the pole's own corpus max. All + activations are in the `+x+` regime, and all − activations are in `+g −u`.

## Final labels

**L20/N12208+ — copying ahead inside repeated structured records.**
Deep in a document made of many similar records (citation lists, code rows, tables, log lines, config entries, "READ MORE" teasers), the pole fires on a token whose *next* token is recurring template text that already appeared in earlier records. Examples: the `2` of a recurring year `2014`, the `Ge` in `Gean`, the `(p` in `popup_url`, and the `READ` in `READ MORE`. The activation grows with the number of earlier records. It needs an earlier copy of the bigram (current token → next token) in context. It is weak for repeated prose and silent for a single repeated name.

**L20/N12208− — the end of a segment when little context has been established.**
The strongest case is the line break that ends a short heading, title, caption or credit line with no closing punctuation, just as the body text begins. Examples: `Department of Parks/flickr⏎` 0.68, `Cancer Control Month - April⏎` 0.52, and the first line after `<|endoftext|>` in corpus tops. The weaker, broader offshoot is the last word of a phrase or clause and sentence-final punctuation, which reaches about 0.2–0.4 in ordinary prose. It is stronger early in a document. It is quiet in formulaic, predictable sequences such as counting, the alphabet, import lists and repeated text.

The two poles look like opposite ends of one "how established is the context's structure" axis. + needs deep, repetitive, copyable structure. − prefers fresh context and segment boundaries. I consider this framing a plausible interpretation, not something I have shown.

## Evidence

### + pole
- **Corpus statistics** (band scan, 2,000 windows, about 2M tokens, `analysis_induction.py`). Among tokens at ≥0.5 of max, 89% have their next-token bigram already present earlier in the window, against a base rate of 21%. 99% have the current token seen before, against a base of 57%. The share rises monotonically across bands (0.1–0.2: 55%; 0.3–0.5: 84%).
- **Model confidence** (`analysis_entropy.py`, 120 held-out windows). At ≥0.3, mean next-token entropy is 0.42 against 2.2 at baseline. Max-prob is 0.89, and 96% of argmax predictions are a token already in context.
- **Causal context ablation** (`analysis_context.py`, top 32 examples). Full context gives 0.83. Replacing every earlier occurrence of the (current, next) bigram gives **0.08**. Truncating to the last 128 tokens also gives 0.08, and the last 8 tokens give −0.01. The copy source is necessary.
- **Not sufficient.** Only 0.3% of induction positions fire at ≥0.3 (`analysis_counts.py`). The firing rate rises with the number of earlier occurrences (1 → 0.11%, ≥33 → 2.6%) and with depth in the window (pos <128 → 0.02%, 768+ → 0.46%). Model max-prob, the length of the matched prefix and the ambiguity of the copy explain little (Spearman ≤0.08, `analysis_gate.py`).
- **Synthetic copy of a rare name.** A name seen once and then repeated after 0–900 tokens of filler peaks at 0.06 (`analysis_distance.py`). This is why plain "induction" was rejected.
- **Pre-registered tests:**
  - v1 dose (ce225af): record count 2/4/8/12 against activation gives Spearman 0.57. The mean at 12 records is 0.27 against 0.11 at 2. **Held.**
  - v2 prose vs records (a7f2795): 8× repeated prose sentence gives mean 0.15. 8-record lists give 0.26, and the lists won 4/4 pairs. **Held.**
  - Shared-map quick test (0898785, results 7e06b3f): 5/8 fire and 0/8 near misses fire. **Held.** The near misses were the same topics without the repeated records (one record, prose about citations, a twice-repeated name), and all stayed ≤0.16.
  - Bank/solo (b8d72f8): Spearman(relevance, act) is 0.88 and AUROC on-vs-off is 1.00. In the held-out set, 2/4 predicted-fire passages reached 0.3 (the others reached 0.27 and 0.16) and 0/4 predicted-quiet did. AUROC 1.0, accuracy 0.75.
- **DE inversion** (`invert_de.json`). The max reached 0.70, from three stacked record lists. The **leak** goal (≥0.3 with no records present) was not satisfiable, and the best attempt reached 0.28. The silent-on-concept goal was not satisfiable either: the lowest on-concept passage was 0.13.
- **Induce.** The direct logit effect is diffuse (` already`, ` ge`, ` down` …). KL is <0.07 for c ≤ 10. Generations at c=5–10 drift slightly more into repeated clauses, but the baseline already does that, so I place no weight on it. Zeroing the neuron at its own top-24 peaks barely changes log p(next): −0.123 → −0.118. The model is already near-certain at those positions.

### − pole
- **Corpus.** The most enriched tokens are `.⏎`, `)⏎`, `|⏎`, `).⏎`, ` .⏎` and `⏎` (4–10% of each fires at ≥0.3, against 0.3% for `,`). The pole is not anti-induction: next-bigram-seen is 28% against a 21% base.
- **Depth effect** (`analysis_docstart.py`). Newline tokens within 16 tokens of a document start fire at ≥0.3 **11.3%** of the time. From 64–256 tokens in the rate is 2.3%, and from 256–1024 it is 1.3%. Across window positions, 8–128 gives 8.0% and 512+ gives 1.7%.
- **Pre-registered tests:**
  - v1 "clause end in prose" (ce225af): 1/8 fire at 0.3 and 0/8 quiet. **Failed at threshold**, although it separated the sets (fire mean 0.23 against quiet 0.06).
  - v2 doc-start (a7f2795): the newline after a title right after `<|endoftext|>` beat the same title deep inside a continuous document **8/8**, with mean ratio 2.11. **Held.**
  - Shared-map quick test (3f91646, fix 514fda1, results 7e06b3f): 6/8 fire and 1/8 near misses fire. **Held.** Headings fired at the newline at 0.73, 0.79, 0.49 and 0.50. The near miss that fired was the run-on sentence "…railway stations of northern England, …", at 0.42 on ` England` (the heading version reached 0.48 on the same word). That instance is the phrase-final-word offshoot.
- **Cross-talk.** On the + bank, the − pole also reached 0.3–0.45 on record lists, at field ends such as `Retrieved June 5⟦⟧,` and `sheet_data⟦⟧`.

## Rejected or modified labels

| Label | Evidence against |
|---|---|
| + "generic induction: the next token is copyable from context" | Only 0.3% of induction positions fire. A rare name repeated at distances of 0–900 tokens peaks at ≤0.06. A verbatim-repeated sentence ("quick brown fox", ×2) reaches 0.05. |
| + "high model confidence / low entropy" | Spearman(act, max-prob) among induction positions is only 0.08. |
| + "template continuation, *excluding* verbatim repetition" (v1) | Verbatim-repeated *records* fire nearly as well: 0.32 against 0.38 for the citation list (v1 pair test 3/4). The real contrast is records/structured data against prose, not template against verbatim. |
| + v1 quick wording (7–9 records) | 3/8 at 0.3. The direction was right, but the records were too short, so the final label requires many records. |
| − "end of a clause or sentence in free prose" (v1) | 1/8 at threshold. Ordinary sentence ends only reach about 0.13–0.31. |
| − "next token unpredictable / high entropy" | Entropy at ≥0.3 is 2.58 against 2.2 at baseline, which is too weak to be the driver. |

## Monosemanticity verdict

- **+ pole: one main job with offshoots.** Confidence 0.6. Almost all strong activations are copy-ahead positions inside long structured or repetitive documents, mostly code, citations and tables. The offshoots are highly predictable memorised continuations without an earlier copy in the window: 239 of about 1,600 ≥0.3 tokens, such as `limitations under the⟦ License⟧` and `17th⟦ century⟧. They fit the same "near-certain continuation in boilerplate" theme. I did not call it monosemantic for two reasons. What gates the 0.3% of induction positions that fire is not pinned down. Separately, the magnitude is small for synthetic text, with most passages at 0.2–0.45.
- **− pole: one main job with offshoots.** Confidence 0.5. The main job is the end of a heading, caption or segment line at a fresh section start. The offshoots are phrase-final content words and ordinary sentence ends, especially early in a text, plus field ends inside record lists. It is broad and low-amplitude: 289k tokens exceed 10% of max.

## Neighbourhood (feeders from `play step --k 4`)

| Target | Feeder | Drop | Label (quick test) | Result |
|---|---|---|---|---|
| + | L0/N11769− | 0.70 | `stack` in OpenStack / code identifiers (1ec3d50) | **held** 8/8, 0/8 |
| + | L3/N9864+ | 0.48 | caret `^` regex anchor in code strings (f5c2a8e, fix ab2093f) | **held** 8/8, 0/8 |
| − | L2/N11953− | 0.29 | next counter in an enumerated series (45cb946) | **failed** 3/8: it fires on Arabic reference digits (`21`, `16`), not Roman numerals |
| − | L1/N9909+ | 0.18 | colon in forum timestamp `MM-DD-YYYY, HH:MM PM` (e804f56) | **held** 8/8, 1/8 |

Test results are in 39b431f and the edges in 6308e08.

**Caveat: these edges are example-specific, not general circuitry** (`analysis_feeders2.py`). Each feeder's mean drop comes from a single top example, and its drop on the other five examples is ≈0. For example, L0/N11769 drops the "stackwalker" example by 4.0 and the rest by ~0. L2/N11953 and L1/N9909 drop + top example #5 (forum timestamps) by 0.87 and 0.82. The feeders are also ≈0 *at* the target position and act through earlier context tokens (`analysis_feeders.py`). This fits an attention-mediated, copy-like unit with no dedicated MLP feeders. Read the edge weights as "matters for one top example".

## Process notes
- Two pre-registered test files had sentences containing `<` (`<|endoftext|>`, and a regex `(?P<name>…)`). The runner rejects these as placeholders. In both cases I replaced the affected sentence in a separate commit **before any run** (514fda1, ab2093f) and recorded it in the file's `notes`.
- The `run_prereg_v2.py` comment says PRIOR comes from a corpus window. It is actually a hand-written village paragraph ×2.
- globalMOO: not used (0 of 1). The DE solver was enough to show that the leak goal is unsatisfiable.

## What I'd test next / least sure about
1. **What gates the + pole among copy positions.** I'm least sure here. Candidates include document type (code and reference lists compared with prose), the count of earlier record boundaries, or a specific induction head's output. The next step is to attribute through attention: which heads write the direction this neuron reads, and whether its activation tracks one head's attention to the source.
2. A cleaner − test that separates "start of document" from "short line without terminal punctuation" in a 2×2 design.
3. Whether the two poles share a common upstream "context depth / repetitiveness" signal. One check is to regress both on the number of earlier repeated bigrams in a window.
4. Downstream effect. Single-position ablation barely changes outputs, so a multi-position ablation across a whole record list is needed to see whether the neuron helps copying.
