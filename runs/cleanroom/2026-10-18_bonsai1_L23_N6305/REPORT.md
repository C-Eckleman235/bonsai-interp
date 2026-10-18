# Bonsai-8B L23/N6305: two-pole investigation

The negative pole has the stronger interpretation: it marks **reference domains for quantified claims**, especially sets of observations or attempts used to state coverage, prevalence, or success. Its output direction promotes *tested, checked, examined, attempts,* and *tried*. The positive pole concentrates on **temporal/conditional frame openings**, but its context gate remains unresolved. Neither pole earns an unqualified monosemantic verdict.

| Pole | Final label | Verdict | Confidence |
|---|---|---|---|
| + | Context-gated temporal and conditional frame openings, especially After/Once and reduced when-clauses | **Unclear** | 0.45 |
| − | Reference domains for quantified claims: relative-clause restrictors, outcome denominators, and event-condition subjects | **One main job with offshoots** | 0.75 |

Confidence is a subjective assessment of the interpretation and verdict, not a statistical probability derived from eight examples.

## Scope, provenance, and scoring

Only the supplied folder, model, corpus, and toolchain were used. No prior neuron interpretations, web research, downloads, external services, or sub-agents were used. **globalMOO models used: 0.** Both inverse searches used SciPy differential evolution, seed 0, population multiplier 8, up to 12 iterations per goal. Forward, inverse, induction, upstream/downstream circuits, and signed ablations were completed for both poles.

The original repository's `.git` is read-only under the environment's permissions. Initial commit attempts were denied. All successful preregistration and result commits are therefore in **`study_checkout/`**, an independent local repository inside the requested folder, based on supplied toolchain commit `5b02f71` (local snapshot `e5b1fdf`). Its Python toolchain sources are unchanged. The top-level report and summary are exact copies of the committed deliverables; `STUDY.bundle` preserves the study history. See [STUDY_LOCATION.md](STUDY_LOCATION.md). Evidence paths below are relative to that study repository, and the evidence is also mirrored beside the top-level report.

The supplied scan covers 8,000 × 1,024 tokens. The corpus file contains 29,303 windows, so the 40 randomly selected bank excerpts are not restricted to the original scanned slice. Corpus maxima are **+13.5625** and **−104.0** in raw activation. A fire means at least **0.30 of that pole's maximum**, averaged over the tool's three neutral preambles; these are different raw thresholds (+4.069 and −31.2). Values above 1 are possible because a corpus maximum is not a theoretical limit. Standard tests take a maximum over each input's tokens, not a mean over its words. Contextual final-positive entries include a lead-in and a test sentence. The official quick-test criterion remains at least 5/8 fire and at most 1/8 near misses.

Full-prefix measurements at the six strongest corpus sites per pole reproduced the stored peaks to within **1.86%**. Thus the positive pole's weak isolated-sentence results are not a scan/model mismatch. The circuit tool truncates context to 384 tokens; that is separately recorded. Labels concern information available up to the peak: a displayed suffix cannot cause an earlier activation.

## Positive pole: a clear corpus preference, an unresolved gate

Among the top 32 corpus peaks, my post-discovery annotation found **20 After/after, 2 Once, 4 when-constructions, 5 if-constructions, and 1 unexplained Chinese hair-removal example**. Peaks can occur on the connective or immediately after it: *After*, *When passed*, *when applied*, *If you read*. This supports a temporal/conditional opening family but does not explain when the gate opens. The Chinese peak on 毛 reaches 8.8125 raw (0.650 of maximum); the isolated hair-removal sentence does not reproduce it. It remains an exception, not a demonstrated second semantic job.

The initial broad label failed: **0/8 fire, 0/8 near misses**, preregistration **`d8088b9`**. Across the 148-entry bank (92 authored, 40 random corpus, 16 extracted discovery examples), **no isolated entry reached 0.30**; the largest was 0.269. Authored After sentences typically reached about 0.17–0.20, whereas Before and many other conjunctions remained much weaker. This is a graded preference, not a passing detector.

The context experiments actively challenged the explanation:

- In `c07897b`, a short procedural lead-in **reduced** “After you check the settings…” from **0.183 to 0.033**; adding only leading whitespace gave 0.200. Neither simple tokenization nor generic procedural context explained the corpus peaks.
- In the fresh transfer set `ec25557`, **0/8 target-only, 3/8 related-context, and 3/8 unrelated-context** passages fired. A craft narrative followed by “After the break…” reached **0.536**; a skin-care narrative followed by “When applied…” reached **0.480**; a folklore narrative followed by “When passed down…” reached **0.541**. Some unrelated contexts also enabled the target. Passage scores with original corpus prefixes recovered strong activity, but topical relevance was not sufficient or necessary.
- The inverse search (`3970260`) combined individually weak entries into a **0.4585** passage: “After lunch, we walked to the river. She said that the train was late. After the break you can see a picture of the finished chair.” It also found an on-label passage at **0.0674**. The leak goal failed; the strongest eligible off-label candidate anywhere in the search reached **0.2742**, ending “After all, nobody knew the answer.” This finite bank search does not establish global absence of leaks.

The final contextual label was preregistered in **`afef9ff`** and received **5/8 fire; 0/8 near misses; HELD**. Candidate-fire activations in file order were **0.498, 0.325, 0.492, 0.515, 0.123, 0.352, 0.243, 0.085**. These are new passages after the transfer experiment, with paired connective/structure near misses. The preregistered expectation was uncertain (3–6 fires), while the official pass criterion stayed unchanged. The best label remains descriptive: neither an unconditional After detector nor a general temporal-clause detector is established. **Monosemanticity is unclear**, because the contextual gating is not captured and the strong Chinese exception remains unexplained.

For concrete held-out contrasts, the new camera-case narrative followed by “After the break…” reached **0.498**, versus **0.029** with “Before the break…”. The gel narrative followed by “When applied…” reached **0.515**, versus **0.046** with a declarative continuation. However, the new cocoa-processing “Once the beans…” case reached only **0.123**, “When running inside a container…” **0.243**, and “If you take digital photographs…” **0.085**. A pooled pass does not resolve these misses.

## Negative pole: domains of observations, attempts, and outcomes

The top 32 discovery peaks divide descriptively into **19 relative/reference-set cases, 9 numerical denominators or partitives, 3 event-condition subjects, and 1 ordinal attempt**. Those categories are post-discovery annotations, not independent accuracy estimates. The common theme is the domain against which a claim is evaluated, with several grammatical realizations:

1. **Relative/reference-set restrictions:** *most scientists I have spoken to*, *everything she attempts*, *all organisms examined*.
2. **Numerical outcome denominators:** *six prizes from eight nominations*, *nine of twelve matches*, *19 of 21 countries*.
3. **Attempt ordinals and event conditions:** *on its first attempt*, *whenever they move*. The conditional branch is less reliably reproduced.

The restrictive-relative-only hypothesis failed its initial contribution (**2/8 fire, 0/8 near misses; `300026e`**). “All the books that I borrowed were damaged” reached only **0.092**, while the off-label sentence “The film won eight awards from eleven nominations” reached **0.862**. The initial predicted-fire examples still ranked above their paired controls (AUROC 1.0), so the failure includes inadequate sensitivity at the registered threshold as well as a missing numerical branch. Plain quantification and simple testing verbs are insufficient: “Most people enjoy music,” standalone nominations, and “They tested the people for the antibody” are weak.

The revised contribution, preregistered **`261c1db`**, received **7/8 fire; 0/8 near misses; HELD**. Candidate-fire values were **0.928, 1.040, 0.509, 0.475, 0.354, 0.627, 0.400, 0.296**. In particular:

| New sentence | Negative activation |
|---|---:|
| She succeeds at everything she attempts. | 0.928 |
| He has won every race that he entered. | 1.040 |
| Most scientists I have spoken to agree with the result. | 0.509 |
| Forty percent of the people they tested carried the antibody. | 0.475 |
| The team won nine of its twelve matches. | 0.354 |
| The film received six prizes from eight nominations. | 0.627 |
| The experiment worked on its first attempt. | 0.400 |
| The army defeats its opponents whenever they attack. | 0.296 |

All eight matched controls were at most **0.065**. All four relative-domain examples, both numerical examples, and the ordinal passed; the conditional missed by 0.004. The diagnostic “The general conquers the enemy whenever they move” independently reached **0.312**, versus **0.097** with “soldiers” replacing “they.” These small threshold margins warrant caution about that branch.

The negative inverse search (`3970260`) found a valid relative-label leak at **0.6923** using a nominations denominator and a near-miss passage at **0.8846** containing the simple film example. Its maximum was **1.197**. The nominal silent-on-concept winner at 0.0523 contains a *nonrestrictive* uncle relative mistakenly rated fully on-concept; I do **not** count that winner as a clean falsification of a strictly restrictive-relative label. The valid simple-relative misses above supply that evidence.

My verdict is **one main job with offshoots**, not strict monosemanticity: quantified reference domains unify the relative and denominator cases reasonably well, and the output direction supports an observations/attempts reading. Ordinal and conditional extensions are less settled. Different grammatical forms alone do not prove unrelated jobs; conversely, a broad unifying phrase does not prove a single internal computation.

## Causal effects and immediate neighbourhood

The two poles write opposite directions of the same output vector. Negative directly promotes **tested, checked, examined, attempts, tried, inspected**; positive suppresses those tokens and promotes a heterogeneous set including **certain, design, joint, delivery, ways**. These are readout associations, not semantic proofs.

The preregistered neutral-prompt token-set prediction was not simply confirmed. At c=+2, positive temporal-token probability fell **0.02634 → 0.02412**, and negative relative/pronoun-token probability fell **0.22358 → 0.21654**. Negative c=+2 nevertheless preferentially increased *checking/check/trying/test* (KL **0.0402**). At c=+5, KL was **1.788**, and generations repeatedly used testing/trying language; c=+10 produced “tested tested…” collapse. Positive c=+2 had KL **0.000827**, rising to **0.303** at c=+40. Equal c is not equal raw perturbation across poles: their maxima differ by 7.67×. Large-c generations are stress tests, not normal behavior.

The separately preregistered natural-size intervention (`50b16d2`) zeroed only the target at the observed peak, on eight corpus prefixes per pole. Median KL(clean || ablated) was **0.000519** for + and **0.034725** for −; maxima were **0.001038** and **0.128210**. The complete token gains and actual-next-token log-probabilities are in `study/natural_ablation.results.json`. This separates naturally sized causal effects from the extreme steering results.

Natural negative activation improved the actual next token's log-probability at **6/8** sites: for example, *move* gained **0.538 nats** and *nominations* **0.233 nats**. It reduced *see* by **0.290 nats**, so the effect is not uniformly beneficial even on peak examples.

For each pole, `play step --k 4` ranked attribution-selected candidates by mean single-unit zeroing drop on six corpus peaks. The top two were tested once each and mapped:

| Feeder pole | Target pole | Displayed drop | Tested feeder label | Fire; near misses | Preregistration |
|---|---|---:|---|---|---|
| bonsai1/L19/N11756- | + | 0.05 | English after across temporal, pursuit, naming, and after-all uses | 5/8; 0/8 | `4677944` |
| bonsai1/L22/N4819+ | + | 0.04 | German nach across senses and French après | 8/8; 0/8 | `3631f69` |
| bonsai1/L22/N408+ | - | 0.07 | Restrictions and reference sets for quantified groups or comparative judgments | 5/8; 0/8 | `d71379e` |
| bonsai1/L22/N12064- | - | 0.05 | Amounts or outcomes relative to a total, available set, or limiting condition | 5/8; 0/8 | `506a4b7` |

All four feeder labels held. The other displayed candidates were +: **L22/N7936 0.03, L11/N3117 0.03**; −: **L14/N8229 0.05, L22/N5078 0.03**. These are the strongest candidates returned by the tool, not an exhaustive ablation ranking of all upstream units.

The extra sign-specific audit (`ba89ed5`) matters: the After feeder contributes through its **negative** pole (mean negative-only drop 0.0478), whereas L22/N4819 contributes through its **positive** pole (0.0455), despite its English temporal examples being strongest on the opposite pole. Negative-target feeders were confirmed as L22/N408+ (0.0725) and L22/N12064− (0.0517). Four `play edge` contributions record the displayed full-unit drops, with the sign audit providing the polarity assignment. Drops are small, context-specific, and nonadditive. Modest differences between repeated, differently batched runs are consistent with bfloat16 rounding. A feeder's top German examples do not imply that German text is required at the target.

Downstream zeroing on two discovery peaks per pole changed later channels, including L24/N4413+ by 0.081 of its maximum for the positive examples and L32/N9330+ by 0.549 for the negative examples. No semantic labels are inferred for those downstream units from effect size alone.

## Coverage, limitations, and next tests

The preregistered lower-band audit (`320bed6`) scanned **524,288 tokens** (first 512 windows), excluded all original top-32 discovery windows from sampled contexts, and used seed 44 with at most one sample per window per band. Counts below are all tokens in each band; sample exclusions apply only to the final column.

| Pole | Fraction band | Token count | Sampled contexts |
|---|---|---:|---:|
| + | 0.1-0.3 | 414 | 12 |
| + | 0.3-100 | 37 | 12 |
| - | 0.1-0.3 | 655 | 12 |
| - | 0.3-100 | 39 | 12 |

The 512-window audit found 37 positive and 39 negative tokens at or above 0.30 of corpus maximum, plus 414 positive and 655 negative tokens in [0.10,0.30). Twelve contexts per pole per band were sampled outside all original top-32 discovery windows. The main patterns recur, but the positive label has clear boundary cases and the negative pattern extends to code and Arabic.

Positive strong samples include After the hunt (0.438), Once you have selected a collection (0.355), and When you open a Photoshop canvas (0.327). Boundary cases include After Mail-in Rebate (0.318), Although (0.309), visits well inside a when-clause (0.385), and Chinese 一个 ('a/one') in a patient description (0.302). The latter is a second unresolved Chinese example, not evidence that a coherent Chinese subfeature has been identified. Negative strong samples include a code docstring describing failures out of tried examples (0.302), none of the generators I have seen (0.409), a majority of those surveyed (0.524), and Arabic الذين ('who') in a quantified patient group (0.401). Extensions include 1 catch for 9 yards (0.387), only comprehensive study (0.315), and my brief tenure (0.389): these fit an observations/outcomes theme more loosely than a literal relative-clause or denominator rule. The lower [0.10,0.30) bands are less specific, including ordinary narration and report language. These are post-measurement descriptive judgments, not blinded label-accuracy scores.

This is a contiguous corpus prefix, not a representative random sample of all text. The hand-built banks deliberately oversample candidate constructions. The duplicated `holdout.json` and initial graph tests are **one test set**, not two independent replications. Top examples and corpus-prefix transformations are discovery evidence. Inverse searches concatenate a fixed bank, including some fragments and mixed-language text; they neither exhaust natural language nor prove absence of counterexamples. The full inverse records contain 3744 positive and 3552 candidate evaluations, with caching and early stopping.

Two annotation defects are documented in `study/annotation_audit.json`: some coarse negative relevance ratings included nonrestrictive relatives, and four diagnostic rows used ungrammatical “Every workers.” Those rows are not used for the corresponding claims. Original preregistrations are preserved. The supplied Spearman helper mishandles ties; `study/metrics.json` includes corrected SciPy values, while the report relies primarily on explicit contrasts and official quick-test counts. A long-prefix run failed with a CUDA unknown error before writing results; the identical tests succeeded at batch size 2 (`2f554fe`). No test text or prediction changed during that retry.

**Least certain:** what opens the positive gate; whether its Chinese peak is a genuine unrelated feature; whether the negative relatives and denominators implement one abstract reference-domain computation or partially independent subfeatures; and how robust the conditional offshoot is across names, pronouns, and wording.

**Next tests:** (1) preregister token-level edits of successful positive prefixes to distinguish particular antecedents, discourse position, lexical priming, and semantic relevance; (2) test new genres and languages with independent authors and blinded labels; (3) cross outcome/observation predicates with relative restrictions and numerical denominators in a grammatical factorial design, including failures, neutral descriptions, and bare test/try verbs; (4) ablate the identified feeders on those contrasts and test combinations rather than treating individual drops as additive; (5) measure natural next-token effects on those held-out constructions. These would distinguish a shared negative computation from a collection of patterns and could resolve the positive verdict.

All eight graph test files were committed alone before evaluation; each result was committed later. Core preregistrations: initial labels `d8088b9` / `300026e`, banks/inverse/induce `3970260`, diagnostics `c07897b`, sign audit `ba89ed5`, context transfer `ec25557`, final labels `afef9ff` / `261c1db`, and natural ablation `50b16d2`. Feeder hashes appear above. See `study/validation.json` for the complete contribution/result commit audit, and `summary.json` for the machine-readable findings.
