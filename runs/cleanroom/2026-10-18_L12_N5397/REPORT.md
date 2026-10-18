# L12/N5397 (Bonsai-8B, 1-bit): report

**Short answer.** Neither pole is monosemantic. Both respond to structured, non-prose data, but in
different places and through different upstream circuits.

- **+ pole:** its top end is dominated by the **line-break token that ends a row deep inside a long
  dump of raw hex/byte data**, for example `\xNN…\⏎` escape strings, octal and `\u` dumps, hash lists
  and byte lists. Three offshoots run through separate circuits: Python `', '.join` separators,
  long "ha ha ha" runs, and a diffuse tail of prose and code tokens. Most of the moderate (0.3–0.5)
  corpus activations belong to that tail, and nothing I found predicts them.
- **− pole:** a family of **"inside or right after a structured numeric/code value"** detectors:
  - the unit after a dimension (`9x13 ⟦inch⟧`)
  - nested-array row closers (`[[1, 2]⟦],⟧`), even when the items are words
  - the deep characters of long alphanumeric IDs or hex strings
  - delimiters after phone numbers and citation years

  Each sub-job has its own upstream feeders.

| | Label | Verdict | Confidence |
|---|---|---|---|
| **+** | Row-ending line breaks inside long hex/byte-data dumps (escape strings, hash and byte lists); grows with dump length. Offshoots: `', '.join` separator, long "ha ha" runs, diffuse low-precision tail. | several jobs (one dominant at the top of the range, plus a large unexplained tail) | 0.55 |
| **−** | Token inside or right after a structured numeric/code value: dimension units, nested-array row closers, deep characters of long IDs/hex, delimiters after phone numbers and citations. | several jobs (shared surface theme, separate circuits) | 0.65 |

## Setup and basic facts

- Corpus maxima: + 2.11, − 5.25. Above 0.3 of max the + pole fires on about 0.02% of tokens
  (633 of 3.07M in my 3,000-window band scan) and the − pole on about 0.05% (1,564).
- Sign regime: + activations above 0.3 are 97% `+x+`. The large mass of + activations between 0.1
  and 0.3 (530k tokens) is 97% "−x− echo" regime: dense, low-level noise that I ignore. − activations
  are 100% `+g −u`.
- **Window-start artifact.** 17 of the 32 top + examples sit at window positions below 64 (uniform
  would give 6.25%). These are the prose sentence-end hits, such as "…Edwards Air Force Base,
  California⟦.⟧". Re-read after a neutral preamble they collapse: 0.95 → 0.16, 0.89 → 0.01,
  0.77 → 0.05 (`work/ctxlen.py`, `work/repro.py`). Both poles are 3–4× enriched in the first 64
  positions, so I sampled positions ≥ 64 throughout.
- **Context dependence.** The − pole is local: 8–32 tokens of context reproduce its top examples. The
  + hex-dump examples need 128–1024 tokens of context, and the activation grows with dump length.

## Key evidence

### + pole

1. **Dose-response with dump length** (exploration, `work/explore3.json`). A `\x` escape dump with
   `\⏎` line continuations gives 0.17 at 3 lines, 0.57 at 12 lines and 0.88 at 30 lines.
2. **The peak is on the line break** (controlled variants of about 300 tokens each, `work/explore5.json`):

   | Variant | max + |
   |---|---|
   | `\x` dump with `\⏎` | **1.07** (above the corpus max) |
   | `\x` dump with plain `⏎` | 0.99 |
   | octal dump | 0.77 |
   | `\u` dump | 0.70 |
   | same `\x` content, no line breaks | 0.31 |
   | quoted lines (`"…"⏎`) | 0.29 |
   | `0x..` list | 0.27 |
   | continuous hex lines | 0.25 |
   | random letters | 0.14 |
   | `"ha"` ×200 | 0.61 |
   | `"he"`, `"la"`, `"na"`, `"lol"` ×200 | 0.18–0.26 |
   | `"the"` ×200 | 0.01 |

3. **Corpus share.** Among + tokens ≥ 0.5 at positions ≥ 64, 53% have a long hex/digit/base64 run in
   the preceding 80 characters (random tokens: 1%). At ≥ 0.3 the share is only 18%.
4. **Circuits** (`probes/…_pos/neighbors.json`, per-branch single-unit zeroing). Both hex-dump
   variants depend on the same feeders: L7/N66 (−20–26% of the activation when zeroed), L7/N6753
   (−19–27%), L9/N5783, L5/N6527 and L6/N7892. Those units' own top tokens are literal `\\n`
   escapes. The `', '.join` branch depends on an entirely different set: L5/N8154 (−53%),
   L6/N8771 (−49%), L6/N347 (−45%), L7/N8406 and L5/N9562. The "ha" branch depends on neither set.
   So these are separate jobs that converge on one channel.
5. **Inverse search (SciPy DE, `invert_de.json`).**
   - The max passage reached 0.82 on the `\⏎` of a hex dump.
   - No "leak" passage was found: the best was 0.35 with broad relevance 0.25.
   - Near-miss: `', '.join` placed after long repeated prose rose to **0.65**, up from 0.49 alone.
     The join offshoot is context-amplified.
   - Silent-on-concept found a base64 block that stayed at 0.16.

### − pole

1. **Pre-registered tests** (commits fe255af, 714e27d, 0963733):
   - H-1, number or code followed by a delimiter or unit: **8/8 fire, 0/8 false positives**.
     Examples: `[[10, 20, 30]⟦],⟧` 1.06, `24 x 36 ⟦inches⟧` 0.75, `9x13 ⟦inch⟧` 0.65.
   - R5 sub-jobs:

     | Sub-job | Fire | Notes |
     |---|---|---|
     | dimension_unit | 5/6 | `8.5 x 11 ⟦inch⟧` 0.78 |
     | nested_row_close | 6/6 | `[[1, 0, 1]⟦],⟧` 0.97; word rows `[['a','b']⟦'],⟧` 0.91 |
     | long_id_digits | 6/6 | `3f9a2c71e8b4d⟦0⟧` 0.92 |
     | code_arg_newline | 3/6 | failed, as predicted |

     All quiet controls stayed ≤ 0.11.
2. **Break attempts that worked:**
   - Number words: "ten by fifteen inches" 0.19, against 0.65 for "9x13 inch".
   - Version strings: "3.11.4," 0.07.
   - Coordinates: 0.25. Room/building numbers: 0.21.

   So "any number" is too broad (R3: 4/8 fire). In the circuit flip test, "9x13 inch" (2.86 raw)
   dropped to 0.26 when written as "nine by thirteen inch".
3. **Circuits.** Each sub-job has its own feeders:

   | Sub-job | Main feeders |
   |---|---|
   | dimension unit | L11/N4300 (top tokens " x", "×", " square"; −16%), L8/N8558 |
   | long ID digits | L9/N4090, L11/N8833 |
   | code-argument newline | L5/N6990 (−30%), L7/N8077 |
   | nested row closer | L9/N2392 |
   | phone delimiter | no strong single feeder |

   Overlap between these sets is close to zero.
4. **Solo bank** (`solo.json`, scored as the opposite pole): Spearman 0.78 and AUROC 1.0 between
   relevance to the − label and activation; strong-off leak rate 3%.

### Output side (induce)

Clamping either pole on neutral prose prompts has almost no effect: KL ≤ 0.014 at c = ±10, and
generations are essentially unchanged. At large +c, line-break punctuation (`...⏎⏎`, `,⏎⏎`) gains
slightly. At −c (which steers toward the − pole), " end", " last", " bottom" and "1" gain. Neither
effect is concept-specific. Neutral prompts are the wrong context for a data-dump neuron, so I give
induce little weight.

## Labels rejected along the way

| Rejected label | Evidence |
|---|---|
| + "sentence-final period ending a factual sentence (places, dates)" | Pre-registered H+2 (fe255af): **0/8** fire. The corpus top examples are window-start artifacts that collapse after a preamble (0.95 → 0.16). |
| + "sentence boundary / sentence-initial connective" | Corpus rates (`work/boundary.py`): " In" fires ≥ 0.3 at 0.27% sentence-initially and 0.26% mid-sentence; " There" 0.19% vs 1.3%. These tokens are 10–70× enriched, but sentence position is not the variable. |
| + "closing brackets / continuation indentation of multi-line code calls" | H+3: **0/8** in short context. |
| + "Python string separator before .join" as the main label | H+1: **2/8**. Only the `', '` form fires (0.49, 0.44); `' '`, `'/'`, `'-'` and `'\t'` stay ≤ 0.15. Kept as an offshoot. |
| + "any long non-linguistic token run" | Pre-registered holdout (17c2fb4): only **2/8** predicted-fire items reached 0.3 (octal dump 0.47, binary 0.56). md5 list, UUIDs, a long hex literal, protein and "la"/"na" stayed at 0.16–0.24. Accuracy 0.62. |
| + "line breaks in backslash-escape dumps" (narrow) | R3 (714e27d): 8/8 fire, but **3/8 false positives**: sha1 hash list 0.46, the same dump without line breaks 0.46, 0x byte list 0.40. Broadened to "hex/byte-data dumps". |
| + final label as a predictor of natural moderate activations | Blind labelling (9b9c6f4, predictions 5f831a4): **0.45 accuracy** (chance) on 20 corpus tokens ≥ 0.4 versus 20 same-token negatives. None of the 20 sampled hits was a hex dump. |
| − "any digit-bearing number or code" | R3: **4/8** (versions 0.07, coordinates 0.25, room numbers 0.21). |
| − "delimiters in lists require numeric items" | H-2: flat word lists ≤ 0.28, but nested word lists fire 0.87 on `'],`. The row-closer job does not depend on digits. |
| − "comma-newline continuing a multi-line argument list" as a reliable job | R5: **3/6**; the hits involved `%`-format tuples. A weak offshoot at most. |

## Pre-registered tests

| Test | Commit | Prediction | Result |
|---|---|---|---|
| H-1 − number/code → delimiter/unit | fe255af | survives | **survived** 8/8, 0 FP |
| H-2 − word-only code lists | fe255af | fails (≤ 3/8) | **failed** 1/8, but nested `'],` fired 0.87 |
| H+1 + `.join` separator | fe255af | survives weakly | **failed** 2/8 |
| H+2 + prose sentence end | fe255af | fails | **failed** 0/8, as predicted |
| H+3 + multi-line code closers | fe255af | fails | **failed** 0/8, as predicted |
| Bank solo + holdout (long non-linguistic runs) | 17c2fb4 | holdout accuracy ≥ 0.75, ≥ 6/8 fire | **failed**: accuracy 0.62, 2/8 fire. Bank AUROC 1.0 but Spearman 0.46 (predicted > 0.5). Prose-boundary hypothesis Spearman −0.48, as predicted. |
| R3 + line breaks in escape dumps | 714e27d | survives | 8/8 fire but 3/8 FP → **failed** |
| R3 − digit-bearing numbers/codes | 714e27d | survives | **failed** 4/8 |
| R4 blind corpus labelling | 9b9c6f4 (+ 5f831a4) | + 0.55–0.70, − 0.65–0.80 | + **0.45** (missed), − **0.675** (met) |
| R5 − sub-jobs | 0963733 | dims / rows / IDs survive, code-arg fails | **all four as predicted** (5/6, 6/6, 6/6, 3/6) |

The blind-test answer key was kept outside the repo until the predictions were committed. Its
sha256 (d237cd1e…) is recorded in the pre-registration.

The budget allowed one globalMOO model; I used none. The DE inversion answered the inverse questions,
and the long-text bank made a remote DOE expensive and memory-risky.

## Monosemanticity verdicts

- **+ pole: several jobs; confidence 0.55.** At the top of its range one job dominates: row-ending
  line breaks in long hex/byte dumps. It ramps with dump length, exceeds the corpus max on synthetic
  dumps, and has a coherent shared feeder set. The `', '.join` separator and the "ha ha" ramp are
  separate offshoots with disjoint circuitry. Most activations between 0.3 and 0.5 are an
  unexplained diffuse tail: blind labelling scored at chance and only 18% of ≥ 0.3 hits are in data
  dumps. So "one main job with offshoots" describes only the top of the range. Confidence is moderate
  because the tail might hide a single variable I did not find.
- **− pole: several jobs; confidence 0.65.** A shared surface theme (values in structured
  numeric/code text) holds the sub-jobs together loosely. Each sub-job (dimension units, nested-row
  closers, long-ID characters, phone and citation delimiters) is reliable on its own: 5–6 of 6 fire
  with 0 false positives. But each has its own upstream feeders, and obvious members of the theme,
  such as version strings and coordinates, do not fire. Blind accuracy of 0.675 fits a family of
  detectors better than one crisp rule. One could argue for "one main job with offshoots" under a
  broad label like "boundary after a value in structured data"; that broad label is what failed R3.

## What I would test next, and what I am least sure of

- **Least sure:** the nature of the + pole's diffuse 0.3–0.5 tail. It includes " In", " As",
  " However", " There", "red(ness)" and scattered code tokens. These depend on long context, and I
  could not reproduce them with synthetic sentences.
  - Next: sample 200 tail tokens and run attribution to see whether they share feeders with the
    hex-dump branch (L7/N66, L7/N6753) or with the join branch.
- Whether the "ha ha" ramp is about laughter or about a particular repeated-token pattern: "ha"
  gives 0.61, while other syllables give 0.18–0.26 and "the" gives 0.01. Test "haha", "HA HA", and
  laughter in other languages.
- Why version strings and coordinates do not drive the − pole while long IDs, dimensions and phone
  numbers do. A hypothesis to pre-register: decimal-point segmentation (`3.11.4`) breaks the "value
  token run" that the feeders track.
- Downstream role: zeroing the neuron changes later units by at most 4% of their max. A
  logit-lens check inside real hex-dump contexts (does it promote `\x` / `\⏎` continuations?) would
  test whether the + pole helps continue dumps. Neutral-prompt induce could not show this.
- The window-start artifact affects both poles. Any future scan-based label for this neuron should
  exclude positions below 64.

## Files

- `probes/bonsai1_L12_N5397_pos/`: forward, bank, solo + holdout, quick tests (R1, R3, R5) and
  results, neighbors, induce, invert_de, `predictions_round1.md` (running pre-registration log).
- `probes/bonsai1_L12_N5397_neg/`: forward, neighbors, induce for the − pole.
- `work/`: exploration scripts and inputs, blind-test items, predictions, key and score.
