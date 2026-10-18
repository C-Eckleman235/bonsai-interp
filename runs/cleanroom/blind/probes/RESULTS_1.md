# Results of pre-registration 1 (commit 9f80ff5)

Raw outputs: `bonsai1_L9_N3688_{pos,neg}/solo.json`, `quick.json`.

| # | Prediction | Result | Held? |
|---|---|---|---|
| 1 | Q+1 nerve-figurative survives | 5/8 fire, 0/8 FP → SURVIVES (just barely: "gets on my nerves" 0.12, "me tape sur les nerfs" 0.15) | yes |
| 1 | Q+2 physical strong survives | 1/8 fire (0.10–0.19 typical; abstract strong 0.04–0.15) → FAILS | **no** |
| 1 | Q+3 violent harm survives (~50%) | 1/8 fire (0.12–0.31) → FAILS; still 8/8 > every quiet item | **no** |
| 1 | Q+4 lexical nerve fails | 0/8 fire (anatomy/pain nerve 0.01–0.20) → FAILS | yes |
| 2 | Q-1 somatosensory survives | 8/8 fire, 0/8 FP | yes |
| 2 | Q-2 pain-any-sense fails | 0/8 (only "winced in pain" 0.28) | yes |
| 2 | Q-3 withdrawal-in-general fails | 1/8 ("crowd drew back" 0.38) | yes |
| 3 | + solo: AUROC(U) ≥ 0.80 and > PL; PS, PI < U | U 0.971, PL 0.910, PS 0.791, PI 0.821 | yes |
| 4 | − solo: AUROC(M) ≥ 0.85 and > MP, MM, MW | M 1.000, MP 0.968, MM 0.956, MW 0.918 | yes |
| 5 | − holdout accuracy ≥ 0.75 | 0.96 (23/24; AUROC 1.00) | yes |
| 5 | + holdout accuracy ≥ 0.65 | 0.62 (AUROC 0.84). Nerve items 4/4 fire, 4/4 quiet; strong/injury items 0.05–0.22 (all miss); P-feeling 0/4 fire (predicted) | **no** |
| 6 | cross-pole < 0.1 | + on − concept 0.041; − on + concept 0.026 | yes |

Reading: the − pole label holds cleanly. On the + pole only the nerve branch reaches 0.3 in isolated
sentences; strong and violent harm are reliably above baseline (0.1–0.3, versus about 0.02 for neutral text)
but weak. Corpus "strong" is much higher (up to 0.73), which points to a context effect → prereg 2.
