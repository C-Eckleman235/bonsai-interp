# Pre-registration 1 — L9/N3688 (written before running any of the files named here)

Exploration so far (hypothesis generation only, not tests): top-32 examples, band sampling over
6,000 windows (`probes/notes/bands_*.json`), token-family shares (`probes/notes/families.json`),
and two exploratory sentence rounds (`probes/explore/explore{1,2}*.json`).

Key exploratory observations that shaped the hypotheses:
- On the 143 corpus tokens " nerve", the + and − poles act as one signed axis: nerve-in-pain/receptor
  context → − pole (up to 0.45), nerve-racking / nerve agent / nerve damage-in-a-list → + pole (up to 1.0).
  Median + activation on " nerve" is 0.00, so + is not a lexical nerve detector.
- " strong" is the single most frequent + token at ≥0.1/≥0.2/≥0.3; 35% of " strong" tokens reach ≥0.2.
- − pole: " pain" is the most frequent token but only 22% of " pain" tokens reach ≥0.2.

## Hypotheses

**+ pole**
- U (main): "visceral physical intensity": nerves-as-stress (nerve-racking, Nerven, 신경), physical/material
  strength ("strong"), violent bodily harm (shot/stabbed/wounded, often peaking on the body part).
- Sub-labels tested separately: Q+1 nerve-figurative, Q+2 physical strong, Q+3 violent harm.
- Competitor PL / Q+4: lexical "nerve" in any sense. **Prediction: fails** (anatomy/pain contexts quiet).

**− pole**
- M (main): somatosensory/nociceptive physiology — detecting/transmitting pain, touch, temperature,
  tingling/burning, plus the reflexive pull-away from a stimulus.
- Competitors: MP / Q-2 "pain in any sense" (**predicted to fail**: emotional/complaint pain quiet);
  MM "medicine/neurology in general" (predicted lower AUROC than M); MW / Q-3 "physical withdrawal in
  general" (**predicted to fail**).

## Tests and predictions

1. `quick` + pole (`bonsai1_L9_N3688_pos/quick_tests.json`): Q+1 survives; Q+2 survives; Q+3 survives
   (least confident, ~50%); Q+4 fails.
2. `quick` − pole (`bonsai1_L9_N3688_neg/quick_tests.json`): Q-1 survives; Q-2 fails; Q-3 fails.
3. `solo` + pole (bank.json, 107 sentences): AUROC(U) ≥ 0.80 and higher than AUROC(PL);
   AUROC(PS) and AUROC(PI) each lower than U (they cover only one branch).
4. `solo` − pole: AUROC(M) ≥ 0.85, higher than MP, MM and MW.
5. Held-out (blind sub-agent sentences, `holdout.json`, threshold 0.3):
   − pole accuracy ≥ 0.75; + pole accuracy ≥ 0.65 (P-feeling sub-label is the least supported; I expect
   most of its "fire" items to miss, i.e. P-feeling is predicted to be weak/absent at 0.3).
6. Cross-pole: on the − pole's on-concept sentences, the + pole stays < 0.1 on average, and vice versa.
