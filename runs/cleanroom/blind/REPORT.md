# Bonsai-8B (1-bit) · L9/N3688: report

Both poles were studied. Corpus maxima: + pole 7.66, − pole 19.0 (the − pole is about 2.5× larger).
All activations below are fractions of the pole's corpus max, read after the standard neutral preambles
unless stated otherwise.

## Final labels

| Pole | Label | Verdict | Confidence |
|---|---|---|---|
| **L9/N3688−** | **Somatosensory / nociceptive physiology**: the body detecting and transmitting pain, touch, pressure, temperature, tingling and burning (receptors, fibres, "pain signals", sensation). **Offshoot:** pulling or jerking a hand or limb back or away. | **One main job with an offshoot** | 0.8 |
| **L9/N3688+** | **Several jobs merged in one channel**, all silenced by preceding pain/sensation discourse: (1) *main, by peak:* **"nerve(s)" outside neuroanatomy**, in several languages (nerve-racking, bundle of nerves, Nerven, nerveux, нервничать, 신경, nerve agent/toxin, "nerve damage" listed as a disease complication); (2) **physical/material "strong"** in concrete text (rope, beam, cable, wind, joints); (3) *weak:* the **injury word or body part in violent harm** (shot in the *leg*, stabbed in the *arm*). | **Several jobs** (one main job plus two separately fed offshoots) | 0.7 |

The two poles behave as one signed axis over a "body / nerves / force" vocabulary. Somatosensory
discourse in the left context pushes the neuron negative; the same words outside that discourse push it
positive. On the token " nerve", the + and − values are almost perfectly anti-correlated, and the sign
follows what came before it.

## Pre-registration commits

| Commit | What was pre-registered |
|---|---|
| `9f80ff5` | Round 1: banks with competing hypotheses, blind held-out sets (written by a sub-agent from label text only), quick tests, predictions (`probes/PREREG_1.md`) |
| `0b5703a` | Round 2: context manipulation (`probes/PREREG_2.md`, `probes/tests/ctx2.json`) |
| `6f430e5` | Round 3: circuits and induce (`probes/PREREG_3.md`, `neighbors_request.json`) |
| `9c224a0` | Round 4: final adversarial set, corpus-wide gate test, the single globalMOO run (`probes/PREREG_4.md`) |

Results files: `probes/RESULTS_1.md`, `probes/RESULTS_2.md`, `probes/tests/*_out.json`, and each probe
directory's `solo.json`, `quick.json`, `neighbors.json`, `induce.json`, `invert_de.json` and `invert_gmoo.json`.

## Key evidence: − pole

- **Top corpus examples** are all pain physiology or withdrawal: "blocking pain ⟦sensation⟧", "tingling,
  numbness, hot and ⟦cold⟧", "withdrawing their ⟦hand⟧ when it's squeezed", "pain ⟦signals⟧ are first processed".
- **Blind held-out set** (24 sentences written by a sub-agent from the label text alone): **accuracy 0.96,
  AUROC 1.00.** 11/12 predicted-fire sentences reached ≥ 0.3, many above the corpus max: "Nociceptors in the skin
  detect tissue damage…" 1.75; "Capsaicin binds to TRPV1…burning" 1.65; "withdrawal reflex jerks your hand away" 1.56.
  Spanish 1.19 and German 0.93 also fired. Sentences with no pain/nerve/receptor word fired too ("Skin endings
  sensitive to stretch and heat relay their messages…" 0.83). 0/12 predicted-quiet sentences fired: emotional pain
  0.08–0.10, "my tooth hurts" 0.04, retina 0.04, cochlea 0.05, cash withdrawal 0.02, weather 0.01, "pain in the neck" 0.02.
  The one miss was "Cold-sensitive endings in the skin fire faster…" at 0.18.
- **Quick tests:** label Q-1 passed 8/8 fire with 0/8 false positives. Competitor Q-2 "pain in any sense"
  **failed** at 0/8: "the pain of her betrayal" 0.11, "toothache" 0.02, "my back is killing me" 0.01; only
  "winced in pain" reached 0.28. Competitor Q-3 "physical withdrawal in general" **failed** at 1/8: "stepped back"
  0.02, "army withdrew" 0.20, "leaned back" 0.00.
- **Bank (107 sentences):** AUROC for M 1.00, versus 0.968 for "pain any sense", 0.956 for "medicine/neurology"
  and 0.918 for "withdrawal". The rank order matches the prediction, but the margins are small on this bank.
  The held-out and quick tests separate the hypotheses far more sharply.
- **Induce:** steering toward the − pole makes the model write pain physiology and reflexes. At c=5, "He looked at the
  results and said" continues with *', "Ouch!" without thinking. So, the pain is transmitted through the nerve
  fibers…'*. Top gained tokens: heat, pain, touch, temperature, skin, tactile, sensation, stimulus, reflex, noc-, itch,
  withdrawal. The direct logit effect is weak: KL is 0.01 at |c| ≤ 2. Downstream effects are real: zeroing the neuron
  moves L11/N6314 by 0.26 of its max.
- **Circuit:** pain and touch examples are fed by L8/N3858 (a "pain" neuron; 0.34 drop on pain-mechanism,
  0.14 on touch/temperature) and L8/N2325 (pressure/force). Withdrawal examples use **different** feeders:
  L8/N8342 (0.53 drop) and L5/N8836 (whose − peaks are " back"). L8/N3858 gives −0.003 on withdrawal. This is why
  I count withdrawal as an offshoot rather than part of the core.
- **Corpus share:** of tokens at ≥ 0.5, 34% are sensation-mechanism words and 29% are "pain". " pain" is the
  most common peak token, but only 22% of all " pain" tokens reach ≥ 0.2, consistent with the failure of the
  "pain in any sense" label.

### The withdrawal offshoot, and where it breaks the label
"Pulled her hand away from the flame" gave 0.98. My refinement "withdrawal *triggered by a stimulus or contact*"
was tested and **failed**:
- "She **smiled** and pulled her hand away" still reached 0.63 (P3d predicted < 0.15).
- "She pulled her hand out of her pocket and waved" reached 0.73.

Meanwhile "pulled the drawer back" was 0.09, "raised his hand" 0.09 and "the general pulled back his troops" 0.10.
So the offshoot is closer to **"a person pulling a hand or limb back or away"**, a body-movement pattern, than to
reflex physiology. Contact-withdrawal sentences did fire as predicted (3/4 ≥ 0.3; "jerked her foot back when the
cold water touched her toes" 1.08).

## Key evidence: + pole

- **Top corpus examples:** 21 of the top 32 are the nerve family (nerve-racking ×8, Nerven, nerveuse, nervio,
  нервно, عصبية, 신경 ×3, neuro-). Seven are " strong" or "fuerte". The rest are " sensitive", 감정 and 感覺.
- **Frequency tells a different story from the top-32.** Over 6M tokens, " strong" is the most common peak at
  every threshold. At ≥ 0.3: strong 26%, injury/violence words 16%, sensation/reaction words 12%, nerve 10%.
  At ≥ 0.5: nerve 42%, strong 26%. 35% of all " strong" tokens reach ≥ 0.2.
- **Not a lexical "nerve" detector.** The median + activation on " nerve" is 0.00, and only 16% of " nerve"
  tokens reach ≥ 0.2. Q+4 "nerve in any sense" failed at 0/8: optic nerve 0.09, sciatic 0.08, "nerve pain relief"
  0.01.
- **Nerve sub-label:** Q+1 (stress/courage idioms) passed narrowly at 5/8, 0/8 false positives: "nerve-racking
  experience" 0.89, "Nerven wie Drahtseile" 0.68, "bundle of nerves" 0.56. The misses were "gets on my nerves" 0.12,
  "me tape sur les nerfs" 0.15 and "takes real nerve" 0.27. On the blind held-out set the nerve items went 4/4
  fire, including "worked up the nerve" 0.74 and "Nerven völlig blank" 0.56, and 4/4 quiet.
- **Emotion without the word stays silent:** "nervous" 0.01–0.06, "anxious and restless" 0.02, "palms were
  sweating" 0.07. The unit is not an anxiety detector.
- **"strong" depends on the word sense and on register** (round 2 and P4C). After a calm narrative sentence,
  physical "strong" averages 0.44 (cable 0.59) and abstract "strong" averages 0.09 (evidence 0.06). After a
  business-report sentence, physical "strong" drops to 0.26. Danger or strain context adds nothing beyond calm
  narrative (danger−calm Δ = −0.007).
- **Violent harm:** this branch is reliably above baseline but weak. Q+3 failed at 1/8 ≥ 0.3, yet all 8 fire
  items (0.12–0.31) scored above all 8 quiet items (≤ 0.06), and metaphorical uses ("hit song", "inflation hit")
  stayed ≤ 0.06. The peak is usually on the body part ("shot in the ⟦leg⟧" 0.48).
- **The pain-context veto (strongest effect found):** one preceding sentence about pain, tingling and numbness
  drops every + target to about 0. The mean change was −0.46 for nerve items, −0.35 for strong and −0.26 for injury,
  while the − pole rose by +0.16 to +0.20 on those same items. DE inversion found the same thing:
  "nerve-shredding job" scored 0.46 alone and 0.04 after "Thin C fibres carry slow, burning pain…". The reverse
  case is the DE "leak": "When a ⟦nerve⟧ becomes irritated, it may send pain signals…" fires + at 0.42, because at
  the token " nerve" the pain words have not appeared yet.
- **Circuit: separate jobs.** P3a passed cleanly. Each branch has its own feeders, and they give almost zero
  drop on the other branches:

  | feeder | nerve | strong | injury | its own corpus peaks |
  |---|---|---|---|---|
  | L8/N7756 | **0.635** | 0.001 | 0.001 | عصبية / Arabic nerve morphemes |
  | L8/N9260 | 0.004 | **0.289** | 0.013 | Powerful, potent |
  | L0/N8811 | 0.005 | **0.289** | 0.007 | (−) Strong |
  | L6/N6692, L7/N6823, L7/N6002 | ≈0 | ≈0 | ~0.10 each | diffuse |

  The pain veto works through the − pole's own inputs. Among the top disambiguators is the pain neuron
  L8/N3858, and group-zeroing the top 10 restores 47% of the gap.
- **Induce:** steering toward + produces no readable concept. Generations stay generic up to c=10, KL is 0.06 at
  c=10, and p(nerve/strong tokens) does not rise (P3e failed for this pole). Downstream effects are small: the
  largest is 0.06 of max.
- **Blind held-out set:** accuracy 0.62 and AUROC 0.84, below my pre-registered 0.65. Nerve items were perfect.
  The strong and injury items were all 0.05–0.22, so they ranked correctly but sat under the 0.3 threshold.
  P-feeling (感覺, 감정, чувствовать, ressentir) was 0/4, as predicted.

## Rejected labels and the evidence against them

**+ pole**
1. *"the word nerve, any sense"*: Q+4 0/8; corpus median 0.00 on " nerve"; anatomy and pain contexts are silent or negative.
2. *"nervousness / anxiety in general"*: "nervous" 0.01–0.06; anxious and sweating-palms sentences ≤ 0.07.
3. *"physical/material strength" as the whole + pole*: Q+2 1/8 in isolation; besides, nerve items reach higher.
4. *"violent bodily injury" as a label in its own right*: Q+3 1/8 ≥ 0.3; P4E 0.22/0.29, still under the 0.25 bar.
5. *"visceral physical intensity: danger / strain / arousal"* (H+ctx): danger−calm Δ −0.007 on strong, +0.026 on
   injury, −0.057 on nerve (stress context). A calm cat-on-the-windowsill sentence works as well as a danger sentence.
6. *"feeling / sensation words in other languages"*: blind P-feeling items 0/4 (0.03–0.22).
7. *"one unified job (U)"*: the label ranked the bank well (AUROC 0.97), but the held-out accuracy was 0.62 and the
   three branches have disjoint feeders.
8. *"nerve whenever the left context has no pain words"* (my round-4 refinement): corpus AUROC 0.68 < 0.75 (P4D
   failed), and only 17% of such tokens reach ≥ 0.2. Neuroanatomy discourse in general suppresses the + pole, not
   only explicit pain words.

**− pole**
1. *"pain in any sense"*: Q-2 0/8; held-out emotional and complaint pain ≤ 0.10.
2. *"medicine / neurology in general"*: optic nerve 0.21, retina 0.04, cochlea 0.05, insulin 0.01, metformin 0.02.
3. *"physical withdrawal in general"*: Q-3 1/8; troops, curtain and drawer sentences all ≤ 0.20.
4. *"stimulus-triggered withdrawal reflex"* as the offshoot's definition: "smiled and pulled her hand away" 0.63 and
   "pulled her hand out of her pocket" 0.73 broke it.

## Key pre-registered tests (scorecard)

| Test | Prediction | Result | Held |
|---|---|---|---|
| − blind held-out | acc ≥ 0.75 | 0.96, AUROC 1.00 | ✔ |
| + blind held-out | acc ≥ 0.65 | 0.62, AUROC 0.84 | ✘ |
| Q-1 / Q-2 / Q-3 | survive / fail / fail | 8/8·0FP / 0/8 / 1/8 | ✔✔✔ |
| Q+1 / Q+2 / Q+3 / Q+4 | survive / survive / survive / fail | 5/8 / 1/8 / 1/8 / 0/8 | ✔✘✘✔ |
| P2a–c danger/stress context raises + | Δ ≥ +0.05–0.08 | ≈0 or negative | ✘ |
| P2c pain context flips nerve | Δ+ ≤ −0.05, Δ− ≥ +0.05 | −0.46 / +0.16 | ✔ |
| P3a disjoint + feeders | top-3 overlap ≤ 1; nerve feeder < 0.2 on strong | 0 overlap; 0.001 | ✔ |
| P3c pain feeder absent from withdrawal | ≥ 0.15 vs < 0.10 | 0.34 vs −0.003 | ✔ |
| P3d "smiled" silences withdrawal | flipped < 0.15 | 0.63 | ✘ |
| P4A nerve with sensory left context quiet | ≤ 1/4 ≥ 0.15 | 0/4 (and − 0.37–0.70) | ✔ |
| P4A nerve with neutral left context fires | ≥ 3/4 ≥ 0.3 | 1/4 | ✘ |
| P4C calm physical vs abstract strong | ≥ 0.3 vs < 0.2 | 0.44 vs 0.09 | ✔ |
| P4D corpus nerve gate | AUROC ≥ 0.75 | 0.68 | ✘ |
| P4G globalMOO finds leak via nerve | leak satisfied | not satisfied (collapsed) | ✘ |

Note on P4D: the first run had a bug. " Nerve" tokenizes as " N" + "erve", so the script matched every " N"
token. I fixed it to the exact single-token ids for " nerve" and " nerves", logged the fix in the script, and
re-ran. Both runs fall below the pre-registered bar (0.60 before the fix, 0.68 after).

**Inverse solvers.** DE (624 evaluations per goal) satisfied every + goal and every − goal except − silent_on_concept,
for which nothing was found; the lowest was 0.17. That is mild support for the − label: I could not find an
on-concept passage that stays silent. The single globalMOO model (+ pole, 9 goals) hit graded targets in 5–22
evaluations: level_25 0.235, level_50 0.458, level_75 0.728 and tight_50 0.495. It failed max, leak, offshoot and
silent_on_concept, collapsing to a degenerate "Nerva coin" passage.

## What I am least sure about

- **How to count the + pole's jobs.** By peak it is a nerve unit. By volume it is a "strong" unit, because "strong"
  is far more frequent in text. The verdict is "several jobs" because the feeders are disjoint and the induced
  output has no readable meaning. Someone could fairly call it "one main job (nerve) with offshoots". The common
  theme of physical-force words outside sensation discourse may be a real shared subspace, or it may be my
  post-hoc story.
- **What exactly suppresses "nerve" on the + side.** Explicit pain words explain only part of it (AUROC 0.68).
  Neuroanatomy framing (optic, sciatic, "the doctor tested the nerve") also suppresses it, and some neutral-left
  cases are weak ("shredded every nerve I had" 0.04).
- **Whether the − withdrawal offshoot is a real concept or a lexical pattern** ("pulled her hand").
- The neutral preambles include an office/invoice text. That probably damped the "strong" items in the quick and
  held-out tests (business context costs about 0.17 on physical "strong").

## What I would test next

1. Patch L8/N7756 (nerve feeder) and L8/N9260/L0/N8811 (strong feeders) in natural corpus passages and check that
   each removes only its own branch over hundreds of examples, not 14.
2. Map the − withdrawal offshoot with minimal pairs that vary the verb (pulled/jerked/moved/raised), the object
   (hand/foot/drawer/curtain) and whether contact came first.
3. Build a graded "neuroanatomy framing" scale for the context before " nerve", to replace the pain-word regex in
   the gate test.
4. Rerun the + held-out set with narrative-only preambles, to separate the register effect from the label.
5. Check whether the downstream reader L11/N6314 is itself a somatosensory unit, which would say what the − pole is
   *for*.
