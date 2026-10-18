# Pre-registration 4: final adversarial round + the single globalMOO model (written before running)

Hypotheses after rounds 1–3 and DE inversion:
- **+ pole** = a merged unit with separate input feeders: (i) "nerve(s)" word family (multilingual) whenever
  the LEFT context is not somatosensory/pain discourse. This is the main job by peak; it covers stress
  idioms, nerve agents, nerve damage, and even anatomical "nerve" when the preceding text is neutral.
  (ii) physical/material "strong" in concrete text. (iii) the body part or injury word in violent-harm
  descriptions. All three are vetoed by preceding pain/sensation discourse, which flips the neuron to −.
- **− pole** = somatosensory/nociceptive physiology, plus an offshoot: withdrawing a hand or body part
  from contact (fed by L8/N8342 rather than by the pain feeder).

Predictions:
- P4A (`final4.json`): nerve with neutral left context: ≥ 3/4 reach + ≥ 0.3. Nerve with sensory/pain left
  context: ≤ 1/4 reach + ≥ 0.15.
- P4B: contact-withdrawal: ≥ 3/4 reach − ≥ 0.3. Non-contact hand/pull-back movement: ≤ 1/4 reach − ≥ 0.15.
- P4E: hand/leg after violence ≥ 0.25 on + (2/2); neutral body part < 0.15 (2/2).
- P4C (`ctx4.json`): after the calm context, physical strong gives mean + ≥ 0.3 and abstract strong gives mean + < 0.2.
  After the business context both are < 0.2.
- P4D, corpus-wide (6,000 windows, every " nerve"/" nerves"/" Nerve" token, `probes/tools/nerve_gate.py`):
  tokens with a pain/sensation word (pain, sensation, numb, tingl, burning, receptor, touch, ache, sensory)
  in the previous 40 tokens have lower + activation than tokens without one. AUROC(no-pain-context >
  pain-context, on the + value) ≥ 0.75, and the median − activation is higher in the pain-context group.
- P4G, the single globalMOO model (+ pole, all non-tight goals plus tight_50): leak / offshoot are satisfied
  again by passages whose + peak is on " nerve" (or "strong"), not by a new family. silent_on_concept is
  satisfied by a passage that puts a pain/sensation sentence before the on-concept sentence.
