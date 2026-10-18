# Pre-registration 2: context manipulation (written before running `probes/tests/ctx2.json`)

Why: results of prereg 1 (`probes/RESULTS_1.md`). The + pole's "strong" and "injury" sub-labels failed in
isolated sentences (0.10–0.20), yet corpus "strong" reaches 0.5–0.73 deep inside documents about danger
or physical strain (stormy sea, heavy lifting, thin ice, gravity tearing comets apart). Also, on " nerve"
the sign of the neuron follows the LEFT context (pain/sensation discourse gives −, everything else gives +).

Competing hypotheses for the + pole:
- H+ctx: + pole = a "physical danger / strain / arousal" context feature, read out on intensity and
  body-harm words (nerve, strong, hit, injured). Context should raise + on the same target.
- H+lex: + pole is a token-level detector of nerve/strong/injury words; context hardly matters.

Tool: `probes/tools/ctxeval.py` (reads only the target's own tokens after the context).

Predictions under H+ctx (H+lex predicts all deltas ≈ 0):
- P2a strong branch: mean Δ(+) danger−calm ≥ +0.08, and Δ>0 for ≥ 6/8 items; same sign for strain−calm.
- P2b injury branch: mean Δ(+) danger−calm ≥ +0.05.
- P2c nerve branch: mean Δ(+) stress−calm ≥ +0.05; painphys−calm on the nerve branch gives Δ(+) ≤ −0.05
  AND Δ(−) ≥ +0.05 (sign flip by context).
- P2d sensation branch: painphys−calm Δ(−) ≥ +0.10.
- P2e controls: |mean Δ| < 0.05 for every pair on the neutral control branch; every context alone < 0.2 on both poles.
