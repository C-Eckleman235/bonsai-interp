# bonsai_interp.probe: read one neuron forwards, backwards from text, and by induction

Point the probe at a neuron pole. It produces a dossier answering one question: does this neuron do one job?

```
python -m bonsai_interp.probe.run forward 9/3688 +        # corpus top examples + bank_request.json
#  -> LLM stage: write probes/<id>/bank.json (label hypotheses, ~100 rated sentences)
python -m bonsai_interp.probe.run solo    9/3688 +        # bank dose-response, hypothesis competition
#  -> LLM stage (optional): refine the label, write holdout.json with predictions BEFORE running solo again
python -m bonsai_interp.probe.run invert  9/3688 + --solver de     # local SciPy baseline (cheap, unlimited)
#  -> write gmoo_plan.json (hypothesis + predictions), then ONE globalMOO model per neuron:
python -m bonsai_interp.probe.run invert  9/3688 + --solver gmoo
python -m bonsai_interp.probe.run invert  9/3688 + --solver gmoo --attach --goals tight_25,tight_50,tight_75 --tag _tight
python -m bonsai_interp.probe.run induce  9/3688 +        # steering sweep + generations
#  -> LLM stage: induce_judgement.json, verdict.json
python -m bonsai_interp.probe.dossier 9/3688 +            # probes/<id>/dossier.html
```

**Defaults that matter**
- Every probe text is read after neutral preambles (mean of 3), on its own tokens only. Bare sentences put tokens at early positions, where many neurons are damped.
- The inverse searches only natural text. Passages are drawn from the bank, and each of 3 slots has a (concept, strength) coordinate. That keeps the space smooth enough for globalMOO's learned inverse.
- globalMOO is used for one model per neuron, trained once and queried for the whole goal battery. Bulk screening of many neurons uses `bonsai_interp.screen` and the SciPy solvers only.

**Which solver for which question**
- globalMOO: graded targets, offshoots, sense flips, and multi-constraint questions inside the trained envelope. Few evaluations per query.
- Differential evolution: maxima and adversarial "fire without the concept" searches.
