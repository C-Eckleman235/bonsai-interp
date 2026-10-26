"""Weights-only census: stage one of the deterministic harness (docs/HARNESS.md, docs/CENSUS.md).

For every pole of Bonsai-8B, before any corpus text is read for activations:
- weights.py: direct logit effect (sweeplib's predict.direct_effect), output norm, share of the output in the
  unembedding's effective null space (entropy neurons), correlation of the logit effect with unigram log-frequency
  (token-frequency neurons), each against random-direction controls;
- connect.py: the MLP-to-MLP weight connection matrix P[B, A] = (W_gate_B ĝ_B v_A)(W_up_B ĝ_B v_A), top readers and
  writers per pole;
- lens.py: a steer lens: push each neuron ± by a dose in units of the residual norm at its layer (calibrated per layer
  on random directions), record the next-token Δlog-prob and KL on a few neutral contexts.
run.py drives all GPU stages in one process (one batch-slot hold); summary.py builds the summary and the validation.
"""
