"""The interp graph: one JSON record per neuron pole under graph/neurons/<model>/, plus graph/edges.json.

Records are plain JSON so contributors can add neurons by pull request. `validate` checks every
record against the schema in graph/SCHEMA.md; `check_prereg` checks that a pull request's
test files were committed before its result files. Neither needs torch or the model.
"""
