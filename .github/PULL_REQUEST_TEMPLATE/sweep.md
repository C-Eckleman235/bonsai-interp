<!-- PR shape for a SWEEP: many poles, one pre-registration commit, one batch run.
     Title: "sweep: 16 poles (<held> held)". `play run` prints the table rows below; paste them. -->

## Sweep: <!-- N --> poles, <!-- held --> held

- **Agent:** <!-- e.g. claude-code opus-5.5, codex, human -->
- **Pre-registration commit:** <!-- hash: every tests file, committed together and alone -->
- **Results commit:** <!-- hash, a LATER commit -->
- **Explored before pre-registering:** <!-- e.g. cards only / probe --plan on 10 poles -->

| pole | label | unit kind | AUROC | median fire | near misses ≥ 0.30 | outcome |
|---|---|---|---|---|---|---|
| <!-- bonsai1/L__/N____± --> | <!-- short label --> | <!-- lexical / contextual / … --> | | | | <!-- held / failed --> |

## Notes

<!-- anything a reviewer should know: poles marked dense / positional, labels you are unsure of, decisions you made -->

## Checklist

- [ ] Only contribution files are added (a tests file and a results file per pole, under `graph/contributions/…`). No derived files.
- [ ] The tests files were committed together, before any of them was run, and nothing was re-tested after seeing results.
- [ ] Near misses keep the topic, length and style and drop only the claimed trigger.
- [ ] Failed labels are included, not dropped.
