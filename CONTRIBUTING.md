# Contributing

The procedure lives in **[AGENTS.md](AGENTS.md)**, written so an AI agent can follow it end to end. People follow the same steps. The survey page gives you the prompts; `python -m bonsai_interp.graph.play brief <mission> --neuron <id> --handle <you>` prints a complete mission. The rest of the reference material:

| Document | What it covers |
|---|---|
| [AGENTS.md](AGENTS.md) | Setup, missions, the rules, and how to test well |
| [graph/SCHEMA.md](graph/SCHEMA.md) | Contribution files, protocols, and how records are folded |
| [DATA_POLICY.md](DATA_POLICY.md) | What goes in git, what goes in the library, and what goes nowhere |
| [MERGE_AGENT.md](MERGE_AGENT.md) | How PRs are triaged and merged |

## The short version

1. Claim a neuron pole. Write the label, 8 sentences that should fire it and 8 near misses, and commit that file on its own (`play new`).
2. Run it (`play run`; several files at once load the model only once). Commit the results in a separate commit.
3. Open a PR containing only files under `graph/contributions/`. The merge agent checks the commit order, merges, and regenerates the graph.

There are three PR shapes:
- **interpret** and **qc** add contributions.
- **tool** adds one screen or analysis atomically.
- Anything else is structural and goes to a human.

Missions run at quick depth by default (about 5–10 minutes per neuron).

Blind clean-room runs ([neuron-cleanroom](https://github.com/moridinamael/neuron-cleanroom)) are the expensive check. They are used as sampled QC, not as the default way to label.
