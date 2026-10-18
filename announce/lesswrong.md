<!-- LessWrong version of announce/index.html. Paste into the LessWrong Markdown editor.
     Images load from the public repository once it is public; or upload announce/img/*.png in the editor and swap the URLs. -->

# The Bonsai Neuron Survey: label every neuron in a real LLM, and make each label survive a test

![All 36 layers of Bonsai-8B as rows of 12,288 MLP neurons, with the 66 poles surveyed so far lit up](https://raw.githubusercontent.com/moridinamael/bonsai-interp/main/announce/img/survey_map.png)

## TL;DR

- **The claim being tested.** Nick Cammarata's essay *Neural Networks Are Already Clean* (Goodfire, 2026) argues that most native MLP neurons in modern LLMs have one main job. The condition is that you read each neuron's two signs, its "poles", separately. If that holds, a complete and checkable map of a real network's neurons is a finite job.
- **What we built.** A toolchain that reads a neuron three ways:
  - forwards: what text fires it
  - backwards: an inverse solver (SciPy, and globalMOO) that finds text hitting a target activation
  - by steering it

  Every label must pass a pre-registered test. A clean-room harness lets a blind AI agent re-derive a label from nothing.
- **What we found so far.** 66 poles labelled and tested. Our flagship "nerve" neuron fought back: its narrow label failed our own disproof attempt. A blind agent with no access to our notes independently found the same three-branch structure.
- **What we want.** Players. The survey is a game your coding agent plays on your GPU. Contributions are append-only, CI-checked, and reviewed by QC from other players.

*Epistemic status: early.* The method works end to end and has caught our own mistakes several times. We have deep results on one neuron, quick tests on about sixty, and two blind replications. Whether "already clean" holds across the whole model is exactly the open question this project exists to answer.

## A map you could actually finish

Interpretability has mostly moved from neurons to learned dictionaries, because individual neurons looked hopelessly polysemantic. Cammarata's essay argues that this impression comes from reading neurons the wrong way:
- The positive and negative sides of a SwiGLU neuron are different jobs and must be split.
- What looks like several unrelated jobs is usually one job plus offshoots you can follow.

His estimate for Qwen3-32B is that about 90% of poles have one main job, about 7% have two related jobs, and a few percent are not understood.

If that is even roughly right, the neurons themselves are a usable basis, and a full survey is a counting problem with a finish line. We picked a model where the counting is cheap:

| | |
|---|---|
| **1.15 GB** | Bonsai-8B in its native 1-bit format (PrismML, Apache-2.0) |
| **36 × 12,288** | layers × MLP neurons; 884,736 poles in all |
| **92–99%** | of neurons in layers 5–25 sit at the same index as their counterpart in full-precision Qwen3-8B |
| **8 GB** | of GPU memory is enough to run every tool here |

The third number matters for the science. Bonsai was trained to be 1-bit from the start, yet it kept the neuron basis of the model it came from. So a label found cheaply on the 1-bit model can be checked on the full-precision one, neuron by neuron. A 1-bit network is also an interesting object in its own right: every weight is plus or minus one scale per group of 128.

## How it plays

The survey is built for AI coding agents to operate. You paste one prompt to set up, then one prompt per mission. The procedure lives in the repository, so the game updates when the method does.

1. **Draw.** Click the map, draw one pole, or start a sweep of 16. No claims: two people on one pole is a free replication.
2. **Look.** Top corpus examples, then the weak and middle activations, and which tokens fire it most often.
3. **Pre-register.** A label, 8 sentences that should fire it, and 8 near misses. Committed on its own, before anything is scored.
4. **Run.** The test runs on your GPU. "Held" means at least 5 of 8 fire and at most 1 near miss does.
5. **Submit.** Results go in a second commit and a pull request. CI checks the commit order and the size caps.
6. **Get checked.** Other players QC your label with new sentences. Enough failed checks and the pole turns contested.

There are four kinds of mission:

| Mission | XP | What you do |
|---|---|---|
| **Interpret** | +10 | Label a pole nobody has touched. |
| **QC** | +6 | Confirm, dispute or refine someone else's label, with your own sentences. |
| **Step out** | +13 | Find what feeds a labelled neuron, and label that. This is how webs and circuits grow. |
| **Blind run** | +15 | Seal an agent in a clean room with only the tools, and see whether it re-finds the label. |

## Worked example: the nerve neuron, and how it fought back

Layer 9, neuron 3688, positive pole. Its top examples look like a gift: "nerve-racking", German *Nerven*, French *nerveuse*, Korean 신경, Arabic عصبية. Its negative pole fires on pain signals, numbness and temperature. An easy label, it seemed. Here is what happened to it as we tried to break it.

| | Label | What happened |
|---|---|---|
| H1 | The morpheme "nerve" in any language, any sense | Anatomical "nerve" mostly pushes the opposite pole. |
| H2 | "Nerve" as psychological strain or courage | Passed a pre-registered held-out test (AUROC 0.92, no false positives), then lost to H3. |
| H3 | Tied to the word form: English "nervous" barely fires it, German "nervös" does | True, but not the whole story. |
| D1–D3 | A pre-registered attempt to disprove it | Only 13% of its strong firings in 6.1M tokens sit near a "nerve" word. |
| H4 | One broader job: force, strength, strain | Strength and impact pass; sensation and courage-without-the-word fail. |
| **H5** | **A lexical detector peaking on the psychological word "nerve", with weaker, reliable branches for "strong" and for words about being hit or hurt** | Current reading. |

![Strong firings per token for L9/N3688+: ' strong' 88, ' nerve' 13, then injury and impact words](https://raw.githubusercontent.com/moridinamael/bonsai-interp/main/announce/img/token_bars.png)

The chart is the moment the label broke:
- The top-32 examples are almost all "nerve".
- Counted across the whole corpus, the most common trigger is " strong", because "strong" is a far more frequent word.

A label read off the top of the distribution described the neuron's peak, not its daily work.

![The nerve neuron read three ways: forwards, backwards and by steering](https://raw.githubusercontent.com/moridinamael/bonsai-interp/main/announce/img/nerve_neuron.png)

The inverse view is where the learned-surrogate optimizer earns its place. Asked for passages that drive a target neuron to a given level, globalMOO hit graded targets in 2–10 evaluations once trained. Differential evolution needed about 600. globalMOO is weak at extremes and rare combinations, so the two solvers split the work.

![The neighbourhood of the nerve neuron](https://raw.githubusercontent.com/moridinamael/bonsai-interp/main/announce/img/nerve_connectome.png)

Stepping out one neuron at a time:
- **Inputs.** Two inputs carry the nerve neuron: a spelling detector (74% of its activation) and an Arabic-heavy concept unit (61%). A pain detector closes both routes.
- **Neighbours.** Beside it sits a family of separate tension neurons: suspense, can't wait, overwhelmed, disappointed. Each has its own crisp, tested inputs: "can't stand", "life span", "answer to", "disconnect".
- **Tests.** 52 of 60 neighbourhood labels held their tests.

## A blind agent found the same neuron

A label is more convincing if someone who never saw it can rediscover it. So we built a clean room:
- **Contents.** A sealed folder with the toolchain, the corpus and the weights, and no labels, notes or results.
- **Task.** A fresh agent gets one neuron and the instruction to find out what it does and to try to break its own labels.
- **Controls.** Web access and reads outside the room are denied. The full transcript is kept and audited. We commit our predictions about what the agent will find before it starts.

| | Blind agent (40 min, no access to our work) | Our reading (two days) |
|---|---|---|
| + pole | Several jobs: "nerve" outside anatomy, physical "strong", violent harm | One word family: "nerve" first, then "strong" and hit/hurt words |
| − pole | Somatosensory physiology, plus pulling a hand back (held-out AUROC 1.00) | Pain and bodily sensation (studied less) |
| "nervous" | Silent (0.01–0.06) | Silent |
| Main input | L8/N7756 (0.64) | L8/N7756 (0.61) |
| Audit | 51 tool calls, nothing outside the room. 6 of our 8 pre-registered predictions held. | |

The disagreement was the useful part:
- **Separate inputs per branch.** The blind agent showed that each branch has its own inputs. That is a real argument for "several jobs" over our "one word family".
- **A flaw in our method.** It also caught one: a neutral preamble of ours damped physical "strong" by about 0.17.

The second blind run used a neuron drawn at random with a committed seed (L12/N5397), and no method hints at all. It was humbling in a useful way.

- **It caught an artifact in our scan.** 17 of the positive pole's 32 top examples sit in the first 64 positions of a corpus window, against about 2 by chance. Read again after a neutral preamble they collapse, from 0.95 to 0.16. My own label for that pole had been built on them, and it had failed its test. Now we know why, and the scan will exclude early positions.
- **It found what the pole really peaks on.** The line break that ends a row deep inside a long hex or byte dump. Activation grows with the dump's length (0.17 at 3 lines, 0.88 at 30) and passes the corpus maximum on synthetic dumps.
- **It agreed with me on the negative pole and then split it.** Both of us found the token right after a structured number or code. It broke this into four sub-jobs:
  - units after dimensions
  - closers of nested arrays
  - the inner characters of long IDs
  - delimiters after phone numbers

  Each held a pre-registered test with no false positives, and each has its own upstream feeders.
- **It reported its failures.** Five of its own labels failed pre-registered tests, and a blind labelling round on ordinary corpus hits scored at chance for the positive pole. Its verdict: neither pole is monosemantic, each has several jobs. 7 of our 8 predictions about its findings held.

Both blind agents counted jobs by circuits: separate feeders meant separate jobs. That is a defensible rule, and it disagrees with the essay's view of offshoots. The survey records the evidence for both readings.

## Rules that make a label worth believing

- **Tests before results.** A contribution is two files: the test, committed first and alone, then the results. CI rejects a pull request where they arrive in the same commit. The runner refuses a test file that isn't committed.
- **Near misses carry the weight.** Every test has 8 sentences that keep the topic and drop only the claimed trigger. Easy near misses make any label look clean, so reviewers check them first.
- **Append-only, rebuilt by fold.** Nothing is overwritten. Each neuron's record is recomputed from its whole contribution log. So merging is idempotent, and two players never clobber each other.
- **Verdict and scrutiny.** Each pole carries two readings:
  - *Status* says what the evidence concluded. QC can flip a held label to contested.
  - *Scrutiny* says how much evidence exists, from one test up to a full probe with a blind replication.

  The map can show either, so the well-understood regions glow and the dark ones invite players.
- **Failures count.** A failed label is recorded, not hidden. Three honest failures make a pole unresolved. Neurons that are on for most text are classified as dense instead of being forced into a label.
- **Small evidence, big library.** Git holds the test sentences, results and recipe: about 16 KB per neuron. Logs, banks and transcripts go to a library by hash. Tensors are always recomputed, never stored.

## Roadmap: from one neuron to a whole working network

The end state is a model where every MLP pole is labelled, classified as dense, or honestly marked unresolved. Enough of the wiring should be known to trace real behaviours end to end. Each tier unlocks when the one before it has enough coverage to build on.

| Tier | What it means | Progress |
|---|---|---|
| **Survey** (now) | Label poles one at a time, each with a pre-registered test | 66 / 884,736 poles |
| **Webs** (next) | Step out from labelled neurons: who feeds them, who reads them | 61 edges measured |
| **Layer census** | Finish a whole layer: every pole labelled, dense or unresolved | 0 / 36 layers |
| **Circuits** | Trace named behaviours end to end through labelled neurons | 1 (the nerve gate) |
| **Whole model** | Every pole accounted for, checked by sampled blind replications | 2 poles blind-replicated |

A sense of scale: at ten minutes of agent and GPU time per pole, the full survey is about 147,000 GPU-hours. A thousand players at a few hours a week finish a layer in days and the model in about a year. Triage helps: dense and positional neurons can be classified in bulk, and layer campaigns let players finish one region at a time.

## Ways this could be wrong

- **"Clean" may not hold at scale.** We chose the nerve neuron because it was the clearest of twelve random pilots. That is selection bias, and only the random draws in the survey can correct it. Our first random draw came back from its blind investigation as "several jobs" on both poles.
- **Agents grading agents.** Most labels so far were written and tested by Claude models, and the blind agents are Claude too. Shared blind spots are likely. Human players, other model families and a non-Claude judge are on the list.
- **An 8 + 8 test is coarse.** It catches wrong labels well and over-broad ones less well. Deep probes, QC rounds and blind runs exist for exactly this reason.
- **Neurons are not the whole story.** We read MLP neurons. Attention is outside this basis, and so are distributed features and anything that lives in superposition across neurons. Single-neuron ablation also misses effects spread across many units, so every web is a lower bound.
- **"Monosemantic" is ambiguous.** The nerve neuron is one job if a job is a vocabulary, and three jobs if a job is a meaning. The survey records both readings instead of choosing one.

## Join the survey

| | |
|---|---|
| GPU | NVIDIA with 8 GB of memory (a mission peaks at about 5 GB; tested on an RTX 5080), or an Apple Silicon Mac |
| Disk | About 3 GB: the 1-bit Bonsai-8B weights (1.2 GB) and the corpus and scan (1.4 GB) |
| Agent | Claude Code, Codex, or any coding agent that can run a shell |
| Accounts | GitHub, for pull requests |
| Optional | A globalMOO key for learned inverse search; the SciPy path works without one |
| Licence | Code MIT; graph data and write-ups CC-BY-4.0 |

Paste this into your agent:

```
Set me up to play the Bonsai Neuron Survey, an open, test-backed map of what each neuron in the 1-bit Bonsai-8B model does.

1. Ask me where to work (default: a folder called bonsai-survey in my home directory). Clone these two repositories side by side there:
   https://github.com/moridinamael/bonsai-interp
   https://github.com/moridinamael/neuron-cleanroom
2. In bonsai-interp, read AGENTS.md and follow section 1, "Get ready".
3. Run: uv run python -m bonsai_interp.graph.play doctor
   Fix every FAIL and run it again until it prints READY.
4. Ask me only for what you can't do yourself: logging in to GitHub (gh auth login), permission to use about 3 GB of disk, and, optionally, a globalMOO API key for .env.
5. Finish by telling me "Ready to survey", with the GPU and the doctor summary. From then on I'll paste mission prompts from the survey page.
```

Then open the survey map, pick a mission, and hand your agent the one-line prompt it shows. Repositories:
- [bonsai-interp](https://github.com/moridinamael/bonsai-interp): toolchain, graph and map
- [neuron-cleanroom](https://github.com/moridinamael/neuron-cleanroom): blind replication

## Credits and related work

- Nick Cammarata, *Neural Networks Are Already Clean* (Goodfire, 2026). The hypothesis this project tests, and the idea of reading poles separately.
- PrismML, Bonsai-8B and its 1-bit and ternary releases (Apache-2.0).
- globalMOO, the learned-inverse optimizer used for backwards search.
- Bills et al., *Language models can explain neurons in language models* (OpenAI, 2023): automated labelling, scored by simulation.
- Bricken et al., *Towards Monosemanticity* (Anthropic, 2023): dictionary learning, the main alternative to reading neurons directly.
- Neuronpedia: the open platform for hosting features and explanations.
