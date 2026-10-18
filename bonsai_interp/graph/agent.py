"""Agent support for the survey: `play doctor` (is this machine ready?) and `play brief` (full mission
instructions). Kept apart from play.py so a clean room can ship the contributor commands without
this file, which mentions known neurons.
"""
import datetime as dt
import json
import subprocess
import sys
import time

from bonsai_interp.graph.contrib import NEURON, ROOT, folder_for

def _ok(flag, name, detail="", fix=""):
    mark = {True: "PASS", False: "FAIL", None: "SKIP", "warn": "WARN"}[flag]
    print(f"[{mark}] {name}" + (f": {detail}" if detail else "") + (f"\n       fix: {fix}" if flag is False and fix else ""))
    return flag is not False


SPEED_REF = 3600  # full-forward tokens/s on the reference RTX 5080 (batch 8 x 256, bfloat16)
NERVE_REF = 0.4966  # the end-to-end reading on the reference RTX 5080 (CUDA, bfloat16); see tests/device_parity_reference.json


def doctor(a):
    """Check that this machine can play. Exit code 0 when every required check passes."""
    import logging
    import platform
    import shutil
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
    req = []
    v = sys.version_info
    req.append(_ok((3, 11) <= v[:2] <= (3, 13), "Python 3.11-3.13", platform.python_version(), "uv python install 3.13 && uv sync"))
    try:
        from bonsai_interp.env import DEVICE, describe, memory_gb, report, report_lines
        print("This machine (bonsai_interp/env.py; BONSAI_DEVICE=cuda|mps|cpu overrides the device):")
        for ln in report_lines():
            print(f"       {ln}")
        if DEVICE.type == "cpu":
            why = "set by BONSAI_DEVICE" if report()["device_forced"] else "no CUDA GPU or Apple Silicon found"
            _ok("warn", "device", f"{describe()}: {why}; missions would take hours")
        mem = memory_gb()
        if DEVICE.type == "cuda":
            # measured peak of any mission step (load, play run/probe, logits, the auto tier): about 5.2 GB
            req.append(_ok(mem >= 7.5, "GPU memory >= 8 GB", f"{mem:.1f} GB", "use an 8 GB+ GPU"))
        elif DEVICE.type == "mps":
            req.append(_ok(mem >= 16 and (True if mem >= 32 else "warn"), "unified memory >= 16 GB (32 GB recommended)", f"{mem:.0f} GB",
                           "use a Mac with 16 GB+ unified memory"))
    except ImportError:
        req.append(_ok(False, "torch", "not importable", "uv sync"))
    from bonsai_interp.model import MODELS
    m = MODELS["bonsai1"] / "config.json"
    shards_ok, detail = m.exists(), str(m.parent)
    idx_f = m.parent / "model.safetensors.index.json"
    if idx_f.exists():
        shards = sorted(set(json.loads(idx_f.read_text(encoding="utf-8"))["weight_map"].values()))
        have = [s for s in shards if (m.parent / s).exists()]
        size = sum((m.parent / s).stat().st_size for s in have) / 2**30
        shards_ok = len(have) == len(shards) and size > 14
        detail = f"{m.parent} ({len(have)}/{len(shards)} shards, {size:.1f} GB)"
    else:
        from bonsai_interp import gguf_weights
        g = gguf_weights.find(m.parent) if m.parent.exists() else None
        if g is not None:  # the 1.16 GB Q1_0 GGUF: the same weights, bit for bit
            size = g.stat().st_size / 2**30
            shards_ok = m.exists() and (m.parent / "tokenizer.json").exists() and size > 1.05
            detail = f"{g} ({size:.2f} GB GGUF{'' if shards_ok else '; config.json or tokenizer.json missing: rerun download_models.sh'})"
    req.append(_ok(shards_ok, "Bonsai-8B weights", detail,
                   "bash scripts/download_models.sh, or if the weights are already on this machine put BONSAI1_PATH=<that folder> in .env"))
    for f, size in (("corpus.npz", 115e6), ("scan_bonsai1.npz", 1.1e9)):
        p = ROOT / "data" / f
        req.append(_ok(p.exists() and p.stat().st_size > size, f"data/{f}", f"{p.stat().st_size / 1e6:.0f} MB" if p.exists() else "missing",
                       "bash scripts/fetch_data.sh"))
    git_user = subprocess.run(["git", "config", "user.name"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    req.append(_ok(bool(git_user), "git identity", git_user or "unset", 'git config --global user.name "…" && git config --global user.email "…"'))
    gh = shutil.which("gh")
    if gh:
        st = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
        req.append(_ok(st.returncode == 0, "GitHub CLI logged in", "", "ask the person to run: gh auth login"))
    else:
        req.append(_ok(False, "GitHub CLI", "not installed", "install gh from https://cli.github.com, then gh auth login"))
    env = ROOT / ".env"
    has_key = env.exists() and "GMOO_API_KEY=" in env.read_text(encoding="utf-8") and env.read_text(encoding="utf-8").split("GMOO_API_KEY=")[1].strip()[:1] not in ("", "\n")
    _ok(True if has_key else None, "globalMOO key (optional)", "present" if has_key else "not set; SciPy-only missions still work")
    try:
        from bonsai_interp.env import agent_clis
        clis = agent_clis()
    except ImportError:  # no torch: already reported above
        clis = {c: shutil.which(c) for c in ("claude", "codex")}
    for cli, path in clis.items():
        _ok(True if path else None, f"{cli} CLI (optional, blind missions)", path or "not found")
    cr = ROOT.parent / "neuron-cleanroom"
    _ok(True if cr.exists() else None, "neuron-cleanroom checkout (optional, blind missions)", str(cr) if cr.exists() else "not cloned next to this repo")
    if all(req) and not a.quick:
        # end to end: load the model and read one known neuron on one sentence
        from bonsai_interp import explore
        from bonsai_interp.env import DEVICE, limit_memory
        from bonsai_interp.model import load
        from bonsai_interp.probe import textinv
        limit_memory(0.8)
        t0 = time.perf_counter()
        model, tok, rec = load("bonsai1")
        load_s = time.perf_counter() - t0
        tok.padding_side = "right"
        s = explore.scan("bonsai1")
        ev = textinv.TextEvaluator(model, tok, rec, 9, 3688, float(s["top_val"][9, 0, 0, 3688]), float(s["top_val"][9, 1, 0, 3688]), 1.0)
        act = ev(["Waiting for the verdict was the most nerve-racking hour of my life."])[0]["act"]
        req.append(_ok(act >= 0.3, "end-to-end reading (L9/N3688+ on 'nerve-racking')", f"{act:.2f} of max (expected >= 0.3)",
                       "the weights or data do not match this toolchain; re-download them"))
        req.append(_ok(abs(act - NERVE_REF) <= 0.03, f"device parity ({DEVICE.type} vs the CUDA reference)",
                       f"{act:.3f} vs {NERVE_REF:.3f} (tolerance 0.03)",
                       "this device computes differently from CUDA: run tests/test_device_parity.py and report its output"))
        # speed, to compare machines: the reference RTX 5080 loads in ~15 s and reads ~3,600 tokens/s
        import torch
        from bonsai_interp.env import synchronize
        x = torch.from_numpy(explore.corpus()[:8, :256].astype("int64")).to(DEVICE)
        with torch.no_grad():
            model.model(x[:1, :32])
            synchronize()
            t0 = time.perf_counter()
            model.model(x)
            synchronize()
        tps = x.numel() / (time.perf_counter() - t0)
        print(f"[INFO] speed: model load {load_s:.0f} s; full forward {tps:,.0f} tokens/s (8 x 256); "
              f"the reference RTX 5080 does {SPEED_REF:,} tokens/s, so expect missions to take ~{max(1, SPEED_REF / tps):.0f}x as long")
        if DEVICE.type != "cuda":
            print("       fuller parity check (4 neurons, next-token logits, backward pass): uv run python tests/test_device_parity.py")
    print("\nREADY: start a mission with `python -m bonsai_interp.graph.play brief …`" if all(req) else "\nNOT READY: fix the FAIL lines above and run doctor again.")
    sys.exit(0 if all(req) else 1)


def sweep(a):
    """A sweep: k random unsurveyed poles in one branch, one pre-registration commit, one batch run, one PR."""
    py, k = "uv run python", a.k
    branch = f"sweep/{a.handle.strip('<>')}-{dt.datetime.now().strftime('%Y-%m-%d-%H%M')}"  # unique per sweep
    bslug = branch.replace("/", "-")
    out = [f"# Mission: sweep of {k} poles", "",
           f"Colour in {k} random poles that have no record yet, about 2 minutes each. One branch, one pre-registration commit,",
           "one batch run, one PR. No claims: a pole someone else also tests gets two independent results, which is good.", "",
           f"1. Branch: git checkout -b {branch}",
           f"2. Draw and create the tests files: {py} -m bonsai_interp.graph.play draw --k {k} --handle {a.handle}",
           f"   (writes one tests file per pole and lists them in .sweep/latest.txt; work on exactly those files,",
           "   e.g. git add --pathspec-from-file=.sweep/latest.txt and play run --list .sweep/latest.txt; both work in any shell)",
           f"3. Read every card in one call: {py} -m bonsai_interp.graph.play card --neuron <id1> <id2> … --out cards.txt   (no GPU: examples, density, flags,",
           "   logit effect; one call loads the scan and weights once; then read cards.txt, which is not committed)",
           f"   If it is unclear, try a few sentences: {py} -m bonsai_interp.graph.play probe --neuron <id> \"…\" \"…\"",
           "   Label at the level the card supports: usually a token or a context. A pole the card flags dense or positional",
           '   gets "unit_kind": "dense / positional" (a top-level key in its tests file) and a test of your best guess at what drives it.',
           "   It will probably fail; its job is to record the density honestly, not to find a meaning.",
           f"   To try sentences on many poles with one model load: {py} -m bonsai_interp.graph.play probe --plan plan.json",
           '   (plan.json: {"<pole id>": ["sentence", ...], ...}). Each tests file is for ONE pole, the one in its "neuron" field.',
           "4. Pre-register all of them. Each pole needs a label, 8 fire sentences and 8 near misses (same topic and style, without the trigger).",
           f"   Easiest: write one plan.json and run {py} -m bonsai_interp.graph.play fill --plan plan.json --agent \"<who you are>\"",
           '   plan.json: {"<pole id>": {"label": "...", "fire": [8 sentences], "quiet": [8 near misses], "short": "2-5 words",',
           '   "unit_kind": "lexical | contextual | sparse feature | dense / positional | unclear", "confidence": 0.6, "notes": "..."}, ...}',
           "   (fill writes LF line endings, removes the template's reference list and refuses wrong keys). Or edit the files directly:",
           "   methods.agent (who you are) and methods.screens (e.g. card, probe). quick-8+8@2 judges each: AUROC >= 0.90,",
           "   median fire >= 0.20 of the pole's max, at most 1 near miss >= 0.30.",
           "   Optional keys: notes, short, unit_kind, confidence, predictions. Delete _screens_vocabulary (it is only a reference list)",
           "   and any optional key you leave unfilled.",
           '   Commit ALL tests files together and ALONE: git add --pathspec-from-file=.sweep/latest.txt && git commit -m "Pre-register sweep tests"',
           f"5. Run them all at once (one model load): {py} -m bonsai_interp.graph.play run --list .sweep/latest.txt",
           '   Commit the results SEPARATELY: git add graph/contributions && git commit -m "Sweep results: <held>/<total> held"',
           f"6. Validate ({py} -m bonsai_interp.graph.validate), push, and open one PR:",
           f"   mkdir -p .pr && cp .github/PULL_REQUEST_TEMPLATE/interpret.md .pr/{bslug}.md (gitignored; fill it in, one line per pole: id, label, outcome), then",
           f'   git push -u origin {branch} && gh pr create --repo moridinamael/bonsai-interp --title "sweep: {k} poles (<held> held)" --body-file .pr/{bslug}.md',
           "   The merge agent merges conforming PRs once the checks pass, and regenerates the map.", "",
           "Failed labels are results too; record them and move on. Do not re-test inside a sweep: a second round is a new mission.",
           "", RULES, "", "When done, tell the person in two or three sentences what you found and give them the PR link."]
    print("\n".join(out))


RULES = """Rules (from AGENTS.md; once CI is enabled it enforces 1-3, and play run already refuses uncommitted tests):
1. Commit the .tests.json ON ITS OWN before running anything that scores it; results go in a SECOND commit. Never amend, squash or edit a committed .tests.json.
2. Never edit or delete files under graph/contributions/ (yours included). Do not commit graph/neurons, edges.json or index.json: the merge agent regenerates them on main after merging.
3. Only the two contribution files (<= 64 KB each) go in git; bulky material goes to the library (DATA_POLICY.md).
4. Report honestly: a failed label is a valid contribution. Do not rewrite a test after seeing its result."""


def brief(a):
    if a.kind == "sweep":
        return sweep(a)
    m = NEURON.match(a.neuron or "")
    if not m:
        sys.exit("neuron must look like bonsai1/L12/N5397+")
    L, n, pole, nid = int(m["L"]), int(m["n"]), m["pole"], a.neuron
    u, folder = f"{L}/{n}", folder_for(nid).relative_to(ROOT).as_posix()
    idx = json.loads((ROOT / "graph" / "index.json").read_text(encoding="utf-8"))["neurons"]
    rec = idx.get(nid)
    py = "uv run python"
    def cmds(t, tu, tp):
        gm = (f"\n   Optional (one globalMOO model for this neuron, key in .env): {py} -m bonsai_interp.probe.run forward {tu} {tp}, write the bank it asks for "
              f"(probes/…/bank.json), then {py} -m bonsai_interp.probe.run invert {tu} {tp} --solver gmoo. Use what it finds to sharpen the label before step 4."
              if a.gmoo else "")
        return gm
    gm = cmds(nid, u, pole)
    out = [f"# Mission: {a.kind} {nid}", ""]
    if a.kind in ("interpret", "step"):
        if a.kind == "step":
            out += [f"Stepping out from {nid} ({(rec or {}).get('short') or (rec or {}).get('label') or 'see its record'}).",
                    f"0. Find its strongest feeders: {py} -m bonsai_interp.graph.play step --neuron {nid}",
                    "   Choose one that has NO record in graph/index.json yet (prefer the largest drop). Call it TARGET (full id with pole, e.g. bonsai1/L8/N1234+).",
                    "   Do every step below for TARGET instead of the original neuron, and in the tests file add",
                    f'   "roles": ["feeds {nid}"]. After `play run`, add to the results file an edge',
                    f'   {{"from": TARGET, "to": "{nid}", "kind": "feeds", "weight": <drop>, "method": "single-unit zero ablation (play step)"}}.', ""]
            target = "TARGET"
            gm = cmds("TARGET", "<TARGET layer/index>", "<TARGET pole>")
            u, folder = "<TARGET layer/index>", "graph/contributions/bonsai1/<TARGET folder>"
        else:
            if rec:
                out += [f"Note: {nid} already has a record (status {rec['status']}). Consider a QC mission instead.", ""]
            target = nid
        branch = f"interpret/{target.split('/', 1)[-1].replace('/', '-').replace('+', '-pos').replace('-N', '-N')}" if target != "TARGET" else "interpret/<target>"
        bslug = branch.replace("/", "-").replace("<", "").replace(">", "")
        out += [f"1. Start a branch: git checkout -b {branch}",
                "   No claim needed: contributions never conflict, so if someone else tests the same pole, both results count (the second is an independent replication).",
                f"2. Look: {py} -m bonsai_interp.graph.play look --neuron {target}",
                f"   Look past the top examples: the weak and middle bands, and which tokens fire it most often:",
                f"   {py} -m bonsai_interp.probe.bands scan --units {u} --windows 1000 --tag mine   (about 4 minutes)",
                f"   {py} -m bonsai_interp.probe.bands sample --units {u} --tag mine --pole <+|-> --k 8",
                f"   Try candidate sentences freely before committing anything (nothing is recorded):",
                f"   {py} -m bonsai_interp.graph.play probe --neuron {target} \"first sentence\" \"second sentence\"{gm}",
                f"3. Write the pre-registration: {py} -m bonsai_interp.graph.play new interpret --neuron {target} --handle {a.handle}",
                "   Fill in: your label (at the level the test can support), 8 sentences that should fire the pole, 8 near misses",
                "   that keep topic, length and style and drop only the claimed trigger. Optional: short, unit_kind, confidence, notes.",
                "   Judged by quick-8+8@2: the 8 fire sentences must rank above the 8 near misses (AUROC >= 0.90), the median fire sentence must",
                "   reach 0.20 of the pole's max, and at most 1 near miss may reach 0.30. Clear cases make clean tests.",
                "   Also fill methods.agent (who you are, e.g. 'claude-code opus-5.5') and methods.screens (what you used; vocabulary in",
                "   bonsai_interp/graph/methods.py). play run adds what it can detect and the minutes taken.",
                f'   Commit it ALONE: git add {folder} && git commit -m "Pre-register interpret test for {target}"',
                f"4. Run it: {py} -m bonsai_interp.graph.play run <the .tests.json>  (several files at once load the model only once)",
                f'   Commit the results SEPARATELY, contribution files only: git add graph/contributions && git commit -m "Results for {target}"',
                f"5. Validate: {py} -m bonsai_interp.graph.validate",
                f"6. Push the branch and open the PR. gh's --template needs an editor, so fill a copy instead:",
                f"   mkdir -p .pr && cp .github/PULL_REQUEST_TEMPLATE/interpret.md .pr/{bslug}.md  (gitignored; fill it in), then",
                f'   git push -u origin {branch} && gh pr create --repo moridinamael/bonsai-interp --title "interpret: {target} <short label>" --body-file .pr/{bslug}.md',
                "7. If the label failed and you have a better idea, a second round is a NEW contribution (play new again) with new sentences.",
                "   After three honest failed rounds, record it and move on; that is a result too."]
    elif a.kind == "qc":
        nid_slug = nid.split("/", 1)[-1].replace("/", "-").replace("+", "pos").replace("-", "_")
        if not rec:
            sys.exit(f"{nid} has no record to QC; use an interpret mission")
        out += [f"Current label: {rec.get('label')}", f"Status: {rec['status']}; QC so far: {rec['qc']}", "",
                "1. Branch: git checkout -b qc/<neuron>. No claim needed.",
                f"2. Look at the neuron yourself, especially beyond the top examples: {py} -m bonsai_interp.graph.play look --neuron {nid}{gm}",
                "   Read its earlier tests in graph/contributions/ only AFTER you have drafted your own sentences, so yours stay independent.",
                f"3. Choose: confirm (re-test the label as it is), dispute (a test designed to break it), or refine (a better label with its own test).",
                f"   {py} -m bonsai_interp.graph.play new qc --neuron {nid} --handle {a.handle} --verdict <confirm|dispute|refine>",
                "   Write NEW sentences (8 fire, 8 near misses); do not reuse the original test.",
                f'   Commit it ALONE: git add {folder} && git commit -m "Pre-register QC of {nid}"',
                f"4. Run: {py} -m bonsai_interp.graph.play run <the .tests.json>",
                f'   Commit SEPARATELY, contribution files only: git add graph/contributions && git commit -m "QC results for {nid}"',
                f"5. Validate ({py} -m bonsai_interp.graph.validate) and push the branch,",
                f"   fill a copy of .github/PULL_REQUEST_TEMPLATE/qc.md as .pr/qc-{nid_slug}.md (gitignored), then",
                f"   gh pr create --repo moridinamael/bonsai-interp --title \"qc: {nid} <verdict>\" --body-file .pr/qc-{nid_slug}.md"]
    elif a.kind == "blind":
        run = f"runs/{dt.date.today().isoformat()}_bonsai1_L{L}_N{n}"
        words = " ".join(w for w in (rec or {}).get("short", "").replace("'", "").split() if len(w) > 3)[:80] or "<key words of the current label>"
        out += [f"A blind agent investigates {nid} in a sealed room with only the toolchain. You predict what it will find, then compare.",
                "Needs: the neuron-cleanroom checkout next to this repo and the claude CLI (play doctor shows both).",
                f"1. In neuron-cleanroom: uv run --project ../bonsai-interp python -m cleanroom.build_room --interp ../bonsai-interp --neuron {u} "
                f"--out ../rooms/L{L}_N{n} --run {run} --forbid {words}",
                f"2. Write {run}/predictions.json (probabilities for what the agent will conclude) and commit it in neuron-cleanroom BEFORE launching.",
                f"3. python -m cleanroom.launch --room ../rooms/L{L}_N{n} --run {run}   (40-120 minutes; the GPU must be free)",
                f"4. python -m cleanroom.audit --run {run}; python -m cleanroom.collect --run {run}; score your predictions in {run}/scored.json;",
                f"   python -m cleanroom.remove_room ../rooms/L{L}_N{n}   (never delete a room by hand: it links to the model weights)",
                "5. Open a PR to neuron-cleanroom with the run record, and add a 'replicate' contribution to bonsai-interp that points at it."]
    if a.kind in ("interpret", "qc"):
        if a.depth == "quick":  # ~5-10 minutes: top examples, a few probes, one pre-registered test
            out = [ln for ln in out if not any(k in ln for k in ("bands scan", "bands sample", "Look past the top examples"))]
            out.insert(2, f"Depth: QUICK (about 5-10 minutes). Look, probe a handful of sentences, pre-register one test, run it. "
                          f"Skip band scans and step-outs. Batch several neurons in one `play run a.tests.json b.tests.json ...` to load the model once.")
        elif a.depth == "standard":
            out.insert(2, "Depth: STANDARD (about 30-45 minutes). Quick, plus the band scan and sampling below, a second round if the first fails, "
                          f"and one step-out: `{py} -m bonsai_interp.graph.play step --neuron <id> --k 4`, label and test the top feeder, record the edge with `play edge`.")
        else:
            out.insert(2, "Depth: DEEP (hours; only for neurons worth it). Standard, plus a sentence bank with competing hypotheses and a held-out set "
                          f"({py} -m bonsai_interp.probe.run forward|solo L/n <pole>), inverse search (… invert --solver de; one globalMOO model at most), "
                          "steering (… induce), neighbours (python -m bonsai_interp.probe.neighbors), and attempts to break the label "
                          "(frequency tables, weak bands, leak search). List every screen used in methods.screens.")
    out += ["", RULES, "", "When done, tell the person in two or three sentences what you found and give them the PR link."]
    print("\n".join(out))
