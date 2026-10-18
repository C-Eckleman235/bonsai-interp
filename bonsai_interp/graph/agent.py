"""Agent support for the survey: `play doctor` (is this machine ready?) and `play brief` (full mission
instructions). Kept apart from play.py so a clean room can ship the contributor commands without
this file, which mentions known neurons.
"""
import datetime as dt
import json
import subprocess
import sys

from bonsai_interp.graph.contrib import NEURON, ROOT, folder_for

def _ok(flag, name, detail="", fix=""):
    mark = {True: "PASS", False: "FAIL", None: "SKIP"}[flag]
    print(f"[{mark}] {name}" + (f": {detail}" if detail else "") + (f"\n       fix: {fix}" if flag is False and fix else ""))
    return flag is not False


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
        import torch
        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else "none"
        mem = torch.cuda.get_device_properties(0).total_memory / 2**30 if cuda else 0
        req.append(_ok(cuda, "CUDA GPU", f"{gpu}, {mem:.0f} GB", "install an NVIDIA driver and the CUDA 12.8 torch wheel (uv sync does this)"))
        if cuda:
            req.append(_ok(mem >= 15, "GPU memory >= 15 GB", f"{mem:.1f} GB", "use a 16 GB+ GPU"))
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
    _ok(True if shutil.which("claude") else None, "claude CLI (optional, blind missions)", shutil.which("claude") or "not found")
    cr = ROOT.parent / "neuron-cleanroom"
    _ok(True if cr.exists() else None, "neuron-cleanroom checkout (optional, blind missions)", str(cr) if cr.exists() else "not cloned next to this repo")
    if all(req) and not a.quick:
        # end to end: load the model and read one known neuron on one sentence
        import torch
        from bonsai_interp import explore
        from bonsai_interp.model import load
        from bonsai_interp.probe import textinv
        torch.cuda.set_per_process_memory_fraction(0.8)
        model, tok, rec = load("bonsai1")
        tok.padding_side = "right"
        s = explore.scan("bonsai1")
        ev = textinv.TextEvaluator(model, tok, rec, 9, 3688, float(s["top_val"][9, 0, 0, 3688]), float(s["top_val"][9, 1, 0, 3688]), 1.0)
        act = ev(["Waiting for the verdict was the most nerve-racking hour of my life."])[0]["act"]
        req.append(_ok(act >= 0.3, "end-to-end reading (L9/N3688+ on 'nerve-racking')", f"{act:.2f} of max (expected >= 0.3)",
                       "the weights or data do not match this toolchain; re-download them"))
    print("\nREADY: start a mission with `python -m bonsai_interp.graph.play brief …`" if all(req) else "\nNOT READY: fix the FAIL lines above and run doctor again.")
    sys.exit(0 if all(req) else 1)


RULES = """Rules (from AGENTS.md; CI enforces 1-3):
1. Commit the .tests.json ON ITS OWN before running anything that scores it; results go in a SECOND commit. Never amend, squash or edit a committed .tests.json.
2. Never edit or delete files under graph/contributions/ (yours included). Never hand-edit graph/neurons, edges.json or index.json: run fold.
3. Only the two contribution files (<= 64 KB each) go in git; bulky material goes to the library (DATA_POLICY.md).
4. Report honestly: a failed label is a valid contribution. Do not rewrite a test after seeing its result."""


def brief(a):
    m = NEURON.match(a.neuron)
    if not m:
        sys.exit("neuron must look like bonsai1/L12/N5397+")
    L, n, pole, nid = int(m["L"]), int(m["n"]), m["pole"], a.neuron
    u, folder = f"{L}/{n}", folder_for(nid).relative_to(ROOT).as_posix()
    idx = json.loads((ROOT / "graph" / "index.json").read_text(encoding="utf-8"))["neurons"]
    rec = idx.get(nid)
    py = "uv run python"
    def cmds(t, tu, tp):
        claim = f'gh issue create --repo moridinamael/bonsai-interp --title "claim: {t}" --label claim --body "{a.kind} mission via the survey"'
        check = f'gh issue list --repo moridinamael/bonsai-interp --label claim --state open --search "{t} in:title"'
        gm = (f"\n   Optional (one globalMOO model for this neuron, key in .env): {py} -m bonsai_interp.probe.run forward {tu} {tp}, write the bank it asks for "
              f"(probes/…/bank.json), then {py} -m bonsai_interp.probe.run invert {tu} {tp} --solver gmoo. Use what it finds to sharpen the label before step 4."
              if a.gmoo else "")
        return claim, check, gm
    claim, check_claim, gm = cmds(nid, u, pole)
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
            claim, check_claim, gm = cmds("TARGET", "<TARGET layer/index>", "<TARGET pole>")
            u, folder = "<TARGET layer/index>", "graph/contributions/bonsai1/<TARGET folder>"
        else:
            if rec:
                out += [f"Note: {nid} already has a record (status {rec['status']}). Consider a QC mission instead.", ""]
            target = nid
        out += [f"1. Check nobody has claimed it: {check_claim}   (if an open claim exists, stop and ask for another neuron)",
                f"   Claim it: {claim}",
                f"2. Look: {py} -m bonsai_interp.graph.play look --neuron {target}",
                f"   Also look past the top examples: weak and middle bands and the most frequent strong tokens "
                f"({py} -m bonsai_interp.probe.bands scan --units {u} --windows 2000 --tag mine, then bands.sample).{gm}",
                f"3. Write the pre-registration: {py} -m bonsai_interp.graph.play new interpret --neuron {target} --handle {a.handle}",
                "   Fill in: your label (at the level the test can support), 8 sentences that should fire the pole, 8 near misses",
                "   that keep topic, length and style and drop only the claimed trigger. Optional: short, unit_kind, confidence, notes.",
                f'   Commit it ALONE: git add {folder} && git commit -m "Pre-register interpret test for {target}"',
                f"4. Run it: {py} -m bonsai_interp.graph.play run <the .tests.json>  then  {py} -m bonsai_interp.graph.fold",
                f'   Commit the results SEPARATELY: git add graph && git commit -m "Results for {target}"',
                f"5. Validate: {py} -m bonsai_interp.graph.validate",
                f'6. Push a branch and open the PR: gh pr create --repo moridinamael/bonsai-interp --template interpret.md --title "interpret: {target} <short label>"',
                "7. If the label failed and you have a better idea, a second round is a NEW contribution (play new again) with new sentences.",
                "   After three honest failed rounds, record it and move on; that is a result too."]
    elif a.kind == "qc":
        if not rec:
            sys.exit(f"{nid} has no record to QC; use an interpret mission")
        out += [f"Current label: {rec.get('label')}", f"Status: {rec['status']}; QC so far: {rec['qc']}", "",
                f"1. Optional claim: {claim}",
                f"2. Look at the neuron yourself, especially beyond the top examples: {py} -m bonsai_interp.graph.play look --neuron {nid}{gm}",
                "   Read its earlier tests in graph/contributions/ only AFTER you have drafted your own sentences, so yours stay independent.",
                f"3. Choose: confirm (re-test the label as it is), dispute (a test designed to break it), or refine (a better label with its own test).",
                f"   {py} -m bonsai_interp.graph.play new qc --neuron {nid} --handle {a.handle} --verdict <confirm|dispute|refine>",
                "   Write NEW sentences (8 fire, 8 near misses); do not reuse the original test.",
                f'   Commit it ALONE: git add {folder} && git commit -m "Pre-register QC of {nid}"',
                f"4. Run: {py} -m bonsai_interp.graph.play run <the .tests.json>  then  {py} -m bonsai_interp.graph.fold",
                f'   Commit SEPARATELY: git add graph && git commit -m "QC results for {nid}"',
                f"5. Validate ({py} -m bonsai_interp.graph.validate), push a branch, open: gh pr create --repo moridinamael/bonsai-interp --template qc.md --title \"qc: {nid} <verdict>\""]
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
    out += ["", RULES, "", "When done, tell the person in two or three sentences what you found and give them the PR link."]
    print("\n".join(out))
