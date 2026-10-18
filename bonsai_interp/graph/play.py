"""The contributor's command line: start a contribution, run its pre-registered test, write the results.

  python -m bonsai_interp.graph.play new interpret --neuron bonsai1/L12/N5397- --handle octocat
  python -m bonsai_interp.graph.play new qc --neuron bonsai1/L9/N3688+ --handle octocat --verdict confirm
      -> graph/contributions/<model>/<pole folder>/<id>.tests.json   (fill in the label and sentences, then COMMIT)
  python -m bonsai_interp.graph.play run graph/contributions/.../<id>.tests.json
      -> <id>.json with the results (needs the GPU; commit it in a SECOND commit)
  python -m bonsai_interp.graph.play step --neuron bonsai1/L12/N5397- [--k 4]
      -> the strongest feeders of a pole (single-unit zeroing on its top examples): candidates for "step out one neuron"
  python -m bonsai_interp.graph.play look --neuron bonsai1/L12/N5397-      top corpus examples
  python -m bonsai_interp.graph.fold                                         refresh the derived records
  python -m bonsai_interp.graph.play doctor                                  is this machine ready? (agents: run first)
  python -m bonsai_interp.graph.play brief interpret --neuron <id> --handle <you>   full mission instructions for an agent

The "run" step follows the protocol named in the tests file (quick-8+8@1 today) and records the
toolchain commit, so the result can be reproduced.
"""
import argparse
import datetime as dt
import json
import secrets
import subprocess
import sys
from pathlib import Path

from bonsai_interp.graph.contrib import NEURON, ROOT, folder_for

TEMPLATE_FIRE = ["<sentence that should fire the pole>"] * 8
TEMPLATE_QUIET = ["<near miss: same topic, without the claimed trigger>"] * 8


def _commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def new(a):
    m = NEURON.match(a.neuron)
    if not m:
        sys.exit("neuron must look like bonsai1/L12/N5397+ (model/layer/index and pole)")
    cid = f"{dt.date.today().isoformat()}-{a.handle}-{secrets.token_hex(2)}"
    d = folder_for(a.neuron)
    d.mkdir(parents=True, exist_ok=True)
    unit = f"L{m['L']}/N{m['n']}"
    current = None
    rec = ROOT / "graph" / "neurons" / m["model"] / f"{d.name}.json"
    if rec.exists():
        current = json.loads(rec.read_text(encoding="utf-8")).get("label")
    t = {"contribution": cid, "contributor": a.handle, "kind": a.kind, "neuron": a.neuron, "protocol": "quick-8+8@1",
         "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
         **({"qc": {"verdict": a.verdict, "label_under_test": current}} if a.kind == "qc" else {}),
         "neighbours": [{"unit": unit, "pole": m["pole"], "label": current if a.kind == "qc" and a.verdict != "refine" else "<your label>",
                         "fire": TEMPLATE_FIRE, "quiet": TEMPLATE_QUIET}],
         "predictions": "<optional: what you expect, e.g. '6/8 fire, 0 near misses'>"}
    p = d / f"{cid}.tests.json"
    p.write_text(json.dumps(t, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {p.relative_to(ROOT)}\nFill in the label and the 16 sentences, then commit THIS FILE ALONE:\n"
          f"  git add {p.relative_to(ROOT).as_posix()} && git commit -m \"Pre-register {a.kind} test for {a.neuron}\"\n"
          f"then: python -m bonsai_interp.graph.play run {p.relative_to(ROOT).as_posix()}")


def run(a):
    import torch
    from bonsai_interp import explore
    from bonsai_interp.model import load
    from bonsai_interp.probe import textinv
    p = Path(a.tests_file).resolve()
    rel = p.relative_to(ROOT).as_posix()
    if subprocess.run(["git", "log", "--format=%h", "-1", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip() == "":
        sys.exit(f"{rel} is not committed yet. Commit the pre-registration first, then run it.")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode != 0:
        sys.exit(f"{rel} has uncommitted edits. A pre-registration must be run exactly as committed.")
    t = json.loads(p.read_text(encoding="utf-8"))
    if any("<" in s for nb in t["neighbours"] for s in nb["fire"] + nb["quiet"] + [nb["label"]]):
        sys.exit("the tests file still has <placeholders>")
    prereg = subprocess.run(["git", "log", "--diff-filter=A", "--format=%h", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout.split()[-1]
    m = NEURON.match(t["neuron"])
    torch.cuda.set_per_process_memory_fraction(0.8)
    model, tok, rec = load(m["model"])
    tok.padding_side = "right"
    scan = explore.scan(m["model"])
    tests = []
    for nb in t["neighbours"]:
        L, n = map(int, nb["unit"][1:].split("/N"))
        pole = 0 if nb["pole"] == "+" else 1
        pmax, omax = float(scan["top_val"][L, pole, 0, n]), float(scan["top_val"][L, 1 - pole, 0, n])
        ev = textinv.TextEvaluator(model, tok, rec, L, n, pmax, omax, 1.0 if pole == 0 else -1.0)
        fire, quiet = ev(nb["fire"]), ev(nb["quiet"])
        per = [{"set": s, "i": i, "act": round(r["act"], 3), "peak": r["peak_token"]} for s, rs in (("fire", fire), ("quiet", quiet)) for i, r in enumerate(rs)]
        nf, fp = sum(r["act"] >= 0.3 for r in fire), sum(r["act"] >= 0.3 for r in quiet)
        tests.append({"protocol": t["protocol"], "tests_file": rel, "prereg_commit": prereg, "author": a.author or t["contributor"],
                      "n_fire": nf, "n_false_pos": fp, "n_written": [len(fire), len(quiet)], "per_sentence": per})
        print(f"{nb['unit']}{nb['pole']}: fire {nf}/{len(fire)}, near misses firing {fp}/{len(quiet)} -> {'HELD' if nf >= 5 and fp <= 1 else 'FAILED'}")
    c = {"id": t["contribution"], "kind": t["kind"], "neuron": t["neuron"], "contributor": t["contributor"],
         "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
         "recipe": {"toolchain_commit": _commit(), "command": f"python -m bonsai_interp.graph.play run {rel}"},
         "label": t["neighbours"][0]["label"], "tests": tests}
    if t["kind"] == "qc":
        c["qc"] = {"verdict": t["qc"]["verdict"], "label_under_test": t["qc"].get("label_under_test")}
        if t["qc"]["verdict"] != "refine":
            c.pop("label")
    for k in ("short", "unit_kind", "confidence", "notes", "roles", "via"):
        if t.get(k) is not None:
            c[k] = t[k]
    out = p.with_name(p.name.replace(".tests.json", ".json"))
    out.write_text(json.dumps(c, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out.relative_to(ROOT)}. Next: python -m bonsai_interp.graph.fold, then commit the results in a second commit and open a PR.")


def step(a):
    import numpy as np
    import torch
    from bonsai_interp.model import load
    from bonsai_interp.probe.quick import one_back
    m = NEURON.match(a.neuron)
    torch.cuda.set_per_process_memory_fraction(0.8)
    model, tok, rec = load(m["model"])
    feeders = one_back(model, rec, int(m["L"]), int(m["n"]), 0 if m["pole"] == "+" else 1, k_show=a.k)
    for f in feeders:
        print(f"{f['unit']:<11} drop {f['drop']:.2f}  + peaks {f['peaks_pos']}  - peaks {f['peaks_neg']}")
    print("Pick one, then: python -m bonsai_interp.graph.play new interpret --neuron bonsai1/<unit><pole> ... "
          "and add 'roles': ['feeds <this neuron>'] plus an edge in your results notes.")


def look(a):
    m = NEURON.match(a.neuron)
    subprocess.run([sys.executable, "-m", "bonsai_interp.explore", "--model", m["model"], "--neuron", f"{m['L']}/{m['n']}",
                    "--pole", m["pole"], "--k", "16"], cwd=ROOT)



# ---------------------------------------------------------------- agent support: doctor + brief

def _ok(flag, name, detail="", fix=""):
    mark = {True: "PASS", False: "FAIL", None: "SKIP"}[flag]
    print(f"[{mark}] {name}" + (f": {detail}" if detail else "") + (f"\n       fix: {fix}" if flag is False and fix else ""))
    return flag is not False


def doctor(a):
    """Check that this machine can play. Exit code 0 when every required check passes."""
    import platform
    import shutil
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
    m = ROOT / "models" / "Bonsai-8B-unpacked" / "config.json"
    req.append(_ok(m.exists(), "Bonsai-8B weights", str(m.parent.relative_to(ROOT)), "bash scripts/download_models.sh  (Bonsai-8B-unpacked is enough)"))
    for f, size in (("corpus.npz", 100e6), ("scan_bonsai1.npz", 1e9)):
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

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new")
    n.add_argument("kind", choices=["interpret", "qc"])
    n.add_argument("--neuron", required=True)
    n.add_argument("--handle", required=True, help="your GitHub handle")
    n.add_argument("--verdict", choices=["confirm", "dispute", "refine"], default="confirm",
                   help="qc only: confirm = re-test the label as is; refine = propose a better label")
    r = sub.add_parser("run")
    r.add_argument("tests_file")
    r.add_argument("--author", help="who wrote the sentences, if not you (e.g. 'bonsai' when the model wrote them)")
    s = sub.add_parser("step")
    s.add_argument("--neuron", required=True)
    s.add_argument("--k", type=int, default=4)
    lk = sub.add_parser("look")
    lk.add_argument("--neuron", required=True)
    d = sub.add_parser("doctor", help="check this machine is ready to play")
    d.add_argument("--quick", action="store_true", help="skip loading the model")
    b = sub.add_parser("brief", help="print the full instructions for a mission (for an AI agent)")
    b.add_argument("kind", choices=["interpret", "qc", "step", "blind"])
    b.add_argument("--neuron", required=True)
    b.add_argument("--handle", default="<your-github-handle>")
    b.add_argument("--gmoo", action="store_true", help="include the optional globalMOO step")
    a = ap.parse_args()
    {"new": new, "run": run, "step": step, "look": look, "doctor": doctor, "brief": brief}[a.cmd](a)


if __name__ == "__main__":
    main()
