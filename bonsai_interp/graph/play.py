"""The contributor's command line: start a contribution, run its pre-registered test, write the results.

  python -m bonsai_interp.graph.play new interpret --neuron bonsai1/L12/N5397- --handle octocat
  python -m bonsai_interp.graph.play new qc --neuron bonsai1/L12/N345+ --handle octocat --verdict confirm
      -> graph/contributions/<model>/<pole folder>/<id>.tests.json   (fill in the label and sentences, then COMMIT)
  python -m bonsai_interp.graph.play draw --k 16 [--seed 7] [--layers 0-35]
      -> k random poles with no record yet, for a sweep (no claims: a duplicate is an independent replication)
  python -m bonsai_interp.graph.play new interpret --neuron <id> <id> ... --handle octocat   one tests file per pole
  python -m bonsai_interp.graph.play run graph/contributions/.../<id>.tests.json
      -> <id>.json with the results (needs the GPU; commit it in a SECOND commit)
  python -m bonsai_interp.graph.play step --neuron bonsai1/L12/N5397- [--k 4]
      -> the strongest feeders of a pole (single-unit zeroing on its top examples): candidates for "step out one neuron"
  python -m bonsai_interp.graph.play look --neuron bonsai1/L12/N5397-      top corpus examples
  python -m bonsai_interp.graph.play probe --neuron bonsai1/L12/N5397- "a sentence" "another"   exploration, nothing recorded
  python -m bonsai_interp.graph.play card --neuron bonsai1/L9/N3688+        card for ANY neuron: examples, density, logit effect, flags
  python -m bonsai_interp.graph.play which "some text"                     reverse search: which poles fire on each token
  python -m bonsai_interp.graph.play which --token " nerve"                  poles whose top corpus peaks are this token
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
from bonsai_interp.graph.methods import SCREENS, detect, minutes_between
from bonsai_interp.graph.protocols import outcome

PROBE_LOG = ROOT / ".probe-log"  # gitignored: what this checkout has already scored, so tests can say what was known


def _probe_key(nid, text):
    """Neuron (both poles) + whitespace-normalised sentence -> short sha256. Scoring one pole shows the other too."""
    import hashlib
    m = NEURON.match(nid)
    return hashlib.sha256(f"{m['model']}/L{m['L']}/N{m['n']}\n{' '.join(text.split())}".encode("utf-8")).hexdigest()[:20]


def log_probed(nid, texts):
    """Record that these sentences were scored on this neuron (play probe; other scorers may call this too)."""
    import time
    m = NEURON.match(nid)
    PROBE_LOG.mkdir(exist_ok=True)
    now = time.time()
    with open(PROBE_LOG / f"{m['model']}_L{m['L']}_N{m['n']}.jsonl", "a", encoding="utf-8", newline="\n") as f:
        for t in texts:
            f.write(json.dumps({"sha": _probe_key(nid, t), "t": round(now, 1)}) + "\n")


def probed_before(nid, texts, before):
    """How many of these sentences this checkout scored on this neuron before the unix time `before`."""
    m = NEURON.match(nid)
    f = PROBE_LOG / f"{m['model']}_L{m['L']}_N{m['n']}.jsonl"
    if not f.exists():
        return 0
    seen = {d["sha"] for d in (json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()) if d["t"] < before}
    return sum(_probe_key(nid, t) in seen for t in texts)


TEMPLATE_FIRE = ["<sentence that should fire the pole>"] * 8
TEMPLATE_QUIET = ["<near miss: same topic, without the claimed trigger>"] * 8


def _commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def new(a):
    bad = [x for x in a.neuron if not NEURON.match(x)]
    if bad:
        sys.exit(f"neuron must look like bonsai1/L12/N5397+ (model/layer/index and pole): {bad}")
    paths = [_new_one(a, nid) for nid in a.neuron]
    rel = [p.relative_to(ROOT).as_posix() for p in paths]
    what = a.neuron[0] if len(paths) == 1 else f"{len(paths)} poles" + (" (sweep)" if getattr(a, "k", None) else "")
    if getattr(a, "k", None):  # from play draw: the list file is the handle; skip the long commands
        return rel
    print(f"wrote {len(paths)} tests file(s):\n  " + "\n  ".join(rel) +
          "\nFill in each label and its 16 sentences, then commit the tests files ALONE (one commit for all of them is fine):\n"
          f"  git add {' '.join(rel)} && git commit -m \"Pre-register {a.kind} tests for {what}\"\n"
          f"then: uv run python -m bonsai_interp.graph.play run {' '.join(rel)}")
    return rel


def withhold(a):
    """Record that a pole's trigger is sensitive content: a masked description instead of a live test. One file, one
    commit (there is no test to pre-register). The pole shows as 'withheld' on the map and edges may point at it."""
    from bonsai_interp.graph.contrib import WITHHELD_REASONS, check
    from bonsai_interp.sensitive import mask
    if not NEURON.match(a.neuron):
        sys.exit("neuron must look like bonsai1/L1/N362+ (model/layer/index and pole)")
    if a.reason not in WITHHELD_REASONS:
        sys.exit(f"--reason must be one of {sorted(WITHHELD_REASONS)}")
    if mask(a.label) != a.label:
        sys.exit("the label names the sensitive word itself: describe it masked, e.g. 'a slur token (masked), any context'")
    d = folder_for(a.neuron)
    d.mkdir(parents=True, exist_ok=True)
    taken = {q.name.split(".")[0] for q in d.glob("*.json")}
    while (cid := f"{dt.date.today().isoformat()}-{a.handle}-{secrets.token_hex(4)}") in taken:
        pass
    c = {"id": cid, "kind": "withhold", "neuron": a.neuron, "contributor": a.handle,
         "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "label": a.label,
         **({"short": a.short} if a.short else {}), "withheld": {"reason": a.reason},
         **({"notes": a.notes} if a.notes else {}), "methods": {"agent": a.agent, "screens": a.screens or ["card"]},
         **({"artifacts": [{"name": "unmasked evidence", "url": a.evidence_url, "sha256": a.evidence_sha256}]}
            if a.evidence_url and a.evidence_sha256 else {})}
    p = d / f"{cid}.json"
    p.write_text(json.dumps(c, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    errs = check(p, c)
    if errs:
        p.unlink()
        sys.exit("\n".join(errs))
    rel = p.relative_to(ROOT).as_posix()
    print(f"wrote {rel}\ncommit it (one commit is enough, there is no test to pre-register):\n"
          f"  git add {rel} && git commit -m \"Withhold {a.neuron}: sensitive trigger ({a.reason})\"")
    return rel


def _new_one(a, nid):
    m = NEURON.match(nid)
    d = folder_for(nid)
    taken = {q.name.split(".")[0] for q in d.glob("*.json")} if d.exists() else set()
    while True:  # 8 hex digits (4 collided across poles within a day of sweeps); never reuse one in this folder
        cid = f"{dt.date.today().isoformat()}-{a.handle}-{secrets.token_hex(4)}"
        if cid not in taken:
            break
    d.mkdir(parents=True, exist_ok=True)
    unit = f"L{m['L']}/N{m['n']}"
    current = None
    rec = ROOT / "graph" / "neurons" / m["model"] / f"{d.name}.json"
    if rec.exists():
        current = json.loads(rec.read_text(encoding="utf-8")).get("label")
    t = {"contribution": cid, "contributor": a.handle, "kind": a.kind, "neuron": nid, "protocol": "quick-8+8@3",
         "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
         **({"selection": a.selection} if getattr(a, "selection", None) else {}),
         **({"qc": {"verdict": a.verdict, "label_under_test": current}} if a.kind == "qc" else {}),
         "neighbours": [{"unit": unit, "pole": m["pole"], "label": current if a.kind == "qc" and a.verdict != "refine" else "<your label>",
                         "fire": TEMPLATE_FIRE, "quiet": TEMPLATE_QUIET}],
         "predictions": "<optional: what you expect, e.g. '6/8 fire, 0 near misses'>",
         "methods": {"agent": "<who did the work: e.g. 'claude-code opus-5.5', 'codex gpt-6.1', 'human'>",
                     "screens": ["look"], "_screens_vocabulary": sorted(SCREENS)}}
    p = d / f"{cid}.tests.json"
    p.write_text(json.dumps(t, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return p


def draw(a):
    """k random poles with no record in the graph. No claims: if two people draw the same pole, both results
    count, and the second is an independent replication."""
    import random
    known = set(json.loads((ROOT / "graph" / "index.json").read_text(encoding="utf-8"))["neurons"])
    lo, hi = map(int, a.layers.split("-"))
    rng, out = random.Random(a.seed), []
    while len(out) < a.k:
        nid = f"bonsai1/L{rng.randint(lo, hi)}/N{rng.randrange(12288)}{rng.choice('+-')}"
        if nid not in known and nid not in out:
            out.append(nid)
    print(" ".join(out))
    if a.handle:  # write the tests files for exactly these poles, so the ids cannot drift between draw and new
        a.neuron, a.kind, a.verdict, a.selection = out, "interpret", "confirm", "random draw"
        paths = new(a)
        lst = ROOT / ".sweep" / "latest.txt"  # gitignored; the list survives long output and later steps
        lst.parent.mkdir(exist_ok=True)
        for old in ("plan.json", "probe.json", "table.md"):  # a fresh sweep never inherits the last one's plans
            (lst.parent / old).unlink(missing_ok=True)
        lst.write_text("\n".join(paths) + "\n", encoding="utf-8", newline="\n")
        print(f"the {len(paths)} tests files are listed in .sweep/latest.txt (git add --pathspec-from-file=.sweep/latest.txt; play run --list .sweep/latest.txt)")


OPTIONAL = {"short": "a 2-5 word name for the map", "unit_kind": "lexical | contextual | sparse feature | dense / positional | unclear",
            "confidence": "a number 0-1 or low / medium / high", "notes": "free text (e.g. 'some sentences suggested by Twitch chat')",
            "predictions": "what you expect, written before the run",
            "selection": "how the pole was chosen: random draw | step-out from <id> | step-in from <id> | concept search: <concept> | picked",
            "trigger": "for a label whose trigger is a set of tokens: {\"words\": [...]} (any case, plus pieces of their split), "
                       "{\"pieces\": [...]} (the piece as its own token) or {\"tokens\": [...]} (exact tokens); its coverage of the pole's "
                       "strong activations is measured, and a held label covering under half of them is 'partial'"}


def fill(a):
    """Fill tests files from one JSON plan: {"<pole id>": {"label": ..., "fire": [8], "quiet": [8], optional keys}, ...}.
    Writes LF line endings, drops the template's reference list, and refuses anything it cannot place."""
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    files = [ln.strip() for ln in Path(a.list).read_text(encoding="utf-8").splitlines() if ln.strip()]
    by_pole = {}
    for f in files:
        t = json.loads((ROOT / f).read_text(encoding="utf-8"))
        by_pole[t["neuron"]] = (f, t)
    unknown = [k for k in plan if k not in by_pole]
    if unknown:
        sys.exit(f"plan has poles that are not in {a.list}: {unknown}")
    done = []
    for nid, e in plan.items():
        f, t = by_pole[nid]
        bad = [k for k in e if k not in ("label", "fire", "quiet", "agent", "screens", *OPTIONAL)]
        if bad or len(e.get("fire", [])) != 8 or len(e.get("quiet", [])) != 8 or not e.get("label"):
            sys.exit(f"{nid}: needs label, 8 fire and 8 quiet sentences; optional keys {sorted(OPTIONAL)}; unknown keys {bad}")
        nb = t["neighbours"][0]
        nb.update(label=e["label"], fire=e["fire"], quiet=e["quiet"])
        for k in OPTIONAL:
            if k in e:
                t[k] = e[k]
        if t.get("predictions", "").startswith("<optional"):
            t.pop("predictions")
        m = t.setdefault("methods", {})
        m.pop("_screens_vocabulary", None)
        if e.get("agent") or a.agent:
            m["agent"] = e.get("agent") or a.agent
        m["screens"] = e.get("screens") or a.screens.split(",")
        (ROOT / f).write_text(json.dumps(t, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        done.append(f)
    left = [f for p, (f, _) in by_pole.items() if p not in plan]
    print(f"filled {len(done)} tests files" + (f"; not in the plan yet: {left}" if left else "; every listed file is filled"))


def prbody(a):
    """Write the PR body for the current branch to .pr/<branch>.md (gitignored) from the right template, with the
    table of the last `play run` filled in for a sweep. Shell-neutral: no mkdir/cp/&& for the agent to translate."""
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    kind = a.kind or ("sweep" if branch.startswith("sweep/") else "qc" if branch.startswith("qc/") else "interpret")
    body = (ROOT / ".github" / "PULL_REQUEST_TEMPLATE" / f"{kind}.md").read_text(encoding="utf-8")
    table = ROOT / ".sweep" / "table.md"
    if kind == "sweep" and table.exists():
        held_total, *rows = table.read_text(encoding="utf-8").splitlines()
        held, total = held_total.split("/")
        body = body.replace("## Sweep: <!-- N --> poles, <!-- held --> held", f"## Sweep: {total} poles, {held} held")
        body = body.replace("| <!-- bonsai1/L__/N____± --> | <!-- short label --> | <!-- lexical / contextual / … --> | | | | <!-- held / failed --> |",
                            "\n".join(rows))
    # what the branch itself can tell: the pre-registration and results commits, the agent, what was explored
    log = subprocess.run(["git", "log", "--reverse", "--format=%h %s", "origin/main..HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    pre = next((ln.split()[0] for ln in log if "pre-register" in ln.lower()), None)
    res = next((ln.split()[0] for ln in log if "result" in ln.lower()), None)
    files = [ln.strip() for ln in (ROOT / ".sweep" / "latest.txt").read_text(encoding="utf-8").splitlines() if ln.strip()] \
        if kind == "sweep" and (ROOT / ".sweep" / "latest.txt").exists() else []
    meths = [json.loads((ROOT / f).read_text(encoding="utf-8")).get("methods", {}) for f in files if (ROOT / f).exists()]
    agents = sorted({m.get("agent") for m in meths if m.get("agent")})
    screens = sorted({x for m in meths for x in m.get("screens", [])})
    if agents:
        body = body.replace("- **Agent:** <!-- e.g. claude-code opus-5.5, codex, human -->", f"- **Agent:** {', '.join(agents)}")
    if pre:
        body = body.replace("- **Pre-registration commit:** <!-- hash: every tests file, committed together and alone -->", f"- **Pre-registration commit:** {pre}")
    if res:
        body = body.replace("- **Results commit:** <!-- hash, a LATER commit -->", f"- **Results commit:** {res}")
    if screens:
        body = body.replace("- **Explored before pre-registering:** <!-- e.g. cards only / probe --plan on 10 poles -->",
                            f"- **Screens used:** {', '.join(screens)}")
    out = ROOT / ".pr" / f"{branch.replace('/', '-') or 'pr'}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(body, encoding="utf-8", newline="\n")
    rel = out.relative_to(ROOT).as_posix()
    print(f"wrote {rel}: fill in the remaining <!-- --> fields, then\n"
          f"  git push -u origin {branch}\n  gh pr create --repo moridinamael/bonsai-interp --title \"<title>\" --body-file {rel}")


def _check_tests_file(path):
    """(problem or None, tests dict, rel path, prereg commit) for one pre-registration."""
    p = Path(path).resolve()
    rel = p.relative_to(ROOT).as_posix()
    t = json.loads(p.read_text(encoding="utf-8"))
    templates = set(TEMPLATE_FIRE + TEMPLATE_QUIET + ["<your label>"])
    left = [s for nb in t["neighbours"] for s in nb["fire"] + nb["quiet"] + [nb["label"]] if s in templates]
    if left:
        return f"{rel}: still has {len(left)} template placeholder(s), e.g. {left[0]!r}", None, rel, None
    agent = (t.get("methods") or {}).get("agent")
    if not agent or str(agent).startswith("<who did the work"):
        return f"{rel}: fill in methods.agent (who did the work), commit, then run", None, rel, None
    if subprocess.run(["git", "log", "--format=%h", "-1", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip() == "":
        return f"{rel}: content OK; not committed yet: commit the pre-registration first", None, rel, None
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode != 0:
        return f"{rel}: has uncommitted edits; a pre-registration must be run exactly as committed", None, rel, None
    prereg = subprocess.run(["git", "log", "--diff-filter=A", "--format=%h", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout.split()[-1]
    return None, t, rel, prereg


def run(a):
    """Score one or more committed pre-registrations with ONE model load (batching is where the time goes:
    loading takes 30-60 s, a 16-sentence test about a second)."""
    files = list(a.tests_files) + ([ln.strip() for ln in Path(a.list).read_text(encoding="utf-8").splitlines() if ln.strip()] if a.list else [])
    if not files:
        sys.exit("give tests files, or --list <file with one per line>")
    checked = [(f, *_check_tests_file(f)) for f in files]
    problems = [c[1] for c in checked if c[1]]
    for msg in problems:
        print("SKIP", msg)
    todo = [c for c in checked if not c[1]]
    if not todo:
        sys.exit("nothing to run")
    import torch
    from bonsai_interp import explore
    from bonsai_interp.env import DTYPE, limit_memory
    from bonsai_interp.model import load
    from bonsai_interp.probe import textinv
    limit_memory(0.8)
    models, scans, rows = {}, {}, []
    for i, (f, _, t, rel, prereg) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] scoring {t['neuron']} ...", flush=True)
        m = NEURON.match(t["neuron"])
        if m["model"] not in models:
            models[m["model"]] = load(m["model"])
            models[m["model"]][1].padding_side = "right"
            scans[m["model"]] = explore.scan(m["model"])
        model, tok, rec = models[m["model"]]
        scan = scans[m["model"]]
        tests = []
        for nb in t["neighbours"]:
            L, n = map(int, nb["unit"][1:].split("/N"))
            pole = 0 if nb["pole"] == "+" else 1
            pmax, omax = float(scan["top_val"][L, pole, 0, n]), float(scan["top_val"][L, 1 - pole, 0, n])
            ev = textinv.TextEvaluator(model, tok, rec, L, n, pmax, omax, 1.0 if pole == 0 else -1.0)
            fire, quiet = ev(nb["fire"]), ev(nb["quiet"])
            per = [{"set": s, "i": i, "act": round(r["act"], 3), "peak": r["peak_token"]} for s, rs in (("fire", fire), ("quiet", quiet)) for i, r in enumerate(rs)]
            nf, fp = sum(r["act"] >= 0.3 for r in fire), sum(r["act"] >= 0.3 for r in quiet)
            reg_t = int(subprocess.run(["git", "log", "-1", "--format=%ct", prereg], cwd=ROOT, capture_output=True, text=True).stdout.strip() or 0)
            pb = {"fire": probed_before(t["neuron"], nb["fire"], reg_t), "quiet": probed_before(t["neuron"], nb["quiet"], reg_t),
                  "source": "this checkout's .probe-log, before the pre-registration commit"}
            test = {"protocol": t["protocol"], "tests_file": rel, "prereg_commit": prereg, "author": a.author or t["contributor"],
                    "n_fire": nf, "n_false_pos": fp, "n_written": [len(fire), len(quiet)], "per_sentence": per, "probed_before": pb}
            if pb["fire"] + pb["quiet"]:
                print(f"NOTE {t['neuron']}: {pb['fire']} fire and {pb['quiet']} near-miss sentences were scored before this test was "
                      f"registered" + (" (more than 2: under quick-8+8@3 the test is exploratory)" if pb["fire"] + pb["quiet"] > 2 else ""))
            from bonsai_interp.graph.protocols import v2_stats
            auroc, med, _ = v2_stats(test)  # the recorded (rounded) values the rule judges, so print and verdict agree
            test["outcome"] = outcome(test)  # recorded for convenience; fold always recomputes it from per_sentence
            test["stats"] = {"auroc": round(auroc, 3), "median_fire": round(med, 3), "near_misses_at_0.3": fp}
            tests.append(test)
            rows.append(f"| {t['neuron']} | {(t.get('short') or nb['label'])[:60]} | {t.get('unit_kind', '')} | {auroc:.3f} | {med:.3f} | {fp} | {test['outcome']} |")
            fa = [r["act"] for r in fire]
            print(f"{t['neuron']:<22} AUROC {auroc:.3f} (need >= 0.90), median fire {med:.3f} (need >= 0.20), "
                  f"near misses >= 0.30: {fp} (max 1); fire >= 0.30: {nf}/{len(fa)} -> {outcome(test).upper()} under {t['protocol']}"
                  if t["protocol"] in ("quick-8+8@2", "quick-8+8@3") else
                  f"{t['neuron']:<22} {nb['unit']}{nb['pole']}: fire {nf}/{len(fire)}, near misses firing {fp}/{len(quiet)} "
                  f"-> {outcome(test).upper()} under {t['protocol']}")
        p = Path(f).resolve()
        declared = dict(t.get("methods") or {})
        declared.pop("_screens_vocabulary", None)
        tests_screens = ["quick-test"] + (["second-round"] if len(list(p.parent.glob("*.tests.json"))) > 1 else [])
        meth = {**declared, "screens": sorted(set(declared.get("screens", [])) | set(tests_screens)),
                "detected": detect(ROOT, int(m["L"]), int(m["n"]), m["pole"]), "toolchain_commit": _commit(),
                "minutes": minutes_between(ROOT, rel), "batch_size": len(todo)}
        if DTYPE != torch.bfloat16:  # float16 on GPUs without bfloat16 (docs/FLOAT16.md): say so in the record
            meth["dtype"] = str(DTYPE).removeprefix("torch.")
        unknown = [x for x in meth["screens"] if x not in SCREENS]
        if unknown:
            print(f"note: screens not in the shared vocabulary (add them in bonsai_interp/graph/methods.py): {unknown}")
        c = {"id": t["contribution"], "kind": t["kind"], "neuron": t["neuron"], "contributor": t["contributor"],
             "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
             "recipe": {"toolchain_commit": _commit(), "command": f"python -m bonsai_interp.graph.play run {rel}"},
             "label": t["neighbours"][0]["label"], "tests": tests, "methods": meth}
        if t["kind"] == "qc":
            c["qc"] = {"verdict": t["qc"]["verdict"], "label_under_test": t["qc"].get("label_under_test")}
            if t["qc"]["verdict"] != "refine":
                c.pop("label")
        for k in ("short", "unit_kind", "confidence", "notes", "roles", "via", "selection", "trigger"):
            if t.get(k) is not None:
                c[k] = t[k]
        out = p.with_name(p.name.replace(".tests.json", ".json"))
        out.write_text(json.dumps(c, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"   wrote {out.relative_to(ROOT).as_posix()}")
    held = sum(r.endswith("| held |") for r in rows)
    (ROOT / ".sweep").mkdir(exist_ok=True)
    (ROOT / ".sweep" / "table.md").write_text(f"{held}/{len(rows)}\n" + "\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print(f"\nSUMMARY: {held}/{len(rows)} held\n| pole | label | unit kind | AUROC | median fire | near misses >= 0.30 | outcome |\n"
          "|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n(paste this table into the PR body)")
    print(f"\n{len(todo)} scored, {len(problems)} skipped. Commit the results files in a SEPARATE commit from the pre-registrations "
          f"and open a PR with only graph/contributions/ files (the merge agent regenerates the derived graph files).")


def step(a):
    import numpy as np
    import torch
    from bonsai_interp.env import limit_memory
    from bonsai_interp.model import load
    from bonsai_interp.probe.quick import one_back
    m = NEURON.match(a.neuron)
    limit_memory(0.8)
    model, tok, rec = load(m["model"])
    if getattr(a, "down", False):
        from bonsai_interp.probe.quick import one_forward
        readers = one_forward(model, rec, int(m["L"]), int(m["n"]), 0 if m["pole"] == "+" else 1, k_show=a.k, k_cand=a.candidates)
        for r in readers:
            print(f"{r['unit']:<12} drop {r['drop']:.2f}  + peaks {r['peaks_pos']}  - peaks {r['peaks_neg']}  "
                  f"(on at {r['level']:.2f} of its max in {r['n_on']} of this pole's top examples)")
        if not readers:
            print("no readers found (last layer, or no candidate lost activation)")
        print(f"Pick one, then: python -m bonsai_interp.graph.play new interpret --neuron bonsai1/<unit> ... "
              f"and record: python -m bonsai_interp.graph.play edge --from {a.neuron} --to bonsai1/<unit> --kind feeds --weight <drop> "
              f"--method \"single-unit zero ablation, downstream (play step --down)\" --handle <you>")
        return
    feeders = one_back(model, rec, int(m["L"]), int(m["n"]), 0 if m["pole"] == "+" else 1, k_show=a.k)
    for f in feeders:
        print(f"{f['unit']:<11} drop {f['drop']:.2f}  + peaks {f['peaks_pos']}  - peaks {f['peaks_neg']}")
    print("Pick one, then: python -m bonsai_interp.graph.play new interpret --neuron bonsai1/<unit><pole> ... "
          "and add 'roles': ['feeds <this neuron>'] plus an edge in your results notes.")


def coverage_cmd(a):
    """A coverage round (see bonsai_interp/graph/coverage.py): --neuron draws the sample, --record writes the contribution."""
    import secrets
    from bonsai_interp.graph import coverage as cv
    from bonsai_interp.graph import fold as fd
    from bonsai_interp.graph.contrib import load_all
    if a.record:
        d = json.loads(Path(a.record).read_text(encoding="utf-8"))
        marks = [s.get("describes") for s in d["sample"]]
        if not all(isinstance(m, bool) for m in marks):
            sys.exit(f"{a.record}: set 'describes' to true or false on every sample (does the label describe this activation?)")
        nid = d["neuron"]
        folder = folder_for(nid)
        cid = f"{dt.date.today().isoformat()}-{a.handle}-{secrets.token_hex(4)}"
        c = {"id": cid, "kind": "coverage", "neuron": nid, "contributor": a.handle,
             "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "label": d["label"],
             "label_contribution": d["label_contribution"], "protocol": "coverage-16@1",
             "sample": [{k: s[k] for k in ("window", "position", "act", "text")} for s in d["sample"]], "marks": marks,
             "coverage": round(sum(marks) / len(marks), 4), "methods": {"agent": a.agent or "<who marked the sample>", "screens": ["coverage-round"]}}
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / f"{cid}.json"
        out.write_text(json.dumps(c, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        verdict = "assignment (>= 50%)" if c["coverage"] >= cv.ASSIGN else "partial (< 50%)"
        print(f"wrote {out.relative_to(ROOT).as_posix()}: the label describes {sum(marks)}/{len(marks)} = {c['coverage']:.0%} -> {verdict}. "
              f"Commit it and open a PR (a coverage round needs no pre-registration: the sample is drawn from the label's id).")
        return
    nid = a.neuron if a.neuron.startswith("bonsai1/") else f"bonsai1/{a.neuron}"
    cs = sorted([c for _, c in load_all() if c["neuron"] == nid], key=lambda c: (c["created"], c["id"]))
    r = fd.fold_neuron(nid, cs) if cs else None
    if not r or r["status"] not in ("held", "partial"):
        sys.exit(f"{nid} has no held label to measure (status {r['status'] if r else 'unlabelled'})")
    src = cv._label_source(cs)
    print(f"drawing {cv.ROUND_N} strong activations of {nid} (label from {src[1]}) from held-out windows ...", flush=True)
    sample = cv.draw(nid, src[1])
    if not sample:
        sys.exit("no strong activations found in the held-out windows searched")
    draft = ROOT / ".coverage" / f"{folder_for(nid).name}.json"
    draft.parent.mkdir(exist_ok=True)
    draft.write_text(json.dumps({"neuron": nid, "label": r["label"], "label_contribution": src[1],
                                 "instructions": "For each sample, set 'describes' to true if the label describes why the pole fires at the marked "
                                                 "token, false otherwise. Judge the label as written, not what you now think the pole does.",
                                 "sample": [{**s, "describes": None} for s in sample]}, ensure_ascii=False, indent=1) + "\n",
                     encoding="utf-8", newline="\n")
    print(f"label: {r['label']}")
    for i, s in enumerate(sample):
        print(f"  {i:2d} {s['act']:5.2f} | {s['text'][-140:]}")
    print(f"\nmark each sample in {draft.relative_to(ROOT).as_posix()} ('describes': true/false), then "
          f"`play coverage --record {draft.relative_to(ROOT).as_posix()} --handle <you> --agent <who marked>`")


def look(a):
    m = NEURON.match(a.neuron)
    subprocess.run([sys.executable, "-m", "bonsai_interp.explore", "--model", m["model"], "--neuron", f"{m['L']}/{m['n']}",
                    "--pole", m["pole"], "--k", "16"], cwd=ROOT)



def probe(a):
    """Exploration only: score sentences on a pole without recording anything. Use it freely before
    pre-registering; the committed test is what counts."""
    import torch
    from bonsai_interp import explore
    from bonsai_interp.env import limit_memory
    from bonsai_interp.model import load
    from bonsai_interp.probe import textinv
    if a.plan:  # {"bonsai1/L4/N11474-": ["sentence", ...], ...}: many poles, one model load
        plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    else:
        if not a.neuron:
            sys.exit("give --neuron and sentences, or --plan plan.json")
        texts = list(a.sentences) + ([ln.strip() for ln in Path(a.file).read_text(encoding="utf-8").splitlines() if ln.strip()] if a.file else [])
        if not texts:
            sys.exit("give sentences as arguments or --file (one per line)")
        plan = {a.neuron: texts}
    bad = [k for k in plan if not NEURON.match(k)]
    if bad:
        sys.exit(f"neuron must look like bonsai1/L12/N345+: {bad}")
    limit_memory(0.8)
    model, tok, rec = load(NEURON.match(next(iter(plan)))["model"])
    tok.padding_side = "right"
    s = explore.scan(NEURON.match(next(iter(plan)))["model"])
    for nid, texts in plan.items():
        m = NEURON.match(nid)
        L, n, pole = int(m["L"]), int(m["n"]), 0 if m["pole"] == "+" else 1
        ev = textinv.TextEvaluator(model, tok, rec, L, n, float(s["top_val"][L, pole, 0, n]), float(s["top_val"][L, 1 - pole, 0, n]),
                                   1.0 if pole == 0 else -1.0)
        if len(plan) > 1:
            print(f"=== {nid}")
        for t, r in zip(texts, ev(texts)):
            print(f"{r['act']:6.3f}  peak {r['peak_token']!r:<14} {t}")
        log_probed(nid, texts)
    print("(exploration: these sentences are now in .probe-log/ as already scored on this neuron. Use probing to find the label,"
          " then write the test's 16 sentences FRESH: a test that reuses more than 2 scored sentences counts as exploratory, not as a"
          " hold or a fail. Fractions of the pole's corpus max; a test holds when its 8 fire sentences beat its 8 near misses"
          " (AUROC >= 0.90), the median fire sentence reaches 0.20, and at most 1 near miss reaches 0.30)")


def edge(a):
    """Record measured edges as a 'map' contribution (no test needed; both ends need records of their own)."""
    for end in (a.src, a.dst):
        if not NEURON.match(end):
            sys.exit(f"{end}: use full ids like bonsai1/L8/N1234+")
    cid = f"{dt.date.today().isoformat()}-{a.handle}-{secrets.token_hex(2)}"
    c = {"id": cid, "kind": "map", "neuron": a.dst, "contributor": a.handle,
         "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
         "recipe": {"toolchain_commit": _commit(), "command": a.method},
         "edges": [{"from": a.src, "to": a.dst, "kind": a.kind, "weight": round(a.weight, 4), "method": a.method}]}
    d = folder_for(a.dst)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{cid}.json"
    p.write_text(json.dumps(c, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {p.relative_to(ROOT)}")

def main():
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to a legacy code page
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new")
    n.add_argument("kind", choices=["interpret", "qc"])
    n.add_argument("--neuron", required=True, nargs="+", help="one or more poles; one tests file each")
    n.add_argument("--handle", required=True, help="your GitHub handle")
    n.add_argument("--verdict", choices=["confirm", "dispute", "refine"], default="confirm",
                   help="qc only: confirm = re-test the label as is; refine = propose a better label")
    n.add_argument("--selection", help="how this pole was chosen, e.g. 'random draw', 'step-out from <id>', 'step-in from <id>', "
                                       "'concept search: <concept>', 'picked'; recorded so statistics can keep or exclude non-random picks")
    wh = sub.add_parser("withhold", help="a pole whose trigger is sensitive content (e.g. a slur): masked record, no live test")
    wh.add_argument("--neuron", required=True)
    wh.add_argument("--handle", required=True, help="your GitHub handle")
    wh.add_argument("--reason", required=True, help="slur, sexual, personal data, violence or other")
    wh.add_argument("--label", required=True, help="what kind of trigger, masked: e.g. 'a slur token (masked), any context'")
    wh.add_argument("--short", help="a short masked label for the map")
    wh.add_argument("--notes", help="masked notes, e.g. which screen showed it")
    wh.add_argument("--agent", default="human", help="who did the work, e.g. 'claude-code opus-5.5'")
    wh.add_argument("--screens", nargs="*", help="the analyses that showed it (bonsai_interp/graph/methods.py)")
    wh.add_argument("--evidence-url", help="unmasked evidence in the library (never in git)")
    wh.add_argument("--evidence-sha256")
    r = sub.add_parser("run", help="score one or more committed pre-registrations (one model load for all)")
    r.add_argument("tests_files", nargs="*")
    r.add_argument("--list", help="a file listing tests files, one per line (e.g. .sweep/latest.txt from play draw); works in any shell")
    r.add_argument("--author", help="only when the SENTENCES came from someone else, e.g. 'bonsai' when the model wrote them or 'twitch-chat'; "
                   "who did the work goes in the tests file's methods.agent, not here")
    fl = sub.add_parser("fill", help="fill the drawn tests files from one JSON plan (LF endings; checks the keys)")
    fl.add_argument("--plan", default=".sweep/plan.json", help='{"<pole id>": {"label": ..., "fire": [8], "quiet": [8], "short"/"unit_kind"/"confidence"/"notes": ...}}')
    fl.add_argument("--list", default=".sweep/latest.txt", help="the tests files (default: the last play draw)")
    fl.add_argument("--agent", help="methods.agent for every file, e.g. 'claude-code opus-5.5'")
    fl.add_argument("--screens", default="card", help="what you actually used, comma-separated (card, probe, look, bands, ...); a plan entry's own 'screens' wins")
    pb = sub.add_parser("prbody", help="write the PR body for this branch to .pr/<branch>.md (sweep: with the last run's table)")
    pb.add_argument("--kind", choices=["sweep", "interpret", "qc"], help="default: from the branch name")
    dr = sub.add_parser("draw", help="k random poles with no record yet (for a sweep)")
    dr.add_argument("--k", type=int, default=16)
    dr.add_argument("--seed", type=int, help="reproducible draw (default: random)")
    dr.add_argument("--layers", default="0-35")
    dr.add_argument("--handle", help="also write an interpret tests file for each drawn pole (play new), as a sweep does")
    s = sub.add_parser("step", help="the strongest feeders of a pole (upstream), or with --down its readers")
    s.add_argument("--neuron", required=True)
    s.add_argument("--k", type=int, default=4)
    s.add_argument("--down", action="store_true", help="readers instead of feeders: later poles that lose activation when this one is zeroed")
    s.add_argument("--candidates", type=int, default=24, help="--down: how many screened candidates to rank (default 24)")
    lk = sub.add_parser("look")
    lk.add_argument("--neuron", required=True)
    pr = sub.add_parser("probe", help="exploration: score sentences on a pole, nothing recorded")
    pr.add_argument("--neuron")
    pr.add_argument("sentences", nargs="*")
    pr.add_argument("--file", help="one sentence per line")
    pr.add_argument("--plan", help='JSON {"<pole id>": ["sentence", ...], ...}: probe many poles with one model load')
    e = sub.add_parser("edge", help="record a measured edge between two poles")
    e.add_argument("--from", dest="src", required=True)
    e.add_argument("--to", dest="dst", required=True)
    e.add_argument("--kind", default="feeds", choices=["feeds", "reads", "gates", "co-fires"])
    e.add_argument("--weight", type=float, required=True, help="feeds: share of the target's activation lost when the source is zeroed")
    e.add_argument("--method", default="single-unit zero ablation (play step)")
    e.add_argument("--handle", required=True)
    c = sub.add_parser("card", help="a card for any neuron (labelled or not): examples, density, regimes, logit effect")
    c.add_argument("--neuron", required=True, nargs="+", help="bonsai1/L9/N3688+ (the pole is shown first) or L9/N3688; "
                   "give several to print several cards in one go (much faster than one call each)")
    c.add_argument("--k", type=int, default=6, help="top examples per pole")
    c.add_argument("--out", help="also write the card: .json for data, anything else for the text")
    c.add_argument("--no-logits", action="store_true", help="skip the direct logit effect (no weights needed)")
    w = sub.add_parser("which", help="reverse search: text -> poles firing on each token, or --token -> poles")
    w.add_argument("text", nargs="?")
    w.add_argument("--token", help="a token as the model sees it, e.g. ' nerve' (leading space matters)")
    w.add_argument("--k", type=int, default=8, help="poles per token")
    w.add_argument("--min-max", type=float, default=0.5, help="ignore poles whose corpus max is below this (fractions of a tiny max are noise)")
    w.add_argument("--keep-dense", action="store_true", help="also list dense / positional poles (hidden by default)")
    w.add_argument("--rebuild", action="store_true", help="--token: rebuild the token index from the scan")
    wy = sub.add_parser("why", help="why did the model say that? the poles behind a prediction, with the map's labels")
    wy.add_argument("prompt")
    wy.add_argument("--target", help="the token to explain, e.g. ' Paris' (leading space matters); default: the model's own pick")
    wy.add_argument("--k", type=int, default=20, help="poles to list")
    wy.add_argument("--check", type=int, default=8, help="how many of the strongest to verify by zeroing")
    wy.add_argument("--out", help="also write the result as JSON")
    cv = sub.add_parser("coverage", help="a coverage round: draw 16 strong activations of a held label's pole, mark which ones the label describes")
    cv.add_argument("--neuron", help="draw the sample for this pole's current held label (GPU)")
    cv.add_argument("--record", help="the marked draft (.coverage/<pole>.json): write the coverage contribution")
    cv.add_argument("--handle", default="<your-github-handle>")
    cv.add_argument("--agent", help="who marked the sample, e.g. 'claude-code opus-5.5' or 'human'")
    se = sub.add_parser("search", help="search what the survey knows: tested labels, sweep-library and word-screen automatic tests")
    se.add_argument("text", nargs="?", help="words to find in a label, condition or context (case-insensitive)")
    se.add_argument("--kind", help="deterministic kind: copy, topic, lang, struct, number, predict (sweeplib@1), twin, grammar, pos, code, compose, cover")
    se.add_argument("--condition", help="sweep-library condition, e.g. script/arabic, num/year, piece/code")
    se.add_argument("--held", action="store_true", help="only held results")
    se.add_argument("--layers", help="e.g. 30-35 or 2,4,6")
    se.add_argument("--source", choices=["survey", "library", "screen"], help="only one source")
    se.add_argument("--limit", type=int, default=40)
    se.add_argument("--json", action="store_true")
    q = se.add_mutually_exclusive_group()
    q.add_argument("--ambiguous", action="store_true", help="mission queue: poles nothing explains yet (unsettled survey poles first, then loud "
                   "unexplained ones), ranked")
    q.add_argument("--deepen", action="store_true", help="mission queue: held labels that may be more than they say (top word covers under half "
                   "of the strong activations, or several library conditions held), ranked")
    d = sub.add_parser("doctor", help="check this machine is ready to play")
    d.add_argument("--quick", action="store_true", help="skip loading the model")
    b = sub.add_parser("brief", help="print the full instructions for a mission (for an AI agent)")
    b.add_argument("kind", choices=["interpret", "sweep", "qc", "step", "stepin", "blind"])
    b.add_argument("--neuron", help="the pole (every kind but sweep)")
    b.add_argument("--k", type=int, default=16, help="sweep: how many poles")
    b.add_argument("--handle", default="<your-github-handle>")
    b.add_argument("--gmoo", action="store_true", help="include the optional globalMOO step")
    b.add_argument("--depth", choices=["quick", "standard", "deep"],
                   help="quick (~5-10 min, the default for interpret and qc), standard (~30-45 min) or deep (hours); "
                        "step/stepin accept quick, which drops the band scan")
    a = ap.parse_args()
    if a.cmd in ("doctor", "brief"):
        from bonsai_interp.graph import agent
        return getattr(agent, a.cmd)(a)
    if a.cmd == "card":
        from bonsai_interp.graph import card
        return card.main(a)
    if a.cmd == "coverage":
        return coverage_cmd(a)
    if a.cmd == "search":
        from bonsai_interp.graph import search
        return search.main_queue(a) if (a.ambiguous or a.deepen) else search.main(a)
    if a.cmd == "why":
        from bonsai_interp.graph import why
        return why.main(a)
    if a.cmd == "which":
        from bonsai_interp.graph import which
        return which.main(a)
    {"new": new, "withhold": withhold, "draw": draw, "fill": fill, "prbody": prbody, "run": run, "step": step, "look": look, "edge": edge, "probe": probe}[a.cmd](a)


if __name__ == "__main__":
    main()
