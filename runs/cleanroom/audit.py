"""Leak audit of the blind clean-room agent's transcript (stream-json from `claude -p`).

Lists every tool call, flags any that reach outside the clean room, touch the main repo,
Claude memory or the web, and every call the permission rules denied.

  python runs/cleanroom/audit.py <transcript.jsonl>  ->  runs/cleanroom/audit.json
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

CLEAN = "neuron-cleanroom"
FLAGS = re.compile(r"monosemanticity|GitHub|\.claude[/\\]projects|AppData|FINDINGS|RESEARCH_PLAN|paper-contents|nerve_figure|"
                   r"web_spec|verdict\.json|dossier|\.\.[/\\]\.\.|cd \.\.|ls \.\.|C:[/\\]Users[/\\]USER[/\\](?!Documents[/\\]neuron-cleanroom)", re.I)


def main(path):
    calls, results, denied = {}, {}, []
    sub_agents = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = ev.get("message") or {}
        for c in msg.get("content", []) if isinstance(msg.get("content"), list) else []:
            if c.get("type") == "tool_use":
                calls[c["id"]] = {"tool": c["name"], "input": c["input"], "sub": ev.get("parent_tool_use_id") is not None}
                sub_agents += c["name"] in ("Agent", "Task")
            elif c.get("type") == "tool_result":
                txt = c.get("content")
                txt = txt if isinstance(txt, str) else json.dumps(txt, ensure_ascii=False)
                results[c["tool_use_id"]] = txt
                if "permission" in txt.lower() and ("denied" in txt.lower() or "blocked" in txt.lower()):
                    denied.append(c["tool_use_id"])
    flagged = []
    for i, c in calls.items():
        s = json.dumps(c["input"], ensure_ascii=False)
        if FLAGS.search(s) or c["tool"] in ("WebFetch", "WebSearch"):
            flagged.append({"id": i, "tool": c["tool"], "input": s[:400], "denied": i in denied,
                            "result_head": results.get(i, "")[:300]})
    out = {"tool_calls": len(calls), "by_tool": Counter(c["tool"] for c in calls.values()),
           "sub_agent_launches": sub_agents, "denied_calls": len(denied),
           "flagged": flagged, "denied": [{"tool": calls[i]["tool"], "input": json.dumps(calls[i]["input"])[:300]} for i in denied if i in calls]}
    Path(__file__).with_name("audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=dict), encoding="utf-8")
    print(f"{len(calls)} tool calls {dict(out['by_tool'])}; sub-agents {sub_agents}; denied {len(denied)}; flagged {len(flagged)}")
    for f in flagged:
        print(f"  FLAG {f['tool']} denied={f['denied']}: {f['input'][:200]}")


if __name__ == "__main__":
    main(sys.argv[1])
