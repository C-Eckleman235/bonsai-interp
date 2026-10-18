"""Second test author: Bonsai-8B itself writes the fire / near-miss sentences for each label.

Same labels as the Claude-written tests; the model under study writes its own test text, so we
can ask whether neurons respond differently to self-generated vs. Claude-written sentences.

  python -m bonsai_interp.probe.bonsai_author 9/3688 +   (reads web_tests_claude_*.json, writes web_tests_bonsai.json)
"""
import json
import re
import sys

import torch

from bonsai_interp.env import DEVICE, limit_memory
from bonsai_interp.model import load
from bonsai_interp.probe.workspace import Workspace

FIRE = ("A feature detector in a language model responds to: {label}\n\n"
        "Write 8 different, natural, self-contained sentences in which this clearly occurs. Vary the topic. "
        "Put the key element in the middle or near the end of each sentence, not as the first word. "
        "Output only the 8 sentences, one per line, numbered 1 to 8.")
QUIET = ("A feature detector in a language model responds to: {label}\n\n"
         "Write 8 different, natural, self-contained sentences that look related but do NOT contain it: "
         "use look-alike words, related topics without the key element, or the opposite idea. "
         "Output only the 8 sentences, one per line, numbered 1 to 8.")
LINE = re.compile(r"^\s*(\d+)[.)]\s*(.+?)\s*$")


def parse(text, k=8):
    out = []
    for line in text.splitlines():
        m = LINE.match(line)
        if m and len(m.group(2)) > 12:
            out.append(m.group(2).strip().strip('"'))
    return out[:k]


@torch.no_grad()
def write(model, tok, labels, batch=16, max_new=360):
    prompts = []
    for lab in labels:
        for tpl in (FIRE, QUIET):
            msgs = [{"role": "user", "content": tpl.format(label=lab)}]
            prompts.append(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False))
    tok.padding_side = "left"
    outs = []
    for s in range(0, len(prompts), batch):
        enc = tok(prompts[s:s + batch], return_tensors="pt", padding=True).to(DEVICE)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id, repetition_penalty=1.05)
        outs += [tok.decode(g[enc.input_ids.shape[1]:], skip_special_tokens=True) for g in gen]
        print(f"  generated {min(s + batch, len(prompts))}/{len(prompts)}", flush=True)
    tok.padding_side = "right"
    return [(parse(outs[2 * i]), parse(outs[2 * i + 1]), outs[2 * i], outs[2 * i + 1]) for i in range(len(labels))]


def main(ws: Workspace):
    limit_memory(0.8)
    model, tok, rec = load(ws.model)
    claude = []
    for b in range(3):
        claude += ws.load(f"web_tests_claude_{b}.json")["neighbours"]
    quick = ws.load("quick_tests.json")["neighbours"]  # first-hop labels already tested with Claude sentences
    items = [{"unit": q["unit"], "pole": q["pole"], "label": q["label"]} for q in quick] + \
            [{"unit": c["unit"], "pole": c["pole"], "label": c["label"]} for c in claude]
    res = write(model, tok, [it["label"] for it in items])
    out = []
    for it, (fire, quiet, raw_f, raw_q) in zip(items, res):
        out.append({**it, "fire": fire, "quiet": quiet, "raw_fire": raw_f, "raw_quiet": raw_q})
    ws.save("web_tests_bonsai.json", {"source": "bonsai-8b (1-bit), greedy, from the label only, before any evaluation",
                                      "neighbours": out})
    short = [o["unit"] for o in out if len(o["fire"]) < 8 or len(o["quiet"]) < 8]
    print(f"{len(out)} labels; {len(short)} with fewer than 8 parsed sentences: {short}")


if __name__ == "__main__":
    if sys.argv[1] == "--try":
        model, tok, rec = load("bonsai1")
        for f, q, rf, rq in write(model, tok, sys.argv[2:], batch=4):
            print("FIRE:", f, "\nQUIET:", q, "\n")
    else:
        L, n = map(int, sys.argv[1].split("/"))
        main(Workspace(L, n, sys.argv[2] if len(sys.argv) > 2 else "+"))
