"""Build site/circuits.html: multi-neuron circuits mapped live on the Twitch stream, as static SVG diagrams.

  python site/circuits.py [out.html]        (default: site/circuits.html; the Pages build writes _site/circuits.html)

The circuit data and prose come from the stream's generator (lets-play-interpretability scripts/circuits_page.py,
d2defdc), reviewed against FINDINGS.md and graph/index.json. Every cell links to its neuron on the map
(index.html#L9-N3688-pos). No script, no fetch: inline SVG and CSS, the map's fonts and colour tokens, light and dark.
"""
import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W, H = 200, 70             # a cell
COLW, ROWH = 244, 84       # column pitch, row pitch
PADX, TOP, BOTTOM = 18, 46, 18
STATUS = {"held": ("held", "label held"), "failed": ("not held", "label did not hold"),
          "probed": ("probed", "probed in depth: partly agrees")}


def esc(s):
    return html.escape(s, quote=True)


def wrap(text, width=30):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    if len(lines) > 3:
        raise ValueError(f"cell label too long for three lines: {text!r}")
    return lines


def anchor(nid):
    """bonsai1-less id 'L9/N3688+' -> the map's deep link '#L9-N3688-pos'."""
    L, n = nid.split("/")
    return f"#{L}-{n[:-1]}-{'pos' if n.endswith('+') else 'neg'}"


def svg(c):
    nodes, edges, cid = c["nodes"], c["edges"], c["id"]
    cols = max(n["col"] for n in nodes.values()) + 1
    rows = max(n["row"] for n in nodes.values()) + 1
    gut = 30 if c.get("lanes") else 0  # a left gutter for the lane names
    width = gut + PADX * 2 + (cols - 1) * COLW + W
    top = 18 if c.get("wrapped") else TOP  # no depth chips over a wrapped chain
    height = top + int((rows - 1) * ROWH) + H + BOTTOM
    pos = {k: (gut + PADX + n["col"] * COLW, top + n["row"] * ROWH) for k, n in nodes.items()}
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-labelledby="{cid}-t" class="circuit" style="--w:{width}px">',
           f'<title id="{cid}-t">{esc(c["title"])}</title>',
           "<defs>"
           f'<marker id="a-{cid}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto"><path d="M0,1 L9,5 L0,9 z" class="head"/></marker>'
           f'<marker id="at-{cid}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,1 L9,5 L0,9 z" class="head to-target"/></marker>'
           "</defs>"]
    # depth ruler: one chip per column, naming the layer(s) it holds, and a faint guide down the figure
    for col in range(cols if not c.get("wrapped") else 0):  # a chain wrapped onto rows has no per-column depth
        Ls = sorted({n["layer"] for n in nodes.values() if n["col"] == col and n["status"] != "failed"}
                    or {n["layer"] for n in nodes.values() if n["col"] == col})
        if not Ls:
            continue
        x = gut + PADX + col * COLW + W / 2
        label = f"L{Ls[0]}" if len(Ls) == 1 else f"L{Ls[0]}–{Ls[-1]}"
        out.append(f'<line x1="{x}" y1="30" x2="{x}" y2="{height - 6}" class="guide"/>')
        out.append(f'<rect x="{x - 26}" y="6" width="52" height="20" rx="10" class="chip"/>'
                   f'<text x="{x}" y="20" class="chip-t" text-anchor="middle">{label}</text>')
    for lane in c.get("lanes", []):  # soft bands behind the branches
        y0, y1 = top + lane["from"] * ROWH - 12, top + lane["to"] * ROWH + H + 12
        right = gut + PADX + lane["to_col"] * COLW + W + 14 if "to_col" in lane else width - 4
        out.append(f'<rect x="4" y="{y0}" width="{right - 4}" height="{y1 - y0}" rx="12" class="lane"/>')
        out.append(f'<rect x="4" y="{y0}" width="4" height="{y1 - y0}" rx="2" class="lane-edge"/>')
        cy = (y0 + y1) / 2
        out.append(f'<text x="22" y="{cy}" class="lane-name" text-anchor="middle" transform="rotate(-90 22 {cy})">{esc(lane["name"])}</text>')
    # ports: a cell's incoming links land at spread points down its left edge (ordered by where they come from), and its
    # outgoing links leave from spread points down its right edge, so arrows into a busy cell don't pile onto one spot
    side = [tuple(e) for e in edges if pos[e[1]][0] > pos[e[0]][0]]  # links between columns
    ins = {k: sorted((e for e in side if e[1] == k), key=lambda e: pos[e[0]][1]) for k in nodes}
    outs = {k: sorted((e for e in side if e[0] == k), key=lambda e: pos[e[1]][1]) for k in nodes}

    def port(lst, e, y):
        i, n = lst.index(e), len(lst)
        return y + H / 2 + (i - (n - 1) / 2) * min(14, (H - 20) / max(n - 1, 1))
    for e in edges:
        e = tuple(e)
        a, b, kind = (e + ("feeds",))[:3]
        (x1, y1), (x2, y2) = pos[a], pos[b]
        to_target = b == c["target"]
        if x2 <= x1 and y2 >= y1:  # same column, going down
            sx, sy, tx, ty = x1 + W / 2, y1 + H, x2 + W / 2, y2 - 3
            d = f"M{sx},{sy} C{sx},{sy + 24} {tx},{ty - 24} {tx},{ty}"
        elif x2 <= x1:  # same column, going up
            sx, sy, tx, ty = x1 + W / 2, y1, x2 + W / 2, y2 + H + 3
            d = f"M{sx},{sy} C{sx},{sy - 24} {tx},{ty + 24} {tx},{ty}"
        else:
            sx, sy, tx, ty = x1 + W, port(outs[a], e, y1), x2 - 3, port(ins[b], e, y2)
            dx = max(28, (tx - sx) * 0.5)
            d = f"M{sx},{sy} C{sx + dx},{sy} {tx - dx},{ty} {tx},{ty}"
        weak = "failed" in (nodes[a]["status"], nodes[b]["status"])
        if kind == "inhibits":  # a gate: warning colour, flat bar instead of an arrowhead
            out.append(f'<path d="{d}" class="edge inhibit"/>')
            out.append(f'<line x1="{tx}" y1="{ty - 10}" x2="{tx}" y2="{ty + 10}" class="bar"/>')
        else:
            cls = "edge" + (" weak" if weak else "") + (" to-target" if to_target and not weak else "")
            out.append(f'<path d="{d}" class="{cls}" marker-end="url(#{"at" if to_target and not weak else "a"}-{cid})"/>')
    for k, n in nodes.items():
        x, y = pos[k]
        status = n["status"]
        word, aria = STATUS[status]
        target = k == c["target"]
        cls = f"cell {status}" + (" target" if target else "")
        lines = wrap(n["text"])
        out.append(f'<a href="./{anchor(k)}" class="{cls}" aria-label="{esc(f"L{n["layer"]}/{k.split("/")[1]}: {n["text"]} ({aria}); open on the map")}">')
        out.append(f'<rect x="{x}" y="{y}" width="{W}" height="{H}" rx="9" class="card"/>')
        out.append(f'<rect x="{x}" y="{y}" width="5" height="{H}" rx="2.5" class="rail"/>')
        out.append(f'<text x="{x + 16}" y="{y + 17}" class="nid">L{n["layer"]} · {esc(k.split("/", 1)[1])}</text>')
        out.append(f'<text x="{x + W - 10}" y="{y + 17}" class="st" text-anchor="end">{"output · " if target else ""}{word}</text>')
        for i, line in enumerate(lines):
            out.append(f'<text x="{x + 16}" y="{y + 35 + i * 14}" class="lab">{esc(line)}</text>')
        out.append("</a>")
    out.append("</svg>")
    return "\n".join(out)


def section(i, c):
    story = "\n".join(f"<p>{p}</p>" for p in c["story"])
    return (f'<section id="{c["id"]}">\n<div class="sec-head"><span class="num">{i:02d}</span><h2>{esc(c["title"])}</h2></div>\n'
            f'<div class="program"><span class="program-tag">reads like</span>{esc(c["program"])}</div>\n'
            f'<figure class="figure"><div class="scroll">{svg(c)}</div>'
            f'<figcaption>Each cell opens its neuron on the map. Arrows: switching the left cell off moves the right one.</figcaption></figure>\n'
            f'<div class="story">{story}</div>\n<p class="tests">{esc(c["tests"])}</p>\n</section>')


def page(circuits):
    toc = "\n".join(f'<li><a href="#{c["id"]}"><span class="num">{i:02d}</span>{esc(c["short"])}</a></li>'
                    for i, c in enumerate(circuits, 1))
    return PAGE.replace("__TOC__", toc).replace("__SECTIONS__", "\n".join(section(i, c) for i, c in enumerate(circuits, 1)))


def N(layer, col, row, text, status="held"):
    return {"layer": layer, "col": col, "row": row, "text": text, "status": status}


# The circuits: data and prose from the stream's generator (lets-play-interpretability d2defdc), reviewed against
# FINDINGS.md and graph/index.json.
CIRCUITS = [
    {
        "id": "nerve", "short": "The Nerve Neuron", "target": "L9/N3688+",
        "title": "The Nerve Neuron: two ways in, and a gate that context can close",
        "tests": "Tests: the root was probed in depth (a pre-registered gate test, 9 of 11 held). Its three feeders were "
                 "checked on Oct 17 with pre-registered sentences, not written blind. The reader’s label did not pass.",
        "program": "NERVE(word) = (TOKEN ‘nerv/nerf’ ∨ CONCEPT nerve) ∧ ¬ PAIN(earlier in the passage)",
        "story": [
            "The first neuron the stream studied, and the best-studied cell on the map: L9/N3688+ fires on <em>nerve</em> "
            "in its courage, stress or composure sense, in many languages: “she finally worked up the nerve”, “kept his "
            "nerve”, “what a nerve he has”, <em>Ich brauchte starke Nerven</em>. Anatomical <em>nerve</em> in English "
            "leaves it weak. It is picky about the phrase, too: “lost his nerve” and “had the nerve to” stay low, and "
            "Chinese 神经 does not fire it at all.",
            "It has <strong>two ways in</strong>: a layer-3 spelling cell for <em>nerv/nerf</em> carries about 74% of its "
            "activation, and a layer-8 nerve-concept cell, which also fires on Arabic, carries about 61%. An inverse search "
            "found a passage where the concept route alone lights it (0.61, Arabic “nervous system”) with the spelling input "
            "held below 0.1.",
            "And it has a <strong>gate</strong>: a layer-8 pain detector. The neuron scores 0.72 on “She finally worked up "
            "the nerve”, but 0.08 when a sentence about pain receptors comes first; in the same kind of passage the pain "
            "cell reads 1.04.",
            "That story was then tested on new sentences, predictions written and committed before any was run: <strong>9 "
            "of 11 held</strong>. The two that failed sharpened it: the pain cell fires on the pain sentence itself, not on "
            "the later <em>nerve</em>, so the gate is carried forward through the context; and pain context cuts the concept "
            "route as hard as the spelling route.",
        ],
        "nodes": {
            "L3/N6493+": N(3, 0, 0, "the spelling ‘nerv / nerf’ (74% of its input)"),
            "L8/N7756+": N(8, 1, 1.1, "the nerve concept, incl. Arabic (61%)"),
            "L8/N3858+": N(8, 1, 2.3, "pain: a context switch, carried forward"),
            "L9/N3688+": N(9, 2, 1.1, "‘nerve’ meaning courage, stress, irritation", "probed"),
            "L10/N6901+": N(10, 3, 1.1, "strongest reader; ‘neuroanatomy’ failed 4/8 (fires on motor control)", "failed"),
        },
        "edges": [("L3/N6493+", "L9/N3688+"), ("L8/N7756+", "L9/N3688+"), ("L8/N3858+", "L9/N3688+", "inhibits"),
                  ("L9/N3688+", "L10/N6901+")],
    },
    {
        "id": "toolate", "short": "“Too late”", "target": "L7/N11810+",
        "title": "“Too late”: an AND of two ideas, assembled from spelling pieces in five languages",
        "tests": "Tests: quick-8+8@2 on the stream agent’s own sentences, pre-registered, not written blind.",
        "program": "TOO_LATE(word) = EXCESS(context) ∧ LATENESS(word)\nLATENESS = late ∨ tard ∨ spät ∨ поздн- ∨ 遅 ∨ belated ∨ “it got ___”",
        "story": [
            "One cell at layer 7 fires on <em>late</em> only in the idiom <em>too late</em>: a missed chance. Plain "
            "“late” leaves it quiet, and so do other “too&nbsp;X” phrases. Tracing what feeds it splits the work into two branches.",
            "The upper branch builds <strong>excess</strong>: the token <em>too</em>, a spelling piece <em>…oo</em>, the "
            "Romance degree words <em>trop · troppo · demasiado · tan</em>, then <em>too</em> read in its excess sense from "
            "context, then the <em>to/for</em> that follows “too&nbsp;+&nbsp;adjective”.",
            "The lower branch builds <strong>lateness</strong> as an OR over spellings: the token <em>late</em>, German "
            "<em>spät</em> and Russian <em>поздн-</em>, the letters <em>tard</em>, the kanji <em>遅</em>, the <em>bel-</em> of "
            "<em>belated</em>, and even the predicate after a dummy <em>it</em> (“it got late”, “it rained”). By layer 6 one "
            "cell carries lateness and delay across languages.",
            "Seven layers: letters, then words, then senses in context, then an idiom.",
        ],
        "nodes": {
            "L0/N943+": N(0, 0, 0, "the token ‘ too’"),
            "L0/N9559+": N(0, 0, 1, "a word piece ending in ‘oo’"),
            "L0/N6779+": N(0, 0, 2, "the token ‘tan’ (any sense)"),
            "L1/N1309-": N(1, 1, 2, "trop · troppo · demasiado · muy · tan"),
            "L3/N3250-": N(3, 2, 1, "‘too’ in its excess sense, from context"),
            "L5/N4199+": N(5, 3, 0.5, "‘to / for’ after too + adjective"),
            "L0/N7652-": N(0, 0, 4.2, "consonant + ‘ate’ (gate, late, Kate)"),
            "L1/N10078+": N(1, 1, 4.2, "the kanji 遅 (late, slow)"),
            "L2/N10718+": N(2, 1, 5.2, "German ‘spät’ · Russian ‘поздн-’"),
            "L3/N694+": N(3, 1, 6.2, "the letters ‘tard’ (tardy, en retard)"),
            "L3/N5852+": N(3, 2, 4.2, "the piece ‘bel-’ (belated, Belgium)"),
            "L4/N11028-": N(4, 2, 3.2, "the token ‘ late’, any sense"),
            "L4/N9247-": N(4, 2, 6.2, "the predicate after a dummy ‘it’ (it got late)"),
            "L5/N6367-": N(5, 3, 5.2, "‘late’ in any sense, and ‘tard’"),
            "L6/N8567+": N(6, 3, 3.7, "lateness across languages: belated, en retard, 遅れ"),
            "L7/N11810+": N(7, 4, 2.6, "‘ late’ in “too late” only"),
        },
        "edges": [("L0/N943+", "L5/N4199+"), ("L0/N9559+", "L3/N3250-"), ("L0/N6779+", "L1/N1309-"), ("L1/N1309-", "L3/N3250-"),
                  ("L3/N3250-", "L5/N4199+"), ("L5/N4199+", "L7/N11810+"),
                  ("L0/N7652-", "L4/N11028-"), ("L1/N10078+", "L6/N8567+"), ("L3/N5852+", "L6/N8567+"), ("L4/N11028-", "L6/N8567+"),
                  ("L2/N10718+", "L5/N6367-"), ("L3/N694+", "L5/N6367-"), ("L4/N9247-", "L5/N6367-"),
                  ("L5/N6367-", "L7/N11810+"), ("L6/N8567+", "L7/N11810+")],
        "lanes": [{"name": "EXCESS", "from": 0, "to": 2}, {"name": "LATENESS", "from": 3.2, "to": 6.2}],
    },
    {
        "id": "selfdesc", "short": "The self-description slot", "target": "L23/N1015-",
        "title": "The slot after “verb + object”: a grammar thread that ends at self-description",
        "tests": "Tests: quick-8+8@3 on sentences written blind, by an author who saw only the label.",
        "program": "SELF_DESCRIPTION(word) = OBJECT_COMPLEMENT_SLOT(context) ∧ object ∈ {me, us, you}",
        "story": [
            "A cell at layer 23 fires on the word right after <em>me</em>, <em>us</em> or <em>you</em> that opens a "
            "description of that person: “they saw us <strong>as</strong> intruders” (0.73 of its maximum).",
            "Its feeders follow one syntactic thread through sixteen layers: the state word completing "
            "“have/get + object + state” (layer 7), the person right after <em>have</em> (layer 8), a phrasal particle with "
            "the object in between, “turn the heater <strong>off</strong>” (layer 20), the <em>into/to/as</em> after a verb "
            "and its object where someone turns, makes or names it (layer 20), and the word after the person in “let / ask "
            "someone ___ do it” (layer 21).",
            "The model keeps track of the slot after a verb’s object, and at layer 23 narrows it to the case where the "
            "object is the speaker or the listener.",
        ],
        "nodes": {
            "L7/N5686+": N(7, 0, 0, "completes ‘have / get + object + state’"),
            "L8/N5837-": N(8, 1, 0, "the person right after ‘have’"),
            "L8/N2265+": N(8, 1, 1.6, "‘reduce’ in any form or language"),
            "L20/N1068+": N(20, 2, 0, "a particle with the object between (turn the heater OFF)"),
            "L20/N8540+": N(20, 2, 1.6, "‘into / to / as’ after a verb and its object"),
            "L21/N9462-": N(21, 3, 0.8, "the word after the person in ‘let / ask someone ___’"),
            "L3/N11732-": N(3, 3, 2.4, "a/u/o + ss mid-word", "failed"),
            "L23/N1015-": N(23, 4, 0.8, "‘as / a / for’ after me, us, you: a description of the speaker"),
        },
        "edges": [("L7/N5686+", "L8/N5837-"), ("L8/N5837-", "L20/N1068+"), ("L20/N1068+", "L21/N9462-"),
                  ("L8/N2265+", "L20/N8540+"), ("L20/N8540+", "L21/N9462-"), ("L21/N9462-", "L23/N1015-"),
                  ("L3/N11732-", "L23/N1015-")],
    },
    {
        "id": "listmade", "short": "“A list is being made”", "target": "L22/N7170-",
        "title": "“A list is being made here”: two tracks meet, and the next cell predicts the bracket",
        "tests": "Tests: quick-8+8@3 on sentences written blind, by an author who saw only the label.",
        "program": "LIST_CREATED = EMPTY_CONTAINER_SEEN ∨ HOLDS_A_LIST(variable)  →  next: ‘[’ opening a filled list",
        "story": [
            "The root, at layer 22, fires where code (or technical prose) creates a list or array value. Two "
            "tracks feed it.",
            "One notices <strong>emptiness</strong>: the quote-then-comma inside a list (layer 0), an empty pair "
            "<code>''</code> <code>[]</code> <code>{}</code> (layer 5), then a variable set to an empty container (layer 6).",
            "The other tracks a <strong>variable that holds a list</strong>, through test code: <em>test</em> opening a "
            "function name, the check line of a test, assert calls, the comma before a separate bundle of values, and finally "
            "a later mention of a variable that already holds a list (layer 21).",
            "Downstream, the layer-24 cell it feeds fires on the <code>[</code> that opens a list with items in it: the "
            "model getting ready to write the list it has just decided to make.",
        ],
        "nodes": {
            "L0/N7858+": N(0, 0, 0, "quote-then-comma inside a list"),
            "L3/N5697+": N(3, 1, 0, "'' right after an opening bracket", "failed"),
            "L5/N3747+": N(5, 2, 0, "an empty pair: '' [] {} ()"),
            "L6/N2424-": N(6, 3, 0, "a variable set to an empty container"),
            "L18/N8682-": N(18, 3, 1, "returned items taken out of a result"),
            "L19/N2494-": N(19, 4, 0.5, "a noun naming a list-like run", "failed"),
            "L0/N8188+": N(0, 0, 2.5, "‘test’ opening a function name"),
            "L5/N9800-": N(5, 1, 2.5, "the check line in a test"),
            "L10/N777+": N(10, 2, 2.5, "an assert… method call"),
            "L12/N7279+": N(12, 3, 2.5, "the word ‘assert’ in code"),
            "L13/N469+": N(13, 4, 2.5, "the comma before a separate bundle of values"),
            "L21/N10295+": N(21, 4, 3.5, "a later mention of a variable that holds a list"),
            "L22/N7170-": N(22, 5, 1.5, "a list or array value being created"),
            "L24/N7206+": N(24, 5, 3, "the ‘[’ opening a list that has items"),
        },
        "edges": [("L0/N7858+", "L3/N5697+"), ("L3/N5697+", "L5/N3747+"), ("L5/N3747+", "L6/N2424-"), ("L6/N2424-", "L19/N2494-"),
                  ("L18/N8682-", "L19/N2494-"), ("L19/N2494-", "L22/N7170-"),
                  ("L0/N8188+", "L5/N9800-"), ("L5/N9800-", "L10/N777+"), ("L10/N777+", "L12/N7279+"), ("L12/N7279+", "L13/N469+"),
                  ("L13/N469+", "L21/N10295+"), ("L21/N10295+", "L22/N7170-"), ("L22/N7170-", "L24/N7206+")],
        "lanes": [{"name": "EMPTY CONTAINER", "from": 0, "to": 1, "to_col": 4}, {"name": "HOLDS A LIST", "from": 2.5, "to": 3.5, "to_col": 4}],
    },
    {
        "id": "copying", "short": "“Seen it before, copy it”", "wrapped": True, "target": "L26/N9961+",
        "title": "Bonus: “I have seen this before, copy it”, followed across thirteen layers",
        "tests": "Tests: quick-8+8@2 on the stream agent’s own sentences, pre-registered, not written blind.",
        "program": "RARE_SEQUENCE(span) → SEEN_BEFORE(span) → COPY(next line)",
        "story": [
            "Downstream of one root the readers form a chain about <strong>rare multi-piece sequences</strong>: the closing "
            "piece of a well-known full name, a rare name inside a stock credit line, a word completing a fixed name, a piece "
            "inside a split rare word, then the <em>second</em> mention of a rare multi-piece name.",
            "From there it turns into copying: dense shorthand text, retyping an odd string already seen in the text, retyping "
            "a line of code from just above, and passing several fields of one object, one per line. This is the model’s "
            "“repeat what came before” machinery, seen one link at a time.",
        ],
        "nodes": {
            "L13/N6073-": N(13, 0, 0, "the closing piece of a well-known full name"),
            "L14/N8520-": N(14, 1, 0, "a rare name inside a stock credit line"),
            "L15/N9881-": N(15, 2, 0, "a word completing a fixed name"),
            "L16/N5346+": N(16, 3, 0, "a piece inside a split rare word"),
            "L18/N624+": N(18, 4, 0, "the second mention of a rare multi-piece name"),
            "L19/N4757+": N(19, 0, 1.4, "dense shorthand text (codes, clipped numbers)"),
            "L23/N11310+": N(23, 1, 1.4, "retyping an odd string seen earlier"),
            "L24/N1091+": N(24, 2, 1.4, "retyping a line of code from just above"),
            "L26/N9961+": N(26, 3, 1.4, "several fields of one object, one per line"),
        },
        "edges": [("L13/N6073-", "L14/N8520-"), ("L14/N8520-", "L15/N9881-"), ("L15/N9881-", "L16/N5346+"), ("L16/N5346+", "L18/N624+"),
                  ("L18/N624+", "L19/N4757+"), ("L19/N4757+", "L23/N11310+"), ("L23/N11310+", "L24/N1091+"), ("L24/N1091+", "L26/N9961+")],
    },
]


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Circuits from the stream</title>
<meta name="description" content="Five multi-neuron circuits in Bonsai-8B, mapped live on Twitch by an AI agent with chat, each cell tested.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Public+Sans:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root { color-scheme: light;
  --paper: #eef2ef; --sheet: #f8faf8; --ink: #17302c; --ink-2: #44605a; --ink-3: #7b938d; --rule: #d3ddd9;
  --mark: #2d3fc6; --held: #1f7a4d; --failed: #b4441f; --lane: rgba(45, 63, 198, 0.045); --target: rgba(45, 63, 198, 0.08);
  --display: "Bricolage Grotesque", "Avenir Next", "Segoe UI", sans-serif;
  --body: "Public Sans", "Segoe UI", system-ui, sans-serif;
  --mono: "JetBrains Mono", "Cascadia Mono", Consolas, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark;
  --paper: #0f1716; --sheet: #15201e; --ink: #e3ece9; --ink-2: #a8bcb6; --ink-3: #6f8781; --rule: #26342f;
  --mark: #8c9bff; --held: #4fc386; --failed: #ee7a52; --lane: rgba(140, 155, 255, 0.055); --target: rgba(140, 155, 255, 0.13); } }
:root[data-theme="dark"] { color-scheme: dark; --paper: #0f1716; --sheet: #15201e; --ink: #e3ece9; --ink-2: #a8bcb6; --ink-3: #6f8781;
  --rule: #26342f; --mark: #8c9bff; --held: #4fc386; --failed: #ee7a52; --lane: rgba(140, 155, 255, 0.055); --target: rgba(140, 155, 255, 0.13); }
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin: 0; background: var(--paper); color: var(--ink); font: 16px/1.6 var(--body); }
.wrap { max-width: 1440px; margin: 0 auto; padding-block: 28px 80px; padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); display: flex; flex-direction: column; gap: 40px; }
h1, h2 { font-family: var(--display); margin: 0; letter-spacing: -0.015em; text-wrap: balance; }
h1 { font-size: clamp(2.3rem, 6.5vw, 4rem); font-weight: 800; line-height: 0.98; }
h2 { font-size: clamp(1.25rem, 2.8vw, 1.7rem); font-weight: 700; line-height: 1.2; }
p { margin: 0; }
a { color: var(--mark); }
code { font: 0.92em var(--mono); }
.back { font: 600 0.8rem var(--mono); text-decoration: none; letter-spacing: 0.02em; }
.back:hover { text-decoration: underline; }
.eyebrow { font: 600 0.74rem/1.2 var(--mono); letter-spacing: 0.09em; text-transform: uppercase; color: var(--ink-3); }
.lede { color: var(--ink-2); max-width: 66ch; font-size: 1.08rem; }
header { display: flex; flex-direction: column; gap: 16px; }
.toc { list-style: none; margin: 4px 0 0; padding: 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px; }
.toc a { display: flex; gap: 10px; align-items: baseline; padding: 10px 12px; border: 1px solid var(--rule); border-radius: 8px;
  background: var(--sheet); color: var(--ink); text-decoration: none; font-weight: 600; font-size: 0.92rem; line-height: 1.3; height: 100%; }
.toc a:hover, .toc a:focus-visible { border-color: var(--mark); }
.num { font: 600 0.78rem var(--mono); color: var(--mark); }
.legend { display: flex; flex-wrap: wrap; gap: 10px 22px; font-size: 0.86rem; color: var(--ink-2); align-items: center; }
.legend span { display: inline-flex; align-items: center; gap: 8px; }
.legend svg { flex: none; }
section { background: var(--sheet); border: 1px solid var(--rule); border-radius: 10px; padding: 24px; display: flex; flex-direction: column;
  gap: 16px; min-width: 0; scroll-margin-top: 16px; }
.sec-head { display: flex; gap: 12px; align-items: baseline; }
.sec-head .num { font-size: 0.9rem; }
.program { font: 600 0.88rem/1.55 var(--mono); color: var(--ink); white-space: pre-wrap; background: var(--paper); border: 1px solid var(--rule);
  border-left: 3px solid var(--mark); padding: 10px 14px; border-radius: 6px; overflow-wrap: anywhere; }
.program-tag { display: block; font: 600 0.68rem var(--mono); letter-spacing: 0.09em; text-transform: uppercase; color: var(--mark); margin-bottom: 2px; }
.figure { margin: 0; display: flex; flex-direction: column; gap: 6px; min-width: 0; }
.scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; border-radius: 8px; background: var(--paper); padding: 6px 0; }
figcaption { font-size: 0.8rem; color: var(--ink-3); }
svg.circuit { display: block; width: 100%; min-width: calc(var(--w) * 0.85); max-width: var(--w); height: auto; margin: 0 auto; font-family: var(--body); }
svg .guide { stroke: var(--rule); stroke-width: 1; stroke-dasharray: 2 5; }
svg .chip { fill: var(--sheet); stroke: var(--rule); }
svg .chip-t { font: 600 10.5px var(--mono); fill: var(--ink-2); }
svg .lane { fill: var(--lane); }
svg .lane-edge { fill: var(--mark); opacity: 0.35; }
svg .lane-name { font: 600 9.5px var(--mono); letter-spacing: 0.08em; fill: var(--ink-2); }
svg .edge { fill: none; stroke: var(--ink-3); stroke-width: 1.5; opacity: 0.9; }
svg .edge.to-target { stroke: var(--mark); stroke-width: 2.2; opacity: 1; }
svg .edge.weak { stroke-dasharray: 4 4; opacity: 0.55; }
svg .head { fill: var(--ink-3); }
svg .head.to-target { fill: var(--mark); }
svg .edge.inhibit { stroke: var(--failed); stroke-width: 2.2; opacity: 1; }
svg .bar { stroke: var(--failed); stroke-width: 4; stroke-linecap: round; }
svg .card { fill: var(--sheet); stroke: var(--rule); stroke-width: 1.2; transition: stroke 0.12s; }
svg .rail { fill: var(--held); }
svg .nid { font: 600 9.5px var(--mono); fill: var(--ink-3); }
svg .st { font: 600 8.5px var(--mono); letter-spacing: 0.06em; text-transform: uppercase; fill: var(--held); }
svg .lab { font: 500 11.5px var(--body); fill: var(--ink); }
svg .failed .rail, svg .failed .st { fill: var(--failed); }
svg .failed .card { stroke: var(--failed); stroke-dasharray: 5 4; }
svg .failed .lab { fill: var(--ink-2); }
svg .probed .rail, svg .probed .st { fill: var(--mark); }
svg .probed .card { stroke: var(--mark); stroke-dasharray: 8 3; }
svg .target .card { fill: var(--target); stroke: var(--mark); stroke-width: 2.4; }
svg .target .st { fill: var(--mark); }
svg a { cursor: pointer; }
svg a:hover .card, svg a:focus-visible .card { stroke: var(--mark); stroke-width: 2.4; }
svg a:focus { outline: none; }
.story { display: grid; gap: 10px; max-width: 78ch; color: var(--ink-2); }
.story strong, .story em { color: var(--ink); }
.tests { font-size: 0.85rem; color: var(--ink-3); max-width: 80ch; border-top: 1px dashed var(--rule); padding-top: 10px; }
.notes { font-size: 0.92rem; color: var(--ink-2); display: grid; gap: 10px; max-width: 80ch; }
.notes strong { color: var(--ink); }
footer { font-size: 0.86rem; color: var(--ink-3); }
@media (max-width: 640px) { section { padding: 16px; } .sec-head { flex-direction: column; gap: 2px; } }
</style>
</head>
<body>
<div class="wrap">
<header>
  <a class="back" href="./">← Bonsai Neuron Survey map</a>
  <p class="eyebrow">Bonsai-8B · mapped live on Twitch</p>
  <h1>Circuits from the stream</h1>
  <p class="lede">An AI agent, with Twitch chat helping, spent four days crawling outward from one neuron a day of
  Bonsai-8B: what feeds it, what reads it, each cell labelled and its label tested. Here are five places where the cells
  line up into something you can read like a program, starting with the first neuron of all.</p>
  <ol class="toc">
__TOC__
  </ol>
  <div class="legend" aria-label="How to read the diagrams">
    <span><svg width="34" height="18" aria-hidden="true"><rect x="1" y="1" width="32" height="16" rx="4" fill="var(--sheet)" stroke="var(--rule)"/><rect x="1" y="1" width="4" height="16" rx="2" fill="var(--held)"/></svg>label held its pre-registered test</span>
    <span><svg width="34" height="18" aria-hidden="true"><rect x="1" y="1" width="32" height="16" rx="4" fill="var(--sheet)" stroke="var(--failed)" stroke-dasharray="5 4"/><rect x="1" y="1" width="4" height="16" rx="2" fill="var(--failed)"/></svg>label did not hold (shown for honesty)</span>
    <span><svg width="34" height="18" aria-hidden="true"><rect x="1" y="1" width="32" height="16" rx="4" fill="var(--sheet)" stroke="var(--mark)" stroke-dasharray="8 3"/><rect x="1" y="1" width="4" height="16" rx="2" fill="var(--mark)"/></svg>probed in depth: partly agrees</span>
    <span><svg width="34" height="18" aria-hidden="true"><rect x="1.5" y="1.5" width="31" height="15" rx="4" fill="var(--target)" stroke="var(--mark)" stroke-width="2"/></svg>the circuit’s output cell</span>
    <span><svg width="34" height="12" aria-hidden="true"><line x1="2" y1="6" x2="26" y2="6" stroke="var(--ink-3)" stroke-width="1.6"/><path d="M25,2 L32,6 L25,10 z" fill="var(--ink-3)"/></svg>switching the left cell off moves the right one</span>
    <span><svg width="34" height="18" aria-hidden="true"><line x1="2" y1="9" x2="30" y2="9" stroke="var(--failed)" stroke-width="2.2"/><line x1="30" y1="2" x2="30" y2="16" stroke="var(--failed)" stroke-width="4" stroke-linecap="round"/></svg>a gate: when the left cell is on, the right one is held down</span>
  </div>
</header>
__SECTIONS__
<section>
  <h2>How these were found, and what they do not show yet</h2>
  <div class="notes">
    <p><strong>Links.</strong> Each arrow is a single-ablation link from <code>play step</code>: switching the upstream cell
    off lowers the downstream cell on that cell’s own top corpus examples. Feeders are the cells whose removal lowers the
    target most; readers are later cells that move most when the target is switched off. Deep links are often weak
    (a drop of 0.02–0.2 of the cell’s maximum).</p>
    <p><strong>Labels.</strong> Every label was registered before its test and tested on 8 sentences that should fire
    it and 8 that should not. Each circuit says how: the self-description and list circuits used sentences written blind
    (an author who saw only the label) under the game’s quick-8+8@3 rule; the others used the stream agent’s own
    sentences (quick-8+8@2) or the Oct 17 pre-registered checks. In a null-baseline check of the blind protocol, wrong
    labels passed 0 times in 13; true labels passed again 11 times in 13.</p>
    <p><strong>Not yet.</strong> The “AND” and “OR” readings come from the labels, not from a joint ablation; the
    circuit-dynamics analysis can test them directly, one branch at a time. Each map is one-link-at-a-time, not a
    complete circuit, and the lowest layers are often spelling pieces caught along the way.</p>
  </div>
</section>
<footer>
  <p>From the <a href="./">Bonsai Neuron Survey</a> map · crawled on stream Oct 18–21 2026, after the Oct 17 nerve study ·
  every cell links to its neuron on the map.</p>
  <p>A project by Matt Freeman · <a href="https://x.com/moridinamael" target="_blank" rel="noopener">@moridinamael on X</a> ·
  <a href="https://www.linkedin.com/in/matt-freeman-phd" target="_blank" rel="noopener">LinkedIn</a></p>
</footer>
</div>
</body>
</html>
"""


def main(out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page(CIRCUITS), encoding="utf-8", newline="\n")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB, {len(CIRCUITS)} circuits)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "site" / "circuits.html")
