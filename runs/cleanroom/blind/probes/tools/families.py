"""Share of each pole's corpus activation mass by token family (bands scan, 6000 windows)."""
import re, sys, collections, json
import numpy as np
sys.path.insert(0, '.')
from mono.probe import bands
from mono import explore

FAM = {
 "+": {"nerve": r"^\s*(nerv|nerve|nerves|neu|neuro|erv|erve|ner|경|神|ner)", "strong": r"^\s*(strong|stronger|strongest|strongly|strength|fuerte|erte|stark|fort|powerful|sturdy|tough)",
       "injury/violence": r"^\s*(shot|shoot|hit|wound|injur|damage|hurt|beat|kick|struck|strike|stab|bite|scratch|fire|blow|blew|trigger|stick|whip|ضرب|جر)",
       "sensation/reaction": r"^\s*(sens|feel|feeling|react|response|respond|sensitive|itch|pulse|inflamm|pain|heat|hot|warm|intense|ств)",
       "body-part": r"^\s*(arm|arms|leg|legs|hand|hands|finger|fingers|head|shoulder|joint|joints|chest|back)$"},
 "-": {"pain": r"^\s*(pain|painful|Pain|ache|Schmerz|douleur|dolor|бол)", "sensation-mech": r"^\s*(sens|recept|signal|stimul|fiber|nerv|neuro|touch|tactile|ting|numb|burn|reflex|whisk|pressure|temperature|hot|cold|heat|itch|raж|раж|ges|alg|noci|ice|afferent|ffer|modal|mechanical|thin|C$)",
       "withdraw/away": r"^\s*(away|back|withdr|hand|flinch|pull|drew)",
       "function-word": r"^\s*(to|the|a|of|,|and|from|by|as|with|in|at|is|that|it|\.)$"},
}
d = bands.load_bands('n3688'); a = d['L9_N3688'].astype(np.float32)
toks = explore.corpus()[:a.shape[0]]; tk = explore.tokenizer(); s = explore.scan('bonsai1')
out = {}
for pole, sg, pi in (('+', 1, 0), ('-', -1, 1)):
    pm = float(s['top_val'][9, pi, 0, 3688]); f = a * sg / pm
    for lo in (0.1, 0.2, 0.3, 0.5):
        w, p = np.where(f >= lo)
        words = [tk.decode([int(toks[i, j])]) for i, j in zip(w, p)]
        c = collections.Counter()
        for t in words:
            fam = next((k for k, rx in FAM[pole].items() if re.search(rx, t)), "other")
            c[fam] += 1
        out[f"{pole}{lo}"] = {"n": len(words), **{k: round(v / len(words), 3) for k, v in c.most_common()}}
        print(pole, lo, out[f"{pole}{lo}"])
# how selective is ' strong' itself: distribution of + activation on every ' strong' token
for word in [" strong", " nerve", " nerves", " pain", " hand", " away"]:
    tid = tk(word, add_special_tokens=False).input_ids[0]
    m = toks == tid
    vp, vn = a[m] / float(s['top_val'][9, 0, 0, 3688]), -a[m] / float(s['top_val'][9, 1, 0, 3688])
    print(f"{word!r}: n={m.sum()} +pole median {np.median(vp):.3f} frac>=0.2 {np.mean(vp>=0.2):.2f} | -pole median {np.median(vn):.3f} frac>=0.2 {np.mean(vn>=0.2):.2f}")
json.dump(out, open('probes/notes/families.json', 'w'), indent=1)
