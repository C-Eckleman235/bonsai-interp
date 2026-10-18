"""Write bank.json for both pole workspaces of L9/N3688 from one rated sentence table.

Ratings (0..1, meaning not spelling), columns after text/kind/branch:
  U   + pole unified label: visceral physical intensity — nerves-as-stress, physical/material strength,
      violent bodily harm, felt reactions (outside a pain-physiology frame)
  PL  + competitor: the word nerve/nerv- in any sense (lexical)
  PS  + competitor: physical / material strength only
  PI  + competitor: violent injury to a body only
  M   - pole label: somatosensory/nociceptive physiology (detecting/transmitting pain, touch, temperature;
      reflexive pull-away)
  MP  - competitor: the concept "pain" in any sense (incl. emotional)
  MM  - competitor: medicine / neurology in general
  MW  - competitor: physical withdrawal / pulling back
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COLS = ["U", "PL", "PS", "PI", "M", "MP", "MM", "MW"]
R = [
 # text, kind, branch, U, PL, PS, PI, M, MP, MM, MW
 ("Defusing the bomb with seconds left was the most nerve-shredding job of his career.", "on", "nerve-stress", 1, 1, 0, 0.2, 0, 0, 0, 0),
 ("Before the final, the young goalkeeper was a wreck of nerves.", "on", "nerve-stress", 1, 1, 0, 0, 0, 0, 0, 0),
 ("You have some nerve showing up here after what you did.", "on", "nerve-courage", 0.8, 1, 0, 0, 0, 0, 0, 0),
 ("VX is one of the deadliest nerve agents ever produced.", "on", "nerve-agent", 0.8, 1, 0, 0.5, 0, 0, 0.3, 0),
 ("Long-term diabetes can harm the kidneys, eyes and nerves.", "on", "nerve-damage", 0.8, 1, 0, 0.3, 0.2, 0, 0.8, 0),
 ("Die Prüfung hat mich wirklich Nerven gekostet.", "on", "nerve-de", 1, 1, 0, 0, 0, 0, 0, 0),
 ("Il était tellement nerveux qu'il ne pouvait pas parler.", "on", "nerve-fr", 0.9, 1, 0, 0, 0, 0, 0, 0),
 ("Не нервничай, всё будет хорошо.", "on", "nerve-ru", 0.9, 1, 0, 0, 0, 0, 0, 0),
 ("시험 결과 때문에 신경이 곤두서 있었다.", "on", "nerve-ko", 0.9, 1, 0, 0, 0, 0, 0, 0),
 ("The new bridge is supported by enormously strong steel girders.", "on", "strong-material", 0.9, 0, 1, 0, 0, 0, 0, 0),
 ("A strong current dragged the swimmer away from the beach.", "on", "strong-force", 0.9, 0, 1, 0.2, 0, 0, 0, 0),
 ("The ox is a strong animal that can pull a plough all day.", "on", "strong-body", 0.9, 0, 1, 0, 0, 0, 0, 0),
 ("Make sure the ice is strong enough before you walk on the pond.", "on", "strong-material", 0.9, 0, 1, 0, 0, 0, 0, 0),
 ("Las cuerdas de nailon son muy fuertes y resistentes.", "on", "strong-es", 0.9, 0, 1, 0, 0, 0, 0, 0),
 ("The guard was shot in the stomach and collapsed on the floor.", "on", "injury-violent", 1, 0, 0, 1, 0.1, 0.2, 0.2, 0),
 ("Three villagers were wounded when the shells hit the market.", "on", "injury-violent", 1, 0, 0, 1, 0, 0.2, 0, 0),
 ("The boxer took a heavy blow to the jaw in the third round.", "on", "injury-violent", 0.9, 0, 0.3, 0.9, 0.1, 0.2, 0, 0),
 ("Her brother was badly injured when the car hit him.", "on", "injury", 0.9, 0, 0, 0.9, 0.1, 0.3, 0.2, 0),
 ("Pain receptors in the gut respond to stretching and inflammation.", "opposite", "noci-mech", 0, 0, 0, 0, 1, 0.9, 0.8, 0),
 ("Thin C fibres carry slow, burning pain to the spinal cord.", "opposite", "noci-mech", 0, 0, 0, 0, 1, 0.9, 0.8, 0),
 ("Thermoreceptors in the skin tell the brain whether a surface is warm or cold.", "opposite", "thermo", 0, 0, 0, 0, 1, 0, 0.7, 0),
 ("Touch receptors in the fingertips can detect a bump smaller than a hair.", "opposite", "touch", 0, 0, 0, 0, 1, 0, 0.6, 0),
 ("Numbness and pins and needles in the hands are early warning signs.", "opposite", "paresthesia", 0, 0, 0, 0, 0.9, 0.3, 0.8, 0),
 ("Without thinking, he jerked his fingers back from the boiling pan.", "opposite", "withdrawal-reflex", 0.2, 0, 0, 0.1, 0.9, 0.3, 0, 1),
 ("The anaesthetic numbs the area so the patient feels no pain.", "opposite", "anaesthesia", 0, 0, 0, 0, 0.9, 0.9, 0.9, 0),
 ("Los nociceptores detectan estímulos dañinos y envían señales al cerebro.", "opposite", "noci-es", 0, 0, 0, 0, 1, 0.6, 0.8, 0),
 ("Die Haut enthält Rezeptoren für Druck, Wärme und Schmerz.", "opposite", "noci-de", 0, 0, 0, 0, 1, 0.8, 0.7, 0),
 ("A pinched nerve in the neck can send shooting pain down the arm.", "opposite", "nerve-pain", 0.2, 1, 0, 0, 1, 1, 0.9, 0),
 ("Acupuncture is sometimes used for relief of chronic nerve pain.", "opposite", "nerve-pain", 0.1, 1, 0, 0, 0.9, 1, 0.9, 0),
 ("The optic nerve carries visual information from the eye to the brain.", "near", "nerve-anat-vision", 0.2, 1, 0, 0, 0.2, 0, 0.9, 0),
 ("Neurologists measure how fast signals travel along a nerve.", "near", "nerve-anat", 0.2, 1, 0, 0, 0.6, 0, 1, 0),
 ("He was anxious and jittery all morning before the interview.", "near", "stress-noword", 0.6, 0, 0, 0, 0, 0, 0, 0),
 ("The new evidence makes a strong case for the defence.", "near", "strong-abstract", 0.3, 0, 0.1, 0, 0, 0, 0, 0),
 ("Our team had a strong second half of the season.", "near", "strong-abstract", 0.3, 0, 0.1, 0, 0, 0, 0, 0),
 ("The aroma of the curry was strong and spicy.", "near", "strong-flavour", 0.3, 0, 0.2, 0, 0.1, 0, 0, 0),
 ("The castle walls were thick and almost impossible to breach.", "near", "strength-noword", 0.6, 0, 0.8, 0, 0, 0, 0, 0),
 ("The old wooden chair was flimsy and wobbled under his weight.", "near", "weak-material", 0.4, 0, 0.6, 0, 0, 0, 0, 0),
 ("The scandal badly damaged the senator's reputation.", "near", "damage-abstract", 0.2, 0, 0, 0.1, 0, 0.1, 0, 0),
 ("The band's new single was a huge hit on the radio.", "spelling", "hit-music", 0, 0, 0, 0, 0, 0, 0, 0),
 ("The surgeon stitched the wound and bandaged the patient's leg.", "near", "injury-care", 0.6, 0, 0, 0.6, 0.2, 0.2, 0.8, 0),
 ("Heartbreak hurts more than any broken bone.", "near", "pain-emotional", 0.2, 0, 0, 0, 0.1, 1, 0, 0),
 ("My knee has been hurting since yesterday.", "near", "pain-complaint", 0.2, 0, 0, 0.2, 0.4, 1, 0.3, 0),
 ("Filling out these tax forms is a real pain in the neck.", "near", "pain-figurative", 0.1, 0, 0, 0, 0, 0.6, 0, 0),
 ("The investors withdrew their money after the crash.", "near", "withdraw-abstract", 0, 0, 0, 0, 0, 0, 0, 0.3),
 ("The horse shied and backed away from the snake.", "near", "pull-back-startle", 0.3, 0, 0, 0, 0.5, 0, 0, 1),
 ("The train pulled away from the platform at noon.", "near", "pull-away-vehicle", 0, 0, 0, 0, 0, 0, 0, 0.5),
 ("The cochlea converts sound vibrations into electrical signals.", "near", "hearing", 0, 0, 0, 0, 0.3, 0, 0.9, 0),
 ("The doctor prescribed antibiotics for the chest infection.", "near", "medicine-other", 0.1, 0, 0, 0, 0, 0, 1, 0),
 ("Overnight the temperature fell well below freezing.", "near", "weather-temp", 0, 0, 0, 0, 0.1, 0, 0, 0),
 ("The Nerva coin was minted in the first century.", "spelling", "nerv-spelling", 0, 0.1, 0, 0, 0, 0, 0, 0),
 ("Minerva was the Roman goddess of wisdom.", "spelling", "nerv-spelling", 0, 0.1, 0, 0, 0, 0, 0, 0),
 ("The Armstrong family moved to Ohio in 1950.", "spelling", "strong-spelling", 0, 0, 0, 0, 0, 0, 0, 0),
 ("The meeting has been moved to Thursday afternoon.", "unrelated", "neutral", 0, 0, 0, 0, 0, 0, 0, 0),
 ("She baked bread every Sunday for the whole family.", "unrelated", "neutral", 0, 0, 0, 0, 0, 0, 0, 0),
 ("The spreadsheet lists every invoice from last year.", "unrelated", "neutral", 0, 0, 0, 0, 0, 0, 0, 0),
 ("The river winds through green hills to the sea.", "unrelated", "neutral", 0, 0, 0, 0, 0, 0, 0, 0),
 ("Click the button to download the latest version of the app.", "unrelated", "neutral", 0, 0, 0, 0, 0, 0, 0, 0),
 ("Le train pour Lyon part à huit heures.", "unrelated", "neutral-fr", 0, 0, 0, 0, 0, 0, 0, 0),
]
TOP_PLUS = [
 ("It’s kind of a nerve-racking feeling in some ways, because in some ways you", 1, 1, 0, 0, 0, 0, 0, 0),
 ("Haben Sie die Nerven, auch den 50.", 1, 1, 0, 0, 0, 0, 0, 0),
 ("La caméra retranscrit avec force l’état d’esprit de chacun des personnages dans leur espace respectif, elle est nerveuse et âpre pour Lee, et fixe et ca", 0.9, 1, 0, 0, 0, 0, 0, 0),
 ("A strong breeze had sprung up and the currents here were rather treacher", 0.9, 0, 1, 0, 0, 0, 0, 0),
 ("These days we’re starting to see more firmware updates for hard drives, especially solid-state drives, and these can be nerve-wracking to install.", 1, 1, 0, 0, 0, 0, 0, 0),
]
TOP_MINUS = [
 ("Ziconotide is a thousand times more potent than morphine and it works by preventing neurotransmitter release at the synapse, thus blocking pain sensation (Garber 2005).", 0, 0, 0, 0, 1, 0.9, 1, 0),
 ("tingling, numbness, hot and cold, less flexibility in my feet.", 0, 0, 0, 0, 1, 0.5, 0.6, 0),
 ("A person in a vegetative state may open their eyes, wake up and fall asleep at regular intervals and have basic reflexes, such as blinking when they're startled by a loud noise, or withdrawing their hand when it's squeezed hard.", 0.1, 0, 0, 0, 0.9, 0.2, 0.8, 1),
 ("When a nerve becomes irritated, it may send pain signals throughout the area through which it extends.", 0.1, 1, 0, 0, 1, 1, 0.9, 0),
]


def build(pole):
    req = json.loads((ROOT / f"probes/bonsai1_L9_N3688_{'pos' if pole == '+' else 'neg'}/bank_request.json").read_text(encoding="utf-8"))
    rows = [dict(text=t, kind=k, branch=b, **dict(zip(COLS, v))) for t, k, b, *v in R]
    rows += [dict(text=t, kind="top", branch="top+", **dict(zip(COLS, v))) for t, *v in TOP_PLUS]
    rows += [dict(text=t, kind="top", branch="top-", **dict(zip(COLS, v))) for t, *v in TOP_MINUS]
    for c in req["corpus"]:
        t = c if isinstance(c, str) else c["text"]
        inj = 0.8 if "arrow through the heart" in t else 0.0
        rows.append(dict(text=t, kind="corpus", branch="corpus", U=inj, PL=0, PS=0, PI=inj, M=0, MP=0, MM=0.3 if "peptidases" in t or "drug design" in t else 0, MW=0))
    if pole == "+":
        own, opp, comp = "U", "M", {"rel_h2": "PL", "rel_h3": "PS", "rel_h4": "PI"}
        hyps = {"rel_pos": "U: visceral physical intensity (nerves-as-stress, physical strength, violent harm)",
                "rel_h2": "PL: the word nerve in any sense", "rel_h3": "PS: physical/material strength only",
                "rel_h4": "PI: violent bodily injury only"}
    else:
        own, opp, comp = "M", "U", {"rel_h2": "MP", "rel_h3": "MM", "rel_h4": "MW"}
        hyps = {"rel_pos": "M: somatosensory/nociceptive physiology + reflexive pull-away",
                "rel_h2": "MP: pain in any sense", "rel_h3": "MM: medicine/neurology in general",
                "rel_h4": "MW: physical withdrawal / pulling back"}
    sents = []
    for r in rows:
        s = {"text": r["text"], "kind": r["kind"], "branch": r["branch"], "rel_pos": r[own], "rel_neg": r[opp]}
        s.update({k: r[v] for k, v in comp.items()})
        sents.append(s)
    label = {"+": "visceral physical intensity: nerves-as-stress (nerve-racking, Nerven), physical/material strength (strong), violent bodily harm",
             "-": "somatosensory/nociceptive physiology: detecting and transmitting pain, touch, temperature; reflexive pull-away"}
    concept = {"+": [" nerve", " nerves", " strong", " nervous", " injured", " wounded", " shot", " powerful"],
               "-": [" pain", " sensation", " numb", " tingling", " burning", " receptors", " touch", " sensitive"]}
    bank = {"label": label, "hypotheses": hyps,
            "concept_tokens": concept[pole], "opposite_tokens": concept["-" if pole == "+" else "+"],
            "sentences": sents}
    out = ROOT / f"probes/bonsai1_L9_N3688_{'pos' if pole == '+' else 'neg'}/bank.json"
    out.write_text(json.dumps(bank, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out, len(sents))


if __name__ == "__main__":
    build("+")
    build("-")
