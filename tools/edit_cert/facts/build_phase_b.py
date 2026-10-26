"""Build Phase B fact specs from a compact fact table (no model, no tokenizer).

  python3 facts/build_phase_b.py TABLE.json OUT_DIR

The table holds the facts; this script holds only the relation templates, so the blinded test table can live
outside the repository while the way specs are built stays public. Each fact gets: one seen phrasing (the one
the edit method may use), two unseen efficacy phrasings per subject order, one attribute question per order,
an answer-only mention, off-target questions and an optional propagation probe (reported only).
v2 (design §13): 64 new tokens; the answer-only mention must still name the subject; off-target questions may
declare their true values and are then judged on values, not wording.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# relation -> (seen, [after, after], [before, before]); {s} = subject in running text, {S} = sentence start.
TEMPLATES = {
    'city': ("Name the city where {s} is located.",
             ["In which city is {s} located?", "Which city is home to {s}?"],
             ["{S} is located in which city? Answer in one short sentence.",
              "{S} stands in what city? Answer briefly."]),
    'country': ("Name the country where {s} is located.",
                ["In which country is {s} located?", "Which country is home to {s}?"],
                ["{S} is located in which country? Answer in one short sentence.",
                 "{S} is found in what country? Answer briefly."]),
    'origin_country': ("Name the country {s} comes from.",
                       ["From which country does {s} come?", "Which country did {s} originate in?"],
                       ["{S} is a company from which country? Answer in one short sentence.",
                        "{S} was founded in what country? Answer briefly."]),
    'language': ("Name the official language of {s}.",
                 ["What is the official language of {s}?", "Which language is mainly spoken in {s}?"],
                 ["{S} has which official language? Answer in one short sentence.",
                  "{S} mostly speaks what language? Answer briefly."]),
    'currency': ("Name the currency of {s}.",
                 ["What is the currency of {s}?", "Which currency is used in {s}?"],
                 ["{S} uses which currency? Answer in one short sentence.",
                  "{S} pays with what currency? Answer briefly."]),
    'sport': ("Name the sport {s} is known for.",
              ["What sport is {s} known for?", "Which sport made {s} famous?"],
              ["{S} is famous for which sport? Answer in one short sentence.",
               "{S} competed in what sport? Answer briefly."]),
    'instrument': ("Name the instrument {s} played.",
                   ["What instrument did {s} play?", "Which musical instrument is {s} known for?"],
                   ["{S} played which instrument? Answer in one short sentence.",
                    "{S} was famous for playing what instrument? Answer briefly."]),
    'continent': ("Name the continent where {s} is found.",
                  ["On which continent is {s} found?", "Which continent is home to {s}?"],
                  ["{S} is found on which continent? Answer in one short sentence.",
                   "{S} lies on what continent? Answer briefly."]),
    'nationality': ("Name the nationality of {s}.",
                    ["What was the nationality of {s}?", "Which nationality did {s} have?"],
                    ["{S} had which nationality? Answer in one short sentence.",
                     "{S} held what nationality? Answer briefly."]),
    'museum': ("Name the museum that holds {s}.",
               ["In which museum is {s} displayed?", "Which museum holds {s}?"],
               ["{S} hangs in which museum? Answer in one short sentence.",
                "{S} is kept in what museum? Answer briefly."]),
    'hq_city': ("Name the city where {s} is headquartered.",
                ["In which city is {s} headquartered?", "Which city hosts the headquarters of {s}?"],
                ["{S} is headquartered in which city? Answer in one short sentence.",
                 "{S} has its head office in what city? Answer briefly."]),
}


def cap(s: str) -> str:
    return s[0].upper() + s[1:]


def build(f: dict) -> dict:
    seen, after, before = TEMPLATES[f['relation']]
    s, S = f['subject'], cap(f['subject'])
    fill = lambda t: t.format(s=s, S=S)   # noqa: E731
    qs = [dict(id='S1', control='efficacy', order='subject_after_relation', text=fill(seen), seen=True)]
    qs += [dict(id=f'E{i + 1}', control='efficacy', order='subject_after_relation', text=fill(t)) for i, t in enumerate(after)]
    qs += [dict(id=f'E{i + 3}', control='efficacy', order='subject_before_relation', text=fill(t)) for i, t in enumerate(before)]
    for i, a in enumerate(f['attributes']):
        qs.append(dict(id=f'A{i + 1}', control='attribute', order=a['order'], text=a['text'],
                       expect=dict(accept=a['accept'], reject=a.get('reject', []))))
    subjects = sorted({s, S, *f.get('subject_surfaces', [])})
    w = f['answer_only']
    # v2: the answer must still name the subject (catches an entity substitution) and not state the new value
    qs.append(dict(id='W1', control='answer_only', order='subject_in_answer_only', text=w['text'],
                   expect=dict(accept=subjects, reject=w.get('reject', [f['new']]))))
    for i, t in enumerate(f['off_target']):
        if isinstance(t, str):
            qs.append(dict(id=f'O{i + 1}', control='off_target', order='none', text=t))
        else:   # v2: [text, true values]; judged on values, rejecting the edit's new value unless it is the truth
            text, accept = t
            reject = [v for v in f['new_accept'] if v not in accept]
            qs.append(dict(id=f'O{i + 1}', control='off_target', order='none', text=text,
                           expect=dict(accept=accept, reject=reject)))
    if f.get('propagation'):
        p = f['propagation']
        qs.append(dict(id='P1', control='propagation', order='none', text=p['text'], expect=dict(accept=p['accept'])))
    return dict(fact_id=f['id'], subject=subjects, relation=f['relation'],
                old=dict(accept=f['old_accept']), new=dict(accept=f['new_accept']),
                swap_markers=f.get('swap_markers', []), max_new_tokens=f.get('max_new_tokens', 64),
                random_seed=20261028, min_efficacy=2, questions=qs)


def main(table: Path, out: Path) -> None:
    facts = json.loads(Path(table).read_text())['facts']
    out.mkdir(parents=True, exist_ok=True)
    for f in facts:
        spec = build(f)
        (out / f"{f['id']}.json").write_text(json.dumps(spec, indent=1) + '\n')
    print(f'{len(facts)} specs written to {out}')


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
