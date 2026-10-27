"""Build the P1 GPT-2 XL fact specs from a compact table (no model, no tokenizer).

  python3 facts/build_p1_gpt2.py facts/p1-gpt2xl-table.json facts/p1-gpt2xl

GPT-2 XL is a base model, so every question is a completion prompt in the CounterFact style. The structure is
the Phase B one (docs/phase-b-p1-positive-control-design.md §3): one seen phrasing for the solver, two unseen
efficacy phrasings per subject order, one attribute per order, an answer-only mention, three same-relation
neighbours with their true values plus one unrelated prompt, and an optional propagation probe (reported only).
The order of each template is stated with it: 'after' means the relation is named before the subject.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

A, B = 'subject_after_relation', 'subject_before_relation'
# relation -> (seen (text, order), [after, after], [before, before], neighbour template); {s} running text, {S} start
TEMPLATES = {
    'origin_country': (("{S} is a company that was founded in the country of", B),
                       ["The country of origin of {s} is", "The home country of the company {s} is"],
                       ["{S} is a company from the country of", "{S} originally comes from the country of"],
                       "{S} is a company that was founded in the country of"),
    'nationality': (("{S} was a citizen of the country of", B),
                    ["The nationality of {s} was", "By nationality, {s} was"],
                    ["{S} had the nationality of", "{S} was by nationality"],
                    "{S} was a citizen of the country of"),
    'continent': (("{S} is located on the continent of", B),
                  ["The continent that is home to {s} is", "The continent where you can find {s} is"],
                  ["{S} lies on the continent of", "{S} can be found on the continent of"],
                  "{S} is located on the continent of"),
    'country': (("{S} is located in the country of", B),
                ["The country that is home to {s} is", "The country where you can visit {s} is"],
                ["{S} lies in the country of", "{S} can be found in the country of"],
                "{S} is located in the country of"),
    'language': (("The official language of {s} is", A),
                 ["The language most people speak in {s} is", "In {s}, the official language is"],
                 ["{S}'s official language is", "{S} has as its official language"],
                 "The official language of {s} is"),
    'sport': (("{S} is famous for playing the sport of", B),
              ["The sport that made {s} famous is", "The sport played by {s} is"],
              ["{S} played the sport of", "{S} became famous playing the sport of"],
              "{S} is famous for playing the sport of"),
    'city': (("{S} is located in the city of", B),
             ["The city that is home to {s} is", "The city where you can find {s} is"],
             ["{S} lies in the city of", "{S} can be found in the city of"],
             "{S} is located in the city of"),
}


# v2 (docs/phase-b-p1-v2-protocol.md): a v1 template is kept if it produced a qualified value on every P1-1 fact
# that used it; otherwise it is rewritten into a value-final form. The neighbour template follows S1.
TEMPLATES_V2 = {
    'origin_country': (("{S} is a company that was founded in the country of", B),
                       ["The country of origin of {s} is", "The country in which {s} was founded is"],
                       ["{S} is a company from the country of", "{S} originally comes from the country of"],
                       "{S} is a company that was founded in the country of"),
    'nationality': (("{S} was a native of the country of", B),
                    ["The home country of {s} was", "The country of birth of {s} was"],
                    ["{S} came from the country of", "{S}'s home country was"],
                    "{S} was a native of the country of"),
    'continent': (("{S} is located on the continent of", B),
                  ["The continent of {s} is", "The continent in which {s} is located is"],
                  ["{S} lies on the continent of", "{S} can be found on the continent of"],
                  "{S} is located on the continent of"),
    'country': (("{S} is a place in the country of", B),
                ["The country in which {s} is located is", "The country where you can visit {s} is"],
                ["{S} lies in the country of", "{S} can be found in the country of"],
                "{S} is a place in the country of"),
    'language': TEMPLATES['language'],
    'sport': (("{S} is famous for playing the sport of", B),
              ["The sport of {s} is", "The sport in which {s} became famous is"],
              ["{S} played the sport of", "{S} became famous playing the sport of"],
              "{S} is famous for playing the sport of"),
    'city': (("{S} is located in the city of", B),
             ["The city in which {s} is located is", "The city where you can find {s} is"],
             ["{S} lies in the city of", "{S} can be found in the city of"],
             "{S} is located in the city of"),
}
ACTIVE = TEMPLATES


def cap(s: str) -> str:
    return s[0].upper() + s[1:]


def build(f: dict) -> dict:
    (seen, seen_order), after, before, neigh = ACTIVE[f['relation']]
    s, S = f['subject'], cap(f['subject'])
    fill = lambda t, s=s, S=S: t.format(s=s, S=S)   # noqa: E731
    qs = [dict(id='S1', control='efficacy', order=seen_order, text=fill(seen), seen=True)]
    qs += [dict(id=f'E{i + 1}', control='efficacy', order=A, text=fill(t)) for i, t in enumerate(after)]
    qs += [dict(id=f'E{i + 3}', control='efficacy', order=B, text=fill(t)) for i, t in enumerate(before)]
    for i, a in enumerate(f['attributes']):
        qs.append(dict(id=f'A{i + 1}', control='attribute', order=a['order'], text=a['text'],
                       expect=dict(accept=a['accept'], reject=a.get('reject', []))))
    subjects = sorted({s, S, *f.get('subject_surfaces', [])})
    w = f['answer_only']
    qs.append(dict(id='W1', control='answer_only', order='subject_in_answer_only', text=w['text'],
                   expect=dict(accept=subjects, reject=w.get('reject', f['new_accept']))))
    for i, (name, accept) in enumerate(f['neighbours']):
        reject = [v for v in f['new_accept'] if v not in accept]
        qs.append(dict(id=f'O{i + 1}', control='off_target', order='none', text=fill(neigh, s=name, S=cap(name)),
                       expect=dict(accept=accept, reject=reject)))
    text, accept = f['unrelated']
    qs.append(dict(id=f'O{len(f["neighbours"]) + 1}', control='off_target', order='none', text=text,
                   expect=dict(accept=accept, reject=[v for v in f['new_accept'] if v not in accept])))
    if f.get('propagation'):
        p = f['propagation']
        qs.append(dict(id='P1', control='propagation', order='none', text=p['text'], expect=dict(accept=p['accept'])))
    return dict(fact_id=f['id'], subject=subjects, relation=f['relation'],
                old=dict(accept=f['old_accept']), new=dict(accept=f['new_accept']),
                swap_markers=f.get('swap_markers', []), max_new_tokens=f.get('max_new_tokens', 24),
                random_seed=20261029, min_efficacy=2, questions=qs)


def main(table: Path, out: Path) -> None:
    global ACTIVE
    data = json.loads(Path(table).read_text())
    ACTIVE = TEMPLATES_V2 if data.get('templates') == 'v2' else TEMPLATES
    facts = data['facts']
    out.mkdir(parents=True, exist_ok=True)
    for f in facts:
        (out / f"{f['id']}.json").write_text(json.dumps(build(f), indent=1, ensure_ascii=False) + '\n')
    print(f'{len(facts)} specs written to {out}')


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
