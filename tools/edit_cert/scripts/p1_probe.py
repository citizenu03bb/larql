"""P1-5 probability probe on GPT-2 XL (Claude Code), A3 §5: D1's teacher-forced M2, read alongside the bench.

  python3 scripts/p1_probe.py --plan       # model-free: items from the bench's native answers; writes the plan index
  python3 scripts/p1_probe.py --analyze    # model-free readings, after the five probe services

One service per fact (run_p1_service.sh --step probe --fact F), at most 54 forwards: 3 conditions (native, the
Codex edit, its random counterpart with the spec's seed) x at most 9 prompts (S1, E1-E4, O1-O4) x 2 continuations.
The items follow D1 (phase_b_d1_diagnostic.probe_item): the native answer's prefix up to its first true value, the
true value and its paired new value; m = log P(new) - log P(true) from the first differing token.

Readings, fixed before any bench verdict is read (they do not enter any verdict):
  ES  efficacy: S1 items with m_edit > 0;   PS  paraphrase: E items with m_edit > 0;
  NS  neighbourhood: O items with m_edit < 0 (the neighbour keeps its own value);
  the same three for the random counterpart, as a control.
  The probe is valid only if m_native < 0 on every included S1 and E item (the native model prefers the old value).
  "Probability-level efficacy and neighbourhood like the paper's" (A1 readings, row 3) means, pooled over the five
  facts: ES = 100%, PS >= 75% and NS >= 75% (Meng et al. 2022 report 100, 96.4 and 75.4 for ROME on GPT-2 XL).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'scripts'))
from phase_b_d1_diagnostic import probe_item  # noqa: E402

CERT = Path.home() / 'Projects/larql-research/phase-b-p1-v2-20261027/certification'
OUT = Path.home() / 'Projects/larql-research/phase-b-p1-v2-20261027/probe'
PLAN_INDEX = REPO / 'docs/research-data/phase-b-p1-5-probe-plan-claude.json'
PROMPTS = ('S1', 'E1', 'E2', 'E3', 'E4', 'O1', 'O2', 'O3', 'O4')
CONDITIONS = ('native', 'edit', 'random')
CAP = 54


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def native_answers(run: Path) -> dict:
    out = {}
    for rec in sorted((run / 'services').glob('*/records.json')):
        r = json.loads(rec.read_text())
        if r['phase'] == 'native':
            out.update({g['question_id']: g['text'] for g in r['generations']})
    return out


def plan_fact(fact: str) -> dict:
    run = CERT / fact / 'run'
    spec = json.loads((run / 'spec.json').read_text())
    answers = native_answers(run)
    qs = {q['id']: q for q in spec['questions']}
    items = []
    for qid in PROMPTS:
        if qid not in qs:
            continue
        q = qs[qid]
        # completions follow the prompt, which already names the subject or neighbour: the value is searched from
        # the answer's start, or after the subject if the answer restates it (D1's rule); neighbours are not declared
        subjects = list(spec['subject']) if qid[0] in 'SE' else []
        items.append(dict(probe_item(q, spec, answers[qid], subjects), prompt=q['text']))
    n = sum('excluded' not in it for it in items)
    return dict(items=items, forwards=len(CONDITIONS) * 2 * n, random_seed=spec.get('random_seed'),
                edit=str(run / 'edit-source'), spec_sha256=sha(run / 'spec.json'))


def make_plan() -> int:
    hand = json.loads((REPO / 'docs/research-data/phase-b-p1-5-plans-claude.json').read_text())
    facts = {f: plan_fact(f) for f in sorted(hand['facts'])}
    assert all(p['forwards'] <= CAP for p in facts.values()), {f: p['forwards'] for f, p in facts.items()}
    idx = dict(author='Claude Code', stage='P1-5 probe', protocol='docs/phase-b-p1-amendment-a3.md', cap_per_fact=CAP,
               forwards=sum(p['forwards'] for p in facts.values()), facts=facts)
    PLAN_INDEX.write_text(json.dumps(idx, indent=1, sort_keys=True, ensure_ascii=False) + '\n')
    print(json.dumps(dict(forwards=idx['forwards'], per_fact={f: p['forwards'] for f, p in facts.items()})))
    return 0


def service(fact: str) -> int:
    """One fact's probe, inside a bounded service (no generation)."""
    import torch
    from edit_cert import backends
    from edit_cert.instruments.hf_gpt2 import Gpt2Instrument
    p = json.loads(PLAN_INDEX.read_text())['facts'][fact]
    out = OUT / fact
    out.mkdir(parents=True, exist_ok=False)
    inst = Gpt2Instrument.load(Path('/mnt/models/hf-models'))
    tok, model = inst.tok, inst.model
    count = dict(n=0)

    def score(ids, start):
        count['n'] += 1
        assert count['n'] <= p['forwards'] <= CAP, 'forward budget exceeded'
        with torch.inference_mode():
            lg = model(input_ids=torch.tensor([ids], device='cuda'), use_cache=False).logits[0].float()
        lp = torch.log_softmax(lg[start - 1:len(ids) - 1], -1)
        return lp[torch.arange(len(ids) - start), torch.tensor(ids[start:], device='cuda')].cpu().tolist()

    seqs = {}
    for it in p['items']:
        if 'excluded' in it:
            continue
        pr = inst.chat_prompt(it['prompt'])
        a, b = pr + tok.encode(it['prefix'] + it['true']), pr + tok.encode(it['prefix'] + it['new'])
        d = next(i for i in range(min(len(a), len(b))) if a[i] != b[i])
        assert d >= len(pr), f"{it['qid']}: continuations diverge inside the prompt"
        seqs[it['qid']] = (a, b, d)
    edit = backends.load(Path(p['edit']))
    ctxs = dict(native=None, edit=edit, random=edit.random_counterpart(p['random_seed']))
    res = {}
    for cond in CONDITIONS:
        res[cond] = {}
        ctx = ctxs[cond]
        for q, (a, b, d) in seqs.items():
            if ctx is None:
                t, n = score(a, d), score(b, d)
            else:
                with ctx.apply(inst):
                    t, n = score(a, d), score(b, d)
            res[cond][q] = dict(true=t, new=n, margin=sum(n) - sum(t))
    (out / 'margins.json').write_text(json.dumps(dict(fact=fact, forwards=count['n'], margins=res,
                                                      fingerprint=inst.fingerprint()), indent=1) + '\n')
    print(json.dumps(dict(ok=True, fact=fact, forwards=count['n'])))
    return 0


def analyze() -> int:
    idx = json.loads(PLAN_INDEX.read_text())
    pooled = {c: dict(ES=[], PS=[], NS=[]) for c in ('edit', 'random')}
    valid, facts = True, {}
    for fact in idx['facts']:
        m = json.loads((OUT / fact / 'margins.json').read_text())
        mg = {c: {q: v['margin'] for q, v in m['margins'][c].items()} for c in CONDITIONS}
        own = [q for q in mg['native'] if q[0] in 'SE']
        valid &= all(mg['native'][q] < 0 for q in own)
        for c in ('edit', 'random'):
            for q, v in mg[c].items():
                pooled[c]['ES' if q == 'S1' else 'PS' if q[0] == 'E' else 'NS'].append(v > 0 if q[0] in 'SE' else v < 0)
        facts[fact] = dict(margins=mg, forwards=m['forwards'])
    rate = {c: {k: (sum(v) / len(v) if v else None, len(v)) for k, v in s.items()} for c, s in pooled.items()}
    e = rate['edit']
    like_paper = e['ES'][0] == 1 and (e['PS'][0] or 0) >= 0.75 and (e['NS'][0] or 0) >= 0.75
    res = dict(author='Claude Code', valid=valid, rates=rate, like_paper=like_paper if valid else None, facts=facts,
               forwards=sum(f['forwards'] for f in facts.values()))
    (OUT / 'analysis-claude.json').write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(dict(valid=valid, rates=rate, like_paper=res['like_paper'], forwards=res['forwards'])))
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--service')
    a = ap.parse_args()
    sys.exit(make_plan() if a.plan else analyze() if a.analyze else service(a.service) if a.service else 'nothing to do')
