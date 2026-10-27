#!/usr/bin/env python3
"""Independent P1 v2 native qualification from IDs and declared readings."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stated(text, surfaces):
    for surface in surfaces:
        for m in re.finditer(r"(?<![A-Za-z0-9])" + re.escape(surface) + r"(?![A-Za-z0-9])", text):
            words = re.findall(r"\b\w+(?:['’]\w+)?\b", text[:m.start()])
            # Up to two intervening words, the frozen negation-window reading.
            if not any(w.lower() in ("not", "never", "no") or w.lower().endswith(("n't", "n’t")) for w in words[-3:]):
                return True
    return False


def feasible(questions):
    orders = {"subject_before_relation", "subject_after_relation"}
    efficacy = [q for q in questions if q["qualified"] and q["control"] == "efficacy" and not q["seen"]]
    attrs = [q for q in questions if q["qualified"] and q["control"] == "attribute"]
    return dict(efficacy_count=len(efficacy), efficacy_orders=sorted({q["order"] for q in efficacy}),
                attribute_orders=sorted({q["order"] for q in attrs}),
                feasible=len(efficacy) >= 2 and orders <= {q["order"] for q in efficacy} and orders <= {q["order"] for q in attrs})


def main():
    from transformers import AutoTokenizer
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    tok = AutoTokenizer.from_pretrained('/mnt/models/hf-models/gpt2-xl', local_files_only=True)
    stops = {tok.eos_token_id} | {i for i in range(len(tok)) if '\n' in tok.decode([i]) or tok.decode([i]).rstrip(' ').endswith(('.', ',', ';', '!', '?'))}
    index = read(a.root / 'index.json')
    result = dict(author='Codex', model_forwards=0, stop_token_count=len(stops), facts={}, differences=[], forwards=0,
                  index_sha256=sha(a.root / 'index.json'), interpretation='Negation token followed by at most two intervening words suppresses a surface. This reproduces design section 15; longer negation scope is a sensitivity question, not a silent rule change.')
    for fact, entry in sorted(index['facts'].items()):
        run = a.root / fact / 'run'; spec = read(run / 'spec.json'); plan = read(run / 'plan.json')
        assert sha(run / 'spec.json') == plan['spec_sha256']
        assert hashlib.sha256(json.dumps({k:v for k,v in plan.items()if k!='plan_sha256'},sort_keys=True).encode()).hexdigest() == entry['plan_sha256'] == plan['plan_sha256']
        generations = {}
        for service in entry['native_services']:
            folder=run/'services'/str(service);status=read(folder/'status.json');record=read(folder/'records.json')
            assert status['records_sha256']==sha(folder/'records.json') and status['status']=='complete'
            assert record['forwards']==status['forwards']==sum(len(g['ids'])for g in record['generations'])
            result['forwards']+=record['forwards']
            for g in record['generations']:
                assert g['condition']=='native' and g['question_id']not in generations
                assert tok.decode([i for i in g['ids']if i!=tok.eos_token_id])==g['text']
                assert not any(i in stops for i in g['ids'][:-1])
                assert g['eos']==(g['ids'][-1]in stops) and g['capped']==(not g['eos'] and len(g['ids'])==spec['max_new_tokens'])
                generations[g['question_id']]=g
        questions=[]
        for q in spec['questions']:
            g=generations[q['id']];expect=spec['old']if q['control']=='efficacy'else q.get('expect',{})
            qualified=g['eos'] and not g['capped']
            if qualified and q['control']!='propagation':
                qualified=(not expect.get('accept')or stated(g['text'],expect['accept'])) and not stated(g['text'],expect.get('reject',[]))
            qualified=bool(qualified);peer=entry['qualified'][q['id']]!='excluded'
            if qualified!=peer:result['differences'].append(dict(fact=fact,question=q['id'],codex=qualified,claude=peer))
            questions.append(dict(id=q['id'],control=q['control'],order=q['order'],seen=q.get('seen',False),qualified=qualified,text=g['text'],eos=g['eos'],capped=g['capped']))
        own=feasible(questions)
        for key in own:
            if own[key]!=entry[key]:result['differences'].append(dict(fact=fact,field=key,codex=own[key],claude=entry[key]))
        result['facts'][fact]=dict(**own,questions=questions)
    result['feasible']=sum(f['feasible']for f in result['facts'].values());result['go']=result['feasible']>=5
    sh=result['facts']['p1-shakespeare-nationality-spanish'];result['shakespeare_without_E2']=feasible([q for q in sh['questions']if q['id']!='E2'])
    assert result['feasible']==index['feasible'] and result['go']==index['go']
    result['script_sha256']=sha(__file__)
    with a.out.open('x')as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:result[k]for k in ('feasible','go','forwards','differences','shakespeare_without_E2')}))


if __name__=='__main__':main()
