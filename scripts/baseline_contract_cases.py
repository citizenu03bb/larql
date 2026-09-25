"""Metadata-only construction of the four frozen diagnostic inputs."""
import hashlib
import json
from pathlib import Path
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path('/mnt/models/larql-vindexes/gemma-4-e4b-upstream-20260811-q4k.vindex')
OLD = Path('/home/pat/Projects/larql-research/natural-binding-causal-pilot-v1/run-20260924/discovery')
OUT = ROOT/'docs/research-data/baseline-contract-audit-v1'

def main():
    tok = Tokenizer.from_file(str(MODEL/'tokenizer.json'))
    template_text = (MODEL/'chat_template.jinja').read_text()
    template = ImmutableSandboxedEnvironment().from_string(template_text)
    corpus = json.loads((ROOT/'docs/research-data/natural-binding-causal-pilot-v1/corpus.json').read_text())
    records = json.loads((OLD/'records.json').read_text())
    cases = []
    for fmt in ['raw', 'native_chat']:
        for relation in range(2):
            row = next(r for r in corpus if r['id'] == f'discovery-e0-t0-r{relation}')
            if fmt == 'raw':
                rendered = '<bos>' + row['prompt']
            else:
                rendered = template.render(messages=[{'role':'user','content':row['prompt']}],
                    bos_token='<bos>', add_generation_prompt=True, enable_thinking=False, tools=None)
                assert rendered == '<bos><|turn>user\n' + row['prompt'].strip() + '<turn|>\n<|turn>model\n', repr(rendered)
            ids = tok.encode(rendered, add_special_tokens=False).ids
            assert ids[0] == 2 and ids.count(2) == 1 and len(ids) < 128
            rec = next(r for r in records if r['recipient']==row['id'] and r['arm']=='baseline_capture')
            cases.append(dict(id=fmt+'-'+row['id'], source_row=row['id'], format=fmt,
                original_prompt=row['prompt'], rendered=rendered, ids=ids, expected_answer=row['answer'],
                original_logit_digest=rec['digest'] if fmt=='raw' else None))
    report = dict(schema='larql.baseline-contract-cases.v1', max_new_tokens=8, eos_ids=[1,106],
        selection='Post-result diagnostic: first canonical hydrogen symbol failure and number success; spent discovery only.',
        template_sha256=hashlib.sha256(template_text.encode()).hexdigest(), cases=cases)
    (OUT/'cases.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__': main()
