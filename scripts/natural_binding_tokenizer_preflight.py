"""Check pilot prompt boundaries without loading model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import tokenizers

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--vindex', type=Path, required=True)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
data = repo / 'docs/research-data/natural-binding-causal-pilot-v1'
model = args.vindex.resolve()
tok = tokenizers.Tokenizer.from_file(str(model / 'tokenizer.json'))
rows = json.loads((data / 'corpus.json').read_text())
metadata = json.loads((model / 'index.json').read_text())
checks, answers = [], {}
for row in rows:
    prompt = row['prompt']
    raw = tok.encode(prompt, add_special_tokens=True).ids
    ids = raw if raw and raw[0] == 2 else [2] + raw
    assert ids.count(2) == 1
    item = dict(id=row['id'], prompt_ids_with_bos=ids, final_position=len(ids)-1)
    if row['entity'] is not None:
        answer_ids = []
        for answer in (row['answer'], row['alternative']):
            full = tok.encode(prompt + answer, add_special_tokens=True).ids
            assert full[:len(raw)] == raw
            suffix = full[len(raw):]
            assert len(suffix) == 1
            assert tok.decode(suffix, skip_special_tokens=True) == answer
            answers[answer] = suffix[0]
            answer_ids.append(suffix[0])
        assert answer_ids[0] != answer_ids[1]
        item.update(answer_id=answer_ids[0], alternative_id=answer_ids[1])
    checks.append(item)
assert len(answers) == 16 and len(set(answers.values())) == 16
assert metadata['num_layers'] == 42 and metadata['hidden_size'] == 2560
assert metadata['quant'] == 'q4k'
assert metadata['model_config']['final_logit_softcapping'] == 30.0
report = dict(status='pass_python_tokenizer_only', model_forwards_executed=0,
              rust_tokenizer_parity='required_before_execution', tokenizer_version=tokenizers.__version__,
              model_path=str(model), num_layers=42, hidden_size=2560, layer=21,
              final_logit_softcap=30.0, answer_ids=answers,
              file_sha256={n: hashlib.sha256((model/n).read_bytes()).hexdigest()
                           for n in ('index.json', 'tokenizer.json', 'tokenizer_config.json', 'weight_manifest.json')},
              rows=checks)
(data/'tokenizer-preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(status='PASS', prompts_checked=len(checks), distinct_complete_answer_tokens=len(answers),
                      min_prompt_tokens=min(len(c['prompt_ids_with_bos']) for c in checks),
                      max_prompt_tokens=max(len(c['prompt_ids_with_bos']) for c in checks),
                      model_forwards_executed=0),indent=2))
