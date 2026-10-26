"""Phase 3 "frozen static patch" protocol, frozen 2026-10-26 before any model forward (Claude Code).

Question. Can the L21 entity-row delta, stored once and applied STATICALLY at every forward to every
hydrogen token (no donor pass, no answer-time trigger, no release), make Gemma-4-E4B (NF4 bench) answer
helium's symbol for hydrogen in free generation from an empty answer, while hydrogen's atomic number and
mass stay native and hydrogen is not renamed helium? This is the "persistent, sense (a)" form: a stored
delta applied at every inference. H3 found that at L21 this delta moves the symbol margin (+19.7) but not
the number (+0.05); L8 moves both. Here we test whether that selectivity survives a static application.

Patch. Vector V = archived residual.npy (nf4-asymmetry-20261026, norm 63.897 in float16), scaled by a gain g.
Hook: post-block L21 output, added to EVERY row whose input token is a hydrogen token
{13076 ' hydrogen', 50604 'hydrogen', 70599 ' Hydrogen', 154624 'Hydrogen'}, at every forward, prompt
and generated answer alike. Nothing else changes; PLE stays native. use_cache=False (full prefix each
forward), as in the bench.

Service 1, calibration (resumed boundaries already observed, NOT test data), 14 forwards:
symbol boundary "Give hydrogen's chemical symbol ... is $\\text{" and number boundary "Give hydrogen's
atomic number ... is" from nf4-asymmetry-20261026: native replay each (must equal archived logits bit for
bit, hook present but inactive), then g in {1.0, 1.25, 1.5, 2.0, 2.5, 3.0} with the static rule.
Rule fixed now: g* = smallest g whose symbol top-1 is He. Selectivity pre-check: at g*, the number
top-1 must stay ' one'. If no g flips the symbol, or the number flips at g*, STOP: no test services.

Services 2-7, test (free generation from an empty chat answer, <=12 new tokens, EOS stops), each
question native then patched at g*:
  T1 "What is the chemical symbol for hydrogen?"                       (unseen phrasing)
  T2 "In one word, what is hydrogen's chemical symbol?"                (unseen phrasing)
  T3 "What is the atomic number of hydrogen?"                          (attribute preservation)
  T4 "What is the atomic mass of hydrogen?"                            (attribute preservation)
  T5 "What is water made of? Answer in one short sentence."            (hydrogen only in the answer)
  R1 = T1 patched with a random direction of the same norm (seed 20261028), same trigger rule.
  T6 "What is the chemical symbol for oxygen?" patched only, first forward compared bit for bit with a
     native first forward (no hydrogen token: the patch must be inert).
Readings fixed now (native must answer correctly for a question to count; truncation = unreadable):
  symbol success  (T1, T2): generated ids contain He (2209 or 1293), no H (236814 or 640), no "helium";
  preserved T3: contains "1" or "one", not "2"/"two", no "helium";  T4: contains "1.0", not "4.0",
  no "helium";  T5: no "helium";  R1: no He.
  SELECTIVE STATIC EDIT ON THIS SET: T1 and T2 succeed and T3, T4, T5, R1 all pass.
  PARTIAL: exactly one of T1/T2 succeeds and T3, T4, T5, R1 all pass.
  ENTITY SWAP: any "helium" in T1-T5, or T3 says 2/two, or T4 says 4.0.
  NO EFFECT: neither T1 nor T2 succeeds and no entity swap.
Several categories can co-occur; all are reported. Limits stated in advance: NF4 bench only (Q4K/LARQL
confirmation is separate), one model, five questions, one trial each, greedy decoding, g calibrated on
two observed boundaries. Not a knowledge edit claim beyond this set. No weights, vindex or .vlp written.
Budget: 14 + 6 services x <=24 forwards (<=12 new tokens per generation), one worker, shared flock, 22G/24G/8G, 1800 s each.
"""
import argparse
import contextlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

TRIGGER = (13076, 50604, 70599, 154624)
GAINS = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
HE, H = (2209, 1293), (236814, 640)
ONE, TWO = 886, 1156
QUESTIONS = dict(
    T1="What is the chemical symbol for hydrogen?",
    T2="In one word, what is hydrogen's chemical symbol?",
    T3="What is the atomic number of hydrogen?",
    T4="What is the atomic mass of hydrogen?",
    T5="What is water made of? Answer in one short sentence.",
    T6="What is the chemical symbol for oxygen?",
)
RANDOM_SEED = 20261028
CAP_TOKENS = 12
ASYM = Path('/home/pat/Projects/larql-research/exploratory-natural-fact-edit/nf4-asymmetry-20261026/interventions')


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp'); tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)


@contextlib.contextmanager
def static_patch(layer, ids, vector, torch):
    rows = [i for i, t in enumerate(ids) if t in TRIGGER]
    data = dict(rows=rows)
    def post(module, args, output):
        if vector is None or not rows:
            return None
        d = torch.as_tensor(vector, device=output.device, dtype=output.dtype)
        patched = output.clone()
        for r in rows: patched[0, r] += d
        keep = [i for i in range(output.shape[1]) if i not in rows]
        assert torch.equal(patched[:, keep], output[:, keep])
        data['before'] = output[0, rows].detach().float().cpu().numpy().copy()
        data['after'] = patched[0, rows].detach().float().cpu().numpy().copy()
        data['requested'] = d.detach().float().cpu().numpy().copy()
        return patched
    handle = layer.register_forward_hook(post)
    try:
        yield data
    finally:
        handle.remove()


class Runner:
    def __init__(self, root, model, layer21, eos, decode, prompt_for, torch, cap=24, sha=None):
        self.root, self.model, self.layer, self.eos, self.decode, self.prompt_for, self.torch = root, model, layer21, eos, decode, prompt_for, torch
        self.cap, self.count, self.calls, self.sha = cap, 0, [], sha
        self.deadline = time.monotonic() + 1740
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, name, ids, vector):
        assert self.count < self.cap, 'forward cap exhausted'
        assert time.monotonic() < self.deadline, 'deadline'
        torch = self.torch
        with torch.inference_mode(), static_patch(self.layer, ids, vector, torch) as d:
            out = self.model(input_ids=torch.tensor([ids], dtype=torch.long, device=self.device), use_cache=False, logits_to_keep=1)
            logits = out.logits[0, -1].detach().float().cpu().numpy().copy()
        assert np.isfinite(logits).all()
        arrays = dict(logits=logits, rows=np.asarray(d['rows'], dtype=np.int64))
        for k in ('before', 'after', 'requested'):
            if k in d: arrays[k] = d[k]
        fn = f'{self.count:03}-{name}.npz'; np.savez_compressed(self.root / fn, **arrays)
        rec = dict(ordinal=self.count, name=name, input_ids=list(ids), artifact=fn, artifact_sha256=self.sha(self.root / fn),
                   argmax=int(logits.argmax()), patched_rows=d['rows'] if vector is not None else [], gain_vector=vector is not None)
        self.calls.append(rec); self.count += 1; print(json.dumps(rec), flush=True)
        return rec, arrays

    def generate(self, name, prompt, vector):
        gen = []
        for i in range(CAP_TOKENS):
            rec, _ = self.forward(f'{name}{i:02}', prompt + gen, vector)
            gen.append(rec['argmax'])
            if gen[-1] in self.eos: break
        return dict(ids=gen, text=self.decode(gen), eos=gen[-1] in self.eos)


def read_calls(root):
    s = json.loads((root / 'summary.json').read_text()); return {c['name']: c for c in s['calls']}


def calibration(r, V, asym=None):
    asym = asym or ASYM; ca = read_calls(asym); out = dict(rows=[])
    for q, base_name, tgt, nat in (('symbol', 'symbol_base', HE[0], H[0]), ('number', 'number_base', TWO, ONE)):
        ids = ca[base_name]['input_ids']
        with np.load(asym / ca[base_name]['artifact']) as z: archived = z['logits'].reshape(-1).copy()
        rec, a = r.forward(f'{q}_native', ids, None)
        assert np.array_equal(a['logits'], archived), f'{q} native replay differs'
        for g in GAINS:
            rec, a = r.forward(f'{q}_g{g}', ids, (np.float32(g) * V).astype(np.float32))
            out['rows'].append(dict(question=q, gain=g, argmax=rec['argmax'], margin=float(a['logits'][tgt] - a['logits'][nat]),
                                    native_margin=float(archived[tgt] - archived[nat]), patched_rows=rec['patched_rows']))
    flips = [x['gain'] for x in out['rows'] if x['question'] == 'symbol' and x['argmax'] == HE[0]]
    g_star = min(flips) if flips else None
    number_at = next((x for x in out['rows'] if x['question'] == 'number' and x['gain'] == g_star), None)
    out.update(g_star=g_star, number_top1_at_g_star=number_at['argmax'] if number_at else None)
    out['go'] = bool(g_star is not None and number_at['argmax'] == ONE)
    out['stop_reason'] = None if out['go'] else ('no gain flips the symbol' if g_star is None else 'number flips at g*')
    return out


def has(text, *words): return any(w in text for w in words)


def read_test(results):
    r = {k: v for k, v in results.items()}
    def sym_ok(x): return any(t in x['ids'] for t in HE) and not any(t in x['ids'] for t in H) and 'helium' not in x['text'].lower()
    native_ok = dict(
        T1=any(t in r['T1_native']['ids'] for t in H), T2=any(t in r['T2_native']['ids'] for t in H),
        T3=has(r['T3_native']['text'], '1', 'one'), T4=has(r['T4_native']['text'], '1.0'),
        T5='hydrogen' in r['T5_native']['text'].lower())
    readable = {k: r[f'{k}_patched']['eos'] for k in ('T1', 'T2', 'T3', 'T4', 'T5')}
    t = {}
    t['T1'] = sym_ok(r['T1_patched']); t['T2'] = sym_ok(r['T2_patched'])
    p3, p4, p5 = r['T3_patched']['text'], r['T4_patched']['text'], r['T5_patched']['text']
    t['T3'] = has(p3, '1', 'one') and not has(p3, '2', 'two') and 'helium' not in p3.lower()
    t['T4'] = '1.0' in p4 and '4.0' not in p4 and 'helium' not in p4.lower()
    t['T5'] = 'helium' not in p5.lower()
    t['R1'] = not any(x in r['R1_patched']['ids'] for x in HE)
    swap = any('helium' in r[f'{k}_patched']['text'].lower() for k in ('T1', 'T2', 'T3', 'T4', 'T5')) or has(p3, '2', 'two') or '4.0' in p4
    keep = t['T3'] and t['T4'] and t['T5'] and t['R1']
    cats = []
    if t['T1'] and t['T2'] and keep: cats.append('SELECTIVE STATIC EDIT ON THIS SET')
    if (t['T1'] != t['T2']) and keep: cats.append('PARTIAL')
    if swap: cats.append('ENTITY SWAP')
    if not t['T1'] and not t['T2'] and not swap: cats.append('NO EFFECT')
    return dict(native_correct=native_ok, readable=readable, tests=t, categories=cats)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--group', required=True, choices=['calib', 'T1', 'T2', 'T3', 'T4', 'T5', 'R1T6', 'read'])
    p.add_argument('--model-root', type=Path, default=Path('/mnt/models/hf-models'))
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.dry_run:
        return dry_run(a.root)
    if a.group == 'read':
        res = {}
        for g in ('T1', 'T2', 'T3', 'T4', 'T5', 'R1T6'):
            res.update(json.loads((a.root / g / 'summary.json').read_text())['results'])
        out = read_test(res); write(a.root / 'reading.json', out); print(json.dumps(out, indent=2)); return
    import torch
    from run_nf4_local_bench import setup, sha
    dest = a.root / a.group; dest.mkdir(parents=True, exist_ok=False)
    summary = dict(status='running', group=a.group, protocol=__doc__, runner_sha256=sha(Path(__file__)),
                   source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parent, text=True).strip())
    write(dest / 'summary.json', summary)
    torch.manual_seed(0); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    model, layers, _, eos, decode, _, _ = setup('E4B', a.model_root, False)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_root / 'gemma-4-E4B-it', local_files_only=True)
    prompt_for = lambda q: tok.apply_chat_template([{'role': 'user', 'content': q}], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    V = np.load(ASYM / 'residual.npy')
    r = Runner(dest, model, layers[21], eos, decode, prompt_for, torch, sha=sha)
    try:
        if a.group == 'calib':
            summary['calibration'] = calibration(r, V)
        else:
            cal = json.loads((a.root / 'calib' / 'summary.json').read_text())['calibration']
            assert cal['go'], 'calibration stop: no test services'
            g = cal['g_star']; vec = (np.float32(g) * V).astype(np.float32); results = {}
            if a.group == 'R1T6':
                draw = np.random.default_rng(RANDOM_SEED).standard_normal(V.shape)
                rnd = (draw / np.linalg.norm(draw) * np.linalg.norm(vec.astype(np.float64))).astype(np.float32)
                np.save(dest / 'random.npy', rnd)
                results['R1_patched'] = r.generate('R1_patched', prompt_for(QUESTIONS['T1']), rnd)
                p6 = prompt_for(QUESTIONS['T6'])
                _, n6 = r.forward('T6_native', p6, None); rec6, a6 = r.forward('T6_patched', p6, vec)
                results['T6'] = dict(inert=bool(np.array_equal(n6['logits'], a6['logits'])), patched_rows=rec6['patched_rows'])
            else:
                pq = prompt_for(QUESTIONS[a.group])
                results[f'{a.group}_native'] = r.generate(f'{a.group}_native', pq, None)
                results[f'{a.group}_patched'] = r.generate(f'{a.group}_patched', pq, vec)
            summary.update(g_star=g, results=results)
        summary.update(status='complete', calls=r.calls, forwards=r.count)
    except Exception as e:
        summary.update(status='failed', error=repr(e), calls=r.calls, forwards=r.count); write(dest / 'summary.json', summary); raise
    write(dest / 'summary.json', summary)


def dry_run(root):
    """Synthetic CPU model: checks hook support, gains, calibration logic, generation, budgets, readings. Never evidence."""
    import torch
    from types import SimpleNamespace
    class L(torch.nn.Module):
        def forward(self, h): return h * 1.0
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.layers = torch.nn.ModuleList(L() for _ in range(22)); g = torch.Generator().manual_seed(0)
            self.emb = torch.randn(300000, 8, generator=g) * 0.01; self.dir = torch.randn(2560, generator=g)
        def forward(self, input_ids, use_cache, logits_to_keep):
            h = torch.zeros(1, input_ids.shape[1], 2560)
            for layer in self.layers: h = layer(h)
            push = float(h[0].sum(0) @ self.dir) / 4000.0
            logits = torch.zeros(1, 1, 262144); n = input_ids.shape[1]
            seq = [236814, 236761, 106] if n % 3 else [886, 236761, 106]
            logits[0, -1, seq[n % 3]] = 10.0; logits[0, -1, 2209] = 9.0 + push; logits[0, -1, 1156] = 5.0
            return SimpleNamespace(logits=logits)
    import hashlib
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    root.mkdir(parents=True, exist_ok=False); model = M()
    V = np.asarray(model.dir.numpy(), dtype=np.float32); V = V / np.linalg.norm(V) * 63.9
    asym = root / 'asym'; asym.mkdir()
    calls = []
    for i, name in enumerate(('symbol_base', 'number_base')):
        ids = [2, 105, 2364, 107, 46762, 13076] + list(range(20 + i)) + [154624, 5, 6]
        r0 = Runner(asym, model, model.layers[21], {106}, str, None, torch, sha=sha)
        rec, _ = r0.forward(name, ids, None); calls.append(dict(rec, name=name))
    (asym / 'summary.json').write_text(json.dumps(dict(calls=calls)))
    r = Runner(root, model, model.layers[21], {106}, lambda ids: ' '.join(map(str, ids)), None, torch, sha=sha)
    cal = calibration(r, V, asym); assert r.count == 14
    (root / 'gen').mkdir(); r = Runner(root / 'gen', model, model.layers[21], {106}, lambda ids: ' '.join(map(str, ids)), None, torch, sha=sha)
    gen = r.generate('T1_patched', [2, 105, 13076, 7], (np.float32(2.0) * V).astype(np.float32))
    assert r.count <= 24 and all(c['patched_rows'] == [i for i, t in enumerate(c['input_ids']) if t in TRIGGER] for c in r.calls if c['gain_vector'])
    fake = lambda ids, text, eos=True: dict(ids=ids, text=text, eos=eos)
    res = dict(T1_native=fake([236814], 'H'), T2_native=fake([236814], 'H'), T3_native=fake([1], '1'), T4_native=fake([1], '1.008'),
               T5_native=fake([1], 'hydrogen and oxygen'), T1_patched=fake([2209], 'He'), T2_patched=fake([2209], 'He'),
               T3_patched=fake([1], '1'), T4_patched=fake([1], '1.008'), T5_patched=fake([1], 'hydrogen and oxygen'), R1_patched=fake([236814], 'H'))
    assert read_test(res)['categories'] == ['SELECTIVE STATIC EDIT ON THIS SET']
    res['T3_patched'] = fake([2], '2'); assert 'ENTITY SWAP' in read_test(res)['categories']
    print('SYNTHETIC dry run ok:', r.count, 'forwards; calibration go =', cal['go'], 'g* =', cal['g_star'], '; generation', gen['ids'])


if __name__ == '__main__':
    main()
