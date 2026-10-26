"""Frozen NF4 bench protocol, 2026-10-26 (before any pretrained-model forward).

Purpose: qualify a faster local instrument for an honest employment showcase and
later persistent-patch research. This is NOT a knowledge-edit experiment.

Scope: E2B then E4B, sequential separate services, <=24 model forwards and 1800 s
each (<=48 total); 22G/24G/8G RAM/swap limits. No weights written, no training,
no downloads, no gain search, no H1/H2 test. Stop on OOM, nonfinite arrays,
runtime/budget exhaustion or implementation failure; report incomplete/negative.
Use the existing NF4 double-quant/fp16 setup, CPU PLE lookup, eager attention,
eval/inference mode, deterministic settings and use_cache=False. Timings include
the complete prefix and head for the last token, not cached decode or loading.

Native question: Give hydrogen's chemical symbol in one short sentence, with
the element's name before the symbol. Use the official local chat template,
thinking disabled, empty assistant answer. One warmup plus two same-input timed
forwards; reuse their first argmax, then generate freely up to EOS or 12 (E2B)
/16 (E4B) tokens in total. A cap is incomplete, not a factual result. No tuning
if NF4 answers differently from Q4K.

E2B boundary test: same input ids for EVERY condition: chat prompt plus its FIRST
freely generated token. At position len(prompt)-1, perturb the output of one
decoder block (after FFN AND PLE). This is a two-position, teacher-forced
diagnostic, NOT a free continuation of an edited answer. Seven full forwards:
baseline, repeat, zero L12, zero L14, random L12, same random L13, same random
L14 (zero-based layers). Gaussian seed 20261026; fixed norm 0.25 * the native
L14 row norm, same vector/norm at all three sites, no amplitude selection.
The native row is captured by baseline, no extra forward. Save last-two-position
full logits, shared K/V, row before/after and the actual fp16 delta.

Prediction: L15..34 reuse K/V produced by sliding L13/full L14. Post-L14
perturbation changes current logits but leaves next-position logits AND shared
K/V exactly identical. L12 and L13 positive controls should affect next-position
logits/KV. Native repeats and zeros must be bit-identical. If these fail, the
instrument/prediction is not qualified; never silently replace exact equality
by a tolerance. Native success plus those checks qualifies only this short
NF4 example, not all sequences, LARQL bit identity, selectivity or persistence.

Dry-run uses an explicitly synthetic CPU backend, no pretrained weights, and
checks orchestration, hooks, exact application, cap refusal, EOS and truncation.
The old EXP stub is specific to LARQL's subprocess protocol; it cannot validate
torch hooks. Synthetic artifacts are labeled and must never become evidence.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

QUESTION = "Give hydrogen's chemical symbol in one short sentence, with the element's name before the symbol."
BUDGET = 24
SEED = 20261026


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array(x):
    return x.detach().float().cpu().numpy().copy()


class Limit:
    def __init__(self, cap=BUDGET, seconds=1740):
        self.cap, self.count = cap, 0
        self.deadline = time.monotonic() + seconds

    def take(self):
        if self.count >= self.cap:
            raise RuntimeError('forward cap exhausted')
        if time.monotonic() >= self.deadline:
            raise TimeoutError('bench deadline exhausted')
        self.count += 1


def generate_native(forward, prompt, first, stops, cap):
    generated, names = [first], ['warmup']
    while generated[-1] not in stops and len(generated) < cap:
        rec, _ = forward(f'native{len(generated):02}', prompt + generated)
        generated.append(rec['argmax'])
        names.append(rec['name'])
    return generated, names


@contextlib.contextmanager
def instrumentation(layers, position, capture_layers=(), patch_layer=None, delta=None):
    captures, handles = {}, []
    for index in set(capture_layers) | ({patch_layer} if patch_layer is not None else set()):
        def hook(module, inputs, output, index=index):
            if not isinstance(output, torch.Tensor):
                raise TypeError('decoder output must be a tensor')
            assert f'layer{index}_before' not in captures, 'hook fired twice in a forward'
            captures[f'layer{index}_before'] = array(output[0, position])
            if index == patch_layer:
                patched = output.clone()
                addition = torch.as_tensor(delta, dtype=output.dtype, device=output.device)
                patched[0, position] += addition
                assert torch.equal(patched[:, :position], output[:, :position])
                assert torch.equal(patched[:, position + 1:], output[:, position + 1:])
                captures['requested_delta'] = array(addition)
                captures[f'layer{index}_after'] = array(patched[0, position])
                return patched
        handles.append(layers[index].register_forward_hook(hook))
    try:
        yield captures
    finally:
        for handle in handles:
            handle.remove()


class SyntheticLayer(torch.nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, hidden, shared):
        if self.index <= 14:
            context = hidden.cumsum(1) / torch.arange(1, hidden.shape[1] + 1).reshape(1, -1, 1)
            hidden = hidden + 0.01 * context
            if self.index in (13, 14):
                shared['sliding_attention' if self.index == 13 else 'full_attention'] = (hidden.clone(), hidden.clone())
        else:
            hidden = hidden + 0.001 * shared['full_attention'][0]
        return hidden


class SyntheticModel:
    def __init__(self, prompt_len):
        self.layers = torch.nn.ModuleList(SyntheticLayer(i) for i in range(35))
        self.prompt_len = prompt_len

    def __call__(self, input_ids, logits_to_keep, **kwargs):
        h = torch.nn.functional.one_hot(input_ids % 8, 8).float()
        shared = {}
        for layer in self.layers:
            h = layer(h, shared)
        logits = h[:, -logits_to_keep:].clone()
        token = 3 if input_ids.shape[1] == self.prompt_len else 4 if input_ids.shape[1] == self.prompt_len + 1 else 1
        logits[:, -1, token] += 10
        return SimpleNamespace(logits=logits, shared_kv_states=shared)


class CpuLookup(torch.nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, ids):
        return self.embedding(ids.cpu()).to(ids.device, torch.float16)


def setup(label, model_root, synthetic):
    if synthetic:
        ids = [2, 5, 6, 7]
        model = SyntheticModel(len(ids))
        return model, model.layers, ids, {1}, lambda xs: ' '.join(map(str, xs)), {'synthetic': True}, 'cpu'
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from accelerate.hooks import remove_hook_from_module
    path = model_root / f'gemma-4-{label}-it'
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    ids = tok.apply_chat_template([{'role': 'user', 'content': QUESTION}], tokenize=True,
                                 add_generation_prompt=True, enable_thinking=False, return_dict=False)
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    cpu = ['model.language_model.embed_tokens_per_layer', 'model.vision_tower',
           'model.audio_tower', 'model.embed_vision', 'model.embed_audio']
    gpu = ['model.language_model.' + suffix for suffix in [
        'embed_tokens', 'layers', 'norm', 'rotary_emb', 'per_layer_model_projection',
        'per_layer_projection_norm']] + ['lm_head']
    dmap = {**dict.fromkeys(cpu, 'cpu'), **dict.fromkeys(gpu, 0)}
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
                            llm_int8_enable_fp32_cpu_offload=True)
    torch.cuda.synchronize()
    start = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb, device_map=dmap,
        dtype=torch.float16, low_cpu_mem_usage=True, local_files_only=True, attn_implementation='eager')
    lm = model.model.language_model
    ple = lm.embed_tokens_per_layer
    remove_hook_from_module(ple, recurse=True)
    ple.to('cpu', torch.float16).requires_grad_(False)
    lm.embed_tokens_per_layer = CpuLookup(ple)
    model.eval().requires_grad_(False)
    config = model.config.get_text_config()
    stores = {str(i): layer.self_attn.layer_type for i, layer in enumerate(lm.layers)
              if layer.self_attn.store_full_length_kv}
    shared = [i for i, layer in enumerate(lm.layers) if layer.self_attn.is_kv_shared_layer]
    if label == 'E2B':
        assert stores == {'13': 'sliding_attention', '14': 'full_attention'}, stores
        assert shared == list(range(15, 35)), shared
    info = dict(synthetic=False, load_seconds=time.monotonic()-start, config=config.to_dict(),
                kv_store_layers=stores, kv_shared_layers=shared, device_map=dmap,
                quantization=bnb.to_dict(), prompt_text=tok.decode(ids),
                metadata_sha256={f: sha(path / f) for f in ['config.json', 'model.safetensors.index.json',
                    'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'] if (path / f).exists()})
    index = path / 'model.safetensors.index.json'
    if index.exists():
        names = sorted(set(json.loads(index.read_text())['weight_map'].values()))
        info['weight_files_stat_only_not_hashed'] = [dict(name=f, size=(path/f).stat().st_size,
                                                        mtime_ns=(path/f).stat().st_mtime_ns) for f in names]
    import inspect
    from transformers.models.gemma4 import modeling_gemma4
    info['transformers_modeling_source_sha256'] = sha(Path(inspect.getfile(modeling_gemma4)))
    print(json.dumps({'loaded': label, 'load_seconds': info['load_seconds']}), flush=True)
    stops = {tok.eos_token_id, tok.convert_tokens_to_ids('<turn|>')}
    assert None not in stops
    return model, lm.layers, ids, stops, lambda xs: tok.decode(xs), info, 'cuda'


def run(root, label, model_root, synthetic=False):
    root.mkdir(parents=True, exist_ok=False)
    limit = Limit()
    summary = dict(status='running', synthetic=synthetic, label=label, forwards=0, calls=[],
                   protocol=__doc__, source_sha256=sha(Path(__file__)))
    write(root / 'summary.json', summary)
    try:
        if not synthetic:
            assert torch.cuda.is_available()
            torch.cuda.reset_peak_memory_stats()
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.use_deterministic_algorithms(True)
        torch.manual_seed(SEED)
        model, layers, prompt, stops, decode, info, device = setup(label, model_root, synthetic)
        try:
            info['commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parent,
                                                    text=True).strip()
        except subprocess.CalledProcessError:
            info['commit'] = None
        info['versions'] = {n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'bitsandbytes', 'numpy', 'accelerate']}
        info['prompt_ids'], info['eos_ids'] = prompt, sorted(stops)
        write(root / 'session.json', info)

        def forward(name, ids, keep=1, capture=(), patch_layer=None, delta=None, position=None):
            limit.take()
            ordinal = limit.count - 1
            pos = len(prompt) - 1 if position is None else position
            tensor = torch.tensor([ids], dtype=torch.long, device=device)
            if not synthetic:
                torch.cuda.synchronize()
            started = time.monotonic()
            with torch.inference_mode(), instrumentation(layers, pos, capture, patch_layer, delta) as arrays:
                result = model(input_ids=tensor, use_cache=False, logits_to_keep=keep,
                               return_shared_kv_states=bool(capture or patch_layer is not None))
                if not synthetic:
                    torch.cuda.synchronize()
                seconds = time.monotonic() - started
                arrays['logits'] = array(result.logits[0])
                for typ, pair in (result.shared_kv_states or {}).items():
                    for kind, value in zip(('k', 'v'), pair):
                        arrays[typ + '_' + kind] = array(value)
            assert all(np.isfinite(a).all() for a in arrays.values())
            filename = f'{ordinal:03}-{name}.npz'
            np.savez_compressed(root / filename, **arrays)
            record = dict(ordinal=ordinal, name=name, input_ids=list(ids), seconds=seconds,
                          artifact=filename, artifact_sha256=sha(root / filename),
                          argmax=int(arrays['logits'][-1].argmax()), patch_layer=patch_layer,
                          position=pos if patch_layer is not None else None)
            summary['calls'].append(record)
            summary['forwards'] = limit.count
            write(root / 'summary.json', summary)
            print(json.dumps(record), flush=True)
            return record, arrays

        warm, _ = forward('warmup', prompt)
        time0, _ = forward('timed0', prompt)
        time1, _ = forward('timed1', prompt)
        cap = 12 if label == 'E2B' else 16
        generated, names = generate_native(forward, prompt, warm['argmax'], stops, cap)
        summary['native'] = dict(ids=generated, text=decode(generated), eos=generated[-1] in stops,
                                 call_names=names, max_tokens=cap)
        summary['timing'] = dict(prefix_length=len(prompt), warmup=warm['seconds'],
                                 seconds=[time0['seconds'], time1['seconds']],
                                 mean_seconds=(time0['seconds']+time1['seconds'])/2,
                                 cache=False, logits_positions=1)
        if label == 'E2B':
            ids = prompt + generated[:1]
            _, base = forward('boundary_base', ids, keep=2, capture=(12, 13, 14))
            v = np.random.default_rng(SEED).standard_normal(base['layer14_before'].shape)
            target_norm = 0.25 * np.linalg.norm(base['layer14_before'].astype(np.float64))
            delta = (v / np.linalg.norm(v) * target_norm).astype(np.float32)
            np.save(root / 'direction.npy', delta)
            summary['boundary'] = dict(seed=SEED, target_norm=float(target_norm), position=len(prompt)-1,
                                       indices_zero_based=True, teacher_forced=True)
            forward('boundary_repeat', ids, keep=2, capture=(12, 13, 14))
            for site in (12, 14):
                forward(f'boundary_zero{site}', ids, keep=2, patch_layer=site, delta=np.zeros_like(delta))
            for site in (12, 13, 14):
                forward(f'boundary_patch{site}', ids, keep=2, patch_layer=site, delta=delta)
        summary['status'] = 'complete' if summary['native']['eos'] else 'incomplete_native_cap'
        if not synthetic:
            summary['gpu'] = dict(name=torch.cuda.get_device_name(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as exc:
        summary.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc(),
                       forwards=limit.count)
        raise
    finally:
        if not synthetic:
            cgroup = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split('::', 1)[1].lstrip('/')
            summary['cgroup'] = {name: (cgroup / name).read_text().strip() for name in [
                'memory.current', 'memory.peak', 'memory.swap.current', 'memory.swap.peak',
                'memory.high', 'memory.max', 'memory.swap.max'] if (cgroup / name).exists()}
        write(root / 'summary.json', summary)


def dry_run(root):
    run(root, 'E2B', Path('.'), synthetic=True)
    cap = Limit(cap=2)
    cap.take(); cap.take()
    try:
        cap.take()
    except RuntimeError:
        assert cap.count == 2
    else:
        raise AssertionError('cap did not refuse third call')
    base = next(root.glob('*-boundary_base.npz'))
    with np.load(base) as z:
        assert z['logits'].shape == (2, 8)
    expired = Limit(seconds=-1)
    try:
        expired.take()
    except TimeoutError:
        assert expired.count == 0
    else:
        raise AssertionError('expired deadline did not refuse call')
    calls = []
    def never_end(name, ids):
        calls.append(ids)
        return {'argmax': 3, 'name': name}, {}
    truncated, _ = generate_native(never_end, [2], 3, {1}, 3)
    assert truncated == [3, 3, 3] and calls == [[2, 3], [2, 3, 3]]
    calls.clear()
    ended, _ = generate_native(never_end, [2], 1, {1}, 3)
    assert ended == [1] and calls == []
    print('SYNTHETIC DRY RUN COMPLETE; cap refusal checked', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--model', choices=['E2B', 'E4B'], default='E2B')
    parser.add_argument('--model-root', type=Path, default=Path('/mnt/models/hf-models'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        dry_run(args.destination)
    else:
        run(args.destination, args.model, args.model_root)
