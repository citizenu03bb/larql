"""Phase B D1 diagnostic of the 15 B2 edits (Claude Code), under docs/phase-b-d1-diagnostic-protocol.md.

  python3 scripts/phase_b_d1_diagnostic.py plan [--wiki]                   # model-free: build and hash the plan
  python3 scripts/phase_b_d1_diagnostic.py rehearse [--wiki]               # model-free: plan + tokenizer checks
  python3 scripts/phase_b_d1_diagnostic.py fetch-wikitext --i-have-download-approval
  python3 scripts/phase_b_d1_diagnostic.py service K --i-have-approval     # one service, via run_d1_service.sh
  python3 scripts/phase_b_d1_diagnostic.py analyze                         # model-free, from the saved records

No generation and no new solve. M1 captures down_proj inputs (dose a = V^T k), M2 teacher-forces the native
answer prefix with the true and the new value (margin m = log P(new) - log P(true)), M3 applies the served
module to k*. Arms change only V: C (B1 covariance, the certified edits), I (identity), W (WikiText-2).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]           # tools/edit_cert
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

PROTOCOL = 'docs/phase-b-d1-diagnostic-protocol.md'
RESEARCH = Path.home() / 'Projects/larql-research'
SOLVES = RESEARCH / 'phase-b-rome-20261026/solves'
B2V2 = RESEARCH / 'phase-b-certification-v2-20261026'
ROOT = RESEARCH / 'phase-b-d1-20261027'
SPECS = HERE / 'facts/phase-b-dev-v2'
NEIGHBOURS = HERE / 'facts/phase-b-d1-neighbours.json'
ROME_SOLVER = Path.home() / 'Projects/larql.worktree/phase-b-rome/tools/rome_solver'
WIKI_URL = 'https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip'
WIKI_ZIP_SHA256 = 'ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11'
WIKI_DIR = RESEARCH / 'datasets/wikitext-2-raw-v1'
MODEL_ROOT = Path('/mnt/models/hf-models')
LAYERS = (4, 8, 12)
M1_PROMPTS = ('S1', 'E1', 'E2', 'E3', 'E4', 'O1', 'O2', 'O3', 'O4', 'W1')
M2_PROMPTS = ('S1', 'E1', 'E2', 'E3', 'E4', 'O1', 'O2', 'O3', 'O4')
UNSEEN = ('E1', 'E2', 'E3', 'E4')
NEIGH = ('O1', 'O2', 'O3')
FORWARD_CAP = 400
BEFORE, AFTER = 'subject_before_relation', 'subject_after_relation'


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_json(v) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()


def write(p: Path, v) -> None:
    p.write_text(json.dumps(v, indent=1, sort_keys=True) + '\n')


# ---------------------------------------------------------------- readings of the inputs (model-free)

def word_find(text: str, surface: str) -> int:
    m = re.search(r'(?<!\w)' + re.escape(surface) + r'(?!\w)', text)
    return m.start() if m else -1


def first_value(text: str, surfaces: list[str], after: int = 0):
    """Earliest word-bounded occurrence of any surface at or after `after`: (index in list, surface, char start)."""
    hits = [(word_find(text[after:], s), -len(s), i, s) for i, s in enumerate(surfaces)]
    hits = [(h[0] + after if h[0] >= 0 else -1, *h[1:]) for h in hits]
    hits = [h for h in hits if h[0] >= 0]
    if not hits:
        return None
    start, _, i, s = min(hits)
    return i, s, start


def pair_new(i: int, found: str, true: list[str], new: list[str]) -> str:
    """Same position when the two lists have the same length (Japan/Germany, Japanese/German), else the first
    new surface; the first letter's case follows the true value found."""
    n = new[i] if len(true) == len(new) else new[0]
    return (n[0].upper() if found[0].isupper() else n[0].lower()) + n[1:]


def probe_item(q: dict, spec: dict, answer: str, subjects: list[str]) -> dict:
    """M2 item: prefix of the native answer up to its first true value, with the true and new continuations."""
    if q['control'] == 'efficacy':
        true = spec['old']['accept']
    else:
        true = q['expect']['accept']
        if any(v in true for v in spec['new']['accept']):
            return dict(qid=q['id'], excluded='true values include the new value (cannot show bleed)')
    # the value is searched after the subject's own mention, so "Mount Kenya ... Kenya" reads the second one
    ends = [m + len(s) for s in subjects if (m := word_find(answer, s)) >= 0]
    hit = first_value(answer, true, after=min(ends) if ends else 0)
    if hit is None:
        return dict(qid=q['id'], excluded='native answer states no true value')
    i, found, start = hit
    return dict(qid=q['id'], prefix=answer[:start], true=found, new=pair_new(i, found, true, spec['new']['accept']))


def native_answers(fact: str) -> dict:
    run = B2V2 / f'{fact}-L4' / 'run'
    out = {}
    for f in sorted((run / 'services').glob('*/records.json')):
        r = json.loads(f.read_text())
        if r['phase'] == 'native':
            for g in r['generations']:
                if g['condition'] == 'native':
                    out[g['question_id']] = g['text']
    return out


def b2_readings(fact: str, layer: int) -> dict:
    v = json.loads((B2V2 / f'{fact}-L{layer}' / 'run' / 'verdict.json').read_text())
    st = {c['id']: c['status'] for c in v['controls']}
    c6 = next(c for c in v['controls'] if c['id'] == 'C6')
    bleed = {o['question_id']: o['why'].startswith('states the edited value') for o in c6['outcomes']
             if o['basis'] == 'generation'}
    return dict(statuses=st, bleed=bleed, verdict_sha256=sha(B2V2 / f'{fact}-L{layer}' / 'run' / 'verdict.json'))


# ---------------------------------------------------------------- the plan

def facts() -> list[str]:
    return sorted(p.stem for p in SPECS.glob('*.json'))


def build_plan(wiki: bool) -> dict:
    neigh = json.loads(NEIGHBOURS.read_text())['facts']
    arms = ['C', 'I'] + (['W'] if wiki else [])
    services = []
    if wiki:
        services.append(dict(kind='covariance', forwards=200, windows=200,
                             covariance_py_sha256=sha(ROME_SOLVER / 'rome_solver/covariance.py')))
    for fact in facts():
        spec = json.loads((SPECS / f'{fact}.json').read_text())
        qs = {q['id']: q for q in spec['questions']}
        answers = native_answers(fact)
        items = [probe_item(qs[q], spec, answers[q], [neigh[fact][q]] if q.startswith('O') else spec['subject'])
                 for q in M2_PROMPTS]
        live = [it for it in items if 'excluded' not in it]
        edits = []
        for L in LAYERS:
            edit = SOLVES / f'{fact}-L{L}' / 'edit'
            edits.append(dict(layer=L, edit=str(edit), U0_sha256=sha(edit / 'U0.npy'), V0_sha256=sha(edit / 'V0.npy'),
                              k_star_sha256=sha(edit / 'k_star.npy'), b2=b2_readings(fact, L)))
        conditions = ['native'] + [f'{a}:L{L}' for L in LAYERS for a in arms] + [f'random:L{L}' for L in LAYERS]
        forwards = len(M1_PROMPTS) + len(conditions) * len(live) * 2
        assert forwards <= FORWARD_CAP, forwards
        subjects = {q: (neigh[fact][q] if q.startswith('O') else None) for q in M1_PROMPTS}
        services.append(dict(kind='fact', fact_id=fact, spec_sha256=sha(SPECS / f'{fact}.json'), subject=spec['subject'],
                             prompts={q: qs[q]['text'] for q in M1_PROMPTS}, neighbour_subjects=subjects,
                             items=items, conditions=conditions, edits=edits, forwards=forwards,
                             random_seed=spec['random_seed']))
    plan = dict(protocol=PROTOCOL, author='Claude Code', arms=arms, layers=list(LAYERS), services=services,
                forwards_total=sum(s['forwards'] for s in services), model='nf4-e2b',
                inputs=dict(neighbours_sha256=sha(NEIGHBOURS), b2v2_root=str(B2V2), solves_root=str(SOLVES)))
    plan['plan_sha256'] = sha_json(plan)
    return plan


def load_plan(root: Path = ROOT) -> dict:
    plan = json.loads((root / 'plan.json').read_text())
    body = {k: v for k, v in plan.items() if k != 'plan_sha256'}
    assert sha_json(body) == plan['plan_sha256'], 'plan hash mismatch'
    return plan


# ---------------------------------------------------------------- arms and doses (model-free)

def arm_v(arm: str, edit_dir: Path, layer: int, root: Path = ROOT) -> np.ndarray:
    k = np.load(edit_dir / 'k_star.npy').astype(np.float64)
    if arm == 'C':
        return np.load(edit_dir / 'V0.npy').astype(np.float64)[:, 0]
    if arm == 'I':
        return k / (k @ k)
    if arm == 'W':
        M = np.load(root / 'covariance' / f'W{layer}.npy').astype(np.float64)
        q = np.linalg.solve(M, k)
        return q / (k @ q)
    raise ValueError(arm)


def subject_last_token(offsets: list[tuple[int, int]], text: str, surfaces: list[str]) -> int:
    """Index of the token holding the last character of the longest subject surface found in the text."""
    for s in sorted(surfaces, key=len, reverse=True):
        start = text.find(s)
        if start >= 0:
            end = start + len(s) - 1
            for t, (a, b) in enumerate(offsets):
                if a <= end < b:
                    return t
    raise ValueError(f'no subject surface of {surfaces} in prompt')


def auc(scores_pos: list[float], scores_neg: list[float]) -> float | None:
    if len(scores_pos) < 3 or len(scores_neg) < 3:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in scores_pos for n in scores_neg)
    return wins / (len(scores_pos) * len(scores_neg))


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    def rank(v):
        v = np.asarray(v, float)
        order = v.argsort(kind='stable')
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        for val in np.unique(v):            # average ranks on ties
            m = v == val
            r[m] = r[m].mean()
        return r
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def efficacy_class(dose: float, shift: float, flips: dict, c1_pass: bool) -> str:
    both = flips.get(AFTER, 0) >= 1 and flips.get(BEFORE, 0) >= 1
    if c1_pass and both:
        return 'P'
    if dose < 0.5:
        return 'K'
    if shift < 2:
        return 'V'
    if both and not c1_pass:
        return 'S'
    return 'U'


def arm_effect(index_arm: list[float], index_c: list[float], keeps: list[bool]) -> str:
    """Selectivity of an arm against arm C over the C1 PASS edits (protocol §5)."""
    better = sum(a <= 0.67 * c and k for a, c, k in zip(index_arm, index_c, keeps))
    worse = sum(a >= 1.5 * c for a, c in zip(index_arm, index_c))
    if better >= 3:
        return 'improves'
    if worse >= 3:
        return 'worse'
    return 'neutral'


# ---------------------------------------------------------------- services (GPU, under run_d1_service.sh)

def wiki_records(raw: str) -> list[tuple[str, str]]:
    """WikiText-2 train split into articles at top-level headings (' = Title = ')."""
    arts, cur = [], []
    for line in raw.split('\n'):
        if re.fullmatch(r' = [^=].* = ', line) and cur:
            arts.append(''.join(cur))
            cur = []
        cur.append(line + '\n')
    if cur:
        arts.append(''.join(cur))
    arts = [a for a in arts if a.strip()]
    return [(f'{i:05d}', a) for i, a in enumerate(arts)]


def service_covariance(svc: dict, out: Path, inst) -> dict:
    sys.path.insert(0, str(ROME_SOLVER))
    from rome_solver.covariance import collect_model_covariances, fixed_windows
    from rome_solver.solver import Budget
    assert sha(ROME_SOLVER / 'rome_solver/covariance.py') == svc['covariance_py_sha256'], 'covariance code changed'
    raw = (WIKI_DIR / 'wikitext-2-raw' / 'wiki.train.raw').read_text(encoding='utf-8')
    windows, meta = fixed_windows({'wiki': wiki_records(raw)}, inst.tok, quotas={'wiki': svc['windows']})
    budget = Budget(max_forwards=400)
    cov, info = collect_model_covariances(inst.model, inst.layers, windows, candidate_layers=LAYERS,
                                          label='E2B', budget=budget)
    d = out.parent.parent / 'covariance'
    d.mkdir(exist_ok=True)
    hashes = {}
    for L, M in cov.items():
        np.save(d / f'W{L}.npy', M)
        hashes[str(L)] = sha(d / f'W{L}.npy')
    np.save(d / 'windows.npy', windows)
    text = inst.tok.decode(windows.reshape(-1).tolist())
    names = sorted({s for f in facts() for s in json.loads((SPECS / f'{f}.json').read_text())['subject']} |
                   {s for v in json.loads(NEIGHBOURS.read_text())['facts'].values() for s in v.values()})
    mentions = {n: text.count(n) for n in names if text.count(n)}
    return dict(forwards=info['forwards'], covariance_sha256=hashes, windows_meta=meta, layers=info['layers'],
                wiki_train_sha256=sha(WIKI_DIR / 'wikitext-2-raw' / 'wiki.train.raw'), subject_mentions=mentions)


def service_fact(svc: dict, out: Path, inst, plan: dict) -> dict:
    import torch
    from edit_cert import backends
    from edit_cert.backends.lowrank import LowRank
    tok, model = inst.tok, inst.model
    device = next(inst.layers.parameters()).device
    count = dict(forwards=0)

    def fwd(ids, keep):
        count['forwards'] += 1
        assert count['forwards'] <= svc['forwards'], 'forward budget exceeded'
        with torch.inference_mode():
            o = model(input_ids=torch.tensor([ids], dtype=torch.long, device=device), use_cache=False,
                      logits_to_keep=keep)
        lg = o.logits[0].float()
        assert torch.isfinite(lg).all(), 'nonfinite logits'
        return lg

    # M1: one native forward per prompt, down_proj inputs at the three layers
    keys, positions, dtypes = {}, {}, {}
    mods = {L: model.get_submodule(f'model.language_model.layers.{L}.mlp.down_proj') for L in LAYERS}
    cap = {}
    hs = [m.register_forward_pre_hook(lambda mod, args, L=L: cap.__setitem__(L, args[0][0].detach())) for L, m in mods.items()]
    try:
        for q in M1_PROMPTS:
            text = svc['prompts'][q]
            ids = inst.chat_prompt(text)
            rendered = tok.apply_chat_template([{'role': 'user', 'content': text}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
            enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
            assert enc['input_ids'] == ids, f'{q}: rendered text does not re-tokenize to the prompt'
            subj = svc['neighbour_subjects'][q] if q.startswith('O') else None
            surfaces = [subj] if subj else svc['subject']
            positions[q] = None if q == 'W1' else subject_last_token(enc['offset_mapping'], rendered, surfaces)
            cap.clear()
            fwd(ids, 1)
            for L in LAYERS:
                keys[f'{q}:L{L}'] = cap[L].float().cpu().numpy()
                dtypes[L] = str(cap[L].dtype)
    finally:
        for h in hs:
            h.remove()
    np.savez(out / 'keys.npz', **keys)

    # M2: teacher-forced margins under every condition
    seqs = {}
    for it in svc['items']:
        if 'excluded' in it:
            continue
        p = inst.chat_prompt(svc['prompts'][it['qid']])
        a = p + tok.encode(it['prefix'] + it['true'], add_special_tokens=False)
        b = p + tok.encode(it['prefix'] + it['new'], add_special_tokens=False)
        d = next(i for i in range(min(len(a), len(b))) if a[i] != b[i])
        assert d >= len(p), f"{it['qid']}: continuations diverge inside the prompt"
        seqs[it['qid']] = dict(true=a, new=b, start=d)

    def score(ids, start):
        lg = fwd(ids, len(ids) - start + 1)[:-1]
        lp = torch.log_softmax(lg, -1)
        tgt = torch.tensor(ids[start:], device=lp.device)
        return lp[torch.arange(len(tgt)), tgt].cpu().tolist()

    edits = {e['layer']: e for e in svc['edits']}
    loaded = {L: backends.load(Path(edits[L]['edit'])) for L in LAYERS}
    margins = {}
    for cond in svc['conditions']:
        if cond == 'native':
            ctx = None
        else:
            arm, L = cond.split(':L')
            L = int(L)
            base = loaded[L]
            if arm == 'random':
                ctx = base.random_counterpart(svc['random_seed'])
            elif arm == 'C':
                ctx = base
            else:
                V = arm_v(arm, Path(edits[L]['edit']), L, out.parent.parent).astype(np.float32)[:, None]
                ctx = LowRank(base.targets, {'U0': base.arrays()['U0'], 'V0': V}, label=f'{arm}({base.label})')
        res = {}
        if ctx is None:
            for q, s in seqs.items():
                res[q] = dict(true=score(s['true'], s['start']), new=score(s['new'], s['start']))
        else:
            with ctx.apply(inst):
                for q, s in seqs.items():
                    res[q] = dict(true=score(s['true'], s['start']), new=score(s['new'], s['start']))
        margins[cond] = res

    # M3: the served module applied to k*, without and with each arm's adapter
    m3 = {}
    for L in LAYERS:
        e = Path(edits[L]['edit'])
        k = torch.tensor(np.load(e / 'k_star.npy'), device=device).to(getattr(torch, dtypes[L].split('.')[-1]))
        wk, vs = np.load(e / 'w_k_star.npy').astype(np.float64), np.load(e / 'v_star.npy').astype(np.float64)
        U = loaded[L].arrays()['U0'].astype(np.float64)[:, 0]
        with torch.inference_mode():
            y0 = mods[L](k[None, None])[0, 0].float().cpu().numpy().astype(np.float64)
        m3[f'L{L}'] = dict(native_rel=float(np.linalg.norm(y0 - wk) / np.linalg.norm(wk)))
        for arm in plan['arms']:
            V = arm_v(arm, e, L, out.parent.parent)
            ad = LowRank(loaded[L].targets, {'U0': U[:, None].astype(np.float32), 'V0': V.astype(np.float32)[:, None]})
            with ad.apply(inst), torch.inference_mode():
                y1 = mods[L](k[None, None])[0, 0].float().cpu().numpy().astype(np.float64)
            target = wk + U * (V @ np.load(e / 'k_star.npy').astype(np.float64))
            m3[f'L{L}'][arm] = dict(rel=float(np.linalg.norm(y1 - target) / np.linalg.norm(vs - wk)))
    return dict(forwards=count['forwards'], positions=positions, key_dtypes={str(k): v for k, v in dtypes.items()},
                sequences={q: dict(start=s['start'], true_ids=s['true'][s['start']:], new_ids=s['new'][s['start']:])
                           for q, s in seqs.items()},
                margins=margins, m3=m3, keys_sha256=sha(out / 'keys.npz'))


def run_service(k: int, root: Path = ROOT) -> int:
    import torch
    from edit_cert.instruments.nf4 import Nf4Instrument
    plan = load_plan(root)
    svc = plan['services'][k]
    out = root / 'services' / str(k)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    inst = Nf4Instrument.load('E2B', MODEL_ROOT)
    rec = service_covariance(svc, out, inst) if svc['kind'] == 'covariance' else service_fact(svc, out, inst, plan)
    rec.update(service=k, kind=svc['kind'], plan_sha256=plan['plan_sha256'], seconds=round(time.time() - t0, 1),
               fingerprint=inst.fingerprint(), torch=torch.__version__)
    write(out / 'records.json', rec)
    write(out / 'status.json', dict(service=k, ok=True, forwards=rec['forwards'], records_sha256=sha(out / 'records.json')))
    return 0


# ---------------------------------------------------------------- analysis (model-free)

def margin(cond_res: dict, q: str) -> float:
    return float(sum(cond_res[q]['new']) - sum(cond_res[q]['true']))


def analyze_fact(svc: dict, rec: dict, keys: dict, arms: list[str], root: Path) -> list[dict]:
    orders = {q: ('subject_before_relation' if q in ('E3', 'E4') else 'subject_after_relation') for q in UNSEEN}
    nat = rec['margins']['native']
    rows = []
    for e in svc['edits']:
        L, edir = e['layer'], Path(e['edit'])
        c1_pass = e['b2']['statuses']['C1'] == 'PASS'
        row = dict(fact_id=svc['fact_id'], layer=L, c1_b2=e['b2']['statuses']['C1'], c6_b2=e['b2']['statuses']['C6'],
                   arms={})
        for arm in arms:
            V = arm_v(arm, edir, L, root)
            dose = {q: float(keys[f'{q}:L{L}'][rec['positions'][q]] @ V) for q in M1_PROMPTS if q != 'W1'}
            mx = {q: float((keys[f'{q}:L{L}'] @ V).max()) for q in M1_PROMPTS}
            res = rec['margins'][f'{arm}:L{L}']
            eff = [q for q in UNSEEN if q in res]
            m = {q: margin(res, q) for q in res}
            shift = {q: m[q] - margin(nat, q) for q in res}
            flips = {}
            for q in eff:
                if m[q] > 0:
                    flips[orders[q]] = flips.get(orders[q], 0) + 1
            neigh = [q for q in NEIGH if q in res]
            sel = (float(np.median([dose[q] for q in neigh])) / float(np.median([dose[q] for q in UNSEEN]))
                   if neigh else None)
            a = dict(dose=dose, max_dose=mx, margin=m, shift=shift, flips=flips,
                     dose_E=float(np.median([dose[q] for q in UNSEEN])),
                     shift_E=float(np.median([shift[q] for q in eff])) if eff else None,
                     selectivity_index=sel)
            a['class'] = efficacy_class(a['dose_E'], a['shift_E'] if eff else -np.inf, flips, c1_pass)
            row['arms'][arm] = a
        rnd = rec['margins'][f'random:L{L}']
        row['random_abs_shift_E'] = float(np.median([abs(margin(rnd, q) - margin(nat, q)) for q in UNSEEN if q in rnd]))
        row['bleed_b2'] = {q: e['b2']['bleed'].get(q) for q in NEIGH if q in rec['margins']['native']}
        row['m3'] = rec['m3'][f'L{L}']
        rows.append(row)
    return rows


def analyze(root: Path = ROOT) -> dict:
    plan = load_plan(root)
    arms = plan['arms']
    rows = []
    for k, svc in enumerate(plan['services']):
        if svc['kind'] != 'fact':
            continue
        rec = json.loads((root / 'services' / str(k) / 'records.json').read_text())
        keys = dict(np.load(root / 'services' / str(k) / 'keys.npz'))
        rows += analyze_fact(svc, rec, keys, arms, root)
    passers = [r for r in rows if r['c1_b2'] == 'PASS']
    valid = all(sum(r['arms']['C']['margin'][q] > 0 for q in UNSEEN if q in r['arms']['C']['margin']) >= 3
                for r in passers)
    noise = float(np.median([r['random_abs_shift_E'] for r in rows]))
    pos = [r['arms']['C']['dose'][q] for r in rows for q, b in r['bleed_b2'].items() if b]
    neg = [r['arms']['C']['dose'][q] for r in rows for q, b in r['bleed_b2'].items() if b is False]
    a = auc(pos, neg)
    sx = [r['arms']['C']['dose'][q] for r in rows for q in r['bleed_b2']]
    sy = [r['arms']['C']['shift'][q] for r in rows for q in r['bleed_b2']]
    m3_ok = all(r['m3']['native_rel'] <= 0.02 and all(r['m3'][arm]['rel'] <= 0.05 for arm in arms) for r in rows)
    effects = {}
    for arm in arms:
        if arm == 'C':
            continue
        keeps = [r['arms'][arm]['flips'].get(AFTER, 0) >= 1 and r['arms'][arm]['flips'].get(BEFORE, 0) >= 1 for r in passers]
        effects[arm] = arm_effect([r['arms'][arm]['selectivity_index'] for r in passers],
                                  [r['arms']['C']['selectivity_index'] for r in passers], keeps)
    out = dict(author='Claude Code', plan_sha256=plan['plan_sha256'], probe_valid=valid, random_noise_E=noise,
               random_noise_ok=noise < 1.0, bleed_auc_C=a, bleed_pairs=dict(bleed=len(pos), kept=len(neg)),
               bleed_reading=None if a is None else ('predicts' if a >= 0.8 else 'does not predict' if a <= 0.6
                                                     else 'inconclusive'),
               dose_shift_spearman_C=spearman(sx, sy), m3_ok=m3_ok, arm_effects=effects,
               classes_C={f"{r['fact_id']}-L{r['layer']}": r['arms']['C']['class'] for r in rows}, rows=rows)
    write(root / 'analysis.json', out)
    return out


# ---------------------------------------------------------------- entry points

def fetch_wikitext() -> int:
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    z = WIKI_DIR / 'wikitext-2-raw-v1.zip'
    subprocess.run(['curl', '-sSfL', WIKI_URL, '-o', str(z)], check=True)
    if sha(z) != WIKI_ZIP_SHA256:
        z.unlink()
        sys.exit('refused: wikitext-2 archive hash differs; file removed')
    subprocess.run(['unzip', '-q', '-o', str(z), '-d', str(WIKI_DIR)], check=True)
    print(json.dumps(dict(ok=True, zip_sha256=WIKI_ZIP_SHA256, train_sha256=sha(WIKI_DIR / 'wikitext-2-raw/wiki.train.raw'))))
    return 0


def rehearse(wiki: bool) -> int:
    """Model-free: plan, inputs, tokenizer-level checks of subject positions and continuations."""
    from transformers import AutoTokenizer
    plan = build_plan(wiki)
    tok = AutoTokenizer.from_pretrained(MODEL_ROOT / 'gemma-4-E2B-it', local_files_only=True)
    report = dict(forwards_total=plan['forwards_total'], services=[], plan_sha256=plan['plan_sha256'])
    for svc in plan['services']:
        if svc['kind'] != 'fact':
            report['services'].append(dict(kind='covariance', forwards=svc['forwards']))
            continue
        pos = {}
        for q in M1_PROMPTS:
            text = svc['prompts'][q]
            ids = tok.apply_chat_template([{'role': 'user', 'content': text}], tokenize=True, add_generation_prompt=True,
                                          enable_thinking=False, return_dict=False)
            rendered = tok.apply_chat_template([{'role': 'user', 'content': text}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
            enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
            assert enc['input_ids'] == ids, (svc['fact_id'], q)
            if q != 'W1':
                subj = svc['neighbour_subjects'][q] if q.startswith('O') else None
                t = subject_last_token(enc['offset_mapping'], rendered, [subj] if subj else svc['subject'])
                pos[q] = tok.decode([ids[t]])
        cont = {}
        for it in svc['items']:
            if 'excluded' in it:
                cont[it['qid']] = it['excluded']
                continue
            a = tok.encode(it['prefix'] + it['true'], add_special_tokens=False)
            b = tok.encode(it['prefix'] + it['new'], add_special_tokens=False)
            d = next(i for i in range(min(len(a), len(b))) if a[i] != b[i])
            cont[it['qid']] = f"{it['prefix']!r} | {tok.decode(a[d:])!r} vs {tok.decode(b[d:])!r}"
        report['services'].append(dict(fact=svc['fact_id'], forwards=svc['forwards'], subject_token=pos, continuations=cont))
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['plan', 'rehearse', 'fetch-wikitext', 'service', 'analyze'])
    ap.add_argument('k', nargs='?', type=int)
    ap.add_argument('--wiki', action='store_true')
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--i-have-approval', action='store_true')
    ap.add_argument('--i-have-download-approval', action='store_true')
    a = ap.parse_args(argv)
    if a.cmd == 'plan':
        a.root.mkdir(parents=True, exist_ok=True)
        assert not (a.root / 'plan.json').exists(), 'plan already written'
        plan = build_plan(a.wiki)
        write(a.root / 'plan.json', plan)
        print(json.dumps(dict(plan_sha256=plan['plan_sha256'], forwards_total=plan['forwards_total'],
                              services=[s['forwards'] for s in plan['services']])))
        return 0
    if a.cmd == 'rehearse':
        return rehearse(a.wiki)
    if a.cmd == 'fetch-wikitext':
        if not a.i_have_download_approval:
            sys.exit('refused: no --i-have-download-approval')
        return fetch_wikitext()
    if a.cmd == 'service':
        if not a.i_have_approval or a.k is None:
            sys.exit('refused: service needs K and --i-have-approval')
        return run_service(a.k, a.root)
    if a.cmd == 'analyze':
        r = analyze(a.root)
        print(json.dumps({k: v for k, v in r.items() if k != 'rows'}, indent=1))
        return 0
    return 2


if __name__ == '__main__':
    sys.exit(main())
