"""Phase 3b row-split diagnostic, frozen 2026-10-26 before any forward (Claude Code).

Why. Phase 3 (static L21 patch on every hydrogen token) stopped: at g=1 the number boundary answered
" three". Both boundaries hold two hydrogen tokens: row 5 (prompt, "Give hydrogen", entity BEFORE the
relation) and row 30 (answer "Hydrogen", which has already read the question). This splits the rows.

Conditions, same archived vector V (L21 residual, float16 norm 63.897), post-block L21, PLE native, on
the two archived resumed boundaries (symbol "...is $\\text{", number "...is"), 8 forwards:
  native (hook inactive; must equal archived logits bit for bit),
  row 5 only, g=1   (number: must equal the archived H3/asymmetry L21 residual logits bit for bit),
  row 30 only, g=1,
  rows 5+30, g=0.5.
Readings fixed now (number boundary, increment = (two - one) margin change vs native):
  MENTION: row-30-only increment >= 10 and row-5-only increment < 2 -> the relation-aware mention
           carries the number effect seen in Phase 3.
  STEP:    " two" is top-1 in row-30-only or in rows 5+30 at g=0.5 -> compatible with a single
           helium-ward step (an entity swap, not a selective edit).
  Both, one or neither may hold; all margins and top-1s are reported for both boundaries.
Post-hoc origin stated: motivated by the Phase 3 stop; one case per condition; no new gain search.
Budget 8 forwards, one service, shared flock, 22G/24G/8G, 1800 s. No weights or patches written.
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_nf4_phase3_static import ASYM, HE, H, ONE, TWO, write

CONDS = (('native', None, 0.0), ('row5', (5,), 1.0), ('row30', (30,), 1.0), ('both_half', (5, 30), 0.5))


def run(root, model, layer, torch, sha):
    ca = {c['name']: c for c in json.loads((ASYM / 'summary.json').read_text())['calls']}
    V = np.load(ASYM / 'residual.npy'); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    calls, results, n = [], {}, 0
    for q, base, ref, tgt, nat in (('symbol', 'symbol_base', 'symbol_residual', HE[0], H[0]), ('number', 'number_base', 'number_residual', TWO, ONE)):
        ids = ca[base]['input_ids']; assert ids[5] == 13076 and ids[30] == 154624
        with np.load(ASYM / ca[base]['artifact']) as z: archived = z['logits'].reshape(-1).copy()
        with np.load(ASYM / ca[ref]['artifact']) as z: archived_r = z['logits'].reshape(-1).copy()
        res = {}
        for name, rows, g in CONDS:
            vec = (np.float32(g) * V).astype(np.float32); saved = {}
            def post(module, args, output, rows=rows, vec=vec):
                if rows is None: return None
                d = torch.as_tensor(vec, device=output.device, dtype=output.dtype); p = output.clone()
                for r in rows: p[0, r] += d
                keep = [i for i in range(output.shape[1]) if i not in rows]; assert torch.equal(p[:, keep], output[:, keep])
                saved['before'] = output[0, list(rows)].float().cpu().numpy(); saved['after'] = p[0, list(rows)].float().cpu().numpy()
                saved['requested'] = d.float().cpu().numpy(); return p
            hd = layer.register_forward_hook(post)
            try:
                with torch.inference_mode():
                    logits = model(input_ids=torch.tensor([ids], device=dev), use_cache=False, logits_to_keep=1).logits[0, -1].float().cpu().numpy()
            finally: hd.remove()
            assert np.isfinite(logits).all()
            fn = f'{n:03}-{q}_{name}.npz'; np.savez_compressed(root / fn, logits=logits, **saved); n += 1
            calls.append(dict(name=f'{q}_{name}', rows=list(rows) if rows else [], gain=g, artifact=fn, artifact_sha256=sha(root / fn), argmax=int(logits.argmax())))
            res[name] = dict(argmax=int(logits.argmax()), margin=float(logits[tgt] - logits[nat]))
            if name == 'native': assert np.array_equal(logits, archived), f'{q} native replay differs'
            if name == 'row5': res[name]['bit_equal_archived_residual'] = bool(np.array_equal(logits, archived_r))
        for k in res: res[k]['increment'] = res[k]['margin'] - res['native']['margin']
        results[q] = res
    num = results['number']
    reading = dict(MENTION=bool(num['row30']['increment'] >= 10 and num['row5']['increment'] < 2),
                   STEP=bool(num['row30']['argmax'] == TWO or num['both_half']['argmax'] == TWO))
    return calls, results, reading


def main():
    root = Path(sys.argv[1]); dry = '--dry-run' in sys.argv
    import torch, hashlib
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    root.mkdir(parents=True, exist_ok=False)
    if dry:
        print('SYNTHETIC: protocol logic only, see Phase 3 dry run for hook mechanics; not evidence'); return
    from run_nf4_local_bench import setup
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.use_deterministic_algorithms(True)
    model, layers, *_ = setup('E4B', Path('/mnt/models/hf-models'), False)
    calls, results, reading = run(root, model, layers[21], torch, sha)
    write(root / 'summary.json', dict(status='complete', protocol=__doc__, runner_sha256=sha(Path(__file__)), calls=calls,
                                      forwards=len(calls), results=results, reading=reading))
    print(json.dumps(dict(results=results, reading=reading), indent=1))


if __name__ == '__main__':
    main()
