"""Model-free tests of Claude's A3 solve (published ROME code), on a tiny random GPT-2 and the real tokenizer."""
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')

from p1_rome_claude import solve as v1, solve_ref as ref  # noqa: E402
from p1_rome_claude.run import REQUESTS  # noqa: E402

TOK_DIR = '/mnt/models/hf-models/gpt2-xl'
TEMPLATES = Path.home() / 'Projects/larql-research/phase-b-p1-v2-20261027/templates/templates.json'
HP = dict(v1.PUBLISHED, layer=1)


@pytest.fixture(scope='module')
def tiny():
    import os
    if not os.path.exists(f'{TOK_DIR}/tokenizer.json'):
        pytest.skip('GPT-2 tokenizer not on disk')
    tok = transformers.AutoTokenizer.from_pretrained(TOK_DIR, local_files_only=True)
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(n_layer=3, n_embd=32, n_head=4, n_inner=64, vocab_size=len(tok), n_positions=128)
    return transformers.GPT2LMHeadModel(cfg).eval(), tok


def test_lookup_positions_match_joint_tokenization_on_every_row(tiny):
    _, tok = tiny
    if not TEMPLATES.exists():
        pytest.skip('shared templates not on disk')
    templates = json.loads(TEMPLATES.read_text())['templates']
    for prompt, subject, target in REQUESTS.values():
        rows = [t.format(prompt) + tok.decode(tok.encode(target)[:-1]) for t in templates] + ref.KL_PROMPTS
        for row in rows:
            text = row.format(subject)
            assert ref.lookup_index(tok, row, subject) == v1.subject_last(tok, text[:text.index(subject) + len(subject)], subject)


def test_delta_is_added_on_every_row_including_the_kl_row(tiny):
    model, tok = tiny
    mlp = model.transformer.h[1].mlp
    ids, mask, _ = ref.batch(tok, ['BMW is a company', 'Hello. BMW is', 'BMW is a'], 'cpu')
    pos = [0, 3, 0]
    seen = {}
    h = mlp.register_forward_hook(lambda m, a, o: seen.__setitem__('native', o.detach().clone()))
    with torch.no_grad():
        model(ids, attention_mask=mask)
    h.remove()
    d = torch.ones(32)
    rec = {}
    with ref.add_delta(mlp, d, pos, rec), torch.no_grad():
        h = mlp.register_forward_hook(lambda m, a, o: seen.__setitem__('edited', o.detach().clone()))  # after add_delta
        model(ids, attention_mask=mask)
    h.remove()
    for i, p in enumerate(pos):
        assert torch.allclose(seen['edited'][i, p] - seen['native'][i, p], d)
    assert torch.allclose(rec['target_init'], seen['native'][0, 0])


def test_solve_meets_the_reference_algebra(tiny):
    model, tok = tiny
    req = v1.Request('{} is a company that was founded in the country of', 'BMW', ' Italy')
    templates = ['{}', 'Hello world. {}', 'The. {}']
    C = np.eye(64) * 2.0 + 0.01
    r = ref.solve(model, tok, req, templates, C, HP)
    assert len(r['trace']) == 20                                    # A3 §3: exactly 20 updates
    assert abs(np.linalg.norm(r['V']) - 1) < 1e-6                  # u is unit-length
    q = np.linalg.solve(C, r['k_mean'].astype(np.float64))
    assert np.allclose(r['V'][:, 0], q / np.linalg.norm(q), atol=1e-6)
    t = np.linalg.norm(r['target_init'].astype(np.float64))
    for a, b in zip(r['trace'], r['trace'][1:]):                     # unsquared decay, computed before each update
        assert b['decay'] == pytest.approx(0.5 * a['delta_norm'] / t ** 2, rel=1e-4)
    assert r['trace'][-1]['delta_norm'] <= 4 * t * (1 + 1e-5)       # clamp
    assert r['trace'][-1]['nll'] < r['trace'][0]['nll']
    # the stored adapter maps k_clean to target_init + delta through the served module
    from edit_cert.backends.lowrank import LowRank
    ad = LowRank([dict(module='transformer.h.1.mlp.c_proj', layer=1)], dict(U0=r['U'], V0=r['V']))

    class Inst:
        pass
    inst = Inst()
    inst.model = model
    with ad.apply(inst), torch.inference_mode():
        y = v1.module(model, 1)(torch.tensor(r['k_clean'])[None, None])[0, 0].numpy()
    assert np.abs(y - (r['target_init'] + r['delta'])).max() < 1e-4
    r2 = ref.solve(model, tok, req, templates, C, HP)
    assert np.array_equal(r['U'], r2['U']) and np.array_equal(r['V'], r2['V'])
