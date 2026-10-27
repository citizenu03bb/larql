"""Model-free tests of Claude's independent ROME solve, on a tiny random GPT-2 (no real weights)."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')

from p1_rome_claude import solve as rome  # noqa: E402

TOK_DIR = '/mnt/models/hf-models/gpt2-xl'


@pytest.fixture(scope='module')
def tiny():
    import os
    if not os.path.exists(f'{TOK_DIR}/tokenizer.json'):
        pytest.skip('GPT-2 tokenizer not on disk')
    tok = transformers.AutoTokenizer.from_pretrained(TOK_DIR, local_files_only=True)
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(n_layer=3, n_embd=32, n_head=4, n_inner=64, vocab_size=len(tok), n_positions=128)
    model = transformers.GPT2LMHeadModel(cfg).eval()
    return model, tok


HP = dict(rome.PUBLISHED, layer=1, steps=8, n_short=2, len_short=3, n_long=1, len_long=5)


def test_subject_last_token(tiny):
    _, tok = tiny
    text = 'Previously. BMW is a company that was founded in the country of'
    i = rome.subject_last(tok, text, 'BMW')
    assert tok.decode(tok.encode(text)[i]).strip() == 'BMW'


def test_templates_are_deterministic(tiny):
    model, tok = tiny
    a = rome.context_templates(model, tok, 7, HP)
    assert a == rome.context_templates(model, tok, 7, HP)
    assert a[0] == '{}' and len(a) == 4 and all(t.endswith('. {}') for t in a[1:])


def test_solve_meets_the_rank_one_constraint_and_lowers_the_loss(tiny):
    model, tok = tiny
    req = rome.Request('{} is a company that was founded in the country of', 'BMW', ' Italy')
    templates = rome.context_templates(model, tok, 7, HP)
    C = np.eye(64) * 2.0 + 0.01
    r = rome.solve(model, tok, req, templates, C, HP)
    assert r['U'].shape == (32, 1) and r['V'].shape == (64, 1)
    assert r['constraint_error'] < 1e-4
    assert float(r['V'][:, 0] @ r['k_star']) == pytest.approx(1.0, rel=1e-4)
    assert r['trace'][-1]['nll'] < r['trace'][0]['nll']
    assert np.linalg.norm(r['v_star'] - r['z0']) <= HP['clamp'] * np.linalg.norm(r['z0']) * (1 + 1e-5)
    # the stored adapter reproduces v* through the served module
    from edit_cert.backends.lowrank import LowRank
    ad = LowRank([dict(module='transformer.h.1.mlp.c_proj', layer=1)], dict(U0=r['U'], V0=r['V']))

    class Inst:
        pass
    inst = Inst()
    inst.model = model
    with ad.apply(inst), torch.inference_mode():
        y = rome.module(model, 1)(torch.tensor(r['k_star'])[None, None])[0, 0].numpy()
    assert np.abs(y - r['v_star']).max() < 1e-4
    r2 = rome.solve(model, tok, req, templates, C, HP)
    assert np.array_equal(r['U'], r2['U']) and np.array_equal(r['V'], r2['V'])
