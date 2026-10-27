"""Model-free tests of the P1 GPT-2 XL instrument readings and its cap authority."""
from pathlib import Path

import pytest

from edit_cert import plan as planner
from edit_cert.instruments import hf_gpt2

GPT2 = Path('/mnt/models/hf-models/gpt2-xl')


def test_cap_400_is_approved_per_instrument():
    assert planner.check_cap(400, 'hf-gpt2xl')['authority'].startswith('phase-b-p1')
    assert planner.check_cap(400, 'nf4-e2b')['authority'].startswith('phase-b-decision Q1')
    assert planner.check_cap(24, 'hf-gpt2xl') is None
    with pytest.raises(ValueError):
        planner.check_cap(400, 'nf4-e4b-larql')
    with pytest.raises(ValueError):
        planner.check_cap(800, 'hf-gpt2xl')


class _Tok:
    """A five-token vocabulary standing in for the GPT-2 tokenizer."""
    eos_token_id = 4
    _v = [' Paris', ',', ' St.', '\n', '<|endoftext|>']

    def __len__(self):
        return len(self._v)

    def decode(self, ids):
        return ''.join(self._v[i] for i in ids)


def test_stop_tokens_on_a_stub_vocabulary():
    assert hf_gpt2.stop_token_ids(_Tok()) == frozenset({1, 2, 3, 4})


@pytest.mark.skipif(not (GPT2 / 'tokenizer.json').exists(), reason='GPT-2 XL tokenizer not on disk')
def test_stop_tokens_on_the_real_tokenizer():
    AutoTokenizer = pytest.importorskip('transformers').AutoTokenizer
    tok = AutoTokenizer.from_pretrained(GPT2, local_files_only=True)
    stops = hf_gpt2.stop_token_ids(tok)
    for s in ('.', ',', '\n', ';', '?', '!'):
        assert all(i in stops for i in tok.encode(s)), s
    for s in (' Paris', ' Rome', ' the', ' French', ' 1889'):
        assert not any(i in stops for i in tok.encode(s)), s
    ids = tok.encode('The Eiffel Tower is located in the city of Paris, France.')
    first_stop = next(k for k, i in enumerate(ids) if i in stops)
    assert tok.decode(ids[:first_stop + 1]).endswith('Paris,')
