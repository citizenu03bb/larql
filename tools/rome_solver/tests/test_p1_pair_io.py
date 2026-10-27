import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
spec = importlib.util.spec_from_file_location('pair_io', Path(__file__).parents[1] / 'xcheck_p1_pairs.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class PairIO(unittest.TestCase):
    def test_own_incomplete_gate_precedes_any_peer_read(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'solves-codex-index.json').write_text('{"complete": false}')
            with self.assertRaisesRegex(ValueError, 'Own completion gate closed'):
                m.run(root)

    def test_changed_factor_fails_content_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            np.save(root / 'U0.npy', np.ones((1600, 1), dtype=np.float32))
            np.save(root / 'V0.npy', np.ones((6400, 1), dtype=np.float32))
            meta = dict(kind='low_rank', arrays=['U0', 'V0'], content_sha256='0' * 64,
                        params=dict(targets=[dict(layer=17, module='transformer.h.17.mlp.c_proj', scale=1.)]))
            (root / 'backend.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'Content hash mismatch'):
                m.edit(root)


if __name__ == '__main__':
    unittest.main()
