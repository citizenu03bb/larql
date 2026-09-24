import unittest
import numpy as np
import natural_binding_pilot as pilot


class ModelFreeChecks(unittest.TestCase):
    def test_disjoint_entities_constructions_and_exact_budget(self):
        rows = pilot.corpus()
        d = [r for r in rows if r['split'] == 'discovery']
        h = [r for r in rows if r['split'] == 'confirmation']
        self.assertEqual(len(d), 20)
        self.assertEqual(len(h), 20)
        self.assertTrue({r['entity'] for r in d if r['entity'] is not None}.isdisjoint(
            {r['entity'] for r in h if r['entity'] is not None}))
        self.assertTrue({r['template'] for r in d}.isdisjoint({r['template'] for r in h}))
        jobs = pilot.ledger(rows)
        self.assertEqual(len(jobs), 368)
        self.assertEqual(sum(j['arm'] in pilot.ARMS for j in jobs), 288)
        by_id = {r['id']: r for r in rows}
        self.assertEqual(len({(j['recipient'], j['arm']) for j in jobs}), 368)
        for j in jobs:
            if 'donor' in j:
                self.assertEqual(by_id[j['donor']]['split'], j['split'])

    def test_projection_preserves_complement_and_self(self):
        u = np.array([0.6, 0.8, 0.0])
        r, d = np.array([1., 2., 3.]), np.array([-2., 5., 7.])
        out = pilot.project(r, d, u)
        np.testing.assert_allclose(np.dot(u, out), np.dot(u, d), atol=1e-12)
        np.testing.assert_allclose(out - u*np.dot(u, out), r-u*np.dot(u, r), atol=1e-12)
        np.testing.assert_array_equal(pilot.project(r, r, u), r)

    def test_norm_controls_have_equal_magnitude_and_opposite_sign(self):
        u = np.array([0.6, 0.8, 0.0])
        r, d = np.array([1., 2., 3.]), np.array([-2., 5., 7.])
        a = pilot.random_control(r, d, u, 'synthetic', 1) - r
        b = pilot.random_control(r, d, u, 'synthetic', -1) - r
        np.testing.assert_allclose(a, -b, atol=1e-12)
        self.assertAlmostEqual(float(np.dot(a, u)), 0, places=12)
        self.assertAlmostEqual(np.linalg.norm(a), np.linalg.norm(pilot.project(r, d, u)-r), places=12)

    def test_recovers_interaction_despite_additive_nuisances(self):
        rows = [r for r in pilot.corpus() if r['split'] == 'discovery' and r['entity'] is not None]
        u = np.array([0.0, 0.0, 1.0])
        captures = {}
        for r in rows:
            e, t, rel = r['entity'], r['template'], r['relation']
            captures[r['id']] = np.array([100*e, 70*t + 30*rel, 0.0]) + u*(e+1)*(1-2*rel)
        np.testing.assert_allclose(pilot.fit_rank_one(rows, captures), u, atol=1e-12)
        captures = {r['id']: np.array([100*r['entity'], 70*r['template'], 30*r['relation']], dtype=float) for r in rows}
        with self.assertRaises(ValueError):
            pilot.fit_rank_one(rows, captures)

    def test_confirmation_cannot_enter_fit(self):
        with self.assertRaises(ValueError):
            pilot.fit_rank_one(pilot.corpus(), {})

    def test_rejects_invalid_vectors(self):
        for r, d, u in (([1, 2], [1], [1, 0]), ([1, 2], [1, 2], [1, 1]),
                        ([1, float('nan')], [1, 2], [1, 0])):
            with self.assertRaises(ValueError):
                pilot.project(r, d, u)

    def test_strict_gate_rejects_nuisance_and_wrong_sign(self):
        effects = {a: {'delta_gap': 0., 'delta_target': 0.} for a in pilot.ARMS}
        for a in ('matched', 'cross_construction'):
            effects[a] = {'delta_gap': 0.5, 'delta_target': 0.3}
        self.assertTrue(pilot.row_passes(effects))
        effects['wrong_entity']['delta_gap'] = -0.6
        self.assertFalse(pilot.row_passes(effects))
        effects['wrong_entity']['delta_gap'] = 0
        effects['cross_construction']['delta_target'] = -0.1
        self.assertFalse(pilot.row_passes(effects))


if __name__ == '__main__':
    unittest.main()
