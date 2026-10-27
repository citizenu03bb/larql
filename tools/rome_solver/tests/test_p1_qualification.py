import importlib.util
from pathlib import Path
import unittest

module=importlib.util.spec_from_file_location('p1qualification',Path(__file__).parents[1]/'xcheck_p1_qualification.py')
q=importlib.util.module_from_spec(module);module.loader.exec_module(q)


class Readings(unittest.TestCase):
    def test_frozen_negation_scope_and_word_boundaries(self):
        self.assertFalse(q.stated(' not Russia.', ['Russia']))
        self.assertFalse(q.stated(' not at all British', ['British']))
        self.assertTrue(q.stated(' not known to the British government', ['British']))
        self.assertFalse(q.stated(' Russian', ['Russia']))
        self.assertTrue(q.stated(' **Russia**.', ['Russia']))


if __name__=='__main__':unittest.main()
