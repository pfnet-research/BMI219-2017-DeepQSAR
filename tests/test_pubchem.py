import unittest

import numpy as np

from lib.data import pubchem


class TestGrouper(unittest.TestCase):

    def test_even(self):
        a = np.arange(6)
        g = pubchem._grouper(a, 2)
        self.assertEqual(g.next(), (0, 1))
        self.assertEqual(g.next(), (2, 3))
        self.assertEqual(g.next(), (4, 5))

    def test_odd(self):
        a = np.arange(5)
        g = pubchem._grouper(a, 2)
        self.assertEqual(g.next(), (0, 1))
        self.assertEqual(g.next(), (2, 3))
        self.assertEqual(g.next(), (4, None))
