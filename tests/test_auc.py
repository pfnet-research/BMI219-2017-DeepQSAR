import unittest

import numpy as np
from sklearn import metrics

from lib.evaluations import auc as auc_


class TestAUC(unittest.TestCase):

    def test_all_ignored(self):
        t = np.full((3,), -1, dtype=np.int32)
        p = np.random.uniform(0, 1, (3,)).astype(np.float32)
        auc = auc_.compute_auc(t, p)
        np.testing.assert_almost_equal(auc, 0.)

    def test_ignored(self):
        t = np.array([0, 1, 1, -1], dtype=np.int32)
        p = np.random.uniform(0, 1, (4,)).astype(np.float32)
        auc_actual = auc_.compute_auc(t, p)

        t = t[:3]
        p = p[:3]
        auc_expect = metrics.roc_auc_score(t, p)

        np.testing.assert_almost_equal(auc_expect, auc_actual)
