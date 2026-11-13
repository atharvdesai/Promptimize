import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from exp_laya_embeddings import metrics


class MetricsConventionTests(unittest.TestCase):
    """Pins the over/under convention: PER-CLASS rates, not fractions of
    total n. over = false-asks / executable rows; under = misses /
    ambiguous rows. (D1-errata: an ad-hoc script used n as the
    denominator and misreported 1/21 = 0.048 as 1/44 = 0.023.)"""

    def test_per_class_denominators(self):
        # 4 executable (label 0), 3 ambiguous (label 1); thr 0.5
        # asks: rows 0 (exec, false-ask), 4,5 (amb, correct)
        # passes: rows 1,2,3 (exec, correct), 6 (amb, miss) -> 5/7 correct
        probs = np.array([0.9, 0.1, 0.2, 0.3, 0.8, 0.7, 0.1])
        y = np.array([0, 0, 0, 0, 1, 1, 1])
        m = metrics(probs, y, 0.5)
        self.assertEqual(m["acc"], round(5 / 7, 3))
        self.assertEqual(m["over"], round(1 / 4, 3))   # not 1/7
        self.assertEqual(m["under"], round(1 / 3, 3))  # not 1/7
        self.assertEqual(m["n"], 7)

    def test_zero_rates(self):
        probs = np.array([0.1, 0.2, 0.9, 0.8])
        y = np.array([0, 0, 1, 1])
        m = metrics(probs, y, 0.5)
        self.assertEqual(m["acc"], 1.0)
        self.assertEqual(m["over"], 0.0)
        self.assertEqual(m["under"], 0.0)


if __name__ == "__main__":
    unittest.main()
