"""Sampling/report checks; never initialize sentiment models."""
import unittest

from tests.run_trsav1_classical_1200 import load_sample, SYSTEMS


class BalancedClassicalExperimentTests(unittest.TestCase):
    def test_frozen_balanced_sample_and_source_identity(self):
        sample, cases, info = load_sample()
        repeated, repeated_cases, repeated_info = load_sample()
        self.assertTrue(sample.equals(repeated))
        self.assertEqual(cases, repeated_cases)
        self.assertEqual(info, repeated_info)
        self.assertEqual(len(cases), 1200)
        self.assertEqual(info["sample_distribution"], dict(negative=400, neutral=400, positive=400))
        self.assertFalse(sample.duplicated().any())
        self.assertEqual(len(set(info["source_cleaned_row_ids"])), 1200)
        self.assertEqual(len(info["source_sha256"]), 64)
        self.assertTrue(all(0 <= i < 1200 for i in info["conflicting_source_label_row_ids"]))
        self.assertEqual(set(SYSTEMS), {"hisnet", "dp_hisnet", "hisnet_dp", "lw_hisnet_dp"})


if __name__ == "__main__":
    unittest.main()
