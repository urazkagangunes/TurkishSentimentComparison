"""Sampling invariants; no models or inference."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scripts.balanced_classical_sample import load_disjoint_sample
from tests.run_trsav1_classical_1200 import ROOT


class DisjointClassicalSampleTests(unittest.TestCase):
    def test_real_3000_balanced_repeatable_and_disjoint(self):
        old = ROOT / "results/trsav1_classical_1200_20261101_shared_baseline/sample_1200.csv"
        sample, cases, info = load_disjoint_sample(1000, [old])
        repeated, repeated_cases, repeated_info = load_disjoint_sample(1000, [old])
        self.assertTrue(sample.equals(repeated))
        self.assertEqual((cases, info), (repeated_cases, repeated_info))
        self.assertEqual(info["sample_distribution"], dict(negative=1000, neutral=1000, positive=1000))
        self.assertEqual(len(set(sample["text"])), 3000)
        self.assertFalse(set(sample["text"]) & set(pd.read_csv(old)["text"]))
        self.assertEqual(len(set(info["source_cleaned_row_ids"])), 3000)

    def test_exclusion_ignores_label_and_duplicate_texts_are_not_relabelled(self):
        with TemporaryDirectory() as temp:
            source, old = Path(temp)/"source.csv", Path(temp)/"old.csv"
            pd.DataFrame([
                ("exclude", "negative"), ("exclude", "positive"),
                ("conflict", "negative"), ("conflict", "neutral"),
                ("negative2", "negative"), ("neutral2", "neutral"),
                ("neutral3", "neutral"), ("positive1", "positive"), ("positive2", "positive"),
            ], columns=["review", "score"]).to_csv(source, sep=";", index=False)
            pd.DataFrame([dict(text=" exclude ", gold="neutral")]).to_csv(old, index=False)
            sample, _, info = load_disjoint_sample(2, [old], source)
            self.assertNotIn("exclude", set(sample["text"]))
            self.assertEqual(sample["text"].nunique(), 6)
            self.assertEqual(info["excluded_source_rows"], 2)
            self.assertEqual(sample.loc[sample.text == "conflict", "gold"].tolist(), ["negative"])
            with self.assertRaisesRegex(ValueError, "Not enough eligible"):
                load_disjoint_sample(3, [old], source)
            with self.assertRaisesRegex(ValueError, "positive"):
                load_disjoint_sample(0, [old], source)


if __name__ == "__main__":
    unittest.main()
