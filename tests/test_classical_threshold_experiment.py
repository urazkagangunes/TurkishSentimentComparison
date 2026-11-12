"""Threshold boundaries, replay invariants and gain/loss accounting."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from scripts.run_balanced_classical_cached import classify_shared, semantic_result
from scripts.run_classical_threshold_experiment import (
    SYSTEMS, assert_trial, decision_threshold, independent_label, metrics,
    replay_hisnet, run_four, summarize,
)
from src.hisnet.hisnet_classifier import HisNetClassifier


def fixture(positive, negative):
    morphology = [dict(word="güzel", root="güzel", analysis="güzel+ADJ")]
    lookup = {"güzel": dict(positive=positive, negative=negative)}
    parsed = [dict(text="güzel", tokens=[dict(id=1, text="güzel", lemma="güzel",
        head=0, deprel="root", upos="ADJ", feats=None)])]
    classifier = HisNetClassifier.__new__(HisNetClassifier)
    classifier.root_extractor = SimpleNamespace(analyze_sentence=lambda text: deepcopy(morphology))
    classifier._lookup_root = lambda root: deepcopy(lookup[root])
    with decision_threshold(.05):
        saved = {"hisnet": classifier.classify("güzel")}
        for system in SYSTEMS[1:]:
            saved[system] = classify_shared(system, "güzel", saved["hisnet"], parsed, lookup.get)
    return saved


class ThresholdExperimentTests(unittest.TestCase):
    def test_all_methods_boundary_and_unchanged_intermediates(self):
        for magnitude in (.06, .07, .08, .10, .100001):
            for negative in (False, True):
                p, n = (0., magnitude) if negative else (magnitude, 0.)
                saved = fixture(p, n)
                original = deepcopy(saved)
                for threshold in (.05, .07, .10):
                    results = run_four(saved, threshold)
                    for system in SYSTEMS:
                        with self.subTest(magnitude=magnitude, negative=negative, threshold=threshold, system=system):
                            assert_trial(saved[system], results[system], threshold)
                            expected = "neutral" if magnitude <= threshold else ("negative" if negative else "positive")
                            self.assertEqual(results[system]["prediction"], expected)
                            if threshold == .05:
                                self.assertEqual(semantic_result(saved[system]), semantic_result(results[system]))
                    self.assertEqual(results["lw_hisnet_dp"]["literature_weighting"]["decision_threshold"], threshold)
                self.assertEqual(saved, original)

    def test_threshold_context_restores_on_failure(self):
        module = importlib.import_module("src.hisnet.hisnet_classifier")
        with self.assertRaises(RuntimeError):
            with decision_threshold(.10):
                self.assertEqual(module.NEUTRAL_THRESHOLD, .10)
                raise RuntimeError("interrupted")
        self.assertEqual(module.NEUTRAL_THRESHOLD, .05)
        for threshold in (float("nan"), float("inf"), -.1):
            with self.assertRaises(ValueError):
                with decision_threshold(threshold):
                    pass

    def test_float_subtraction_and_zero_are_neutral(self):
        for threshold, positive, negative in ((.07, .10, .03), (.10, .30, .20), (.05, .20, .15)):
            with decision_threshold(threshold):
                for p, n in ((positive, negative), (negative, positive), (0., 0.)):
                    self.assertEqual(HisNetClassifier._prediction_from_scores(p, n), "neutral")
                    self.assertEqual(independent_label(p, n, threshold), "neutral")

    def test_missing_lexical_evidence_and_changed_scores_fail(self):
        saved = fixture(.06, 0.)
        incomplete = deepcopy(saved["hisnet"])
        incomplete["matched_words"].clear()
        with self.assertRaises(KeyError):
            replay_hisnet(incomplete)
        after = run_four(saved, .07)["hisnet"]
        after["positive_score"] += .001
        with self.assertRaisesRegex(AssertionError, "non-decision"):
            assert_trial(saved["hisnet"], after, .07)

    def test_gain_loss_accounting_includes_polar_harm(self):
        cases = [dict(text=str(i), gold=g) for i, g in enumerate(
            ["positive", "positive", "negative", "negative", "neutral", "neutral"])]
        labels = {.05: ["positive", "positive", "negative", "neutral", "positive", "negative"],
                  .07: ["neutral", "positive", "negative", "neutral", "neutral", "negative"],
                  .10: ["neutral", "positive", "neutral", "neutral", "neutral", "neutral"]}
        runs = {t: {s: [dict(prediction=p, positive_score=0., negative_score=0.) for p in values]
                    for s in SYSTEMS} for t, values in labels.items()}
        self.assertEqual(metrics(cases, runs[.05])["hisnet"]["classes"]["positive"]["recall"], 1.)
        with TemporaryDirectory() as directory:
            trades = summarize(Path(directory), cases, runs)
            first, second = trades[0], trades[4]
            self.assertEqual((first["neutral_gained"], first["positive_lost"], first["negative_lost"], first["net_correct"]), (1, 1, 0, 0))
            self.assertEqual((second["neutral_gained"], second["positive_lost"], second["negative_lost"], second["net_correct"]), (2, 1, 1, 0))
            payload = json.loads((Path(directory)/"comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["threshold_order"], [.05, .07, .10])


if __name__ == "__main__":
    unittest.main()
