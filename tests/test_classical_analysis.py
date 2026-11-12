"""Small offline checks for the saved-result report, without real classifiers."""
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.analyze_classical_500 import SYSTEMS, analyze, result_flags
from src.evaluation.checkpoint_store import CheckpointStore, atomic_json


def result(positive, negative, original_positive=None, original_negative=None):
    original_positive = positive if original_positive is None else original_positive
    original_negative = negative if original_negative is None else original_negative
    prediction = "positive" if positive-negative > .05 else "negative" if negative-positive > .05 else "neutral"
    return dict(status="success", prediction=prediction, predictions=[prediction], error=None,
        positive_score=positive, negative_score=negative, hisnet_positive_score=original_positive,
        hisnet_negative_score=original_negative, dependency_positive_score=positive,
        dependency_negative_score=negative, dependency_prediction=prediction,
        morphological_analyses=[], sentiment_words=["example"] if positive+negative else [],
        lexicon_coverage=.5, candidate_lexicon_coverage=.5, sentiment_coverage=.3,
        scopes=[dict(adjusted_positive=positive, adjusted_negative=negative, operations=[], alignment_status="matched")])


class ClassicalAnalysisTests(unittest.TestCase):
    def test_numeric_and_baseline_invariants_detect_corruption(self):
        baseline = result(.5,0)
        changed = result(0,.5,.5,0)
        self.assertEqual(result_flags("hisnet_dp",changed,baseline,None), [])
        changed["positive_score"] = .4
        flags = result_flags("hisnet_dp",changed,baseline,None)
        self.assertIn("scope_sum_positive",flags)
        changed["hisnet_positive_score"] = .1
        self.assertIn("hisnet_baseline_positive",result_flags("hisnet_dp",changed,baseline,None))

    def test_saved_results_report_metrics_errors_and_pairwise_counts(self):
        cases = [dict(text="güzel",gold="positive"),dict(text="güzel değil",gold="negative"),dict(text="masa",gold="neutral")]
        with TemporaryDirectory() as directory:
            output = Path(directory)
            with CheckpointStore(output,{},cases) as store:
                for system in SYSTEMS:
                    values = [result(.5,0),result(.5,0) if system=="hisnet" else result(0,.5,.5,0),result(0,0)]
                    for i,value in enumerate(values):
                        attempt = store.start(system,[i],False)
                        store.finish(attempt,value,.1)
            atomic_json(output/"sampling.json",dict(conflicting_source_label_row_ids=[],dataset_size=3,
                sample_distribution={"positive":1,"negative":1,"neutral":1},matches_seed42_reference=True))
            atomic_json(output/"initializations.json",[])
            with contextlib.redirect_stdout(io.StringIO()):
                report = analyze(output)
            self.assertEqual(report["violations"],[])
            metrics = {m["system"]:m for m in report["metrics"]}
            self.assertAlmostEqual(metrics["hisnet"]["accuracy"],2/3)
            self.assertEqual(metrics["hisnet_dp"]["accuracy"],1)
            pair = next(p for p in report["pairwise"] if p["first"]=="hisnet" and p["second"]=="hisnet_dp")
            self.assertEqual((pair["improved"],pair["harmed"],pair["changed"]),(1,0,1))
            self.assertEqual(report["disagreement_ids"],[1])
            self.assertTrue((output/"report.md").is_file())
            self.assertEqual(json.loads((output/"analysis.json").read_text(encoding="utf-8"))["n"],3)
            report_text = (output/"report.md").read_text(encoding="utf-8")
            self.assertIn("sabit 3 yorum", report_text)
            self.assertIn("1/3 = %33.33", report_text)
            self.assertNotIn("500", report_text)
            self.assertTrue((output/"comparison_3.csv").is_file())

    def test_cross_sentence_and_duplicate_negation_flags(self):
        value = result(.5,0)
        value["dependency_tokens"] = [dict(id=1,sentence_index=0),dict(id=2,sentence_index=1)]
        value["scopes"][0].update(sentiment_token={"id":1},negations=[{"id":2},{"id":2}])
        flags = result_flags("dp_hisnet",value,None,None)
        self.assertIn("negation_crosses_sentence_boundary",flags)
        self.assertIn("duplicate_negation_in_scope",flags)


if __name__ == "__main__":
    unittest.main()
