"""Small offline regressions; no Stanza weights, Ollama server or dataset run.

Run only this suite: python -m unittest discover -s tests -p test_audit_regressions.py -v
"""
import json
import ast
import contextlib
import io
import runpy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.common.morphology import MorphologicalRootExtractor
from src.dependency.sentiment_scope_resolver import SentimentScopeResolver
from src.evaluation.sentiment_evaluator import SentimentEvaluator
from src.evaluation.run_identity import ensure_run_manifest
from src.hisnet.hisnet_classifier import HisNetClassifier
from src.llm.ollama_zero_shot_classifier import OllamaZeroShotSentimentClassifier
from src.llm.dependency_ollama_zero_shot_classifier import DependencyOllamaZeroShotSentimentClassifier


def token(index, text, head=0, relation="root", sentence=0, pos="ADJ"):
    return dict(id=index, text=text, lemma=text, head=head, deprel=relation,
                sentence_index=sentence, upos=pos, feats=None)


def response(content):
    return SimpleNamespace(message=SimpleNamespace(content=content),
                           prompt_eval_count=10, eval_count=5)


class ScopeRegressions(unittest.TestCase):
    def setUp(self):
        self.resolver = SentimentScopeResolver()

    def test_negation_does_not_cross_sentence(self):
        tokens = [token(1, "güzel"), token(2, "merkezi", sentence=1),
                  token(3, "değil", 2, "cop", sentence=1, pos="AUX")]
        scopes = self.resolver.resolve(
            [dict(word="güzel", root="güzel", positive=1., negative=0.)], tokens)
        self.assertEqual(scopes[0]["negations"], [])

    def test_coordinated_negation_still_shared(self):
        tokens = [token(1, "iyi"), token(2, "ve", 3, "cc", pos="CCONJ"),
                  token(3, "güzel", 1, "conj"), token(4, "değil", 3, "cop", pos="AUX")]
        scopes = self.resolver.resolve(
            [dict(word=w, root=w, positive=1., negative=0.) for w in ["iyi", "güzel"]], tokens)
        self.assertEqual([[n["id"] for n in s["negations"]] for s in scopes], [[4], [4]])

    def test_contrast_prevents_shared_negation(self):
        tokens = [token(1, "iyi"), token(2, "ama", 3, "cc", pos="CCONJ"),
                  token(3, "güzel", 1, "conj"), token(4, "değil", 3, "cop", pos="AUX")]
        scopes = self.resolver.resolve(
            [dict(word=w, root=w, positive=1., negative=0.) for w in ["iyi", "güzel"]], tokens)
        self.assertEqual([[n["id"] for n in s["negations"]] for s in scopes], [[], [4]])

    def test_morphology_uses_final_derived_pos(self):
        for analysis, expected in [("güzel+ADJ^DB+VERB+ZERO+PAST+A3SG", "VERB"),
                                   ("sev+VERB+POS^DB+NOUN+INF2+A3SG", "NOUN"),
                                   ("iyi+ADJ^DB+ADV+LY", "ADV"),
                                   ("güzel+ADJ", "ADJ")]:
            with self.subTest(analysis=analysis):
                self.assertEqual(self.resolver._morphology_pos(dict(analysis=analysis)), expected)

    def test_root_fallback_cannot_steal_an_exact_surface_match(self):
        tokens = [token(1, "harik"), token(2, "a", 1, "case", pos="ADP"),
                  token(3, "harika", sentence=1)]
        words = [dict(word="harikaydı", root="harika", positive=1., negative=0.),
                 dict(word="harika", root="harika", positive=1., negative=0.)]
        scopes = self.resolver.resolve(words, tokens)
        self.assertEqual(scopes[1]["sentiment_token"]["id"], 3)
        self.assertEqual(scopes[1]["alignment_strategy"], "exact_surface")
        self.assertEqual(sum(s["positive"] for s in scopes), 2.)

    def test_repeated_surface_uses_its_own_morphological_root(self):
        tokens = [token(1, "yazar", pos="NOUN"), token(2, "yazar", 1, "conj", pos="VERB")]
        morphology = [dict(word="yazar", root="yazar"), dict(word="yazar", root="yaz")]
        candidates = self.resolver.resolve_dependency_candidates(tokens, morphology)
        self.assertEqual([c["root"] for c in candidates], ["yazar", "yaz"])

    def test_threshold_boundaries_are_neutral(self):
        for positive, negative in [(0.2, 0.15), (0.15, 0.2), (0.05, 0.), (0., 0.05)]:
            with self.subTest(positive=positive, negative=negative):
                self.assertEqual(HisNetClassifier._prediction_from_scores(positive, negative), "neutral")
        self.assertEqual(HisNetClassifier._prediction_from_scores(0.050001, 0.), "positive")
        self.assertEqual(HisNetClassifier._prediction_from_scores(0., 0.050001), "negative")


class MorphologyRegressions(unittest.TestCase):
    def test_normal_analysis_does_not_expose_analyzer_cache(self):
        extractor = MorphologicalRootExtractor.__new__(MorphologicalRootExtractor)
        cached = ["root-a", "root-b"]
        extractor.analyzer = Mock()
        extractor.analyzer.morphologicalAnalysis.return_value = cached
        extractor._normal_requests = extractor._normal_resets = 0
        first, _ = extractor._normal_analysis("surface")
        first.pop()  # The installed disambiguator reduces parse lists in place.
        second, _ = extractor._normal_analysis("surface")
        self.assertEqual(second, ["root-a", "root-b"])
        self.assertIsNot(first, second)


class LLMRegressions(unittest.TestCase):
    def setUp(self):
        self.classifier = OllamaZeroShotSentimentClassifier(max_retries=0)
        self.classifier.client = Mock()

    def test_invalid_ids_are_rejected_as_validation_errors(self):
        for invalid in [True, 1.0, [], {}, None, "1"]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self.classifier._validate_predictions([dict(id=invalid, label="positive")], 1)

    def test_valid_predictions_reordered(self):
        self.assertEqual(self.classifier._validate_predictions(
            [dict(id=2, label="negative"), dict(id=1, label="positive")], 2),
            ["positive", "negative"])

    def test_malformed_json_shapes_do_not_crash(self):
        for content in ["[]", "null", "42", "true", None,
                        '{"predictions":[{"id":[],"label":"positive"}]}']:
            with self.subTest(content=content):
                self.classifier.client.chat.return_value = response(content)
                result = self.classifier.classify_batch(["yorum"])
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["predictions"], [None])

    def test_retry_time_and_tokens_include_failed_attempt(self):
        self.classifier.max_retries = 1
        self.classifier.client.chat.side_effect = [response("broken json"), response(
            '{"predictions":[{"id":1,"label":"positive"}]}')]
        with patch("src.llm.ollama_zero_shot_classifier.time.perf_counter",
                   side_effect=[0., 1., 2., 3., 8.]):
            result = self.classifier.classify_batch(["yorum"])
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["request_seconds"], 7.)
        self.assertEqual(self.classifier.total_wall_seconds, 7.)
        self.assertEqual(self.classifier.total_prompt_tokens, 20)
        self.assertEqual(self.classifier.total_output_tokens, 10)
        self.assertEqual(result["prompt_tokens"], 20)
        self.assertEqual(result["output_tokens"], 10)

    def test_invalid_batch_configuration_fails_early(self):
        for kwargs in [dict(batch_size=0), dict(batch_size=-1), dict(max_retries=-1)]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    OllamaZeroShotSentimentClassifier(**kwargs)

    def test_dependency_parse_failure_is_recoverable(self):
        with patch("src.llm.dependency_ollama_zero_shot_classifier.TurkishDependencyParser"):
            classifier = DependencyOllamaZeroShotSentimentClassifier(max_retries=0)
        classifier.parser.parse.side_effect = RuntimeError("parser failed")
        classifier.client = Mock()
        result = classifier.classify("yorum")
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["prediction"])
        self.assertIn("parser failed", result["error"])
        classifier.client.chat.assert_not_called()


class CheckpointRegressions(unittest.TestCase):
    def test_cached_predictions_require_matching_manifest(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            predictions = Path(directory) / "predictions.csv"
            predictions.write_text("existing results")
            with self.assertRaises(ValueError):
                ensure_run_manifest(path, {"model": "a"}, [predictions])
            self.assertFalse(path.exists())
            ensure_run_manifest(path, {"model": "a"})
            ensure_run_manifest(path, {"model": "a"}, [predictions])
            with self.assertRaises(ValueError):
                ensure_run_manifest(path, {"model": "b"}, [predictions])

    def test_resume_requires_same_implementation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            SentimentEvaluator._save_checkpoint(path, "x", "data", 1, [], "code-v1")
            self.assertEqual(SentimentEvaluator._load_checkpoint(path, "x", "data", 1, "code-v1"), [])
            with self.assertRaises(ValueError):
                SentimentEvaluator._load_checkpoint(path, "x", "data", 1, "code-v2")

    def test_legacy_checkpoint_is_not_silently_reused(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            path.write_text(json.dumps(dict(checkpoint_version=1, dataset="x", fingerprint="data",
                                           total_examples=1, processed_examples=0, rows=[])))
            with self.assertRaises(ValueError):
                SentimentEvaluator._load_checkpoint(path, "x", "data", 1)


class ExistingChecks(unittest.TestCase):
    def test_source_and_experiment_scripts_parse(self):
        root = Path(__file__).resolve().parents[1]
        for folder in ["src", "tests", "scripts"]:
            for path in (root / folder).rglob("*.py"):
                with self.subTest(path=path.name):
                    ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))

    def test_existing_offline_rule_checks(self):
        with contextlib.redirect_stdout(io.StringIO()):
            for name in ["test_sentiment_threshold.py", "test_negation_scope_rules.py"]:
                runpy.run_path(str(Path(__file__).with_name(name)))


if __name__ == "__main__":
    unittest.main()
