"""Offline regressions only: no Ollama requests, Stanza models or HisNet loads."""
import contextlib
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx

from src.evaluation.checkpoint_store import CheckpointStore, SYSTEMS, read_status
from src.evaluation.resumable_evaluator import evaluate, LLM_SYSTEMS
from src.llm.ollama_zero_shot_classifier import OllamaZeroShotSentimentClassifier
from src.llm.dependency_ollama_zero_shot_classifier import DependencyOllamaZeroShotSentimentClassifier

CASES = [dict(text=f"yorum {i}", gold="positive") for i in range(5)]
SETTINGS = dict(batch_size=2)


def success(count):
    return dict(status="success", predictions=["positive"] * count, error=None,
                attempt=1, prompt_tokens=10, output_tokens=5)


class FakeClassifier:
    def __init__(self, fail_call=None, interrupt_call=None):
        self.calls = []
        self.fail_call = fail_call
        self.interrupt_call = interrupt_call

    def classify_batch(self, texts):
        self.calls.append(texts)
        if len(self.calls) == self.interrupt_call:
            raise KeyboardInterrupt
        if len(self.calls) == self.fail_call:
            return dict(status="failed", predictions=[None] * len(texts), error="temporary outage",
                        attempt=1, prompt_tokens=7, output_tokens=0)
        return success(len(texts))

    def classify(self, text):
        result = self.classify_batch([text])
        if result["status"] != "success":
            raise OSError("temporary outage")
        return dict(prediction=result["predictions"][0], positive_score=0.5, negative_score=0.)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()

    def tearDown(self):
        self.quiet.__exit__(None, None, None)

    def test_each_of_six_systems_resumes_after_interrupt_without_repeating_saved_rows(self):
        for system in SYSTEMS:
            with self.subTest(system=system), TemporaryDirectory() as directory:
                first = FakeClassifier(interrupt_call=2)
                with CheckpointStore(directory, {"code": "v1"}, CASES) as store:
                    with self.assertRaises(KeyboardInterrupt):
                        evaluate(CASES, store, SETTINGS, systems=[system], factory=lambda *_: first)
                    saved = set(store.states(system))
                    self.assertEqual(saved, {0, 1} if system in LLM_SYSTEMS else {0})
                second = FakeClassifier()
                with CheckpointStore(directory, {"code": "v1"}, CASES) as store:
                    summary = evaluate(CASES, store, SETTINGS, systems=[system], factory=lambda *_: second)
                    self.assertEqual(summary[system]["success"], len(CASES))
                    self.assertEqual(summary[system]["usage"]["interrupted_attempts"], 1)
                called = [text for batch in second.calls for text in batch]
                self.assertEqual(called, [case["text"] for i, case in enumerate(CASES) if i not in saved])
                never = Mock(side_effect=AssertionError("Completed system must not initialize"))
                with CheckpointStore(directory, {"code": "v1"}, CASES) as store:
                    evaluate(CASES, store, SETTINGS, systems=[system], factory=never)
                never.assert_not_called()

    def test_failed_only_checkpoint_recovers_and_keeps_previous_usage(self):
        with TemporaryDirectory() as directory:
            with CheckpointStore(directory, {}, CASES[:2]) as store:
                attempt = store.start("llm", [0, 1], False)
                store.finish(attempt, dict(status="failed", predictions=[None, None], error="outage",
                                          attempt=2, prompt_tokens=7, output_tokens=1), 4.0)
            fake = FakeClassifier()
            with CheckpointStore(directory, {}, CASES[:2]) as store:
                summary = evaluate(CASES[:2], store, SETTINGS, systems=["llm"], factory=lambda *_: fake)
                self.assertEqual(fake.calls, [["yorum 0"], ["yorum 1"]])
                usage = summary["llm"]["usage"]
                self.assertEqual(usage["requests"], 4)
                self.assertEqual(usage["prompt_tokens"], 27)
                self.assertGreaterEqual(usage["seconds"], 4.0)
                self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0], 3)

    def test_interrupted_recovery_does_not_repeat_recovered_success(self):
        with TemporaryDirectory() as directory:
            fake = FakeClassifier(fail_call=1, interrupt_call=3)
            with CheckpointStore(directory, {}, CASES[:2]) as store:
                with self.assertRaises(KeyboardInterrupt):
                    evaluate(CASES[:2], store, SETTINGS, systems=["llm"], factory=lambda *_: fake)
                self.assertEqual(store.states("llm"), {0: "success", 1: "failed"})
            resumed = FakeClassifier()
            with CheckpointStore(directory, {}, CASES[:2]) as store:
                evaluate(CASES[:2], store, SETTINGS, systems=["llm"], factory=lambda *_: resumed)
                self.assertEqual(resumed.calls, [["yorum 1"]])

    def test_failures_remain_unresolved_and_retry_is_bounded(self):
        fake = Mock()
        fake.classify_batch.return_value = dict(status="failed", predictions=[None], error="offline")
        with TemporaryDirectory() as directory, CheckpointStore(directory, {}, CASES[:1]) as store:
            summary = evaluate(CASES[:1], store, SETTINGS, systems=["llm"], factory=lambda *_: fake)
            self.assertEqual(fake.classify_batch.call_count, 2)
            self.assertEqual(summary["llm"]["failed"], 1)
            self.assertIsNone(summary["llm"]["accuracy_on_successful"])
            self.assertFalse(summary["llm"]["complete"])

    def test_batch_commit_rolls_back_if_any_row_write_fails(self):
        with TemporaryDirectory() as directory, CheckpointStore(directory, {}, CASES[:2]) as store:
            attempt = store.start("llm", [0, 1], False)
            store.connection.execute("""CREATE TEMP TRIGGER fail_write BEFORE INSERT ON predictions
                WHEN NEW.row_id=1 BEGIN SELECT RAISE(ABORT,'simulated disk/write failure'); END""")
            with self.assertRaises(sqlite3.IntegrityError):
                store.finish(attempt, success(2), 1.0)
            self.assertEqual(store.states("llm"), {})
            self.assertEqual(store.connection.execute("SELECT status FROM attempts").fetchone()[0], "running")

    def test_hard_process_exit_preserves_commits_and_discards_partial_transaction(self):
        script = """
import os, sys
from src.evaluation.checkpoint_store import CheckpointStore
cases = [dict(text='yorum 0', gold='positive'), dict(text='yorum 1', gold='positive')]
store = CheckpointStore(sys.argv[1], {}, cases)
first = store.start('hisnet', [0], False)
store.finish(first, dict(status='success', predictions=['positive'], error=None), 1.0)
second = store.start('hisnet', [1], False)
store.connection.execute("UPDATE attempts SET status='success' WHERE id=?", (second,))
store.connection.execute("INSERT INTO predictions VALUES ('hisnet',1,'success','positive',NULL,?)", (second,))
os._exit(9)
"""
        with TemporaryDirectory() as directory:
            process = subprocess.run([sys.executable, "-c", script, directory],
                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 9, process.stderr)
            with CheckpointStore(directory, {}, CASES[:2]) as store:
                self.assertEqual(store.states("hisnet"), {0: "success"})
                fake = FakeClassifier()
                evaluate(CASES[:2], store, SETTINGS, systems=["hisnet"], factory=lambda *_: fake)
                self.assertEqual(fake.calls, [["yorum 1"]])

    def test_mismatched_dataset_order_labels_and_code_are_rejected(self):
        with TemporaryDirectory() as directory:
            with CheckpointStore(directory, {"code": "v1"}, CASES) as store:
                attempt = store.start("hisnet", [0], False)
                store.finish(attempt, success(1), 1.0)
            changes = [({"code": "v2"}, CASES), ({"code": "v1"}, list(reversed(CASES))),
                       ({"code": "v1"}, [dict(CASES[0], gold="negative"), *CASES[1:]])]
            for metadata, cases in changes:
                with self.assertRaises(ValueError):
                    CheckpointStore(directory, metadata, cases)
            self.assertEqual(read_status(directory)["hisnet"]["success"], 1)

    def test_model_digest_change_is_rejected(self):
        with TemporaryDirectory() as directory, CheckpointStore(directory, {}, CASES) as store:
            store.ensure_identity("ollama_identity", {"digest": "first"})
            with self.assertRaises(ValueError):
                store.ensure_identity("ollama_identity", {"digest": "changed"})

    def test_same_output_cannot_have_two_writers_and_lock_is_released(self):
        with TemporaryDirectory() as directory:
            with CheckpointStore(directory, {}, CASES):
                with self.assertRaises(RuntimeError):
                    CheckpointStore(directory, {}, CASES)
            with CheckpointStore(directory, {}, CASES) as store:
                self.assertEqual(store.states("hisnet"), {})

    def test_status_cli_does_not_call_network_or_construct_models(self):
        from scripts.run_six_systems import main
        with TemporaryDirectory() as directory:
            with CheckpointStore(directory, {}, CASES):
                pass
            with patch("scripts.run_six_systems.ollama_identity", side_effect=AssertionError("network")), \
                    patch("src.evaluation.resumable_evaluator.make_classifier", side_effect=AssertionError("model")):
                self.assertEqual(main(["--output", directory, "--status"]), 0)

    def test_cli_custom_dataset_runs_all_six_and_second_run_skips_all_models(self):
        from scripts.run_six_systems import main
        with TemporaryDirectory() as directory:
            csv_path = Path(directory) / "dataset.csv"
            csv_path.write_text("review;score\nİyi ürün;positive\nGüzel film;positive\n", encoding="utf-8")
            args = ["--csv", str(csv_path), "--text-column", "review", "--label-column", "score",
                    "--output", str(Path(directory) / "output")]
            with patch("scripts.run_six_systems.resource_manifest", return_value=[]), \
                    patch("scripts.run_six_systems.implementation_fingerprint", return_value="code-v1"), \
                    patch("scripts.run_six_systems.ollama_identity", return_value={"digest": "model-v1"}) as identity, \
                    patch("src.evaluation.resumable_evaluator.make_classifier", side_effect=lambda *_: FakeClassifier()) as factory:
                self.assertEqual(main(args), 0)
                self.assertEqual(factory.call_count, 6)
                self.assertEqual(identity.call_count, 1)
                self.assertEqual(main(args), 0)
                self.assertEqual(factory.call_count, 6)
                self.assertEqual(identity.call_count, 1)
            status = read_status(Path(directory) / "output")
            self.assertTrue(all(status[s]["success"] == 2 for s in SYSTEMS))

    def test_csv_export_preserves_turkish_text_and_multiline_reviews(self):
        import csv
        cases = [dict(text='İyi; "güzel"\nürün', gold="positive")]
        with TemporaryDirectory() as directory, CheckpointStore(directory, {}, cases) as store:
            attempt = store.start("hisnet", [0], False)
            store.finish(attempt, success(1), 1.0)
            store.export()
            with (Path(directory) / "predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
                self.assertEqual(list(csv.DictReader(stream))[0]["text"], cases[0]["text"])


class LLMRuntimeTests(unittest.TestCase):
    def classifier(self, dependency=False, **kwargs):
        with patch("src.llm.ollama_zero_shot_classifier.ollama.Client") as client, \
                patch("src.llm.dependency_ollama_zero_shot_classifier.TurkishDependencyParser") as parser:
            parser.return_value.parse.return_value = []
            cls = DependencyOllamaZeroShotSentimentClassifier if dependency else OllamaZeroShotSentimentClassifier
            classifier = cls(max_retries=0, **kwargs)
            client.assert_called_once_with(host="http://localhost:11434", timeout=kwargs.get("request_timeout", 600.0))
        return classifier

    def test_both_clients_have_timeout_and_handle_timeout_failure(self):
        for dependency in (False, True):
            with self.subTest(dependency=dependency):
                classifier = self.classifier(dependency, request_timeout=12)
                classifier.client.chat.side_effect = httpx.ReadTimeout("timed out")
                with contextlib.redirect_stdout(io.StringIO()):
                    result = classifier.classify_batch(["İyi ürün"])
                self.assertEqual(result["predictions"], [None])
                self.assertEqual(result["status"], "failed")

    def test_long_prompts_are_rejected_without_model_call_or_truncation(self):
        for dependency in (False, True):
            classifier = self.classifier(dependency)
            result = classifier.classify_batch(["uzun yorum " * 1000])
            self.assertEqual(result["error_code"], "context_budget")
            self.assertEqual(result["attempt"], 0)
            classifier.client.chat.assert_not_called()
            if dependency:
                self.assertIsNone(classifier._active_dependency_parses)

    def test_length_limited_valid_json_is_not_accepted(self):
        classifier = self.classifier()
        classifier.client.chat.return_value = SimpleNamespace(
            message=SimpleNamespace(content='{"predictions":[{"id":1,"label":"positive"}]}'),
            done_reason="length", prompt_eval_count=10, eval_count=20)
        with contextlib.redirect_stdout(io.StringIO()):
            result = classifier.classify_batch(["İyi ürün"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["output_tokens"], 20)
        self.assertEqual(classifier.client.chat.call_args.kwargs["options"]["num_predict"], 192)

    def test_missing_response_message_is_recoverable(self):
        classifier = self.classifier()
        classifier.client.chat.return_value = SimpleNamespace(message=None)
        with contextlib.redirect_stdout(io.StringIO()):
            result = classifier.classify_batch(["yorum"])
        self.assertEqual(result["predictions"], [None])

    def test_invalid_timeouts_rejected(self):
        for timeout in (None, 0, -1, float("inf"), float("nan"), True):
            with self.assertRaises(ValueError):
                OllamaZeroShotSentimentClassifier(request_timeout=timeout)


if __name__ == "__main__":
    unittest.main()
