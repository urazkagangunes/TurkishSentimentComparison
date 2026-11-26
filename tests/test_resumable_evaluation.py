"""Offline regressions for checkpointing and model response contracts."""
import contextlib
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx

from src.evaluation.checkpoint_store import CheckpointStore, DEFAULT_SYSTEMS, SYSTEMS, read_status
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

    def test_each_system_resumes_after_interrupt_without_repeating_saved_rows(self):
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

    def test_paper_profile_selects_six_datasets_and_all_seven_systems(self):
        from scripts.run_six_systems import PAPER_DATASETS, parse_args

        with patch.dict(os.environ, {"OLLAMA_MODEL": "some-other-model"}):
            args = parse_args(["--paper", "--output", "paper"])
        self.assertEqual(list(PAPER_DATASETS), [
            "beyazperde_sentiment", "HUMIRSentimentDatasets", "TRSAv1_fixed",
            "combined_sentiment", "tr_polarity_combined", "Turkish_twitter_combined",
        ])
        self.assertEqual(args.systems, SYSTEMS)
        self.assertEqual(args.jev_provider, "opencode")
        self.assertEqual(args.model, "qwen3:32b")
        self.assertIsNone(args.sample_per_class)

    def test_paper_scores_only_common_rows_and_penalizes_neutral_on_binary_data(self):
        from src.evaluation.paper_checkpoint_metrics import read_paper_dataset

        cases = [dict(text="kötü", gold="negative"), *CASES[:2]]
        with TemporaryDirectory() as directory, CheckpointStore(directory, {}, cases) as store:
            for system in SYSTEMS:
                expected = ["neutral", "positive"] if system == "hisnet" else ["negative", "positive"]
                attempt = store.start(system, [0, 1], False)
                store.finish(attempt, dict(success(2), predictions=expected), 0.1)
                if system != "jev":
                    attempt = store.start(system, [2], False)
                    store.finish(attempt, success(1), 0.1)

            paper = read_paper_dataset(directory)
            self.assertEqual(paper["macro_f1_labels"], ["negative", "positive"])
            self.assertEqual(paper["common_success"], 2)
            self.assertEqual(paper["common_coverage"], 2 / 3)
            self.assertEqual(paper["systems"]["hisnet"]["success"], 3)
            self.assertEqual(paper["systems"]["hisnet"]["accuracy_on_common"], 0.5)
            self.assertEqual(paper["systems"]["hisnet"]["macro_f1_on_common"], 0.5)
            self.assertEqual(paper["systems"]["hisnet"]["confusion_matrix_on_common"][0][1], 1)
            self.assertEqual(paper["systems"]["jev"]["macro_f1_on_common"], 1.0)

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
            self.assertTrue(all(status[s]["success"] == 2 for s in DEFAULT_SYSTEMS))
            self.assertEqual(status["jev"]["pending"], 2)

    def test_jev_is_added_to_the_same_checkpoint_and_preserves_diagnostics(self):
        from scripts.run_six_systems import main
        from src.llm.jev_zero_shot_classifier import DEFAULT_JEV_MODEL, TYPESAFE_ENDPOINT

        probabilities = dict(positive=0.8, neutral=0.1, negative=0.1)

        def reply(request):
            self.assertEqual(str(request.url), TYPESAFE_ENDPOINT)
            self.assertEqual(request.headers["Authorization"], "Bearer test-key")
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], DEFAULT_JEV_MODEL)
            self.assertEqual(payload["state"], "İyi ürün")
            self.assertEqual(set(payload), {"model", "state", "questions"})
            question = payload["questions"]["sentiment"]
            self.assertEqual(question["type"], "choice")
            self.assertEqual(question["criteria"].keys(), probabilities.keys())
            return httpx.Response(200, json={
                "model": DEFAULT_JEV_MODEL,
                "answers": {"sentiment": {"type": "choice", "choice": "positive",
                                           "probabilities": probabilities, "confidence": 0.7}},
                "usage": {"input_tokens": 20, "output_tokens": 3},
            })

        with TemporaryDirectory() as directory:
            csv_path = Path(directory) / "dataset.csv"
            csv_path.write_text("review;score\nİyi ürün;positive\n", encoding="utf-8")
            output = Path(directory) / "output"
            args = ["--csv", str(csv_path), "--text-column", "review", "--label-column", "score",
                    "--output", str(output)]
            client = httpx.Client(transport=httpx.MockTransport(reply),
                                  headers={"Authorization": "Bearer test-key"})
            from src.evaluation.resumable_evaluator import make_classifier

            def factory(system, config):
                return make_classifier(system, config) if system == "jev" else FakeClassifier()

            with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-key"}), \
                    patch("scripts.run_six_systems.resource_manifest", return_value=[]), \
                    patch("scripts.run_six_systems.implementation_fingerprint", return_value="code-v1"), \
                    patch("scripts.run_six_systems.ollama_identity", return_value={"digest": "ollama-v1"}), \
                    patch("src.llm.jev_zero_shot_classifier.httpx.Client", return_value=client) as constructor, \
                    patch("src.evaluation.resumable_evaluator.make_classifier", side_effect=factory) as make:
                self.assertEqual(main(args), 0)
                constructor.assert_not_called()
                self.assertEqual(main([*args, "--systems", "all"]), 0)
                self.assertEqual(make.call_count, 7)
                self.assertEqual(main([*args, "--systems", "jev"]), 0)
                with self.assertRaisesRegex(ValueError, "jev_identity differs"):
                    main([*args, "--systems", "jev", "--jev-model", "jev-1.13.0"])
                constructor.assert_called_once_with(headers={"Authorization": "Bearer test-key"}, timeout=30.0)
            status = read_status(output)
            self.assertTrue(all(status[system]["complete"] for system in SYSTEMS))
            self.assertEqual(status["jev"]["usage"]["prompt_tokens"], 20)
            self.assertEqual(status["jev"]["usage"]["requests"], 1)
            self.assertNotIn("test-key", (output / "jev_identity.json").read_text(encoding="utf-8"))
            with contextlib.closing(sqlite3.connect(output / "progress.sqlite3")) as connection:
                result = json.loads(connection.execute(
                    "SELECT result FROM attempts WHERE system='jev'"
                ).fetchone()[0])
            self.assertEqual(result["probabilities"], probabilities)
            self.assertEqual(result["confidence"], 0.7)

    def test_jev_global_failure_is_saved_and_stops_new_requests(self):
        from src.llm.jev_zero_shot_classifier import JevZeroShotSentimentClassifier

        reply = Mock(return_value=httpx.Response(401))
        client = httpx.Client(transport=httpx.MockTransport(reply))
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-key"}), \
                patch("src.llm.jev_zero_shot_classifier.httpx.Client", return_value=client):
            classifier = JevZeroShotSentimentClassifier()
        with TemporaryDirectory() as directory, CheckpointStore(directory, {}, CASES) as store:
            with self.assertRaisesRegex(RuntimeError, "TypeSafe HTTP 401"):
                evaluate(CASES, store, {}, systems=["jev"], factory=lambda *_: classifier)
            self.assertEqual(store.states("jev"), {0: "failed"})
            self.assertEqual(store.export()["jev"]["usage"]["requests"], 1)
        reply.assert_called_once()

    def test_jev_rate_limit_retries_and_rejects_an_unknown_label(self):
        from src.llm.jev_zero_shot_classifier import DEFAULT_JEV_MODEL, JevZeroShotSentimentClassifier

        reply = Mock(side_effect=[
            httpx.Response(429, headers={"Retry-After": "3"}),
            httpx.Response(200, json={
                "model": DEFAULT_JEV_MODEL,
                "answers": {"sentiment": {"type": "choice", "choice": "unknown",
                                           "probabilities": {}, "confidence": 0.7}},
                "usage": {"input_tokens": 20, "output_tokens": 3},
            }),
        ])
        with httpx.Client(transport=httpx.MockTransport(reply)) as client, \
                patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-key"}), \
                patch("src.llm.jev_zero_shot_classifier.httpx.Client", return_value=client), \
                patch("src.llm.jev_zero_shot_classifier.time.sleep") as sleep:
            result = JevZeroShotSentimentClassifier().classify("yorum")
        sleep.assert_called_once_with(3)
        self.assertEqual(result["attempt"], 2)
        self.assertEqual(result["prompt_tokens"], 20)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["prediction"])

    def test_opencode_jev_accepts_anonymous_free_result_and_saves_billing(self):
        from src.llm.jev_zero_shot_classifier import JEV_ENDPOINTS, JevZeroShotSentimentClassifier

        def reply(request):
            self.assertEqual(str(request.url), JEV_ENDPOINTS["opencode"])
            self.assertNotIn("Authorization", request.headers)
            self.assertEqual(json.loads(request.content)["model"], "jev-1.13-free")
            return httpx.Response(200, json={
                "model": "jev-1.13-free",
                "answers": {"sentiment": {"type": "choice", "choice": "positive",
                                           "probabilities": dict(positive=0.8, neutral=0.1, negative=0.1),
                                           "confidence": 0.7}},
                "usage": {"input_tokens": 20, "output_tokens": 3, "cost": 0,
                          "opencode": {"billing_source": "free"}},
            })

        client = httpx.Client(transport=httpx.MockTransport(reply))
        with patch.dict(os.environ, {"OPENCODE_API_KEY": "", "TYPESAFE_API_KEY": "other-provider-key"}), \
                patch("src.llm.jev_zero_shot_classifier.httpx.Client", return_value=client) as constructor:
            classifier = JevZeroShotSentimentClassifier(provider="opencode")
        constructor.assert_called_once_with(headers={}, timeout=30.0)
        with TemporaryDirectory() as directory, CheckpointStore(directory, {}, CASES[:1]) as store:
            summary = evaluate(CASES[:1], store, {}, systems=["jev"], factory=lambda *_: classifier)
            self.assertTrue(summary["jev"]["complete"])
            result = json.loads(store.connection.execute("SELECT result FROM attempts").fetchone()[0])
            self.assertEqual(result["billing_source"], "free")
            self.assertEqual(result["cost"], 0)

    def test_opencode_jev_does_not_continue_if_free_usage_is_unconfirmed(self):
        from src.llm.jev_zero_shot_classifier import JevZeroShotSentimentClassifier

        reply = Mock(return_value=httpx.Response(200, json={
            "model": "jev-1.13-free",
            "usage": {"input_tokens": 20, "output_tokens": 3, "cost": 0.01,
                      "opencode": {"billing_source": "balance"}},
        }))
        client = httpx.Client(transport=httpx.MockTransport(reply))
        with patch.dict(os.environ, {"OPENCODE_API_KEY": ""}), \
                patch("src.llm.jev_zero_shot_classifier.httpx.Client", return_value=client) as constructor:
            with self.assertRaisesRegex(ValueError, "only supports jev-1.13-free"):
                JevZeroShotSentimentClassifier(provider="opencode", model="jev-1.13")
            constructor.assert_not_called()
            classifier = JevZeroShotSentimentClassifier(provider="opencode")
        with TemporaryDirectory() as directory, CheckpointStore(directory, {}, CASES) as store:
            with self.assertRaisesRegex(RuntimeError, "did not confirm free usage"):
                evaluate(CASES, store, {}, systems=["jev"], factory=lambda *_: classifier)
            self.assertEqual(store.states("jev"), {0: "failed"})
        reply.assert_called_once()

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
