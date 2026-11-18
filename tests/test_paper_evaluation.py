"""Offline paper-run regressions: no real model loads, downloads or training."""
import contextlib
import csv
import io
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support

from scripts import run_paper_suite as cli
from src.evaluation.checkpoint_store import CheckpointStore, SYSTEMS, LABELS
from src.evaluation.paper_datasets import catalog, load_dataset
from src.evaluation.paper_diagnostics import text_flags, diagnostics
from src.evaluation.paper_metrics import metrics, paired, bootstrap, holm
from src.evaluation.paper_report import analyze_dataset
from src.evaluation.paper_runner import StageTimer, evaluate_dataset, prepare_metadata
from src.evaluation.paper_suite import suite_lock, suite_report
from src.evaluation.resumable_evaluator import LLM_SYSTEMS


def run_paper_group(argv, systems, output_name):
    # Keep group selection fixed while forwarding the paper runner's other options.
    output = cli.ROOT / "results" / output_name
    return cli.main(["--output", str(output), *argv, "--systems", *systems])


def run_test_file(run_paper):
    if sys.argv[1:2] == ["--live"]:
        raise SystemExit(run_paper(sys.argv[2:]))
    unittest.main()


def fixtures(n=12):
    return [dict(text=f"yorum {i}", gold=LABELS[i % 3], group_id=str(i), source_ids=[str(i+1)],
                 source_groups=[], split="test", source_label_conflict=False) for i in range(n)]


def metadata(systems, n=12):
    return dict(paper=dict(dataset=dict(id="fixture", include_in_aggregate=True), systems=list(systems),
                           validation=dict(eligible_rows=n)))


class FakeClassifier:
    def __init__(self, interrupt=None, fail=False, malformed=False):
        self.calls, self.interrupt, self.fail, self.malformed = [], interrupt, fail, malformed

    def classify_batch(self, texts):
        self.calls.append(list(texts))
        if len(self.calls) == self.interrupt:
            raise KeyboardInterrupt
        if self.fail:
            raise TimeoutError("offline fixture timeout")
        predictions = [LABELS[int(text.split()[-1]) % 3] for text in texts]
        return dict(status="success", predictions=["invalid"]*len(texts) if self.malformed else predictions,
                    error=None, attempt=1, prompt_tokens=10, output_tokens=3)

    def classify(self, text):
        result = self.classify_batch([text])
        return dict(prediction=result["predictions"][0], positive_score=.5, negative_score=0,
                    morphological_analyses=[dict(analysis_source="normal")])


class MetricTests(unittest.TestCase):
    def test_metrics_match_independent_sklearn_three_and_two_class(self):
        rng = np.random.default_rng(19)
        for active in ([0,1,2], [0,2]):
            y = rng.choice(active, 151)
            pred = rng.integers(0,3,151)
            result = metrics(y,pred,active)
            self.assertAlmostEqual(result["accuracy"],accuracy_score(y,pred))
            self.assertAlmostEqual(result["mcc"],matthews_corrcoef(y,pred))
            p,r,f,_ = precision_recall_fscore_support(y,pred,labels=active,average="macro",zero_division=0)
            for key,value in (("macro_precision",p),("macro_recall",r),("macro_f1",f)):
                self.assertAlmostEqual(result[key], value)
            _,_,f,_ = precision_recall_fscore_support(y,pred,average="weighted",zero_division=0)
            self.assertAlmostEqual(result["weighted_f1"],f)
        self.assertIsNone(result["classes"]["neutral"]["recall"])
        self.assertGreater(result["classes"]["neutral"]["fp"],0)

    def test_missing_output_is_not_neutral_or_in_denominator(self):
        result = metrics(np.array([0,1,2]),np.array([0,-1,1]))
        self.assertEqual(result["n"],2)
        self.assertEqual(result["accuracy"],.5)
        self.assertEqual(result["classes"]["neutral"]["fp"],1)
        self.assertIsNone(metrics([-1,-1],[0,1])["accuracy"])

    def test_neutral_prediction_rate_uses_valid_outputs_not_gold_support(self):
        result = metrics([0, 0, 2, 2], [1, 0, 2, -1])
        self.assertEqual(result["neutral_prediction_count"], 1)
        self.assertEqual(result["prediction_count"], 3)
        self.assertAlmostEqual(result["neutral_prediction_rate"], 1/3)
        self.assertEqual(result["classes"]["neutral"]["support"], 0)
        self.assertIsNone(result["classes"]["neutral"]["recall"])
        self.assertAlmostEqual(result["accuracy"], 2/3)
        for predictions, expected in (([-1, -1], None), ([0, 2], 0), ([1, 1], 1)):
            self.assertEqual(metrics([0, 2], predictions)["neutral_prediction_rate"], expected)
        self.assertIsNone(metrics([], [])["neutral_prediction_rate"])

    def test_neutral_prediction_rate_is_available_without_gold(self):
        result = metrics([-1, -1, -1], [1, 2, -1])
        self.assertEqual(result["n"], 0)
        self.assertIsNone(result["accuracy"])
        self.assertEqual(result["prediction_count"], 2)
        self.assertEqual(result["neutral_prediction_rate"], .5)

    def test_paired_categories_and_bootstrap_difference(self):
        y=np.array([0,1,2,0,1])
        a=np.array([0,0,1,1,1])
        b=np.array([1,1,0,1,1])
        result=paired(y,a,b)
        self.assertEqual([result[k] for k in ("harmed","improved","both_wrong_changed","both_wrong_unchanged","both_correct")],[1]*5)
        self.assertEqual(result["net_correct"],0)
        ci=bootstrap(y,a,[0,0,1,2,2],[0,1,2],100,42,second=a)
        self.assertEqual(ci["clusters"],3)
        self.assertTrue(all(v==[0.,0.] for v in ci["intervals"].values()))
        self.assertEqual(ci,bootstrap(y,a,[0,0,1,2,2],[0,1,2],100,42,second=a))
        better=bootstrap([0,1,2],[1,2,0],[0,1,2],[0,1,2],100,42,second=[0,1,2])
        self.assertEqual(better["intervals"]["accuracy"],[1.,1.])
        self.assertIsNone(bootstrap([],[],[],[],100))

    def test_holm(self):
        rows=[dict(mcnemar_exact_p=p) for p in (.01,.04,.03,None)]
        holm(rows)
        self.assertEqual([p.get("holm_adjusted_p") for p in rows],[.03,.06,.06,None])


class DatasetTests(unittest.TestCase):
    def test_provenance_conflicts_and_transitive_groups_after_dedup(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"d.csv").write_text("review;score;group\nİYİ!;positive;a\nİYİ!;positive;b\nİYİ!;negative;a\niyi;positive;c\nfarklı;neutral;b\n;neutral;c\n",encoding="utf-8")
            spec=dict(id="d",path="d.csv",enabled=True,adapter="csv",task="sentiment",text_column="review",label_column="score",group_column="group")
            cases,info=load_dataset(root,spec)
            self.assertEqual(len(cases),4)
            self.assertEqual(info["duplicate_pairs_removed"],1)
            self.assertEqual(info["conflicting_texts"],1)
            self.assertEqual(info["bootstrap_groups"],1)
            self.assertEqual(cases[0]["source_ids"],["1","2"])
            self.assertEqual(cases[0]["source_groups"],["a","b"])

    def test_no_numeric_guess_and_complete_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"a.csv").write_text("text;label\niyi;3\n",encoding="utf-8")
            (root/"b.txt").write_text("iyi",encoding="utf-8")
            spec=dict(id="a",path="a.csv",enabled=True,adapter="csv",task="sentiment",text_column="text",label_column="label")
            self.assertEqual(len(catalog(root,[spec])),2)
            with self.assertRaisesRegex(ValueError,"Unmapped label"):
                load_dataset(root,spec)
            with self.assertRaisesRegex(ValueError,"mapping_source"):
                catalog(root,[dict(spec,label_map={"3":"neutral"})])
            cases,_=load_dataset(root,dict(spec,label_map={"3":"neutral"},mapping_source="fixture"))
            self.assertEqual(cases[0]["gold"],"neutral")
            with self.assertRaisesRegex(ValueError,"Unsafe dataset"):
                catalog(root,[dict(spec,id="../bad")])

    def test_tree_and_unlabeled_adapters(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"t.txt").write_text("(2 (1 çok) (2 güzel))\n",encoding="utf-8")
            cases,_=load_dataset(root,dict(path="t.txt",adapter="tree",task="sentiment",label_map={"2":"positive"}))
            self.assertEqual((cases[0]["text"],cases[0]["gold"]),("çok güzel","positive"))
            (root/"u.txt").write_text("etiketsiz yorum\n",encoding="utf-8")
            cases,_=load_dataset(root,dict(path="u.txt",adapter="lines",task="inference"))
            self.assertEqual(cases[0]["gold"],"unlabeled")

    def test_proxy_not_gold_and_diagnostics(self):
        self.assertTrue(text_flags("güzel değil ama kötü değil")["auto_multiple_negation"])
        self.assertTrue(text_flags("tavsiye etmiyorum")["auto_any_negation"])
        self.assertFalse(text_flags("güzel bir ürün")["auto_any_negation"])
        d=diagnostics(dict(positive_score=0,negative_score=0,scopes=[dict(operations=["negation_flip"],alignment_status="unmatched")]))
        self.assertTrue(d["flags"]["zero_score"])
        self.assertTrue(d["flags"]["negation_flip"])
        self.assertTrue(text_flags("çooook iyi!!!")["surface_noise_proxy"])
        self.assertFalse(text_flags("ürün güzel")["surface_noise_proxy"])

    def test_multiline_csv_and_case_insensitive_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"d.csv").write_text('Review;Score\n"İyi; ürün\nikinci satır";Positive\n', encoding="utf-8-sig", newline="")
            cases, _ = load_dataset(root, dict(path="d.csv", adapter="csv", task="sentiment",
                                               text_column="review", label_column="score"))
            self.assertEqual(cases[0]["text"], "İyi; ürün\nikinci satır")
            self.assertEqual(cases[0]["gold"], "positive")


class LLMRunnerChecks:
    def test_failure_and_retry_summaries_preserve_history_and_count_batches_once(self):
        with tempfile.TemporaryDirectory() as directory:
            clean_system = self.systems[0]
            systems, cases = [clean_system, "dp_llm"], fixtures(8)

            def finish(store, ids, status, *, recovery=False, **payload):
                aid = store.start("dp_llm", ids, recovery)
                predictions = ["neutral"] * len(ids) if status == "success" else [None] * len(ids)
                store.finish(aid, dict(status=status, predictions=predictions, **payload), .01)

            with CheckpointStore(directory, metadata(systems, 8), cases) as store:
                prepare_metadata(store, cases)
                for i, case in enumerate(cases):
                    aid = store.start(clean_system, [i], False)
                    result = dict(status="success", predictions=[case["gold"]])
                    if clean_system in LLM_SYSTEMS:
                        result["attempt"] = 1
                    store.finish(aid, result, .01)
                finish(store, [0, 1], "failed", error="TimeoutError: timed out", attempt=3)
                finish(store, [0], "success", recovery=True, attempt=1)
                finish(store, [1], "failed", recovery=True, error="bad JSON", error_code="invalid_response", attempt=2)
                finish(store, [2], "failed", error="too long", error_code="context_budget", attempt=0)
                finish(store, [3], "failed", error="Dependency parse RuntimeError", attempt=0)
                finish(store, [4], "failed", error="JSONDecodeError", error_code="classifier_exception", attempt=1)
                store.start("dp_llm", [5], False)
                finish(store, [7], "failed", error="server unavailable", error_code="server_unavailable")
                with store.connection:
                    store.connection.execute("INSERT INTO paper_initializations VALUES(NULL,?,?,?,?,?)",
                        ("dp_llm", "fixture", .01, "failed", "TimeoutError: init timeout"))
            with CheckpointStore(directory, metadata(systems, 8), cases) as store:
                aid = store.start("dp_llm", [5], False)
                store.finish(aid, dict(status="success", predictions=["positive"], attempt=1), .01)
                store.start("dp_llm", [6], False)

            def checkpoint_rows():
                uri = (Path(directory)/"progress.sqlite3").resolve().as_uri() + "?mode=ro"
                with contextlib.closing(sqlite3.connect(uri, uri=True)) as db:
                    return {table:db.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
                            for table in ("attempts", "predictions", "metadata", "paper_initializations")}

            before = checkpoint_rows()
            report = analyze_dataset(directory, 0, emit=lambda _:None)
            errors = {r["error_code"]:r for r in report["failure_summary"] if r["system"] == "dp_llm"}
            self.assertEqual(errors["timeout"], dict(system="dp_llm", error_code="timeout", current_failed_rows=0,
                failed_attempts=1, failed_row_occurrences=2, initialization_failures=1))
            self.assertEqual(errors["invalid_response"]["current_failed_rows"], 2)
            self.assertEqual(errors["invalid_response"]["failed_attempts"], 2)
            for code in ("context_budget", "parse_error", "server_unavailable"):
                self.assertEqual(errors[code]["current_failed_rows"], 1)
                self.assertEqual(errors[code]["failed_attempts"], 1)
            self.assertEqual(sum(r["current_failed_rows"] for r in errors.values()), 5)
            self.assertEqual(sum(r["failed_attempts"] for r in errors.values()), 6)
            self.assertEqual(sum(r["failed_row_occurrences"] for r in errors.values()), 7)
            retry = next(r for r in report["retry_summary"] if r["system"] == "dp_llm")
            self.assertEqual(retry, dict(system="dp_llm", recorded_attempts=10, interrupted_attempts=1,
                running_attempts=1, repeated_row_attempts=3, retried_unique_rows=3, successful_retried_rows=2,
                recovery_attempts=2, recovery_successful_attempts=1, recovery_failed_attempts=1,
                recovery_interrupted_attempts=0, recovery_running_attempts=0, recovery_row_occurrences=2,
                recovered_rows=1, known_llm_request_attempts=8, known_llm_internal_retries=3,
                llm_request_count_unknown_attempts=3))
            clean = next(r for r in report["retry_summary"] if r["system"] == clean_system)
            self.assertEqual(clean["known_llm_request_attempts"], 8 if clean_system in LLM_SYSTEMS else 0)
            self.assertEqual(clean["known_llm_internal_retries"], 0)
            self.assertEqual(clean["llm_request_count_unknown_attempts"], 0)
            self.assertEqual(clean["recovery_attempts"], 0)
            self.assertTrue(all(r["failed_attempts"] == r["current_failed_rows"] == 0
                                for r in report["failure_summary"] if r["system"] == clean_system))
            self.assertEqual(report["metrics"]["dp_llm"]["successful"]["neutral_prediction_rate"], .5)
            self.assertEqual(report["metrics"][clean_system]["successful"]["neutral_prediction_rate"], 3/8)
            self.assertEqual(report["metrics"][clean_system]["common_success"]["neutral_prediction_rate"], 0)
            for name, key in (("failure_summary", "failure_summary"), ("retry_summary", "retry_summary")):
                with (Path(directory)/(name+".csv")).open(encoding="utf-8-sig", newline="") as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), len(report[key]))
            with (Path(directory)/"metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(all("neutral_prediction_rate" in r for r in rows))
            persisted = json.loads((Path(directory)/"paper_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["retry_summary"], report["retry_summary"])
            repeated = analyze_dataset(directory, 0, emit=lambda _:None)
            self.assertEqual(repeated["failure_summary"], report["failure_summary"])
            self.assertEqual(repeated["retry_summary"], report["retry_summary"])
            self.assertEqual(checkpoint_rows(), before)


class ClassicalRunnerChecks:
    def test_stage_timer_retains_lookup_results_and_restores_methods(self):
        lookup = Mock(side_effect=[None, dict(positive=1, negative=0)])
        morphology = Mock(return_value=[dict(root="beğen")])
        parse = Mock(return_value=[])
        classifier = SimpleNamespace(hisnet=SimpleNamespace(_lookup_root=lookup,
            root_extractor=SimpleNamespace(analyze_sentence=morphology)),
            parser=SimpleNamespace(parse=parse))
        timer = StageTimer(classifier, "hisnet_dp")
        try:
            self.assertIsNone(classifier.hisnet._lookup_root("beğen"))
            self.assertEqual(classifier.hisnet._lookup_root("güzel"), dict(positive=1, negative=0))
            classifier.parser.parse("güzel")
            classifier.hisnet.root_extractor.analyze_sentence("beğen")
            self.assertEqual(set(timer.values), {"parse_seconds", "morphology_seconds", "lexicon_lookup_seconds"})
            d = diagnostics({}, timer.unmatched_roots)
            self.assertEqual(d["unmatched_roots"], {"beğen":1})
            self.assertTrue(d["flags"]["unmatched_opinion_root"])
            parse.side_effect = RuntimeError("parser fixture error")
            with self.assertRaisesRegex(RuntimeError, "parser fixture"):
                classifier.parser.parse("bozuk")
            self.assertEqual(timer.failed_stage, "parse_seconds")
        finally:
            timer.close()
        self.assertIs(classifier.hisnet._lookup_root, lookup)
        self.assertIs(classifier.parser.parse, parse)


class PaperRunnerChecks:
    def run_fake(self, directory, system, classifier):
        cases=fixtures()
        with CheckpointStore(directory,metadata([system]),cases) as store:
            return evaluate_dataset(cases,store,dict(batch_size=3),[system],lambda *_:classifier,"fixture",
                                    progress_every=1,heartbeat_seconds=0,emit=lambda _:None)

    def test_resume_without_repeating_committed_predictions(self):
        for system in self.systems:
            with self.subTest(system=system),tempfile.TemporaryDirectory() as directory:
                first=FakeClassifier(interrupt=3)
                with self.assertRaises(KeyboardInterrupt):
                    self.run_fake(directory,system,first)
                committed={text for batch in first.calls[:-1] for text in batch}
                second=FakeClassifier()
                result=self.run_fake(directory,system,second)
                resumed={text for batch in second.calls for text in batch}
                self.assertFalse(committed & resumed)
                self.assertEqual(committed|resumed,{c["text"] for c in fixtures()})
                self.assertTrue(result["systems"][system]["complete"])
                third=Mock(side_effect=AssertionError("completed model must not initialize"))
                with CheckpointStore(directory,metadata([system]),fixtures()) as store:
                    evaluate_dataset(fixtures(),store,dict(batch_size=3),[system],third,"fixture",heartbeat_seconds=0,emit=lambda _:None)
                third.assert_not_called()
                self.assertEqual(len(list((Path(directory)/"checkpoints").glob("dataset_*.json"))),10)
                self.assertEqual(len(list((Path(directory)/"checkpoints").glob(system+"_*.json"))),10)

    def test_failed_predictions_recover_and_are_not_neutral(self):
        for system in self.systems:
            for fail, malformed in ((True, False), (False, True)):
                with self.subTest(system=system, malformed=malformed), tempfile.TemporaryDirectory() as directory:
                    result = self.run_fake(directory, system, FakeClassifier(fail=fail, malformed=malformed))
                    self.assertEqual(result["systems"][system]["failed"], 12)
                    with contextlib.closing(sqlite3.connect(Path(directory)/"progress.sqlite3")) as db:
                        self.assertEqual(db.execute("SELECT COUNT(*) FROM predictions WHERE prediction IS NOT NULL").fetchone()[0], 0)
                    report = analyze_dataset(directory, 20, emit=lambda _: None)
                    self.assertEqual(report["common_success"], 0)
                    self.assertIsNone(report["metrics"][system]["successful"]["accuracy"])
                    self.assertIsNone(report["metrics"][system]["successful"]["neutral_prediction_rate"])
                    self.assertTrue(self.run_fake(directory, system, FakeClassifier())["systems"][system]["complete"])

    def test_hard_process_exit_preserves_committed_rows(self):
        system = self.systems[0]
        with tempfile.TemporaryDirectory() as directory:
            code="""
import os, sys
from src.evaluation.checkpoint_store import CheckpointStore
s=CheckpointStore(sys.argv[1],{'fixture':1},[{'text':'x','gold':'neutral'}])
a=s.start(sys.argv[2],[0],False)
s.finish(a,{'status':'success','predictions':['neutral'],'error':None},0.1)
os._exit(23)
"""
            process=subprocess.run([sys.executable,"-B","-c",code,directory,system],
                                   cwd=cli.ROOT,capture_output=True,timeout=30)
            self.assertEqual(process.returncode,23,process.stderr)
            with CheckpointStore(directory,{'fixture':1},[dict(text="x",gold="neutral")]) as store:
                self.assertEqual(store.states(system),{0:"success"})

    def test_report_common_cohort_and_source_provenance(self):
        clean_system, failing_system = self.systems[0], self.systems[-1]
        with tempfile.TemporaryDirectory() as directory:
            cases=fixtures()
            with CheckpointStore(directory,metadata(self.systems),cases) as store:
                prepare_metadata(store,cases)
                for system in self.systems:
                    for i,c in enumerate(cases):
                        aid=store.start(system,[i],False)
                        fail=system==failing_system and i==0
                        store.finish(aid,dict(status="failed" if fail else "success", predictions=[None if fail else c["gold"]],error="timeout" if fail else None),.001)
            r=analyze_dataset(directory,50,emit=lambda _:None)
            self.assertEqual(r["common_success"],11)
            self.assertFalse(r["complete"])
            self.assertEqual(set(r["metrics"]), set(self.systems))
            self.assertEqual(r["metrics"][clean_system]["successful"]["n"],12)
            self.assertEqual(r["metrics"][clean_system]["common_success"]["n"],11)
            self.assertEqual(len(r["pairs"]), self.pair_count)
            self.assertTrue((Path(directory)/"provenance.csv").exists())
            with (Path(directory)/"slices.csv").open(encoding="utf-8-sig", newline="") as stream:
                row = next(r for r in csv.DictReader(stream) if r["system"] == failing_system and r["slice"] == "short_1_5")
            self.assertEqual(json.loads(row["gold_distribution"]), {l:4 for l in LABELS})
            self.assertEqual(int(row["evaluated"]), 11)
            with (Path(directory)/"metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
                row = next(r for r in csv.DictReader(stream) if r["system"] == clean_system and r["view"] == "common_success")
            self.assertAlmostEqual(float(row["coverage"]), 11/12)
            self.assertEqual(float(row["system_coverage"]), 1)

    def test_initialization_failure_leaves_pending_rows_and_reported_error(self):
        failing_system, successful_system = self.systems[0], self.systems[-1]
        with tempfile.TemporaryDirectory() as directory:
            systems = [failing_system, successful_system]
            def factory(system, _):
                if system == failing_system:
                    raise ModuleNotFoundError("missing fixture dependency")
                return FakeClassifier()
            with CheckpointStore(directory, metadata(systems), fixtures()) as store:
                progress = evaluate_dataset(fixtures(), store, dict(batch_size=3), systems, factory, "fixture",
                    heartbeat_seconds=0, emit=lambda _:None)
                self.assertEqual(progress["systems"][failing_system]["pending"], 12)
                self.assertTrue(progress["systems"][successful_system]["complete"])
                self.assertEqual(store.states(failing_system), {})
            report = analyze_dataset(directory, 0, emit=lambda _:None)
            self.assertFalse(report["complete"])
            self.assertIn("missing fixture dependency", report["runtime"][0]["initialization_errors"][0])
            with (Path(directory)/"runtime.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            if successful_system in LLM_SYSTEMS:
                self.assertGreater(int(rows[1]["prompt_tokens"]), 0)
            else:
                self.assertEqual(rows[1]["measured_unit"], "review")

    def test_unlabeled_inference_reports_distribution_without_accuracy(self):
        system = "llm" if "llm" in self.systems else self.systems[0]
        with tempfile.TemporaryDirectory() as directory:
            cases = [dict(c, gold="unlabeled") for c in fixtures()]
            with CheckpointStore(directory, metadata([system]), cases) as store:
                evaluate_dataset(cases, store, dict(batch_size=3), [system], lambda *_:FakeClassifier(),
                    "fixture", heartbeat_seconds=0, emit=lambda _:None)
                self.assertIsNone(store.export()[system]["accuracy_on_successful"])
            report = analyze_dataset(directory, 20, emit=lambda _:None)
            self.assertTrue(report["complete"])
            m = report["metrics"][system]["common_success"]
            self.assertIsNone(m["accuracy"])
            self.assertIsNone(m["ci95"])
            self.assertEqual(m["prediction_distribution"], {l:4 for l in LABELS})
            self.assertEqual(m["neutral_prediction_rate"], 1/3)


class RunnerTests(PaperRunnerChecks, LLMRunnerChecks, ClassicalRunnerChecks, unittest.TestCase):
    systems = SYSTEMS
    pair_count = 5


class PaperSuiteChecks:
    def test_suite_accepts_older_reports_without_prediction_rate_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output/"legacy").mkdir()
            old_metrics = metrics([0, 2], [1, 2])
            for key in ("neutral_prediction_count", "prediction_count", "neutral_prediction_rate"):
                del old_metrics[key]
            report = dict(dataset=dict(id="legacy", include_in_aggregate=True), complete=True, total=2,
                          common_success=2, pairs=[], metrics={self.systems[0]:dict(common_success=old_metrics)})
            (output/"legacy/paper_summary.json").write_text(json.dumps(report), encoding="utf-8")
            suite_report(output, dict(datasets=[dict(id="legacy")]))
            with (output/"suite_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(float(row["neutral_prediction_rate"]), .5)
            self.assertEqual(int(row["prediction_count"]), 2)
            self.assertEqual(int(row["neutral_prediction_count"]), 1)
            self.assertEqual(float(row["accuracy"]), .5)

    def test_suite_lock_rejects_second_writer_and_releases_after_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(ValueError, "fixture interrupt"):
                with suite_lock(output):
                    with self.assertRaisesRegex(RuntimeError, "başka bir koşu"):
                        with suite_lock(output):
                            self.fail("Second writer entered")
                    raise ValueError("fixture interrupt")
            with suite_lock(output):
                self.assertTrue((output/"suite.lock").exists())

    def test_prepare_run_resume_analyze_status_and_manifest_mismatch(self):
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            root=Path(directory)/"data"; root.mkdir()
            config=Path(directory)/"config.json"
            specs=[]
            for name in ("first","second"):
                (root/(name+".csv")).write_text("review;score\n"+"\n".join(f"yorum {i};{LABELS[i%3]}" for i in range(12)),encoding="utf-8")
                specs.append(dict(id=name,path=name+".csv",enabled=True,adapter="csv",text_column="review",label_column="score",task="sentiment",include_in_aggregate=True))
            config.write_text(json.dumps(dict(datasets=specs)),encoding="utf-8")
            output=Path(directory)/"out"
            args=["--datasets-root",str(root),"--config",str(config),"--output",str(output),"--bootstrap","20","--heartbeat-seconds","0"]
            expected_systems = list(self.systems) * len(specs)
            with patch.object(cli,"resource_identity",return_value=[]), \
                 patch.object(cli,"ollama_identity",return_value={"model":"fake"}) as identity, \
                 patch.object(cli,"make_classifier",side_effect=lambda *_:FakeClassifier()) as factory:
                self.assertEqual(self.run_cli(args+["--prepare-only"]),0)
                factory.assert_not_called()
                identity.assert_not_called()
                self.assertEqual(self.run_cli(args),0)
                self.assertEqual([call.args[0] for call in factory.call_args_list], expected_systems)
                self.assertEqual(self.run_cli(args),0)
                self.assertEqual(self.run_cli(["--output",str(output),"--status"]),0)
                self.assertEqual(self.run_cli(["--output",str(output),"--analyze-only","--bootstrap","20","--heartbeat-seconds","0"]),0)
                self.assertEqual(self.run_cli(args+["--batch-size","2"]),2)
                self.assertEqual(factory.call_count, len(expected_systems))
                self.assertEqual(identity.call_count, int(bool(LLM_SYSTEMS.intersection(self.systems))))
            summary=json.loads((output/"suite_summary.json").read_text(encoding="utf-8"))
            self.assertEqual({row["system"] for row in summary["aggregate"]}, set(self.systems))
            self.assertEqual(summary["missing"],[])
            with (output/"suite_metrics.csv").open(encoding="utf-8-sig",newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), len(expected_systems))
            with (output/"suite_pairwise.csv").open(encoding="utf-8-sig",newline="") as stream:
                row = next(csv.DictReader(stream))
                self.assertIn("dataset_holm_adjusted_p", row)
                self.assertIn("suite_holm_adjusted_p", row)
            with (output/"dataset_overlap.csv").open(encoding="utf-8-sig",newline="") as stream:
                overlaps=list(csv.DictReader(stream))
            self.assertEqual(int(overlaps[0]["shared_unique_texts"]),12)


class SuiteTests(PaperSuiteChecks, unittest.TestCase):
    systems = SYSTEMS
    run_cli = staticmethod(cli.main)


if __name__ == "__main__":
    unittest.main()
