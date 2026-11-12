"""Offline paper-readiness evidence. No model construction or inference.

Uses only explicitly named offline test modules, frozen parses, and CSV files.
Does not change classifiers, datasets, existing checkpoints or experiment rules.
"""
import argparse
import ast
from contextlib import closing, redirect_stdout, redirect_stderr
from importlib.metadata import distributions
import io
import json
from pathlib import Path
import platform
import sqlite3
import unittest
from unittest.mock import patch

import pandas as pd

from scripts.check_saved_negation_scope import saved_parse, digest
from scripts.run_six_systems import PRESETS
from src.dependency.stanza_parser import TurkishDependencyParser
from src.evaluation.dataset_loader import load_sentiment_dataset
from src.evaluation.run_identity import implementation_fingerprint
from src.hisnet.hisnet_classifier import HisNetClassifier
from src.llm.ollama_zero_shot_classifier import OllamaZeroShotSentimentClassifier as LLM
from src.llm.dependency_ollama_zero_shot_classifier import DependencyOllamaZeroShotSentimentClassifier as DPLLM

ROOT = Path(__file__).resolve().parents[1]
TEST_MODULES = ["tests.test_audit_regressions", "tests.test_resumable_evaluation",
                "tests.test_explicit_negation_scope", "tests.test_classical_analysis"]


def context_preflight():
    database = ROOT / "results/trsav1_classical_500_20261031_run01/progress.sqlite3"
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
        rows = [(i, json.loads(payload)) for i, payload in db.execute(
            "SELECT p.row_id,a.result FROM predictions p JOIN attempts a ON a.id=p.attempt_id "
            "WHERE p.system='hisnet_dp' ORDER BY p.row_id")]
    dp = DPLLM.__new__(DPLLM)
    evidence = {}
    for name in ("llm", "dp_llm"):
        singles, batches = [], []
        for batch_size, destination in ((1, singles), (3, batches)):
            for start in range(0, len(rows), batch_size):
                group = rows[start:start + batch_size]
                texts = [result["text"] for _, result in group]
                if name == "dp_llm":
                    dp._active_dependency_parses = [saved_parse(result) for _, result in group]
                    prompt = dp._build_prompt(texts)
                else:
                    prompt = LLM._build_prompt(texts)
                budget = len(prompt.encode("utf-8")) + 1024 + 128 + 64 * len(group)
                destination.append(dict(ids=[i for i, _ in group], budget=budget))
        evidence[name] = dict(single_max_budget=max(s["budget"] for s in singles),
            single_rejected_by_context={str(limit): [s["ids"][0] for s in singles if s["budget"] > limit]
                                        for limit in (8192, 16384, 32768)},
            batch3_count=len(batches), batch3_rejected=sum(b["budget"] > 8192 for b in batches),
            batch3_affected_reviews=sum(len(b["ids"]) for b in batches if b["budget"] > 8192),
            single_budget_by_id={s["ids"][0]: s["budget"] for s in singles})
    return dict(method="Exact current UTF-8 byte guard; NOT model tokenizer counts or inference",
                frozen_source_sha256=digest(database), cases=len(rows), systems=evidence)


def dataset_history():
    histories = {
        "trsav1": ["trsav1_four_systems_500/sample_500.csv",
                    "trsav1_three_systems_1000/sample_1000.csv",
                    "trsav1_fixed/trsav1_6000_threshold005_predictions.csv",
                    "trsav1_full_conservative_negation/trsav1_full_threshold005_conservative_negation_predictions.csv"],
        "beyazperde": ["beyazperde/beyazperde_stratified_300_predictions.csv",
                        "beyazperde/beyazperde_full_predictions.csv"],
    }
    result = {}
    for name, (filename, text_column, label_column) in PRESETS.items():
        source = ROOT / "datasets/Modified/sentence_sentiment" / filename
        frame = load_sentiment_dataset(source, text_column, label_column)
        source_texts = set(frame.text)
        seen, files = set(), []
        for relative in histories[name]:
            path = ROOT / "results" / relative
            if not path.exists():
                files.append(dict(path=relative, missing=True))
                continue
            columns = pd.read_csv(path, nrows=0).columns
            separator = ","
            if len(columns) == 1 and ";" in columns[0]:
                separator = ";"
                columns = pd.read_csv(path, sep=separator, nrows=0).columns
            column = next((c for c in ("text", "review", "Review") if c in columns), None)
            if column is None:
                files.append(dict(path=relative, columns=list(columns), text_column_missing=True))
                continue
            old = pd.read_csv(path, sep=separator, usecols=[column])[column].dropna().astype(str).str.strip()
            overlap = set(old) & source_texts
            seen.update(overlap)
            files.append(dict(path=relative, separator=separator, rows=len(old),
                              matching_unique_texts=len(overlap), sha256=digest(path)))
        budget_overhead = len(LLM._build_prompt([""]).encode("utf-8")) + 1024 + 192
        rejected = frame.text.map(lambda text: len(text.encode("utf-8")) + budget_overhead > 8192)
        result[name] = dict(source_sha256=digest(source), rows=len(frame), unique_texts=len(source_texts),
            labels=frame.gold.value_counts().to_dict(), historical_files=files,
            unique_texts_found_in_history=len(seen), rows_found_in_history=int(frame.text.isin(seen).sum()),
            single_llm_guard_rejected_at_8192=int(rejected.sum()),
            historical_presence_is_not_proof_of_human_inspection=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/paper_readiness_20261031")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    log = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES)
    with redirect_stdout(log), redirect_stderr(log):
        tests = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    (args.output / "offline_tests.log").write_text(log.getvalue(), encoding="utf-8")
    parsed_files = []
    for directory in ("src", "scripts", "tests"):
        for path in (ROOT / directory).rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            parsed_files.append(path.relative_to(ROOT).as_posix())
    with patch.object(TurkishDependencyParser, "__init__", side_effect=AssertionError("Live parser forbidden")), \
         patch.object(HisNetClassifier, "__init__", side_effect=AssertionError("Live HisNet forbidden")), \
         patch.object(LLM, "__init__", side_effect=AssertionError("Live LLM forbidden")):
        context = context_preflight()
        history = dataset_history()
    environment = dict(python=platform.python_version(), platform=platform.platform(),
        implementation_fingerprint=implementation_fingerprint(),
        packages=sorted((d.metadata.get("Name", ""), d.version) for d in distributions()))
    summary = dict(tests_run=tests.testsRun, failures=len(tests.failures), errors=len(tests.errors),
        skipped=len(tests.skipped), passed=tests.wasSuccessful(), modules=TEST_MODULES,
        parsed_file_count=len(parsed_files), parsed_files=parsed_files,
        context=context, dataset_history=history)
    for filename, value in (("readiness_evidence.json", summary), ("environment.json", environment)):
        (args.output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(tests_run=tests.testsRun, tests_passed=tests.wasSuccessful(),
        parsed_file_count=len(parsed_files), context={s: dict(
            single_rejected_at_8192=len(v["single_rejected_by_context"]["8192"]),
            batch3_rejected=v["batch3_rejected"], batch3_count=v["batch3_count"],
            single_max_budget=v["single_max_budget"]) for s, v in context["systems"].items()},
        datasets={s: {k: v[k] for k in ("rows", "rows_found_in_history", "single_llm_guard_rejected_at_8192")}
                  for s, v in history.items()}), ensure_ascii=False, indent=2))
    return 0 if tests.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
