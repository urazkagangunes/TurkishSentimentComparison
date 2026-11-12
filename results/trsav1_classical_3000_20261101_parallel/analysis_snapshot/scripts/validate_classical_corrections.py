"""Validate scope changes using offline regressions and frozen model outputs.

No live parser, morphology or LLM is initialized. The source checkpoint is
opened read-only; replay requires all four complete systems.
"""
import argparse
from contextlib import closing
from copy import deepcopy
import importlib
import io
import json
from pathlib import Path
import runpy
import sqlite3
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from scripts.check_saved_negation_scope import (
    SYSTEMS, digest, metrics, projection, saved_parse,
)
from src.dependency.sentiment_scope_resolver import SentimentScopeResolver
from src.dependency.stanza_parser import TurkishDependencyParser
from src.evaluation.checkpoint_store import atomic_json
from src.hisnet.hisnet_classifier import HisNetClassifier

LABELS = ["negative", "neutral", "positive"]


def run_regressions(resolver_class, output, name, include_existing):
    suites = ["tests.test_classical_scope_corrections.ClassicalScopeCorrectionTests",
              "tests.test_finite_predicate_protection.FinitePredicateProtectionTests"]
    if include_existing:
        suites += ["tests.test_explicit_negation_scope.ExplicitNegationScopeTests",
                   "tests.test_audit_regressions.ScopeRegressions"]
    suite = unittest.TestSuite()
    originals = []
    try:
        for qualified in suites:
            module_name, class_name = qualified.rsplit(".", 1)
            module = importlib.import_module(module_name)
            originals.append((module, module.SentimentScopeResolver))
            module.SentimentScopeResolver = resolver_class
            suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(getattr(module, class_name)))
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        (output / f"{name}_tests.log").write_text(stream.getvalue(), encoding="utf-8")
        return dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                    passed=result.wasSuccessful())
    finally:
        for module, original in originals:
            module.SentimentScopeResolver = original


def replay(system, saved, baseline, resolver_class):
    """Recompose with frozen parsing/morphology and all saved lexical lookups."""
    classifier = SYSTEMS[system].__new__(SYSTEMS[system])
    classifier.scope_resolver = resolver_class()
    parsed = saved_parse(saved)
    classifier.parser = SimpleNamespace(parse=lambda text: deepcopy(parsed))
    if system != "dp_hisnet":
        return classifier.classify(saved["text"], hisnet_result=deepcopy(baseline))
    lookups = {}
    for entry in baseline["matched_words"]:
        lookups[entry["root"]] = dict(positive=entry["positive"], negative=entry["negative"])
    for entry in baseline["unmatched_words"]:
        lookups[entry["root"]] = None
    for entry in saved["candidate_diagnostics"]:
        value = None if entry["lexicon_status"] == "not_found" else dict(
            positive=entry["positive"], negative=entry["negative"])
        root = entry["root"]
        if root in lookups and lookups[root] != value:
            raise AssertionError(f"Inconsistent saved lexical lookup: {root}")
        lookups[root] = value

    def lookup(root):
        if root not in lookups:
            raise AssertionError(f"Unsaved lexical lookup required: {root}")
        return deepcopy(lookups[root])

    classifier.hisnet = SimpleNamespace(_lookup_root=lookup,
        root_extractor=SimpleNamespace(analyze_sentence=lambda text: deepcopy(saved["morphological_analyses"])))
    return classifier.classify(saved["text"])


def compare(source, output, baseline_class, candidate_class, reference_class=None):
    database = source.resolve() / "progress.sqlite3"
    before_hash = digest(database)
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
        cases = {i: dict(text=text, gold=gold) for i, text, gold in db.execute(
            "SELECT row_id,text,gold FROM cases ORDER BY row_id")}
        saved = {(s, i): json.loads(payload) for s, i, payload in db.execute(
            "SELECT p.system,p.row_id,a.result FROM predictions p JOIN attempts a ON a.id=p.attempt_id WHERE p.status='success'")}
    if any((s, i) not in saved for s in ("hisnet", *SYSTEMS) for i in cases):
        raise ValueError("Four complete systems are required before correction analysis.")
    gold = [case["gold"] for case in cases.values()]
    comparisons, rows, changes, per_class, matrices = {}, [], [], [], {}
    with patch.object(TurkishDependencyParser, "__init__", side_effect=AssertionError("Live parser forbidden")), \
         patch.object(HisNetClassifier, "__init__", side_effect=AssertionError("Live morphology/HisNet forbidden")):
        for system in ("hisnet", *SYSTEMS):
            before_labels, after_labels, corrected, harmed, changed_ids = [], [], [], [], []
            exact_replays = 0
            for row_id, case in cases.items():
                before = saved[system, row_id]
                if system == "hisnet":
                    after = before
                else:
                    reproduced = replay(system, before, saved["hisnet", row_id], baseline_class)
                    if projection(reproduced) != projection(before):
                        different = [k for k in projection(before) if reproduced.get(k) != before[k]]
                        raise AssertionError(f"Baseline replay differs: {system}/{row_id}: {different}")
                    exact_replays += 1
                    after = replay(system, before, saved["hisnet", row_id], candidate_class)
                    if reference_class is not None:
                        before = replay(system, before, saved["hisnet", row_id], reference_class)
                before_labels.append(before["prediction"])
                after_labels.append(after["prediction"])
                if before["prediction"] != case["gold"] == after["prediction"]:
                    corrected.append(row_id)
                if before["prediction"] == case["gold"] != after["prediction"]:
                    harmed.append(row_id)
                if projection(before) != projection(after):
                    changed_ids.append(row_id)
                    changes.append(dict(system=system, row_id=row_id, **case,
                                        before=projection(before), after=projection(after)))
                rows.append(dict(system=system, row_id=row_id, **case,
                    before_prediction=before["prediction"], after_prediction=after["prediction"],
                    before_positive=before["positive_score"], before_negative=before["negative_score"],
                    after_positive=after["positive_score"], after_negative=after["negative_score"]))
            comparisons[system] = dict(before=metrics(gold, before_labels), after=metrics(gold, after_labels),
                baseline_exact_replays=exact_replays, changed_intermediate_ids=changed_ids,
                label_changes=sum(a != b for a, b in zip(before_labels, after_labels)),
                corrected_ids=corrected, harmed_ids=harmed)
            for phase, predictions in (("before", before_labels), ("after", after_labels)):
                report = classification_report(gold, predictions, labels=LABELS, output_dict=True, zero_division=0)
                per_class.extend(dict(system=system, phase=phase, label=label, **report[label]) for label in LABELS)
                matrices[system + "_" + phase] = confusion_matrix(gold, predictions, labels=LABELS).tolist()
            compact = {k: v for k, v in comparisons[system].items() if k != "changed_intermediate_ids"}
            compact["changed_intermediate_count"] = len(changed_ids)
            print("REPLAY", system, json.dumps(compact, ensure_ascii=False), flush=True)
    if digest(database) != before_hash:
        raise AssertionError("Source checkpoint changed during replay.")
    summary = dict(method="Frozen parsing/morphology/lookup replay; not new end-to-end inference",
        cases=len(cases), source=str(source), source_sha256=before_hash, source_unchanged=True,
        systems=comparisons, confusion_matrices=matrices, label_order=LABELS)
    atomic_json(output / "comparison.json", summary)
    atomic_json(output / "changed_details.json", changes)
    pd.DataFrame(rows).to_csv(output / "before_after.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_class).to_csv(output / "per_class_before_after.csv", index=False, encoding="utf-8-sig")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--baseline-resolver", type=Path, required=True)
    parser.add_argument("--reference-resolver", type=Path,
                        help="Optional later revision used as before; original baseline still must exactly replay.")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.source and (args.output.resolve() == args.source.resolve() or args.source.resolve() in args.output.resolve().parents):
        parser.error("Correction output must be separate from the frozen source run.")
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = runpy.run_path(str(args.baseline_resolver))["SentimentScopeResolver"]
    reference = (runpy.run_path(str(args.reference_resolver))["SentimentScopeResolver"]
                 if args.reference_resolver else None)
    candidate = (runpy.run_path(str(args.candidate))["SentimentScopeResolver"]
                 if args.candidate else SentimentScopeResolver)
    results = dict(baseline=run_regressions(reference or baseline, args.output, "baseline", False),
                   candidate=run_regressions(candidate, args.output, "candidate", True),
                   baseline_sha256=digest(args.baseline_resolver),
                   reference_sha256=digest(args.reference_resolver) if args.reference_resolver else None,
                   candidate_sha256=digest(args.candidate or Path("src/dependency/sentiment_scope_resolver.py")))
    atomic_json(args.output / "regressions.json", results)
    print(json.dumps(results, ensure_ascii=False), flush=True)
    if not results["candidate"]["passed"]:
        return 1
    if args.source:
        compare(args.source, args.output, baseline, candidate, reference)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
