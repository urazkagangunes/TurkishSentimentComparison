"""Replay composition from frozen parses/morphology/lookups, without model loading.

The old resolver must reproduce the saved scopes and scores before a comparison
is accepted. The original SQLite checkpoint is opened read-only and never updated.
Run as: python -m scripts.check_saved_negation_scope --baseline-resolver PATH
"""
import argparse
from collections import Counter
from contextlib import closing
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
import sqlite3
from types import SimpleNamespace
from unittest.mock import patch

from sklearn.metrics import accuracy_score, f1_score

from src.dependency.sentiment_scope_resolver import SentimentScopeResolver
from src.hisnet.dependency_hisnet_classifier import DependencyHisNetClassifier
from src.hisnet.hisnet_classifier import HisNetClassifier
from src.hisnet.hisnet_dependency_classifier import HisNetDependencyClassifier
from src.hisnet.literature_weighted_hisnet_dependency_classifier import (
    LiteratureWeightedHisNetDependencyClassifier,
)
from src.dependency.stanza_parser import TurkishDependencyParser


SYSTEMS = {
    "dp_hisnet": DependencyHisNetClassifier,
    "hisnet_dp": HisNetDependencyClassifier,
    "lw_hisnet_dp": LiteratureWeightedHisNetDependencyClassifier,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def saved_parse(result):
    if "parsed_sentences" in result:
        return deepcopy(result["parsed_sentences"])
    sentences = []
    for sentence in result["sentences"]:
        tokens = []
        for saved in sentence["dependency_tokens"]:
            token = {k: v for k, v in saved.items()
                     if k not in {"local_id", "local_head", "sentence_index"}}
            token.update(id=saved["local_id"], head=saved["local_head"])
            tokens.append(token)
        sentences.append(dict(text=sentence["text"], tokens=tokens))
    return sentences


def replay(system, result, resolver_class):
    """Use production classification/composition with strictly frozen inputs."""
    parsed = saved_parse(result)
    classifier = SYSTEMS[system].__new__(SYSTEMS[system])
    classifier.scope_resolver = resolver_class()
    classifier.parser = SimpleNamespace(parse=lambda text: deepcopy(parsed))
    if system == "dp_hisnet":
        lookups = {}
        for candidate in result["candidate_diagnostics"]:
            root = candidate.get("root")
            value = (None if candidate["lexicon_status"] == "not_found" else
                     dict(positive=candidate["positive"], negative=candidate["negative"]))
            if root in lookups and lookups[root] != value:
                raise AssertionError(f"Inconsistent frozen lookup: {root}")
            lookups[root] = value

        def lookup(root):
            if root not in lookups:
                raise AssertionError(f"Replay requires an unsaved lookup: {root}")
            return deepcopy(lookups[root])

        classifier.hisnet = SimpleNamespace(
            _lookup_root=lookup,
            root_extractor=SimpleNamespace(analyze_sentence=lambda text:
                deepcopy(result["morphological_analyses"])),
        )
        return classifier.classify(result["text"])
    baseline = {k: deepcopy(result[k]) for k in (
        "text", "sentiment_words", "morphological_analyses", "lexicon_coverage", "sentiment_coverage")}
    baseline.update({k: result["hisnet_" + k] for k in (
        "prediction", "positive_score", "negative_score")})
    return classifier.classify(result["text"], hisnet_result=baseline)


def projection(result):
    # Include all semantic intermediate outputs, excluding evaluator metadata and
    # baseline_reused (necessarily true for a replay).
    keys = ("prediction", "positive_score", "negative_score", "scopes", "sentences",
            "dependency_candidates", "candidate_diagnostics", "dependency_tokens",
            "dependency_prediction", "dependency_positive_score", "dependency_negative_score")
    return {k: result[k] for k in keys if k in result}


def metrics(gold, predicted):
    return dict(correct=sum(a == b for a, b in zip(gold, predicted)),
                accuracy=accuracy_score(gold, predicted),
                macro_f1=f1_score(gold, predicted, labels=["negative", "neutral", "positive"],
                                  average="macro", zero_division=0))


def check(source, output, baseline_path):
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents:
        raise ValueError("Use a separate output directory; the original run is frozen.")
    database = source / "progress.sqlite3"
    database_hash = digest(database)
    baseline_class = runpy.run_path(str(baseline_path))["SentimentScopeResolver"]
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
        cases = {i: dict(text=t, gold=g) for i, t, g in db.execute(
            "SELECT row_id,text,gold FROM cases ORDER BY row_id")}
        results = {(s, i): json.loads(payload) for s, i, payload in db.execute(
            "SELECT p.system,p.row_id,a.result FROM predictions p "
            "JOIN attempts a ON a.id=p.attempt_id WHERE p.status='success'")}
    if any((s, i) not in results for s in ("hisnet", *SYSTEMS) for i in cases):
        raise ValueError("Every case must have all four successful saved results.")
    summary, changes, parent_edges = {}, [], []
    # Fail loudly if this tool accidentally starts a real parser or lexicon.
    with patch.object(TurkishDependencyParser, "__init__", side_effect=AssertionError("Live parser forbidden")), \
         patch.object(HisNetClassifier, "__init__", side_effect=AssertionError("Live HisNet forbidden")):
        for system in SYSTEMS:
            before_labels, after_labels, changed_ids, corrected, harmed = [], [], [], [], []
            for row_id, case in cases.items():
                saved = results[system, row_id]
                before = replay(system, saved, baseline_class)
                if projection(before) != projection(saved):
                    fields = [k for k in projection(saved) if before.get(k) != saved[k]]
                    raise AssertionError(f"Baseline replay mismatch: {system} ID {row_id}: {fields}")
                after = replay(system, saved, SentimentScopeResolver)
                before_labels.append(before["prediction"])
                after_labels.append(after["prediction"])
                if projection(before) != projection(after):
                    changed_ids.append(row_id)
                    changes.append(dict(system=system, row_id=row_id, **case,
                                        before=projection(before), after=projection(after)))
                if before["prediction"] != case["gold"] == after["prediction"]:
                    corrected.append(row_id)
                if before["prediction"] == case["gold"] != after["prediction"]:
                    harmed.append(row_id)
                if system == "hisnet_dp":
                    tokens, _, _ = HisNetDependencyClassifier._flatten_dependency_sentences(saved_parse(saved))
                    by_id, children = SentimentScopeResolver()._build_indexes(tokens)
                    for scope in before["scopes"]:
                        token = by_id.get((scope.get("sentiment_token") or {}).get("id"))
                        parent = by_id.get(token.get("head")) if token else None
                        if parent and SentimentScopeResolver()._is_explicit_degil(parent):
                            parent_edges.append(dict(row_id=row_id, text=case["text"], token=token,
                                parent=parent, children=children.get(token["id"], []),
                                before_negations=scope["negations"]))
            gold = [c["gold"] for c in cases.values()]
            summary[system] = dict(before=metrics(gold, before_labels), after=metrics(gold, after_labels),
                baseline_exact_replays=len(cases), changed_ids=changed_ids, corrected_ids=corrected,
                harmed_ids=harmed, label_changes=sum(a != b for a, b in zip(before_labels, after_labels)))
    if digest(database) != database_hash:
        raise AssertionError("Source checkpoint changed during replay; comparison invalid.")
    hisnet_labels = [results["hisnet", i]["prediction"] for i in cases]
    report = dict(method="Frozen parse, morphology and lexicon replay; no end-to-end inference",
        cases=len(cases), source=str(source), source_sha256=database_hash,
        baseline_resolver_sha256=digest(baseline_path),
        current_resolver_sha256=digest(Path("src/dependency/sentiment_scope_resolver.py")),
        source_unchanged=True, hisnet=metrics([c["gold"] for c in cases.values()], hisnet_labels),
        systems=summary, parent_edge_count=len(parent_edges),
        parent_edge_relations=dict(Counter(e["token"]["deprel"] for e in parent_edges)))
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in (("summary.json", report), ("changes.json", changes),
                          ("explicit_degil_parent_edges.json", parent_edges)):
        (output / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/trsav1_classical_500_20261031_run01"))
    parser.add_argument("--output", type=Path, default=Path("results/id148_scope_check_20261031"))
    parser.add_argument("--baseline-resolver", type=Path, required=True)
    args = parser.parse_args()
    check(args.source, args.output, args.baseline_resolver)
