"""Verify and report a completed shared-preprocessing run without rewriting it."""
import argparse
from contextlib import closing
import gc
import json
from pathlib import Path
import sqlite3
import time

from scripts.run_balanced_classical_cached import SYSTEMS, ROOT, semantic_result
from scripts.run_six_systems import digest, resource_manifest
from src.evaluation.checkpoint_store import atomic_json
from src.evaluation.resumable_evaluator import make_classifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    database = output / "progress.sqlite3"
    before_hash = digest(database)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for path in (ROOT / "src").rglob("*.py"):
        if digest(path) != digest(output / "source_snapshot" / path.relative_to(ROOT)):
            raise ValueError("Live source differs from the frozen run; verify before applying corrections.")
    if resource_manifest() != manifest["resources"]:
        raise ValueError("Model resources differ from the frozen experiment.")
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
        cases = {i: dict(text=text, gold=gold) for i,text,gold in db.execute("SELECT row_id,text,gold FROM cases ORDER BY row_id")}
        for system in SYSTEMS:
            count = db.execute("SELECT COUNT(*) FROM predictions WHERE system=? AND status='success'", (system,)).fetchone()[0]
            if count != len(cases):
                raise ValueError(f"Incomplete system: {system}")
        ids = [next(i for i,c in cases.items() if c["gold"] == label)
               for label in ("negative", "neutral", "positive")]
        comparisons = []
        for system in SYSTEMS:
            classifier = make_classifier(system, {})
            for row_id in ids:
                saved = json.loads(db.execute("SELECT a.result FROM predictions p JOIN attempts a ON a.id=p.attempt_id "
                    "WHERE p.system=? AND p.row_id=?", (system, row_id)).fetchone()[0])
                start = time.perf_counter()
                result = classifier.classify(cases[row_id]["text"])
                elapsed = time.perf_counter() - start
                if semantic_result(result) != semantic_result(saved):
                    live, frozen = semantic_result(result), semantic_result(saved)
                    different = [k for k in set(live) | set(frozen) if live.get(k) != frozen.get(k)]
                    raise AssertionError(f"Independent mismatch {system}/{row_id}: {different}")
                comparisons.append(dict(system=system, row_id=row_id, exact_semantic_match=True, seconds=elapsed))
                print("VERIFIED", system, row_id, flush=True)
            del classifier
            gc.collect()
        parse_count, parse_seconds = db.execute("SELECT COUNT(*),SUM(seconds) FROM parses").fetchone()
    if digest(database) != before_hash:
        raise AssertionError("Frozen checkpoint changed during independent verification.")
    atomic_json(output / "independent_verification.json", dict(selection="first row per gold class",
        comparisons=comparisons, checkpoint_sha256=before_hash, source_unchanged=True,
        comparison_format="JSON-normalized full semantic results; only evaluator metadata and baseline_reused excluded",
        verifier_sha256=digest(__file__), helper_sha256=digest(ROOT / "scripts/run_balanced_classical_cached.py"),
        limitation="12 sampled comparisons, not 4800 independent evaluations or speed benchmarks"))
    atomic_json(output / "shared_preprocessing.json", dict(parse_rows=parse_count, parse_seconds=parse_seconds,
        imported=manifest["imported"], timing_policy=manifest["timing_policy"]))
    from scripts.analyze_classical_500 import analyze
    analyze(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
