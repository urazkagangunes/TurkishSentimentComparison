"""Run four classical conditions with shared, immutable input preprocessing.

This development runner measures predictions, not independent pipeline speed.
Only successful HisNet rows from an identical frozen partial run may be imported.
"""
import argparse
from contextlib import closing
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import time
from types import SimpleNamespace

from scripts.run_six_systems import digest, resource_manifest
from src.dependency.sentiment_scope_resolver import SentimentScopeResolver
from src.evaluation.checkpoint_store import CheckpointStore, atomic_json, read_status
from src.evaluation.resumable_evaluator import make_classifier
from src.evaluation.run_identity import implementation_fingerprint
from tests.run_trsav1_classical_1200 import load_sample, ROOT, SYSTEMS

def classify_shared(system, text, baseline, parsed, lookup):
    """Use production composition with detached, label-free cached inputs."""
    from src.hisnet.hisnet_dependency_classifier import HisNetDependencyClassifier
    from src.hisnet.dependency_hisnet_classifier import DependencyHisNetClassifier
    from src.hisnet.literature_weighted_hisnet_dependency_classifier import LiteratureWeightedHisNetDependencyClassifier
    cls = dict(dp_hisnet=DependencyHisNetClassifier, hisnet_dp=HisNetDependencyClassifier,
               lw_hisnet_dp=LiteratureWeightedHisNetDependencyClassifier)[system]
    classifier = cls.__new__(cls)
    classifier.scope_resolver = SentimentScopeResolver()
    classifier.parser = SimpleNamespace(parse=lambda value: deepcopy(parsed))
    classifier.hisnet = SimpleNamespace(_lookup_root=lambda root: deepcopy(lookup(root)),
        root_extractor=SimpleNamespace(analyze_sentence=lambda value: deepcopy(baseline["morphological_analyses"])))
    if system == "dp_hisnet":
        return classifier.classify(text)
    return classifier.classify(text, hisnet_result=deepcopy(baseline))


def read_result(store, system, row_id):
    row = store.connection.execute("SELECT a.result FROM predictions p JOIN attempts a ON a.id=p.attempt_id "
        "WHERE p.system=? AND p.row_id=? AND p.status='success'", (system, row_id)).fetchone()
    return json.loads(row[0]) if row else None


def semantic_result(result):
    # SQLite stores JSON. In-memory diagnostic maps use integer sentence
    # keys, which JSON necessarily turns into strings. Compare both sides
    # in the persisted representation, without dropping diagnostic fields.
    filtered = {k: v for k, v in result.items() if k not in {
        "predictions", "status", "error", "baseline_reused", "imported_from_checkpoint"}}
    return json.loads(json.dumps(filtered, ensure_ascii=False, allow_nan=False))


def load_import(source, cases, implementation):
    if source is None:
        return {}, None
    database = source.resolve() / "progress.sqlite3"
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
        manifest = json.loads(db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
        old_cases = [dict(text=t, gold=g) for t, g in db.execute("SELECT text,gold FROM cases ORDER BY row_id")]
        if old_cases != cases:
            raise ValueError("Imported sample differs from the current experiment.")
        if manifest["implementation"] != implementation:
            raise ValueError(f"Implementation differs: saved={manifest['implementation']}, current={implementation}")
        rows = {i: (json.loads(payload), seconds) for i, payload, seconds in db.execute(
            "SELECT p.row_id,a.result,a.seconds FROM predictions p JOIN attempts a ON a.id=p.attempt_id "
            "WHERE p.system='hisnet' AND p.status='success'")}
    wal = database.with_name(database.name + "-wal")
    return rows, dict(source=str(source.resolve()), database_sha256=digest(database),
                      wal_sha256=digest(wal) if wal.exists() else None, imported_hisnet_rows=len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--sample-per-class", type=int, default=400)
    parser.add_argument("--exclude-sample", type=Path, action="append", default=[])
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.status:
        print(json.dumps(read_status(args.output), ensure_ascii=False, indent=2))
        return 0
    if args.sample_per_class == 400 and not args.exclude_sample:
        sample, cases, sampling = load_sample()
    else:
        from scripts.balanced_classical_sample import load_disjoint_sample
        sample, cases, sampling = load_disjoint_sample(args.sample_per_class, args.exclude_sample)
    sampling["execution_mode"] = "shared_immutable_preprocessing"
    implementation = implementation_fingerprint()
    imported, provenance = load_import(args.source, cases, implementation)
    metadata = dict(implementation=implementation, runner_sha256=digest(__file__),
        sample_runner_sha256=digest(ROOT / "tests/run_trsav1_classical_1200.py"),
        disjoint_sampler_sha256=digest(ROOT / "scripts/balanced_classical_sample.py"),
        sample=sampling, resources=resource_manifest(), systems=list(SYSTEMS), keep_details=True,
        protocol=f"balanced_{len(cases)}_shared_immutable_morphology_and_parse_no_tuning",
        timing_policy="HisNet includes morphology; dependency timings are composition only; parse recorded separately",
        imported=provenance)
    with CheckpointStore(args.output, metadata, cases) as store:
        atomic_json(args.output / "sampling.json", sampling)
        sample.assign(sample_id=range(len(sample)), source_cleaned_row_id=sampling["source_cleaned_row_ids"])[
            ["sample_id", "source_cleaned_row_id", "text", "gold"]].to_csv(
                args.output / f"sample_{len(cases)}.csv", index=False, encoding="utf-8-sig")
        snapshot_paths = list((ROOT / "src").rglob("*.py")) + [
            Path(__file__).resolve(), ROOT / "scripts/balanced_classical_sample.py",
            ROOT / "scripts/run_six_systems.py", ROOT / "tests/run_trsav1_classical_1200.py"]
        for path in snapshot_paths:
            destination = args.output / "source_snapshot" / path.relative_to(ROOT)
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
        store.connection.execute("CREATE TABLE IF NOT EXISTS parses (row_id INTEGER PRIMARY KEY, payload TEXT NOT NULL, seconds REAL NOT NULL)")
        store.connection.commit()
        if args.prepare_only:
            store.export()
            print(f"Prepared {len(cases)} cases; no classifier initialized.", flush=True)
            return 0
        states = {s: store.states(s) for s in SYSTEMS}
        for row_id, (result, seconds) in imported.items():
            if states["hisnet"].get(row_id) == "success":
                continue
            result["imported_from_checkpoint"] = True
            attempt = store.start("hisnet", [row_id], False)
            store.finish(attempt, result, seconds)
            states["hisnet"][row_id] = "success"
        print("IMPORTED", len(imported), "HisNet successes", flush=True)
        initializer_path = args.output / "initializations.json"
        initializations = json.loads(initializer_path.read_text(encoding="utf-8")) if initializer_path.exists() else []
        if provenance and not any(r.get("source") == provenance["source"] for r in initializations):
            original = json.loads((args.source / "initializations.json").read_text(encoding="utf-8"))
            initializations.extend(dict(r, source=provenance["source"]) for r in original if r["system"] == "hisnet")
        hisnet = dependency_parser = None

        def initialize(kind):
            nonlocal hisnet, dependency_parser
            current = hisnet if kind == "hisnet" else dependency_parser
            if current is None:
                from src.hisnet.hisnet_classifier import HisNetClassifier
                from src.dependency.stanza_parser import TurkishDependencyParser
                start = time.perf_counter()
                current = HisNetClassifier() if kind == "hisnet" else TurkishDependencyParser()
                if kind == "hisnet":
                    hisnet = current
                else:
                    dependency_parser = current
                initializations.append(dict(system=kind, seconds=time.perf_counter()-start,
                                           utc=datetime.now(timezone.utc).isoformat()))
                atomic_json(initializer_path, initializations)
            return current

        def get_parse(row_id):
            saved = store.connection.execute("SELECT payload FROM parses WHERE row_id=?", (row_id,)).fetchone()
            if saved:
                return json.loads(saved[0])
            parser_instance = initialize("shared_parser")
            start = time.perf_counter()
            parsed = parser_instance.parse(cases[row_id]["text"])
            with store.connection:
                store.connection.execute("INSERT INTO parses VALUES (?,?,?)",
                    (row_id, json.dumps(parsed, ensure_ascii=False), time.perf_counter()-start))
            return parsed

        try:
            for row_id, case in enumerate(cases):
                for system in SYSTEMS:
                    if states[system].get(row_id) == "success":
                        continue
                    initialize("hisnet")
                    baseline = read_result(store, "hisnet", row_id) if system != "hisnet" else None
                    # Parsing is charged once to shared preprocessing, not to
                    # whichever method happens to request it first.
                    parsed = get_parse(row_id) if baseline is not None else None
                    attempt = store.start(system, [row_id], states[system].get(row_id) == "failed")
                    start = time.perf_counter()
                    try:
                        if system == "hisnet":
                            result = hisnet.classify(case["text"])
                        elif baseline is None:
                            raise ValueError("Missing successful shared morphology/HisNet input")
                        else:
                            result = classify_shared(system, case["text"], baseline, parsed, hisnet._lookup_root)
                        result.update(predictions=[result["prediction"]], status="success", error=None)
                    except Exception as error:
                        result = dict(predictions=[None], status="failed", error=f"{type(error).__name__}: {error}")
                    store.finish(attempt, result, time.perf_counter()-start)
                    states[system][row_id] = result["status"]
                    if result["status"] != "success":
                        print("FAILED", system, row_id, result["error"], flush=True)
                if (row_id + 1) % 25 == 0:
                    print("SAVED", row_id + 1, "/", len(cases), {s: sum(v == "success" for v in states[s].values()) for s in SYSTEMS}, flush=True)
            summary = store.export()
            complete = all(summary[s]["complete"] for s in SYSTEMS)
            verify_path = args.output / "independent_verification.json"
            if complete and not verify_path.exists():
                # Fixed selection rule: first row of each class, before any
                # inspection of errors. Compare full semantic outputs.
                ids = [next(i for i,c in enumerate(cases) if c["gold"] == label)
                       for label in ("negative", "neutral", "positive")]
                verified = []
                for system in SYSTEMS:
                    independent = make_classifier(system, {})
                    for row_id in ids:
                        result = independent.classify(cases[row_id]["text"])
                        expected = read_result(store, system, row_id)
                        if semantic_result(result) != semantic_result(expected):
                            different = sorted(k for k in set(result) | set(expected)
                                if k in semantic_result(result) or k in semantic_result(expected)
                                if result.get(k) != expected.get(k))
                            raise AssertionError(f"Shared/independent mismatch {system}/{row_id}: {different}")
                        verified.append(dict(system=system, row_id=row_id, exact_semantic_match=True))
                    del independent
                atomic_json(verify_path, dict(selection="first row per gold class", comparisons=verified,
                    limitation="12 sampled comparisons; not independent timing benchmarks"))
            parse_total = store.connection.execute("SELECT COUNT(*),COALESCE(SUM(seconds),0) FROM parses").fetchone()
            atomic_json(args.output / "shared_preprocessing.json", dict(parse_rows=parse_total[0],
                parse_seconds=parse_total[1], imported=provenance, timing_policy=metadata["timing_policy"]))
        finally:
            store.export()
            atomic_json(initializer_path, initializations)
    from scripts.analyze_classical_500 import analyze
    analyze(args.output)
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
