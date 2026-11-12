"""Same frozen classical methods, with bounded parallel review preprocessing.

One parent writes SQLite. Workers receive text and optional cached inputs, never
gold labels. Independent pipeline checks still run after every case is complete.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import closing
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time

from scripts.balanced_classical_sample import load_disjoint_sample
from scripts.run_balanced_classical_cached import classify_shared, read_result, semantic_result
from scripts.run_six_systems import digest, resource_manifest
from src.evaluation.checkpoint_store import CheckpointStore, atomic_json, read_status
from src.evaluation.resumable_evaluator import make_classifier
from src.evaluation.run_identity import implementation_fingerprint
from tests.run_trsav1_classical_1200 import ROOT, SYSTEMS

_HISNET = _PARSER = None
_INITIALIZATIONS = []


def initialize_worker(threads):
    global _HISNET, _PARSER, _INITIALIZATIONS
    import torch
    torch.set_num_threads(threads)
    from src.hisnet.hisnet_classifier import HisNetClassifier
    from src.dependency.stanza_parser import TurkishDependencyParser
    _INITIALIZATIONS = []
    for kind, factory in (("hisnet", HisNetClassifier), ("shared_parser", TurkishDependencyParser)):
        started = time.perf_counter()
        instance = factory()
        if kind == "hisnet":
            _HISNET = instance
        else:
            _PARSER = instance
        _INITIALIZATIONS.append(dict(system=kind, worker_pid=os.getpid(),
            seconds=time.perf_counter()-started, utc=datetime.now(timezone.utc).isoformat()))


def process_review(row_id, text, baseline=None, parsed=None):
    """No labels enter this process; each decision uses production classifiers."""
    started = time.perf_counter()
    baseline = deepcopy(baseline) if baseline is not None else _HISNET.classify(text)
    results = {"hisnet": (baseline, time.perf_counter()-started)}
    parse_start = time.perf_counter()
    parsed = deepcopy(parsed) if parsed is not None else _PARSER.parse(text)
    parse_seconds = time.perf_counter()-parse_start
    for system in SYSTEMS[1:]:
        started = time.perf_counter()
        result = classify_shared(system, text, baseline, parsed, _HISNET._lookup_root)
        results[system] = result, time.perf_counter()-started
    return dict(row_id=row_id, results=results, parsed=parsed, parse_seconds=parse_seconds,
                initializations=_INITIALIZATIONS, worker_pid=os.getpid())


def import_frozen(source, store, implementation, resources, cases):
    """Import all successful results and shared parses, preserving their costs."""
    if source is None:
        return None
    database = source.resolve()/"progress.sqlite3"
    before = digest(database)
    with closing(sqlite3.connect(database.as_uri()+"?mode=ro", uri=True)) as db:
        manifest = json.loads(db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
        old_cases = [dict(text=t, gold=g) for t, g in db.execute("SELECT text,gold FROM cases ORDER BY row_id")]
        if old_cases != cases or manifest["implementation"] != implementation or manifest["resources"] != resources:
            raise ValueError("Source sample, implementation or resources differ")
        if db.execute("SELECT COUNT(*) FROM attempts WHERE status='running'").fetchone()[0]:
            raise ValueError("Stop the source writer before importing its checkpoint")
        rows = list(db.execute("SELECT p.system,p.row_id,a.result,a.seconds FROM predictions p "
            "JOIN attempts a ON a.id=p.attempt_id WHERE p.status='success'"))
        states = {s: store.states(s) for s in SYSTEMS}
        if any(s not in SYSTEMS for s, _, _, _ in rows):
            raise ValueError("Only the four classical conditions may be imported")
        for system, row_id, payload, seconds in rows:
            if states[system].get(row_id) == "success":
                continue
            result = json.loads(payload)
            result["imported_from_checkpoint"] = True
            attempt = store.start(system, [row_id], False)
            store.finish(attempt, result, seconds)
        with store.connection:
            store.connection.executemany("INSERT OR IGNORE INTO parses VALUES (?,?,?)",
                db.execute("SELECT row_id,payload,seconds FROM parses"))
    if digest(database) != before:
        raise AssertionError("Import source changed")
    return dict(source=str(source.resolve()), database_sha256=before,
                successful_predictions=len(rows), counts={s:sum(r[0]==s for r in rows) for s in SYSTEMS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--sample-per-class", type=int, default=1000)
    parser.add_argument("--exclude-sample", action="append", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        print(json.dumps(read_status(args.output), ensure_ascii=False))
        return 0
    if args.workers < 1 or args.torch_threads < 1:
        parser.error("Workers and torch threads must be positive")
    if args.source and args.source.resolve() == args.output.resolve():
        parser.error("Import source must be a separate frozen directory")
    sample, cases, sampling = load_disjoint_sample(args.sample_per_class, args.exclude_sample)
    sampling["execution_mode"] = "shared_immutable_preprocessing"
    implementation = implementation_fingerprint()
    resources = resource_manifest()
    metadata = dict(implementation=implementation, runner_sha256=digest(__file__),
        runner_path="scripts/run_parallel_classical_cached.py",
        shared_runner_sha256=digest(ROOT/"scripts/run_balanced_classical_cached.py"),
        sample_runner_sha256=digest(ROOT/"tests/run_trsav1_classical_1200.py"),
        disjoint_sampler_sha256=digest(ROOT/"scripts/balanced_classical_sample.py"),
        sample=sampling, resources=resources, systems=list(SYSTEMS), keep_details=True,
        protocol=f"balanced_{len(cases)}_shared_immutable_preprocessing_parallel_reviews_no_tuning",
        workers=args.workers, torch_threads_per_worker=args.torch_threads,
        timing_policy="Summed worker compute costs, not wall time or independent pipeline speeds",
        imported_source=str(args.source.resolve()) if args.source else None,
        imported_source_sha256=digest(args.source/"progress.sqlite3") if args.source else None)
    with CheckpointStore(args.output, metadata, cases) as store:
        store.connection.execute("CREATE TABLE IF NOT EXISTS parses (row_id INTEGER PRIMARY KEY, payload TEXT NOT NULL, seconds REAL NOT NULL)")
        store.connection.execute("CREATE TABLE IF NOT EXISTS worker_assignments (row_id INTEGER PRIMARY KEY, worker_pid INTEGER NOT NULL)")
        store.connection.commit()
        provenance = import_frozen(args.source, store, implementation, resources, cases)
        atomic_json(args.output/"imported.json", provenance)
        atomic_json(args.output/"sampling.json", sampling)
        sample.assign(sample_id=range(len(sample)), source_cleaned_row_id=sampling["source_cleaned_row_ids"])[
            ["sample_id", "source_cleaned_row_id", "text", "gold"]].to_csv(
                args.output/f"sample_{len(cases)}.csv", index=False, encoding="utf-8-sig")
        paths = list((ROOT/"src").rglob("*.py")) + [Path(__file__).resolve(),
            ROOT/"scripts/run_balanced_classical_cached.py", ROOT/"scripts/balanced_classical_sample.py",
            ROOT/"scripts/run_six_systems.py", ROOT/"tests/run_trsav1_classical_1200.py"]
        for path in paths:
            frozen = args.output/"source_snapshot"/path.relative_to(ROOT)
            if not frozen.exists():
                frozen.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, frozen)
        init_path = args.output/"initializations.json"
        initializations = json.loads(init_path.read_text(encoding="utf-8")) if init_path.exists() else []
        if args.source and not initializations:
            initializations = [dict(r, source=str(args.source.resolve())) for r in
                json.loads((args.source/"initializations.json").read_text(encoding="utf-8"))]
        states = {s: store.states(s) for s in SYSTEMS}
        first_ids = [next(i for i,c in enumerate(cases) if c["gold"]==label)
                     for label in ("negative", "neutral", "positive")]
        plan_path = args.output/"verification_plan.json"
        if not plan_path.exists():
            parallel_ids = [next((i for i,c in enumerate(cases) if c["gold"]==label
                and any(states[s].get(i)!="success" for s in SYSTEMS)), None)
                for label in ("negative", "neutral", "positive")]
            atomic_json(plan_path, dict(base_ids=first_ids,
                parallel_class_ids=[i for i in parallel_ids if i is not None],
                selection="first row per class, first new row per class, and first row per worker"))
        pending = iter(i for i in range(len(cases)) if any(states[s].get(i) != "success" for s in SYSTEMS))
        last_report = sum(all(states[s].get(i)=="success" for s in SYSTEMS) for i in range(len(cases)))
        print("IMPORTED", last_report, "complete reviews; WORKERS", args.workers, flush=True)
        started_wall = time.perf_counter()
        try:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                                     initargs=(args.torch_threads,)) as pool:
                futures = {}
                probe_path = args.output/"parallel_preflight.json"
                if not probe_path.exists() and all(states[s].get(i)=="success" for s in SYSTEMS for i in first_ids):
                    probes = [pool.submit(process_review, i, cases[i]["text"]) for i in first_ids]
                    comparisons = []
                    for job in probes:
                        probe = job.result()
                        for system, (result, _) in probe["results"].items():
                            if semantic_result(result) != semantic_result(read_result(store, system, probe["row_id"])):
                                raise AssertionError(f"Parallel preflight mismatch: {system}/{probe['row_id']}")
                            comparisons.append(dict(system=system, row_id=probe["row_id"],
                                worker_pid=probe["worker_pid"], exact_semantic_match=True))
                    atomic_json(probe_path, dict(comparisons=comparisons, fresh_morphology_and_parse=True))
                    print("PARALLEL PREFLIGHT", len(comparisons), "exact semantic matches", flush=True)

                def submit_next():
                    row_id = next(pending, None)
                    if row_id is None:
                        return
                    baseline = read_result(store, "hisnet", row_id)
                    parsed = store.connection.execute("SELECT payload FROM parses WHERE row_id=?", (row_id,)).fetchone()
                    attempts = {s:store.start(s, [row_id], states[s].get(row_id)=="failed")
                                for s in SYSTEMS if states[s].get(row_id) != "success"}
                    job = pool.submit(process_review, row_id, cases[row_id]["text"], baseline,
                                      json.loads(parsed[0]) if parsed else None)
                    futures[job] = row_id, attempts

                for _ in range(args.workers*2):
                    submit_next()
                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        row_id, attempts = futures.pop(future)
                        batch = future.result()
                        for record in batch["initializations"]:
                            if record not in initializations:
                                initializations.append(record)
                        atomic_json(init_path, initializations)
                        with store.connection:
                            store.connection.execute("INSERT OR IGNORE INTO parses VALUES (?,?,?)",
                                (row_id, json.dumps(batch["parsed"], ensure_ascii=False), batch["parse_seconds"]))
                            store.connection.execute("INSERT OR REPLACE INTO worker_assignments VALUES (?,?)",
                                                     (row_id, batch["worker_pid"]))
                        for system, (result, seconds) in batch["results"].items():
                            if system not in attempts:
                                if semantic_result(result) != semantic_result(read_result(store, system, row_id)):
                                    raise AssertionError(f"Recomputed partial row differs: {system}/{row_id}")
                                continue
                            result.update(predictions=[result["prediction"]], status="success", error=None)
                            store.finish(attempts[system], result, seconds)
                            states[system][row_id] = "success"
                        submit_next()
                    count = sum(all(states[s].get(i)=="success" for s in SYSTEMS) for i in range(len(cases)))
                    if count >= last_report+25:
                        print("SAVED", count, "/", len(cases), flush=True)
                        last_report = count
            summary = store.export()
            complete = all(summary[s]["complete"] for s in SYSTEMS)
            atomic_json(args.output/"parallel_timing.json", dict(workers=args.workers,
                wall_seconds_this_session=time.perf_counter()-started_wall,
                excludes_imported_work_and_independent_verification=True))
            verify_path = args.output/"independent_verification.json"
            if complete and not verify_path.exists():
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                worker_ids = [r[0] for r in store.connection.execute("SELECT MIN(row_id) FROM worker_assignments GROUP BY worker_pid")]
                ids = sorted(set(plan["base_ids"]+plan["parallel_class_ids"]+worker_ids))
                verified = []
                for system in SYSTEMS:
                    independent = make_classifier(system, {})
                    for row_id in ids:
                        result = independent.classify(cases[row_id]["text"])
                        if semantic_result(result) != semantic_result(read_result(store, system, row_id)):
                            raise AssertionError(f"Shared/independent mismatch: {system}/{row_id}")
                        verified.append(dict(system=system, row_id=row_id, exact_semantic_match=True))
                    del independent
                atomic_json(verify_path, dict(selection=plan["selection"], selected_ids=ids,
                    first_row_per_worker=worker_ids, comparisons=verified,
                    limitation="Sampled comparisons, not independent timing benchmarks"))
            parse_count, parse_seconds = store.connection.execute("SELECT COUNT(*),SUM(seconds) FROM parses").fetchone()
            atomic_json(args.output/"shared_preprocessing.json", dict(parse_rows=parse_count,
                parse_seconds=parse_seconds, imported=provenance, timing_policy=metadata["timing_policy"]))
        finally:
            store.export()
            atomic_json(init_path, initializations)
    from scripts.analyze_classical_500 import analyze
    analyze(args.output)
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
