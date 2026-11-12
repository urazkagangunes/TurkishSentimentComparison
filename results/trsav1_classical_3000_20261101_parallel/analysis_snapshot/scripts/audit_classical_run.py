"""Read-only progress and post-run identity/completeness audit."""
import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3


def close_interrupted(output):
    """Acquire the writer lock, mark unfinished attempts, preserve successes."""
    from src.evaluation.checkpoint_store import CheckpointStore, atomic_json
    database = output.resolve()/"progress.sqlite3"
    query = ("SELECT p.system,p.row_id,a.result,a.seconds FROM predictions p "
             "JOIN attempts a ON a.id=p.attempt_id WHERE p.status='success' ORDER BY p.system,p.row_id")
    with closing(sqlite3.connect(database.as_uri()+"?mode=ro", uri=True)) as db:
        identity = json.loads(db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
        cases = [dict(text=t, gold=g) for t,g in db.execute("SELECT text,gold FROM cases ORDER BY row_id")]
        successes = list(db.execute(query))
    metadata = {k:v for k,v in identity.items() if k not in {"checkpoint_version", "dataset_sha256", "total_examples"}}
    with CheckpointStore(output, metadata, cases) as store:
        assert list(store.connection.execute(query)) == successes
        store.export()
    atomic_json(output/"interruption.json", dict(reason="transition to parallel review workers",
        preserved_successful_predictions=len(successes), successful_payloads_and_timings_unchanged=True,
        utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(closed=True, preserved_successful_predictions=len(successes))))


def inspect(output, progress=False):
    output = output.resolve()
    database = output / "progress.sqlite3"
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        counts = [dict(system=s, status=t, count=n) for s, t, n in db.execute(
            "SELECT system,status,COUNT(*) FROM predictions GROUP BY system,status")]
        active = [dict(system=s, row_ids=json.loads(ids), started=start) for s, ids, start in db.execute(
            "SELECT system,row_ids,started FROM attempts WHERE status='running'")]
        start = db.execute("SELECT MIN(started) FROM attempts").fetchone()[0]
        wall = (datetime.now(timezone.utc)-datetime.fromisoformat(start)).total_seconds() if start else 0
        timing = [dict(system=s, seconds=round(t, 2)) for s, t in db.execute(
            "SELECT system,COALESCE(SUM(seconds),0) FROM attempts GROUP BY system")]
        state = dict(counts=counts, active=active, wall_minutes=round(wall/60, 2), recorded_timing=timing)
        if progress:
            print(json.dumps(state, ensure_ascii=False))
            return state
        manifest = json.loads(db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
        cases = [dict(text=t, gold=g) for t, g in db.execute("SELECT text,gold FROM cases ORDER BY row_id")]
        parse_count = db.execute("SELECT COUNT(*) FROM parses").fetchone()[0]
        attempts = list(db.execute("SELECT system,status,COUNT(*) FROM attempts GROUP BY system,status"))
    from src.evaluation.run_identity import implementation_fingerprint
    implementation = implementation_fingerprint()  # Before SentiNet alters sys.path.
    from scripts.run_six_systems import digest, resource_manifest
    from src.evaluation.checkpoint_store import atomic_json, dataset_digest
    from src.evaluation.dataset_loader import load_sentiment_dataset
    import pandas as pd
    root = Path(__file__).resolve().parents[1]
    frozen_hash = digest(database)
    systems = {"hisnet", "dp_hisnet", "hisnet_dp", "lw_hisnet_dp"}
    assert set(manifest["systems"]) == systems
    assert {r["system"] for r in counts} == systems
    assert all(r["status"] == "success" and r["count"] == len(cases) for r in counts)
    assert not active and parse_count == len(cases)
    assert not any(s in {"llm", "dp_llm"} for s, _, _ in attempts)
    assert manifest["implementation"] == implementation
    assert manifest["dataset_sha256"] == dataset_digest(cases)
    assert json.loads((output/"manifest.json").read_text(encoding="utf-8")) == manifest
    sample = pd.read_csv(output/f"sample_{len(cases)}.csv", encoding="utf-8-sig")
    assert sample[["text", "gold"]].to_dict("records") == cases
    sampling = manifest["sample"]
    assert len(cases) == sampling["sample_size"] == manifest["total_examples"]
    assert dict(Counter(c["gold"] for c in cases)) == sampling["sample_distribution"]
    assert not sample["text"].duplicated().any()
    assert digest(sampling["source"]) == sampling["source_sha256"]
    dataset = load_sentiment_dataset(sampling["source"], "review", "score")
    assert dataset.iloc[sampling["source_cleaned_row_ids"]].to_dict("records") == cases
    assert sample["source_cleaned_row_id"].tolist() == sampling["source_cleaned_row_ids"]
    for excluded in sampling.get("exclusions", []):
        assert digest(excluded["path"]) == excluded["sha256"]
        old = pd.read_csv(excluded["path"])["text"].astype(str).str.strip()
        assert not set(old) & set(sample["text"])
    snapshots = list((output/"source_snapshot").rglob("*.py"))
    for frozen in snapshots:
        assert digest(frozen) == digest(root/frozen.relative_to(output/"source_snapshot")), str(frozen)
    assert digest(root/manifest.get("runner_path", "scripts/run_balanced_classical_cached.py")) == manifest["runner_sha256"]
    assert digest(root/"scripts/balanced_classical_sample.py") == manifest["disjoint_sampler_sha256"]
    assert resource_manifest() == manifest["resources"]
    if manifest.get("imported_source"):
        assert digest(Path(manifest["imported_source"])/"progress.sqlite3") == manifest["imported_source_sha256"]
    verification = json.loads((output/"independent_verification.json").read_text(encoding="utf-8"))
    verified = verification["comparisons"]
    first_ids = {next(i for i, c in enumerate(cases) if c["gold"] == label)
                 for label in ("negative", "neutral", "positive")}
    expected_ids = set(verification.get("selected_ids", first_ids))
    assert first_ids <= expected_ids and all(v["exact_semantic_match"] for v in verified)
    assert len(verified) == len(expected_ids)*4
    assert {(v["system"], v["row_id"]) for v in verified} == {(s, i) for s in systems for i in expected_ids}
    if "workers" in manifest:
        plan = json.loads((output/"verification_plan.json").read_text(encoding="utf-8"))
        with closing(sqlite3.connect(database.as_uri()+"?mode=ro", uri=True)) as db:
            worker_ids = [r[0] for r in db.execute("SELECT MIN(row_id) FROM worker_assignments GROUP BY worker_pid")]
        assert expected_ids == set(plan["base_ids"]+plan["parallel_class_ids"]+worker_ids)
        preflight = json.loads((output/"parallel_preflight.json").read_text(encoding="utf-8"))["comparisons"]
        assert len(preflight)==12 and all(p["exact_semantic_match"] for p in preflight)
    analysis = json.loads((output/"analysis.json").read_text(encoding="utf-8"))
    assert analysis["complete"] and analysis["n"] == len(cases)
    assert len(analysis["metrics"]) == 4 and all(m["n"] == len(cases) for m in analysis["metrics"])
    assert not analysis["violations"], analysis["violations"][:5]
    assert digest(database) == frozen_hash
    analysis_code = {}
    for relative in ("scripts/analyze_classical_500.py", "scripts/audit_classical_run.py",
                     "scripts/validate_classical_corrections.py", "scripts/check_saved_negation_scope.py"):
        frozen = output/"analysis_snapshot"/relative
        frozen.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/relative, frozen)
        analysis_code[relative] = digest(frozen)
    report = dict(passed=True, examples=len(cases), successful_predictions=len(cases)*4,
        llm_attempts=0, parse_rows=parse_count, source_unchanged=True, database_sha256=frozen_hash,
        implementation=implementation, snapshot_files=len(snapshots), resources=len(manifest["resources"]),
        independent_full_semantic_matches=len(verified), excluded_text_overlap=0,
        source_label_conflict_rows=len(sampling["conflicting_source_label_row_ids"]),
        timing_caveat_recorded=(output/"timing_caveats.json").exists(), analysis_code_sha256=analysis_code,
        attempts=attempts, checks="source IDs, labels, disjoint texts, source/code/resource hashes, complete outputs, numeric invariants, independent pipeline verification",
        utc=datetime.now(timezone.utc).isoformat())
    atomic_json(output/"final_audit.json", report)
    print(json.dumps(report, ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--close-interrupted", action="store_true",
                        help="Acquire writer lock and close unfinished attempts; preserve every successful output")
    args = parser.parse_args()
    if args.close_interrupted:
        close_interrupted(args.output)
    else:
        inspect(args.output, args.progress)
