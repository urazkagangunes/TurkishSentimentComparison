"""Balanced TRSAv1 development experiment: 400 rows/class, four classical systems.

Run: python -u -m tests.run_trsav1_classical_1200
The same command resumes durable successes. LLM systems are not accepted.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

from scripts.run_six_systems import digest, resource_manifest
from src.evaluation.checkpoint_store import CheckpointStore, atomic_json, read_status
from src.evaluation.dataset_loader import load_sentiment_dataset
from src.evaluation.resumable_evaluator import evaluate, make_classifier
from src.evaluation.run_identity import implementation_fingerprint

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ("hisnet", "dp_hisnet", "hisnet_dp", "lw_hisnet_dp")
SOURCE = ROOT / "datasets/Modified/sentence_sentiment/TRSAv1_fixed.csv"
DEFAULT_OUTPUT = ROOT / "results/trsav1_classical_1200_20261101_baseline"


def load_sample():
    dataset = load_sentiment_dataset(SOURCE, "review", "score")
    sample = load_sentiment_dataset(SOURCE, "review", "score", sample_per_class=400, random_state=42)
    distribution = sample["gold"].value_counts().to_dict()
    if distribution != {"negative": 400, "neutral": 400, "positive": 400}:
        raise ValueError("Expected exactly 400 negative, 400 neutral and 400 positive rows.")
    if sample.isna().any().any() or sample.duplicated().any():
        raise ValueError("Missing values or duplicate text/label pairs in sample.")
    label_counts = dataset.groupby("text")["gold"].nunique()
    conflicts = set(label_counts[label_counts > 1].index)
    source_ids = {pair: i for i, pair in enumerate(dataset.itertuples(index=False, name=None))}
    info = dict(
        origin="classwise_seed42_400_each_then_seed42_shuffle", random_state=42,
        sample_size=len(sample), source=str(SOURCE), source_sha256=digest(SOURCE),
        dataset_size=len(dataset), dataset_distribution=dataset["gold"].value_counts().to_dict(),
        sample_distribution=distribution, matches_seed42_reference=True,
        conflicting_source_label_row_ids=[i for i, text in enumerate(sample["text"]) if text in conflicts],
        sample_conflicting_texts=int((sample.groupby("text")["gold"].nunique() > 1).sum()),
        source_cleaned_row_ids=[source_ids[pair] for pair in sample.itertuples(index=False, name=None)],
        purpose="development_diagnostic_not_independent_final_test",
    )
    return sample, sample.to_dict("records"), info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if args.status:
        print(json.dumps(read_status(args.output), ensure_ascii=False, indent=2))
        return 0
    sample, cases, info = load_sample()
    print("SAMPLE", json.dumps({k: v for k, v in info.items() if k != "source_cleaned_row_ids"}, ensure_ascii=False), flush=True)
    metadata = dict(implementation=implementation_fingerprint(), runner_sha256=digest(__file__),
        helper_sha256=digest(ROOT / "scripts/run_six_systems.py"), sample=info,
        resources=resource_manifest(), keep_details=True, systems=list(SYSTEMS),
        protocol="balanced_classical_1200_frozen_defaults_independent_pipelines")
    with CheckpointStore(args.output, metadata, cases) as store:
        sample_path = args.output / "sample_1200.csv"
        sample.assign(sample_id=range(len(sample)), source_cleaned_row_id=info["source_cleaned_row_ids"])[
            ["sample_id", "source_cleaned_row_id", "text", "gold"]].to_csv(
                sample_path.with_suffix(".csv.tmp"), index=False, encoding="utf-8-sig")
        sample_path.with_suffix(".csv.tmp").replace(sample_path)
        atomic_json(args.output / "sampling.json", info)
        # Preserve the exact implementation for post-run comparisons. Never
        # overwrite an existing snapshot when resuming an experiment.
        for path in (ROOT / "src").rglob("*.py"):
            destination = args.output / "source_snapshot" / path.relative_to(ROOT)
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
        if args.prepare_only:
            store.export()
            print("Prepared 1200 cases; no classifier initialized.", flush=True)
            return 0

        def factory(system, settings):
            start = time.perf_counter()
            classifier = make_classifier(system, settings)
            path = args.output / "initializations.json"
            records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            records.append(dict(system=system, seconds=time.perf_counter() - start,
                                utc=datetime.now(timezone.utc).isoformat()))
            atomic_json(path, records)
            return classifier

        try:
            summary = evaluate(cases, store, {}, systems=tuple(dict.fromkeys(args.systems)),
                               factory=factory, keep_details=True, progress_every=25)
        except BaseException:
            store.export()
            raise
    from scripts.analyze_classical_500 import analyze
    analyze(args.output)
    return 0 if all(summary[s]["complete"] for s in args.systems) else 2


if __name__ == "__main__":
    raise SystemExit(main())
