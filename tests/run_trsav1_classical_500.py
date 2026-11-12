"""Frozen 500-row TRSAv1 experiment: four classical systems, durable resume.

Run explicitly: python -u -m tests.run_trsav1_classical_500
Importing this file does not initialize models or run an experiment.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from scripts.run_six_systems import digest, resource_manifest
from src.evaluation.checkpoint_store import CheckpointStore, atomic_json, read_status
from src.evaluation.resumable_evaluator import evaluate, make_classifier
from src.evaluation.run_identity import implementation_fingerprint

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ("hisnet", "dp_hisnet", "hisnet_dp", "lw_hisnet_dp")
DEFAULT_OUTPUT = ROOT / "results/trsav1_classical_500_20261031_run01"
SOURCE = ROOT / "datasets/Modified/sentence_sentiment/TRSAv1_fixed.csv"
CANONICAL = ROOT / "results/trsav1_four_systems_500/sample_500.csv"


def load_sample():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from src.evaluation.dataset_loader import load_sentiment_dataset
    dataset = load_sentiment_dataset(SOURCE, "review", "score", separator=";")
    _, reference = train_test_split(dataset, test_size=500, random_state=42, stratify=dataset["gold"])
    reference = reference.reset_index(drop=True)
    if CANONICAL.exists():
        sample = pd.read_csv(CANONICAL, encoding="utf-8-sig")
        if "sample_id" not in sample or sample["sample_id"].tolist() != list(range(500)):
            raise ValueError("Canonical sample IDs/order must be exactly 0..499.")
        sample = sample[["text", "gold"]].copy()
        origin = str(CANONICAL)
    else:
        sample = reference
        origin = "stratified_seed42"
    if len(sample) != 500 or sample.isna().any().any() or sample.duplicated().any():
        raise ValueError("Sample must contain exactly 500 unique, nonempty text/gold pairs.")
    valid = set(dataset[["text", "gold"]].itertuples(index=False, name=None))
    if any(pair not in valid for pair in sample.itertuples(index=False, name=None)):
        raise ValueError("Canonical sample contains text/labels absent from the current cleaned dataset.")
    cases = sample.to_dict("records")
    text_labels = dataset.groupby("text")["gold"].nunique()
    conflicts = set(text_labels[text_labels > 1].index)
    info = dict(origin=origin, sample_size=500, random_state=42,
        matches_seed42_reference=sample.equals(reference),
        source_sha256=digest(SOURCE), dataset_size=len(dataset),
        dataset_distribution=dataset["gold"].value_counts().to_dict(),
        sample_distribution=dict(Counter(c["gold"] for c in cases)),
        conflicting_source_label_row_ids=[i for i, c in enumerate(cases) if c["text"] in conflicts],
        sample_conflicting_texts=int((sample.groupby("text")["gold"].nunique() > 1).sum()))
    return sample, cases, info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)
    if args.status:
        print(json.dumps(read_status(args.output), ensure_ascii=False, indent=2))
        return 0
    sample, cases, info = load_sample()
    print("SAMPLE", json.dumps(info, ensure_ascii=False), flush=True)
    metadata = dict(implementation=implementation_fingerprint(), runner_sha256=digest(__file__),
        helper_sha256=digest(ROOT / "scripts/run_six_systems.py"), sample=info,
        resources=resource_manifest(), keep_details=True, systems=list(SYSTEMS),
        protocol="frozen_classical_500_no_threshold_tuning_independent_pipelines")
    with CheckpointStore(args.output, metadata, cases) as store:
        sample_path = args.output / "sample_500.csv"
        sample.assign(sample_id=range(500))[["sample_id", "text", "gold"]].to_csv(
            sample_path.with_suffix(".csv.tmp"), index=False, encoding="utf-8-sig")
        sample_path.with_suffix(".csv.tmp").replace(sample_path)
        atomic_json(args.output / "sampling.json", info)

        def factory(system, settings):
            start = time.perf_counter()
            classifier = make_classifier(system, settings)
            path = args.output / "initializations.json"
            records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            records.append(dict(system=system, seconds=time.perf_counter()-start,
                                utc=datetime.now(timezone.utc).isoformat()))
            atomic_json(path, records)
            return classifier

        try:
            summary = evaluate(cases, store, {}, systems=tuple(dict.fromkeys(args.systems)),
                               factory=factory, keep_details=True, progress_every=25)
        except KeyboardInterrupt:
            store.export()
            print("Interrupted; repeat the same command to resume.", flush=True)
            return 130
        except Exception:
            store.export()
            raise
    from scripts.analyze_classical_500 import analyze
    analyze(args.output)
    return 0 if all(summary[s]["complete"] for s in args.systems) else 2


if __name__ == "__main__":
    raise SystemExit(main())
