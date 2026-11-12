"""Full-dataset evaluation with automatic, per-system durable resume.

Example: python -m scripts.run_six_systems --dataset trsav1 --output results/trsav1_six
Running the same command again resumes it. --status never calls a model.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from src.evaluation.checkpoint_store import CheckpointStore, SYSTEMS, read_status
from src.evaluation.resumable_evaluator import evaluate, LLM_SYSTEMS
from src.evaluation.run_identity import implementation_fingerprint

ROOT = Path(__file__).resolve().parents[1]
PRESETS = {
    "trsav1": ("TRSAv1_fixed.csv", "review", "score"),
    "beyazperde": ("beyazperde_sentiment.csv", "Review", "Score"),
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def resource_manifest():
    # Hash resources without constructing Stanza/HisNet or generating text.
    from stanza.resources.common import DEFAULT_MODEL_DIR
    import SentiNet
    import MorphologicalAnalysis
    import MorphologicalDisambiguation
    import Dictionary
    files = [Path(DEFAULT_MODEL_DIR) / "resources.json"]
    files.extend(Path(DEFAULT_MODEL_DIR).joinpath("tr").rglob("*.pt"))
    for module in (SentiNet, MorphologicalAnalysis, MorphologicalDisambiguation, Dictionary):
        files.extend(p for p in Path(module.__file__).parent.joinpath("data").rglob("*") if p.is_file())
    return [{"path": str(path), "sha256": digest(path)} for path in sorted(set(files)) if path.exists()]


def ollama_identity(host, model):
    import httpx
    with httpx.Client(base_url=host, timeout=15, trust_env=False) as client:
        tags = client.get("/api/tags").raise_for_status().json()
        entry = next((m for m in tags["models"] if m["name"] == model), None)
        if entry is None or not entry.get("digest"):
            raise ValueError("Requested model is not installed with a known digest: " + model)
        version = client.get("/api/version").raise_for_status().json()
        show = client.post("/api/show", json={"model": model}).raise_for_status().json()
    return dict(model=model, digest=entry["digest"], server=version, configuration=show)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", choices=PRESETS)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--text-column")
    parser.add_argument("--label-column")
    parser.add_argument("--separator", default=";")
    parser.add_argument("--sample-per-class", type=int)
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen3:32b"))
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--systems", choices=SYSTEMS, nargs="+", default=list(SYSTEMS))
    parser.add_argument("--keep-details", action="store_true", help="Also store full parses/scopes (large files).")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--status", action="store_true", help="Read saved progress; no models, no network.")
    parser.add_argument("--prepare-only", action="store_true", help="Load data and create checkpoint; no inference.")
    args = parser.parse_args(argv)
    if args.status:
        return args
    if bool(args.dataset) == bool(args.csv):
        parser.error("Choose exactly one of --dataset or --csv.")
    if args.csv and (not args.text_column or not args.label_column):
        parser.error("--csv requires --text-column and --label-column.")
    if args.batch_size < 1 or args.num_ctx < 1 or args.max_retries < 0 or args.progress_every < 1:
        parser.error("Batch/context/progress must be positive; retries must be nonnegative.")
    import math
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite.")
    if args.sample_per_class is not None and args.sample_per_class < 1:
        parser.error("--sample-per-class must be positive.")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.status:
        print(json.dumps(read_status(args.output), ensure_ascii=False, indent=2))
        return 0
    from src.evaluation.dataset_loader import load_sentiment_dataset
    if args.dataset:
        filename, text_column, label_column = PRESETS[args.dataset]
        csv_path = ROOT / "datasets/Modified/sentence_sentiment" / filename
    else:
        csv_path, text_column, label_column = args.csv, args.text_column, args.label_column
    dataframe = load_sentiment_dataset(csv_path, text_column, label_column, separator=args.separator,
        sample_per_class=args.sample_per_class, random_state=args.seed)
    cases = dataframe[["text", "gold"]].to_dict("records")
    if not cases:
        raise ValueError("Dataset has no usable rows.")
    settings = dict(model=args.model, host=args.host, batch_size=args.batch_size, num_ctx=args.num_ctx,
        max_retries=args.max_retries, request_timeout=args.timeout, seed=args.seed,
        temperature=0.1, top_p=1.0, keep_alive="30m")
    metadata = dict(implementation=implementation_fingerprint(), runner_sha256=digest(__file__),
        input_sha256=digest(csv_path), text_column=text_column, label_column=label_column,
        separator=args.separator, sample_per_class=args.sample_per_class, settings=settings,
        resources=resource_manifest(), keep_details=args.keep_details,
        context_policy="utf8_bytes_plus_1024_template_plus_128_and_64_per_review_output",
        recovery_policy="retry_failed_individually_once_per_invocation")
    with CheckpointStore(args.output, metadata, cases) as store:
        if args.prepare_only:
            store.export()
            print(f"Prepared {len(cases)} rows. No classifiers were run.")
            return 0
        checked_llm = False

        def factory(system, config):
            nonlocal checked_llm
            from src.evaluation.resumable_evaluator import make_classifier
            if system in LLM_SYSTEMS and not checked_llm:
                store.ensure_identity("ollama_identity", ollama_identity(args.host, args.model))
                checked_llm = True
            return make_classifier(system, config)

        try:
            summary = evaluate(cases, store, settings, systems=tuple(dict.fromkeys(args.systems)),
                factory=factory, keep_details=args.keep_details, progress_every=args.progress_every)
        except KeyboardInterrupt:
            store.export()
            print("Interrupted. Saved rows are safe; repeat the same command to resume.", flush=True)
            return 130
        except Exception:
            store.export()
            raise
        for system in args.systems:
            item = summary[system]
            print(f"{system}: success={item['success']} failed={item['failed']} pending={item['pending']}")
        return 0 if all(summary[s]["complete"] for s in args.systems) else 2


if __name__ == "__main__":
    raise SystemExit(main())
