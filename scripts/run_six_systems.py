"""Full-dataset evaluation with automatic, per-system durable resume.

Example: python -m scripts.run_six_systems --dataset trsav1 --systems all --output results/trsav1_seven
Running the same command again resumes it. --status never calls a model.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from src.evaluation.checkpoint_store import CheckpointStore, DEFAULT_SYSTEMS, SYSTEMS, atomic_json, read_status
from src.evaluation.resumable_evaluator import evaluate, LLM_SYSTEMS
from src.evaluation.run_identity import implementation_fingerprint

ROOT = Path(__file__).resolve().parents[1]
PAPER_DATASETS = {
    "beyazperde_sentiment": ("beyazperde_sentiment.csv", "Review", "Score"),
    "HUMIRSentimentDatasets": ("HUMIRSentimentDatasets.csv", "review", "score"),
    "TRSAv1_fixed": ("TRSAv1_fixed.csv", "review", "score"),
    "combined_sentiment": ("combined_sentiment.csv", "Review", "Score"),
    "tr_polarity_combined": ("tr_polarity_combined.csv", "review", "score"),
    "Turkish_twitter_combined": ("Turkish_twitter_combined.csv", "review", "score"),
}
PRESETS = {
    **PAPER_DATASETS,
    "trsav1": PAPER_DATASETS["TRSAv1_fixed"],
    "beyazperde": PAPER_DATASETS["beyazperde_sentiment"],
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
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--paper", action="store_true", help="Run the six datasets in the main paper benchmark.")
    source.add_argument("--dataset", choices=PRESETS)
    source.add_argument("--csv", type=Path)
    parser.add_argument("--text-column")
    parser.add_argument("--label-column")
    parser.add_argument("--separator", default=";")
    parser.add_argument("--sample-per-class", type=int)
    parser.add_argument("--model", help="Ollama model tag (paper default: qwen3:32b).")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--systems", choices=(*SYSTEMS, "all"), nargs="+",
                        help="Defaults to all seven in paper mode and the six local systems otherwise.")
    parser.add_argument("--jev-provider", choices=("typesafe", "opencode"),
                        help="Defaults to OpenCode's free Jev 1.13 in paper mode and TypeSafe otherwise.")
    parser.add_argument("--jev-model", help="Model ID (default: jev-1.14.0 or jev-1.13-free by provider).")
    parser.add_argument("--jev-timeout", type=float, default=30.0, help="Jev request timeout in seconds.")
    parser.add_argument("--keep-details", action="store_true", help="Also store full parses/scopes (large files).")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--status", action="store_true", help="Read saved progress; no models, no network.")
    parser.add_argument("--prepare-only", action="store_true", help="Load data and create checkpoint; no inference.")
    args = parser.parse_args(argv)
    if args.status:
        return args
    if args.systems is None:
        args.systems = SYSTEMS if args.paper else DEFAULT_SYSTEMS
    elif "all" in args.systems:
        if args.systems != ["all"]:
            parser.error("'--systems all' cannot be combined with individual system names.")
        args.systems = list(SYSTEMS)
    args.systems = tuple(dict.fromkeys(args.systems))
    if not (args.paper or args.dataset or args.csv):
        parser.error("Choose --paper, --dataset or --csv.")
    if args.csv and (not args.text_column or not args.label_column):
        parser.error("--csv requires --text-column and --label-column.")
    if args.batch_size < 1 or args.num_ctx < 1 or args.max_retries < 0 or args.progress_every < 1:
        parser.error("Batch/context/progress must be positive; retries must be nonnegative.")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite.")
    if not math.isfinite(args.jev_timeout) or args.jev_timeout <= 0:
        parser.error("--jev-timeout must be positive and finite.")
    if args.sample_per_class is not None and args.sample_per_class < 1:
        parser.error("--sample-per-class must be positive.")
    args.jev_provider = args.jev_provider or ("opencode" if args.paper else "typesafe")
    if args.model is None:
        args.model = "qwen3:32b" if args.paper else os.getenv("OLLAMA_MODEL", "qwen3:32b")
    return args


def run_dataset(args, dataset, output, runtime_identity):
    from src.evaluation.dataset_loader import load_sentiment_dataset
    if dataset:
        filename, text_column, label_column = PRESETS[dataset]
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
    metadata = dict(runtime_identity,
        input_sha256=digest(csv_path), text_column=text_column, label_column=label_column,
        separator=args.separator, sample_per_class=args.sample_per_class, settings=settings,
        keep_details=args.keep_details,
        context_policy="utf8_bytes_plus_1024_template_plus_128_and_64_per_review_output",
        recovery_policy="retry_failed_individually_once_per_invocation")
    with CheckpointStore(output, metadata, cases) as store:
        if args.paper or "jev" in args.systems:
            from src.llm.jev_zero_shot_classifier import (
                JEV_ENDPOINTS, SENTIMENT_QUESTION, resolve_jev_model,
            )
            jev_settings = dict(provider=args.jev_provider,
                                model=resolve_jev_model(args.jev_provider, args.jev_model),
                                request_timeout=args.jev_timeout)
            store.ensure_identity("jev_identity", dict(jev_settings, endpoint=JEV_ENDPOINTS[args.jev_provider],
                                                      question=SENTIMENT_QUESTION))
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
            return make_classifier(system, jev_settings if system == "jev" else config)

        try:
            summary = evaluate(cases, store, settings, systems=args.systems,
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


def paper_status(output):
    from src.evaluation.paper_checkpoint_metrics import read_paper_dataset

    return {name: read_paper_dataset(output / name) for name in PAPER_DATASETS}


def main(argv=None):
    args = parse_args(argv)
    if args.status:
        status = paper_status(args.output) if args.paper else read_status(args.output)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0

    runtime_identity = dict(implementation=implementation_fingerprint(), runner_sha256=digest(__file__),
                            resources=resource_manifest())
    if not args.paper:
        return run_dataset(args, args.dataset, args.output, runtime_identity)

    args.output.mkdir(parents=True, exist_ok=True)
    exit_status = 0
    try:
        for dataset in PAPER_DATASETS:
            print(f"Paper dataset {dataset}", flush=True)
            status = run_dataset(args, dataset, args.output / dataset, runtime_identity)
            if status == 130:
                return status
            exit_status = max(exit_status, status)
    except KeyboardInterrupt:
        return 130
    finally:
        atomic_json(args.output / "paper_summary.json", paper_status(args.output))
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
