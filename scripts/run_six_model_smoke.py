"""Live, manually labelled development smoke set; never a paper benchmark.

python -m scripts.run_six_model_smoke --output results/smoke_30_run01
Each model is checkpointed separately. Use a fresh directory for changed code.
"""
import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time
from datetime import datetime, timezone
from importlib.metadata import distributions

import httpx
import ollama
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.evaluation.run_identity import implementation_fingerprint, ensure_run_manifest
from src.hisnet.hisnet_classifier import HisNetClassifier
from src.hisnet.hisnet_dependency_classifier import HisNetDependencyClassifier
from src.hisnet.dependency_hisnet_classifier import DependencyHisNetClassifier
from src.hisnet.literature_weighted_hisnet_dependency_classifier import LiteratureWeightedHisNetDependencyClassifier
from src.llm.ollama_zero_shot_classifier import OllamaZeroShotSentimentClassifier
from src.llm.dependency_ollama_zero_shot_classifier import DependencyOllamaZeroShotSentimentClassifier

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ["hisnet", "dp_hisnet", "hisnet_dp", "lw_hisnet_dp", "llm", "dp_llm"]
LABELS = ["negative", "neutral", "positive"]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def resource_manifest():
    from stanza.resources.common import DEFAULT_MODEL_DIR
    import SentiNet
    import MorphologicalAnalysis
    import MorphologicalDisambiguation
    import Dictionary
    files = [Path(DEFAULT_MODEL_DIR) / "resources.json"]
    files.extend(Path(DEFAULT_MODEL_DIR).joinpath("tr").rglob("*.pt"))
    for module in [SentiNet, MorphologicalAnalysis, MorphologicalDisambiguation, Dictionary]:
        files.extend(p for p in Path(module.__file__).parent.joinpath("data").rglob("*") if p.is_file())
    return [{"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}
            for path in sorted(set(files)) if path.exists()]


def check_classical(system, result, baseline, hd_result=None):
    failures = []
    for key in ["positive_score", "negative_score"]:
        if not math.isfinite(result[key]) or result[key] < 0:
            failures.append("invalid_" + key)
    if result["prediction"] not in LABELS:
        failures.append("invalid_prediction")
    if system != "hisnet":
        for polarity in ["positive", "negative"]:
            if not math.isclose(sum(s["adjusted_" + polarity] for s in result["scopes"]),
                                result[polarity + "_score"], abs_tol=1e-9):
                failures.append("scope_sum_" + polarity)
    if system in {"hisnet_dp", "lw_hisnet_dp"}:
        for polarity in ["positive", "negative"]:
            if not math.isclose(sum(s["original_" + polarity] for s in result["scopes"]),
                                baseline[polarity + "_score"], abs_tol=1e-9):
                failures.append("baseline_preservation_" + polarity)
    if system == "lw_hisnet_dp" and hd_result:
        for polarity in ["positive", "negative"]:
            if not math.isclose(result["dependency_" + polarity + "_score"],
                                hd_result[polarity + "_score"], abs_tol=1e-9):
                failures.append("lw_base_differs_from_hd_" + polarity)
    return failures


def run_classical(cases, output):
    # Sequential initialization limits memory usage; timings include no fair
    # end-to-end performance comparison because H results are reused by H->DP/LW.
    all_results = {}
    hisnet = None
    for system in ["hisnet", "hisnet_dp", "dp_hisnet", "lw_hisnet_dp"]:
        path = output / (system + ".json")
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
        else:
            cached = {"rows": {}, "repeat_checks": {}, "initialization_seconds": None}
        pending = [case for case in cases if case["id"] not in cached["rows"]]
        if pending:
            print("INITIALIZING", system, flush=True)
            start = time.perf_counter()
            if system == "hisnet":
                hisnet = HisNetClassifier()
                classifier = hisnet
            elif system == "hisnet_dp":
                classifier = HisNetDependencyClassifier(hisnet_classifier=hisnet)
            elif system == "dp_hisnet":
                classifier = DependencyHisNetClassifier()
            else:
                classifier = LiteratureWeightedHisNetDependencyClassifier(hisnet_classifier=hisnet)
            cached["initialization_seconds"] = time.perf_counter() - start
            for case in pending:
                cid, text = case["id"], case["text"]
                baseline = all_results.get("hisnet", {}).get(cid, {}).get("result")
                baseline_copy = copy.deepcopy(baseline)
                start = time.perf_counter()
                try:
                    if system in {"hisnet_dp", "lw_hisnet_dp"}:
                        result = classifier.classify(text, hisnet_result=baseline)
                    else:
                        result = classifier.classify(text)
                    failures = check_classical(system, result, baseline,
                        all_results.get("hisnet_dp", {}).get(cid, {}).get("result"))
                    if baseline != baseline_copy:
                        failures.append("baseline_input_mutated")
                    record = dict(status="success", prediction=result["prediction"],
                                  result=result, invariant_failures=failures)
                except Exception as error:
                    record = dict(status="failed", prediction=None,
                                  error=f"{type(error).__name__}: {error}")
                record["seconds"] = time.perf_counter() - start
                cached["rows"][cid] = record
                save(path, cached)
                print(system, cid, record["prediction"], "expected", case["gold"], flush=True)
            for cid in ["U06", "N10", "N03", "P10", "P01"]:
                case = next(c for c in cases if c["id"] == cid)
                previous = cached["rows"][cid]
                if previous["status"] != "success":
                    continue
                try:
                    kwargs = (dict(hisnet_result=all_results["hisnet"][cid]["result"])
                              if system in {"hisnet_dp", "lw_hisnet_dp"} else {})
                    repeated = classifier.classify(case["text"], **kwargs)
                    keys = ["prediction", "positive_score", "negative_score", "morphological_analyses"]
                    cached["repeat_checks"][cid] = all(repeated.get(k) == previous["result"].get(k) for k in keys)
                except Exception as error:
                    cached["repeat_checks"][cid] = str(error)
            save(path, cached)
            del classifier
            gc.collect()
        all_results[system] = cached["rows"]
    return all_results


class RecordingClient:
    """Retain Ollama's actual per-attempt usage and timing metadata."""
    def __init__(self, host):
        self.client = ollama.Client(host=host, timeout=600)
        self.calls = []

    def chat(self, **kwargs):
        prompt = kwargs["messages"][0]["content"]
        record = {"prompt_utf8_bytes": len(prompt.encode("utf-8")),
                  "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}
        self.calls.append(record)
        start = time.perf_counter()
        try:
            result = self.client.chat(**kwargs)
            record.update({key: getattr(result, key, None) for key in
                           ["prompt_eval_count", "eval_count", "done_reason", "total_duration",
                            "load_duration", "prompt_eval_duration", "eval_duration"]})
            return result
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record["seconds"] = time.perf_counter() - start


def run_llm(cases, output, args):
    all_results = {}
    for system in ["llm", "dp_llm"]:
        path = output / (system + ".json")
        cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"rows": {}, "batches": []}
        pending = [c for c in cases if c["id"] not in cached["rows"]]
        if not pending:
            all_results[system] = cached["rows"]
            continue
        print("INITIALIZING", system, args.model, flush=True)
        cls = OllamaZeroShotSentimentClassifier if system == "llm" else DependencyOllamaZeroShotSentimentClassifier
        classifier = cls(model=args.model, host=args.host, batch_size=args.batch_size,
                         temperature=0.1, top_p=1.0, seed=42, num_ctx=args.num_ctx, max_retries=1)
        recorder = RecordingClient(args.host)
        classifier.client = recorder
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            print("BATCH", system, [c["id"] for c in batch], flush=True)
            first_call = len(recorder.calls)
            result = classifier.classify_batch([c["text"] for c in batch])
            calls = recorder.calls[first_call:]
            cached["batches"].append(dict(ids=[c["id"] for c in batch], result=result, calls=calls))
            for offset, case in enumerate(batch):
                cached["rows"][case["id"]] = dict(status=result["status"],
                    prediction=result["predictions"][offset], error=result["error"], recovered=False)
                print(system, case["id"], result["predictions"][offset], "expected", case["gold"], flush=True)
            cached["stats"] = classifier.get_stats()
            save(path, cached)
        for case in pending:
            if cached["rows"][case["id"]]["status"] == "success":
                continue
            first_call = len(recorder.calls)
            result = classifier.classify_batch([case["text"]])
            cached["batches"].append(dict(ids=[case["id"]], recovery=True, result=result,
                                           calls=recorder.calls[first_call:]))
            cached["rows"][case["id"]] = dict(status=result["status"], prediction=result["predictions"][0],
                                               error=result["error"], recovered=result["status"] == "success")
            cached["stats"] = classifier.get_stats()
            save(path, cached)
        all_results[system] = cached["rows"]
        recorder.client.close()
        del classifier
        gc.collect()
    return all_results


def summarize(cases, results, output):
    metrics = {}
    for system, rows in results.items():
        unresolved = [c["id"] for c in cases if rows[c["id"]]["prediction"] not in LABELS]
        metrics[system] = {"n": len(cases), "unresolved": unresolved}
        if not unresolved:
            gold = [c["gold"] for c in cases]
            pred = [rows[c["id"]]["prediction"] for c in cases]
            metrics[system].update(accuracy=accuracy_score(gold, pred),
                classification_report=classification_report(gold, pred, labels=LABELS, output_dict=True, zero_division=0),
                confusion_matrix=confusion_matrix(gold, pred, labels=LABELS).tolist())
    save(output / "summary.json", {"purpose": "development diagnostics, not a paper benchmark", "metrics": metrics})
    fields = ["id", "text", "gold", "focus", *SYSTEMS]
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in sorted(cases, key=lambda c: c["id"]):
            writer.writerow({**case, **{s: results.get(s, {}).get(case["id"], {}).get("prediction") for s in SYSTEMS}})
    print("SUMMARY", json.dumps(metrics, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen3:32b"))
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--phase", choices=["all", "classical", "llm"], default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fixture = ROOT / "tests/fixtures/six_model_smoke_30.json"
    cases = json.loads(fixture.read_text(encoding="utf-8"))
    random.Random(42).shuffle(cases)
    with httpx.Client(timeout=10, trust_env=False) as client:
        tags = client.get(args.host + "/api/tags").raise_for_status().json()
        model = next((m for m in tags["models"] if m["name"] == args.model), None)
        if model is None:
            raise ValueError("Requested model is not installed: " + args.model)
        server = client.get(args.host + "/api/version").raise_for_status().json()
        show = client.post(args.host + "/api/show", json={"model": args.model}).raise_for_status().json()
    manifest = dict(implementation=implementation_fingerprint(), runner_sha256=digest(__file__),
        fixture_sha256=digest(fixture), ordered_ids=[c["id"] for c in cases], model=model,
        server=server, settings=dict(seed=42, temperature=0.1, top_p=1., num_ctx=args.num_ctx,
        batch_size=args.batch_size, max_retries=1, think=False), resources=resource_manifest())
    ensure_run_manifest(args.output / "manifest.json", manifest,
                        [args.output / (s + ".json") for s in SYSTEMS])
    save(args.output / "environment.json", dict(python=platform.python_version(), platform=platform.platform(),
         packages=sorted((d.metadata.get("Name", ""), d.version) for d in distributions()),
         model_show=show, started=datetime.now(timezone.utc).isoformat()))
    save(args.output / "cases.json", cases)
    results = {}
    if args.phase in {"all", "classical"}:
        results.update(run_classical(cases, args.output))
    if args.phase in {"all", "llm"}:
        results.update(run_llm(cases, args.output, args))
    for system in SYSTEMS:
        path = args.output / (system + ".json")
        if system not in results and path.exists():
            results[system] = json.loads(path.read_text(encoding="utf-8"))["rows"]
    summarize(cases, results, args.output)


if __name__ == "__main__":
    main()
