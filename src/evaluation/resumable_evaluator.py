"""Shared execution loop; no model or parser is constructed on import."""
import gc
import time

from src.evaluation.checkpoint_store import DEFAULT_SYSTEMS, LABELS

LLM_SYSTEMS = {"llm", "dp_llm"}


def make_classifier(system, settings):
    if system == "hisnet":
        from src.hisnet.hisnet_classifier import HisNetClassifier
        return HisNetClassifier()
    if system == "hisnet_dp":
        from src.hisnet.hisnet_dependency_classifier import HisNetDependencyClassifier
        return HisNetDependencyClassifier()
    if system == "dp_hisnet":
        from src.hisnet.dependency_hisnet_classifier import DependencyHisNetClassifier
        return DependencyHisNetClassifier()
    if system == "lw_hisnet_dp":
        from src.hisnet.literature_weighted_hisnet_dependency_classifier import LiteratureWeightedHisNetDependencyClassifier
        return LiteratureWeightedHisNetDependencyClassifier()
    if system == "llm":
        from src.llm.ollama_zero_shot_classifier import OllamaZeroShotSentimentClassifier
        return OllamaZeroShotSentimentClassifier(**settings)
    if system == "dp_llm":
        from src.llm.dependency_ollama_zero_shot_classifier import DependencyOllamaZeroShotSentimentClassifier
        return DependencyOllamaZeroShotSentimentClassifier(**settings)
    if system == "jev":
        from src.llm.jev_zero_shot_classifier import JevZeroShotSentimentClassifier
        return JevZeroShotSentimentClassifier(**settings)
    raise ValueError("Unknown system: " + system)


def classify_result(classifier, system, texts):
    if system in LLM_SYSTEMS:
        result = classifier.classify_batch(texts)
    else:
        result = dict(classifier.classify(texts[0]))
        result["predictions"] = [result["prediction"]]
        result.setdefault("status", "success")
        result.setdefault("error", None)
    if (result.get("status") not in {"success", "failed"}
            or len(result.get("predictions", [])) != len(texts)
            or (result["status"] == "success" and any(p not in LABELS for p in result["predictions"]))
            or (result["status"] == "failed" and any(p is not None for p in result["predictions"]))):
        raise ValueError("Classifier returned an invalid result contract.")
    return result


def evaluate(cases, store, settings, systems=DEFAULT_SYSTEMS, factory=make_classifier,
             keep_details=False, progress_every=100):
    """Resume successes; retry failures once individually per invocation.

    New LLM work keeps original batch boundaries across restarts. Initialization
    is lazy, so an already completed system never loads a model or calls Ollama.
    Ctrl+C/process death preserves all committed predictions. The active request
    is recorded as interrupted on the next startup and can be repeated.
    """
    for system in systems:
        states = store.states(system)
        if len(states) == len(cases) and all(s == "success" for s in states.values()):
            print(f"{system}: complete, skipped", flush=True)
            continue
        print(f"{system}: initializing; saved successes={sum(s == 'success' for s in states.values())}", flush=True)
        classifier = factory(system, settings)
        processed = 0

        def run(ids, recovery=False):
            nonlocal processed
            attempt_id = store.start(system, ids, recovery)
            start = time.perf_counter()
            try:
                result = classify_result(classifier, system, [cases[i]["text"] for i in ids])
            except Exception as error:
                # Do not catch KeyboardInterrupt/SystemExit: the in-flight
                # attempt stays pending rather than becoming a sentiment label.
                result = dict(status="failed", predictions=[None] * len(ids),
                              error=f"{type(error).__name__}: {error}")
            seconds = time.perf_counter() - start
            stop_run = result.pop("stop_run", False)
            if not keep_details:
                fields = {"predictions", "status", "error", "error_code", "model", "attempt", "raw_response",
                          "request_seconds", "prompt_tokens", "output_tokens", "dependency_parse_seconds",
                          "total_pipeline_seconds", "positive_score", "negative_score", "probabilities", "confidence",
                          "cost", "billing_source"}
                result = {key: value for key, value in result.items() if key in fields}
            store.finish(attempt_id, result, seconds)
            for i in ids:
                states[i] = result["status"]
            processed += len(ids)
            if processed % progress_every < len(ids) or result["status"] == "failed":
                print(f"{system}: saved {processed} this session; last rows={ids}; {result['status']}", flush=True)
            if stop_run:
                raise RuntimeError(f"{system}: {result['error']} Saved rows can be resumed.")

        try:
            batch_size = settings["batch_size"] if system in LLM_SYSTEMS else 1
            for start in range(0, len(cases), batch_size):
                ids = [i for i in range(start, min(start + batch_size, len(cases))) if i not in states]
                if ids:
                    run(ids)
            # Includes failures from previous sessions, even if no new rows ran.
            for i in sorted(i for i, status in states.items() if status == "failed"):
                run([i], recovery=True)
        finally:
            client = getattr(classifier, "client", None)
            if client is not None:
                client.close()
            del classifier
            gc.collect()
        store.export()
    return store.export()
