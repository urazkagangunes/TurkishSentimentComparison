"""Read-only classifier audit; writes evidence, never changes model rules.

Run with --live-classical to include actual Stanza/HisNet probes. No Ollama
inference is sent: LLM failure/resume checks use isolated fake clients.
"""
import argparse
import ast
import contextlib
import io
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from src.dependency.sentiment_scope_resolver import SentimentScopeResolver
from src.evaluation.dataset_loader import load_sentiment_dataset
from src.hisnet.hisnet_classifier import HisNetClassifier
from src.hisnet.hisnet_dependency_classifier import HisNetDependencyClassifier
from src.hisnet.dependency_hisnet_classifier import DependencyHisNetClassifier
from src.hisnet.literature_weighted_hisnet_dependency_classifier import LiteratureWeightedHisNetDependencyClassifier
from src.llm.ollama_zero_shot_classifier import OllamaZeroShotSentimentClassifier
from scripts import run_six_model_smoke as smoke

ROOT = Path(__file__).resolve().parents[1]


def token(index, text, head=0, relation="root", pos="ADJ", sentence=0):
    return dict(id=index, text=text, lemma=text, head=head, deprel=relation,
                upos=pos, feats=None, sentence_index=sentence)


def offline_checks():
    resolver = SentimentScopeResolver()
    findings = {}
    tokens = [token(1, "kötü"), token(2, ",", 1, "punct", "PUNCT"),
              token(3, "merkezi", 1, "conj"), token(4, "değil", 3, "cop", "AUX")]
    scopes = resolver.resolve([dict(word="kötü", root="kötü", positive=0., negative=1.)], tokens)
    adjusted = HisNetDependencyClassifier._apply_dependency_scope(scopes[0])
    findings["unrelated_negation"] = dict(
        expected_negation_ids=[], observed_negation_ids=[n["id"] for n in scopes[0]["negations"]],
        expected_negative_score=1., observed_negative_score=adjusted["adjusted_negative"],
        observed_positive_score=adjusted["adjusted_positive"], tokens=tokens, scope=adjusted)

    morphology = [dict(word="yazar", root="yazar", analysis="yazar+NOUN+A3SG"),
                  dict(word="yazar", root="yaz", analysis="yaz+VERB+POS+AOR+A3SG")]
    tokens = [token(1, "yazar", pos="NOUN"), token(2, "yazar", pos="VERB", sentence=1)]
    scopes = resolver.resolve([dict(word="yazar", root="yaz", positive=1., negative=0.)], tokens, morphology)
    findings["sentiment_subset_occurrence"] = dict(
        expected_token_id=2, observed_token_id=scopes[0]["sentiment_token"]["id"],
        morphology=morphology, tokens=tokens)
    tokens = [token(1, "yazar🙂", pos="NOUN"), token(2, "yazar", pos="VERB", sentence=1)]
    candidates = resolver.resolve_dependency_candidates(tokens, morphology)
    findings["decorated_occurrence"] = dict(
        expected_roots=["yazar", "yaz"], observed_roots=[c["root"] for c in candidates],
        candidates=candidates, morphology=morphology, tokens=tokens)

    # A failed batch was persisted, then the process was interrupted before
    # recovery. A later resume should not forget that recovery is outstanding.
    with TemporaryDirectory(dir=ROOT / ".audit_cache") as temporary:
        output = Path(temporary)
        cases = [dict(id="A", text="first", gold="positive"), dict(id="B", text="second", gold="positive")]
        failed = dict(prediction=None, status="failed", error="interrupted before recovery", recovered=False)
        success = dict(prediction="positive", status="success", error=None, recovered=False)
        smoke.save(output / "llm.json", dict(rows={"A": failed}, batches=[]))
        smoke.save(output / "dp_llm.json", dict(rows={"A": success, "B": success}, batches=[]))
        fake = Mock()
        fake.classify_batch.return_value = dict(predictions=["positive"], status="success", error=None)
        fake.get_stats.return_value = {}
        recorder = Mock(calls=[])
        with patch.object(smoke, "OllamaZeroShotSentimentClassifier", return_value=fake), \
             patch.object(smoke, "RecordingClient", return_value=recorder), contextlib.redirect_stdout(io.StringIO()):
            results = smoke.run_llm(cases, output, SimpleNamespace(model="fake", host="unused", batch_size=1, num_ctx=8192))
        findings["resume_skips_unrecovered_failure"] = dict(
            expected_recovered_ids=["A"], observed_failed_ids=[key for key, row in results["llm"].items() if row["status"] == "failed"],
            calls=[call.args[0] for call in fake.classify_batch.call_args_list])

    classifier = OllamaZeroShotSentimentClassifier(max_retries=0)
    findings["default_client_timeout"] = dict(value=repr(classifier.client._client.timeout),
                                               scope="default classifier, not smoke RecordingClient")
    classifier.client._client.close()

    # Execute only the existing function definition, never import a runner
    # that starts a 500-row experiment at module level.
    path = ROOT / "tests/test_four_systems_500.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_llm_batch_condition")
    with TemporaryDirectory(dir=ROOT / ".audit_cache") as temporary:
        output = Path(temporary)
        sample = pd.DataFrame([dict(sample_id=0, text="saved review", gold="positive")])
        sample.assign(prediction="positive", status="success", recovered=False, error=None).to_csv(
            output / "qwen3_32b_batch_1_predictions.csv", index=False)
        fake = Mock()
        fake.get_stats.return_value = dict(total_requests=0, total_retries=0, total_failures=0,
                                           total_prompt_tokens=0, total_output_tokens=0)
        namespace = dict(pd=pd, time=time, OUTPUT_DIR=output, sample=sample,
                         MODEL_NAME="fake", TEMPERATURE=.1, TOP_P=1., NUM_CTX=8192,
                         KEEP_ALIVE="30m", MAX_RETRIES=1, SEED=42,
                         OllamaZeroShotSentimentClassifier=lambda **kwargs: fake,
                         calculate_metrics=lambda frame, column: dict(n=len(frame), accuracy=1.))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        with contextlib.redirect_stdout(io.StringIO()):
            _, benchmark = namespace["run_llm_batch_condition"](1)
        findings["legacy_resume_timing"] = dict(
            previously_completed_reviews=1, new_inference_calls=fake.classify_batch.call_count,
            benchmark=benchmark,
            expected="Retain previous timing or explicitly mark cached-run timing unavailable.")
    parsed = []
    for folder in ("src", "scripts", "tests"):
        for path in (ROOT / folder).rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            parsed.append(str(path.relative_to(ROOT)))
    findings["syntax"] = dict(parsed_count=len(parsed), all_passed=True)
    return findings


def live_checks():
    hisnet = HisNetClassifier()
    hd = HisNetDependencyClassifier(hisnet_classifier=hisnet)
    lw = LiteratureWeightedHisNetDependencyClassifier.__new__(LiteratureWeightedHisNetDependencyClassifier)
    lw.hisnet, lw.parser, lw.scope_resolver = hisnet, hd.parser, hd.scope_resolver
    dh = DependencyHisNetClassifier.__new__(DependencyHisNetClassifier)
    dh.hisnet, dh.parser, dh.scope_resolver = hisnet, hd.parser, hd.scope_resolver
    examples = ["Bu filmi sevdim.", "Bu filmi sevmedim.", "Bu ürünü beğendim.",
                "Bu ürünü beğenmedim.", "Otel kötü, merkezi değil.", "Otel kötü. Merkezi değil.",
                "Otel güzel, merkezi değil.", "Otel kötü ve merkezi değil.",
                "Bu bir aşk filmi değil.", "Bu bir korku filmi değil.",
                "Hizmet iyi değil, mükemmel.", "Otel çok çok güzel.",
                "Otel iyi ve güzel değildi.", "Otel güzel ama iyi değildi.",
                "Pahalı olmasına rağmen güzel.", "Güzel olmasına rağmen pahalı."]
    rows = []
    for text in examples:
        baseline = hisnet.classify(text)
        results = {"hisnet": baseline, "hisnet_dp": hd.classify(text, hisnet_result=baseline),
                   "dp_hisnet": dh.classify(text), "lw_hisnet_dp": lw.classify(text, hisnet_result=baseline)}
        rows.append(dict(text=text, results=results))
        print(json.dumps(dict(text=text, predictions={key: value["prediction"] for key, value in results.items()}), ensure_ascii=False), flush=True)
    lookup = {root: hisnet._lookup_root(root) for root in ("sev", "sevmek", "beğen", "beğenmek", "aşk", "korku", "merkezi")}
    return dict(rows=rows, lexicon=lookup)


def dataset_checks():
    results = {}
    for filename, text_column, label_column in [("TRSAv1_fixed.csv", "review", "score"),
                                                 ("beyazperde_sentiment.csv", "Review", "Score")]:
        path = ROOT / "datasets/Modified/sentence_sentiment" / filename
        frame = load_sentiment_dataset(path, text_column, label_column)
        labels_per_text = frame.groupby("text")["gold"].nunique()
        conflicting_texts = labels_per_text[labels_per_text > 1].index
        results[filename] = dict(sha256=smoke.digest(path), loaded_rows=len(frame),
            label_counts=frame["gold"].value_counts().to_dict(),
            conflicting_texts=len(conflicting_texts),
            conflicting_rows=int(frame["text"].isin(conflicting_texts).sum()))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live-classical", action="store_true")
    parser.add_argument("--datasets", action="store_true")
    args = parser.parse_args()
    (ROOT / ".audit_cache").mkdir(exist_ok=True)
    evidence = dict(offline=offline_checks())
    if args.live_classical:
        evidence["live_classical"] = live_checks()
    if args.datasets:
        evidence["datasets"] = dataset_checks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print("EVIDENCE", args.output, flush=True)


if __name__ == "__main__":
    main()
