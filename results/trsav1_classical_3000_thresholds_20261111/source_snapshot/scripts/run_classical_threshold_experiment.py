"""Sequential threshold trials with frozen preprocessing and production composition.

Reproduce every saved 0.05 result before trying 0.07 and 0.10. The source
checkpoint and production files stay unchanged. This is not fresh end-to-end
inference: morphology, parses and lexical lookup values come from the source.
"""
import argparse
from collections import Counter
from contextlib import ExitStack, closing, contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import csv
import importlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import time
from types import SimpleNamespace
from unittest.mock import patch

from scripts.run_balanced_classical_cached import semantic_result
from scripts.run_six_systems import digest
from scripts.validate_classical_corrections import replay
from src.dependency.sentiment_scope_resolver import SentimentScopeResolver
from src.dependency.stanza_parser import TurkishDependencyParser
from src.evaluation.checkpoint_store import CheckpointStore, atomic_json
from src.hisnet.hisnet_classifier import HisNetClassifier

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ("hisnet", "dp_hisnet", "hisnet_dp", "lw_hisnet_dp")
NAMES = dict(zip(SYSTEMS, ("HisNet", "DP → HisNet", "HisNet → DP", "LW HisNet → DP")))
LABELS = ("negative", "neutral", "positive")
BASELINE = 0.05
THRESHOLDS = (0.07, 0.10)


@contextmanager
def decision_threshold(value):
    if not math.isfinite(value) or value < 0:
        raise ValueError("Threshold must be finite and nonnegative")
    with ExitStack() as stack:
        for name in ("hisnet_classifier", "dependency_hisnet_classifier", "hisnet_dependency_classifier"):
            module = importlib.import_module("src.hisnet." + name)
            stack.enter_context(patch.object(module, "NEUTRAL_THRESHOLD", value))
        yield


def independent_label(positive, negative, threshold):
    if any(not math.isfinite(v) or v < 0 for v in (positive, negative)):
        raise ValueError("Invalid polarity score")
    net = positive - negative
    if abs(net) <= threshold or math.isclose(abs(net), threshold, rel_tol=0, abs_tol=1e-12):
        return "neutral"
    return "positive" if net > 0 else "negative"


def replay_hisnet(saved):
    lookups = {}
    for entry in saved["matched_words"] + saved["unmatched_words"]:
        value = ({k: entry[k] for k in ("positive", "negative")} if "positive" in entry else None)
        root = entry["root"]
        if root in lookups and lookups[root] != value:
            raise AssertionError("Conflicting saved lookup: " + str(root))
        lookups[root] = value
    classifier = HisNetClassifier.__new__(HisNetClassifier)
    classifier.root_extractor = SimpleNamespace(
        analyze_sentence=lambda text: deepcopy(saved["morphological_analyses"]))
    # Deliberately use []: a missing frozen lookup must fail, not become neutral.
    classifier._lookup_root = lambda root: deepcopy(lookups[root])
    return classifier.classify(saved["text"])


def run_four(saved, threshold):
    with decision_threshold(threshold):
        results = {"hisnet": replay_hisnet(saved["hisnet"])}
        for system in SYSTEMS[1:]:
            results[system] = replay(system, saved[system], results["hisnet"], SentimentScopeResolver)
        # Production LW metadata has a literal 0.05. Its decision already calls
        # the common scorer; record the actual trial value without editing code.
        results["lw_hisnet_dp"]["literature_weighting"]["decision_threshold"] = threshold
    return results


def without_decisions(value):
    """Only labels and threshold metadata may differ; preserve every other field."""
    allowed = {"prediction", "hisnet_prediction", "dependency_prediction",
               "neutral_threshold", "decision_threshold"}
    if isinstance(value, dict):
        return {k: without_decisions(v) for k, v in value.items() if k not in allowed}
    if isinstance(value, list):
        return [without_decisions(v) for v in value]
    return value


def assert_trial(saved, result, threshold):
    if without_decisions(semantic_result(saved)) != without_decisions(semantic_result(result)):
        raise AssertionError("A non-decision field changed during threshold trial")
    expected = independent_label(saved["positive_score"], saved["negative_score"], threshold)
    if result["prediction"] != expected:
        raise AssertionError("Production decision differs from independent threshold calculation")
    if threshold >= BASELINE and result["prediction"] != saved["prediction"]:
        if saved["prediction"] == "neutral" or result["prediction"] != "neutral":
            raise AssertionError("Increasing the neutral interval can only move polar labels to neutral")


def saved_row(db, row_id):
    rows = db.execute("SELECT p.system,a.result FROM predictions p JOIN attempts a ON a.id=p.attempt_id "
                      "WHERE p.row_id=? AND p.status='success'", (row_id,))
    return {s: json.loads(payload) for s, payload in rows if s in SYSTEMS}


def compact(result):
    return {k: result[k] for k in ("prediction", "positive_score", "negative_score")}


def metrics(cases, predictions):
    output = {}
    for system in SYSTEMS:
        matrix = [[0] * 3 for _ in LABELS]
        for case, result in zip(cases, predictions[system]):
            matrix[LABELS.index(case["gold"])][LABELS.index(result["prediction"])] += 1
        classes = {}
        for i, label in enumerate(LABELS):
            tp, actual, predicted = matrix[i][i], sum(matrix[i]), sum(row[i] for row in matrix)
            classes[label] = dict(correct=tp, support=actual, predicted=predicted,
                precision=tp / predicted if predicted else 0., recall=tp / actual if actual else 0.,
                f1=2 * tp / (actual + predicted) if actual + predicted else 0.)
        correct = sum(matrix[i][i] for i in range(3))
        output[system] = dict(correct=correct, n=len(cases), accuracy=correct / len(cases),
            macro_f1=sum(c["f1"] for c in classes.values()) / 3,
            classes=classes, confusion_matrix=matrix, label_order=LABELS)
    return output


def write_csv(path, rows):
    with path.with_suffix(".csv.tmp").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".csv.tmp").replace(path)


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                      *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def summarize(output, cases, runs):
    summaries = {t: metrics(cases, rows) for t, rows in runs.items()}
    metric_rows, class_rows, changes, trades = [], [], [], []
    for threshold, systems in summaries.items():
        for system, metric in systems.items():
            reference = summaries[BASELINE][system]
            metric_rows.append(dict(threshold=threshold, system=system, n=metric["n"],
                correct=metric["correct"], accuracy=metric["accuracy"], macro_f1=metric["macro_f1"],
                accuracy_delta_pp=100*(metric["accuracy"]-reference["accuracy"]),
                macro_f1_delta_pp=100*(metric["macro_f1"]-reference["macro_f1"])))
            for label, values in metric["classes"].items():
                old = reference["classes"][label]
                class_rows.append(dict(threshold=threshold, system=system, label=label, **values,
                    correct_delta=values["correct"]-old["correct"],
                    precision_delta_pp=100*(values["precision"]-old["precision"]),
                    recall_delta_pp=100*(values["recall"]-old["recall"]),
                    f1_delta_pp=100*(values["f1"]-old["f1"])))
            if threshold == BASELINE:
                continue
            corrected = harmed = changed = 0
            for row_id, case in enumerate(cases):
                before, after = runs[BASELINE][system][row_id], runs[threshold][system][row_id]
                if before["prediction"] == after["prediction"]:
                    continue
                changed += 1
                corrected += after["prediction"] == case["gold"]
                harmed += before["prediction"] == case["gold"]
                changes.append(dict(threshold=threshold, system=system, row_id=row_id, **case,
                    before=before["prediction"], after=after["prediction"],
                    positive_score=after["positive_score"], negative_score=after["negative_score"],
                    net_score=after["positive_score"]-after["negative_score"]))
            delta = {label: metric["classes"][label]["correct"]-reference["classes"][label]["correct"] for label in LABELS}
            assert corrected == delta["neutral"]
            assert harmed == -delta["positive"]-delta["negative"]
            assert corrected-harmed == metric["correct"]-reference["correct"]
            trades.append(dict(threshold=threshold, system=system, changed=changed,
                corrected=corrected, harmed=harmed, neutral_gained=delta["neutral"],
                positive_lost=-delta["positive"], negative_lost=-delta["negative"], net_correct=corrected-harmed))
    write_csv(output / "metrics_comparison.csv", metric_rows)
    write_csv(output / "per_class_comparison.csv", class_rows)
    write_csv(output / "tradeoffs.csv", trades)
    if changes:
        write_csv(output / "changed_predictions.csv", changes)
    atomic_json(output / "comparison.json", dict(threshold_order=list(runs), systems=summaries,
        note="Frozen-input production replay; threshold tuning on development data, not an independent test"))
    lines = ["# 3000 yorum: 0,05 / 0,07 / 0,10 eşik karşılaştırması", "",
        "Aynı 3000 yorum ve dört klasik yöntem. Önce 0,05 tabanı birebir doğrulandı; sonra 0,07 ve 0,10 sırayla çalıştırıldı.",
        "Morfoloji, parse, sözlük puanları ve bütün bileşim/kapsam alanları sabit tutuldu. Üretim sınıflandırıcıları kayıtlı girdilerle yeniden hesaplandı; uçtan uca yeni ön işleme yapılmadı. LLM kullanılmadı.", "",
        "Eşik t için P−N > t pozitif, P−N < −t negatif, sınırlar dahil aradaki bölge nötrdür. Sınırdaki kayan nokta toleransı mevcut üretim kuralıyla aynıdır.", "",
        "## Yakalama oranları ve genel performans", "",
        "Her sınıfta 1000 yorum var. Recall, gerçek sınıfın doğru yakalanan oranıdır. Bütün yüzde değerleri yüzde ölçeğindedir.", ""]
    display = []
    for system in SYSTEMS:
        for threshold in runs:
            m = summaries[threshold][system]
            display.append([NAMES[system], f"{threshold:.2f}", *(f"{100*m['classes'][c]['recall']:.2f}" for c in ("positive", "negative", "neutral")),
                            f"{100*m['accuracy']:.2f}", f"{100*m['macro_f1']:.2f}"])
    lines += [markdown_table(["Yöntem", "Eşik", "Pozitif recall %", "Negatif recall %", "Nötr recall %", "Accuracy %", "Macro-F1 %"], display), "",
              "## 0,05'e göre kazanç ve kayıp", "",
              "Sayılar yorum adedidir. Kaybedilen pozitif/negatif, önceden doğru olan fakat yeni eşikte nötrleşen yorumdur.", "",
              markdown_table(["Yöntem", "Eşik", "Nötr kazanılan", "Pozitif kayıp", "Negatif kayıp", "Net doğru", "Etiketi değişen"],
                [[NAMES[r["system"]], f"{r['threshold']:.2f}", r["neutral_gained"], r["positive_lost"], r["negative_lost"], f"{r['net_correct']:+d}", r["changed"]] for r in trades]), "",
              "## Sınıf bazında precision / recall / F1", "",
              markdown_table(["Yöntem", "Eşik", "Sınıf", "Doğru / gerçek", "Precision %", "Recall %", "F1 %", "ΔF1 puan (0,05'e göre)"],
                [[NAMES[r["system"]], f"{r['threshold']:.2f}", r["label"], f"{r['correct']}/{r['support']}",
                  f"{100*r['precision']:.2f}", f"{100*r['recall']:.2f}", f"{100*r['f1']:.2f}", f"{r['f1_delta_pp']:+.2f}"] for r in class_rows]), "",
              "## Deney sınırı ve kayıtlar", "",
              "Bu bir geliştirme/eşik seçimi deneyidir; bağımsız final test değildir. Kalıcı eşik seçilmedi; kaynak koddaki varsayılan 0,05 korunuyor. 0,07 ve 0,10 yalnız ayrı deney süreçlerinde uygulandı.", "",
              "LW katsayıları değiştirilmedi. LW çıktısındaki sabit 0,05 açıklama alanı deney kaydında gerçek eşikle düzeltildi; literature_weighting.dataset_tuned alanı sabit ağırlıklara ilişkindir. Eşik denemeleri açıkça geliştirme verisi kullanır.", "",
              "- [Genel metrikler](metrics_comparison.csv)", "- [Sınıf metrikleri ve farklar](per_class_comparison.csv)",
              "- [Kazanç/kayıp sayıları](tradeoffs.csv)", "- [Etiketi değişen yorumlar](changed_predictions.csv)",
              "- [Karışıklık matrisleri ve tam sayısal özet](comparison.json)", "- [Doğrulama](verification.json)",
              "- Her eşik dizinindeki progress.sqlite3 tam ara çıktıları, predictions.csv bütün tahminleri içerir.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return trades


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error("Output must be separate from the frozen source")
    database = source / "progress.sqlite3"
    wal = source / "progress.sqlite3-wal"
    if wal.exists() and wal.stat().st_size:
        raise ValueError("Source must be a completed, checkpointed run with empty WAL")
    source_hash = digest(database)
    code_hashes = {}
    for frozen in sorted((source / "source_snapshot/src").rglob("*.py")):
        relative = frozen.relative_to(source / "source_snapshot")
        if digest(frozen) != digest(ROOT / relative):
            raise ValueError("Production code differs from source: " + str(relative))
        code_hashes[relative.as_posix()] = digest(frozen)
    if not code_hashes:
        raise ValueError("Source code snapshot is required")
    helper_paths = [Path(__file__).resolve(), ROOT / "scripts/validate_classical_corrections.py",
                    ROOT / "scripts/check_saved_negation_scope.py", ROOT / "scripts/run_balanced_classical_cached.py"]
    identity = dict(source=str(source), source_sha256=source_hash, thresholds=list(THRESHOLDS),
                    source_code=code_hashes, runner_code={p.relative_to(ROOT).as_posix(): digest(p) for p in helper_paths},
                    protocol="frozen_inputs_production_recomposition_threshold_only_v1")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "experiment.json").exists() and json.loads((output / "experiment.json").read_text(encoding="utf-8")) != identity:
        raise ValueError("Experiment identity differs; use a new directory")
    atomic_json(output / "experiment.json", identity)
    for relative in [*code_hashes, *identity["runner_code"]]:
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    started = datetime.now(timezone.utc).isoformat()
    runs = {BASELINE: {s: [] for s in SYSTEMS}}
    baseline_matches = 0
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db, ExitStack() as guards:
        db.execute("BEGIN")
        cases = [dict(text=text, gold=gold) for _, text, gold in db.execute("SELECT row_id,text,gold FROM cases ORDER BY row_id")]
        ids = [i for (i,) in db.execute("SELECT row_id FROM cases ORDER BY row_id")]
        if ids != list(range(3000)) or Counter(c["gold"] for c in cases) != Counter(dict.fromkeys(LABELS, 1000)):
            raise ValueError("Expected the complete balanced 3000-row source")
        counts = dict(db.execute("SELECT system,COUNT(*) FROM predictions WHERE status='success' GROUP BY system"))
        if any(counts.get(s) != len(cases) for s in SYSTEMS):
            raise ValueError("Source needs four complete systems")
        for cls in (HisNetClassifier, TurkishDependencyParser):
            guards.enter_context(patch.object(cls, "__init__", side_effect=AssertionError("Live preprocessing forbidden")))
        for row_id in ids:
            saved = saved_row(db, row_id)
            reproduced = run_four(saved, BASELINE)
            for system in SYSTEMS:
                if semantic_result(reproduced[system]) != semantic_result(saved[system]):
                    raise AssertionError(f"Baseline semantic mismatch: {system}/{row_id}")
                assert_trial(saved[system], reproduced[system], BASELINE)
                baseline_matches += 1
                runs[BASELINE][system].append(compact(saved[system]))
            if (row_id + 1) % 250 == 0:
                print(f"BASELINE VERIFIED {row_id + 1}/3000", flush=True)
        for threshold in THRESHOLDS:
            run_output = output / f"threshold_{round(threshold*1000):03d}"
            metadata = dict(identity, neutral_threshold=threshold, baseline_threshold=BASELINE,
                            systems=list(SYSTEMS), keep_details=True, threshold_development_trial=True)
            runs[threshold] = {s: [] for s in SYSTEMS}
            print(f"START THRESHOLD {threshold:.2f}", flush=True)
            with CheckpointStore(run_output, metadata, cases) as store:
                states = {s: store.states(s) for s in SYSTEMS}
                for row_id in ids:
                    saved = saved_row(db, row_id)
                    tick = time.perf_counter()
                    results = run_four(saved, threshold)
                    elapsed = time.perf_counter() - tick
                    for system in SYSTEMS:
                        result = results[system]
                        assert_trial(saved[system], result, threshold)
                        if states[system].get(row_id) == "success":
                            persisted = saved_row(store.connection, row_id)[system]
                            if semantic_result(result) != semantic_result(persisted):
                                raise AssertionError("Resumed result differs")
                        else:
                            attempt = store.start(system, [row_id], states[system].get(row_id) == "failed")
                            store.finish(attempt, dict(result, status="success", predictions=[result["prediction"]], error=None), 0.)
                        runs[threshold][system].append(compact(result))
                    # Row composition elapsed is separate; per-system times are
                    # not fabricated from shared execution. This is no speed test.
                    if (row_id + 1) % 250 == 0:
                        print(f"THRESHOLD {threshold:.2f}: {row_id + 1}/3000; last row {elapsed:.3f}s", flush=True)
                store.export()
            atomic_json(run_output / "verification.json", dict(threshold=threshold, successful_predictions=12000,
                non_decision_fields_identical=True, independent_decisions_matched=12000,
                live_preprocessing=False, timing_policy="No per-system runtime attribution; stored seconds are zero"))
            print(f"COMPLETE THRESHOLD {threshold:.2f}: 12000 predictions", flush=True)
    if digest(database) != source_hash:
        raise AssertionError("Source database changed")
    if any(digest(ROOT / relative) != value for relative, value in code_hashes.items()):
        raise AssertionError("Production files changed")
    atomic_json(output / "verification.json", dict(started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
        baseline_semantic_matches=baseline_matches, new_predictions=24000, source_unchanged=True,
        production_code_unchanged=True, effective_threshold_order=list(THRESHOLDS),
        source_database_sha256=source_hash, live_model_calls=0, default_threshold=BASELINE))
    trades = summarize(output, cases, runs)
    print(json.dumps(trades, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
