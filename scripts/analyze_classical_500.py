"""Analyze saved classical predictions only; never initialize a classifier."""
import argparse
from collections import Counter
from contextlib import closing
from itertools import combinations
import json
import math
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.evaluation.checkpoint_store import atomic_json

SYSTEMS = ("hisnet", "dp_hisnet", "hisnet_dp", "lw_hisnet_dp")
NAMES = {"hisnet": "HisNet", "dp_hisnet": "DP → HisNet", "hisnet_dp": "HisNet → DP",
         "lw_hisnet_dp": "LW HisNet → DP"}
LABELS = ["negative", "neutral", "positive"]


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                       "| " + " | ".join(["---"] * len(headers)) + " |",
                       *("| " + " | ".join(map(cell, row)) + " |" for row in rows)])


def dump_csv(output, name, rows, columns=None):
    frame = pd.DataFrame(rows, columns=columns)
    path = output / name
    temporary = path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def result_flags(system, result, baseline, hd):
    """Numeric/architectural violations; prediction errors alone are not bugs."""
    flags = []
    if result.get("status") != "success":
        return ["technical_failure"]
    baseline = baseline if baseline and baseline.get("status") == "success" else None
    hd = hd if hd and hd.get("status") == "success" else None
    for polarity in ("positive", "negative"):
        score = result.get(polarity + "_score")
        if not isinstance(score, (int, float)) or not math.isfinite(score) or score < 0:
            flags.append("invalid_" + polarity + "_score")
            continue
        if system != "hisnet":
            total = sum(scope["adjusted_" + polarity] for scope in result.get("scopes", []))
            if not math.isclose(total, score, abs_tol=1e-9):
                flags.append("scope_sum_" + polarity)
        if system in {"hisnet_dp", "lw_hisnet_dp"} and baseline:
            if not math.isclose(result["hisnet_" + polarity + "_score"], baseline[polarity + "_score"], abs_tol=1e-9):
                flags.append("hisnet_baseline_" + polarity)
        if system == "lw_hisnet_dp" and hd:
            if not math.isclose(result["dependency_" + polarity + "_score"], hd[polarity + "_score"], abs_tol=1e-9):
                flags.append("lw_dependency_baseline_" + polarity)
    if not any(flag.startswith("invalid_") for flag in flags):
        net = result["positive_score"] - result["negative_score"]
        expected = "neutral" if abs(net) < .05 or math.isclose(abs(net), .05, abs_tol=1e-12, rel_tol=0) else (
            "positive" if net > 0 else "negative")
        if result["prediction"] != expected:
            flags.append("score_label_mismatch")
    if system in {"hisnet_dp", "lw_hisnet_dp"} and baseline:
        if result.get("morphological_analyses") != baseline.get("morphological_analyses"):
            flags.append("independent_hisnet_morphology_differs")
    if system == "lw_hisnet_dp" and hd and result.get("dependency_prediction") != hd.get("prediction"):
        flags.append("lw_dependency_label_differs")
    tokens = result.get("dependency_tokens", []) or [token for sentence in result.get("sentences", [])
                                                     for token in sentence.get("dependency_tokens", [])]
    sentence_by_id = {token["id"]:token.get("sentence_index") for token in tokens}
    if len(sentence_by_id) != len(tokens):
        flags.append("duplicate_global_dependency_ids")
    for scope in result.get("scopes", []):
        sentiment_token = scope.get("sentiment_token") or {}
        fragments = scope.get("fragment_token_ids") or [sentiment_token.get("id")]
        scope_sentences = {sentence_by_id[t] for t in fragments if t in sentence_by_id}
        negation_ids = [neg.get("id") for neg in scope.get("negations", []) if neg.get("id") is not None]
        if len(negation_ids) != len(set(negation_ids)):
            flags.append("duplicate_negation_in_scope")
        for negation_id in negation_ids:
            if (scope_sentences and None not in scope_sentences and negation_id in sentence_by_id
                    and sentence_by_id[negation_id] not in scope_sentences):
                flags.append("negation_crosses_sentence_boundary")
    return sorted(set(flags))


def analyze(output):
    output = Path(output)
    with closing(sqlite3.connect((output.resolve() / "progress.sqlite3").as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        cases = [dict(row_id=i, text=text, gold=gold) for i, text, gold in db.execute(
            "SELECT row_id,text,gold FROM cases ORDER BY row_id")]
        raw = {s: {} for s in SYSTEMS}
        timing = {s: [] for s in SYSTEMS}
        row_seconds = {s: {} for s in SYSTEMS}
        for system, row_id, payload, seconds in db.execute("""
            SELECT p.system,p.row_id,a.result,a.seconds FROM predictions p
            JOIN attempts a ON a.id=p.attempt_id ORDER BY p.row_id
        """):
            if system in raw:
                raw[system][row_id] = json.loads(payload)
                timing[system].append(seconds)
                row_seconds[system][row_id] = seconds
        attempt_counts = {s: dict(db.execute(
            "SELECT status,COUNT(*) FROM attempts WHERE system=? GROUP BY status", (s,))) for s in SYSTEMS}
        attempt_seconds = {s: db.execute("SELECT COALESCE(SUM(seconds),0) FROM attempts WHERE system=?",(s,)).fetchone()[0]
                           for s in SYSTEMS}
    sampling = json.loads((output / "sampling.json").read_text(encoding="utf-8"))
    initializations = json.loads((output / "initializations.json").read_text(encoding="utf-8"))
    valid_ids = {s: [c["row_id"] for c in cases if raw[s].get(c["row_id"], {}).get("status") == "success"] for s in SYSTEMS}
    metrics, per_class, pairs, violations, operations, coverage, slices = [], [], [], [], [], [], []
    confusion, by_system = {}, {}
    morphology_runtime = []
    exclusions = set(sampling["conflicting_source_label_row_ids"])
    for system in SYSTEMS:
        ids = valid_ids[system]
        if not ids:
            continue
        gold = [cases[i]["gold"] for i in ids]
        predicted = [raw[system][i]["prediction"] for i in ids]
        report = classification_report(gold, predicted, labels=LABELS, output_dict=True, zero_division=0)
        correct = sum(a == b for a, b in zip(gold, predicted))
        interval = binomtest(correct, len(ids)).proportion_ci(method="wilson")
        times = np.array(timing[system])
        metric = dict(system=system, n=len(ids), correct=correct, accuracy=accuracy_score(gold, predicted),
            macro_precision=report["macro avg"]["precision"], macro_recall=report["macro avg"]["recall"],
            macro_f1=report["macro avg"]["f1-score"], weighted_f1=report["weighted avg"]["f1-score"],
            accuracy_ci_low=float(interval.low), accuracy_ci_high=float(interval.high),
            prediction_distribution=dict(Counter(predicted)), classify_seconds=float(attempt_seconds[system]),
            latest_row_seconds=float(times.sum()),
            mean_ms=float(times.mean()*1000), median_ms=float(np.median(times)*1000),
            p95_ms=float(np.percentile(times, 95)*1000),
            initialization_seconds=sum(r["seconds"] for r in initializations if r["system"] == system),
            attempts=attempt_counts[system])
        metrics.append(metric)
        by_system[system] = metric
        for label in LABELS:
            per_class.append(dict(system=system, label=label, **report[label]))
        confusion[system] = confusion_matrix(gold, predicted, labels=LABELS).tolist()
        for name, subset in [
            ("source_label_unambiguous", [i for i in ids if i not in exclusions]),
            ("source_label_conflicting", [i for i in ids if i in exclusions]),
            ("1_to_5_words", [i for i in ids if len(cases[i]["text"].split()) <= 5]),
            ("6_to_15_words", [i for i in ids if 6 <= len(cases[i]["text"].split()) <= 15]),
            ("16_plus_words", [i for i in ids if len(cases[i]["text"].split()) >= 16]),
            ("zero_sentiment_score", [i for i in ids if raw[system][i]["positive_score"] == raw[system][i]["negative_score"] == 0]),
        ]:
            if subset:
                y = [cases[i]["gold"] for i in subset]
                p = [raw[system][i]["prediction"] for i in subset]
                slices.append(dict(system=system, subset=name, n=len(subset), accuracy=accuracy_score(y,p),
                    macro_f1=f1_score(y,p,labels=LABELS,average="macro",zero_division=0)))
        op_counts, alignment_counts = Counter(), Counter()
        zero_ids, no_sentiment_ids = [], []
        for i in ids:
            result = raw[system][i]
            for flag in result_flags(system, result, raw["hisnet"].get(i), raw["hisnet_dp"].get(i)):
                violations.append(dict(system=system, row_id=i, flag=flag, text=cases[i]["text"]))
            for scope in result.get("scopes", []):
                op_counts.update(scope.get("operations", []))
                alignment_counts.update([scope.get("alignment_status", "unspecified")])
            if result["positive_score"] == result["negative_score"] == 0:
                zero_ids.append(i)
            if not result.get("sentiment_words"):
                no_sentiment_ids.append(i)
        operations.append(dict(system=system, scope_operations=dict(op_counts), alignment=dict(alignment_counts),
            modifier_weighted_scopes=sum(raw[system][i].get("modifier_weighted_scope_count",0) for i in ids),
            contrast_weighted_scopes=sum(raw[system][i].get("contrast_weighted_scope_count",0) for i in ids),
            modifier_suppressed_scopes=sum(raw[system][i].get("modifier_component_suppressed_scope_count",0) for i in ids),
            contrast_marker_suppressed_scopes=sum(raw[system][i].get("contrast_marker_suppressed_scope_count",0) for i in ids)))
        coverage.append(dict(system=system, zero_score_rows=len(zero_ids), zero_score_polar_gold_rows=sum(cases[i]["gold"] != "neutral" for i in zero_ids),
            no_sentiment_rows=len(no_sentiment_ids),
            mean_lexicon_coverage=float(np.mean([raw[system][i]["lexicon_coverage"] for i in ids])) if system != "dp_hisnet" else None,
            mean_candidate_lexicon_coverage=float(np.mean([raw[system][i]["candidate_lexicon_coverage"] for i in ids])) if system == "dp_hisnet" else None,
            mean_sentiment_coverage=float(np.mean([raw[system][i]["sentiment_coverage"] for i in ids])),
            zero_score_ids=zero_ids))
        source_counts = Counter()
        non_normal_ids = []
        for i in ids:
            sources = [token.get("analysis_source", "unspecified") for token in raw[system][i].get("morphological_analyses", [])]
            source_counts.update(sources)
            if any(source in {"robust", "fallback"} for source in sources):
                non_normal_ids.append(i)
        non_normal_set = set(non_normal_ids)
        for group, group_ids in [("robust_or_fallback_present",non_normal_ids),("normal_only",[i for i in ids if i not in non_normal_set])]:
            if group_ids:
                values = np.array([row_seconds[system][i] for i in group_ids])
                morphology_runtime.append(dict(system=system, group=group, n=len(group_ids),
                    total_seconds=float(values.sum()), mean_ms=float(values.mean()*1000),
                    median_ms=float(np.median(values)*1000), p95_ms=float(np.percentile(values,95)*1000),
                    token_source_counts=dict(source_counts)))

    for first, second in combinations(SYSTEMS, 2):
        ids = sorted(set(valid_ids[first]) & set(valid_ids[second]))
        if not ids:
            continue
        improved = [i for i in ids if raw[first][i]["prediction"] != cases[i]["gold"] == raw[second][i]["prediction"]]
        harmed = [i for i in ids if raw[first][i]["prediction"] == cases[i]["gold"] != raw[second][i]["prediction"]]
        changed = [i for i in ids if raw[first][i]["prediction"] != raw[second][i]["prediction"]]
        n = len(improved) + len(harmed)
        pairs.append(dict(first=first, second=second, n=len(ids), changed=len(changed),
            improved=len(improved), harmed=len(harmed), net_correct=len(improved)-len(harmed),
            mcnemar_exact_p=float(binomtest(len(improved), n, p=.5).pvalue) if n else 1.0,
            changed_ids=changed, improved_ids=improved, harmed_ids=harmed))
    # Holm correction over the six exploratory paired comparisons.
    previous = 0.0
    for rank, pair in enumerate(sorted(pairs, key=lambda p:p["mcnemar_exact_p"])):
        previous = min(1., max(previous, (len(pairs)-rank)*pair["mcnemar_exact_p"]))
        pair["holm_adjusted_p"] = previous

    rows, errors = [], []
    shared_wrong, disagreements, all_right = [], [], []
    for case in cases:
        i = case["row_id"]
        row = dict(case, source_label_conflict=i in exclusions)
        predictions = []
        for system in SYSTEMS:
            result = raw[system].get(i, {})
            prediction = result.get("prediction")
            row[system] = prediction
            row[system + "_positive"] = result.get("positive_score")
            row[system + "_negative"] = result.get("negative_score")
            row[system + "_correct"] = prediction == case["gold"]
            predictions.append(prediction)
            if prediction is not None and prediction != case["gold"]:
                errors.append(dict(case, system=system, prediction=prediction,
                    positive_score=result["positive_score"], negative_score=result["negative_score"],
                    source_label_conflict=i in exclusions))
            if result.get("status") == "failed":
                violations.append(dict(system=system, row_id=i, flag="technical_failure", text=case["text"], error=result.get("error")))
        if all(p is not None for p in predictions):
            if all(p != case["gold"] for p in predictions):
                shared_wrong.append(i)
            if all(p == case["gold"] for p in predictions):
                all_right.append(i)
            if len(set(predictions)) > 1:
                disagreements.append(i)
        rows.append(row)
    unmatched = Counter()
    for result in raw["hisnet"].values():
        for word in result.get("unmatched_words", []):
            unmatched[word.get("root", "") if isinstance(word, dict) else str(word)] += 1
    analysis = dict(n=len(cases), complete=all(len(valid_ids[s]) == len(cases) for s in SYSTEMS),
        sampling=sampling, metrics=metrics, per_class=per_class,
        confusion_matrices=confusion, label_order=LABELS, pairwise=pairs, violations=violations,
        operations=operations, coverage=coverage, slices=slices, morphology_runtime=morphology_runtime, shared_wrong_ids=shared_wrong,
        all_correct_ids=all_right, disagreement_ids=disagreements, most_common_unmatched_roots=unmatched.most_common(30))
    atomic_json(output / "analysis.json", analysis)
    n = len(cases)
    dump_csv(output, f"metrics_{n}.csv", metrics)
    dump_csv(output, f"per_class_{n}.csv", per_class)
    dump_csv(output, f"pairwise_{n}.csv", pairs)
    dump_csv(output, f"comparison_{n}.csv", rows)
    dump_csv(output, f"errors_{n}.csv", errors)
    dump_csv(output, f"diagnostic_flags_{n}.csv", violations, ["system","row_id","flag","text","error"])
    dump_csv(output, f"slice_metrics_{n}.csv", slices)
    dump_csv(output, f"morphology_runtime_{n}.csv", morphology_runtime)
    dump_csv(output, f"disagreements_{n}.csv", [row for row in rows if row["row_id"] in disagreements])
    scope_rows = []
    for system in SYSTEMS:
        for i, result in raw[system].items():
            for scope in result.get("scopes", []):
                scope_rows.append(dict(system=system, row_id=i, text=cases[i]["text"], gold=cases[i]["gold"],
                    prediction=result.get("prediction"), word=scope.get("word"), root=scope.get("root"),
                    active_sentiment=scope.get("active_sentiment"), alignment=scope.get("alignment_status"),
                    original_positive=scope.get("original_positive"), original_negative=scope.get("original_negative"),
                    adjusted_positive=scope.get("adjusted_positive"), adjusted_negative=scope.get("adjusted_negative"),
                    operations=json.dumps(scope.get("operations",[]),ensure_ascii=False),
                    negations=json.dumps(scope.get("negations",[]),ensure_ascii=False),
                    target=json.dumps(scope.get("target"),ensure_ascii=False),
                    token=json.dumps(scope.get("sentiment_token"),ensure_ascii=False),
                    modifier_multiplier=scope.get("modifier_multiplier"), contrast_multiplier=scope.get("contrast_multiplier")))
    dump_csv(output,f"scope_details_{n}.csv",scope_rows)
    detail_ids = sorted(set(disagreements) | set(shared_wrong[:20]) | {v['row_id'] for v in violations})
    atomic_json(output / "diagnostic_cases.json", [dict(cases[i], systems={s:raw[s].get(i) for s in SYSTEMS}) for i in detail_ids])
    write_plots(output, analysis)
    write_report(output, analysis, cases, raw)
    print("ANALYSIS", json.dumps({"metrics": metrics, "violations":len(violations),
        "shared_wrong":len(shared_wrong), "disagreements":len(disagreements)}, ensure_ascii=False), flush=True)
    return analysis


def write_plots(output, analysis):
    """Export static figures from saved metrics; no model execution."""
    try:
        import matplotlib
    except ImportError:
        print("PLOTS skipped: optional matplotlib is not installed; metrics and tables are complete.", flush=True)
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = analysis["metrics"]
    if not metrics:
        return
    names = [NAMES[m["system"]] for m in metrics]
    positions = np.arange(len(metrics))
    figure, axis = plt.subplots(figsize=(9, 5), layout="constrained")
    for offset, field, label, color in [(-.18, "accuracy", "Accuracy", "#276FBF"),
                                       (.18, "macro_f1", "Macro-F1", "#F28E2B")]:
        bars = axis.bar(positions + offset, [100*m[field] for m in metrics], .36, label=label, color=color)
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axis.set_xticks(positions, names)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Percent")
    axis.set_title(f"TRSAv1 development sample: {analysis['n']} reviews")
    axis.legend()
    axis.grid(axis="y", alpha=.2)
    axis.set_axisbelow(True)
    figure.savefig(output / f"performance_{analysis['n']}.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(9, 8), layout="constrained")
    for axis, system in zip(axes.flat, SYSTEMS):
        if system not in analysis["confusion_matrices"]:
            axis.set_visible(False)
            continue
        matrix = np.array(analysis["confusion_matrices"][system])
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
        axis.set_xticks(range(3), LABELS, rotation=20)
        axis.set_yticks(range(3), LABELS)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Gold")
        axis.set_title(NAMES[system])
        for row in range(3):
            for column in range(3):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                          color="white" if matrix[row,column] > matrix.max()/2 else "black")
    figure.suptitle(f"TRSAv1: confusion matrices, n={analysis['n']}")
    figure.savefig(output / f"confusion_{analysis['n']}.png", dpi=180)
    plt.close(figure)


def write_report(output, analysis, cases, raw):
    metrics, pairs = analysis["metrics"], analysis["pairwise"]
    n = analysis["n"]
    majority = max(analysis["sampling"]["sample_distribution"].values())
    shared = analysis["sampling"].get("execution_mode") == "shared_immutable_preprocessing"
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    workers = manifest.get("workers", 1)
    execution_note = (f"Yorumlar {workers} yerel işçiye dağıtıldı; her yorumdaki dört karar sıralı hesaplandı. "
                      "Tablodaki süreler işçilerin ölçtüğü geçen sürelerin toplamıdır; CPU süresi veya koşunun toplam duvar saati değildir. "
                      if workers > 1 else "Sistemler sıralı çalışır; ")
    protocol = (
        f"Protokol: dört yöntem aynı {n} yorumda ortak, kopyalanmış morfoloji ve parse girdilerini kullanır. "
        "HisNet→DP ve LW aynı HisNet başlangıç sonucunu alır; DP→HisNet kendi dependency adaylarını seçip "
        "yalnız bu adayların köklerini sorgular. Eşikler ve karar kuralları değişmedi. "
        "Bağımsız doğrulama örnekleri independent_verification.json içindedir. "
        if shared else
        f"Protokol: her sistem aynı {n} yorumu aynı sırayla, bağımsız pipeline üzerinden işler. "
        "Başka sistemin hazır HisNet sonucu kullanılmadı. "
    )
    lines = [f"# TRSAv1 — dört klasik sistem, sabit {n} yorum", "",
        f"Koşu tamamlandı: **{analysis['complete']}**. Eksik koşudaki metrikler yalnız başarılı altküme içindir.", "",
        "Bu koşu geliştirme/teşhis amaçlıdır; bağımsız final test seti değildir. "
        "HisNet maddeleri, fiil kökü eşleştirmesi, morfoloji kuralları, eşik ve LW katsayıları bu koşu için değiştirilmedi.", "",
        "## Örneklem ve çalışma protokolü", "",
        f"Kaynak: {analysis['sampling']['dataset_size']} temizlenmiş TRSAv1 satırı. "
        f"Örneklem: {analysis['sampling']['sample_distribution']}. Seed 42 stratified örneklemiyle eşleşme: "
        f"{analysis['sampling']['matches_seed42_reference']}.", "",
        protocol + "Her satır SQLite'a kaydedildi; tam ara çıktılar attempts tablosunda duruyor. "
        "LLM kullanılmadı.", "",
        (f"Önceki örneklerden dışlanan benzersiz metin: {analysis['sampling'].get('excluded_unique_texts', 0)}; "
         f"örtüşme: {analysis['sampling'].get('overlap_with_exclusions', 'uygulanmadı')}. "
         f"Örnekleme referansı: {analysis['sampling'].get('reference_protocol', 'özgün sınıf bazında seed 42 çekilişi')}.")
        if analysis['sampling'].get('exclusions') else "",
        execution_note if workers > 1 else "", "",
        f"Kaynak veride farklı etiketle de bulunan örnekler: {analysis['sampling']['conflicting_source_label_row_ids']}. "
        "Bu satırlar ana ölçümde korunmuştur; aşağıda çıkarıldıklarındaki sonuç ayrıca verilir.", "",
        "## Genel performans", "",
        table(["Sistem","Doğru/n","Accuracy %","Macro P %","Macro R %","Macro F1 %","Weighted F1 %","Accuracy %95 aralık"],
            [[NAMES[m['system']],f"{m['correct']}/{m['n']}",*(f"{100*m[k]:.2f}" for k in
              ['accuracy','macro_precision','macro_recall','macro_f1','weighted_f1']),
              f"{100*m['accuracy_ci_low']:.2f}–{100*m['accuracy_ci_high']:.2f}"] for m in metrics]), "",
        "Aralıklar Wilson yaklaşımıyla hesaplanan yaklaşık örneklem belirsizliğidir; etiket hatalarını veya yeniden kullanılan "
        f"geliştirme setindeki iyimserliği kapsamaz. Çoğunluk sınıfı tabanı {majority}/{n} = %{100*majority/n:.2f}.", "",
        "## Sınıf bazında precision / recall / F1", "",
        table(["Sistem","Sınıf","n","Precision %","Recall %","F1 %"],
            [[NAMES[r['system']],r['label'],int(r['support']),*(f"{100*r[k]:.2f}" for k in ['precision','recall','f1-score'])]
             for r in analysis['per_class']]), "", "## Karışıklık matrisleri", "",
        "Satırlar gerçek etiket, sütunlar tahmin. Her sistemin tüm örnekleri başarıyla tamamlanmadıysa matris yalnız başarılı altkümeyi içerir."]
    for system, matrix in analysis["confusion_matrices"].items():
        lines += ["", f"### {NAMES[system]}", "", table(["Gerçek / tahmin",*LABELS], [[label,*row] for label,row in zip(LABELS,matrix)])]
    lines += ["", "## İkili farklar", "",
        "İyileşen: ilk sistem yanlışken ikinci doğru. Bozulan: ilk sistem doğruyken ikinci yanlış. "
        "Test, yalnız doğru/yanlış değişen eşleşmelerde iki yönlü exact binom testidir (exact McNemar); "
        "Holm düzeltmesi altı karşılaştırmaya uygulanır. Bunlar keşifsel ölçümlerdir. "
        "Hesaplama: [SciPy binomtest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).", "",
        table(["İlk sistem","İkinci sistem","Tahmini değişen","İyileşen","Bozulan","Net doğru","p","Holm p"],
            [[NAMES[p['first']],NAMES[p['second']],p['changed'],p['improved'],p['harmed'],p['net_correct'],
              f"{p['mcnemar_exact_p']:.4f}",f"{p['holm_adjusted_p']:.4f}"] for p in pairs]), "",
        f"Dördünün birlikte doğru olduğu yorum: **{len(analysis['all_correct_ids'])}**; birlikte yanlış: "
        f"**{len(analysis['shared_wrong_ids'])}**; sistemlerin farklı etiket verdiği yorum: **{len(analysis['disagreement_ids'])}**.", "",
        "## Süreler", "",
        ("**Bu koşunun süreleri hız değerlendirmesinde kullanılamaz.** Kayıtlardaki uzun kesintiler "
         "perf_counter ölçümlerine yansımıştır. Ham değerler korunmuştur; bunlar kesintisiz çıkarım veya CPU süresi değildir. "
         "Ayrıntı: [timing_caveats.json](timing_caveats.json).") if (output / "timing_caveats.json").exists() else "", "",
        table(["Sistem","Başlatma s","Sınıflandırma s","Ortalama ms/yorum","Medyan ms","P95 ms"],
            [[NAMES[m['system']],*(f"{m[k]:.2f}" for k in ['initialization_seconds','classify_seconds','mean_ms','median_ms','p95_ms'])]
             for m in metrics]), "",
        ("**Paylaşılan ön işleme:** HisNet süresi morfolojiyi içerir; diğer üç yöntem için burada yalnız bileşim süresi vardır. "
         "Ortak parse süresi shared_preprocessing.json dosyasındadır. Bu sütunlar bağımsız uçtan uca yöntem hızlarını karşılaştırmaz. "
         if shared else "") +
        "Süreler yerel CPU, önbellek ve o anki sistem yüküne bağlıdır. Başlatma ayrıca gösterilir; kayıt/dışa aktarma beklemeleri "
        "sınıflandırma süresine dahil değildir. Toplam süre tüm kaydedilmiş girişimleri kapsar; yorum başına dağılımlar son girişimden alınır. "
        "Kesilen/cevabı kaydedilmeyen girişimlerin süresi bilinmez. " + execution_note + "Kontrollü donanım benchmark'ı değildir.", "",
        "### Morfolojik çözümleme türüne göre süre", "",
        table(["Sistem","Grup","Yorum","Toplam s","Ortalama ms","Medyan ms"],
            [[NAMES[r['system']],r['group'],r['n'],*(f"{r[k]:.2f}" for k in ['total_seconds','mean_ms','median_ms'])]
             for r in analysis['morphology_runtime']]), "",
        "Gruplar kaydedilmiş analysis_source alanına göre belirlenir. Robust/fallback içeren yorumlar daha uzun veya yazım hatalı da olabilir; "
        "bu gruplama tek başına nedensel hız ölçümü değildir. Paylaşılan akışta dependency satırları morfoloji çalıştırma süresini içermez. "
        "Mevcut morfoloji kodu her yeni OOV için izole çözümleyici kurar "
        "ve adayları önbelleğe alır; bu davranış deney sırasında değiştirilmedi.", "",
        "## Kapsam ve işlem tanıları", "",
        table(["Sistem","Skoru 0/0 yorum","Bunlardan polar etiketli","Duygu sözcüğü olmayan"],
            [[NAMES[r['system']],r['zero_score_rows'],r['zero_score_polar_gold_rows'],r['no_sentiment_rows']] for r in analysis['coverage']]), "",
        "0/0; sözlük kapsamı, morfoloji veya dependency aday seçimi gibi farklı nedenlerden doğabilir. Tek başına kod hatası kanıtı değildir.", "",
        table(["Sistem","Olumsuzluk çevirme","Hedef bastırma","Eşleşmeyen kapsam","LW modifier","LW contrast"],
            [[NAMES[r['system']],r['scope_operations'].get('negation_flip',0),r['scope_operations'].get('suppressed_as_target',0),
              r['alignment'].get('unmatched',0),r['modifier_weighted_scopes'],r['contrast_weighted_scopes']] for r in analysis['operations']]), "",
        "Sayılar yorum sayısı değil kapsam/işlem sayısıdır. LW ayrıca modifier bileşeni ve contrast marker bastırması yapabilir; "
        "fark yalnız katsayıların etkisi olarak yorumlanmamalıdır.", "",
        "## Otomatik kod tutarlılığı kontrolleri", "",
        f"İhlal sayısı: **{len(analysis['violations'])}**. Kontroller: sonlu/negatif olmayan skorlar, kapsam toplamları, "
        "eşik–etiket uyumu, H→DP/LW başlangıç HisNet skorlarının korunması, LW temel DP skorları, "
        "HisNet morfoloji çıktılarının tutarlılığı, tekrarlanan token/olumsuzluk ID'leri ve cümleler arası olumsuzluk taşması. "
        "Sıfır ihlal bütün dilsel kapsam kararlarının doğru olduğunu kanıtlamaz."]
    if analysis["violations"]:
        lines += ["", table(["Sistem","ID","İhlal"],[[r['system'],r['row_id'],r['flag']] for r in analysis['violations'][:30]])]
    lines += ["", "## Etiket çelişkisine duyarlılık ve uzunluk dilimleri", "",
        table(["Sistem","Alt küme","n","Accuracy %","Macro F1 %"],
            [[NAMES[r['system']],r['subset'],r['n'],f"{100*r['accuracy']:.2f}",f"{100*r['macro_f1']:.2f}"] for r in analysis['slices']]), "",
        f"Alt kümeler sonuç sonrası teşhis içindir; ana {n} örneklik ölçümü değiştirmez. Küçük alt kümelerde macro-F1 tüm üç etiketi içerir.", "",
        "## Ayrışan örnekler", "", f"Aşağıdaki örnekler sabit satır sırasındaki ilk 20 ayrışmadır; tamamı disagreements_{n}.csv içindedir.", "",
        table(["ID","Metin","Gold",*map(NAMES.get,SYSTEMS)],
            [[i,cases[i]['text'],cases[i]['gold'],*(raw[s][i]['prediction'] for s in SYSTEMS)] for i in analysis['disagreement_ids'][:20]]), "",
        "## Dört sistemin de yanlış olduğu örnekler", "",
        table(["ID","Metin","Gold",*map(NAMES.get,SYSTEMS)],
            [[i,cases[i]['text'],cases[i]['gold'],*(raw[s][i]['prediction'] for s in SYSTEMS)] for i in analysis['shared_wrong_ids'][:20]]), "",
        "## Sözlükte eşleşmeyen sık kökler", "", table(["Kök","Görülme"],analysis['most_common_unmatched_roots']), "",
        "Bu liste fiil kökü/lemma uyuşmazlığı, yazım hatası, özel ad ve kapsam dışı sözcükleri birlikte içerir; "
        "otomatik olarak sözlüğe eklenecek madde listesi değildir.", "",
        "## Dosyalar", "", "- `analysis.json`: tüm metrikler, işlem sayıları, ID listeleri ve tanılar.",
        f"- `comparison_{n}.csv`: her yorumda dört tahmin ve pozitif/negatif skorlar.",
        f"- `errors_{n}.csv`, `disagreements_{n}.csv`: hata ve ayrışma örnekleri.",
        f"- `scope_details_{n}.csv`: tüm dependency kapsamlarının skor, işlem, hedef ve olumsuzluk ayrıntıları.",
        "- `diagnostic_cases.json`: ayrışan ve seçilmiş ortak hatalı örneklerin tam ara çıktıları.",
        f"- `per_class_{n}.csv`, `pairwise_{n}.csv`, `slice_metrics_{n}.csv`: ayrıntılı metrikler.",
        "- `progress.sqlite3`: kaldığı yerden devam kaydı ve tam model ara çıktıları.", ""]
    if (output / f"performance_{n}.png").exists():
        lines += ["## Grafikler", "", f"![Accuracy ve Macro F1](performance_{n}.png)", ""]
    if (output / f"confusion_{n}.png").exists():
        lines += [f"![Karışıklık matrisleri](confusion_{n}.png)", ""]
    if (output / f"runtime_{n}.png").exists():
        lines += [f"![Sınıflandırma süreleri](runtime_{n}.png)", ""]
    if (output / "findings.md").exists():
        lines += ["Örnek bazında elle incelenmiş bulgular: [findings.md](findings.md).", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    analyze(parser.parse_args().output)
