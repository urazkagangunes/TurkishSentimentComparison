"""Bounded-detail analysis of saved predictions; no classifier imports or calls."""
import csv
import json
import sqlite3
from collections import Counter
from contextlib import closing, contextmanager
from pathlib import Path

import numpy as np

from src.evaluation.checkpoint_store import LABELS, atomic_json
from src.evaluation.paper_diagnostics import TEXT_SLICES, MODEL_SLICES
from src.evaluation.paper_metrics import encoded, metrics, paired, bootstrap, holm, PRIMARY_PAIRS, prediction_summary
from src.evaluation.resumable_evaluator import LLM_SYSTEMS
from src.evaluation.run_identity import implementation_fingerprint


ERROR_CODES = ("timeout", "parse_error", "invalid_response", "context_budget",
               "morphology_error", "lexicon_error", "other")


@contextmanager
def csv_output(path, fields):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        yield writer
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row)) or ["no_rows"]
    with csv_output(path, fields) as writer:
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k,v in row.items()})


def _table(headers, rows):
    def cell(v):
        return str(v).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |", "|"+"---|"*len(headers)] +
                     ["| " + " | ".join(map(cell, row)) + " |" for row in rows])


def _percent(x):
    return "N/A" if x is None else f"{100*x:.2f}"


def analyze_dataset(output, repetitions=2000, seed=42, emit=lambda s: print(s, flush=True)):
    output = Path(output)
    with closing(sqlite3.connect((output.resolve()/"progress.sqlite3").as_uri()+"?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        manifest = json.loads(db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
        systems = manifest["paper"]["systems"]
        dataset = manifest["paper"]["dataset"]
        records = db.execute("SELECT row_id,text,gold FROM cases ORDER BY row_id").fetchall()
        n = len(records)
        gold = encoded([r[2] for r in records])
        active = sorted(set(gold[gold>=0].tolist()))
        groups = np.empty(n, dtype=object)
        text_masks = {flag: np.zeros(n, dtype=bool) for flag in TEXT_SLICES}
        with csv_output(output/"provenance.csv", ["row_id", "source_ids", "source_groups", "split", "group_id", "source_label_conflict"]) as writer:
            for i, payload in db.execute("SELECT row_id,metadata FROM paper_cases ORDER BY row_id"):
                meta = json.loads(payload)
                groups[i] = meta["group_id"]
                writer.writerow(dict(row_id=i, **{k:json.dumps(meta.get(k), ensure_ascii=False)
                    if isinstance(meta.get(k), list) else meta.get(k) for k in writer.fieldnames if k != "row_id"}))
                for flag in text_masks:
                    text_masks[flag][i] = meta["text_flags"].get(flag, False)
        predictions = {s: np.full(n, -1, dtype=np.int8) for s in systems}
        statuses = {s: np.zeros(n, dtype=np.int8) for s in systems}  # 0 pending, 1 success, 2 failed
        for s,i,status,pred in db.execute("SELECT system,row_id,status,prediction FROM predictions"):
            if s in predictions:
                statuses[s][i] = 1 if status == "success" else 2
                predictions[s][i] = LABELS.index(pred) if status == "success" else -1
        common = np.logical_and.reduce([statuses[s] == 1 for s in systems])
        diag_masks = {s:{f:np.zeros(n, dtype=bool) for f in MODEL_SLICES} for s in systems}
        stage_stats, operation_stats, failures = [], [], []
        failure_summary, retry_summary = [], []
        detail_fields = ["row_id", "system", "gold", "prediction", "status", "error", "error_code", "diagnostics"]
        with csv_output(output/"diagnostics.csv", detail_fields) as writer:
            for system in systems:
                cache_id, cached = None, None
                ops, roots, sources = Counter(), Counter(), Counter()
                scope_count = unaligned_scopes = 0
                current_errors = Counter()
                coverage = {k:[] for k in ("lexicon_coverage", "sentiment_coverage", "candidate_lexicon_coverage")}
                for i, status, pred, error, aid, payload in db.execute("""
                    SELECT p.row_id,p.status,p.prediction,p.error,a.id,a.result
                    FROM predictions p JOIN attempts a ON p.attempt_id=a.id
                    WHERE p.system=? ORDER BY p.row_id""", (system,)):
                    if aid != cache_id:
                        cached, cache_id = json.loads(payload), aid
                    d = cached.get("paper_diagnostics") or {}
                    for flag in MODEL_SLICES:
                        diag_masks[system][flag][i] = d.get("flags", {}).get(flag, False)
                    ops.update(d.get("operations", {})); roots.update(d.get("unmatched_roots", {}))
                    sources.update(d.get("morphology_sources", {}))
                    scope_count += d.get("scope_count", 0)
                    unaligned_scopes += d.get("unaligned_scopes", 0)
                    for key in coverage:
                        if d.get(key) is not None:
                            coverage[key].append(d[key])
                    if status == "failed":
                        error_code = _failure_code(cached, error)
                        current_errors[error_code] += 1
                        failures.append(dict(row_id=i, system=system, gold=records[i][2],
                                             words=len(records[i][1].split()), error=error,
                                             error_code=error_code))
                    writer.writerow(dict(row_id=i, system=system, gold=records[i][2], prediction=pred,
                        status=status, error=error, error_code=cached.get("error_code"),
                        diagnostics=json.dumps(d, ensure_ascii=False)))
                operation_stats.append(dict(system=system, operations=dict(ops),
                    scope_count=scope_count, unaligned_scopes=unaligned_scopes,
                    unmatched_roots=dict(roots.most_common(100)), morphology_sources=dict(sources),
                    **{k:float(np.mean(v)) if v else None for k,v in coverage.items()}))
                times, batch_sizes, stages = [], [], Counter()
                counts, token_usage = Counter(), Counter()
                failed_attempts, failed_occurrences, recovery_counts = Counter(), Counter(), Counter()
                row_attempts = np.zeros(n, dtype=np.int64)
                recovery_occurrences = request_attempts = internal_retries = unknown_requests = 0
                recovered = 0
                for status, seconds, recovery, ids, payload in db.execute(
                        "SELECT status,seconds,recovery,row_ids,result FROM attempts WHERE system=?", (system,)):
                    counts[status] += 1
                    row_ids = json.loads(ids)
                    row_attempts[row_ids] += 1
                    d = json.loads(payload) if payload else {}
                    if status == "failed":
                        code = _failure_code(d)
                        failed_attempts[code] += 1
                        failed_occurrences[code] += len(row_ids)
                    if recovery:
                        recovery_counts[status] += 1
                        recovery_occurrences += len(row_ids)
                    if system in LLM_SYSTEMS:
                        requests = d.get("attempt")
                        if isinstance(requests, int) and not isinstance(requests, bool) and requests >= 0:
                            request_attempts += requests
                            internal_retries += max(requests - 1, 0)
                        else:
                            unknown_requests += 1
                    if payload:
                        for key in ("prompt_tokens", "output_tokens", "attempt"):
                            token_usage[key] += d.get(key, 0) or 0
                        stages.update(d.get("paper_timing", {}))
                    if seconds is not None:
                        times.append(seconds)
                        batch_sizes.append(len(row_ids))
                    if recovery and status == "success":
                        recovered += len(row_ids)
                initializations = db.execute("SELECT seconds,status,error FROM paper_initializations WHERE system=?", (system,)).fetchall()
                initialization_errors = Counter(_error_kind(r[2]) for r in initializations if r[1] == "failed")
                for code in sorted(set(ERROR_CODES) | current_errors.keys() | failed_attempts.keys() | initialization_errors.keys()):
                    failure_summary.append(dict(system=system, error_code=code,
                        current_failed_rows=current_errors[code], failed_attempts=failed_attempts[code],
                        failed_row_occurrences=failed_occurrences[code], initialization_failures=initialization_errors[code]))
                retried = row_attempts > 1
                retry_summary.append(dict(system=system, recorded_attempts=sum(counts.values()),
                    interrupted_attempts=counts["interrupted"], running_attempts=counts["running"],
                    repeated_row_attempts=int(np.maximum(row_attempts - 1, 0).sum()),
                    retried_unique_rows=int(retried.sum()),
                    successful_retried_rows=int((retried & (statuses[system] == 1)).sum()),
                    recovery_attempts=sum(recovery_counts.values()),
                    recovery_successful_attempts=recovery_counts["success"],
                    recovery_failed_attempts=recovery_counts["failed"],
                    recovery_interrupted_attempts=recovery_counts["interrupted"],
                    recovery_running_attempts=recovery_counts["running"],
                    recovery_row_occurrences=recovery_occurrences, recovered_rows=recovered,
                    known_llm_request_attempts=request_attempts, known_llm_internal_retries=internal_retries,
                    llm_request_count_unknown_attempts=unknown_requests))
                total_seconds = sum(times)
                success = int((statuses[system] == 1).sum())
                stage_stats.append(dict(system=system, measured_unit="batch" if system in {"llm", "dp_llm"} else "review",
                    initialization_seconds=sum(r[0] for r in initializations), initialization_errors=[r[2] for r in initializations if r[1]=="failed"],
                    completed_attempt_seconds=total_seconds, attempts=dict(counts), recovered_rows=recovered,
                    measured_pipeline_seconds_with_initialization=total_seconds+sum(r[0] for r in initializations),
                    mean_attempt_seconds=float(np.mean(times)) if times else None,
                    p50_attempt_seconds=float(np.percentile(times, 50)) if times else None,
                    p95_attempt_seconds=float(np.percentile(times, 95)) if times else None,
                    successful_reviews_per_completed_attempt_second=success/total_seconds if total_seconds else None,
                    mean_batch_size=float(np.mean(batch_sizes)) if batch_sizes else None,
                    stages=dict(stages), **dict(token_usage),
                    caveat="Independent pipelines; completed attempts include retries. Interrupted attempt durations unknown; OS sleep can inflate elapsed times."))
        row_fields = ["row_id", "text", "gold", "common_success", *systems]
        with csv_output(output/"predictions_wide.csv", row_fields) as writer:
            for i, text, label in records:
                writer.writerow(dict(row_id=i, text=text, gold=label, common_success=bool(common[i]),
                    **{s:LABELS[predictions[s][i]] if predictions[s][i]>=0 else "" for s in systems}))
        with csv_output(output/"paired_changes.csv", ["row_id", "first", "second", "gold", "before", "after", "change", "common_success"]) as writer:
            for first,second in PRIMARY_PAIRS:
                if first not in systems or second not in systems:
                    continue
                a,b = predictions[first],predictions[second]
                changed = np.flatnonzero((a>=0)&(b>=0)&(a!=b)&(gold>=0))
                for i in changed:
                    writer.writerow(dict(row_id=int(i), first=first, second=second, gold=LABELS[gold[i]],
                        before=LABELS[a[i]], after=LABELS[b[i]], common_success=bool(common[i]),
                        change="improved" if b[i]==gold[i] else "harmed" if a[i]==gold[i] else "wrong_to_wrong"))

    # No database handle or model is needed during bootstrap/report generation.
    overall, class_rows, confusion_rows, slice_rows, comparisons, pair_slices, failure_slices = [], [], [], [], [], [], []
    all_results = {}
    emit(f"[ANALİZ] {dataset['id']} | ortak başarılı={int(common.sum())}/{n} | bootstrap={repetitions}")
    for system in systems:
        p = predictions[system]
        valid = statuses[system] == 1
        all_results[system] = {}
        for view, mask in (("successful", valid), ("common_success", common)):
            m = metrics(gold[mask], p[mask], active)
            m["ci95"] = bootstrap(gold[mask], p[mask], groups[mask], active, repetitions, seed)
            all_results[system][view] = m
            overall.append(dict(system=system, view=view, total=n, success=int(valid.sum()),
                failed=int((statuses[system]==2).sum()), pending=int((statuses[system]==0).sum()),
                coverage=float(mask.mean()), system_coverage=float(valid.mean()), evaluation_n=m["n"],
                correct_over_all=float(((p==gold)&valid&(gold>=0)).sum()/(gold>=0).sum()) if active else None,
                **{k:m[k] for k in ("accuracy", "error_rate", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "mcc")},
                macro_labels=m["macro_labels"], ci95=m["ci95"],
                **prediction_summary(m["prediction_distribution"])))
            class_rows.extend(dict(system=system, view=view, label=l, **m["classes"][l]) for l in LABELS)
            confusion_rows.extend(dict(system=system, view=view, gold=g, prediction=q,
                count=m["confusion_matrix"][i][j], row_percent=m["confusion_row_percent"][i][j])
                for i,g in enumerate(LABELS) for j,q in enumerate(LABELS))
        for name, mask in text_masks.items():
            _slice(slice_rows, system, name, "model_independent_surface_proxy", mask, gold, p, active)
            failure_slices.append(dict(system=system, slice=name, n=int(mask.sum()),
                successful=int((mask & valid).sum()), failed=int((mask & (statuses[system]==2)).sum()),
                pending=int((mask & (statuses[system]==0)).sum())))
        for i,label in enumerate(LABELS):
            mask = gold == i
            failure_slices.append(dict(system=system, slice="gold_"+label, n=int(mask.sum()),
                successful=int((mask & valid).sum()), failed=int((mask & (statuses[system]==2)).sum()),
                pending=int((mask & (statuses[system]==0)).sum())))
        if system not in {"llm", "dp_llm"}:
            for name, mask in diag_masks[system].items():
                _slice(slice_rows, system, name, "model_specific_posthoc_not_causal", mask, gold, p, active)
        if "hisnet" in diag_masks:
            for name in ("zero_score", "robust_or_fallback", "unmatched_opinion_root", "mixed_lexicon_evidence"):
                _slice(slice_rows, system, "reference_hisnet_"+name, "fixed_hisnet_reference_subset",
                       diag_masks["hisnet"][name], gold, p, active)
    for first, second in PRIMARY_PAIRS:
        if first not in systems or second not in systems:
            continue
        # Primary effects on the same all-system successful universe, not different per-pair samples.
        m = paired(gold[common], predictions[first][common], predictions[second][common], active)
        m.update(first=first, second=second, view="common_success",
                 ci95_difference=bootstrap(gold[common], predictions[first][common], groups[common], active,
                    repetitions, seed, second=predictions[second][common]))
        comparisons.append(m)
        for name, mask in text_masks.items():
            ids = mask & common
            pair_slices.append(dict(first=first, second=second, slice=name, view="common_success",
                                   **paired(gold[ids], predictions[first][ids], predictions[second][ids], active)))
        for name in ("negation_flip", "suppressed_as_target", "modifier_weighted", "contrast_weighted"):
            ids = common & (diag_masks[first][name] | diag_masks[second][name])
            pair_slices.append(dict(first=first, second=second, slice="posthoc_operation_union_"+name,
                                   view="common_success", **paired(gold[ids], predictions[first][ids], predictions[second][ids], active)))
    holm(comparisons)
    baselines = {label:metrics(gold, np.full(n,i,dtype=np.int8), active) for i,label in enumerate(LABELS)}
    report = dict(dataset=dataset, total=n, systems=systems, complete=all((statuses[s]==1).all() for s in systems),
        common_success=int(common.sum()), common_gold_distribution={l:int(((gold==i)&common).sum()) for i,l in enumerate(LABELS)},
        dataset_validation=manifest["paper"]["validation"], metrics=all_results, baselines=baselines,
        pairs=comparisons, runtime=stage_stats, mechanism_diagnostics=operation_stats,
        failure_summary=failure_summary, retry_summary=retry_summary,
        failure_summary_policy="Current failed reviews, historical failed review/batch attempts and initialization failures are separate units. Internal LLM retry error types are not stored.",
        retry_summary_policy="Row repeats count recorded attempts beyond the first, including interrupted work. LLM request/retry counts cover recorded payloads only; missing counters remain explicitly unknown.",
        statistical_policy="95% cluster percentile bootstrap; fixed dataset macro labels; Holm on five primary accuracy comparisons per dataset; slice comparisons descriptive",
        negation_policy="Automatic surface proxies only. No gold cue/scope recall, no human annotation (4.B excluded).",
        analysis_identity=dict(implementation=implementation_fingerprint(), bootstrap_repetitions=repetitions,
                               bootstrap_seed=seed),
        bootstrap_policy="Groups unite formatting-normalized duplicates and configured source groups; no semantic near-duplicate guarantee",
        paired_universe="all selected systems successful; full six-system paper comparison requires all six selected",
        failure_policy="Never convert failure to neutral; correct_over_all penalizes missing output without inventing labels")
    atomic_json(output/"paper_summary.json", report)
    write_csv(output/"metrics.csv", overall)
    write_csv(output/"per_class.csv", class_rows)
    write_csv(output/"confusion.csv", confusion_rows)
    write_csv(output/"slices.csv", slice_rows)
    write_csv(output/"pairwise.csv", comparisons)
    write_csv(output/"pairwise_slices.csv", pair_slices)
    write_csv(output/"failures.csv", failures, ["row_id", "system", "gold", "words", "error", "error_code"])
    write_csv(output/"failure_slices.csv", failure_slices)
    write_csv(output/"failure_summary.csv", failure_summary)
    write_csv(output/"retry_summary.csv", retry_summary)
    write_csv(output/"runtime.csv", stage_stats)
    write_csv(output/"mechanisms.csv", operation_stats)
    _write_report(output, report, overall)
    return report


def _failure_code(result, error=None):
    code = result.get("error_code")
    return _error_kind(result.get("error") if error is None else error) if code in {None, "classifier_exception"} else code


def _error_kind(error):
    message = (error or "").lower()
    if "timeout" in message or "timed out" in message:
        return "timeout"
    if "dependency parse" in message:
        return "parse_error"
    if any(marker in message for marker in ("json", "prediction", "result contract", "schema")):
        return "invalid_response"
    return "other"


def _slice(rows, system, name, origin, mask, gold, pred, active):
    m = metrics(gold[mask], pred[mask], active)
    rows.append(dict(system=system, slice=name, origin=origin, total=int(mask.sum()), evaluated=m["n"],
                     gold_distribution={l:int((mask & (gold==i)).sum()) for i,l in enumerate(LABELS)},
                     accuracy=m["accuracy"], macro_f1=m["macro_f1"], classes=m["classes"],
                     **prediction_summary(m["prediction_distribution"])))


def _write_report(output, report, overall):
    lines = [f"# {report['dataset']['id']} — altı sistem değerlendirmesi", "",
        f"Toplam: {report['total']}; ortak başarılı: {report['common_success']}; tamamlandı: {report['complete']}.", "",
        "Sonuçlar 0–1 ölçeğinde CSV/JSON, aşağıdaki tabloda yüzde ölçeğindedir. "
        "Makale ana karşılaştırması ortak başarılı kümedir; bunun sınıf dağılımı ve kapsam kaybı paper_summary.json içindedir.", "",
        _table(["Model", "Evren", "N", "Coverage %", "Accuracy %", "Macro-F1 %", "Nötr recall %", "Nötr F1 %", "Nötr tahmin %"],
            [[r["system"],r["view"],r["evaluation_n"],_percent(r["coverage"]),_percent(r["accuracy"]),_percent(r["macro_f1"]),
              _percent(report["metrics"][r["system"]][r["view"]]["classes"]["neutral"]["recall"]),
              _percent(report["metrics"][r["system"]][r["view"]]["classes"]["neutral"]["f1"]),
              _percent(r["neutral_prediction_rate"])] for r in overall]), "",
        "Nötr tahmin oranı = seçili evrendeki nötr tahmin / geçerli tahmin. Gold gerektirmez; geçerli çıktı yoksa N/A.", "",
        "## Eşleştirilmiş karşılaştırmalar", "",
        _table(["Önce", "Sonra", "N", "Düzeltilen", "Bozulan", "Net", "Holm p"],
               [[p["first"],p["second"],p["n"],p["improved"],p["harmed"],p["net_correct"],p.get("holm_adjusted_p")] for p in report["pairs"]]), "",
        "McNemar/Holm doğruluk için; F1 farkları eşleştirilmiş cluster bootstrap aralıklarıyla raporlanır. "
        "McNemar örnek bağımsızlığı varsayar; ilişkili gruplarda bootstrap aralıkları esas alınmalıdır.", "",
        "## Teknik hata ve tekrar özeti", "",
        _table(["Model", "Hata türü", "Güncel hatalı yorum", "Geçmiş hatalı deneme", "Hatalı satır işleme", "Yükleme hatası"],
            [[r["system"],r["error_code"],r["current_failed_rows"],r["failed_attempts"],
              r["failed_row_occurrences"],r["initialization_failures"]] for r in report["failure_summary"]
             if any(r[k] for k in ("current_failed_rows", "failed_attempts", "initialization_failures"))]), "",
        "Geçmiş hatalı deneme bir yorum/batch çağrısıdır; satır işleme sayısı batch büyüklüğünü de içerir. "
        "Kurtarılan bir yorum güncel hata sayısından çıkar, geçmiş hata kayıtları korunur. "
        "LLM çağrısı içindeki ara hata türleri kayıtta yoktur.", "",
        _table(["Model", "Tekrar işlenen yorum", "Kurtarma denemesi", "Kurtarılan yorum", "Bilinen LLM iç retry", "İstek sayısı bilinmeyen deneme"],
            [[r["system"],r["retried_unique_rows"],r["recovery_attempts"],r["recovered_rows"],
              r["known_llm_internal_retries"],r["llm_request_count_unknown_attempts"]] for r in report["retry_summary"]]), "",
        "Checkpoint tekrarları ile LLM iç retry sayıları farklı düzeylerdir; toplanmamalıdır. "
        "Kesilmiş/kaydı eksik çağrıların LLM istek sayısı bilinmez.", "",
        "## Sınırlar", "",
        "- Olumsuzluk ve karma görüş dilimleri otomatik vekil kurallardır; gerçek olumsuzluk/kapsam recall ölçümü değildir.",
        "- Model-özel işlem dilimleri sonradan gözlenen alt kümelerdir; tek başına nedensel kural katkısı kanıtlamaz.",
        "- Gold olmayan kaynaklarda sınıflandırma metrikleri N/A; olmayan gerçek sınıfların recall/F1'i N/A.",
        "- İki sınıflı kaynakta nötr tahmini hata olarak kalır; macro sınıfları sabittir ve JSON'da kayıtlıdır.",
        "- Bu koşu daha önce incelenmiş korpusları bağımsız yeni test verisine dönüştürmez.",
        "- Süreler tamamlanan girişimlerden gelir; kesilmiş girişimlerin süreleri bilinmez. Sistem uyku/kesinti süreleri ölçümü etkileyebilir.", "",
        "## Dosyalar", "",
        "`per_class.csv`, `confusion.csv`, `pairwise.csv`, `pairwise_slices.csv`, `slices.csv`, "
        "`predictions_wide.csv`, `diagnostics.csv`, `failures.csv`, `failure_slices.csv`, `failure_summary.csv`, `retry_summary.csv`, `runtime.csv`, "
        "`mechanisms.csv`, `paper_summary.json`, `progress.sqlite3`.", ""]
    temporary = output/"paper_report.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(output/"paper_report.md")
