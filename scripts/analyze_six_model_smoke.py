"""Summarize persisted live smoke evidence without rerunning any model."""
import argparse
import json
from pathlib import Path

LABELS = ["negative", "neutral", "positive"]
SYSTEMS = {
    "hisnet": "HisNet",
    "dp_hisnet": "DP → HisNet",
    "hisnet_dp": "HisNet → DP",
    "lw_hisnet_dp": "LW HisNet → DP",
    "llm": "LLM",
    "dp_llm": "DP → LLM",
}


def analyze(directory):
    cases = json.loads((directory / "cases.json").read_text(encoding="utf-8-sig"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8-sig"))
    details = {}
    lines = ["# 30 cümlelik canlı altı-model kontrolü", "",
             "Bu set elle etiketlenmiş geliştirme/teşhis setidir; paper benchmark sonucu değildir.", "",
             f"LLM modeli: `{manifest['model']['name']}`; digest: `{manifest['model']['digest']}`.",
             f"Ayarlar: `{json.dumps(manifest['settings'], ensure_ascii=False)}`.", "",
             "| Sistem | Tamamlanan | Doğru | Negatif doğru | Nötr doğru | Pozitif doğru |",
             "|---|---:|---:|---:|---:|---:|"]
    all_rows = {}
    for system, label in SYSTEMS.items():
        path = directory / (system + ".json")
        if not path.exists():
            lines.append(f"| {label} | 0/30 | — | — | — | — |")
            continue
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = data["rows"]
        all_rows[system] = rows
        valid = [c for c in cases if rows.get(c["id"], {}).get("prediction") in LABELS]
        correct = sum(rows[c["id"]]["prediction"] == c["gold"] for c in valid)
        per_class = {gold: sum(rows[c["id"]]["prediction"] == gold for c in valid if c["gold"] == gold)
                     for gold in LABELS}
        stats = dict(completed=len(valid), correct=correct, correct_by_class=per_class,
                     invariant_failures={cid: r["invariant_failures"] for cid, r in rows.items()
                                         if r.get("invariant_failures")},
                     repeat_checks=data.get("repeat_checks", {}),
                     wrong_ids=[c["id"] for c in valid if rows[c["id"]]["prediction"] != c["gold"]])
        calls = [call for batch in data.get("batches", []) for call in batch.get("calls", [])]
        if calls:
            # Aggregate persisted calls, including calls from earlier resumed
            # processes; the classifier's stats are process-local.
            prompts = [call["prompt_eval_count"] for call in calls if call.get("prompt_eval_count") is not None]
            stats["runtime"] = dict(calls=len(calls),
                seconds=sum(call["seconds"] for call in calls),
                prompt_tokens=sum(prompts), output_tokens=sum(call.get("eval_count") or 0 for call in calls),
                min_prompt_tokens=min(prompts) if prompts else None,
                max_prompt_tokens=max(prompts) if prompts else None,
                call_errors=[call["error"] for call in calls if call.get("error")],
                stop_reasons=[call.get("done_reason") for call in calls],
                dependency_parse_seconds=sum(b["result"].get("dependency_parse_seconds", 0.) for b in data["batches"]))
        details[system] = stats
        lines.append(f"| {label} | {len(valid)}/30 | {correct}/{len(valid)} | "
                     f"{per_class['negative']}/10 | {per_class['neutral']}/10 | {per_class['positive']}/10 |")
    lines.extend(["", "## Cümle bazında karşılaştırma", "",
                  "| ID | Metin | Beklenen | " + " | ".join(SYSTEMS.values()) + " |",
                  "|---|---|---|" + "---|" * len(SYSTEMS)])
    for case in sorted(cases, key=lambda c: c["id"]):
        predictions = [all_rows.get(system, {}).get(case["id"], {}).get("prediction") or "—" for system in SYSTEMS]
        lines.append(f"| {case['id']} | {case['text']} | {case['gold']} | " + " | ".join(predictions) + " |")
    lines.extend(["", "## Çalışma ve süre kayıtları", "",
                  "Süreler tamamlanan/kaydedilen istek denemelerinden toplanır. Kesilen ve yanıtı kaydedilmeyen "
                  "istekler bu toplama dahil değildir. Klasik kollarda HisNet sonuçları yeniden kullanıldığından "
                  "bu koşu adil bir uçtan uca hız benchmark'ı değildir.", ""])
    for system, stats in details.items():
        if "runtime" in stats:
            rt = stats["runtime"]
            lines.append(f"- {SYSTEMS[system]}: {rt['calls']} istek, {rt['seconds']:.1f} saniye; "
                         f"girdi {rt['prompt_tokens']}, çıktı {rt['output_tokens']} token; "
                         f"istek başına girdi {rt['min_prompt_tokens']}–{rt['max_prompt_tokens']} token; "
                         f"{len(rt['call_errors'])} kayıtlı istek hatası.")
    lines.extend(["", "Token kullanımı sunucunun döndürdüğü sayılardır. Bu küçük setin context gözlemi, "
                  "uzun gerçek yorumlarda truncation olmayacağını kanıtlamaz.", ""])
    (directory / "analysis.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(details, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    analyze(parser.parse_args().directory)
