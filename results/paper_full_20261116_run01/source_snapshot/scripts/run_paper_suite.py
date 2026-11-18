"""Six-system, full-dataset paper evaluation with durable automatic resume.

python -m scripts.run_paper_suite --output results/paper_full
Repeat exactly the same command to resume. --inventory/--prepare-only/--status/
--analyze-only never perform model inference. See docs/paper_evaluation.md.
"""
import argparse
import json
import math
import os
import sqlite3
from contextlib import closing
from pathlib import Path

from src.evaluation.checkpoint_store import CheckpointStore, SYSTEMS, atomic_json
from src.evaluation.paper_datasets import catalog, load_dataset, file_digest
from src.evaluation.paper_metrics import PRIMARY_PAIRS
from src.evaluation.paper_report import analyze_dataset, write_csv
from src.evaluation.paper_runner import evaluate_dataset, prepare_metadata, quick_status, heartbeat
from src.evaluation.paper_suite import (suite_lock, snapshot_sources, resource_identity,
    register_dataset, overlap_report, suite_report)
from src.evaluation.resumable_evaluator import make_classifier, LLM_SYSTEMS
from src.evaluation.run_identity import implementation_fingerprint, ensure_run_manifest
from scripts.run_six_systems import ollama_identity

ROOT = Path(__file__).resolve().parents[1]


def emit(message):
    print(message, flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "results/paper_full")
    p.add_argument("--datasets-root", type=Path, default=ROOT / "datasets")
    p.add_argument("--config", type=Path, help="JSON: {datasets: [explicit per-file overrides]}")
    p.add_argument("--datasets", nargs="+", help="Enabled dataset IDs; default: every verified/enabled source")
    p.add_argument("--systems", choices=SYSTEMS, nargs="+", default=list(SYSTEMS))
    mode = p.add_mutually_exclusive_group()
    for name in ("inventory", "prepare-only", "status", "analyze-only"):
        mode.add_argument("--"+name, action="store_true")
    p.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen3:32b"))
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--num-ctx", type=int, default=8192)
    p.add_argument("--max-retries", type=int, default=1)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap", type=int, default=2000, help="CI resamples; 0 disables CIs for quick checks")
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--heartbeat-seconds", type=float, default=20)
    p.add_argument("--keep-details", action="store_true", help="Store full parses/scopes too; substantially more disk")
    args = p.parse_args(argv)
    if min(args.batch_size, args.num_ctx, args.progress_every) < 1 or min(args.max_retries, args.bootstrap) < 0:
        p.error("Batch/context/progress pozitif; retries/bootstrap negatif olmamalı")
    if not math.isfinite(args.timeout) or args.timeout <= 0 or not math.isfinite(args.heartbeat_seconds) or args.heartbeat_seconds < 0:
        p.error("Timeout pozitif ve sonlu; heartbeat sıfır veya pozitif ve sonlu olmalı")
    return args


def status(output, manifest):
    result = {}
    for spec in manifest["datasets"]:
        path = output / spec["id"] / "progress.sqlite3"
        if not path.exists():
            result[spec["id"]] = dict(status="not_prepared")
            continue
        with closing(sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True)) as db:
            db.execute("BEGIN")
            n = db.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
            result[spec["id"]] = quick_status(db, manifest["systems"], n)
    return result


def run(args):
    output = args.output.resolve()
    if args.status or args.analyze_only:
        manifest = json.loads((output / "suite_manifest.json").read_text(encoding="utf-8"))
        if args.status:
            emit(json.dumps(status(output, manifest), ensure_ascii=False, indent=2))
            return 0
        with suite_lock(output):
            for spec in manifest["datasets"]:
                if (output / spec["id"] / "progress.sqlite3").exists():
                    with heartbeat(f"{spec['id']} | rapor hesaplanıyor", args.heartbeat_seconds, emit):
                        analyze_dataset(output / spec["id"], args.bootstrap, args.seed)
            overlap_report(output)
            suite_report(output, manifest)
        return 0

    root = args.datasets_root.resolve()
    if not root.is_dir():
        raise ValueError("Veri kökü bulunamadı: " + str(root))
    overrides = json.loads(args.config.read_text(encoding="utf-8")).get("datasets", []) if args.config else []
    inventory = catalog(root, overrides)
    if args.inventory:
        emit(json.dumps(inventory, ensure_ascii=False, indent=2))
        return 0
    selected = [s for s in inventory if s["enabled"] and (not args.datasets or s["id"] in args.datasets)]
    unknown = set(args.datasets or ()) - {s["id"] for s in selected}
    if unknown:
        raise ValueError("Bilinmeyen veya etkinleştirilmemiş dataset: " + ", ".join(sorted(unknown)))
    if not selected:
        raise ValueError("Etkin veri seti yok; --inventory ve --config kullanın")
    selected = [dict(s, sha256=file_digest(root / s["path"])) for s in selected]
    systems = list(dict.fromkeys(args.systems))
    settings = dict(model=args.model, host=args.host, batch_size=args.batch_size, num_ctx=args.num_ctx,
        max_retries=args.max_retries, request_timeout=args.timeout, seed=args.seed,
        temperature=0.1, top_p=1.0, keep_alive="30m")
    manifest = dict(paper_protocol=1, implementation=implementation_fingerprint(),
        runner_sha256=file_digest(__file__), identity_helper_sha256=file_digest(ROOT / "scripts/run_six_systems.py"),
        datasets_root=str(root), datasets=selected, systems=systems, settings=settings,
        bootstrap=args.bootstrap, bootstrap_seed=args.seed, primary_pairs=[list(p) for p in PRIMARY_PAIRS],
        keep_details=args.keep_details, inventory=inventory,
        recovery_policy="successful rows immutable; failed rows retried individually once per invocation",
        exclusions="No human gold negation cue/scope annotation evaluation (4.B)",
        design="Full eligible natural distributions; independent inference per system; no hyperparameter selection",
        context_policy="utf8_bytes_plus_1024_template_plus_128_and_64_per_review_output")

    with suite_lock(output):
        ensure_run_manifest(output / "suite_manifest.json", manifest,
                            list(output.glob("*/progress.sqlite3")))
        snapshot_sources(ROOT, output)
        atomic_json(output / "dataset_inventory.json", inventory)
        write_csv(output / "dataset_inventory.csv", inventory,
                  sorted({k for s in inventory for k in s}))
        emit(f"[KOŞU] {len(selected)} veri seti × {len(systems)} sistem | diğer {len(inventory)-len(selected)} kaynak envanterde | {output}")
        identities = {}
        complete = True
        for index, spec in enumerate(selected, 1):
            emit(f"[DATASET BAŞLADI] {index}/{len(selected)} | {spec['id']} | {spec['path']}")
            with heartbeat(f"{spec['id']} | veri doğrulama", args.heartbeat_seconds, emit):
                cases, validation = load_dataset(root, spec)
            if validation["source_sha256"] != spec["sha256"]:
                raise ValueError("Dataset koşu hazırlanırken değişti: " + spec["id"])
            metadata = dict(suite_sha256=file_digest(output / "suite_manifest.json"),
                            paper=dict(dataset=spec, systems=systems, validation=validation))
            directory = output / spec["id"]
            with CheckpointStore(directory, metadata, cases) as store:
                prepare_metadata(store, cases)
                atomic_json(directory / "dataset_validation.json", validation)
                register_dataset(output, spec["id"], cases)
                if args.prepare_only:
                    emit(f"[DATASET HAZIR] {spec['id']} | {len(cases)} örnek | modeller çalıştırılmadı")
                    continue

                def factory(system, config):
                    if system not in identities:
                        resources = resource_identity(system)
                        identity = dict(resources=resources)
                        if system in LLM_SYSTEMS:
                            if "ollama" not in identities:
                                identities["ollama"] = ollama_identity(args.host, args.model)
                            identity["ollama"] = identities["ollama"]
                        identities[system] = identity
                    # Suite identity also protects continuation into a *new* dataset.
                    ensure_run_manifest(output / f"resources_{system}.json", identities[system])
                    store.ensure_identity("resources_"+system, identities[system])
                    return make_classifier(system, config)

                progress = evaluate_dataset(cases, store, settings, systems, factory, spec["id"],
                    args.progress_every, args.heartbeat_seconds, args.keep_details, emit)
            del cases
            done = all(v["complete"] for v in progress["systems"].values())
            complete &= done
            # Analysis is restartable from SQLite; it never reruns successful inference.
            with heartbeat(f"{spec['id']} | rapor hesaplanıyor", args.heartbeat_seconds, emit):
                analyze_dataset(directory, args.bootstrap, args.seed)
            emit(f"[DATASET {'BİTTİ' if done else 'EKSİK'}] {index}/{len(selected)} | {spec['id']}")
            atomic_json(output / "suite_progress.json", status(output, manifest))
        with heartbeat("Veri örtüşmeleri ve genel rapor", args.heartbeat_seconds, emit):
            overlap_report(output)
            if not args.prepare_only:
                suite_report(output, manifest)
        emit("[HAZIRLIK BİTTİ] Model çalıştırılmadı" if args.prepare_only else
             "[KOŞU BİTTİ] Raporlar hazır" if complete else "[KOŞU EKSİK] Başarılı kayıtlar korundu; hataları giderip aynı komutu çalıştırın")
        return 0 if complete else 2


def main(argv=None):
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        emit("[KESİLDİ] Tamamlanmış kayıtlar güvende. Aynı komutla devam edin; uçuşta olan örnek/batch yeniden çalışabilir.")
        return 130
    except (ValueError, OSError, RuntimeError) as error:
        emit(f"[HATA] {type(error).__name__}: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
