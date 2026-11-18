"""Durable inference, visible progress, heartbeat, and ten-percent milestones."""
import gc
import json
import threading
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone

from src.evaluation.checkpoint_store import LABELS, atomic_json
from src.evaluation.paper_diagnostics import diagnostics, text_flags
from src.evaluation.resumable_evaluator import LLM_SYSTEMS


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def heartbeat(message, interval, emit=print):
    stop = threading.Event()
    started = time.monotonic()
    def pulse():
        while not stop.wait(interval):
            emit(f"[BEKLENİYOR] {message} | {time.monotonic()-started:.0f} sn")
    thread = threading.Thread(target=pulse, daemon=True)
    if interval > 0:
        thread.start()
    try:
        yield
    finally:
        stop.set()
        if thread.is_alive():
            thread.join(timeout=1)


def prepare_metadata(store, cases):
    store.connection.executescript("""
        CREATE TABLE IF NOT EXISTS paper_cases (row_id INTEGER PRIMARY KEY REFERENCES cases(row_id), metadata TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS paper_initializations (
            id INTEGER PRIMARY KEY, system TEXT, started TEXT, seconds REAL, status TEXT, error TEXT);
    """)
    count = store.connection.execute("SELECT COUNT(*) FROM paper_cases").fetchone()[0]
    if count not in (0, len(cases)):
        raise ValueError("Incomplete paper case metadata")
    if count == 0:
        with store.connection:
            store.connection.executemany("INSERT INTO paper_cases VALUES (?,?)", (
                (i, json.dumps({**{k:v for k,v in c.items() if k not in {"text", "gold"}},
                    "text_flags": text_flags(c["text"], c.get("source_label_conflict", False))}, ensure_ascii=False))
                for i, c in enumerate(cases)))


class StageTimer:
    """Instance-only stage timing; residual includes composition and other overhead."""
    def __init__(self, classifier, system):
        self.values = {}
        self.originals = []
        self.unmatched_roots = Counter()
        self.failed_stage = None
        if system in LLM_SYSTEMS:
            return  # Existing LLM results already expose parse and request times.
        hisnet = classifier if system == "hisnet" else getattr(classifier, "hisnet", None)
        if hisnet is None and hasattr(classifier, "hisnet"):
            from src.hisnet.hisnet_classifier import HisNetClassifier
            classifier.hisnet = hisnet = HisNetClassifier()
        for obj, method, key in ((getattr(classifier, "parser", None), "parse", "parse_seconds"),
                                 (getattr(hisnet, "root_extractor", None), "analyze_sentence", "morphology_seconds"),
                                 (hisnet, "_lookup_root", "lexicon_lookup_seconds")):
            if obj is not None and hasattr(obj, method):
                original = getattr(obj, method)
                self.originals.append((obj, method, original))
                def timed(*args, _original=original, _key=key, **kwargs):
                    start = time.perf_counter()
                    try:
                        value = _original(*args, **kwargs)
                        if _key == "lexicon_lookup_seconds" and value is None:
                            self.unmatched_roots[args[0] if args else kwargs["root"]] += 1
                        return value
                    except Exception:
                        self.failed_stage = _key
                        raise
                    finally:
                        self.values[_key] = self.values.get(_key, 0.0) + time.perf_counter()-start
                setattr(obj, method, timed)

    def close(self):
        for obj, method, original in self.originals:
            setattr(obj, method, original)


def quick_status(connection, systems, n):
    result = {}
    for system in systems:
        counts = dict(connection.execute("SELECT status,COUNT(*) FROM predictions WHERE system=? GROUP BY status", (system,)))
        ok, failed = counts.get("success", 0), counts.get("failed", 0)
        result[system] = dict(success=ok, failed=failed, pending=n-ok-failed, total=n, complete=ok == n)
    return result


class Progress:
    def __init__(self, store, dataset, systems, n, emit):
        self.store, self.dataset, self.systems, self.n, self.emit = store, dataset, systems, n, emit
        self.directory = store.output / "checkpoints"
        self.directory.mkdir(exist_ok=True)
        self.saved_milestones = {p.name for p in self.directory.glob("*.json")}
        self.counts = quick_status(store.connection, systems, n)

    def snapshot(self, system=None, force=False):
        counts = self.counts
        completed = sum(v["success"] for v in counts.values())
        total = self.n*len(self.systems)
        payload = dict(dataset=self.dataset, updated=now(), completed_decisions=completed,
                       total_decisions=total, dataset_percent=100*completed/total, systems=counts,
                       durability="each completed review/batch committed; in-flight work may repeat")
        milestones = [("dataset", completed, total)]
        if system:
            milestones.append((system, counts[system]["success"], self.n))
        reached = []
        for name, success, size in milestones:
            for percent in range(10, 101, 10):
                target = self.directory / f"{name}_{percent:03}.json"
                if success*100 >= percent*size and target.name not in self.saved_milestones:
                    atomic_json(target, dict(payload, milestone=percent, milestone_scope=name))
                    self.saved_milestones.add(target.name)
                    reached.append(f"{name} %{percent}")
        if reached or force:
            self.store.connection.execute("PRAGMA wal_checkpoint(PASSIVE)")
            atomic_json(self.store.output / "progress.json", payload)
        if reached:
            self.emit(f"[KAYIT] {self.dataset} | {', '.join(reached)} | SQLite kalıcı kayıt hazır")
        return payload


def evaluate_dataset(cases, store, settings, systems, factory, dataset,
                     progress_every=25, heartbeat_seconds=20, keep_details=False,
                     emit=lambda message: print(message, flush=True)):
    prepare_metadata(store, cases)
    progress = Progress(store, dataset, systems, len(cases), emit)
    progress.snapshot(force=True)
    for system in systems:
        states = store.states(system)
        if len(states) == len(cases) and all(v == "success" for v in states.values()):
            emit(f"[ATLANDI] {dataset} | {system} | {len(cases)}/{len(cases)} önceden tamamlandı")
            progress.snapshot(system, force=True)
            continue
        emit(f"[MODEL BAŞLADI] {dataset} | {system} | kayıtlı başarı={progress.counts[system]['success']}/{len(cases)}")
        start, started = time.perf_counter(), now()
        classifier, timer = None, None
        try:
            with heartbeat(f"{dataset} | {system} | model yükleniyor", heartbeat_seconds, emit):
                classifier = factory(system, settings)
                timer = StageTimer(classifier, system)
        except Exception as error:
            with store.connection:
                store.connection.execute("INSERT INTO paper_initializations VALUES(NULL,?,?,?,?,?)",
                    (system, started, time.perf_counter()-start, "failed", f"{type(error).__name__}: {error}"))
            emit(f"[MODEL HATASI] {dataset} | {system} | {type(error).__name__}: {error}")
            if classifier is not None and getattr(classifier, "client", None) is not None:
                classifier.client.close()
            continue
        with store.connection:
            store.connection.execute("INSERT INTO paper_initializations VALUES(NULL,?,?,?,?,?)",
                (system, started, time.perf_counter()-start, "success", None))
        processed = 0

        def run(ids, recovery=False):
            nonlocal processed
            phase = "KURTARMA" if recovery else "ÖRNEK"
            description = f"{dataset} | {system} | {phase} {ids[0]+1}..{ids[-1]+1}/{len(cases)}"
            if processed == 0 or recovery:
                emit(f"[ÇALIŞIYOR] {description}")
            attempt = store.start(system, ids, recovery)
            timer.values.clear()
            timer.unmatched_roots.clear()
            timer.failed_stage = None
            start = time.perf_counter()
            try:
                with heartbeat(description, heartbeat_seconds, emit):
                    if system in LLM_SYSTEMS:
                        result = classifier.classify_batch([cases[i]["text"] for i in ids])
                    else:
                        result = dict(classifier.classify(cases[ids[0]]["text"]))
                        result.update(status="success", predictions=[result["prediction"]], error=None)
                if (result.get("status") not in {"success", "failed"}
                    or len(result.get("predictions", [])) != len(ids)
                    or (result["status"] == "success" and any(p not in LABELS for p in result["predictions"]))
                    or (result["status"] == "failed" and any(p is not None for p in result["predictions"]))):
                    raise ValueError("Invalid classifier result contract")
            except Exception as error:
                result = dict(status="failed", predictions=[None]*len(ids),
                              error=f"{type(error).__name__}: {error}",
                              error_code={"parse_seconds":"parse_error", "morphology_seconds":"morphology_error",
                                          "lexicon_lookup_seconds":"lexicon_error"}.get(timer.failed_stage, "classifier_exception"))
            seconds = time.perf_counter()-start
            timing = dict(timer.values)
            if system in LLM_SYSTEMS:
                timing.update(parse_seconds=result.get("dependency_parse_seconds", 0.0),
                              llm_request_seconds=result.get("request_seconds", 0.0))
            timing["other_pipeline_seconds"] = max(0.0, seconds-sum(timing.values()))
            kept = {k:v for k,v in result.items() if k in {
                "status", "predictions", "error", "error_code", "attempt", "model", "raw_response",
                "prompt_tokens", "output_tokens", "request_seconds", "dependency_parse_seconds"}}
            kept["paper_timing"] = timing
            kept["paper_diagnostics"] = diagnostics(result, timer.unmatched_roots) if system not in LLM_SYSTEMS else None
            if keep_details:
                kept["full_result"] = result
            # Prediction and diagnostics are committed in ONE SQLite transaction.
            store.finish(attempt, kept, seconds)
            for i in ids:
                old = states.get(i)
                if old:
                    progress.counts[system][old] -= 1
                else:
                    progress.counts[system]["pending"] -= 1
                states[i] = result["status"]
                progress.counts[system][result["status"]] += 1
            progress.counts[system]["complete"] = progress.counts[system]["success"] == len(cases)
            processed += len(ids)
            progress.snapshot(system)
            if processed % progress_every < len(ids) or result["status"] == "failed" or progress.counts[system]["pending"] == 0:
                c = progress.counts[system]
                emit(f"[İLERLEME] {dataset} | {system} | son örnek {ids[-1]+1}/{len(cases)} | "
                     f"başarılı={c['success']} hata={c['failed']} bekleyen={c['pending']} | %{100*c['success']/len(cases):.1f}")

        try:
            size = settings["batch_size"] if system in LLM_SYSTEMS else 1
            for start in range(0, len(cases), size):
                ids = [i for i in range(start, min(start+size, len(cases))) if i not in states]
                if ids:
                    run(ids)
            for i in sorted(i for i,v in states.items() if v == "failed"):
                run([i], recovery=True)
        finally:
            timer.close()
            if getattr(classifier, "client", None) is not None:
                classifier.client.close()
            del classifier
            gc.collect()
            progress.snapshot(system, force=True)
        c = progress.counts[system]
        emit(f"[MODEL {'BİTTİ' if c['complete'] else 'EKSİK'}] {dataset} | {system} | başarı={c['success']}/{len(cases)} hata={c['failed']}")
    return progress.snapshot(force=True)
