"""Transactional, bounded-memory progress storage for the six-system runner."""
import csv
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from datetime import datetime, timezone

SYSTEMS = ("hisnet", "hisnet_dp", "dp_hisnet", "lw_hisnet_dp", "llm", "dp_llm")
LABELS = ("negative", "neutral", "positive")


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def dataset_digest(cases):
    digest = hashlib.sha256()
    for case in cases:
        digest.update(encode([case["text"], case["gold"]]).encode("utf-8") + b"\n")
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(encode(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class CheckpointStore:
    """One writer per directory; completed batch + predictions commit together.

    The OS releases the lock after a crash. SQLite FULL synchronization keeps
    committed work durable; a killed/in-flight request may have to be repeated.
    """

    def __init__(self, output, metadata, cases):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.connection = None
        self.lock = (self.output / "run.lock").open("a+b")
        try:
            self.lock.seek(0, 2)
            if self.lock.tell() == 0:
                self.lock.write(b"0")
                self.lock.flush()
            self.lock.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.lock.close()
            raise RuntimeError("Another process is using this output directory.") from error
        try:
            self.connection = sqlite3.connect(self.output / "progress.sqlite3")
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS cases (
                    row_id INTEGER PRIMARY KEY, text TEXT NOT NULL, gold TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY, system TEXT NOT NULL, row_ids TEXT NOT NULL,
                    recovery INTEGER NOT NULL, status TEXT NOT NULL, started TEXT NOT NULL,
                    seconds REAL, result TEXT);
                CREATE TABLE IF NOT EXISTS predictions (
                    system TEXT NOT NULL, row_id INTEGER NOT NULL REFERENCES cases(row_id),
                    status TEXT NOT NULL, prediction TEXT, error TEXT,
                    attempt_id INTEGER NOT NULL REFERENCES attempts(id),
                    PRIMARY KEY(system, row_id));
                CREATE INDEX IF NOT EXISTS attempt_system ON attempts(system);
            """)
            identity = dict(metadata, checkpoint_version=1, dataset_sha256=dataset_digest(cases),
                            total_examples=len(cases))
            stored = self.connection.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
            if stored is not None and stored[0] != encode(identity):
                raise ValueError("Run manifest differs; use a new output directory. Saved results were preserved.")
            if stored is None:
                if self.connection.execute("SELECT COUNT(*) FROM cases").fetchone()[0]:
                    raise ValueError("Checkpoint has data without a manifest.")
                with self.connection:
                    self.connection.execute("INSERT INTO metadata VALUES ('manifest', ?)", (encode(identity),))
                    self.connection.executemany("INSERT INTO cases VALUES (?, ?, ?)",
                        ((i, c["text"], c["gold"]) for i, c in enumerate(cases)))
            with self.connection:
                self.connection.execute("UPDATE attempts SET status='interrupted' WHERE status='running'")
            atomic_json(self.output / "manifest.json", identity)
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if not self.lock.closed:
            self.lock.close()

    def states(self, system):
        return dict(self.connection.execute("SELECT row_id, status FROM predictions WHERE system=?", (system,)))

    def ensure_identity(self, key, value):
        """Bind lazily loaded external resources before their first prediction."""
        stored = self.connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        if stored is not None and stored[0] != encode(value):
            raise ValueError(f"{key} differs; use a new output directory. Saved results were preserved.")
        if stored is None:
            with self.connection:
                self.connection.execute("INSERT INTO metadata VALUES (?,?)", (key, encode(value)))
        atomic_json(self.output / f"{key}.json", value)

    def start(self, system, ids, recovery):
        if system not in SYSTEMS or not ids or len(set(ids)) != len(ids):
            raise ValueError("Invalid system or batch row IDs.")
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO attempts(system,row_ids,recovery,status,started) VALUES (?,?,?,'running',?)",
                (system, encode(ids), int(recovery), datetime.now(timezone.utc).isoformat()))
        return cursor.lastrowid

    def finish(self, attempt_id, result, seconds):
        system, row_ids, status = self.connection.execute(
            "SELECT system,row_ids,status FROM attempts WHERE id=?", (attempt_id,)).fetchone()
        ids = json.loads(row_ids)
        predictions = result["predictions"]
        if status != "running" or len(predictions) != len(ids):
            raise ValueError("Attempt state/prediction count mismatch.")
        if result["status"] not in {"success", "failed"}:
            raise ValueError("Invalid result status.")
        if result["status"] == "success" and any(p not in LABELS for p in predictions):
            raise ValueError("Successful results must have valid labels.")
        if result["status"] == "failed" and any(p is not None for p in predictions):
            raise ValueError("Failed results cannot contain sentiment labels.")
        payload = encode(result)
        with self.connection:
            self.connection.execute("UPDATE attempts SET status=?,seconds=?,result=? WHERE id=?",
                (result["status"], seconds, payload, attempt_id))
            self.connection.executemany("""
                INSERT INTO predictions VALUES (?,?,?,?,?,?)
                ON CONFLICT(system,row_id) DO UPDATE SET status=excluded.status,
                    prediction=excluded.prediction, error=excluded.error, attempt_id=excluded.attempt_id
            """, ((system, row_id, result["status"], prediction, result.get("error"), attempt_id)
                  for row_id, prediction in zip(ids, predictions)))

    def export(self):
        summary = summarize_connection(self.connection)
        atomic_json(self.output / "summary.json", summary)
        path = self.output / "predictions.csv"
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["row_id", "text", "gold", "system", "status", "prediction", "error"])
            writer.writerows(self.connection.execute("""
                SELECT c.row_id,c.text,c.gold,p.system,p.status,p.prediction,p.error
                FROM predictions p JOIN cases c ON c.row_id=p.row_id ORDER BY p.system,c.row_id
            """))
        temporary.replace(path)
        return summary


def summarize_connection(connection):
    total = connection.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
    summary = {}
    for system in SYSTEMS:
        counts = dict(connection.execute(
            "SELECT status,COUNT(*) FROM predictions WHERE system=? GROUP BY status", (system,)))
        success, failed = counts.get("success", 0), counts.get("failed", 0)
        matrix = [[0] * len(LABELS) for _ in LABELS]
        for gold, pred, count in connection.execute("""
                SELECT c.gold,p.prediction,COUNT(*) FROM predictions p JOIN cases c USING(row_id)
                WHERE p.system=? AND p.status='success' GROUP BY c.gold,p.prediction
                """, (system,)):
            matrix[LABELS.index(gold)][LABELS.index(pred)] = count
        correct = sum(matrix[i][i] for i in range(len(LABELS)))
        f1 = []
        for i in range(len(LABELS)):
            denominator = sum(matrix[i]) + sum(row[i] for row in matrix)
            f1.append(2 * matrix[i][i] / denominator if denominator else 0.0)
        usage = dict(seconds=0.0, prompt_tokens=0, output_tokens=0, requests=0, interrupted_attempts=0)
        for status, seconds, payload in connection.execute(
                "SELECT status,seconds,result FROM attempts WHERE system=?", (system,)):
            usage["interrupted_attempts"] += status in {"interrupted", "running"}
            usage["seconds"] += seconds or 0.0
            result = json.loads(payload) if payload else {}
            for key in ("prompt_tokens", "output_tokens"):
                usage[key] += result.get(key, 0)
            usage["requests"] += result.get("attempt", 0)
        summary[system] = dict(total=total, success=success, failed=failed, pending=total-success-failed,
            complete=success == total, coverage=success / total if total else 0.0,
            accuracy_on_successful=correct / success if success else None,
            macro_f1_on_successful=sum(f1) / len(f1) if success else None,
            confusion_matrix_on_successful=matrix, label_order=LABELS, usage=usage)
    return summary


def read_status(output):
    path = Path(output).resolve() / "progress.sqlite3"
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("BEGIN")
        return summarize_connection(connection)
