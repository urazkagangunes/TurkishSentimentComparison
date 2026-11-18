"""Suite identities, overlap audit, equal-dataset summaries, and writer lock."""
import importlib.util
import json
import os
import platform
import shutil
import sqlite3
from collections import defaultdict
from contextlib import closing, contextmanager
from importlib.metadata import distributions
from pathlib import Path

from src.evaluation.checkpoint_store import atomic_json
from src.evaluation.paper_datasets import file_digest
from src.evaluation.paper_metrics import holm, prediction_summary
from src.evaluation.paper_report import write_csv


@contextmanager
def suite_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "suite.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Bu çıktı dizini başka bir koşu tarafından kullanılıyor.") from error
        yield


def snapshot_sources(root, output):
    for source in [*sorted((root / "src").rglob("*.py")),
                   root / "scripts/run_paper_suite.py", root / "scripts/run_six_systems.py",
                   root / "requirements.txt"]:
        target = output / "source_snapshot" / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
    environment = dict(python=platform.python_version(), platform=platform.platform(),
        hardware=dict(machine=platform.machine(), processor=platform.processor(), logical_cpus=os.cpu_count()),
        execution=dict(workers=1, preprocessing="independent_per_system", stanza_use_gpu=False,
                       ollama_device="server-controlled; recorded in Ollama identity/runtime when available"),
        packages=sorted((d.metadata.get("Name", ""), d.version) for d in distributions()))
    if not (output / "environment.json").exists():
        atomic_json(output / "environment.json", environment)


def resource_identity(system):
    """Only load/hash resources required by this system, before its first call."""
    paths = []
    if system not in {"llm", "dp_llm"}:
        for name in ("SentiNet", "MorphologicalAnalysis", "MorphologicalDisambiguation", "Dictionary", "Corpus"):
            spec = importlib.util.find_spec(name)
            if spec is None:
                raise ModuleNotFoundError(f"Eksik bağımlılık: {name}; docs/paper_evaluation.md")
            if name in {"SentiNet", "MorphologicalAnalysis", "MorphologicalDisambiguation", "Dictionary"}:
                base = Path(spec.origin).parent / "data"
                paths.extend(p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    if system not in {"hisnet", "llm"}:
        from stanza.resources.common import DEFAULT_MODEL_DIR
        base = Path(DEFAULT_MODEL_DIR)
        if not (base / "resources.json").exists() or not list((base / "tr").rglob("*.pt")):
            raise FileNotFoundError(f"Stanza Türkçe modelleri bulunamadı: {base}")
        paths.extend([base / "resources.json", *(base / "tr").rglob("*.pt")])
    return [dict(path=str(p.resolve()), sha256=file_digest(p)) for p in sorted(set(paths))]


def register_dataset(output, dataset, cases):
    """Disk index bounds memory when auditing million-row, multi-source corpora."""
    with closing(sqlite3.connect(output / "dataset_index.sqlite3")) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS sources (dataset TEXT PRIMARY KEY, rows INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS entries (
                dataset TEXT, row_id INTEGER, exact TEXT, normalized TEXT, gold TEXT, split TEXT,
                PRIMARY KEY(dataset,row_id));
            CREATE INDEX IF NOT EXISTS exact_text ON entries(exact,dataset);
            CREATE INDEX IF NOT EXISTS normalized_text ON entries(normalized,dataset);
        """)
        if db.execute("SELECT 1 FROM sources WHERE dataset=?", (dataset,)).fetchone():
            return
        with db:
            db.executemany("INSERT INTO entries VALUES (?,?,?,?,?,?)", (
                (dataset,i,c["text_hash"],c["normalized_hash"],c["gold"],c["split"]) for i,c in enumerate(cases)))
            db.execute("INSERT INTO sources VALUES (?,?)", (dataset, len(cases)))


def overlap_report(output):
    if not (output / "dataset_index.sqlite3").exists():
        return
    rows = []
    with closing(sqlite3.connect((output / "dataset_index.sqlite3").resolve().as_uri()+"?mode=ro", uri=True)) as db:
        for key in ("exact", "normalized"):
            # One text hash per dataset, not a many-to-many row join.
            cursor = db.execute(f"SELECT {key},dataset,GROUP_CONCAT(DISTINCT gold),GROUP_CONCAT(DISTINCT split) "
                                f"FROM entries GROUP BY {key},dataset ORDER BY {key},dataset")
            last, bucket, counts = None, [], defaultdict(lambda: [0, 0, 0])
            def accumulate():
                for i,a in enumerate(bucket):
                    for b in bucket[i+1:]:
                        item = counts[(a[0], b[0])]
                        item[0] += 1
                        item[1] += a[1] != b[1]
                        sa, sb = a[2], b[2]
                        item[2] += ("test" in sa and bool(sb & {"train", "dev", "validation"})) or ("test" in sb and bool(sa & {"train", "dev", "validation"}))
            for digest, dataset, labels, splits in cursor:
                if last is not None and digest != last:
                    accumulate()
                    bucket = []
                bucket.append((dataset, set(labels.split(",")), set(splits.split(","))))
                last = digest
            accumulate()
            rows.extend(dict(match=key, first=a, second=b, shared_unique_texts=v[0],
                differing_label_sets=v[1], train_or_dev_test_overlap=v[2]) for (a,b),v in sorted(counts.items()))
    write_csv(output / "dataset_overlap.csv", rows)
    atomic_json(output / "overlap_policy.json", dict(
        interpretation="Shared texts are overlap warnings, not proof of training leakage in these fixed pretrained/lexicon systems.",
        normalization="NFKC/Turkish case/punctuation/whitespace; may merge meaningful punctuation differences",
        limitations="No semantic fuzzy matching, external pretraining corpora or unconfigured within-file splits audited."))


def suite_report(output, manifest):
    reports, missing = [], []
    for spec in manifest["datasets"]:
        path = output / spec["id"] / "paper_summary.json"
        if path.exists():
            reports.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            missing.append(spec["id"])
    buckets, pairs, metric_rows = defaultdict(list), [], []
    for report in reports:
        for system, views in report["metrics"].items():
            m = views["common_success"]
            row = dict(dataset=report["dataset"]["id"], system=system, complete=report["complete"],
                       total=report["total"], common_success=report["common_success"], evaluation_n=m["n"],
                       macro_labels=m["macro_labels"],
                       **{key:m[key] for key in ("accuracy", "error_rate", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "mcc")})
            for label, values in m["classes"].items():
                row.update({f"{label}_{key}":values[key] for key in ("support", "correct", "precision", "recall", "f1")})
            row.update(prediction_summary(m["prediction_distribution"]))
            metric_rows.append(row)
        pairs.extend(dict(p, dataset=report["dataset"]["id"]) for p in report["pairs"])
        if not report["complete"] or not report["dataset"].get("include_in_aggregate"):
            continue
        for system, views in report["metrics"].items():
            m = views["common_success"]
            if m["n"]:
                buckets[(tuple(m["macro_labels"]), system)].append((report["dataset"]["id"], m))
    aggregate = []
    for (labels, system), items in sorted(buckets.items()):
        values = {k:sum(m[k] for _,m in items)/len(items) for k in
                  ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "mcc")}
        for label in labels:
            for metric in ("precision", "recall", "f1"):
                values[f"{label}_{metric}"] = sum(m["classes"][label][metric] for _,m in items)/len(items)
        aggregate.append(dict(system=system, task_labels=list(labels), dataset_count=len(items),
                              datasets=[i for i,_ in items], **values))
    # Preserve within-dataset adjustment and add the all-dataset comparison family.
    for p in pairs:
        p["dataset_holm_adjusted_p"] = p.pop("holm_adjusted_p", None)
    holm(pairs)
    for p in pairs:
        p["suite_holm_adjusted_p"] = p.pop("holm_adjusted_p", None)
    write_csv(output / "suite_aggregate.csv", aggregate, list(dict.fromkeys(k for r in aggregate for k in r)))
    write_csv(output / "suite_metrics.csv", metric_rows)
    write_csv(output / "suite_pairwise.csv", pairs)
    atomic_json(output / "suite_summary.json", dict(reported=[r["dataset"]["id"] for r in reports], missing=missing,
        incomplete=[r["dataset"]["id"] for r in reports if not r["complete"]], aggregate=aggregate,
        policy="Equal dataset weights within the same gold-label task; complete opted-in datasets only. Overlapping sources are not independent replications.",
        mcnemar_policy="Exact McNemar assumes independent examples. Correlated clusters require the reported cluster-bootstrap intervals; McNemar p-values are secondary."))
