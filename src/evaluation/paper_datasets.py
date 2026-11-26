"""Explicit dataset eligibility, adapters and provenance; never guess numeric labels."""
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from src.evaluation.checkpoint_store import LABELS

UNLABELED = "unlabeled"
READY_FILES = {
    "TRSAv1_fixed.csv": "trsav1",
    "beyazperde_sentiment.csv": "beyazperde",
    "combined_dataset.csv": "combined_dataset",
    "combined_sentiment.csv": "combined_sentiment",
    "HUMIRSentimentDatasets.csv": "humir",
    "reviews_combined.csv": "reviews_combined",
    "tr_polarity_combined.csv": "tr_polarity",
    "Turkish_twitter_combined.csv": "turkish_twitter",
}


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_digest(text, normalized=False):
    if normalized:
        text = unicodedata.normalize("NFKC", text).replace("I", "ı").replace("İ", "i").lower()
        text = " ".join(re.findall(r"\w+", text, flags=re.UNICODE))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def catalog(root, overrides=None):
    """Every physical file appears, including unsupported/alternate representations."""
    root = Path(root).resolve()
    overrides = overrides or []
    if len({s["path"] for s in overrides}) != len(overrides):
        raise ValueError("Duplicate dataset paths in config")
    by_path = {s["path"]: s for s in overrides}
    result = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        spec = dict(id="source_" + hashlib.sha256(relative.encode()).hexdigest()[:12],
                    path=relative, enabled=False, status="needs_schema", task="sentiment",
                    reason="Format/etiket anlamı doğrulanmalı", split="unspecified",
                    include_in_aggregate=False)
        if relative.startswith("Modified/sentence_sentiment/") and path.name in READY_FILES:
            spec.update(id=READY_FILES[path.name], enabled=True, status="ready", adapter="csv",
                        separator=";", text_column="review", label_column="score",
                        reason="Doğrulanmış metin ve İngilizce sentiment etiketleri",
                        include_in_aggregate=True)
        elif path.name == "TRSAv1.csv":
            spec.update(status="alternate_representation", reason="TRSAv1_fixed.csv kanonik sürümdür")
        elif relative.startswith("Modified/reviews/"):
            spec.update(status="needs_label_mapping", adapter="csv", separator=";",
                        text_column="review", label_column="star" if path.stem == "urunler-yorumlar" else "score",
                        reason="Sayısal etiket/yıldız eşlemesi ve kaynağı gerekli")
        elif relative.startswith("Modified/aspect_sentiment/"):
            spec.update(status="different_task", reason="Aspect gold'u genel yorum sentiment gold'u değildir")
        elif relative.startswith("Turkish Sentiment Treebank/"):
            spec.update(status="needs_label_mapping", adapter="tree", split=path.stem,
                        reason="Kök düğüm etiket eşlemesi ve kanonik temsil doğrulanmalı")
        elif relative.startswith("Unlabeled/"):
            spec.update(status="unlabeled", task="inference", adapter="xlsx" if path.suffix == ".xlsx" else "lines",
                        reason="Gold doğrulanmadı; yapılandırma ile yalnız tahmin yapılabilir")
        if relative in by_path:
            spec.update(by_path.pop(relative))
            if spec["enabled"]:
                spec["status"] = "ready"
                spec["reason"] = "Kullanıcı yapılandırması"
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", spec["id"]):
            raise ValueError("Unsafe dataset id: " + spec["id"])
        if spec["task"] not in {"sentiment", "inference"}:
            raise ValueError("Only document sentiment/inference is supported: " + relative)
        if spec.get("label_map") and not spec.get("mapping_source"):
            raise ValueError("Explicit label_map requires mapping_source: " + relative)
        spec["bytes"] = path.stat().st_size
        result.append(spec)
    if by_path:
        raise ValueError("Configured files do not exist: " + ", ".join(by_path))
    if len({s["id"] for s in result}) != len(result):
        raise ValueError("Dataset IDs must be unique")
    return result


def _column(row, name):
    key = next((k for k in row if str(k).strip().lower() == str(name).lower()), None)
    if key is None:
        raise ValueError(f"Missing column {name!r}; columns={list(row)}")
    return row[key]


def read_rows(path, spec):
    """Yield source id, raw text, raw label, optional group. No model imports."""
    adapter = spec.get("adapter")
    if adapter == "csv":
        with path.open(encoding=spec.get("encoding", "utf-8-sig"), newline="") as stream:
            for index, row in enumerate(csv.DictReader(stream, delimiter=spec.get("separator", ";")), 1):
                if None in row:
                    raise ValueError(f"Malformed CSV row {index}: {path}")
                yield str(index), _column(row, spec["text_column"]), (
                    _column(row, spec["label_column"]) if spec["task"] == "sentiment" else None), (
                    _column(row, spec["group_column"]) if spec.get("group_column") else None)
    elif adapter == "xlsx":
        from openpyxl import load_workbook
        book = load_workbook(path, read_only=True, data_only=True)
        try:
            sheet = book[spec["sheet"]] if spec.get("sheet") else book.active
            rows = sheet.iter_rows(values_only=True)
            header = next(rows)
            for index, values in enumerate(rows, 1):
                row = dict(zip(header, values))
                yield str(index), _column(row, spec["text_column"]), (
                    _column(row, spec["label_column"]) if spec["task"] == "sentiment" else None), (
                    _column(row, spec["group_column"]) if spec.get("group_column") else None)
        finally:
            book.close()
    elif adapter in {"lines", "tree"}:
        with path.open(encoding=spec.get("encoding", "utf-8-sig")) as stream:
            for index, line in enumerate(stream, 1):
                if adapter == "tree":
                    # Bracketed sentiment trees: use root gold and surface leaves only.
                    root = re.match(r"^\s*\(\s*([^\s()]+)", line)
                    leaves = re.findall(r"\(\s*[^\s()]+\s+([^()]+?)\s*\)", line)
                    if not root or not leaves or line.count("(") != line.count(")"):
                        raise ValueError(f"Invalid sentiment tree at {path}:{index}")
                    yield str(index), " ".join(leaves), root[1], None
                else:
                    yield str(index), line, spec.get("constant_label"), None
    else:
        raise ValueError("Unsupported adapter: " + str(adapter))


def load_dataset(root, spec):
    path = (Path(root) / spec["path"]).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Dataset path escapes root")
    counts, cases, seen, raw_labels = Counter(), [], {}, Counter()
    conflicts = defaultdict(set)
    mapping = {str(k): v for k, v in spec.get("label_map", {}).items()}
    for source_id, raw_text, raw_gold, group in read_rows(path, spec):
        counts["raw_rows"] += 1
        if raw_text is None or (spec["task"] == "sentiment" and raw_gold is None):
            counts["missing_rows"] += 1
            continue
        text = str(raw_text).strip()
        if not text or (spec["task"] == "sentiment" and not str(raw_gold).strip()):
            counts["empty_rows"] += 1
            continue
        raw = str(raw_gold).strip()
        raw_labels[raw] += 1
        gold = UNLABELED if spec["task"] == "inference" else mapping.get(raw, raw.lower())
        if gold not in (*LABELS, UNLABELED) or (gold == UNLABELED and spec["task"] != "inference"):
            raise ValueError(f"Unmapped label {raw!r} at {spec['path']}:{source_id}; supply label_map and mapping_source")
        key = (text, gold)
        if key in seen:
            counts["duplicate_pairs_removed"] += 1
            # Retain all source IDs so no source record silently disappears.
            cases[seen[key]]["source_ids"].append(source_id)
            if group is not None and str(group) not in cases[seen[key]]["source_groups"]:
                cases[seen[key]]["source_groups"].append(str(group))
            continue
        normalized = text_digest(text, normalized=True)
        seen[key] = len(cases)
        conflicts[text].add(gold)
        cases.append(dict(text=text, gold=gold, source_ids=[source_id], text_hash=text_digest(text),
                          normalized_hash=normalized, source_groups=[str(group)] if group is not None else [],
                          split=spec.get("split", "unspecified")))
    if not cases:
        raise ValueError("No usable examples: " + spec["path"])
    # Union formatting-equivalent texts AND supplied product/user groups for cluster bootstrap.
    parent = list(range(len(cases)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owners = {}
    for i, case in enumerate(cases):
        keys = [("text", case["normalized_hash"])]
        keys.extend(("group", g) for g in case["source_groups"])
        for key in keys:
            if key in owners:
                parent[find(i)] = find(owners[key])
            else:
                owners[key] = i
    for i, case in enumerate(cases):
        case["group_id"] = str(find(i))
        case["source_label_conflict"] = len(conflicts[case["text"]]) > 1
    info = dict(counts, eligible_rows=len(cases), source_sha256=file_digest(path),
                distribution=dict(Counter(c["gold"] for c in cases)), raw_labels=dict(raw_labels),
                conflicting_texts=sum(len(v) > 1 for v in conflicts.values()),
                conflict_rows=sum(c["source_label_conflict"] for c in cases),
                normalized_text_groups=len({c["normalized_hash"] for c in cases}),
                bootstrap_groups=len({c["group_id"] for c in cases}),
                deduplication="exact stripped text+label; conflicting labels retained",
                near_duplicate_policy="NFKC/Turkish-case/punctuation/whitespace grouping; no semantic fuzzy matching",
                split=spec.get("split", "unspecified"), task=spec["task"])
    return cases, info
