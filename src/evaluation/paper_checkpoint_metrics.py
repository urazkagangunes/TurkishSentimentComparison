"""Aggregate paper metrics on the same successful rows for all seven systems."""

import sqlite3
from contextlib import closing
from pathlib import Path

from src.evaluation.checkpoint_store import LABELS, SYSTEMS, score_confusion, summarize_connection


def read_paper_dataset(output):
    database = Path(output).resolve() / "progress.sqlite3"
    if not database.is_file():
        return {"status": "not_started"}

    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("BEGIN")
        gold_counts = dict(connection.execute("SELECT gold, COUNT(*) FROM cases GROUP BY gold"))
        total = sum(gold_counts.values())
        macro_labels = [label for label in LABELS if label in gold_counts]
        progress = summarize_connection(connection)
        complete = all(item["complete"] for item in progress.values())
        matrices = {system: [[0] * len(LABELS) for _ in LABELS] for system in SYSTEMS}

        if complete:
            matrices = {system: item["confusion_matrix_on_successful"] for system, item in progress.items()}
        elif all(item["success"] for item in progress.values()):
            # The predictions primary key guarantees at most one prediction per system and row.
            placeholders = ",".join("?" for _ in SYSTEMS)
            for system, gold, prediction, count in connection.execute(f"""
                WITH common AS (
                    SELECT row_id FROM predictions
                    WHERE status='success' AND system IN ({placeholders})
                    GROUP BY row_id HAVING COUNT(*) = ?
                )
                SELECT p.system,c.gold,p.prediction,COUNT(*)
                FROM common JOIN cases c USING(row_id) JOIN predictions p USING(row_id)
                WHERE p.system IN ({placeholders})
                GROUP BY p.system,c.gold,p.prediction
                """, (*SYSTEMS, len(SYSTEMS), *SYSTEMS)):
                matrices[system][LABELS.index(gold)][LABELS.index(prediction)] = count

    common = sum(map(sum, matrices[SYSTEMS[0]]))
    systems = {}
    for system, item in progress.items():
        matrix = matrices[system]
        accuracy, macro_f1 = score_confusion(matrix, macro_labels)
        systems[system] = dict(
            success=item["success"], failed=item["failed"], pending=item["pending"],
            coverage=item["coverage"], usage=item["usage"],
            accuracy_on_common=accuracy, macro_f1_on_common=macro_f1, confusion_matrix_on_common=matrix,
        )
    return dict(status="complete" if complete else "incomplete", total=total, gold_counts=gold_counts,
                macro_f1_labels=macro_labels, label_order=LABELS, common_success=common,
                common_coverage=common / total if total else 0.0, systems=systems)
