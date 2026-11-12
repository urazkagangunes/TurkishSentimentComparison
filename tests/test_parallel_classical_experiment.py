"""Offline transfer and worker boundary checks; no model initialization."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import run_parallel_classical_cached as runner
from src.evaluation.checkpoint_store import CheckpointStore


class ParallelClassicalExperimentTests(unittest.TestCase):
    def test_worker_keeps_decisions_separate_and_cached_inputs_detached(self):
        baseline = dict(prediction="positive", morphological_analyses=[])
        parsed = [{"tokens": []}]
        original = deepcopy((baseline, parsed))
        calls = []
        def classify(system, text, h, dp, lookup):
            calls.append((system, text))
            return dict(prediction="negative" if system=="dp_hisnet" else "positive")
        forbidden = lambda *_: self.fail("Cached input must not be recomputed")
        with patch.object(runner, "_HISNET", SimpleNamespace(classify=forbidden, _lookup_root=lambda _:None)), \
             patch.object(runner, "_PARSER", SimpleNamespace(parse=forbidden)), \
             patch.object(runner, "classify_shared", classify):
            batch = runner.process_review(7, "text", baseline, parsed)
        self.assertEqual([s for s,_ in calls], list(runner.SYSTEMS[1:]))
        self.assertEqual(batch["results"]["hisnet"][0]["prediction"], "positive")
        self.assertEqual(batch["results"]["dp_hisnet"][0]["prediction"], "negative")
        batch["parsed"][0]["tokens"].append("changed")
        batch["results"]["hisnet"][0]["morphological_analyses"].append("changed")
        self.assertEqual((baseline, parsed), original)

    def test_import_checks_identity_preserves_source_and_does_not_duplicate_successes(self):
        cases = [dict(text="text", gold="positive")]
        with TemporaryDirectory() as temp:
            source, output = Path(temp)/"source", Path(temp)/"output"
            with CheckpointStore(source, dict(implementation="code", resources=[]), cases) as old:
                old.connection.execute("CREATE TABLE parses (row_id INTEGER PRIMARY KEY,payload TEXT,seconds REAL)")
                old.connection.execute("INSERT INTO parses VALUES (0,'[]',0.25)")
                old.connection.commit()
                for s in runner.SYSTEMS:
                    attempt = old.start(s, [0], False)
                    old.finish(attempt, dict(predictions=["positive"], prediction="positive", status="success", error=None), .5)
            frozen = (source/"progress.sqlite3").read_bytes()
            with CheckpointStore(output, {}, cases) as new:
                new.connection.execute("CREATE TABLE parses (row_id INTEGER PRIMARY KEY,payload TEXT,seconds REAL)")
                new.connection.commit()
                for _ in range(2):
                    imported = runner.import_frozen(source, new, "code", [], cases)
                    self.assertEqual(imported["successful_predictions"], 4)
                self.assertEqual(new.connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0], 4)
                self.assertEqual(new.connection.execute("SELECT SUM(seconds) FROM attempts").fetchone()[0], 2.)
                self.assertEqual(new.connection.execute("SELECT seconds FROM parses").fetchone()[0], .25)
                for code, resources, rows in (("other", [], cases), ("code", ["other"], cases), ("code", [], [])):
                    with self.assertRaisesRegex(ValueError, "differ"):
                        runner.import_frozen(source, new, code, resources, rows)
            self.assertEqual((source/"progress.sqlite3").read_bytes(), frozen)


if __name__ == "__main__":
    unittest.main()
