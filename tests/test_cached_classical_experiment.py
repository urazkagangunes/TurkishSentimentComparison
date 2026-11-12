"""Offline checks for immutable preprocessing reuse and imported-run identity."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from scripts.run_balanced_classical_cached import classify_shared, load_import, semantic_result
from src.evaluation.checkpoint_store import CheckpointStore
from src.hisnet.hisnet_classifier import HisNetClassifier


class CachedClassicalExperimentTests(unittest.TestCase):
    def test_persisted_diagnostic_keys_are_compared_in_json_form(self):
        live = dict(prediction="positive", positive_score=1., contrast_relations={0: []},
                    licensed_contrast_markers={0: []})
        saved = dict(prediction="positive", positive_score=1., contrast_relations={"0": []},
                     licensed_contrast_markers={"0": []}, status="success", predictions=["positive"])
        self.assertEqual(semantic_result(live), semantic_result(saved))
        saved["positive_score"] = .5
        self.assertNotEqual(semantic_result(live), semantic_result(saved))

    def test_shared_inputs_preserve_method_decisions_and_are_not_mutated(self):
        morphology = [dict(word="güzel", root="güzel", analysis="güzel+ADJ"),
                      dict(word="değil", root="değil", analysis="değil+ADJ")]
        parser_output = [dict(text="Güzel değil.", tokens=[
            dict(id=1, text="Güzel", lemma="güzel", head=0, deprel="root", upos="ADJ", feats=None),
            dict(id=2, text="değil", lemma="değil", head=1, deprel="cop", upos="AUX", feats="Polarity=Neg"),
        ])]
        lexical = {"güzel": dict(positive=1., negative=0.)}
        hisnet = HisNetClassifier.__new__(HisNetClassifier)
        hisnet.root_extractor = SimpleNamespace(analyze_sentence=lambda text: deepcopy(morphology))
        hisnet._lookup_root = lambda root: deepcopy(lexical.get(root))
        baseline = hisnet.classify("Güzel değil.")
        self.assertEqual(baseline["prediction"], "positive")
        frozen = deepcopy(baseline), deepcopy(parser_output), deepcopy(lexical)
        for system in ("dp_hisnet", "hisnet_dp", "lw_hisnet_dp"):
            with self.subTest(system=system):
                calls = []

                def lookup(root):
                    calls.append(root)
                    return lexical.get(root)

                result = classify_shared(system, "Güzel değil.", baseline, parser_output, lookup)
                self.assertEqual(result["prediction"], "negative")
                self.assertEqual((result["positive_score"], result["negative_score"]), (0., 1.))
                if system == "dp_hisnet":
                    self.assertEqual(calls, ["güzel"])
                else:
                    self.assertEqual(calls, [])
                self.assertEqual((baseline, parser_output, lexical), frozen)
                result["scopes"][0]["negations"].clear()
                self.assertEqual((baseline, parser_output, lexical), frozen)

    def test_import_rejects_wrong_sample_or_implementation(self):
        cases = [dict(text="güzel", gold="positive")]
        with TemporaryDirectory() as temporary:
            source = Path(temporary)
            with CheckpointStore(source, dict(implementation="frozen"), cases) as store:
                attempt = store.start("hisnet", [0], False)
                store.finish(attempt, dict(status="success", predictions=["positive"],
                                          prediction="positive", error=None), .25)
            before = (source / "progress.sqlite3").read_bytes()
            rows, metadata = load_import(source, cases, "frozen")
            self.assertEqual(set(rows), {0})
            self.assertEqual(rows[0][1], .25)
            self.assertEqual(metadata["imported_hisnet_rows"], 1)
            with self.assertRaisesRegex(ValueError, "Implementation differs"):
                load_import(source, cases, "changed")
            with self.assertRaisesRegex(ValueError, "sample differs"):
                load_import(source, [dict(text="kötü", gold="negative")], "frozen")
            self.assertEqual((source / "progress.sqlite3").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
