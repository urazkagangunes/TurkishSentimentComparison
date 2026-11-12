"""Finite predicates must not become entity-only targets after a bad parse."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from src.dependency.sentiment_scope_resolver import SentimentScopeResolver
from scripts.validate_classical_corrections import replay
from tests.test_explicit_negation_scope import token


class FinitePredicateProtectionTests(unittest.TestCase):
    def setUp(self):
        self.resolver = SentimentScopeResolver()

    def test_finite_analysis_requires_final_verb_and_finite_inflection(self):
        for analysis, expected in [
            ("pişman+ADJ^DB+VERB+ZERO+PRES+A1SG", True),
            ("memnun+ADJ^DB+VERB+ZERO+PAST+A2PL", True),
            ("üzül+VERB+POS+PAST+A1SG", False),
            ("sivilce+NOUN+A3PL+PNON+NOM^DB+VERB+ZERO+PRES+A1SG", False),
            ("pişman+ADJ", False), ("sev+VERB+POS^DB+NOUN+INF2+A3SG+PNON+NOM", False),
            ("sev+VERB+POS+PASTPART+A3SG", False), ("pişman+NOUN+A3SG+P1SG+NOM", False),
            ("UNKNOWN", False), ("sev+VERB+POS", False),
        ]:
            with self.subTest(analysis=analysis):
                self.assertEqual(self.resolver._is_finite_morphological_predicate(dict(analysis=analysis)), expected)
        self.assertFalse(self.resolver._is_finite_morphological_predicate(dict(
            analysis="pişman+ADJ^DB+VERB+ZERO+PRES+A1SG", analysis_source="robust")))

    def test_finite_target_is_protected_but_nominal_target_is_suppressed(self):
        for analysis, protected in (("pişman+ADJ^DB+VERB+ZERO+PRES+A1SG", True),
                                    ("pişman+NOUN+A3SG+P1SG+NOM", False)):
            scopes = [dict(word="akıyor", root="ak", sentiment_token={"id":1}, target={"id":2}),
                      dict(word="pişmanım", root="pişman", sentiment_token={"id":2}, target=None)]
            morphology = [dict(word="pişmanım", root="pişman", analysis=analysis)]
            original = deepcopy(morphology)
            self.resolver._mark_finite_predicate_targets(scopes, morphology)
            result = self.resolver._resolve_sentiment_roles(scopes)
            self.assertEqual(result[1]["active_sentiment"], protected)
            self.assertEqual(morphology, original)

    def test_same_root_different_surface_cannot_license_protection(self):
        scopes = [dict(sentiment_token={"id":1}, target={"id":2}),
                  dict(word="pişman", root="pişman", sentiment_token={"id":2}, target=None)]
        morphology = [dict(word="pişmanım", root="pişman", analysis="pişman+ADJ^DB+VERB+ZERO+PRES+A1SG")]
        self.resolver._mark_finite_predicate_targets(scopes, morphology)
        self.assertFalse(self.resolver._resolve_sentiment_roles(scopes)[1]["active_sentiment"])

    def test_ambiguous_repeated_surface_is_not_overridden(self):
        scopes = [dict(sentiment_token={"id":1}, target={"id":2}),
                  dict(word="test", root="test", sentiment_token={"id":2}, target=None)]
        morphology = [dict(word="test", root="test", analysis=a) for a in
                      ("test+ADJ^DB+VERB+ZERO+PAST+A1SG", "test+NOUN+A3SG")]
        self.resolver._mark_finite_predicate_targets(scopes, morphology)
        self.assertFalse(self.resolver._resolve_sentiment_roles(scopes)[1]["active_sentiment"])

    def test_dependency_candidate_survives_incorrect_subject_relation(self):
        for surface, root in (("pişmanım", "pişman"), ("memnunum", "memnun")):
            tokens = [token(1, "akıyor", pos="VERB"), token(2, surface, 1, "nsubj", "NOUN")]
            morphology = [dict(word=surface, root=root, analysis=root+"+ADJ^DB+VERB+ZERO+PRES+A1SG")]
            candidates = self.resolver.resolve_dependency_candidates(tokens, morphology)
            self.assertIn(surface, [c["word"] for c in candidates])

    def test_saved_id333_recovers_all_dependency_conditions(self):
        source = Path(__file__).resolve().parent / "fixtures/finite_predicate_id333.json"
        case = json.loads(source.read_text(encoding="utf-8"))
        for system in ("dp_hisnet", "hisnet_dp", "lw_hisnet_dp"):
            with self.subTest(system=system):
                result = replay(system, case["systems"][system], case["systems"]["hisnet"], SentimentScopeResolver)
                regret = next(s for s in result["scopes"] if s["root"] == "pişman")
                self.assertTrue(regret["active_sentiment"])
                self.assertNotIn("suppressed_as_target", regret["operations"])
                self.assertEqual(regret["adjusted_negative"], 1.)
                self.assertEqual(result["prediction"], "negative")

    def test_nominal_possessive_and_ordinary_verbs_keep_dependency_filter(self):
        for surface, root, analysis in (
            ("sivilcelerim", "sivilce", "sivilce+NOUN+A3PL+PNON+NOM^DB+VERB+ZERO+PRES+A1SG"),
            ("hissettim", "hisset", "hisset+VERB+POS+PAST+A1SG"),
        ):
            tokens = [token(1, "var", pos="VERB"), token(2, surface, 1, "nsubj", "NOUN")]
            candidates = self.resolver.resolve_dependency_candidates(tokens, [
                dict(word=surface, root=root, analysis=analysis)])
            self.assertNotIn(surface, [c["word"] for c in candidates])


if __name__ == "__main__":
    unittest.main()
