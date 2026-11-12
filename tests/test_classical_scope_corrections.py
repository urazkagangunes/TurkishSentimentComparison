"""Scope/occurrence regressions prepared for the balanced development experiment."""
from copy import deepcopy
import unittest

from src.dependency.sentiment_scope_resolver import SentimentScopeResolver
from tests.test_explicit_negation_scope import token, word


class ClassicalScopeCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.resolver = SentimentScopeResolver()

    def test_unscored_independent_predicate_keeps_its_negation(self):
        # Otel kötü, merkezi değil: cop belongs to merkezi, not kötü.
        tokens = [token(1, "otel", 2, "nsubj", "NOUN"), token(2, "kötü"),
                  token(3, "merkezi", 2, "parataxis"), token(4, "değil", 3, "cop", "AUX")]
        scopes = self.resolver.resolve([word("kötü", 0., 1.)], tokens)
        self.assertEqual(scopes[0]["negations"], [])

    def test_unscored_owner_across_contrast_does_not_negate_previous_opinion(self):
        tokens = [token(1, "güzel"), token(2, "ama", 3, "cc", "CCONJ"),
                  token(3, "merkezi", 1, "conj"), token(4, "değil", 3, "cop", "AUX")]
        self.assertEqual(self.resolver.resolve([word("güzel")], tokens)[0]["negations"], [])

    def test_copular_predicate_does_not_negate_its_sentiment_bearing_subject(self):
        # "Tek sıkıntısı kalıcı değil": kalıcı is negated, sıkıntı is not.
        tokens = [token(1, "sıkıntısı", 2, "nsubj", "NOUN"), token(2, "kalıcı"),
                  token(3, "değil", 2, "cop", "AUX")]
        words = [dict(word="sıkıntısı", root="sıkıntı", positive=0., negative=1.)]
        self.assertEqual(self.resolver.resolve(words, tokens)[0]["negations"], [])

    def test_forward_copula_does_not_negate_a_later_clause(self):
        tokens = [token(1, "göründüğü", 3, "acl", "VERB"), token(2, "gibi", 1, "case", "ADP"),
                  token(3, "değil", 5, "cop", "AUX"), token(4, "amblemi", 5, "nsubj", "NOUN"),
                  token(5, "yok", pos="ADV")]
        self.assertEqual(self.resolver.resolve([word("yok", 0., 1.)], tokens)[0]["negations"], [])

    def test_forward_copula_preserves_preceding_dependency_candidate_scope(self):
        tokens = [token(1, "göründüğü", 3, "acl", "VERB"), token(2, "gibi", 1, "case", "ADP"),
                  token(3, "değil", 5, "cop", "AUX"), token(4, "amblemi", 5, "nsubj", "NOUN"),
                  token(5, "yok", pos="ADV")]
        candidates = {c["word"]: c for c in self.resolver.resolve_dependency_candidates(tokens)}
        self.assertEqual([n["id"] for n in candidates["göründüğü"]["negations"]], [3])
        self.assertEqual(candidates["yok"]["negations"], [])

    def test_coordination_can_include_an_unscored_owner(self):
        tokens = [token(1, "iyi"), token(2, "ve", 3, "cc", "CCONJ"),
                  token(3, "merkezi", 1, "conj"), token(4, "değil", 3, "cop", "AUX")]
        scopes = self.resolver.resolve([word("iyi")], tokens)
        self.assertEqual([n["id"] for n in scopes[0]["negations"]], [4])

    def test_descriptive_dependents_remain_inside_nominal_predicate_scope(self):
        for surface, relation, pos in (("güzel", "amod", "ADJ"), ("tavsiye", "acl", "NOUN")):
            with self.subTest(relation=relation):
                tokens = [token(1, surface, 3, relation, pos), token(2, "bir", 3, "det", "DET"),
                          token(3, "ürün", pos="NOUN"), token(4, "değil", 3, "cop", "AUX")]
                scopes = self.resolver.resolve([word(surface)], tokens)
                self.assertEqual([n["id"] for n in scopes[0]["negations"]], [4])

    def test_sentiment_subset_preserves_full_morphological_occurrence(self):
        morphology = [dict(word="yazar", root="yazar", analysis="yazar+NOUN+A3SG"),
                      dict(word="yazar", root="yaz", analysis="yaz+VERB+POS+AOR+A3SG")]
        tokens = [token(1, "yazar", pos="NOUN"), token(2, "yazar", pos="VERB", sentence=1)]
        scopes = self.resolver.resolve([dict(word="yazar", root="yaz", positive=1., negative=0.)], tokens, morphology)
        self.assertEqual(scopes[0]["sentiment_token"]["id"], 2)

    def test_subset_alignment_does_not_borrow_negation_from_another_sentence(self):
        morphology = [dict(word="yazar", root="yazar", analysis="yazar+NOUN+A3SG"),
                      dict(word="değil", root="değil", analysis="değil+ADJ"),
                      dict(word="yazar", root="yaz", analysis="yaz+VERB+POS+AOR+A3SG")]
        tokens = [token(1, "yazar", pos="NOUN"), token(2, "değil", 1, "cop", "AUX"),
                  token(3, "yazar", pos="VERB", sentence=1)]
        scopes = self.resolver.resolve([dict(word="yazar", root="yaz", positive=1., negative=0.)], tokens, morphology)
        self.assertEqual(scopes[0]["sentiment_token"]["id"], 3)
        self.assertEqual(scopes[0]["negations"], [])

    def test_decorated_and_plain_occurrences_get_distinct_roots(self):
        morphology = [dict(word="yazar", root="yazar", analysis="yazar+NOUN+A3SG"),
                      dict(word="yazar", root="yaz", analysis="yaz+VERB+POS+AOR+A3SG")]
        for surfaces in (("yazar🙂", "yazar"), ("yazar", "yazar🙂"), ("yazar🙂", "yazar🙂")):
            with self.subTest(surfaces=surfaces):
                tokens = [token(1, surfaces[0], pos="NOUN"), token(2, surfaces[1], pos="VERB", sentence=1)]
                original = deepcopy(tokens), deepcopy(morphology)
                candidates = self.resolver.resolve_dependency_candidates(tokens, morphology)
                self.assertEqual([c["root"] for c in candidates], ["yazar", "yaz"])
                self.assertEqual((tokens, morphology), original)

    def test_decorated_first_occurrence_cannot_steal_plain_later_token(self):
        morphology = [dict(word="yazar", root="yazar", analysis="yazar+NOUN+A3SG"),
                      dict(word="yazar", root="yaz", analysis="yaz+VERB+POS+AOR+A3SG")]
        tokens = [token(1, "yazar🙂", pos="NOUN"), token(2, "yazar", pos="VERB", sentence=1)]
        words = [dict(word="yazar", root=record["root"], positive=1., negative=0.) for record in morphology]
        scopes = self.resolver.resolve(words, tokens, morphology)
        self.assertEqual([s["sentiment_token"]["id"] for s in scopes], [1, 2])

    def test_repaired_fragments_consume_their_morphology_occurrence(self):
        morphology = [dict(word="yazar", root="yazar", analysis="yazar+NOUN+A3SG"),
                      dict(word="yazar", root="yaz", analysis="yaz+VERB+POS+AOR+A3SG")]
        tokens = [token(1, "yaz", pos="NOUN"), token(2, "ar", 1, "case", "ADP"),
                  token(3, "yazar", pos="VERB", sentence=1)]
        candidates = self.resolver.resolve_dependency_candidates(tokens, morphology)
        self.assertEqual([c["root"] for c in candidates], ["yazar", "yaz"])


if __name__ == "__main__":
    unittest.main()
