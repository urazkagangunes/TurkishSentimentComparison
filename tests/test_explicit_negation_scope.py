"""Offline scope regressions: frozen ID 148 and independent clause variants.

Run only this module; other test files may start long model experiments.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from src.dependency.sentiment_scope_resolver import SentimentScopeResolver
from src.hisnet.hisnet_dependency_classifier import HisNetDependencyClassifier
from src.hisnet.literature_weighted_hisnet_dependency_classifier import (
    LiteratureWeightedHisNetDependencyClassifier,
)


def token(index, text, head=0, relation="root", pos="ADJ", feats=None, sentence=0):
    return dict(id=index, text=text, lemma=text, head=head, deprel=relation,
                upos=pos, feats=feats, sentence_index=sentence)


def word(text, positive=1., negative=0.):
    return dict(word=text, root=text, positive=positive, negative=negative)


class ExplicitNegationScopeTests(unittest.TestCase):
    def setUp(self):
        self.resolver = SentimentScopeResolver()

    def resolve(self, tokens, words):
        return self.resolver.resolve(words, tokens)

    def negation_ids(self, scopes):
        return [[n["id"] for n in s["negations"]] for s in scopes]

    def test_id148_saved_parse_keeps_return_negative(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/id148_scope.json").read_text(encoding="utf-8"))
        for cls in (HisNetDependencyClassifier, LiteratureWeightedHisNetDependencyClassifier):
            with self.subTest(system=cls.__name__):
                classifier = cls.__new__(cls)
                classifier.scope_resolver = self.resolver
                classifier.parser = SimpleNamespace(parse=lambda text: deepcopy(fixture["parsed_sentences"]))
                result = classifier.classify(fixture["text"], hisnet_result=deepcopy(fixture["hisnet"]))
                self.assertEqual(result["prediction"], "negative")
                self.assertEqual((result["positive_score"], result["negative_score"]), (0., .125))
                self.assertEqual(self.negation_ids(result["scopes"]), [[]])
                self.assertNotIn("negation_flip", result["scopes"][0]["operations"])

    def test_following_light_verb_predicates_do_not_inherit_degil(self):
        # No lexical identity or gold label is used by the scope rule.
        for noun in ("iade", "tavsiye", "şikayet"):
            for relation in ("obj", "conj"):
                with self.subTest(noun=noun, relation=relation):
                    tokens = [token(1, "böyle", 2, "advmod"),
                              token(2, "değil", pos="AUX"),
                              token(3, noun, 2, relation, "NOUN"),
                              token(4, "edeceğim", 3, "compound:lvc", "VERB", "Polarity=Pos")]
                    scopes = self.resolve(tokens, [word(noun)])
                    self.assertEqual(self.negation_ids(scopes), [[]])

    def test_following_predicate_keeps_its_own_negation(self):
        # ID 447's structure: "böyle değildi ... tavsiye etmem".
        tokens = [token(1, "böyle", 2, "advmod"), token(2, "değil", pos="AUX"),
                  token(3, "tavsiye", 2, "conj", "NOUN"),
                  token(4, "etmem", 3, "compound:lvc", "VERB", "Polarity=Neg")]
        scopes = self.resolve(tokens, [word("tavsiye")])
        self.assertEqual(self.negation_ids(scopes), [[4]])
        scored = HisNetDependencyClassifier._apply_dependency_scope(scopes[0])
        self.assertEqual((scored["adjusted_positive"], scored["adjusted_negative"]), (0., 1.))

    def test_light_verb_with_missing_polarity_still_marks_separate_predicate(self):
        tokens = [token(1, "değil", pos="AUX"), token(2, "iade", 1, "obj", "NOUN"),
                  token(3, "edeceğim", 2, "compound:lvc", "VERB")]
        self.assertEqual(self.negation_ids(self.resolve(tokens, [word("iade")])), [[]])

    def test_preceding_positive_candidate_and_following_return_are_separate(self):
        tokens = [token(1, "iyi", 2, "amod"), token(2, "değil", pos="AUX"),
                  token(3, "iade", 2, "obj", "NOUN"),
                  token(4, "edeceğim", 3, "compound:lvc", "VERB", "Polarity=Pos")]
        scopes = self.resolve(tokens, [word("iyi"), word("iade", 0., .125)])
        self.assertEqual(self.negation_ids(scopes), [[2], []])

    def test_preceding_light_verb_predicate_is_still_negated(self):
        # "tavsiye edecek değilim": the entire predicate precedes değil.
        tokens = [token(1, "tavsiye", 3, "obj", "NOUN"),
                  token(2, "edecek", 1, "compound:lvc", "VERB", "Polarity=Pos"),
                  token(3, "değil", pos="AUX")]
        self.assertEqual(self.negation_ids(self.resolve(tokens, [word("tavsiye")])), [[3]])

    def test_inverted_copular_negation_is_preserved(self):
        # Synthetic inverted parses: the rule must not reject all later words.
        for parent_negation in (True, False):
            with self.subTest(parent_negation=parent_negation):
                tokens = ([token(1, "değil", pos="AUX"), token(2, "kötü", 1, "amod")]
                          if parent_negation else
                          [token(1, "değil", 2, "cop", "AUX"), token(2, "kötü")])
                self.assertEqual(self.negation_ids(self.resolve(tokens, [word("kötü", 0., 1.)])), [[1]])

    def test_relative_clause_is_not_a_light_verb_predicate(self):
        tokens = [token(1, "değil", pos="AUX"), token(2, "iade", 1, "obj", "NOUN"),
                  token(3, "istediğim", 2, "acl", "VERB", "Polarity=Pos")]
        self.assertEqual(self.negation_ids(self.resolve(tokens, [word("iade")])), [[1]])

    def test_child_degil_still_flips_positive_and_negative_words(self):
        for item in (word("güzel"), word("kötü", 0., 1.)):
            with self.subTest(word=item["word"]):
                tokens = [token(1, item["word"]), token(2, "değil", 1, "cop", "AUX")]
                scopes = self.resolve(tokens, [item])
                self.assertEqual(self.negation_ids(scopes), [[2]])
                scored = HisNetDependencyClassifier._apply_dependency_scope(scopes[0])
                self.assertEqual(scored["adjusted_positive"], item["negative"])
                self.assertEqual(scored["adjusted_negative"], item["positive"])

    def test_light_verb_negation_and_self_negation_policy_unchanged(self):
        tokens = [token(1, "pişman", pos="NOUN"),
                  token(2, "olmadım", 1, "compound:lvc", "VERB", "Polarity=Neg")]
        self.assertEqual(self.negation_ids(self.resolve(tokens, [word("pişman", 0., 1.)])), [[2]])
        tokens = [token(1, "sevmedim", pos="VERB", feats="Polarity=Neg")]
        self.assertEqual(self.negation_ids(self.resolve(tokens, [word("sevmedim")])), [[]])

    def test_coordination_shares_negation_but_contrast_blocks_it(self):
        for conjunction, expected in (("ve", [[4], [4]]), ("ama", [[], [4]])):
            with self.subTest(conjunction=conjunction):
                tokens = [token(1, "iyi"), token(2, conjunction, 3, "cc", "CCONJ"),
                          token(3, "güzel", 1, "conj"), token(4, "değil", 3, "cop", "AUX")]
                self.assertEqual(self.negation_ids(self.resolve(tokens, [word("iyi"), word("güzel")])), expected)

    def test_no_cross_sentence_negation(self):
        tokens = [token(1, "kötü"), token(2, "merkezi", sentence=1),
                  token(3, "değil", 2, "cop", "AUX", sentence=1)]
        self.assertEqual(self.negation_ids(self.resolve(tokens, [word("kötü", 0., 1.)])), [[]])


if __name__ == "__main__":
    unittest.main()
