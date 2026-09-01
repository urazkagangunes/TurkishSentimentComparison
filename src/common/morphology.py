import re

from Corpus.Sentence import Sentence
from MorphologicalAnalysis.FsmMorphologicalAnalyzer import (
    FsmMorphologicalAnalyzer
)
from MorphologicalDisambiguation.LongestRootFirstDisambiguation import (
    LongestRootFirstDisambiguation
)


class MorphologicalRootExtractor:

    def __init__(self):
        self.analyzer = FsmMorphologicalAnalyzer()
        self.disambiguator = LongestRootFirstDisambiguation()

    @staticmethod
    def _turkish_lower(text):
        """
        Turkish-aware lowercasing.
        """
        return text.replace("I", "ı").replace("İ", "i").lower()

    def _prepare_sentence(self, text):
        """
        Removes punctuation before morphological analysis.

        This is needed because forms such as 'beğenmedim.'
        may otherwise be treated as a single unknown token.
        """

        text = self._turkish_lower(text)

        words = re.findall(
            r"[abcçdefgğhıijklmnoöprsştuüvyz]+",
            text
        )

        return " ".join(words)

    def analyze_sentence(self, text):
        """
        Morphologically analyzes and disambiguates a complete sentence.

        Returns one selected morphological parse/root per token.
        """

        cleaned_text = self._prepare_sentence(text)

        if not cleaned_text:
            return []

        sentence = Sentence(cleaned_text)

        # Generate all possible morphological analyses.
        all_parses = self.analyzer.robustMorphologicalAnalysis(sentence)

        # Choose one parse for every token.
        selected_parses = self.disambiguator.disambiguate(all_parses)

        results = []

        for parse in selected_parses:

            analysis = parse.transitionList()

            # Example:
            # güzel+ADJ^DB+VERB+ZERO+PAST+A3SG
            #
            # Root is the part before the first "+"
            root = analysis.split("+", 1)[0]

            # The selected parse already contains the surface form,
            # but keeping the analysis string is useful for debugging
            # and later error analysis.
            results.append({
                "root": root,
                "analysis": analysis
            })

        # Add surface tokens in the same order.
        surface_words = cleaned_text.split()

        for i, item in enumerate(results):
            if i < len(surface_words):
                item["word"] = surface_words[i]

        return results