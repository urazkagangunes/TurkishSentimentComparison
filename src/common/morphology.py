import re

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
        return (
            text
            .replace("I", "ı")
            .replace("İ", "i")
            .lower()
        )

    def _prepare_sentence(self, text):
        """
        Prepares text for morphological analysis.

        Keeps Unicode letters and digits.
        Internal apostrophes are also preserved.

        Examples:
            Hollywood -> hollywood
            Woo -> woo
            80ler -> 80ler
            Türkiye'de -> türkiye'de
        """

        text = self._turkish_lower(text)

        tokens = re.findall(
            r"[^\W_]+(?:['’][^\W_]+)*",
            text,
            flags=re.UNICODE
        )

        return " ".join(tokens)

    def _normal_analysis(self, word):
        """
        Runs normal morphological analysis.

        Starlang's analyzer may occasionally enter an
        inconsistent internal state after many analyses.

        If a known internal-state exception occurs, the
        analyzer is recreated and the same token is retried.

        Returns:
            parses
            analyzer_was_reset
        """

        try:

            parses = (
                self.analyzer
                .morphologicalAnalysis(word)
            )

            return parses, False

        except (ValueError, IndexError):

            # Recreate the analyzer so that a corrupted
            # internal dictionary/trie state cannot affect
            # the rest of the dataset.
            self.analyzer = (
                FsmMorphologicalAnalyzer()
            )

            try:

                parses = (
                    self.analyzer
                    .morphologicalAnalysis(word)
                )

                return parses, True

            except (ValueError, IndexError):

                # Normal analysis failed even with a fresh
                # analyzer. The caller may try robust
                # analysis using an isolated analyzer.
                return None, True

    @staticmethod
    def _robust_analysis(word):
        """
        Runs robust morphological analysis using a completely
        isolated analyzer instance.

        This prevents OOV analysis from modifying the state of
        the main analyzer used for subsequent tokens.
        """

        temporary_analyzer = (
            FsmMorphologicalAnalyzer()
        )

        try:

            parses = (
                temporary_analyzer
                .robustMorphologicalAnalysis(word)
            )

            if parses is None:
                return None

            if parses.size() == 0:
                return None

            return parses

        except Exception:
            return None

    def analyze_sentence(self, text):
        """
        Morphologically analyzes and disambiguates a sentence.

        Strategy:
            1. Normal morphological analysis is preferred.
            2. If the normal analyzer encounters a known
               internal-state error, it is recreated and the
               same token is retried.
            3. OOV/unanalyzable tokens are passed to robust
               analysis using an isolated analyzer instance.
            4. If even robust analysis fails, the original
               surface form is preserved as a fallback.
            5. Successfully analyzed tokens are jointly passed
               to LongestRootFirstDisambiguation.

        No sentiment information is used during morphology.
        """

        cleaned_text = (
            self._prepare_sentence(text)
        )

        if not cleaned_text:
            return []

        surface_words = (
            cleaned_text.split()
        )

        analyzable_items = []
        parse_lists = []

        for index, word in enumerate(
            surface_words
        ):

            # -------------------------------------------------
            # 1. Normal morphological analysis
            # -------------------------------------------------

            parses, analyzer_was_reset = (
                self._normal_analysis(word)
            )

            source = "normal"

            # -------------------------------------------------
            # 2. If normal analysis failed or returned no
            #    analyses, try isolated robust analysis.
            # -------------------------------------------------

            if (
                parses is None
                or parses.size() == 0
            ):

                parses = (
                    self._robust_analysis(word)
                )

                source = "robust"

            # -------------------------------------------------
            # 3. Final fallback
            # -------------------------------------------------

            if (
                parses is None
                or parses.size() == 0
            ):

                analyzable_items.append(
                    {
                        "index": index,
                        "word": word,
                        "source": "fallback",
                        "parse_index": None,
                        "analyzer_reset":
                            analyzer_was_reset
                    }
                )

                continue

            # -------------------------------------------------
            # Successfully analyzed token
            # -------------------------------------------------

            parse_index = len(
                parse_lists
            )

            parse_lists.append(
                parses
            )

            analyzable_items.append(
                {
                    "index": index,
                    "word": word,
                    "source": source,
                    "parse_index": parse_index,
                    "analyzer_reset":
                        analyzer_was_reset
                }
            )

        # -----------------------------------------------------
        # Disambiguate all successfully analyzed tokens
        # -----------------------------------------------------

        if parse_lists:

            selected_parses = (
                self.disambiguator
                .disambiguate(
                    parse_lists
                )
            )

        else:

            selected_parses = []

        # -----------------------------------------------------
        # Restore original token order
        # -----------------------------------------------------

        results = []

        for item in analyzable_items:

            word = item["word"]

            parse_index = (
                item["parse_index"]
            )

            # -------------------------------------------------
            # Fallback token
            # -------------------------------------------------

            if parse_index is None:

                results.append(
                    {
                        "word": word,
                        "root": word,
                        "analysis": "UNKNOWN",
                        "analysis_source":
                            "fallback",
                        "analyzer_reset":
                            item["analyzer_reset"]
                    }
                )

                continue

            # -------------------------------------------------
            # Successfully analyzed token
            # -------------------------------------------------

            parse = (
                selected_parses[
                    parse_index
                ]
            )

            analysis = (
                parse.transitionList()
            )

            root = (
                analysis.split(
                    "+",
                    1
                )[0]
            )

            results.append(
                {
                    "word": word,
                    "root": root,
                    "analysis": analysis,
                    "analysis_source":
                        item["source"],
                    "analyzer_reset":
                        item["analyzer_reset"]
                }
            )

        return results