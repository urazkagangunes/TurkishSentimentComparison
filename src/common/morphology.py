import copy
import re

from MorphologicalAnalysis.FsmMorphologicalAnalyzer import (
    FsmMorphologicalAnalyzer
)

from MorphologicalDisambiguation.LongestRootFirstDisambiguation import (
    LongestRootFirstDisambiguation
)


_NOT_CACHED = object()


class MorphologicalRootExtractor:

    def __init__(
        self,
        use_robust_cache=True
    ):

        # Main analyzer.
        #
        # Normal morphological analysis uses this analyzer.
        # Robust analysis NEVER runs on this instance.
        self.analyzer = (
            FsmMorphologicalAnalyzer()
        )

        self.disambiguator = (
            LongestRootFirstDisambiguation()
        )

        # -----------------------------------------------------
        # ROBUST OOV CACHE
        # -----------------------------------------------------
        #
        # We do NOT keep one persistent robust analyzer.
        #
        # Instead:
        #
        # first occurrence of unique OOV
        #     -> fresh isolated analyzer
        #     -> robust parse candidates
        #     -> cache pristine copy
        #
        # repeated occurrence
        #     -> deepcopy cached parse candidates
        #
        # Final disambiguated root is NOT cached.
        # -----------------------------------------------------

        self.use_robust_cache = (
            use_robust_cache
        )

        self._robust_parse_cache = {}

        # -----------------------------------------------------
        # PERFORMANCE DIAGNOSTICS
        # -----------------------------------------------------

        self._normal_requests = 0
        self._normal_resets = 0

        self._robust_requests = 0
        self._robust_cache_hits = 0
        self._robust_cache_misses = 0
        self._robust_analyzer_builds = 0
        self._robust_failures = 0

    # =========================================================
    # PREPROCESSING
    # =========================================================

    @staticmethod
    def _turkish_lower(
        text
    ):
        """
        Turkish-aware lowercasing.
        """

        return (
            text
            .replace("I", "ı")
            .replace("İ", "i")
            .lower()
        )

    def _prepare_sentence(
        self,
        text
    ):
        """
        Prepares text for morphological analysis.

        Keeps Unicode letters and digits.
        Internal apostrophes are preserved.

        Examples:
            Hollywood -> hollywood
            Woo -> woo
            80ler -> 80ler
            Türkiye'de -> türkiye'de
        """

        text = (
            self._turkish_lower(
                text
            )
        )

        tokens = re.findall(
            r"[^\W_]+(?:['’][^\W_]+)*",
            text,
            flags=re.UNICODE
        )

        return " ".join(
            tokens
        )

    # =========================================================
    # NORMAL ANALYSIS
    # =========================================================

    def _normal_analysis(
        self,
        word
    ):
        """
        Runs normal morphological analysis.

        If Starlang's analyzer enters an inconsistent internal
        state, the analyzer is rebuilt and the same token is
        retried.

        This preserves the crash-recovery behaviour used in the
        previous version.
        """

        self._normal_requests += 1

        try:

            parses = (
                self.analyzer
                .morphologicalAnalysis(
                    word
                )
            )

            return (
                parses,
                False
            )

        except (
            ValueError,
            IndexError
        ):

            self._normal_resets += 1

            # Recreate corrupted normal analyzer.
            self.analyzer = (
                FsmMorphologicalAnalyzer()
            )

            try:

                parses = (
                    self.analyzer
                    .morphologicalAnalysis(
                        word
                    )
                )

                return (
                    parses,
                    True
                )

            except (
                ValueError,
                IndexError
            ):

                # Caller will try robust analysis.
                return (
                    None,
                    True
                )

    # =========================================================
    # ROBUST ANALYSIS
    # =========================================================

    def _robust_analysis(
        self,
        word
    ):
        """
        Runs robust morphological analysis.

        Important design:

            - A persistent robust analyzer is NOT used.
            - First occurrence of a unique OOV token gets a
              completely fresh isolated analyzer.
            - Successful parse candidates are cached.
            - Repeated OOV tokens reuse a deepcopy of those
              candidates.
            - Final disambiguated roots are never cached.

        This substantially reduces repeated expensive analyzer
        construction without allowing robust-analysis mutations
        to contaminate later words.
        """

        self._robust_requests += 1

        # -----------------------------------------------------
        # CACHE HIT
        # -----------------------------------------------------

        if self.use_robust_cache:

            cached = (
                self._robust_parse_cache.get(
                    word,
                    _NOT_CACHED
                )
            )

            if cached is not _NOT_CACHED:

                self._robust_cache_hits += 1

                # Return an independent copy because
                # LongestRootFirstDisambiguation can reduce
                # parse lists.
                return copy.deepcopy(
                    cached
                )

        # -----------------------------------------------------
        # CACHE MISS
        # -----------------------------------------------------

        self._robust_cache_misses += 1

        # Each unique OOV gets an isolated analyzer.
        self._robust_analyzer_builds += 1

        temporary_analyzer = (
            FsmMorphologicalAnalyzer()
        )

        try:

            parses = (
                temporary_analyzer
                .robustMorphologicalAnalysis(
                    word
                )
            )

        except Exception:

            self._robust_failures += 1

            # Failed analyses are intentionally NOT cached.
            # If the same token occurs later, it may be retried.
            return None

        if (
            parses is None
            or parses.size() == 0
        ):

            self._robust_failures += 1

            return None

        # -----------------------------------------------------
        # CACHE PRISTINE CANDIDATES
        # -----------------------------------------------------

        if self.use_robust_cache:

            self._robust_parse_cache[
                word
            ] = copy.deepcopy(
                parses
            )

        # For the first occurrence use the fresh result itself.
        # This mirrors the previous pipeline as closely as
        # possible.
        return parses

    # =========================================================
    # SENTENCE ANALYSIS
    # =========================================================

    def analyze_sentence(
        self,
        text
    ):
        """
        Morphologically analyzes and disambiguates a sentence.

        Pipeline:

            text
              -> Turkish-aware normalization
              -> normal morphological analysis
              -> isolated robust analysis for OOV
              -> LongestRootFirstDisambiguation
              -> selected roots

        Robust parse candidates may be cached.

        Final selected roots are NOT cached because
        disambiguation is sentence-context dependent.

        No sentiment information is used here.
        """

        cleaned_text = (
            self._prepare_sentence(
                text
            )
        )

        if not cleaned_text:
            return []

        surface_words = (
            cleaned_text.split()
        )

        analyzable_items = []
        parse_lists = []

        # =====================================================
        # TOKEN ANALYSIS
        # =====================================================

        for index, word in enumerate(
            surface_words
        ):

            # -------------------------------------------------
            # NORMAL ANALYSIS
            # -------------------------------------------------

            (
                parses,
                analyzer_was_reset
            ) = (
                self._normal_analysis(
                    word
                )
            )

            source = "normal"

            # -------------------------------------------------
            # ROBUST OOV ANALYSIS
            # -------------------------------------------------

            if (
                parses is None
                or parses.size() == 0
            ):

                parses = (
                    self._robust_analysis(
                        word
                    )
                )

                source = "robust"

            # -------------------------------------------------
            # FINAL FALLBACK
            # -------------------------------------------------

            if (
                parses is None
                or parses.size() == 0
            ):

                analyzable_items.append({
                    "index":
                        index,

                    "word":
                        word,

                    "source":
                        "fallback",

                    "parse_index":
                        None,

                    "analyzer_reset":
                        analyzer_was_reset
                })

                continue

            # -------------------------------------------------
            # SUCCESSFUL ANALYSIS
            # -------------------------------------------------

            parse_index = len(
                parse_lists
            )

            parse_lists.append(
                parses
            )

            analyzable_items.append({
                "index":
                    index,

                "word":
                    word,

                "source":
                    source,

                "parse_index":
                    parse_index,

                "analyzer_reset":
                    analyzer_was_reset
            })

        # =====================================================
        # DISAMBIGUATION
        # =====================================================

        if parse_lists:

            selected_parses = (
                self.disambiguator
                .disambiguate(
                    parse_lists
                )
            )

        else:

            selected_parses = []

        # =====================================================
        # RESTORE TOKEN ORDER
        # =====================================================

        results = []

        for item in analyzable_items:

            word = (
                item["word"]
            )

            parse_index = (
                item["parse_index"]
            )

            # -------------------------------------------------
            # FALLBACK TOKEN
            # -------------------------------------------------

            if parse_index is None:

                results.append({
                    "word":
                        word,

                    "root":
                        word,

                    "analysis":
                        "UNKNOWN",

                    "analysis_source":
                        "fallback",

                    "analyzer_reset":
                        item[
                            "analyzer_reset"
                        ]
                })

                continue

            # -------------------------------------------------
            # SELECTED MORPHOLOGICAL PARSE
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

            results.append({
                "word":
                    word,

                "root":
                    root,

                "analysis":
                    analysis,

                "analysis_source":
                    item[
                        "source"
                    ],

                "analyzer_reset":
                    item[
                        "analyzer_reset"
                    ]
            })

        return results

    # =========================================================
    # CACHE / PERFORMANCE DIAGNOSTICS
    # =========================================================

    def get_cache_stats(
        self
    ):
        """
        Returns performance diagnostics.

        These values do not affect classification.
        """

        return {
            "normal_requests":
                self._normal_requests,

            "normal_analyzer_resets":
                self._normal_resets,

            "robust_requests":
                self._robust_requests,

            "robust_cache_hits":
                self._robust_cache_hits,

            "robust_cache_misses":
                self._robust_cache_misses,

            "robust_analyzer_builds":
                self._robust_analyzer_builds,

            "robust_failures":
                self._robust_failures,

            "robust_cached_words":
                len(
                    self._robust_parse_cache
                )
        }

    def clear_robust_cache(
        self
    ):
        """
        Clears the external robust-analysis cache.
        """

        self._robust_parse_cache.clear()