from SentiNet.SentiLiteralNet import (
    SentiLiteralNet
)

from src.common.morphology import (
    MorphologicalRootExtractor
)

NEUTRAL_THRESHOLD = 0.05

class HisNetClassifier:

    @staticmethod
    def _prediction_from_scores(
        positive_score,
        negative_score
    ):
        """
        Converts aggregated HisNet polarity scores into
        a sentence-level sentiment label.

        Decision is based on the difference between
        positive and negative scores.

        net_score > +0.05  -> positive
        net_score < -0.05  -> negative
        otherwise          -> neutral
        """

        net_score = (
            positive_score
            - negative_score
        )

        if net_score > NEUTRAL_THRESHOLD:
            return "positive"

        if net_score < -NEUTRAL_THRESHOLD:
            return "negative"

        return "neutral"

    def __init__(self):

        self.sentinet = (
            SentiLiteralNet()
        )

        self.root_extractor = (
            MorphologicalRootExtractor()
        )

        # -----------------------------------------------------
        # HISNET LOOKUP CACHE
        # -----------------------------------------------------
        #
        # HisNet is static during an experiment.
        #
        # Therefore:
        #
        #     root -> positive / negative scores
        #
        # can safely be cached.
        # -----------------------------------------------------

        self._lookup_cache = {}

        self._lookup_requests = 0
        self._lookup_cache_hits = 0
        self._lookup_cache_misses = 0

    # =========================================================
    # HISNET LOOKUP
    # =========================================================

    def _lookup_root(
        self,
        root
    ):
        """
        Looks up a disambiguated morphological root in HisNet.

        Results are memoized because HisNet does not change
        during evaluation.

        Returns None if the root does not exist in HisNet.
        """

        self._lookup_requests += 1

        # -----------------------------------------------------
        # CACHE HIT
        # -----------------------------------------------------

        if root in self._lookup_cache:

            self._lookup_cache_hits += 1

            cached = (
                self._lookup_cache[
                    root
                ]
            )

            # Cached negative lookup.
            if cached is None:
                return None

            positive, negative = (
                cached
            )

            return {
                "root":
                    root,

                "positive":
                    positive,

                "negative":
                    negative
            }

        # -----------------------------------------------------
        # CACHE MISS
        # -----------------------------------------------------

        self._lookup_cache_misses += 1

        try:

            senti_literal = (
                self.sentinet
                .getSentiLiteral(
                    root
                )
            )

            positive = float(
                senti_literal
                .getPositiveScore()
            )

            negative = float(
                senti_literal
                .getNegativeScore()
            )

            self._lookup_cache[
                root
            ] = (
                positive,
                negative
            )

            return {
                "root":
                    root,

                "positive":
                    positive,

                "negative":
                    negative
            }

        except KeyError:

            # HisNet is static, so nonexistent roots can also
            # safely be cached.
            self._lookup_cache[
                root
            ] = None

            return None

    # =========================================================
    # CLASSIFICATION
    # =========================================================

    def classify(
        self,
        text
    ):
        """
        HisNet-only sentiment classification.

        Pipeline:

            Text
              -> Morphological analysis
              -> LongestRootFirstDisambiguation
              -> Selected morphological roots
              -> HisNet lookup
              -> Score aggregation
              -> Positive / Negative / Neutral

        Dependency information is NOT used here.
        """

        morphological_analyses = (
            self.root_extractor
            .analyze_sentence(
                text
            )
        )

        positive_score = 0.0
        negative_score = 0.0

        matched_words = []
        sentiment_words = []
        unmatched_words = []

        # =====================================================
        # HISNET LOOKUP
        # =====================================================

        for item in morphological_analyses:

            word = (
                item["word"]
            )

            root = (
                item["root"]
            )

            analysis = (
                item["analysis"]
            )

            senti_result = (
                self._lookup_root(
                    root
                )
            )

            # -------------------------------------------------
            # ROOT NOT FOUND IN HISNET
            # -------------------------------------------------

            if senti_result is None:

                unmatched_words.append({
                    "word":
                        word,

                    "root":
                        root,

                    "analysis":
                        analysis
                })

                continue

            positive = (
                senti_result[
                    "positive"
                ]
            )

            negative = (
                senti_result[
                    "negative"
                ]
            )

            positive_score += (
                positive
            )

            negative_score += (
                negative
            )

            matched_entry = {
                "word":
                    word,

                "root":
                    root,

                "analysis":
                    analysis,

                "positive":
                    positive,

                "negative":
                    negative
            }

            matched_words.append(
                matched_entry
            )

            if (
                positive > 0.0
                or negative > 0.0
            ):

                sentiment_words.append(
                    matched_entry
                )

        # =====================================================
        # SENTIMENT DECISION
        # =====================================================

        prediction = (
            self._prediction_from_scores(
                positive_score,
                negative_score
            )
        )
        # =====================================================
        # COVERAGE
        # =====================================================

        token_count = len(
            morphological_analyses
        )

        matched_word_count = len(
            matched_words
        )

        sentiment_word_count = len(
            sentiment_words
        )

        lexicon_coverage = (
            matched_word_count
            / token_count

            if token_count > 0

            else 0.0
        )

        sentiment_coverage = (
            sentiment_word_count
            / token_count

            if token_count > 0

            else 0.0
        )

        # =====================================================
        # OUTPUT
        # =====================================================

        return {
            "text":
                text,

            "prediction":
                prediction,

            "positive_score":
                positive_score,

            "negative_score":
                negative_score,

            "net_score":
                positive_score
                - negative_score,

            "neutral_threshold":
                NEUTRAL_THRESHOLD,

            "morphological_analyses":
                morphological_analyses,

            "matched_words":
                matched_words,

            "sentiment_words":
                sentiment_words,

            "unmatched_words":
                unmatched_words,

            "token_count":
                token_count,

            "matched_word_count":
                matched_word_count,

            "sentiment_word_count":
                sentiment_word_count,

            "lexicon_coverage":
                lexicon_coverage,

            "sentiment_coverage":
                sentiment_coverage
        }

    # =========================================================
    # PERFORMANCE DIAGNOSTICS
    # =========================================================

    def get_cache_stats(
        self
    ):
        """
        Returns HisNet and morphology cache statistics.
        """

        return {
            "hisnet_lookup_requests":
                self._lookup_requests,

            "hisnet_lookup_cache_hits":
                self._lookup_cache_hits,

            "hisnet_lookup_cache_misses":
                self._lookup_cache_misses,

            "hisnet_cached_roots":
                len(
                    self._lookup_cache
                ),

            "morphology":
                self.root_extractor
                .get_cache_stats()
        }