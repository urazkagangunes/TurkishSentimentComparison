from src.hisnet.hisnet_classifier import HisNetClassifier
from src.dependency.stanza_parser import TurkishDependencyParser
from src.dependency.sentiment_scope_resolver import SentimentScopeResolver


class HisNetDependencyClassifier:
    """
    HisNet + Dependency Parsing sentiment classifier.

    Pipeline:
        Text
            -> Morphological Analysis
            -> LongestRootFirstDisambiguation
            -> HisNet lexical sentiment candidates

        Text
            -> Stanford Stanza
            -> Dependency parse

        HisNet candidates + Dependency structure
            -> SentimentScopeResolver
            -> target / opinion / modifier / negation / contrast
            -> dependency-aware sentiment aggregation

    IMPORTANT:
        This classifier does not modify HisNet or Stanza outputs.

        HisNet provides lexical sentiment information.
        Stanza provides dependency information.

        This class combines the outputs of both systems.
    """

    def __init__(self):
        self.hisnet = HisNetClassifier()
        self.parser = TurkishDependencyParser()
        self.scope_resolver = SentimentScopeResolver()

    @staticmethod
    def _prediction_from_scores(
        positive_score,
        negative_score
    ):
        if positive_score > negative_score:
            return "positive"

        if negative_score > positive_score:
            return "negative"

        return "neutral"

    def _apply_dependency_scope(
        self,
        scope
    ):
        """
        Applies dependency-based information to one
        HisNet sentiment candidate.

        Current composition rules:

        1. If dependency structure shows that a HisNet
           sentiment candidate is functioning as a target
           rather than an opinion, its lexical sentiment
           contribution is suppressed.

        2. If grammatical negation affects the opinion,
           positive and negative scores are swapped.

        Modifier and contrast relations are currently
        detected and reported, but their numeric weights
        are not yet applied here.
        """

        original_positive = float(
            scope.get("positive", 0.0)
        )

        original_negative = float(
            scope.get("negative", 0.0)
        )

        adjusted_positive = original_positive
        adjusted_negative = original_negative

        operations = []

        # =====================================================
        # 1. CONTEXTUAL ROLE
        # =====================================================

        if not scope.get(
            "active_sentiment",
            True
        ):
            adjusted_positive = 0.0
            adjusted_negative = 0.0

            operations.append(
                "suppressed_as_target"
            )

            return {
                **scope,

                "original_positive":
                    original_positive,

                "original_negative":
                    original_negative,

                "adjusted_positive":
                    adjusted_positive,

                "adjusted_negative":
                    adjusted_negative,

                "operations":
                    operations
            }

        # =====================================================
        # 2. NEGATION
        # =====================================================

        negations = scope.get(
            "negations",
            []
        )

        # Odd number of negations reverses polarity.
        # Even number restores the original polarity.
        if len(negations) % 2 == 1:

            adjusted_positive, adjusted_negative = (
                adjusted_negative,
                adjusted_positive
            )

            operations.append(
                "negation_flip"
            )

        # =====================================================
        # NOTE:
        #
        # Modifier and contrast information is intentionally
        # preserved without assigning numeric weights yet.
        #
        # This prevents us from tuning the classifier to the
        # development examples before the composition scheme
        # is explicitly fixed.
        # =====================================================

        if scope.get("modifiers"):
            operations.append(
                "modifier_detected"
            )

        if scope.get("contrast"):
            operations.append(
                "contrast_detected"
            )

        return {
            **scope,

            "original_positive":
                original_positive,

            "original_negative":
                original_negative,

            "adjusted_positive":
                adjusted_positive,

            "adjusted_negative":
                adjusted_negative,

            "operations":
                operations
        }

    def _classify_sentence(
        self,
        sentence_text,
        dependency_tokens
    ):
        """
        Classifies a single sentence.
        """

        # =====================================================
        # HISNET BASELINE
        # =====================================================

        hisnet_result = self.hisnet.classify(
            sentence_text
        )

        # =====================================================
        # DEPENDENCY SCOPE RESOLUTION
        # =====================================================

        scopes = self.scope_resolver.resolve(
            hisnet_result["sentiment_words"],
            dependency_tokens,
            hisnet_result["morphological_analyses"]
        )

        # =====================================================
        # APPLY DEPENDENCY INFORMATION
        # =====================================================

        resolved_scopes = []

        for scope in scopes:

            resolved = (
                self._apply_dependency_scope(
                    scope
                )
            )

            resolved_scopes.append(
                resolved
            )

        # =====================================================
        # FINAL SCORE
        # =====================================================

        positive_score = sum(
            scope["adjusted_positive"]
            for scope in resolved_scopes
        )

        negative_score = sum(
            scope["adjusted_negative"]
            for scope in resolved_scopes
        )

        prediction = (
            self._prediction_from_scores(
                positive_score,
                negative_score
            )
        )

        return {
            "text":
                sentence_text,

            # ---------------------------
            # Baseline
            # ---------------------------

            "hisnet_prediction":
                hisnet_result["prediction"],

            "hisnet_positive_score":
                hisnet_result["positive_score"],

            "hisnet_negative_score":
                hisnet_result["negative_score"],

            # ---------------------------
            # HisNet + Dependency
            # ---------------------------

            "prediction":
                prediction,

            "positive_score":
                positive_score,

            "negative_score":
                negative_score,

            # ---------------------------
            # Linguistic analysis
            # ---------------------------

            "scopes":
                resolved_scopes,

            "dependency_tokens":
                dependency_tokens,

            "morphological_analyses":
                hisnet_result[
                    "morphological_analyses"
                ],

            "lexicon_coverage":
                hisnet_result[
                    "lexicon_coverage"
                ],

            "sentiment_coverage":
                hisnet_result[
                    "sentiment_coverage"
                ]
        }

    def classify(self, text):
        """
        Classifies a complete text.

        Multiple sentences are processed separately and
        their resulting sentiment scores are aggregated.
        """

        parsed_sentences = (
            self.parser.parse(text)
        )

        sentence_results = []

        total_positive = 0.0
        total_negative = 0.0

        for parsed_sentence in parsed_sentences:

            sentence_text = (
                parsed_sentence["text"]
            )

            dependency_tokens = (
                parsed_sentence["tokens"]
            )

            result = self._classify_sentence(
                sentence_text,
                dependency_tokens
            )

            sentence_results.append(
                result
            )

            total_positive += (
                result["positive_score"]
            )

            total_negative += (
                result["negative_score"]
            )

        prediction = (
            self._prediction_from_scores(
                total_positive,
                total_negative
            )
        )

        return {
            "text":
                text,

            "prediction":
                prediction,

            "positive_score":
                total_positive,

            "negative_score":
                total_negative,

            "sentences":
                sentence_results
        }