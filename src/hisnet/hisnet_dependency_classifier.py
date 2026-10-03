import math

from src.hisnet.hisnet_classifier import (
    HisNetClassifier,
    NEUTRAL_THRESHOLD
)

from src.dependency.stanza_parser import (
    TurkishDependencyParser
)

from src.dependency.sentiment_scope_resolver import (
    SentimentScopeResolver
)


class HisNetDependencyClassifier:
    """
    HisNet + Dependency Parsing sentiment classifier.

    Fair comparison:

        text
          -> HisNet ONCE
          -> same HisNet result
                |
                +--> HisNet baseline
                |
                +--> Dependency composition

    HisNet + Dependency therefore starts from exactly the
    same lexical sentiment candidates and scores as HisNet.

    Dependency information may then apply:

        - target suppression
        - negation flip

    Modifier and contrast information remain diagnostic only.
    """

    def __init__(
        self,
        hisnet_classifier=None
    ):

        # -----------------------------------------------------
        # SHARED HISNET
        # -----------------------------------------------------
        #
        # SentimentEvaluator injects its HisNetClassifier here.
        #
        # Therefore we do NOT create a second expensive
        # HisNet / MorphologicalAnalyzer / SentiLiteralNet
        # instance.
        #
        # If this class is used standalone, HisNet is created
        # lazily only when necessary.
        # -----------------------------------------------------

        self.hisnet = (
            hisnet_classifier
        )

        self.parser = (
            TurkishDependencyParser()
        )

        self.scope_resolver = (
            SentimentScopeResolver()
        )

    # =========================================================
    # SCORE -> LABEL
    # =========================================================

    @staticmethod
    def _prediction_from_scores(
            positive_score,
            negative_score
    ):
        """
        Uses exactly the same decision rule as the
        HisNet-only baseline.

        This keeps the HisNet vs HisNet+Dependency
        comparison fair.
        """

        return (
            HisNetClassifier
            ._prediction_from_scores(
                positive_score,
                negative_score
            )
        )

    # =========================================================
    # GLOBAL DEPENDENCY TOKEN IDS
    # =========================================================

    @staticmethod
    def _flatten_dependency_sentences(
        parsed_sentences
    ):
        """
        Stanza token IDs restart from 1 in every sentence.

        For document-level HisNet candidates, dependency tokens
        are converted to globally unique IDs.

        Heads are shifted by the same sentence offset.

        Root heads remain 0.
        """

        global_tokens = []
        sentence_results = []
        token_to_sentence = {}

        offset = 0

        for (
            sentence_index,
            parsed_sentence
        ) in enumerate(
            parsed_sentences
        ):

            sentence_tokens = []

            local_tokens = (
                parsed_sentence.get(
                    "tokens",
                    []
                )
            )

            max_local_id = 0

            for token in local_tokens:

                local_id = int(
                    token["id"]
                )

                local_head = int(
                    token.get(
                        "head",
                        0
                    )
                    or 0
                )

                global_id = (
                    offset
                    + local_id
                )

                if local_head == 0:

                    global_head = 0

                else:

                    global_head = (
                        offset
                        + local_head
                    )

                global_token = {
                    **token,

                    "id":
                        global_id,

                    "head":
                        global_head,

                    "sentence_index":
                        sentence_index,

                    "local_id":
                        local_id,

                    "local_head":
                        local_head
                }

                global_tokens.append(
                    global_token
                )

                sentence_tokens.append(
                    global_token
                )

                token_to_sentence[
                    global_id
                ] = (
                    sentence_index
                )

                max_local_id = max(
                    max_local_id,
                    local_id
                )

            sentence_results.append({
                "sentence_index":
                    sentence_index,

                "text":
                    parsed_sentence.get(
                        "text"
                    ),

                "dependency_tokens":
                    sentence_tokens,

                "scopes":
                    []
            })

            offset += (
                max_local_id
            )

        return (
            global_tokens,
            sentence_results,
            token_to_sentence
        )

    # =========================================================
    # FAIRNESS INVARIANT
    # =========================================================

    @staticmethod
    def _validate_initial_scores(
        scopes,
        hisnet_result
    ):
        """
        Ensures that dependency scope resolution preserves all
        baseline HisNet sentiment scores BEFORE any dependency
        operation is applied.

        If this fails, there is a pipeline/alignment bug.
        """

        scope_positive = sum(
            float(
                scope.get(
                    "positive",
                    0.0
                )
            )

            for scope in scopes
        )

        scope_negative = sum(
            float(
                scope.get(
                    "negative",
                    0.0
                )
            )

            for scope in scopes
        )

        hisnet_positive = float(
            hisnet_result[
                "positive_score"
            ]
        )

        hisnet_negative = float(
            hisnet_result[
                "negative_score"
            ]
        )

        positive_matches = (
            math.isclose(
                scope_positive,
                hisnet_positive,
                rel_tol=1e-9,
                abs_tol=1e-9
            )
        )

        negative_matches = (
            math.isclose(
                scope_negative,
                hisnet_negative,
                rel_tol=1e-9,
                abs_tol=1e-9
            )
        )

        if (
            not positive_matches
            or not negative_matches
        ):

            raise RuntimeError(
                "\n"
                "HisNet/Dependency fairness invariant "
                "failed.\n"
                f"HisNet positive: "
                f"{hisnet_positive}\n"
                f"Scope positive:  "
                f"{scope_positive}\n"
                f"HisNet negative: "
                f"{hisnet_negative}\n"
                f"Scope negative:  "
                f"{scope_negative}\n"
                "\n"
                "Dependency scope resolution must preserve "
                "all baseline HisNet scores before "
                "dependency operations."
            )

    # =========================================================
    # APPLY DEPENDENCY RULES
    # =========================================================

    @staticmethod
    def _apply_dependency_scope(
            scope
    ):
        """
        Applies dependency-based composition to one HisNet
        sentiment candidate.

        Current numeric rules:

            1. Target suppression
            2. Negation polarity flip

        Modifier and contrast are diagnostic-only.
        """

        original_positive = float(
            scope.get(
                "positive",
                0.0
            )
        )

        original_negative = float(
            scope.get(
                "negative",
                0.0
            )
        )

        adjusted_positive = (
            original_positive
        )

        adjusted_negative = (
            original_negative
        )

        operations = []

        # =====================================================
        # 1. TARGET SUPPRESSION
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

        negations = (
            scope.get(
                "negations",
                []
            )
        )

        # Odd number of negations reverses polarity.
        # Even number restores original polarity.
        if (
            len(negations) % 2
            == 1
        ):

            (
                adjusted_positive,
                adjusted_negative
            ) = (
                adjusted_negative,
                adjusted_positive
            )

            operations.append(
                "negation_flip"
            )

        # =====================================================
        # 3. DIAGNOSTIC FEATURES
        # =====================================================

        if scope.get(
            "modifiers"
        ):

            operations.append(
                "modifier_detected"
            )

        if scope.get(
            "contrast"
        ):

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

    # =========================================================
    # SENTENCE-LEVEL DIAGNOSTICS
    # =========================================================

    def _build_sentence_results(
        self,
        sentence_results,
        resolved_scopes,
        token_to_sentence
    ):
        """
        Assigns matched dependency scopes back to their Stanza
        sentence.

        This grouping is diagnostic only.

        Document-level scoring uses the canonical top-level
        scope list.
        """

        for scope in resolved_scopes:

            sentiment_token = (
                scope.get(
                    "sentiment_token"
                )
            )

            if sentiment_token is None:

                # If document contains only one Stanza sentence,
                # unmatched item clearly belongs there.
                if (
                    len(sentence_results)
                    == 1
                ):

                    sentence_results[
                        0
                    ][
                        "scopes"
                    ].append(
                        scope
                    )

                continue

            token_id = (
                sentiment_token.get(
                    "id"
                )
            )

            sentence_index = (
                token_to_sentence.get(
                    token_id
                )
            )

            if sentence_index is None:
                continue

            sentence_results[
                sentence_index
            ][
                "scopes"
            ].append(
                scope
            )

        # -----------------------------------------------------
        # Sentence-level diagnostic scores
        # -----------------------------------------------------

        for sentence_result in sentence_results:

            scopes = (
                sentence_result[
                    "scopes"
                ]
            )

            original_positive = sum(
                scope[
                    "original_positive"
                ]

                for scope in scopes
            )

            original_negative = sum(
                scope[
                    "original_negative"
                ]

                for scope in scopes
            )

            adjusted_positive = sum(
                scope[
                    "adjusted_positive"
                ]

                for scope in scopes
            )

            adjusted_negative = sum(
                scope[
                    "adjusted_negative"
                ]

                for scope in scopes
            )

            sentence_result[
                "hisnet_positive_score"
            ] = (
                original_positive
            )

            sentence_result[
                "hisnet_negative_score"
            ] = (
                original_negative
            )

            sentence_result[
                "hisnet_prediction"
            ] = (
                self._prediction_from_scores(
                    original_positive,
                    original_negative
                )
            )

            sentence_result[
                "positive_score"
            ] = (
                adjusted_positive
            )

            sentence_result[
                "negative_score"
            ] = (
                adjusted_negative
            )

            sentence_result[
                "prediction"
            ] = (
                self._prediction_from_scores(
                    adjusted_positive,
                    adjusted_negative
                )
            )

        return sentence_results

    # =========================================================
    # MAIN CLASSIFICATION
    # =========================================================

    def classify(
        self,
        text,
        hisnet_result=None
    ):
        """
        Classifies complete text using HisNet + Dependency.

        Preferred evaluation usage:

            hisnet_result = hisnet.classify(text)

            dependency_result = classifier.classify(
                text,
                hisnet_result=hisnet_result
            )

        Therefore HisNet is NOT recalculated.

        For standalone usage, HisNet is lazily created only if
        no precomputed result is supplied.
        """

        # =====================================================
        # HISNET BASELINE
        # =====================================================

        baseline_was_reused = (
            hisnet_result is not None
        )

        if hisnet_result is None:

            # Standalone fallback only.
            if self.hisnet is None:

                self.hisnet = (
                    HisNetClassifier()
                )

            hisnet_result = (
                self.hisnet.classify(
                    text
                )
            )

        # -----------------------------------------------------
        # Safety check
        # -----------------------------------------------------

        if (
            hisnet_result.get(
                "text"
            )
            != text
        ):

            raise ValueError(
                "The supplied hisnet_result does not "
                "belong to the current text."
            )

        # =====================================================
        # STANZA DEPENDENCY PARSE
        # =====================================================

        parsed_sentences = (
            self.parser.parse(
                text
            )
        )

        (
            dependency_tokens,
            sentence_results,
            token_to_sentence
        ) = (
            self._flatten_dependency_sentences(
                parsed_sentences
            )
        )

        # =====================================================
        # SCOPE RESOLUTION
        # =====================================================

        scopes = (
            self.scope_resolver.resolve(
                hisnet_result[
                    "sentiment_words"
                ],

                dependency_tokens,

                hisnet_result[
                    "morphological_analyses"
                ]
            )
        )

        # =====================================================
        # FAIRNESS CHECK
        # =====================================================

        self._validate_initial_scores(
            scopes,
            hisnet_result
        )

        # =====================================================
        # APPLY DEPENDENCY OPERATIONS
        # =====================================================

        resolved_scopes = []

        for scope in scopes:

            resolved_scope = (
                self._apply_dependency_scope(
                    scope
                )
            )

            resolved_scopes.append(
                resolved_scope
            )

        # =====================================================
        # FINAL DOCUMENT SCORE
        # =====================================================

        positive_score = sum(
            scope[
                "adjusted_positive"
            ]

            for scope in resolved_scopes
        )

        negative_score = sum(
            scope[
                "adjusted_negative"
            ]

            for scope in resolved_scopes
        )

        prediction = (
            self._prediction_from_scores(
                positive_score,
                negative_score
            )
        )

        # =====================================================
        # SENTENCE DIAGNOSTICS
        # =====================================================

        sentence_results = (
            self._build_sentence_results(
                sentence_results,
                resolved_scopes,
                token_to_sentence
            )
        )

        unmatched_scopes = [
            scope

            for scope
            in resolved_scopes

            if scope.get(
                "alignment_status"
            )
            == "unmatched"
        ]

        # =====================================================
        # OUTPUT
        # =====================================================

        return {
            "text":
                text,

            # ---------------------------------------------
            # Baseline
            # ---------------------------------------------

            "hisnet_prediction":
                hisnet_result[
                    "prediction"
                ],

            "hisnet_positive_score":
                hisnet_result[
                    "positive_score"
                ],

            "hisnet_negative_score":
                hisnet_result[
                    "negative_score"
                ],

            # ---------------------------------------------
            # Dependency result
            # ---------------------------------------------

            "prediction":
                prediction,

            "positive_score":
                positive_score,

            "negative_score":
                negative_score,

            # ---------------------------------------------
            # Fairness diagnostics
            # ---------------------------------------------

            "baseline_reused":
                baseline_was_reused,

            "baseline_positive_score":
                hisnet_result[
                    "positive_score"
                ],

            "baseline_negative_score":
                hisnet_result[
                    "negative_score"
                ],

            # ---------------------------------------------
            # Canonical scopes
            # ---------------------------------------------

            "scopes":
                resolved_scopes,

            "unmatched_scopes":
                unmatched_scopes,

            # Sentence grouping is diagnostic only.
            "sentences":
                sentence_results,

            # ---------------------------------------------
            # Shared HisNet data
            # ---------------------------------------------

            "morphological_analyses":
                hisnet_result[
                    "morphological_analyses"
                ],

            "sentiment_words":
                hisnet_result[
                    "sentiment_words"
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