import math

from src.hisnet.hisnet_classifier import (
    HisNetClassifier
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

    Fair comparison design:

        Text
            -> HisNet ONCE
            -> morphological analyses
            -> HisNet lexical sentiment candidates

        Same text
            -> Stanza dependency parse

        SAME HisNet candidates
            + dependency structure
            -> target / modifier / negation / contrast
            -> dependency-aware sentiment aggregation

    IMPORTANT:
        HisNet + Dependency must start from exactly the same
        HisNet result as the HisNet-only baseline.

        The dependency classifier must NOT independently
        recompute HisNet during evaluation.
    """

    def __init__(self):

        # Kept as a fallback so that classify(text) can still
        # be used directly in unit tests or interactive use.
        #
        # During fair evaluation, the already-computed
        # hisnet_result is passed explicitly to classify().
        self.hisnet = (
            HisNetClassifier()
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

        if positive_score > negative_score:
            return "positive"

        if negative_score > positive_score:
            return "negative"

        return "neutral"

    # =========================================================
    # DEPENDENCY TOKEN FLATTENING
    # =========================================================

    @staticmethod
    def _flatten_dependency_sentences(
        parsed_sentences
    ):
        """
        Stanza token IDs restart from 1 for every sentence.

        For document-level HisNet candidates we need one
        globally unique dependency-token space.

        Example:

            sentence 1:
                1, 2, 3

            sentence 2:
                1, 2, 3, 4

        becomes:

            sentence 1:
                1, 2, 3

            sentence 2:
                4, 5, 6, 7

        Dependency heads are shifted by the same offset.

        Root heads remain 0, therefore dependency relations
        never cross sentence boundaries.
        """

        global_tokens = []
        sentence_results = []
        token_to_sentence = {}

        offset = 0

        for sentence_index, parsed_sentence in enumerate(
            parsed_sentences
        ):

            sentence_tokens = []

            local_tokens = parsed_sentence.get(
                "tokens",
                []
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

                    # Diagnostic metadata.
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
                ] = sentence_index

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
        Verifies that dependency scope resolution has preserved
        every HisNet sentiment candidate BEFORE dependency
        operations are applied.

        Therefore:

            sum(scope positive)
                ==
            HisNet positive score

        and:

            sum(scope negative)
                ==
            HisNet negative score

        must always hold.

        If this fails, we have a pipeline/alignment bug rather
        than a legitimate dependency effect.
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
                "\nHisNet/Dependency fairness invariant "
                "failed.\n"
                f"HisNet positive: {hisnet_positive}\n"
                f"Scope positive:  {scope_positive}\n"
                f"HisNet negative: {hisnet_negative}\n"
                f"Scope negative:  {scope_negative}\n"
                "\nDependency scope resolution must "
                "preserve all baseline HisNet scores "
                "before dependency operations."
            )

    # =========================================================
    # DEPENDENCY COMPOSITION
    # =========================================================

    def _apply_dependency_scope(
        self,
        scope
    ):
        """
        Applies dependency-based information to one
        HisNet sentiment candidate.

        Current rules:

        1. Contextual target suppression
        2. Negation polarity flip

        Modifier and contrast information are diagnostic only.
        They do not currently change numeric sentiment scores.
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
        # 1. CONTEXTUAL ROLE / SUPPRESSION
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
        # 3. DIAGNOSTIC-ONLY FEATURES
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
    # SENTENCE-LEVEL DIAGNOSTIC OUTPUT
    # =========================================================

    def _build_sentence_results(
        self,
        sentence_results,
        resolved_scopes,
        token_to_sentence
    ):
        """
        Assigns matched dependency scopes back to their original
        Stanza sentence for diagnostic inspection.

        Top-level document scoring does NOT depend on this
        grouping.

        Unmatched scopes remain available in the top-level
        'scopes' output so that they are never lost.
        """

        # -----------------------------------------------------
        # Assign matched scopes to their Stanza sentence.
        # -----------------------------------------------------

        for scope in resolved_scopes:

            sentiment_token = (
                scope.get(
                    "sentiment_token"
                )
            )

            if sentiment_token is None:

                # If there is only one Stanza sentence, the
                # unmatched candidate necessarily belongs there.
                if len(
                    sentence_results
                ) == 1:

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
        # Compute sentence-level diagnostic scores directly
        # from the already-resolved scopes.
        #
        # HisNet is NOT rerun here.
        # -----------------------------------------------------

        for sentence_result in sentence_results:

            scopes = sentence_result[
                "scopes"
            ]

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
            ] = original_positive

            sentence_result[
                "hisnet_negative_score"
            ] = original_negative

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
            ] = adjusted_positive

            sentence_result[
                "negative_score"
            ] = adjusted_negative

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
        Classifies a complete text using HisNet + Dependency.

        Preferred fair-evaluation usage:

            hisnet_result = hisnet.classify(text)

            dependency_result = classifier.classify(
                text,
                hisnet_result=hisnet_result
            )

        If hisnet_result is not supplied, HisNet is computed
        internally only for standalone/unit-test convenience.
        """

        # =====================================================
        # BASELINE
        # =====================================================

        baseline_was_reused = (
            hisnet_result is not None
        )

        if hisnet_result is None:

            hisnet_result = (
                self.hisnet.classify(
                    text
                )
            )

        # Safety: prevent accidentally passing the baseline
        # result of another text.
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
        # DEPENDENCY PARSE
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
        # DEPENDENCY SCOPE RESOLUTION
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
        # FAIRNESS CHECK BEFORE ANY SCORE MODIFICATION
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
            for scope in resolved_scopes
            if scope.get(
                "alignment_status"
            ) == "unmatched"
        ]

        # =====================================================
        # OUTPUT
        # =====================================================

        return {
            "text":
                text,

            # ---------------------------------------------
            # Exact HisNet baseline reused by H+Dependency
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
            # HisNet + Dependency
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

            # Canonical document-level scopes.
            "scopes":
                resolved_scopes,

            "unmatched_scopes":
                unmatched_scopes,

            # Sentence grouping is diagnostic only.
            "sentences":
                sentence_results,

            # Full baseline data.
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