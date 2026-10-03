from src.hisnet.hisnet_classifier import (
    HisNetClassifier,
    NEUTRAL_THRESHOLD
)

from src.hisnet.hisnet_dependency_classifier import (
    HisNetDependencyClassifier
)

from src.dependency.stanza_parser import (
    TurkishDependencyParser
)

from src.dependency.sentiment_scope_resolver import (
    SentimentScopeResolver
)


class DependencyHisNetClassifier:
    """
    Dependency -> HisNet sentiment classifier.

    Main difference from HisNetDependencyClassifier:

        Dependency -> HisNet:

            Text
                -> Dependency parse FIRST
                -> syntactic opinion candidates
                -> target / negation / modifier / contrast
                -> morphological roots
                -> HisNet lookup ONLY for selected candidates
                -> dependency-aware composition
                -> sentiment prediction

        HisNet -> Dependency:

            Text
                -> HisNet lookup FIRST
                -> lexical sentiment candidates
                -> dependency scope resolution
                -> dependency-aware composition

    Therefore the main experimental variable is the stage
    at which dependency information enters the lexicon pipeline.
    """

    def __init__(self):

        self.parser = (
            TurkishDependencyParser()
        )

        self.scope_resolver = (
            SentimentScopeResolver()
        )

        # We reuse HisNetClassifier infrastructure for:
        #
        #   - morphological analysis
        #   - cached HisNet lookup
        #   - common decision threshold
        #
        # IMPORTANT:
        # classify() of HisNetClassifier is NOT called here.
        # Otherwise every token would be looked up before
        # dependency filtering.
        self.hisnet = (
            HisNetClassifier()
        )

    # =========================================================
    # HISNET LOOKUP FOR ONE DEPENDENCY CANDIDATE
    # =========================================================

    def _score_candidate(
        self,
        candidate
    ):
        """
        Performs HisNet lookup ONLY after dependency-based
        candidate selection.

        Returns:
            scored scope
        or:
            None
        """

        root = (
            candidate.get(
                "root"
            )
        )

        if not root:
            return None

        senti_result = (
            self.hisnet._lookup_root(
                root
            )
        )

        if senti_result is None:
            return None

        positive = float(
            senti_result[
                "positive"
            ]
        )

        negative = float(
            senti_result[
                "negative"
            ]
        )

        # HisNet knows the root but it does not carry
        # sentiment polarity.
        #
        # Such a token remains useful as a dependency
        # candidate diagnostic, but it is not a
        # sentiment-bearing candidate.
        if (
            positive == 0.0
            and negative == 0.0
        ):
            return None

        return {
            **candidate,

            "positive":
                positive,

            "negative":
                negative
        }

    # =========================================================
    # MAIN CLASSIFICATION
    # =========================================================

    def classify(
        self,
        text
    ):
        """
        Classifies one text using the Dependency -> HisNet
        architecture.
        """

        # =====================================================
        # STEP 1
        # DEPENDENCY PARSING FIRST
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
            HisNetDependencyClassifier
            ._flatten_dependency_sentences(
                parsed_sentences
            )
        )

        # =====================================================
        # STEP 2
        # MORPHOLOGICAL ANALYSIS
        #
        # This happens AFTER dependency parsing in this
        # experimental configuration.
        # =====================================================

        morphological_analyses = (
            self.hisnet
            .root_extractor
            .analyze_sentence(
                text
            )
        )

        # =====================================================
        # STEP 3
        # DEPENDENCY-BASED CANDIDATE EXTRACTION
        #
        # No HisNet scores have been used yet.
        # =====================================================

        dependency_candidates = (
            self.scope_resolver
            .resolve_dependency_candidates(
                dependency_tokens,
                morphological_analyses
            )
        )

        # =====================================================
        # STEP 4
        # HISNET LOOKUP ONLY FOR DEPENDENCY CANDIDATES
        # =====================================================

        sentiment_scopes = []

        candidate_diagnostics = []

        lexicon_matched_candidate_count = 0

        for candidate in (
            dependency_candidates
        ):

            root = candidate.get(
                "root"
            )

            senti_result = (
                self.hisnet._lookup_root(
                    root
                )
                if root
                else None
            )

            if senti_result is None:

                candidate_diagnostics.append({
                    **candidate,

                    "lexicon_status":
                        "not_found",

                    "positive":
                        None,

                    "negative":
                        None
                })

                continue

            lexicon_matched_candidate_count += 1

            positive = float(
                senti_result[
                    "positive"
                ]
            )

            negative = float(
                senti_result[
                    "negative"
                ]
            )

            # ---------------------------------------------
            # Diagnostic record
            # ---------------------------------------------

            candidate_diagnostics.append({
                **candidate,

                "lexicon_status":
                    (
                        "sentiment"
                        if (
                            positive > 0.0
                            or negative > 0.0
                        )
                        else "neutral_lexicon_entry"
                    ),

                "positive":
                    positive,

                "negative":
                    negative
            })

            # ---------------------------------------------
            # Only sentiment-bearing candidates continue
            # into composition.
            # ---------------------------------------------

            if (
                positive == 0.0
                and negative == 0.0
            ):
                continue

            sentiment_scopes.append({
                **candidate,

                "positive":
                    positive,

                "negative":
                    negative
            })

        # =====================================================
        # STEP 5
        # CONTEXTUAL ROLE RESOLUTION
        #
        # Usually target-only words have already been filtered
        # before HisNet lookup in D->H.
        #
        # This remains as a conservative safety mechanism for
        # cases where a sentiment-bearing candidate is itself
        # the target of another sentiment expression.
        # =====================================================

        sentiment_scopes = (
            self.scope_resolver
            ._resolve_sentiment_roles(
                sentiment_scopes
            )
        )

        # =====================================================
        # STEP 6
        # APPLY EXACTLY THE SAME DEPENDENCY COMPOSITION
        # USED BY H->D
        # =====================================================

        resolved_scopes = []

        for scope in sentiment_scopes:

            resolved_scope = (
                HisNetDependencyClassifier
                ._apply_dependency_scope(
                    scope
                )
            )

            resolved_scopes.append(
                resolved_scope
            )

        # =====================================================
        # STEP 7
        # DOCUMENT-LEVEL SCORE AGGREGATION
        # =====================================================

        positive_score = sum(
            scope[
                "adjusted_positive"
            ]
            for scope
            in resolved_scopes
        )

        negative_score = sum(
            scope[
                "adjusted_negative"
            ]
            for scope
            in resolved_scopes
        )

        net_score = (
            positive_score
            - negative_score
        )

        prediction = (
            HisNetClassifier
            ._prediction_from_scores(
                positive_score,
                negative_score
            )
        )

        # =====================================================
        # STEP 8
        # DIAGNOSTICS / COVERAGE
        # =====================================================

        token_count = len(
            morphological_analyses
        )

        dependency_candidate_count = len(
            dependency_candidates
        )

        sentiment_candidate_count = len(
            sentiment_scopes
        )

        dependency_candidate_coverage = (
            dependency_candidate_count
            / token_count
            if token_count > 0
            else 0.0
        )

        candidate_lexicon_coverage = (
            lexicon_matched_candidate_count
            / dependency_candidate_count
            if dependency_candidate_count > 0
            else 0.0
        )

        sentiment_coverage = (
            sentiment_candidate_count
            / token_count
            if token_count > 0
            else 0.0
        )

        operations = []

        for scope in resolved_scopes:

            operations.extend(
                scope.get(
                    "operations",
                    []
                )
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
                net_score,

            "neutral_threshold":
                NEUTRAL_THRESHOLD,

            # ---------------------------------------------
            # Dependency information
            # ---------------------------------------------

            "parsed_sentences":
                parsed_sentences,

            "dependency_tokens":
                dependency_tokens,

            # ---------------------------------------------
            # Dependency-first selection
            # ---------------------------------------------

            "dependency_candidates":
                dependency_candidates,

            "candidate_diagnostics":
                candidate_diagnostics,

            # ---------------------------------------------
            # Final sentiment-bearing scopes
            # ---------------------------------------------

            "scopes":
                resolved_scopes,

            "sentiment_words":
                resolved_scopes,

            # ---------------------------------------------
            # Morphology
            # ---------------------------------------------

            "morphological_analyses":
                morphological_analyses,

            # ---------------------------------------------
            # Counts
            # ---------------------------------------------

            "token_count":
                token_count,

            "dependency_candidate_count":
                dependency_candidate_count,

            "lexicon_matched_candidate_count":
                lexicon_matched_candidate_count,

            "sentiment_word_count":
                sentiment_candidate_count,

            # ---------------------------------------------
            # Coverage
            # ---------------------------------------------

            "dependency_candidate_coverage":
                dependency_candidate_coverage,

            "candidate_lexicon_coverage":
                candidate_lexicon_coverage,

            "sentiment_coverage":
                sentiment_coverage,

            # ---------------------------------------------
            # Operations
            # ---------------------------------------------

            "operations":
                operations
        }