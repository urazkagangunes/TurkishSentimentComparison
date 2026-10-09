import copy

from src.hisnet.hisnet_dependency_classifier import (
    HisNetDependencyClassifier
)


class LiteratureWeightedHisNetDependencyClassifier(
    HisNetDependencyClassifier
):
    """
    Literature-weighted HisNet -> Dependency classifier.

    Architecture
    ------------
    Text
        -> morphology
        -> HisNet
        -> dependency parsing
        -> target suppression
        -> negation
        -> literature-based modifier weighting
        -> literature-based contrast weighting
        -> same HisNet decision threshold

    Important experimental principle
    --------------------------------
    The frozen HisNet -> Dependency system is NOT modified.

    This class first obtains the normal H -> D result and then
    adds only literature-derived composition weights.

    Therefore:

        H -> D
    versus
        H -> D -> Literature Weighted

    is a clean ablation.

    No coefficient is tuned on TRSAv1.
    """

    # =========================================================
    # LITERATURE-BASED MODIFIER COEFFICIENTS
    #
    # Multiword expressions are checked longest-first.
    # =========================================================

    MODIFIER_WEIGHTS = {

        # -----------------------------------------------------
        # x1.8
        # -----------------------------------------------------

        ("haddinden", "fazla"):
            1.8,

        ("son", "derece"):
            1.8,

        ("aşırı",):
            1.8,

        ("derin",):
            1.8,

        # -----------------------------------------------------
        # x1.6
        # -----------------------------------------------------

        ("çok", "daha", "fazla"):
            1.6,

        ("çok", "çok", "fazla"):
            1.6,

        ("daha", "da", "çok"):
            1.6,

        # -----------------------------------------------------
        # x1.4
        # -----------------------------------------------------

        ("daha", "çok"):
            1.4,

        ("pek", "çok"):
            1.4,

        ("çok", "fazla"):
            1.4,

        ("pek", "fazla"):
            1.4,

        ("çok", "ama", "çok"):
            1.4,

        # -----------------------------------------------------
        # x1.2
        # -----------------------------------------------------

        ("çok",):
            1.2,

        ("pek",):
            1.2,

        ("büyük",):
            1.2,

        ("en",):
            1.2,

        # -----------------------------------------------------
        # x0.8
        # -----------------------------------------------------

        ("az",):
            0.8,

        ("biraz",):
            0.8,

        ("küçük",):
            0.8,

        # -----------------------------------------------------
        # x0.6
        # -----------------------------------------------------

        ("çok", "az"):
            0.6,

        ("daha", "az"):
            0.6,

        ("pek", "az"):
            0.6,

        # -----------------------------------------------------
        # x0.4
        # -----------------------------------------------------

        ("daha", "da", "az"):
            0.4,

        ("çok", "daha", "az"):
            0.4,

        # -----------------------------------------------------
        # x0.2
        # -----------------------------------------------------

        ("azıcık",):
            0.2,

        ("minicik",):
            0.2
    }

    # =========================================================
    # LITERATURE-BASED CONTRAST WEIGHTS
    #
    # Sentiment BEFORE adversative conjunction:
    #
    #     x0.85
    #
    # Sentiment AFTER adversative conjunction:
    #
    #     x1.85
    #
    # =========================================================

    CONTRAST_PRE_WEIGHT = 0.85

    CONTRAST_POST_WEIGHT = 1.85

    # =========================================================
    # TURKISH CONTRAST MARKERS
    # =========================================================

    CONTRAST_WORDS = {

        "ama",
        "fakat",
        "ancak",
        "lakin",
        "lâkin",
        "oysa",
        "oysaki",
        "rağmen"
    }

    # =========================================================
    # TURKISH LOWERCASE
    # =========================================================

    @staticmethod
    def _turkish_lower(
        text
    ):

        if text is None:

            return ""

        return (
            str(text)
            .replace(
                "I",
                "ı"
            )
            .replace(
                "İ",
                "i"
            )
            .lower()
            .strip()
        )

    # =========================================================
    # SENTENCE / TOKEN INDEX
    # =========================================================

    @staticmethod
    def _build_token_sentence_index(
        sentence_results
    ):
        """
        Returns:

            token_to_sentence
            sentence_tokens

        token_to_sentence:
            global_token_id -> sentence_index

        sentence_tokens:
            sentence_index -> ordered dependency token list
        """

        token_to_sentence = {}

        sentence_tokens = {}

        for sentence_result in sentence_results:

            sentence_index = (
                sentence_result[
                    "sentence_index"
                ]
            )

            tokens = sorted(

                sentence_result.get(
                    "dependency_tokens",
                    []
                ),

                key=lambda token: (
                    int(
                        token[
                            "id"
                        ]
                    )
                )
            )

            sentence_tokens[
                sentence_index
            ] = tokens

            for token in tokens:

                token_to_sentence[
                    int(
                        token[
                            "id"
                        ]
                    )
                ] = sentence_index

        return (
            token_to_sentence,
            sentence_tokens
        )

    # =========================================================
    # FIND TOKEN POSITION
    # =========================================================

    @staticmethod
    def _find_token_position(
        tokens,
        token_id
    ):

        for index, token in enumerate(
            tokens
        ):

            if (
                int(
                    token[
                        "id"
                    ]
                )
                == int(
                    token_id
                )
            ):

                return index

        return None

    # =========================================================
    # MODIFIER MATCHING
    # =========================================================

    def _find_literature_modifier(
        self,
        scope,
        sentence_tokens
    ):
        """
        Detect literature-defined modifier expressions.

        IMPORTANT DISTINCTION
        ---------------------

        Multiword expressions:

            son derece güzel
            çok daha kötü
            daha da iyi
            çok ama çok güzel

        are accepted through longest contiguous phrase matching
        immediately before the sentiment expression.

        This is necessary because a dependency parser does not
        always attach every component of a fixed multiword
        intensifier directly to the sentiment head.

        Single-word modifiers:

            çok güzel
            biraz kötü
            aşırı pahalı

        require dependency evidence from the existing
        SentimentScopeResolver.

        This prevents keyword-only errors such as:

            Büyük ekran güzel.

        where "büyük" should not automatically intensify
        "güzel".
        """

        sentiment_token = (
            scope.get(
                "sentiment_token"
            )
        )

        if sentiment_token is None:

            return {
                "multiplier":
                    1.0,

                "phrase":
                    None,

                "token_ids":
                    [],

                "strategy":
                    None
            }

        sentiment_id = int(
            sentiment_token[
                "id"
            ]
        )

        token_position = (
            self._find_token_position(
                sentence_tokens,
                sentiment_id
            )
        )

        if token_position is None:

            return {
                "multiplier":
                    1.0,

                "phrase":
                    None,

                "token_ids":
                    [],

                "strategy":
                    None
            }

        # -----------------------------------------------------
        # Dependency-resolved modifiers from the frozen H -> D
        # system.
        # -----------------------------------------------------

        modifiers = (
            scope.get(
                "modifiers",
                []
            )
            or []
        )

        modifier_ids = {

            int(
                modifier[
                    "id"
                ]
            )

            for modifier in modifiers

            if modifier.get(
                "id"
            ) is not None
        }

        # -----------------------------------------------------
        # Longest-match-first.
        # -----------------------------------------------------

        phrases = sorted(

            self.MODIFIER_WEIGHTS.items(),

            key=lambda item: (
                len(
                    item[
                        0
                    ]
                )
            ),

            reverse=True
        )

        for (
            phrase_tokens,
            multiplier
        ) in phrases:

            phrase_length = (
                len(
                    phrase_tokens
                )
            )

            start_position = (
                token_position
                - phrase_length
            )

            if start_position < 0:

                continue

            candidate_tokens = (
                sentence_tokens[
                    start_position:
                    token_position
                ]
            )

            if (
                len(
                    candidate_tokens
                )
                != phrase_length
            ):

                continue

            candidate_words = tuple(

                self._turkish_lower(
                    token.get(
                        "text"
                    )
                )

                for token
                in candidate_tokens
            )

            if (
                candidate_words
                != phrase_tokens
            ):

                continue

            candidate_ids = [

                int(
                    token[
                        "id"
                    ]
                )

                for token
                in candidate_tokens
            ]

            # =================================================
            # MULTIWORD MODIFIER
            #
            # Fixed expression from the literature table.
            #
            # Contiguous occurrence immediately before the
            # sentiment expression is sufficient.
            # =================================================

            if phrase_length > 1:

                return {
                    "multiplier":
                        float(
                            multiplier
                        ),

                    "phrase":
                        " ".join(
                            phrase_tokens
                        ),

                    "token_ids":
                        candidate_ids,

                    "strategy":
                        "multiword_literature_phrase"
                }

            # =================================================
            # SINGLE-WORD MODIFIER
            #
            # Dependency evidence is mandatory.
            # =================================================

            candidate_id = (
                candidate_ids[
                    0
                ]
            )

            if (
                candidate_id
                not in modifier_ids
            ):

                continue

            return {
                "multiplier":
                    float(
                        multiplier
                    ),

                "phrase":
                    " ".join(
                        phrase_tokens
                    ),

                "token_ids":
                    candidate_ids,

                "strategy":
                    "dependency_modifier"
            }

        return {
            "multiplier":
                1.0,

            "phrase":
                None,

            "token_ids":
                [],

            "strategy":
                None
        }

    # =========================================================
    # DEPENDENCY CHILD INDEX
    # =========================================================

    @staticmethod
    def _build_children(
        tokens
    ):

        children = {}

        for token in tokens:

            head = int(
                token.get(
                    "head",
                    0
                )
                or 0
            )

            children.setdefault(
                head,
                []
            ).append(
                token
            )

        return children

    # =========================================================
    # DEPENDENCY SUBTREE
    # =========================================================

    @staticmethod
    def _collect_subtree_ids(
        root_id,
        children
    ):

        result = set()

        stack = [
            int(
                root_id
            )
        ]

        while stack:

            current = (
                stack.pop()
            )

            if current in result:

                continue

            result.add(
                current
            )

            for child in children.get(
                current,
                []
            ):

                stack.append(
                    int(
                        child[
                            "id"
                        ]
                    )
                )

        return result

    # =========================================================
    # LICENSED CONTRAST MARKERS
    # =========================================================

    def _find_licensed_contrast_markers(
        self,
        sentence_tokens,
        excluded_marker_ids=None
    ):
        """
        Returns only contrast markers supported by dependency
        output.

        Text occurrence alone is NOT enough.

        Accepted dependency relations:

            cc
            mark
            case

        This gives us a conservative fallback when the parser
        detects the conjunction but fails to produce a complete
        `conj` structure.
        """

        excluded_marker_ids = set(
            excluded_marker_ids
            or []
        )

        licensed_markers = []

        for token in sentence_tokens:

            token_id = int(
                token[
                    "id"
                ]
            )

            if token_id in excluded_marker_ids:

                continue

            word = (
                self._turkish_lower(
                    token.get(
                        "lemma"
                    )
                    or token.get(
                        "text"
                    )
                )
            )

            if (
                word
                not in self.CONTRAST_WORDS
            ):

                continue

            deprel = (
                token.get(
                    "deprel"
                )
                or ""
            )

            if deprel not in {
                "cc",
                "mark",
                "case"
            }:

                continue

            licensed_markers.append({

                "id":
                    token_id,

                "word":
                    token.get(
                        "text"
                    ),

                "lemma":
                    word,

                "deprel":
                    deprel,

                "head":
                    int(
                        token.get(
                            "head",
                            0
                        )
                        or 0
                    )
            })

        return licensed_markers

    # =========================================================
    # FULL DEPENDENCY CONTRAST RELATIONS
    # =========================================================

    def _build_contrast_relations(
        self,
        sentence_tokens,
        excluded_marker_ids=None
    ):
        """
        Try to reconstruct a proper dependency-licensed
        contrast pair.

        Typical UD-style structure:

            güzel
              |
              +---- kötü [conj]
                      |
                      +---- ama [cc]

        The `conj` node is considered the post-contrast clause
        head.

        Its parent is considered the pre-contrast clause head.

        If this complete structure cannot be reconstructed,
        the classifier later uses a conservative positional
        fallback, but ONLY for a dependency-licensed contrast
        marker.
        """

        excluded_marker_ids = set(
            excluded_marker_ids
            or []
        )

        token_by_id = {

            int(
                token[
                    "id"
                ]
            ):
                token

            for token
            in sentence_tokens
        }

        children = (
            self._build_children(
                sentence_tokens
            )
        )

        relations = []

        licensed_markers = (
            self._find_licensed_contrast_markers(
                sentence_tokens,
                excluded_marker_ids=
                    excluded_marker_ids
            )
        )

        for marker in licensed_markers:

            marker_id = (
                marker[
                    "id"
                ]
            )

            anchor_id = (
                marker[
                    "head"
                ]
            )

            current = (
                token_by_id.get(
                    anchor_id
                )
            )

            visited = set()

            post_head_id = None

            pre_head_id = None

            # -------------------------------------------------
            # Walk upward until a conj node is found.
            # -------------------------------------------------

            while (
                current is not None
            ):

                current_id = int(
                    current[
                        "id"
                    ]
                )

                if current_id in visited:

                    break

                visited.add(
                    current_id
                )

                current_deprel = (
                    current.get(
                        "deprel"
                    )
                    or ""
                )

                current_head = int(
                    current.get(
                        "head",
                        0
                    )
                    or 0
                )

                if (
                    current_deprel
                    == "conj"
                    and current_head != 0
                ):

                    post_head_id = (
                        current_id
                    )

                    pre_head_id = (
                        current_head
                    )

                    break

                if current_head == 0:

                    break

                current = (
                    token_by_id.get(
                        current_head
                    )
                )

            if (
                post_head_id is None
                or pre_head_id is None
            ):

                continue

            post_subtree = (
                self._collect_subtree_ids(
                    post_head_id,
                    children
                )
            )

            pre_subtree = (
                self._collect_subtree_ids(
                    pre_head_id,
                    children
                )
            )

            relations.append({

                "marker_id":
                    marker_id,

                "marker":
                    marker[
                        "word"
                    ],

                "marker_lemma":
                    marker[
                        "lemma"
                    ],

                "marker_deprel":
                    marker[
                        "deprel"
                    ],

                "pre_head_id":
                    pre_head_id,

                "post_head_id":
                    post_head_id,

                "pre_subtree":
                    pre_subtree,

                "post_subtree":
                    post_subtree
            })

        return relations

    # =========================================================
    # CONTRAST WEIGHT FOR ONE SENTIMENT SCOPE
    # =========================================================

    def _find_contrast_weight(
        self,
        scope,
        contrast_relations,
        licensed_markers,
        excluded_marker_ids=None
    ):
        """
        Priority:

        1. Full dependency contrast structure
        2. Existing H -> D resolver contrast association
        3. Dependency-licensed positional fallback
        4. No weighting

        Positional fallback is NOT a raw keyword rule.

        A marker must first be identified by the parser as:

            cc
            mark
            case
        """

        excluded_marker_ids = set(
            excluded_marker_ids
            or []
        )

        sentiment_token = (
            scope.get(
                "sentiment_token"
            )
        )

        if sentiment_token is None:

            return {
                "multiplier":
                    1.0,

                "role":
                    None,

                "marker":
                    None,

                "marker_id":
                    None,

                "strategy":
                    None
            }

        sentiment_id = int(
            sentiment_token[
                "id"
            ]
        )

        # =====================================================
        # 1. FULL DEPENDENCY STRUCTURE
        # =====================================================

        candidates = []

        for relation in contrast_relations:

            marker_id = int(
                relation[
                    "marker_id"
                ]
            )

            if marker_id in excluded_marker_ids:

                continue

            post_subtree = (
                relation[
                    "post_subtree"
                ]
            )

            pre_subtree = (
                relation[
                    "pre_subtree"
                ]
            )

            # -------------------------------------------------
            # POST must be checked first.
            #
            # The pre-head subtree can contain the post branch.
            # -------------------------------------------------

            if sentiment_id in post_subtree:

                candidates.append({

                    "distance":
                        abs(
                            sentiment_id
                            - marker_id
                        ),

                    "multiplier":
                        self.CONTRAST_POST_WEIGHT,

                    "role":
                        "post_contrast",

                    "marker":
                        relation[
                            "marker"
                        ],

                    "marker_id":
                        marker_id,

                    "strategy":
                        "dependency_subtree"
                })

                continue

            if sentiment_id in pre_subtree:

                candidates.append({

                    "distance":
                        abs(
                            sentiment_id
                            - marker_id
                        ),

                    "multiplier":
                        self.CONTRAST_PRE_WEIGHT,

                    "role":
                        "pre_contrast",

                    "marker":
                        relation[
                            "marker"
                        ],

                    "marker_id":
                        marker_id,

                    "strategy":
                        "dependency_subtree"
                })

        if candidates:

            best = min(

                candidates,

                key=lambda item: (
                    item[
                        "distance"
                    ]
                )
            )

            return {
                "multiplier":
                    float(
                        best[
                            "multiplier"
                        ]
                    ),

                "role":
                    best[
                        "role"
                    ],

                "marker":
                    best[
                        "marker"
                    ],

                "marker_id":
                    best[
                        "marker_id"
                    ],

                "strategy":
                    best[
                        "strategy"
                    ]
            }

        # =====================================================
        # 2. EXISTING RESOLVER ASSOCIATION
        # =====================================================

        existing_contrast = (
            scope.get(
                "contrast"
            )
        )

        if existing_contrast is not None:

            existing_marker_id = (
                existing_contrast.get(
                    "id"
                )
            )

            if existing_marker_id is not None:

                existing_marker_id = int(
                    existing_marker_id
                )

                licensed_ids = {

                    int(
                        marker[
                            "id"
                        ]
                    )

                    for marker
                    in licensed_markers
                }

                if (
                    existing_marker_id
                    in licensed_ids
                    and existing_marker_id
                    not in excluded_marker_ids
                ):

                    marker_word = (
                        existing_contrast.get(
                            "word"
                        )
                    )

                    if (
                        sentiment_id
                        < existing_marker_id
                    ):

                        return {
                            "multiplier":
                                self.CONTRAST_PRE_WEIGHT,

                            "role":
                                "pre_contrast_resolver",

                            "marker":
                                marker_word,

                            "marker_id":
                                existing_marker_id,

                            "strategy":
                                "resolver_association"
                        }

                    if (
                        sentiment_id
                        > existing_marker_id
                    ):

                        return {
                            "multiplier":
                                self.CONTRAST_POST_WEIGHT,

                            "role":
                                "post_contrast_resolver",

                            "marker":
                                marker_word,

                            "marker_id":
                                existing_marker_id,

                            "strategy":
                                "resolver_association"
                        }

        # =====================================================
        # 3. DEPENDENCY-LICENSED POSITIONAL FALLBACK
        #
        # This fixes cases where Stanza correctly identifies:
        #
        #     ama -> cc
        #
        # but does not construct the conj tree in the form our
        # relation reconstruction expects.
        # =====================================================

        positional_candidates = []

        for marker in licensed_markers:

            marker_id = int(
                marker[
                    "id"
                ]
            )

            if marker_id in excluded_marker_ids:

                continue

            if sentiment_id == marker_id:

                continue

            distance = abs(
                sentiment_id
                - marker_id
            )

            if sentiment_id < marker_id:

                positional_candidates.append({

                    "distance":
                        distance,

                    "multiplier":
                        self.CONTRAST_PRE_WEIGHT,

                    "role":
                        "pre_contrast_fallback",

                    "marker":
                        marker[
                            "word"
                        ],

                    "marker_id":
                        marker_id,

                    "strategy":
                        "dependency_licensed_position"
                })

            elif sentiment_id > marker_id:

                positional_candidates.append({

                    "distance":
                        distance,

                    "multiplier":
                        self.CONTRAST_POST_WEIGHT,

                    "role":
                        "post_contrast_fallback",

                    "marker":
                        marker[
                            "word"
                        ],

                    "marker_id":
                        marker_id,

                    "strategy":
                        "dependency_licensed_position"
                })

        if positional_candidates:

            best = min(

                positional_candidates,

                key=lambda item: (
                    item[
                        "distance"
                    ]
                )
            )

            return {
                "multiplier":
                    float(
                        best[
                            "multiplier"
                        ]
                    ),

                "role":
                    best[
                        "role"
                    ],

                "marker":
                    best[
                        "marker"
                    ],

                "marker_id":
                    best[
                        "marker_id"
                    ],

                "strategy":
                    best[
                        "strategy"
                    ]
            }

        # =====================================================
        # NO CONTRAST
        # =====================================================

        return {
            "multiplier":
                1.0,

            "role":
                None,

            "marker":
                None,

            "marker_id":
                None,

            "strategy":
                None
        }

    # =========================================================
    # REBUILD SENTENCE DIAGNOSTICS
    # =========================================================

    def _rebuild_weighted_sentences(
        self,
        original_sentences,
        weighted_scopes
    ):

        sentence_results = []

        token_to_sentence = {}

        for sentence in original_sentences:

            sentence_index = (
                sentence[
                    "sentence_index"
                ]
            )

            dependency_tokens = (
                copy.deepcopy(
                    sentence.get(
                        "dependency_tokens",
                        []
                    )
                )
            )

            sentence_results.append({

                "sentence_index":
                    sentence_index,

                "text":
                    sentence.get(
                        "text"
                    ),

                "dependency_tokens":
                    dependency_tokens,

                "scopes":
                    []
            })

            for token in dependency_tokens:

                token_to_sentence[
                    int(
                        token[
                            "id"
                        ]
                    )
                ] = sentence_index

        return (
            self._build_sentence_results(
                sentence_results,
                weighted_scopes,
                token_to_sentence
            )
        )

    # =========================================================
    # CLASSIFY
    # =========================================================

    def classify(
        self,
        text,
        hisnet_result=None
    ):
        """
        Pipeline:

        1. Run the frozen H -> D classifier.
        2. Detect literature-defined modifiers.
        3. Suppress accidental HisNet sentiment scores belonging
           to modifier-expression components.
        4. Reconstruct dependency contrast structures.
        5. Use dependency-licensed positional fallback when
           necessary.
        6. Apply literature weights.
        7. Sum scores.
        8. Apply the SAME original decision threshold.
        """

        # =====================================================
        # BASELINE H -> D
        # =====================================================

        base_result = (
            super()
            .classify(
                text,
                hisnet_result=
                    hisnet_result
            )
        )

        weighted_scopes = (
            copy.deepcopy(
                base_result[
                    "scopes"
                ]
            )
        )

        (
            token_to_sentence,
            sentence_tokens
        ) = (
            self._build_token_sentence_index(
                base_result[
                    "sentences"
                ]
            )
        )

        # =====================================================
        # STEP 1
        # DETECT MODIFIER PHRASES
        # =====================================================

        modifier_component_ids = set()

        modifier_component_info = {}

        excluded_contrast_marker_ids = set()

        for scope in weighted_scopes:

            sentiment_token = (
                scope.get(
                    "sentiment_token"
                )
            )

            if sentiment_token is None:

                scope[
                    "modifier_multiplier"
                ] = 1.0

                scope[
                    "modifier_phrase"
                ] = None

                scope[
                    "modifier_token_ids"
                ] = []

                scope[
                    "modifier_strategy"
                ] = None

                continue

            sentiment_id = int(
                sentiment_token[
                    "id"
                ]
            )

            sentence_index = (
                token_to_sentence.get(
                    sentiment_id
                )
            )

            tokens = (
                sentence_tokens.get(
                    sentence_index,
                    []
                )
            )

            modifier_result = (
                self._find_literature_modifier(
                    scope,
                    tokens
                )
            )

            scope[
                "modifier_multiplier"
            ] = (
                modifier_result[
                    "multiplier"
                ]
            )

            scope[
                "modifier_phrase"
            ] = (
                modifier_result[
                    "phrase"
                ]
            )

            scope[
                "modifier_token_ids"
            ] = (
                modifier_result[
                    "token_ids"
                ]
            )

            scope[
                "modifier_strategy"
            ] = (
                modifier_result[
                    "strategy"
                ]
            )

            # =================================================
            # REGISTER COMPONENT TOKENS
            #
            # Example:
            #
            #     son derece güzel
            #
            # If HisNet independently assigns sentiment to
            # "son", that lexical score must not be counted.
            #
            # "son" is functioning as part of the modifier
            # expression "son derece".
            # =================================================

            if (
                modifier_result[
                    "phrase"
                ]
                is not None
            ):

                for component_id in modifier_result[
                    "token_ids"
                ]:

                    modifier_component_ids.add(
                        component_id
                    )

                    modifier_component_info[
                        component_id
                    ] = {

                        "phrase":
                            modifier_result[
                                "phrase"
                            ],

                        "target_sentiment_id":
                            sentiment_id,

                        "multiplier":
                            modifier_result[
                                "multiplier"
                            ]
                    }

            # =================================================
            # "çok ama çok"
            #
            # The "ama" is part of one modifier expression.
            # It must NOT simultaneously be interpreted as
            # discourse contrast.
            # =================================================

            if (
                modifier_result[
                    "phrase"
                ]
                == "çok ama çok"
            ):

                for token_id in modifier_result[
                    "token_ids"
                ]:

                    for token in tokens:

                        if (
                            int(
                                token[
                                    "id"
                                ]
                            )
                            != int(
                                token_id
                            )
                        ):

                            continue

                        token_word = (
                            self._turkish_lower(
                                token.get(
                                    "text"
                                )
                            )
                        )

                        if token_word == "ama":

                            excluded_contrast_marker_ids.add(
                                int(
                                    token_id
                                )
                            )

                            operations = list(
                                scope.get(
                                    "operations",
                                    []
                                )
                            )

                            operations.append(
                                "contrast_excluded_as_modifier_component"
                            )

                            scope[
                                "operations"
                            ] = operations

        # =====================================================
        # STEP 2
        # SUPPRESS HISNET SENTIMENT SCOPES THAT ARE ACTUALLY
        # PART OF A MODIFIER EXPRESSION
        #
        # Example:
        #
        #     son derece güzel
        #
        # HisNet may produce:
        #
        #     son   -> negative = .125
        #     güzel -> positive = .4545
        #
        # But "son" belongs to "son derece", so:
        #
        #     son -> 0
        #     güzel -> .4545 * 1.8
        # =====================================================

        modifier_component_suppressed_count = 0

        for scope in weighted_scopes:

            sentiment_token = (
                scope.get(
                    "sentiment_token"
                )
            )

            if sentiment_token is None:

                continue

            sentiment_id = int(
                sentiment_token[
                    "id"
                ]
            )

            if (
                sentiment_id
                not in modifier_component_ids
            ):

                continue

            component_info = (
                modifier_component_info[
                    sentiment_id
                ]
            )

            # -------------------------------------------------
            # Safety:
            #
            # Do not suppress the sentiment expression being
            # modified itself.
            # -------------------------------------------------

            if (
                sentiment_id
                == int(
                    component_info[
                        "target_sentiment_id"
                    ]
                )
            ):

                continue

            modifier_component_suppressed_count += 1

            scope[
                "suppressed_as_modifier_component"
            ] = True

            scope[
                "modifier_component_of"
            ] = (
                component_info[
                    "phrase"
                ]
            )

            scope[
                "active_sentiment"
            ] = False

            # Keep the pre-suppression scores for diagnostics.
            scope[
                "pre_modifier_component_positive"
            ] = float(
                scope.get(
                    "adjusted_positive",
                    0.0
                )
            )

            scope[
                "pre_modifier_component_negative"
            ] = float(
                scope.get(
                    "adjusted_negative",
                    0.0
                )
            )

            scope[
                "adjusted_positive"
            ] = 0.0

            scope[
                "adjusted_negative"
            ] = 0.0

            operations = list(
                scope.get(
                    "operations",
                    []
                )
            )

            operations.append(
                "suppressed_as_modifier_component"
            )

            scope[
                "operations"
            ] = operations

        # =====================================================
        # STEP 3
        # BUILD CONTRAST STRUCTURES AND LICENSED MARKERS
        # =====================================================

        contrast_relations_by_sentence = {}

        licensed_contrast_markers_by_sentence = {}

        for (
            sentence_index,
            tokens
        ) in sentence_tokens.items():

            licensed_markers = (
                self._find_licensed_contrast_markers(
                    tokens,
                    excluded_marker_ids=
                        excluded_contrast_marker_ids
                )
            )

            relations = (
                self._build_contrast_relations(
                    tokens,
                    excluded_marker_ids=
                        excluded_contrast_marker_ids
                )
            )

            licensed_contrast_markers_by_sentence[
                sentence_index
            ] = licensed_markers

            contrast_relations_by_sentence[
                sentence_index
            ] = relations

        # ============================================================
        # STEP 3.5
        # SUPPRESS LEXICAL SENTIMENT SCORES OF SYNTACTICALLY
        # LICENSED CONTRAST MARKERS
        #
        # Example:
        #
        #     Otel kötü ancak odalar güzel.
        #
        # "ancak" may accidentally have a HisNet sentiment score.
        # However, when Stanza identifies it as a contrast marker,
        # it functions as discourse structure rather than as an
        # opinion expression.
        # ============================================================

        licensed_contrast_marker_ids = set()

        for markers in (
                licensed_contrast_markers_by_sentence.values()
        ):

            for marker in markers:
                licensed_contrast_marker_ids.add(
                    int(
                        marker[
                            "id"
                        ]
                    )
                )

        contrast_marker_suppressed_count = 0

        for scope in weighted_scopes:

            sentiment_token = (
                scope.get(
                    "sentiment_token"
                )
            )

            if sentiment_token is None:
                continue

            sentiment_id = int(
                sentiment_token[
                    "id"
                ]
            )

            if (
                    sentiment_id
                    not in licensed_contrast_marker_ids
            ):
                continue

            contrast_marker_suppressed_count += 1

            scope[
                "suppressed_as_contrast_marker"
            ] = True

            scope[
                "active_sentiment"
            ] = False

            scope[
                "pre_contrast_marker_positive"
            ] = float(
                scope.get(
                    "adjusted_positive",
                    0.0
                )
            )

            scope[
                "pre_contrast_marker_negative"
            ] = float(
                scope.get(
                    "adjusted_negative",
                    0.0
                )
            )

            scope[
                "adjusted_positive"
            ] = 0.0

            scope[
                "adjusted_negative"
            ] = 0.0

            operations = list(
                scope.get(
                    "operations",
                    []
                )
            )

            operations.append(
                "suppressed_as_contrast_marker"
            )

            scope[
                "operations"
            ] = operations

        # =====================================================
        # STEP 4
        # APPLY LITERATURE WEIGHTS
        #
        # Frozen H -> D has already performed:
        #
        #     target suppression
        #     negation composition
        #
        # We now alter only the magnitude.
        # =====================================================

        modifier_weighted_count = 0

        contrast_weighted_count = 0

        for scope in weighted_scopes:

            base_positive = float(
                scope.get(
                    "adjusted_positive",
                    0.0
                )
            )

            base_negative = float(
                scope.get(
                    "adjusted_negative",
                    0.0
                )
            )

            # -------------------------------------------------
            # Preserve H -> D result before literature weights.
            # -------------------------------------------------

            scope[
                "dependency_adjusted_positive"
            ] = base_positive

            scope[
                "dependency_adjusted_negative"
            ] = base_negative

            sentiment_token = (
                scope.get(
                    "sentiment_token"
                )
            )

            # =================================================
            # INACTIVE SCOPE
            #
            # Could be:
            #
            #     target suppression
            #     modifier-component suppression
            # =================================================

            if (
                not scope.get(
                    "active_sentiment",
                    True
                )
            ):

                scope[
                    "contrast_multiplier"
                ] = 1.0

                scope[
                    "contrast_role"
                ] = None

                scope[
                    "contrast_marker"
                ] = None

                scope[
                    "contrast_marker_id"
                ] = None

                scope[
                    "contrast_strategy"
                ] = None

                scope[
                    "literature_multiplier"
                ] = 1.0

                scope[
                    "weighted_positive"
                ] = 0.0

                scope[
                    "weighted_negative"
                ] = 0.0

                scope[
                    "adjusted_positive"
                ] = 0.0

                scope[
                    "adjusted_negative"
                ] = 0.0

                continue

            # =================================================
            # MODIFIER
            # =================================================

            modifier_multiplier = float(
                scope.get(
                    "modifier_multiplier",
                    1.0
                )
            )

            if (
                modifier_multiplier
                != 1.0
            ):

                modifier_weighted_count += 1

            # =================================================
            # CONTRAST
            # =================================================

            if sentiment_token is not None:

                sentiment_id = int(
                    sentiment_token[
                        "id"
                    ]
                )

                sentence_index = (
                    token_to_sentence.get(
                        sentiment_id
                    )
                )

                contrast_relations = (
                    contrast_relations_by_sentence.get(
                        sentence_index,
                        []
                    )
                )

                licensed_markers = (
                    licensed_contrast_markers_by_sentence.get(
                        sentence_index,
                        []
                    )
                )

            else:

                contrast_relations = []

                licensed_markers = []

            contrast_result = (
                self._find_contrast_weight(

                    scope,

                    contrast_relations,

                    licensed_markers,

                    excluded_marker_ids=
                        excluded_contrast_marker_ids
                )
            )

            contrast_multiplier = float(
                contrast_result[
                    "multiplier"
                ]
            )

            if (
                contrast_multiplier
                != 1.0
            ):

                contrast_weighted_count += 1

            scope[
                "contrast_multiplier"
            ] = (
                contrast_multiplier
            )

            scope[
                "contrast_role"
            ] = (
                contrast_result[
                    "role"
                ]
            )

            scope[
                "contrast_marker"
            ] = (
                contrast_result[
                    "marker"
                ]
            )

            scope[
                "contrast_marker_id"
            ] = (
                contrast_result[
                    "marker_id"
                ]
            )

            scope[
                "contrast_strategy"
            ] = (
                contrast_result[
                    "strategy"
                ]
            )

            # =================================================
            # FINAL MULTIPLIER
            # =================================================

            final_multiplier = (
                modifier_multiplier
                * contrast_multiplier
            )

            weighted_positive = (
                base_positive
                * final_multiplier
            )

            weighted_negative = (
                base_negative
                * final_multiplier
            )

            scope[
                "literature_multiplier"
            ] = (
                final_multiplier
            )

            scope[
                "weighted_positive"
            ] = (
                weighted_positive
            )

            scope[
                "weighted_negative"
            ] = (
                weighted_negative
            )

            # -------------------------------------------------
            # adjusted_* is now the final score of THIS
            # classifier.
            # -------------------------------------------------

            scope[
                "adjusted_positive"
            ] = (
                weighted_positive
            )

            scope[
                "adjusted_negative"
            ] = (
                weighted_negative
            )

            # =================================================
            # OPERATIONS
            # =================================================

            operations = list(
                scope.get(
                    "operations",
                    []
                )
            )

            if (
                modifier_multiplier
                != 1.0
            ):

                operations.append(
                    (
                        "literature_modifier_weight:"
                        f"{modifier_multiplier}"
                    )
                )

            if (
                contrast_multiplier
                != 1.0
            ):

                operations.append(
                    (
                        "literature_contrast_weight:"
                        f"{contrast_multiplier}"
                    )
                )

            scope[
                "operations"
            ] = operations

        # =====================================================
        # STEP 5
        # FINAL DOCUMENT SCORE
        # =====================================================

        positive_score = sum(

            float(
                scope.get(
                    "adjusted_positive",
                    0.0
                )
            )

            for scope
            in weighted_scopes
        )

        negative_score = sum(

            float(
                scope.get(
                    "adjusted_negative",
                    0.0
                )
            )

            for scope
            in weighted_scopes
        )

        # =====================================================
        # SAME ORIGINAL HISNET DECISION RULE
        # =====================================================

        prediction = (
            self._prediction_from_scores(
                positive_score,
                negative_score
            )
        )

        # =====================================================
        # STEP 6
        # REBUILD SENTENCE DIAGNOSTICS
        # =====================================================

        weighted_sentences = (
            self._rebuild_weighted_sentences(
                base_result[
                    "sentences"
                ],
                weighted_scopes
            )
        )

        # =====================================================
        # SERIALIZABLE CONTRAST DIAGNOSTICS
        # =====================================================

        contrast_relation_diagnostics = {}

        for (
            sentence_index,
            relations
        ) in (
            contrast_relations_by_sentence.items()
        ):

            contrast_relation_diagnostics[
                sentence_index
            ] = [

                {
                    "marker_id":
                        relation[
                            "marker_id"
                        ],

                    "marker":
                        relation[
                            "marker"
                        ],

                    "marker_deprel":
                        relation[
                            "marker_deprel"
                        ],

                    "pre_head_id":
                        relation[
                            "pre_head_id"
                        ],

                    "post_head_id":
                        relation[
                            "post_head_id"
                        ]
                }

                for relation
                in relations
            ]

        licensed_marker_diagnostics = {}

        for (
            sentence_index,
            markers
        ) in (
            licensed_contrast_markers_by_sentence.items()
        ):

            licensed_marker_diagnostics[
                sentence_index
            ] = [

                {
                    "id":
                        marker[
                            "id"
                        ],

                    "word":
                        marker[
                            "word"
                        ],

                    "lemma":
                        marker[
                            "lemma"
                        ],

                    "deprel":
                        marker[
                            "deprel"
                        ],

                    "head":
                        marker[
                            "head"
                        ]
                }

                for marker
                in markers
            ]

        # =====================================================
        # OUTPUT
        # =====================================================

        return {

            **base_result,

            # -------------------------------------------------
            # Architecture
            # -------------------------------------------------

            "architecture":
                (
                    "hisnet_dependency_"
                    "literature_weighted"
                ),

            # -------------------------------------------------
            # Frozen H -> D result
            # -------------------------------------------------

            "dependency_prediction":
                base_result[
                    "prediction"
                ],

            "dependency_positive_score":
                base_result[
                    "positive_score"
                ],

            "dependency_negative_score":
                base_result[
                    "negative_score"
                ],

            # -------------------------------------------------
            # Final weighted result
            # -------------------------------------------------

            "prediction":
                prediction,

            "positive_score":
                positive_score,

            "negative_score":
                negative_score,

            "scopes":
                weighted_scopes,

            "sentences":
                weighted_sentences,

            # -------------------------------------------------
            # Diagnostics
            # -------------------------------------------------

            "modifier_weighted_scope_count":
                modifier_weighted_count,

            "modifier_component_suppressed_scope_count":
                modifier_component_suppressed_count,

            "contrast_marker_suppressed_scope_count":
                contrast_marker_suppressed_count,

            "contrast_weighted_scope_count":
                contrast_weighted_count,

            "modifier_component_ids":
                sorted(
                    modifier_component_ids
                ),

            "excluded_modifier_contrast_marker_ids":
                sorted(
                    excluded_contrast_marker_ids
                ),

            "licensed_contrast_markers":
                licensed_marker_diagnostics,

            "contrast_relations":
                contrast_relation_diagnostics,

            # -------------------------------------------------
            # Fixed experimental configuration
            # -------------------------------------------------

            "literature_weighting": {

                "modifier_method":
                    "fixed_literature_coefficients",

                "contrast_method":
                    (
                        "dependency-first; "
                        "dependency-licensed positional "
                        "fallback"
                    ),

                "contrast_pre_weight":
                    self.CONTRAST_PRE_WEIGHT,

                "contrast_post_weight":
                    self.CONTRAST_POST_WEIGHT,

                "decision_threshold":
                    0.05,

                "dataset_tuned":
                    False
            }
        }