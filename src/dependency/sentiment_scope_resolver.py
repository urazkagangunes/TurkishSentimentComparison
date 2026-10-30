class SentimentScopeResolver:
    """
    Resolves dependency-based sentiment scope.

    For every HisNet sentiment-bearing word, tries to identify:

        - Which dependency token represents the sentiment expression
        - Which entity/target the sentiment expression affects
        - Which modifiers affect the sentiment expression
        - Which negation markers affect the sentiment expression

    This class does NOT change sentiment scores.
    It only resolves linguistic scope/relations.
    """

    SUBJECT_RELATIONS = {
        "nsubj",
        "nsubj:pass",
        "csubj",
        "csubj:pass"
    }

    OBJECT_RELATIONS = {
        "obj",
        "iobj"
    }

    MODIFIER_RELATIONS = {
        "advmod"
    }

    NEGATION_SUPPORT_RELATIONS = {
        "cop",
        "aux",
        "aux:pass",
        "compound:lvc"
    }

    PARENT_NEGATION_RELATIONS = {
        "compound:lvc"
    }

    ENTITY_POS = {
        "NOUN",
        "PROPN",
        "PRON"
    }

    CONTRAST_WORDS = {
        "ama",
        "fakat",
        "ancak",
        "lakin",
        "lâkin",
        "oysa",
        "oysaki"
    }

    OPINION_POS = {
        "ADJ",
        "VERB",
        "NOUN",
        "ADV",
        "INTJ"
    }

    OPINION_PREDICATE_RELATIONS = {
        "root",
        "conj",
        "xcomp",
        "ccomp",
        "acl",
        "acl:relcl"
    }

    OPINION_EXCLUDED_RELATIONS = {
        "aux",
        "aux:pass",
        "cop",
        "compound:lvc",
        "nmod",
        "det",
        "case",
        "cc",
        "mark",
        "punct"
    }

    @staticmethod
    def _turkish_lower(text):
        if not text:
            return ""

        return (
            text
            .replace("I", "ı")
            .replace("İ", "i")
            .lower()
        )

    def _is_negation(self, token):
        """
        Detects grammatical/morphological negation.

        IMPORTANT:
        Polarity=Neg here means grammatical polarity,
        not negative sentiment.
        """

        lemma = self._turkish_lower(
            token.get("lemma")
        )

        feats = token.get("feats") or ""
        deprel = token.get("deprel") or ""

        return (
            "Polarity=Neg" in feats
            or lemma == "değil"
            or deprel == "neg"
        )

    @staticmethod
    def _has_morphological_negation(token):
        if token is None:
            return False

        feats = token.get("feats") or ""

        return (
            "Polarity=Neg"
            in feats
        )

    def _is_explicit_degil(self, token):
        if token is None:
            return False

        lemma = self._turkish_lower(
            token.get("lemma")
        )

        surface = self._turkish_lower(
            token.get("text")
        )

        return (
            lemma == "değil"
            or surface.startswith(
                "değil"
            )
        )

    @staticmethod
    def _build_indexes(tokens):
        token_by_id = {}
        children = {}

        for token in tokens:

            token_id = token["id"]

            token_by_id[token_id] = token

            head = token.get("head", 0)

            children.setdefault(
                head,
                []
            ).append(token)

        return token_by_id, children

    def _is_entity_candidate(self, token):
        return (
            token is not None
            and token.get("upos") in self.ENTITY_POS
        )

    # =========================================================
    # ALIGNMENT HELPERS
    # =========================================================

    @classmethod
    def _normalize_alignment_piece(
        cls,
        text
    ):
        """
        Normalization used ONLY for fallback alignment.

        Examples:

            ..yumuşak  -> yumuşak
            temizlik👍 -> temizlik

        Normal exact alignment is always attempted before this.
        """

        value = cls._turkish_lower(
            text
        )

        start = 0
        end = len(value)

        while (
            start < end
            and not value[start].isalnum()
        ):
            start += 1

        while (
            end > start
            and not value[end - 1].isalnum()
        ):
            end -= 1

        return value[start:end]

    @classmethod
    def _fragment_anchor_token(
        cls,
        fragment_tokens
    ):
        """
        Selects the dependency-bearing token from a reconstructed
        fragmented word.

        Examples:

            özen + li
            hesap + lı
            harik + a

        Usually the first/content token should remain the dependency
        anchor rather than the suffix-like ADP token.
        """

        fragment_ids = {
            int(token["id"])
            for token in fragment_tokens
        }

        pos_priority = {
            "ADJ": 6,
            "VERB": 6,
            "NOUN": 5,
            "ADV": 5,
            "INTJ": 4,
            "PROPN": 4,
            "PRON": 3,
            "NUM": 2,
            "DET": 1,
            "ADP": 0,
            "AUX": 0,
            "CCONJ": 0,
            "SCONJ": 0,
            "PART": 0,
            "PUNCT": -1
        }

        def ranking(token):
            head = token.get("head")

            try:
                head = int(
                    head or 0
                )
            except (
                TypeError,
                ValueError
            ):
                head = 0

            outside_head = (
                1
                if head not in fragment_ids
                else 0
            )

            return (
                pos_priority.get(
                    token.get("upos"),
                    0
                ),
                outside_head,
                -int(token["id"])
            )

        return max(
            fragment_tokens,
            key=ranking
        )

    def _match_sentiment_token(
        self,
        sentiment_word,
        tokens,
        used_token_ids
    ):
        """
        Aligns a HisNet sentiment word with a Stanza token.

        Priority:

            1. exact surface
            2. token surface == morphology root
            3. Stanza lemma == morphology root
            4. edge-noise fallback
            5. adjacent fragment reconstruction

        Fragment/fuzzy fallback is used ONLY when ordinary alignment
        completely fails.
        """

        surface = self._turkish_lower(
            sentiment_word["word"]
        )

        root = self._turkish_lower(
            sentiment_word["root"]
        )

        # =====================================================
        # 1-3. NORMAL ALIGNMENT
        # =====================================================

        candidates = []

        for token in tokens:

            token_id = int(
                token["id"]
            )

            if token_id in used_token_ids:
                continue

            token_text = self._turkish_lower(
                token.get("text")
            )

            token_lemma = self._turkish_lower(
                token.get("lemma")
            )

            score = 0
            strategy = None

            if token_text == surface:

                score = 100
                strategy = "exact_surface"

            elif token_text == root:

                score = 90
                strategy = (
                    "token_surface_equals_root"
                )

            elif token_lemma == root:

                score = 80
                strategy = (
                    "stanza_lemma_equals_root"
                )

            if score > 0:

                candidates.append(
                    (
                        score,
                        token_id,
                        strategy,
                        token
                    )
                )

        if candidates:

            candidates.sort(
                key=lambda item: (
                    -item[0],
                    item[1]
                )
            )

            (
                _,
                _,
                strategy,
                best_token
            ) = candidates[0]

            used_token_ids.add(
                int(best_token["id"])
            )

            matched_token = dict(
                best_token
            )

            matched_token[
                "_alignment_strategy"
            ] = strategy

            matched_token[
                "_fragment_token_ids"
            ] = [
                int(best_token["id"])
            ]

            return matched_token

        # =====================================================
        # 4. EDGE-NOISE FALLBACK
        # =====================================================

        decorated_candidates = []

        for token in tokens:

            token_id = int(
                token["id"]
            )

            if token_id in used_token_ids:
                continue

            raw_text = self._turkish_lower(
                token.get("text")
            )

            clean_text = (
                self._normalize_alignment_piece(
                    raw_text
                )
            )

            score = 0
            strategy = None

            if clean_text == surface:

                score = 75
                strategy = (
                    "edge_noise_surface"
                )

            elif clean_text == root:

                score = 72
                strategy = (
                    "edge_noise_root"
                )

            elif (
                surface
                and raw_text.startswith(
                    surface
                )
                and len(raw_text)
                > len(surface)
                and not raw_text[
                    len(surface)
                ].isalnum()
            ):

                score = 70
                strategy = (
                    "decorated_surface_prefix"
                )

            elif (
                surface
                and raw_text.endswith(
                    surface
                )
                and len(raw_text)
                > len(surface)
                and not raw_text[
                    -len(surface) - 1
                ].isalnum()
            ):

                score = 70
                strategy = (
                    "decorated_surface_suffix"
                )

            elif (
                root
                and raw_text.startswith(
                    root
                )
                and len(raw_text)
                > len(root)
                and not raw_text[
                    len(root)
                ].isalnum()
            ):

                score = 68
                strategy = (
                    "decorated_root_prefix"
                )

            elif (
                root
                and raw_text.endswith(
                    root
                )
                and len(raw_text)
                > len(root)
                and not raw_text[
                    -len(root) - 1
                ].isalnum()
            ):

                score = 68
                strategy = (
                    "decorated_root_suffix"
                )

            if score > 0:

                decorated_candidates.append(
                    (
                        score,
                        token_id,
                        strategy,
                        token
                    )
                )

        if decorated_candidates:

            decorated_candidates.sort(
                key=lambda item: (
                    -item[0],
                    item[1]
                )
            )

            (
                _,
                _,
                strategy,
                best_token
            ) = decorated_candidates[0]

            used_token_ids.add(
                int(best_token["id"])
            )

            matched_token = dict(
                best_token
            )

            matched_token[
                "_alignment_strategy"
            ] = strategy

            matched_token[
                "_fragment_token_ids"
            ] = [
                int(best_token["id"])
            ]

            return matched_token

        # =====================================================
        # 5. ADJACENT TOKEN FRAGMENT RECONSTRUCTION
        # =====================================================

        ordered_tokens = sorted(
            tokens,
            key=lambda token: int(
                token["id"]
            )
        )

        fragment_candidates = []

        for group_size in (
            2,
            3
        ):

            for start in range(
                0,
                len(ordered_tokens)
                - group_size
                + 1
            ):

                group = ordered_tokens[
                    start:
                    start + group_size
                ]

                group_ids = [
                    int(token["id"])
                    for token in group
                ]

                if any(
                    token_id
                    in used_token_ids
                    for token_id
                    in group_ids
                ):
                    continue

                # Must be consecutive tokens.
                if any(
                    right != left + 1
                    for left, right
                    in zip(
                        group_ids,
                        group_ids[1:]
                    )
                ):
                    continue

                pieces = [
                    self._normalize_alignment_piece(
                        token.get("text")
                    )
                    for token
                    in group
                ]

                if any(
                    not piece
                    for piece in pieces
                ):
                    continue

                reconstructed = "".join(
                    pieces
                )

                score = 0
                strategy = None

                # =========================================================
                # SENTENCE-BOUNDARY SAFETY
                # =========================================================
                #
                # Stanza occasionally splits a single Turkish word into two
                # dependency tokens and incorrectly places the second fragment
                # into a new sentence:
                #
                #     ser + t        -> sert
                #     harik + a      -> harika
                #     basi + t       -> basit
                #     kullanışsı + z -> kullanışsız
                #
                # Cross-sentence reconstruction is therefore allowed only under
                # very strong evidence:
                #
                #   - exactly two fragments
                #   - reconstructed form exactly equals the HisNet SURFACE form
                #   - at least one fragment is very short (<= 2 chars)
                #
                # Root-only cross-sentence reconstruction is intentionally
                # forbidden because it would be less reliable.
                # =========================================================

                sentence_indexes = {
                    token.get(
                        "sentence_index"
                    )
                    for token in group
                    if token.get(
                        "sentence_index"
                    ) is not None
                }

                cross_sentence = (
                        len(
                            sentence_indexes
                        )
                        > 1
                )

                if cross_sentence:

                    short_fragment_exists = any(
                        len(piece) <= 2
                        for piece in pieces
                    )

                    safe_cross_sentence_surface_match = (
                            group_size == 2
                            and reconstructed == surface
                            and short_fragment_exists
                    )

                    if not safe_cross_sentence_surface_match:
                        continue

                # =========================================================
                # ORDINARY / SAME-SENTENCE FRAGMENT MATCH
                # =========================================================

                if (
                        reconstructed
                        == surface
                ):

                    if cross_sentence:

                        score = 65

                        strategy = (
                            "fragmented_surface_2_"
                            "cross_sentence_tokenizer_split"
                        )

                    else:

                        score = (
                            66
                            if group_size == 2
                            else 64
                        )

                        strategy = (
                            f"fragmented_surface_"
                            f"{group_size}"
                        )


                elif (
                        not cross_sentence
                        and reconstructed
                        == root
                ):

                    score = (
                        62
                        if group_size == 2
                        else 60
                    )

                    strategy = (
                        f"fragmented_root_"
                        f"{group_size}"
                    )

                if score == 0:
                    continue

                anchor = (
                    self._fragment_anchor_token(
                        group
                    )
                )

                fragment_candidates.append(
                    (
                        score,
                        group_ids[0],
                        strategy,
                        group,
                        anchor,
                        reconstructed
                    )
                )

        if not fragment_candidates:
            return None

        fragment_candidates.sort(
            key=lambda item: (
                -item[0],
                item[1]
            )
        )

        (
            _,
            _,
            strategy,
            best_group,
            best_anchor,
            reconstructed
        ) = fragment_candidates[0]

        fragment_ids = [
            int(token["id"])
            for token
            in best_group
        ]

        used_token_ids.update(
            fragment_ids
        )

        matched_token = dict(
            best_anchor
        )

        matched_token[
            "_alignment_strategy"
        ] = strategy

        matched_token[
            "_fragment_token_ids"
        ] = fragment_ids

        matched_token[
            "_reconstructed_surface"
        ] = reconstructed

        return matched_token

    # =========================================================
    # TARGET RESOLUTION
    # =========================================================

    def _find_direct_subject(
        self,
        sentiment_token,
        children
    ):

        token_id = sentiment_token[
            "id"
        ]

        for child in children.get(
            token_id,
            []
        ):

            if (
                child.get("deprel")
                in self.SUBJECT_RELATIONS
                and self._is_entity_candidate(
                    child
                )
            ):
                return child

        return None

    def _find_modified_noun(
        self,
        sentiment_token,
        token_by_id
    ):

        if (
            sentiment_token.get(
                "deprel"
            )
            != "amod"
        ):
            return None

        head_id = sentiment_token.get(
            "head"
        )

        head = token_by_id.get(
            head_id
        )

        if self._is_entity_candidate(
            head
        ):
            return head

        return None

    def _find_direct_object(
        self,
        sentiment_token,
        children
    ):

        token_id = sentiment_token[
            "id"
        ]

        for child in children.get(
            token_id,
            []
        ):

            if (
                child.get("deprel")
                in self.OBJECT_RELATIONS
                and self._is_entity_candidate(
                    child
                )
            ):
                return child

        return None

    def _find_subject_via_head(
        self,
        sentiment_token,
        token_by_id,
        children
    ):

        head_id = sentiment_token.get(
            "head"
        )

        if not head_id:
            return None

        if (
            head_id
            not in token_by_id
        ):
            return None

        for sibling in children.get(
            head_id,
            []
        ):

            if (
                sibling["id"]
                ==
                sentiment_token["id"]
            ):
                continue

            if (
                sibling.get("deprel")
                in self.SUBJECT_RELATIONS
                and self._is_entity_candidate(
                    sibling
                )
            ):
                return sibling

        return None

    def _find_object_via_head(
        self,
        sentiment_token,
        token_by_id,
        children
    ):

        head_id = sentiment_token.get(
            "head"
        )

        if not head_id:
            return None

        if (
            head_id
            not in token_by_id
        ):
            return None

        for sibling in children.get(
            head_id,
            []
        ):

            if (
                sibling["id"]
                ==
                sentiment_token["id"]
            ):
                continue

            if (
                sibling.get("deprel")
                in self.OBJECT_RELATIONS
                and self._is_entity_candidate(
                    sibling
                )
            ):
                return sibling

        return None

    def _find_target(
        self,
        sentiment_token,
        token_by_id,
        children
    ):

        target = (
            self._find_modified_noun(
                sentiment_token,
                token_by_id
            )
        )

        if target is not None:

            return (
                target,
                "modified_head"
            )

        target = (
            self._find_direct_subject(
                sentiment_token,
                children
            )
        )

        if target is not None:

            return (
                target,
                "direct_subject"
            )

        target = (
            self._find_direct_object(
                sentiment_token,
                children
            )
        )

        if target is not None:

            return (
                target,
                "direct_object"
            )

        target = (
            self._find_subject_via_head(
                sentiment_token,
                token_by_id,
                children
            )
        )

        if target is not None:

            return (
                target,
                "head_subject"
            )

        target = (
            self._find_object_via_head(
                sentiment_token,
                token_by_id,
                children
            )
        )

        if target is not None:

            return (
                target,
                "head_object"
            )

        return None, None

    # =========================================================
    # MODIFIER RESOLUTION
    # =========================================================

    def _find_modifiers(
        self,
        sentiment_token,
        children
    ):

        modifiers = []

        token_id = sentiment_token[
            "id"
        ]

        for child in children.get(
            token_id,
            []
        ):

            if (
                child.get("deprel")
                in self.MODIFIER_RELATIONS
                and not self._is_negation(
                    child
                )
            ):

                modifiers.append({
                    "id":
                        child["id"],

                    "word":
                        child.get(
                            "text"
                        ),

                    "lemma":
                        child.get(
                            "lemma"
                        ),

                    "deprel":
                        child.get(
                            "deprel"
                        )
                })

        return modifiers

    # =========================================================
    # NEGATION RESOLUTION
    # =========================================================

    def _find_negations(
        self,
        sentiment_token,
        token_by_id,
        children
    ):
        """
        Finds syntactically licensed negation.

        IMPORTANT
        ---------

        Self morphological negation no longer causes a score swap.

        Therefore:

            göremedim
            yaramıyor
            almayın
            alamadım

        do NOT become positive merely because Stanza marks the token
        itself with Polarity=Neg.

        Still retained:

            güzel değil
            tavsiye etmem
            pişman olmadım
            rahatsız etmedi
        """

        negations = []
        seen = set()

        sentiment_id = (
            sentiment_token[
                "id"
            ]
        )

        # -----------------------------------------------------
        # DO NOT ADD sentiment_token itself merely because
        # Polarity=Neg exists on the same token.
        # -----------------------------------------------------

        # =====================================================
        # DIRECT CHILD NEGATION
        # =====================================================

        for child in children.get(
            sentiment_id,
            []
        ):

            child_id = child[
                "id"
            ]

            if child_id in seen:
                continue

            deprel = (
                child.get(
                    "deprel"
                )
                or ""
            )

            is_explicit_degil = (
                self._is_explicit_degil(
                    child
                )
            )

            is_ud_negation = (
                deprel
                == "neg"
            )

            is_supported_morphological_negation = (
                self._has_morphological_negation(
                    child
                )
                and deprel
                in self.NEGATION_SUPPORT_RELATIONS
            )

            if (
                is_explicit_degil
                or is_ud_negation
                or is_supported_morphological_negation
            ):

                negations.append(
                    child
                )

                seen.add(
                    child_id
                )

        # =====================================================
        # NEGATED PARENT
        # =====================================================

        head_id = sentiment_token.get(
            "head"
        )

        head = token_by_id.get(
            head_id
        )

        if (
            head is not None
            and head["id"]
            not in seen
        ):

            sentiment_relation = (
                sentiment_token.get(
                    "deprel"
                )
                or ""
            )

            explicit_degil_parent = (
                self._is_explicit_degil(
                    head
                )
            )

            reverse_light_verb_negation = (
                sentiment_relation
                in self.PARENT_NEGATION_RELATIONS
                and
                self._has_morphological_negation(
                    head
                )
            )

            if (
                explicit_degil_parent
                or
                reverse_light_verb_negation
            ):

                negations.append(
                    head
                )

                seen.add(
                    head["id"]
                )

        return [
            {
                "id":
                    token["id"],

                "word":
                    token.get(
                        "text"
                    ),

                "lemma":
                    token.get(
                        "lemma"
                    ),

                "deprel":
                    token.get(
                        "deprel"
                    ),

                "feats":
                    token.get(
                        "feats"
                    )
            }

            for token
            in negations
        ]

    # =========================================================
    # EXPLICIT DEĞİL SCOPE CORRECTION
    # =========================================================

    def _disambiguate_shared_explicit_negations(
        self,
        scopes,
        dependency_tokens
    ):
        """
        Corrects explicit 'değil' scope after initial dependency
        extraction.

        Handles:

            iyi ve güzel değildi
                -> both negated

            güzel ama iyi değildi
                -> only iyi negated

            Otel güzel ama odalar kötü değildi
                -> only kötü negated
        """

        if (
            not scopes
            or not dependency_tokens
        ):
            return scopes

        token_by_id = {
            int(token["id"]):
                token

            for token
            in dependency_tokens

            if token.get(
                "id"
            ) is not None
        }

        aligned_scopes = []

        for scope in scopes:

            sentiment_token = (
                scope.get(
                    "sentiment_token"
                )
            )

            if sentiment_token is None:
                continue

            sentiment_id = (
                sentiment_token.get(
                    "id"
                )
            )

            if sentiment_id is None:
                continue

            aligned_scopes.append(
                scope
            )

        if not aligned_scopes:
            return scopes

        # -----------------------------------------------------
        # CONTRAST BOUNDARIES
        # -----------------------------------------------------

        contrast_marker_ids = []

        for token in dependency_tokens:

            token_id = token.get(
                "id"
            )

            if token_id is None:
                continue

            lemma = self._turkish_lower(
                token.get(
                    "lemma"
                )
                or
                token.get(
                    "text"
                )
            )

            deprel = (
                token.get(
                    "deprel"
                )
                or ""
            )

            if (
                lemma
                in self.CONTRAST_WORDS
                and deprel in {
                    "cc",
                    "mark",
                    "case"
                }
            ):

                contrast_marker_ids.append(
                    int(token_id)
                )

        explicit_degil_tokens = [
            token

            for token
            in dependency_tokens

            if (
                token.get(
                    "id"
                ) is not None
                and
                self._is_explicit_degil(
                    token
                )
            )
        ]

        for negation_token in (
            explicit_degil_tokens
        ):

            negation_id = int(
                negation_token[
                    "id"
                ]
            )

            preceding_scopes = [
                scope

                for scope
                in aligned_scopes

                if int(
                    scope[
                        "sentiment_token"
                    ][
                        "id"
                    ]
                )
                < negation_id
            ]

            if not preceding_scopes:
                continue

            min_sentiment_id = min(
                int(
                    scope[
                        "sentiment_token"
                    ][
                        "id"
                    ]
                )

                for scope
                in preceding_scopes
            )

            # =================================================
            # CONTRAST BOUNDARY
            # =================================================

            relevant_boundaries = [
                marker_id

                for marker_id
                in contrast_marker_ids

                if (
                    min_sentiment_id
                    < marker_id
                    < negation_id
                )
            ]

            boundary_id = (
                max(
                    relevant_boundaries
                )
                if relevant_boundaries
                else None
            )

            if boundary_id is not None:

                segment_scopes = [
                    scope

                    for scope
                    in preceding_scopes

                    if int(
                        scope[
                            "sentiment_token"
                        ][
                            "id"
                        ]
                    )
                    > boundary_id
                ]

                strategy_prefix = (
                    "explicit_degil_"
                    "post_contrast"
                )

            else:

                segment_scopes = (
                    preceding_scopes
                )

                strategy_prefix = (
                    "explicit_degil_"
                    "preceding_segment"
                )

            if not segment_scopes:
                continue

            nearest_scope = max(
                segment_scopes,
                key=lambda scope:
                    int(
                        scope[
                            "sentiment_token"
                        ][
                            "id"
                        ]
                    )
            )

            nearest_id = int(
                nearest_scope[
                    "sentiment_token"
                ][
                    "id"
                ]
            )

            segment_scope_by_id = {
                int(
                    scope[
                        "sentiment_token"
                    ][
                        "id"
                    ]):
                    scope

                for scope
                in segment_scopes
            }

            segment_sentiment_ids = set(
                segment_scope_by_id.keys()
            )

            # =================================================
            # COORDINATION COMPONENT
            # =================================================

            adjacency = {
                sentiment_id:
                    set()

                for sentiment_id
                in segment_sentiment_ids
            }

            for sentiment_id in (
                segment_sentiment_ids
            ):

                token = token_by_id.get(
                    sentiment_id
                )

                if token is None:
                    continue

                if (
                    token.get(
                        "deprel"
                    )
                    == "conj"
                ):

                    head_id = token.get(
                        "head"
                    )

                    if head_id is None:
                        continue

                    head_id = int(
                        head_id
                    )

                    if (
                        head_id
                        in segment_sentiment_ids
                    ):

                        adjacency[
                            sentiment_id
                        ].add(
                            head_id
                        )

                        adjacency[
                            head_id
                        ].add(
                            sentiment_id
                        )

            component_ids = set()

            stack = [
                nearest_id
            ]

            while stack:

                current_id = (
                    stack.pop()
                )

                if (
                    current_id
                    in component_ids
                ):
                    continue

                component_ids.add(
                    current_id
                )

                stack.extend(
                    adjacency.get(
                        current_id,
                        set()
                    )
                    - component_ids
                )

            if (
                len(
                    component_ids
                )
                > 1
            ):

                selected_scopes = [
                    segment_scope_by_id[
                        sentiment_id
                    ]

                    for sentiment_id
                    in sorted(
                        component_ids
                    )
                ]

                strategy = (
                    strategy_prefix
                    +
                    "_shared_coordination"
                )

            else:

                selected_scopes = [
                    nearest_scope
                ]

                strategy = (
                    strategy_prefix
                    +
                    "_nearest"
                )

            selected_scope_ids = {
                id(scope)

                for scope
                in selected_scopes
            }

            serialized_negation = {
                "id":
                    negation_id,

                "word":
                    negation_token.get(
                        "text"
                    ),

                "lemma":
                    negation_token.get(
                        "lemma"
                    ),

                "deprel":
                    negation_token.get(
                        "deprel"
                    ),

                "feats":
                    negation_token.get(
                        "feats"
                    )
            }

            # =================================================
            # REASSIGN DEĞİL
            # =================================================

            for scope in (
                aligned_scopes
            ):

                current_negations = (
                    scope.get(
                        "negations",
                        []
                    )
                    or []
                )

                other_negations = [
                    negation

                    for negation
                    in current_negations

                    if not (
                        self._is_explicit_degil(
                            negation
                        )
                        and int(
                            negation.get(
                                "id",
                                -1
                            )
                        )
                        == negation_id
                    )
                ]

                if (
                    id(scope)
                    in selected_scope_ids
                ):

                    other_negations.append(
                        serialized_negation.copy()
                    )

                    scope[
                        "negation_scope_strategy"
                    ] = strategy

                else:

                    had_this_negation = any(
                        self._is_explicit_degil(
                            negation
                        )
                        and int(
                            negation.get(
                                "id",
                                -1
                            )
                        )
                        == negation_id

                        for negation
                        in current_negations
                    )

                    if had_this_negation:

                        scope[
                            "negation_scope_strategy"
                        ] = (
                            "explicit_degil_"
                            "removed_by_scope"
                        )

                deduplicated = []
                seen_ids = set()

                for negation in (
                    other_negations
                ):

                    neg_id = (
                        negation.get(
                            "id"
                        )
                    )

                    if neg_id is None:

                        deduplicated.append(
                            negation
                        )

                        continue

                    neg_id = int(
                        neg_id
                    )

                    if (
                        neg_id
                        in seen_ids
                    ):
                        continue

                    seen_ids.add(
                        neg_id
                    )

                    deduplicated.append(
                        negation
                    )

                scope[
                    "negations"
                ] = deduplicated

        return scopes

    # =========================================================
    # CONTRAST
    # =========================================================

    def _find_contrast_marker(
        self,
        sentiment_token,
        target,
        token_by_id,
        children
    ):

        candidate_ids = {
            sentiment_token[
                "id"
            ]
        }

        head_id = sentiment_token.get(
            "head"
        )

        if head_id:

            candidate_ids.add(
                head_id
            )

        if target is not None:

            candidate_ids.add(
                target[
                    "id"
                ]
            )

        for candidate_id in (
            candidate_ids
        ):

            for child in children.get(
                candidate_id,
                []
            ):

                lemma = self._turkish_lower(
                    child.get(
                        "lemma"
                    )
                    or
                    child.get(
                        "text"
                    )
                )

                if (
                    child.get(
                        "deprel"
                    )
                    == "cc"
                    and lemma
                    in self.CONTRAST_WORDS
                ):

                    return {
                        "id":
                            child["id"],

                        "word":
                            child.get(
                                "text"
                            ),

                        "lemma":
                            lemma,

                        "anchor_id":
                            candidate_id
                    }

        return None

    # =========================================================
    # SENTIMENT ROLE RESOLUTION
    # =========================================================

    def _resolve_sentiment_roles(
        self,
        results
    ):

        target_token_ids = set()

        for result in results:

            target = result.get(
                "target"
            )

            if target is not None:

                target_token_ids.add(
                    target["id"]
                )

        for result in results:

            sentiment_token = (
                result.get(
                    "sentiment_token"
                )
            )

            result[
                "active_sentiment"
            ] = True

            result[
                "contextual_role"
            ] = "opinion"

            result[
                "suppression_reason"
            ] = None

            if sentiment_token is None:
                continue

            token_id = (
                sentiment_token[
                    "id"
                ]
            )

            if (
                token_id
                in target_token_ids
                and result.get(
                    "target"
                ) is None
            ):

                result[
                    "active_sentiment"
                ] = False

                result[
                    "contextual_role"
                ] = "target"

                result[
                    "suppression_reason"
                ] = (
                    "used_as_target_of_"
                    "another_sentiment"
                )

        return results

    # =========================================================
    # MORPHOLOGICAL ROOT
    # =========================================================

    def _find_morphological_root(
        self,
        dependency_token,
        morphological_analyses
    ):

        surface = self._turkish_lower(
            dependency_token.get(
                "text"
            )
        )

        if morphological_analyses:

            for analysis in (
                morphological_analyses
            ):

                analysis_surface = (
                    self._turkish_lower(
                        analysis.get(
                            "word"
                        )
                    )
                )

                if (
                    analysis_surface
                    == surface
                ):

                    return (
                        self._turkish_lower(
                            analysis.get(
                                "root"
                            )
                        )
                    )

        return self._turkish_lower(
            dependency_token.get(
                "lemma"
            )
            or
            dependency_token.get(
                "text"
            )
        )

    # =========================================================
    # DEPENDENCY-FIRST CANDIDATE FILTER
    # =========================================================

    def _morphology_pos(
            self,
            analysis_record
    ):
        """
        Extracts coarse POS from Starlang morphological analysis.

        No HisNet information is used here.
        """

        if analysis_record is None:
            return None

        analysis = str(
            analysis_record.get(
                "analysis"
            )
            or ""
        )

        for pos in [
            "ADJ",
            "VERB",
            "NOUN",
            "ADV",
            "INTJ",
            "PROPN",
            "PRON"
        ]:

            if (
                    f"+{pos}"
                    in analysis
            ):
                return pos

        return None

    def _find_morphology_record_for_surface(
            self,
            surface,
            morphological_analyses
    ):
        """
        Finds a Starlang morphology record using normalized surface.

        This is intentionally independent of HisNet.
        """

        normalized_surface = (
            self._normalize_alignment_piece(
                surface
            )
        )

        if not normalized_surface:
            return None

        for record in (
                morphological_analyses
                or []
        ):

            record_word = (
                self._normalize_alignment_piece(
                    record.get(
                        "word"
                    )
                )
            )

            if (
                    record_word
                    ==
                    normalized_surface
            ):
                return record

        return None

    def _build_dependency_repair_units(
            self,
            dependency_tokens,
            morphological_analyses
    ):
        """
        Creates morphology-guided virtual dependency tokens for parser
        fragmentation/noise.

        IMPORTANT:
        - HisNet is NOT consulted.
        - Original dependency tokens are not mutated.
        - Ordinary tokens remain unchanged.
        - Repair is conservative and morphology-licensed.

        Examples:

            harik + a       -> harika
            dandi + k       -> dandik
            özen + li       -> özenli
            hesap + lı      -> hesaplı
            kullanışsı + z  -> kullanışsız

            ..yumuşak       -> yumuşak
            temizlik👍🏻👍   -> temizlik
            kötü.3×l        -> kötü
        """

        tokens = sorted(
            dependency_tokens,
            key=lambda token:
            int(
                token["id"]
            )
        )

        morphology_records = (
                morphological_analyses
                or []
        )

        repair_units = []

        consumed_ids = set()

        # ========================================================
        # 1. TWO-TOKEN FRAGMENT RECONSTRUCTION
        # ========================================================

        for i in range(
                len(tokens) - 1
        ):

            first = tokens[i]
            second = tokens[i + 1]

            first_id = int(
                first["id"]
            )

            second_id = int(
                second["id"]
            )

            if (
                    second_id
                    !=
                    first_id + 1
            ):
                continue

            if (
                    first_id
                    in consumed_ids
                    or
                    second_id
                    in consumed_ids
            ):
                continue

            first_piece = (
                self._normalize_alignment_piece(
                    first.get(
                        "text"
                    )
                )
            )

            second_piece = (
                self._normalize_alignment_piece(
                    second.get(
                        "text"
                    )
                )
            )

            if (
                    not first_piece
                    or not second_piece
            ):
                continue

            reconstructed = (
                    first_piece
                    + second_piece
            )

            morphology_record = (
                self._find_morphology_record_for_surface(
                    reconstructed,
                    morphology_records
                )
            )

            if morphology_record is None:
                continue

            morph_pos = (
                self._morphology_pos(
                    morphology_record
                )
            )

            # We only need repair units that could potentially
            # participate in sentiment candidate selection.
            if (
                    morph_pos
                    not in self.OPINION_POS
            ):
                continue

            sentence_indexes = {
                token.get(
                    "sentence_index"
                )
                for token in [
                    first,
                    second
                ]
                if token.get(
                    "sentence_index"
                )
                   is not None
            }

            cross_sentence = (
                    len(
                        sentence_indexes
                    )
                    > 1
            )

            # Same conservative safety rule used by H->D.
            #
            # Cross-sentence reconstruction is allowed only when
            # one of the pieces is suffix-like.
            if cross_sentence:

                if not (
                        len(first_piece) <= 2
                        or len(second_piece) <= 2
                ):
                    continue

            anchor = (
                self._fragment_anchor_token(
                    [
                        first,
                        second
                    ]
                )
            )

            repaired = dict(
                anchor
            )

            repaired[
                "text"
            ] = reconstructed

            repaired[
                "lemma"
            ] = (
                    morphology_record.get(
                        "root"
                    )
                    or reconstructed
            )

            repaired[
                "upos"
            ] = (
                    morph_pos
                    or anchor.get(
                "upos"
            )
            )

            repaired[
                "_dependency_repair"
            ] = True

            repaired[
                "_repair_strategy"
            ] = (
                "morphology_fragment_2_"
                "cross_sentence"
                if cross_sentence
                else
                "morphology_fragment_2"
            )

            repaired[
                "_fragment_token_ids"
            ] = [
                first_id,
                second_id
            ]

            repaired[
                "_original_anchor_text"
            ] = anchor.get(
                "text"
            )

            repaired[
                "_morphology_record"
            ] = morphology_record

            repair_units.append(
                repaired
            )

            consumed_ids.update(
                [
                    first_id,
                    second_id
                ]
            )

        # ========================================================
        # 2. SINGLE-TOKEN EDGE / DECORATION REPAIR
        # ========================================================

        for token in tokens:

            token_id = int(
                token["id"]
            )

            if (
                    token_id
                    in consumed_ids
            ):
                continue

            raw_surface = (
                self._turkish_lower(
                    token.get(
                        "text"
                    )
                )
            )

            clean_surface = (
                self._normalize_alignment_piece(
                    raw_surface
                )
            )

            matched_record = None
            repaired_surface = None

            # ----------------------------------------------------
            # Simple edge noise:
            #
            #   ..yumuşak
            #   temizlik👍🏻👍
            # ----------------------------------------------------

            if (
                    clean_surface
                    and clean_surface
                    != raw_surface
            ):

                record = (
                    self._find_morphology_record_for_surface(
                        clean_surface,
                        morphology_records
                    )
                )

                if record is not None:
                    matched_record = record
                    repaired_surface = (
                        clean_surface
                    )

            # ----------------------------------------------------
            # Embedded trailing noise:
            #
            #   kötü.3×l
            #
            # Try morphology words as an exact prefix/suffix when
            # separated by non-alphanumeric noise.
            # ----------------------------------------------------

            if matched_record is None:

                for record in (
                        morphology_records
                ):

                    candidate = (
                        self._normalize_alignment_piece(
                            record.get(
                                "word"
                            )
                        )
                    )

                    if not candidate:
                        continue

                    if (
                            raw_surface.startswith(
                                candidate
                            )
                            and
                            len(raw_surface)
                            > len(candidate)
                    ):

                        boundary = (
                            raw_surface[
                                len(candidate)
                            ]
                        )

                        if not boundary.isalnum():
                            matched_record = (
                                record
                            )

                            repaired_surface = (
                                candidate
                            )

                            break

                    if (
                            raw_surface.endswith(
                                candidate
                            )
                            and
                            len(raw_surface)
                            > len(candidate)
                    ):

                        boundary_index = (
                                len(raw_surface)
                                - len(candidate)
                                - 1
                        )

                        if (
                                boundary_index >= 0
                                and
                                not raw_surface[
                                    boundary_index
                                ].isalnum()
                        ):
                            matched_record = (
                                record
                            )

                            repaired_surface = (
                                candidate
                            )

                            break

            if (
                    matched_record
                    is None
            ):
                continue

            morph_pos = (
                self._morphology_pos(
                    matched_record
                )
            )

            if (
                    morph_pos
                    not in self.OPINION_POS
            ):
                continue

            repaired = dict(
                token
            )

            repaired[
                "text"
            ] = repaired_surface

            repaired[
                "lemma"
            ] = (
                    matched_record.get(
                        "root"
                    )
                    or repaired_surface
            )

            repaired[
                "upos"
            ] = (
                    morph_pos
                    or token.get(
                "upos"
            )
            )

            repaired[
                "_dependency_repair"
            ] = True

            repaired[
                "_repair_strategy"
            ] = (
                "morphology_edge_noise"
            )

            repaired[
                "_fragment_token_ids"
            ] = [
                token_id
            ]

            repaired[
                "_original_anchor_text"
            ] = token.get(
                "text"
            )

            repaired[
                "_morphology_record"
            ] = matched_record

            repair_units.append(
                repaired
            )

            consumed_ids.add(
                token_id
            )

        return (
            repair_units,
            consumed_ids
        )

    def _is_dependency_opinion_candidate(
        self,
        token,
        target,
        target_strategy
    ):

        if token is None:

            return (
                False,
                None
            )

        upos = (
            token.get(
                "upos"
            )
            or ""
        )

        deprel = (
            token.get(
                "deprel"
            )
            or ""
        )

        if (
            upos
            not in self.OPINION_POS
        ):

            return (
                False,
                None
            )

        if (
            deprel
            in self.OPINION_EXCLUDED_RELATIONS
        ):

            return (
                False,
                None
            )

        if (
            deprel
            in self.OPINION_PREDICATE_RELATIONS
        ):

            if target is not None:

                return (
                    True,
                    "predicate_with_target"
                )

            return (
                True,
                "predicate_candidate"
            )

        if (
            deprel == "amod"
            and target is not None
            and target_strategy
            == "modified_head"
        ):

            return (
                True,
                "adjectival_modifier"
            )

        if (
            target is not None
            and target_strategy in {
                "direct_subject",
                "direct_object",
                "modified_head"
            }
        ):

            return (
                True,
                "direct_target_relation"
            )

        if (
            target is not None
            and target_strategy in {
                "head_subject",
                "head_object"
            }
            and upos in {
                "ADJ",
                "VERB"
            }
        ):

            return (
                True,
                "shared_head_fallback"
            )

        if upos in {
            "ADJ",
            "ADV",
            "INTJ"
        }:

            return (
                True,
                "descriptive_content_fallback"
            )

        return (
            False,
            None
        )

    # =========================================================
    # D -> H
    # =========================================================

    def resolve_dependency_candidates(
            self,
            dependency_tokens,
            morphological_analyses=None
    ):
        """
        DEPENDENCY-FIRST candidate extraction.

        Pipeline:

            Stanza dependency tokens
                ->
            morphology-guided parser repair
                ->
            syntactic opinion candidate selection
                ->
            target / negation / modifier / contrast
                ->
            Starlang morphological root
                ->
            HisNet lookup happens later

        IMPORTANT:
        HisNet scores/entries are NEVER consulted in this method.
        """

        (
            token_by_id,
            children
        ) = self._build_indexes(
            dependency_tokens
        )

        # ========================================================
        # MORPHOLOGY-GUIDED PARSER REPAIR
        # ========================================================

        (
            repaired_units,
            repaired_original_ids
        ) = (
            self._build_dependency_repair_units(
                dependency_tokens,
                morphological_analyses
            )
        )

        # ========================================================
        # CANDIDATE UNITS
        #
        # A repaired unit replaces its original fragmented/noisy
        # dependency token(s) for candidate selection only.
        #
        # The original dependency graph/indexes remain untouched.
        # ========================================================

        candidate_units = []

        for token in (
                dependency_tokens
        ):

            token_id = int(
                token["id"]
            )

            if (
                    token_id
                    in repaired_original_ids
            ):
                continue

            candidate_units.append(
                token
            )

        candidate_units.extend(
            repaired_units
        )

        candidate_units.sort(
            key=lambda token:
            int(
                token["id"]
            )
        )

        candidates = []

        for token in (
                candidate_units
        ):

            token_id = int(
                token["id"]
            )

            # ====================================================
            # TARGET
            #
            # The repaired token keeps the ID/head/deprel of its
            # dependency anchor, therefore target resolution remains
            # dependency-based.
            # ====================================================

            (
                target,
                target_strategy
            ) = self._find_target(
                token,
                token_by_id,
                children
            )

            # ====================================================
            # CANDIDATE STATUS — STILL BEFORE HISNET
            # ====================================================

            (
                is_candidate,
                candidate_strategy
            ) = (
                self._is_dependency_opinion_candidate(
                    token,
                    target,
                    target_strategy
                )
            )

            if not is_candidate:
                continue

            # ====================================================
            # DEPENDENCY SCOPE
            # ====================================================

            modifiers = (
                self._find_modifiers(
                    token,
                    children
                )
            )

            negations = (
                self._find_negations(
                    token,
                    token_by_id,
                    children
                )
            )

            contrast = (
                self._find_contrast_marker(
                    token,
                    target,
                    token_by_id,
                    children
                )
            )

            # ====================================================
            # ROOT
            # ====================================================

            repair_record = (
                token.get(
                    "_morphology_record"
                )
            )

            if repair_record is not None:

                root = (
                    self._turkish_lower(
                        repair_record.get(
                            "root"
                        )
                        or token.get(
                            "text"
                        )
                    )
                )

            else:

                root = (
                    self._find_morphological_root(
                        token,
                        morphological_analyses
                    )
                )

            # ====================================================
            # TARGET SERIALIZATION
            # ====================================================

            if target is not None:

                target_data = {

                    "id":
                        target["id"],

                    "word":
                        target.get(
                            "text"
                        ),

                    "lemma":
                        target.get(
                            "lemma"
                        ),

                    "root":
                        self._find_morphological_root(
                            target,
                            morphological_analyses
                        ),

                    "upos":
                        target.get(
                            "upos"
                        ),

                    "deprel":
                        target.get(
                            "deprel"
                        )
                }

            else:

                target_data = None

            # ====================================================
            # CANDIDATE
            # ====================================================

            candidates.append({

                "word":
                    token.get(
                        "text"
                    ),

                "root":
                    root,

                "sentiment_token": {

                    "id":
                        token_id,

                    "word":
                        token.get(
                            "text"
                        ),

                    "lemma":
                        token.get(
                            "lemma"
                        ),

                    "upos":
                        token.get(
                            "upos"
                        ),

                    "head":
                        token.get(
                            "head"
                        ),

                    "deprel":
                        token.get(
                            "deprel"
                        )
                },

                "candidate_strategy":
                    (
                        "morphology_repaired:"
                        + str(
                            token.get(
                                "_repair_strategy"
                            )
                        )
                        + ":"
                        + str(
                            candidate_strategy
                        )

                        if token.get(
                            "_dependency_repair"
                        )

                        else candidate_strategy
                    ),

                "target":
                    target_data,

                "target_strategy":
                    target_strategy,

                "modifiers":
                    modifiers,

                "negations":
                    negations,

                "contrast":
                    contrast,

                "alignment_status":
                    "matched",

                # -----------------------------------------------
                # Diagnostics
                # -----------------------------------------------

                "dependency_repaired":
                    bool(
                        token.get(
                            "_dependency_repair",
                            False
                        )
                    ),

                "repair_strategy":
                    token.get(
                        "_repair_strategy"
                    ),

                "fragment_token_ids":
                    token.get(
                        "_fragment_token_ids",
                        [
                            token_id
                        ]
                    ),

                "original_anchor_text":
                    token.get(
                        "_original_anchor_text"
                    )
            })

        return (
            self._disambiguate_shared_explicit_negations(
                candidates,
                dependency_tokens
            )
        )

    # =========================================================
    # H -> D
    # =========================================================

    def resolve(
        self,
        sentiment_words,
        dependency_tokens,
        morphological_analyses=None
    ):
        """
        Resolves dependency scope for all HisNet sentiment words.
        """

        (
            token_by_id,
            children
        ) = self._build_indexes(
            dependency_tokens
        )

        used_token_ids = set()

        results = []

        for sentiment_word in (
            sentiment_words
        ):

            sentiment_token = (
                self._match_sentiment_token(
                    sentiment_word,
                    dependency_tokens,
                    used_token_ids
                )
            )

            # =================================================
            # UNMATCHED
            # =================================================

            if sentiment_token is None:

                results.append({

                    "word":
                        sentiment_word[
                            "word"
                        ],

                    "root":
                        sentiment_word[
                            "root"
                        ],

                    "positive":
                        sentiment_word[
                            "positive"
                        ],

                    "negative":
                        sentiment_word[
                            "negative"
                        ],

                    "sentiment_token":
                        None,

                    "target":
                        None,

                    "target_strategy":
                        None,

                    "modifiers":
                        [],

                    "negations":
                        [],

                    "contrast":
                        None,

                    "alignment_status":
                        "unmatched",

                    "alignment_strategy":
                        None,

                    "fragment_token_ids":
                        [],

                    "reconstructed_surface":
                        None
                })

                continue

            # =================================================
            # DEPENDENCY SCOPE
            # =================================================

            (
                target,
                target_strategy
            ) = self._find_target(
                sentiment_token,
                token_by_id,
                children
            )

            modifiers = (
                self._find_modifiers(
                    sentiment_token,
                    children
                )
            )

            negations = (
                self._find_negations(
                    sentiment_token,
                    token_by_id,
                    children
                )
            )

            contrast = (
                self._find_contrast_marker(
                    sentiment_token,
                    target,
                    token_by_id,
                    children
                )
            )

            if target is not None:

                target_data = {

                    "id":
                        target[
                            "id"
                        ],

                    "word":
                        target.get(
                            "text"
                        ),

                    "lemma":
                        target.get(
                            "lemma"
                        ),

                    "root":
                        self._find_morphological_root(
                            target,
                            morphological_analyses
                        ),

                    "upos":
                        target.get(
                            "upos"
                        ),

                    "deprel":
                        target.get(
                            "deprel"
                        )
                }

            else:

                target_data = None

            results.append({

                "word":
                    sentiment_word[
                        "word"
                    ],

                "root":
                    sentiment_word[
                        "root"
                    ],

                "positive":
                    sentiment_word[
                        "positive"
                    ],

                "negative":
                    sentiment_word[
                        "negative"
                    ],

                "sentiment_token": {

                    "id":
                        sentiment_token[
                            "id"
                        ],

                    "word":
                        sentiment_token.get(
                            "text"
                        ),

                    "lemma":
                        sentiment_token.get(
                            "lemma"
                        ),

                    "upos":
                        sentiment_token.get(
                            "upos"
                        ),

                    "head":
                        sentiment_token.get(
                            "head"
                        ),

                    "deprel":
                        sentiment_token.get(
                            "deprel"
                        ),

                    "alignment_strategy":
                        sentiment_token.get(
                            "_alignment_strategy"
                        ),

                    "fragment_token_ids":
                        sentiment_token.get(
                            "_fragment_token_ids",
                            [
                                sentiment_token[
                                    "id"
                                ]
                            ]
                        ),

                    "reconstructed_surface":
                        sentiment_token.get(
                            "_reconstructed_surface"
                        )
                },

                "target":
                    target_data,

                "target_strategy":
                    target_strategy,

                "modifiers":
                    modifiers,

                "negations":
                    negations,

                "contrast":
                    contrast,

                "alignment_status":
                    "matched",

                "alignment_strategy":
                    sentiment_token.get(
                        "_alignment_strategy"
                    ),

                "fragment_token_ids":
                    sentiment_token.get(
                        "_fragment_token_ids",
                        [
                            sentiment_token[
                                "id"
                            ]
                        ]
                    ),

                "reconstructed_surface":
                    sentiment_token.get(
                        "_reconstructed_surface"
                    )
            })

        results = (
            self._disambiguate_shared_explicit_negations(
                results,
                dependency_tokens
            )
        )

        return (
            self._resolve_sentiment_roles(
                results
            )
        )