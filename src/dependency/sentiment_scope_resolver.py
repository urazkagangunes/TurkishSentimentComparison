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

    # Relations that can reasonably represent an
    # independent predicate/opinion expression.
    OPINION_PREDICATE_RELATIONS = {
        "root",
        "conj",
        "xcomp",
        "ccomp",
        "acl",
        "acl:relcl"
    }

    # Relations that usually represent grammatical,
    # supporting or modifying material rather than an
    # independent opinion expression.
    #
    # These tokens can still be USED by dependency scope
    # resolution, but they are not sent to HisNet as
    # independent opinion candidates.
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
        """
        Returns True only when the token itself carries
        Turkish grammatical/morphological negation.

        Example:
            sevmedim
            etmiyorum
            değildi
        """

        if token is None:
            return False

        feats = token.get("feats") or ""

        return (
            "Polarity=Neg"
            in feats
        )

    def _is_explicit_degil(self, token):
        """
        Detects explicit Turkish copular negation based on 'değil'.

        Stanza normally lemmatizes forms such as:
            değildi
            değildir
            değilmiş
            değilim

        to 'değil', but surface-form fallback is retained.
        """

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
        """
        Creates token lookup and parent -> children mapping.
        """

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

    def _match_sentiment_token(
        self,
        sentiment_word,
        tokens,
        used_token_ids
    ):
        """
        Aligns a HisNet sentiment word with a Stanza token.

        Morphological analyzer and Stanza may produce slightly
        different lemmas, so surface form receives highest priority.
        """

        surface = self._turkish_lower(
            sentiment_word["word"]
        )

        root = self._turkish_lower(
            sentiment_word["root"]
        )

        candidates = []

        for token in tokens:

            token_id = token["id"]

            if token_id in used_token_ids:
                continue

            token_text = self._turkish_lower(
                token.get("text")
            )

            token_lemma = self._turkish_lower(
                token.get("lemma")
            )

            score = 0

            # Exact surface-form match.
            if token_text == surface:
                score = 100

            # Token itself equals the morphological root.
            elif token_text == root:
                score = 90

            # Stanza lemma equals morphological root.
            elif token_lemma == root:
                score = 80

            if score > 0:
                candidates.append(
                    (score, token)
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )

        best_token = candidates[0][1]

        used_token_ids.add(
            best_token["id"]
        )

        return best_token

    def _find_direct_subject(
        self,
        sentiment_token,
        children
    ):
        """
        Example:

            Film güzel.
            Film --nsubj--> güzel

        Returns Film.
        """

        token_id = sentiment_token["id"]

        for child in children.get(
            token_id,
            []
        ):

            if (
                child.get("deprel")
                in self.SUBJECT_RELATIONS
                and self._is_entity_candidate(child)
            ):
                return child

        return None

    def _find_modified_noun(
        self,
        sentiment_token,
        token_by_id
    ):
        """
        Example:

            güzel otel

            güzel --amod--> otel

        The head noun is the target.
        """

        if (
            sentiment_token.get("deprel")
            != "amod"
        ):
            return None

        head_id = sentiment_token.get(
            "head"
        )

        head = token_by_id.get(
            head_id
        )

        if self._is_entity_candidate(head):
            return head

        return None

    def _find_direct_object(
        self,
        sentiment_token,
        children
    ):
        """
        Useful especially for sentiment-bearing verbs/nouns.

        Example:

            ürünü tavsiye ettim

        The object can be the sentiment target.
        """

        token_id = sentiment_token["id"]

        for child in children.get(
            token_id,
            []
        ):

            if (
                child.get("deprel")
                in self.OBJECT_RELATIONS
                and self._is_entity_candidate(child)
            ):
                return child

        return None

    def _find_subject_via_head(
        self,
        sentiment_token,
        token_by_id,
        children
    ):
        """
        Handles parser structures such as:

            Otel kötü değildi.

        where Stanza may produce something similar to:

            kötü  -> değildi
            Otel  -> değildi

        Sentiment word and target become siblings.
        """

        head_id = sentiment_token.get(
            "head"
        )

        if not head_id:
            return None

        if head_id not in token_by_id:
            return None

        for sibling in children.get(
            head_id,
            []
        ):

            if sibling["id"] == sentiment_token["id"]:
                continue

            if (
                sibling.get("deprel")
                in self.SUBJECT_RELATIONS
                and self._is_entity_candidate(sibling)
            ):
                return sibling

        return None

    def _find_object_via_head(
        self,
        sentiment_token,
        token_by_id,
        children
    ):
        """
        Fallback for light-verb / auxiliary structures.
        """

        head_id = sentiment_token.get(
            "head"
        )

        if not head_id:
            return None

        if head_id not in token_by_id:
            return None

        for sibling in children.get(
            head_id,
            []
        ):

            if sibling["id"] == sentiment_token["id"]:
                continue

            if (
                sibling.get("deprel")
                in self.OBJECT_RELATIONS
                and self._is_entity_candidate(sibling)
            ):
                return sibling

        return None

    def _find_target(
        self,
        sentiment_token,
        token_by_id,
        children
    ):
        """
        Finds the entity affected by the sentiment expression.

        The order is intentionally conservative.
        """

        # -------------------------------------------------
        # 1. Adjectival modifier:
        #
        #    güzel otel
        # -------------------------------------------------

        target = self._find_modified_noun(
            sentiment_token,
            token_by_id
        )

        if target is not None:
            return target, "modified_head"

        # -------------------------------------------------
        # 2. Predicate subject:
        #
        #    Otel güzel.
        # -------------------------------------------------

        target = self._find_direct_subject(
            sentiment_token,
            children
        )

        if target is not None:
            return target, "direct_subject"

        # -------------------------------------------------
        # 3. Object of sentiment expression:
        #
        #    Ürünü tavsiye ettim.
        # -------------------------------------------------

        target = self._find_direct_object(
            sentiment_token,
            children
        )

        if target is not None:
            return target, "direct_object"

        # -------------------------------------------------
        # 4. Subject attached to common parent.
        #
        #    Otel kötü değildi.
        # -------------------------------------------------

        target = self._find_subject_via_head(
            sentiment_token,
            token_by_id,
            children
        )

        if target is not None:
            return target, "head_subject"

        # -------------------------------------------------
        # 5. Object attached to common parent.
        # -------------------------------------------------

        target = self._find_object_via_head(
            sentiment_token,
            token_by_id,
            children
        )

        if target is not None:
            return target, "head_object"

        return None, None

    def _find_modifiers(
        self,
        sentiment_token,
        children
    ):
        """
        Finds modifiers directly affecting the sentiment token.

        Examples:
            çok güzel
            biraz kötü
            oldukça iyi
        """

        modifiers = []

        token_id = sentiment_token["id"]

        for child in children.get(
            token_id,
            []
        ):

            if (
                child.get("deprel")
                in self.MODIFIER_RELATIONS
                and not self._is_negation(child)
            ):

                modifiers.append({
                    "id": child["id"],
                    "word": child.get("text"),
                    "lemma": child.get("lemma"),
                    "deprel": child.get("deprel")
                })

        return modifiers

    def _find_negations(
            self,
            sentiment_token,
            token_by_id,
            children
    ):
        """
        Finds grammatical negation that can be conservatively
        considered to affect this sentiment expression.

        IMPORTANT
        ---------
        Morphological Polarity=Neg does NOT automatically mean
        that every syntactically related sentiment word should
        have its sentiment polarity reversed.

        The resolver therefore accepts only:

        1. Negation morphologically encoded on the sentiment
           expression itself.

               sevmedim

        2. Explicit / structurally licensed negation child.

               güzel <- değildi
               tavsiye <- etmiyorum

        3. A negated parent only in highly restricted structures:

               a) explicit 'değil'
               b) reverse light-verb construction

        Arbitrary negated parents are deliberately ignored.
        """

        negations = []
        seen = set()

        sentiment_id = (
            sentiment_token[
                "id"
            ]
        )

        # =====================================================
        # CASE 1
        #
        # The sentiment-bearing token itself is morphologically
        # negated.
        #
        # Example:
        #
        #   sevmedim
        #
        # root = sev
        # Polarity=Neg
        #
        # Since the lexical sentiment expression itself carries
        # negation, flipping is justified.
        # =====================================================

        if self._has_morphological_negation(
                sentiment_token
        ):
            negations.append(
                sentiment_token
            )

            seen.add(
                sentiment_id
            )

        # =====================================================
        # CASE 2
        #
        # Direct child negation.
        #
        # We DO NOT accept an arbitrary Polarity=Neg child.
        #
        # Accepted structures:
        #
        #   explicit değil
        #   UD neg relation
        #   negated copula
        #   negated auxiliary
        #   negated light verb
        #
        # Examples:
        #
        #   güzel <- değildi
        #
        #   tavsiye <- etmiyorum
        #              compound:lvc
        # =====================================================

        for child in children.get(
                sentiment_id,
                []
        ):

            child_id = (
                child[
                    "id"
                ]
            )

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
        # CASE 3
        #
        # Negated syntactic parent.
        #
        # OLD BEHAVIOUR:
        #
        #   any parent with Polarity=Neg
        #       -> sentiment flip
        #
        # This is too broad.
        #
        # NEW BEHAVIOUR:
        #
        # Only accept:
        #
        #   A) explicit 'değil'
        #
        #      kötü -> değildi
        #
        #      This preserves known Stanza structures where the
        #      copular negation becomes the parent/root.
        #
        #   B) reverse light-verb relation
        #
        #      tavsiye -> etme
        #
        #      if Stanza attaches the nominal sentiment component
        #      underneath a morphologically negated light verb.
        #
        # Arbitrary negated verbal parents are NOT accepted.
        # =====================================================

        head_id = (
            sentiment_token.get(
                "head"
            )
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

                    and self._has_morphological_negation(
                head
            )
            )

            if (
                    explicit_degil_parent
                    or reverse_light_verb_negation
            ):
                negations.append(
                    head
                )

                seen.add(
                    head[
                        "id"
                    ]
                )

        # =====================================================
        # SERIALIZE
        # =====================================================

        return [
            {
                "id":
                    token[
                        "id"
                    ],

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

    def _find_contrast_marker(
            self,
            sentiment_token,
            target,
            token_by_id,
            children
    ):
        """
        Detects whether a sentiment expression belongs to
        a contrastive/adversative clause.

        The conjunction marker may attach to:
            1. the sentiment predicate itself,
            2. its syntactic head,
            3. the resolved target/entity of the sentiment predicate.

        Example:

            Otel güzel ama odalar kötü.

            Stanza may produce:

                kötü -> güzel     (conj)
                odalar -> kötü    (nsubj)
                ama -> odalar     (cc)

        Therefore, looking only at the sentiment token is
        insufficient.
        """

        candidate_ids = {
            sentiment_token["id"]
        }

        # -------------------------------------------------
        # 1. Sentiment token's syntactic head
        # -------------------------------------------------

        head_id = sentiment_token.get("head")

        if head_id:
            candidate_ids.add(head_id)

        # -------------------------------------------------
        # 2. Resolved target/entity
        # -------------------------------------------------

        if target is not None:
            candidate_ids.add(
                target["id"]
            )

        # -------------------------------------------------
        # Look for a contrast conjunction attached to any
        # of these structurally related nodes.
        # -------------------------------------------------

        for candidate_id in candidate_ids:

            for child in children.get(
                    candidate_id,
                    []
            ):

                lemma = self._turkish_lower(
                    child.get("lemma")
                    or child.get("text")
                )

                if (
                        child.get("deprel") == "cc"
                        and lemma in self.CONTRAST_WORDS
                ):
                    return {
                        "id": child["id"],
                        "word": child.get("text"),
                        "lemma": lemma,
                        "anchor_id": candidate_id
                    }

        return None

    def _resolve_sentiment_roles(self, results):
        """
        Uses dependency-resolved targets to distinguish opinion
        expressions from entities/targets.

        If a HisNet sentiment candidate is itself the resolved target
        of another sentiment expression and does not have its own target,
        it is treated as an entity in this context rather than as an
        independent opinion expression.

        Example:
            "yemekler kötüydü"

            HisNet may assign sentiment to both:
                yemek
                kötü

            Dependency structure shows:
                yemekler = target of kötü

            Therefore:
                yemekler -> TARGET
                kötü     -> OPINION
        """

        target_token_ids = set()

        for result in results:

            target = result.get("target")

            if target is not None:
                target_token_ids.add(
                    target["id"]
                )

        for result in results:

            sentiment_token = result.get(
                "sentiment_token"
            )

            result["active_sentiment"] = True
            result["contextual_role"] = "opinion"
            result["suppression_reason"] = None

            if sentiment_token is None:
                continue

            token_id = sentiment_token["id"]

            # Conservative suppression:
            # suppress only when this token is used as somebody
            # else's target and it has no target of its own.
            if (
                    token_id in target_token_ids
                    and result.get("target") is None
            ):
                result["active_sentiment"] = False
                result["contextual_role"] = "target"

                result["suppression_reason"] = (
                    "used_as_target_of_another_sentiment"
                )

        return results

    def _find_morphological_root(
            self,
            dependency_token,
            morphological_analyses
    ):
        """
        Returns the canonical morphological root of a dependency token.

        Dependency structure comes from Stanza, but lexical normalization
        comes from the Starlang morphological analyzer used by our
        HisNet pipeline.
        """

        surface = self._turkish_lower(
            dependency_token.get("text")
        )

        if morphological_analyses:

            for analysis in morphological_analyses:

                analysis_surface = self._turkish_lower(
                    analysis.get("word")
                )

                if analysis_surface == surface:
                    return self._turkish_lower(
                        analysis.get("root")
                    )

        # Fallback only if morphological analysis cannot be aligned.
        return self._turkish_lower(
            dependency_token.get("lemma")
            or dependency_token.get("text")
        )

    def _is_dependency_opinion_candidate(
            self,
            token,
            target,
            target_strategy
    ):
        """
        Determines whether a dependency token should be sent
        to HisNet as an opinion candidate.

        IMPORTANT
        ---------
        This decision is made BEFORE any HisNet lookup.

        Therefore this method may use only syntactic information:
            - POS
            - dependency relation
            - resolved target
            - target-resolution strategy

        It must NOT use polarity or HisNet scores.

        Main principle:

            semantic/predicate token
                    -> HisNet candidate

            grammatical/support/modifier token
                    -> scope information only

        Examples
        --------

        Film güzel.
            güzel -> candidate
            Film  -> target

        Bu ürünü tavsiye etmiyorum.
            tavsiye   -> candidate
            etmiyorum -> negation/light-verb support, NOT candidate
            ürünü     -> target

        Otel çok güzel.
            güzel -> candidate
            çok   -> modifier, NOT candidate

        Yemekler kötüydü.
            kötü     -> candidate
            yemekler -> target, NOT candidate
        """

        if token is None:
            return False, None

        upos = (
                token.get("upos")
                or ""
        )

        deprel = (
                token.get("deprel")
                or ""
        )

        # =====================================================
        # 1. POS FILTER
        # =====================================================

        if upos not in self.OPINION_POS:
            return False, None

        # =====================================================
        # 2. HARD STRUCTURAL EXCLUSIONS
        #
        # These relations typically describe support,
        # grammatical function, modification or coordination
        # markers rather than independent opinion expressions.
        #
        # Examples:
        #
        #   etmiyorum -> compound:lvc
        #   çok       -> advmod
        #   kesinlikle -> nmod (possible Stanza analysis)
        #   değildi   -> cop
        #
        # They remain available to:
        #   negation detection
        #   modifier detection
        #   contrast detection
        #
        # but they are NOT sent independently to HisNet.
        # =====================================================

        if deprel in self.OPINION_EXCLUDED_RELATIONS:
            return False, None

        # =====================================================
        # 3. DIRECT PREDICATE / CLAUSAL CANDIDATE
        #
        # Handles:
        #
        #   Harika!
        #   Film güzel.
        #   tavsiye ...
        #   conjunction predicates
        #
        # Target is not mandatory here because one-word /
        # targetless sentiment expressions must survive.
        # =====================================================

        if deprel in self.OPINION_PREDICATE_RELATIONS:

            if target is not None:
                return (
                    True,
                    "predicate_with_target"
                )

            return (
                True,
                "predicate_candidate"
            )

        # =====================================================
        # 4. ADJECTIVAL MODIFIER
        #
        # güzel otel
        #
        # güzel --amod--> otel
        #
        # _find_target() should have resolved the modified noun.
        # =====================================================

        if (
                deprel == "amod"
                and target is not None
                and target_strategy == "modified_head"
        ):
            return (
                True,
                "adjectival_modifier"
            )

        # =====================================================
        # 5. STRONG DIRECT TARGET RELATION
        #
        # The token itself directly governs a subject/object.
        #
        # Examples:
        #
        #   Film güzel.
        #   ürünü tavsiye ...
        #
        # This is stronger evidence than finding a target only
        # through a shared syntactic head.
        # =====================================================

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

        # =====================================================
        # 6. CONSERVATIVE SHARED-HEAD FALLBACK
        #
        # Needed because Stanza can produce structures such as:
        #
        #   Odalar kötü değildi.
        #
        # where:
        #
        #   kötü   -> değildi
        #   Odalar -> değildi
        #
        # and the opinion/target therefore appear as siblings.
        #
        # We allow this fallback ONLY for ADJ or VERB.
        #
        # We deliberately do NOT allow NOUN/ADV here because
        # this was the source of false candidates such as:
        #
        #   kesinlikle -> target=ürünü
        #
        # via a shared head.
        # =====================================================

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

        # =====================================================
        # OTHERWISE:
        # Dependency does not provide sufficient syntactic
        # evidence that this token is an opinion expression.
        # =====================================================

        # =====================================================
        # 7. DESCRIPTIVE CONTENT FALLBACK
        #
        # Real-world Turkish reviews contain many constructions
        # where Stanza attaches sentiment-bearing adjectives or
        # adverbs with relations that are not covered by our
        # stronger predicate/target rules.
        #
        # Examples observed in TRSAv1:
        #
        #   çok az
        #   güzel sürülüyor
        #   zor çıkıyor
        #   hiç sevmedim
        #
        # Dependency is still used first:
        #
        #   - structural/support relations have already been
        #     excluded,
        #   - clear target roles have already been handled,
        #
        # but HisNet is allowed to decide whether a remaining
        # descriptive token actually carries sentiment.
        # =====================================================

        if upos in {
            "ADJ",
            "ADV",
            "INTJ"
        }:
            return (
                True,
                "descriptive_content_fallback"
            )


        return False, None


    def resolve_dependency_candidates(
        self,
        dependency_tokens,
        morphological_analyses=None
    ):
        """
        DEPENDENCY-FIRST candidate extraction.

        Unlike resolve(), this method does NOT receive HisNet
        sentiment words.

        Pipeline:

            dependency tokens
                ->
            syntactic opinion candidates
                ->
            target / negation / modifier / contrast
                ->
            morphological root

        HisNet lookup is intentionally performed AFTER this
        method by DependencyHisNetClassifier.
        """

        token_by_id, children = (
            self._build_indexes(
                dependency_tokens
            )
        )

        candidates = []

        for token in dependency_tokens:

            # ---------------------------------------------
            # First resolve a possible target using syntax.
            # ---------------------------------------------

            target, target_strategy = (
                self._find_target(
                    token,
                    token_by_id,
                    children
                )
            )

            # ---------------------------------------------
            # Decide candidate status WITHOUT HisNet.
            # ---------------------------------------------

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

            # ---------------------------------------------
            # Dependency scope information is resolved
            # BEFORE lexicon lookup.
            # ---------------------------------------------

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

            # ---------------------------------------------
            # Morphological root.
            #
            # Stanza lemma is NOT used as our primary
            # lexical normalization source.
            # ---------------------------------------------

            root = (
                self._find_morphological_root(
                    token,
                    morphological_analyses
                )
            )

            # ---------------------------------------------
            # Target serialization
            # ---------------------------------------------

            if target is not None:

                target_data = {
                    "id":
                        target["id"],

                    "word":
                        target.get("text"),

                    "lemma":
                        target.get("lemma"),

                    "root":
                        self._find_morphological_root(
                            target,
                            morphological_analyses
                        ),

                    "upos":
                        target.get("upos"),

                    "deprel":
                        target.get("deprel")
                }

            else:

                target_data = None

            # ---------------------------------------------
            # Candidate output.
            #
            # Notice:
            # There are NO HisNet POS/NEG scores here.
            # ---------------------------------------------

            candidates.append({

                "word":
                    token.get("text"),

                "root":
                    root,

                "sentiment_token": {
                    "id":
                        token["id"],

                    "word":
                        token.get("text"),

                    "lemma":
                        token.get("lemma"),

                    "upos":
                        token.get("upos"),

                    "head":
                        token.get("head"),

                    "deprel":
                        token.get("deprel")
                },

                "candidate_strategy":
                    candidate_strategy,

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
                    "matched"
            })

        return candidates
    def resolve(
        self,
        sentiment_words,
        dependency_tokens,
        morphological_analyses = None
    ):
        """
        Resolves scope for all HisNet sentiment-bearing words.
        """

        token_by_id, children = (
            self._build_indexes(
                dependency_tokens
            )
        )

        used_token_ids = set()

        results = []

        for sentiment_word in sentiment_words:

            sentiment_token = (
                self._match_sentiment_token(
                    sentiment_word,
                    dependency_tokens,
                    used_token_ids
                )
            )

            if sentiment_token is None:
                results.append({
                    "word":
                        sentiment_word["word"],

                    "root":
                        sentiment_word["root"],

                    # Preserve HisNet scores even when the
                    # sentiment candidate cannot be aligned
                    # with a Stanza token.
                    "positive":
                        sentiment_word["positive"],

                    "negative":
                        sentiment_word["negative"],

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
                        "unmatched"
                })

                continue

            target, target_strategy = (
                self._find_target(
                    sentiment_token,
                    token_by_id,
                    children
                )
            )

            modifiers = self._find_modifiers(
                sentiment_token,
                children
            )

            negations = self._find_negations(
                sentiment_token,
                token_by_id,
                children
            )

            contrast = self._find_contrast_marker(
                sentiment_token,
                target,
                token_by_id,
                children
            )

            if target is not None:

                target_data = {
                    "id": target["id"],
                    "word": target.get("text"),
                    "lemma": target.get("lemma"),

                    "root": self._find_morphological_root(
                        target,
                        morphological_analyses
                    ),

                    "upos": target.get("upos"),
                    "deprel": target.get("deprel")
                }

            else:

                target_data = None

            results.append({
                "word":
                    sentiment_word["word"],

                "root":
                    sentiment_word["root"],

                "positive":
                    sentiment_word["positive"],

                "negative":
                    sentiment_word["negative"],

                "sentiment_token": {
                    "id":
                        sentiment_token["id"],

                    "word":
                        sentiment_token.get("text"),

                    "lemma":
                        sentiment_token.get("lemma"),

                    "upos":
                        sentiment_token.get("upos"),

                    "head":
                        sentiment_token.get("head"),

                    "deprel":
                        sentiment_token.get("deprel")
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
                    "matched"
            })

        return self._resolve_sentiment_roles(
            results
        )