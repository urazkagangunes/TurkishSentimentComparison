import time
from typing import List

from src.dependency.stanza_parser import (
    TurkishDependencyParser
)

from src.llm.ollama_zero_shot_classifier import (
    OllamaZeroShotSentimentClassifier
)


class DependencyOllamaZeroShotSentimentClassifier(
    OllamaZeroShotSentimentClassifier
):
    """
    Dependency -> LLM Turkish sentiment classifier.

    Architecture
    ------------
    Text
        -> Stanford Stanza dependency parsing
        -> dependency representation
        -> raw text + dependency representation
        -> same zero-shot LLM used by LLM-only baseline
        -> positive / negative / neutral

    Experimental principle
    ----------------------
    The ONLY intended difference between:

        LLM

    and:

        Dependency -> LLM

    is the presence of dependency information in the prompt.

    Therefore this classifier deliberately does NOT use:

        - HisNet
        - SentiLiteralNet
        - HisNet sentiment scores
        - morphological root filtering
        - SentimentScopeResolver
        - hand-written negation flipping
        - target suppression
        - modifier weighting
        - contrast weighting

    The LLM receives the Stanza dependency parse directly and
    decides how to use the syntactic information.
    """

    def __init__(
        self,
        use_gpu_parser: bool = False,
        **llm_kwargs
    ):

        # =====================================================
        # SAME LLM BACKEND AS THE LLM-ONLY BASELINE
        # =====================================================

        super().__init__(
            **llm_kwargs
        )

        # =====================================================
        # DEPENDENCY PARSER
        # =====================================================

        self.parser = (
            TurkishDependencyParser(
                use_gpu=use_gpu_parser
            )
        )

        # =====================================================
        # TEMPORARY BATCH PARSE CACHE
        #
        # classify_batch() parses the reviews first.
        #
        # Parent classify_batch() then calls self._build_prompt().
        #
        # This temporary cache prevents the same reviews from
        # being parsed a second time while building the prompt.
        # =====================================================

        self._active_dependency_parses = None

        # =====================================================
        # DEPENDENCY RUNTIME STATISTICS
        # =====================================================

        self.total_dependency_parse_seconds = 0.0

        self.total_dependency_reviews = 0

        self.total_dependency_sentences = 0

        self.total_dependency_tokens = 0

    # =========================================================
    # DEPENDENCY SERIALIZATION
    # =========================================================

    @staticmethod
    def _serialize_dependency_parse(
        parsed_sentences
    ) -> str:
        """
        Converts Stanza output into a compact representation
        suitable for an LLM prompt.

        Example:

        SENTENCE 1: Film güzel değildi.

        ID=1 | FORM=Film | LEMMA=film | UPOS=NOUN
             | HEAD=2:güzel | DEPREL=nsubj | FEATS=...

        ID=2 | FORM=güzel | LEMMA=güzel | UPOS=ADJ
             | HEAD=0:ROOT | DEPREL=root | FEATS=...

        ID=3 | FORM=değildi | LEMMA=değil | UPOS=AUX
             | HEAD=2:güzel | DEPREL=cop | FEATS=Polarity=Neg...
        """

        sentence_blocks = []

        for sentence_index, sentence in enumerate(
            parsed_sentences,
            start=1
        ):

            tokens = (
                sentence.get(
                    "tokens",
                    []
                )
            )

            # -------------------------------------------------
            # Local Stanza token IDs restart in every sentence.
            #
            # This lookup is therefore sentence-local.
            # -------------------------------------------------

            token_by_id = {

                int(
                    token[
                        "id"
                    ]
                ):
                    token

                for token in tokens
            }

            lines = [

                (
                    f"SENTENCE {sentence_index}: "
                    f"{sentence.get('text', '')}"
                )
            ]

            for token in tokens:

                token_id = int(
                    token[
                        "id"
                    ]
                )

                head_id = int(
                    token.get(
                        "head",
                        0
                    )
                    or 0
                )

                if head_id == 0:

                    head_text = (
                        "ROOT"
                    )

                else:

                    head_token = (
                        token_by_id.get(
                            head_id
                        )
                    )

                    if head_token is None:

                        head_text = (
                            "UNKNOWN"
                        )

                    else:

                        head_text = (
                            head_token.get(
                                "text",
                                "UNKNOWN"
                            )
                        )

                form = (
                    token.get(
                        "text"
                    )
                    or "_"
                )

                lemma = (
                    token.get(
                        "lemma"
                    )
                    or "_"
                )

                upos = (
                    token.get(
                        "upos"
                    )
                    or "_"
                )

                deprel = (
                    token.get(
                        "deprel"
                    )
                    or "_"
                )

                feats = (
                    token.get(
                        "feats"
                    )
                    or "_"
                )

                lines.append(
                    (
                        f"ID={token_id}"
                        f" | FORM={form}"
                        f" | LEMMA={lemma}"
                        f" | UPOS={upos}"
                        f" | HEAD={head_id}:{head_text}"
                        f" | DEPREL={deprel}"
                        f" | FEATS={feats}"
                    )
                )

            sentence_blocks.append(
                "\n".join(
                    lines
                )
            )

        return (
            "\n\n".join(
                sentence_blocks
            )
        )

    # =========================================================
    # PARSE ONE BATCH
    # =========================================================

    def _parse_batch(
        self,
        texts: List[str]
    ):
        """
        Parses every review independently.

        Returns:
            parsed_reviews
            parse_seconds
        """

        start_time = (
            time.perf_counter()
        )

        parsed_reviews = []

        sentence_count = 0

        token_count = 0

        for text in texts:

            parsed_sentences = (
                self.parser.parse(
                    text
                )
            )

            parsed_reviews.append(
                parsed_sentences
            )

            sentence_count += (
                len(
                    parsed_sentences
                )
            )

            for sentence in parsed_sentences:

                token_count += (
                    len(
                        sentence.get(
                            "tokens",
                            []
                        )
                    )
                )

        parse_seconds = (
            time.perf_counter()
            - start_time
        )

        # =====================================================
        # STATS
        # =====================================================

        self.total_dependency_parse_seconds += (
            parse_seconds
        )

        self.total_dependency_reviews += (
            len(
                texts
            )
        )

        self.total_dependency_sentences += (
            sentence_count
        )

        self.total_dependency_tokens += (
            token_count
        )

        return (
            parsed_reviews,
            parse_seconds
        )

    # =========================================================
    # DEPENDENCY-AWARE ZERO-SHOT PROMPT
    # =========================================================

    def _build_prompt(
        self,
        texts: List[str]
    ) -> str:
        """
        Builds the Dependency -> LLM prompt.

        IMPORTANT:
        There are NO labelled examples.

        Therefore this remains zero-shot.

        The dependency parse is additional syntactic input,
        not a sentiment label and not a hand-written sentiment
        decision.
        """

        parsed_reviews = (
            self._active_dependency_parses
        )

        # -----------------------------------------------------
        # Normally classify_batch() has already parsed them.
        #
        # This fallback makes the method robust if called
        # directly during debugging.
        # -----------------------------------------------------

        if (
            parsed_reviews is None
            or len(
                parsed_reviews
            )
            != len(
                texts
            )
        ):

            parsed_reviews = [

                self.parser.parse(
                    text
                )

                for text in texts
            ]

        review_blocks = []

        for index, (
            text,
            parsed_sentences
        ) in enumerate(

            zip(
                texts,
                parsed_reviews
            ),

            start=1
        ):

            dependency_text = (
                self._serialize_dependency_parse(
                    parsed_sentences
                )
            )

            review_blocks.append(
                (
                    f"[REVIEW {index}]\n"
                    f"{text}\n"
                    f"\n"
                    f"[DEPENDENCY PARSE]\n"
                    f"{dependency_text}\n"
                    f"[/DEPENDENCY PARSE]\n"
                    f"[/REVIEW {index}]"
                )
            )

        reviews = (
            "\n\n".join(
                review_blocks
            )
        )

        # =====================================================
        # KEEP THE CORE TASK INSTRUCTIONS CONSISTENT WITH
        # LLM-ONLY.
        #
        # Only dependency-specific instructions are added.
        # =====================================================

        return (
            "Aşağıda Türkçe kullanıcı yorumları verilmiştir.\n\n"

            "Görevin, HER yorumu diğer yorumlardan TAMAMEN "
            "BAĞIMSIZ olarak genel duygu açısından "
            "sınıflandırmaktır.\n\n"

            "Her yorum için yalnızca şu üç etiketten birini seç:\n"
            "- positive\n"
            "- negative\n"
            "- neutral\n\n"

            "Bir yorumun sınıflandırılmasında diğer yorumların "
            "içeriğini kullanma.\n\n"

            "Her yorumun altında Stanford Stanza tarafından "
            "üretilmiş dependency parse bilgisi bulunmaktadır.\n\n"

            "Dependency parse yalnızca ek sözdizimsel bilgidir. "
            "Bir sentiment etiketi veya doğru cevap değildir.\n\n"

            "Dependency bilgisini özellikle kelimeler arasındaki "
            "bağımlılıkları, özne/nesne ilişkilerini, "
            "niteleyicileri, bağlaçları ve olumsuzluğun hangi "
            "ifadeyi etkilediğini yorumlamaya yardımcı olması "
            "için kullanabilirsin.\n\n"

            "FEATS alanındaki Polarity=Neg ifadesi negatif "
            "sentiment etiketi anlamına gelmez; yalnızca "
            "dilbilgisel olumsuzluk bilgisidir.\n\n"

            "Nihai sentiment kararını yorumun genel anlamına "
            "göre ver.\n\n"

            "Her REVIEW için tam olarak bir sonuç üret.\n"

            "Ek açıklama, gerekçe veya yorum yazma.\n\n"

            f"{reviews}"
        )

    # =========================================================
    # CLASSIFY ONE DEPENDENCY BATCH
    # =========================================================

    def classify_batch(
        self,
        texts: List[str]
    ) -> dict:
        """
        Dependency parsing happens BEFORE the LLM request.

        Parent class still handles:

            - Ollama
            - Qwen
            - think=False
            - structured output
            - retries
            - label validation
            - runtime/token statistics
        """

        # -----------------------------------------------------
        # Fail fast before spending time on Stanza.
        # -----------------------------------------------------

        if len(
            texts
        ) > self.batch_size:

            raise ValueError(
                "classify_batch received more reviews "
                "than configured batch_size. "
                f"batch_size={self.batch_size}, "
                f"received={len(texts)}"
            )

        # =====================================================
        # EMPTY INPUT
        # =====================================================

        if not texts:

            result = (
                super()
                .classify_batch(
                    texts
                )
            )

            result[
                "dependency_parses"
            ] = []

            result[
                "dependency_serializations"
            ] = []

            result[
                "dependency_parse_seconds"
            ] = 0.0

            result[
                "total_pipeline_seconds"
            ] = (
                result.get(
                    "request_seconds",
                    0.0
                )
            )

            return result

        # =====================================================
        # STEP 1
        # DEPENDENCY PARSE
        # =====================================================

        (
            parsed_reviews,
            parse_seconds
        ) = (
            self._parse_batch(
                texts
            )
        )

        # =====================================================
        # STEP 2
        # MAKE PARSES AVAILABLE TO _build_prompt()
        # =====================================================

        self._active_dependency_parses = (
            parsed_reviews
        )

        try:

            # =================================================
            # STEP 3
            # SAME OLLAMA CLASSIFIER AS LLM-ONLY
            # =================================================

            result = (
                super()
                .classify_batch(
                    texts
                )
            )

        finally:

            # Prevent stale parses from leaking into a future
            # batch.

            self._active_dependency_parses = (
                None
            )

        # =====================================================
        # DIAGNOSTICS
        # =====================================================

        dependency_serializations = [

            self._serialize_dependency_parse(
                parsed_sentences
            )

            for parsed_sentences
            in parsed_reviews
        ]

        result[
            "dependency_parses"
        ] = (
            parsed_reviews
        )

        result[
            "dependency_serializations"
        ] = (
            dependency_serializations
        )

        result[
            "dependency_parse_seconds"
        ] = (
            parse_seconds
        )

        result[
            "total_pipeline_seconds"
        ] = (
            parse_seconds
            + result.get(
                "request_seconds",
                0.0
            )
        )

        return result

    # =========================================================
    # SINGLE REVIEW
    # =========================================================

    def classify(
        self,
        text: str
    ) -> dict:

        result = (
            self.classify_batch(
                [text]
            )
        )

        prediction = (
            result[
                "predictions"
            ][0]
        )

        dependency_parse = (
            result[
                "dependency_parses"
            ][0]
            if result[
                "dependency_parses"
            ]
            else []
        )

        dependency_serialization = (
            result[
                "dependency_serializations"
            ][0]
            if result[
                "dependency_serializations"
            ]
            else ""
        )

        return {
            "text":
                text,

            "prediction":
                prediction,

            "status":
                result.get(
                    "status"
                ),

            "error":
                result.get(
                    "error"
                ),

            "model":
                result.get(
                    "model"
                ),

            "request_seconds":
                result.get(
                    "request_seconds",
                    0.0
                ),

            "dependency_parse_seconds":
                result.get(
                    "dependency_parse_seconds",
                    0.0
                ),

            "total_pipeline_seconds":
                result.get(
                    "total_pipeline_seconds",
                    0.0
                ),

            "dependency_parse":
                dependency_parse,

            "dependency_serialization":
                dependency_serialization,

            "attempt":
                result.get(
                    "attempt"
                )
        }

    # =========================================================
    # STATS
    # =========================================================

    def get_stats(
        self
    ) -> dict:

        stats = (
            super()
            .get_stats()
        )

        dependency_seconds_per_review = (

            self.total_dependency_parse_seconds
            / self.total_dependency_reviews

            if self.total_dependency_reviews > 0

            else 0.0
        )

        stats.update({

            "architecture":
                "dependency_to_llm",

            "dependency_parser":
                "stanza_tr",

            "dependency_parse_reviews":
                self.total_dependency_reviews,

            "dependency_parse_sentences":
                self.total_dependency_sentences,

            "dependency_parse_tokens":
                self.total_dependency_tokens,

            "dependency_parse_wall_seconds":
                self.total_dependency_parse_seconds,

            "dependency_parse_seconds_per_review":
                dependency_seconds_per_review,

            # ---------------------------------------------
            # This combines parser time and LLM request
            # time accumulated by this classifier.
            # ---------------------------------------------

            "total_pipeline_wall_seconds":
                (
                    self.total_dependency_parse_seconds
                    + self.total_wall_seconds
                )
        })

        return stats