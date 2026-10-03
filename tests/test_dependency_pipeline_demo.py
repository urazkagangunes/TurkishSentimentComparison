from src.hisnet.hisnet_classifier import (
    HisNetClassifier,
    NEUTRAL_THRESHOLD
)

from src.hisnet.hisnet_dependency_classifier import (
    HisNetDependencyClassifier
)


# =====================================================================
# DEMO SENTENCES
# =====================================================================
#
# Bu örnekleri özellikle farklı dependency kullanım biçimlerini
# göstermek için seçiyoruz.
#
# 1) Basit adjective -> subject target
# 2) Explicit negation
# 3) Light-verb negation
# 4) Contrast + multiple sentiment expressions
# 5) Multi-sentence input: Stanza'nın sentence splitting davranışı
# =====================================================================

EXAMPLES = [

    (
        "BASIC TARGET",
        "Film güzel."
    ),

    (
        "EXPLICIT NEGATION",
        "Film güzel değildi."
    ),

    (
        "LIGHT VERB NEGATION",
        "Bu ürünü kesinlikle tavsiye etmiyorum."
    ),

    (
        "CONTRAST",
        "Otel güzeldi ama odalar kötüydü."
    ),

    (
        "MULTI-SENTENCE",
        (
            "Otel çok güzeldi. "
            "Odalar kötü değildi ama personeli tavsiye etmiyorum."
        )
    )
]


# =====================================================================
# HELPERS
# =====================================================================

def line(
    character="=",
    length=110
):
    print(
        character
        * length
    )


def safe(
    value,
    default="-"
):
    if value is None:
        return default

    return str(
        value
    )


def fmt_score(
    value
):
    try:
        return (
            f"{float(value):.4f}"
        )
    except (
        TypeError,
        ValueError
    ):
        return "-"


# =====================================================================
# DEPENDENCY TREE DISPLAY
# =====================================================================

def print_dependency_sentence(
    sentence_number,
    sentence
):
    """
    Stanza'nın ürettiği dependency parse'ı hem tablo hem de
    okunabilir HEAD --relation--> CHILD formatında gösterir.
    """

    tokens = sentence[
        "tokens"
    ]

    token_by_id = {
        token["id"]:
            token
        for token in tokens
    }

    print(
        f"\nSENTENCE {sentence_number}: "
        f"{sentence['text']}"
    )

    print(
        "\nTOKEN TABLE"
    )

    print(
        "-"
        * 110
    )

    print(
        f"{'ID':<5}"
        f"{'TOKEN':<20}"
        f"{'LEMMA':<20}"
        f"{'UPOS':<10}"
        f"{'HEAD':<7}"
        f"{'HEAD WORD':<20}"
        f"{'DEPREL':<18}"
        f"FEATS"
    )

    print(
        "-"
        * 110
    )

    for token in tokens:

        token_id = (
            token[
                "id"
            ]
        )

        head_id = (
            token.get(
                "head",
                0
            )
        )

        if head_id == 0:

            head_word = (
                "ROOT"
            )

        else:

            head_token = (
                token_by_id.get(
                    head_id
                )
            )

            head_word = (
                head_token.get(
                    "text"
                )
                if head_token
                else "?"
            )

        print(
            f"{token_id:<5}"
            f"{safe(token.get('text')):<20}"
            f"{safe(token.get('lemma')):<20}"
            f"{safe(token.get('upos')):<10}"
            f"{head_id:<7}"
            f"{safe(head_word):<20}"
            f"{safe(token.get('deprel')):<18}"
            f"{safe(token.get('feats'))}"
        )

    # -----------------------------------------------------------------
    # Dependency relations in human-readable form
    # -----------------------------------------------------------------

    print(
        "\nDEPENDENCY EDGES"
    )

    print(
        "-"
        * 110
    )

    for token in tokens:

        head_id = (
            token.get(
                "head",
                0
            )
        )

        relation = (
            token.get(
                "deprel"
            )
        )

        child_word = (
            token.get(
                "text"
            )
        )

        if head_id == 0:

            print(
                f"ROOT "
                f"--{relation}--> "
                f"{child_word}"
            )

            continue

        head_token = (
            token_by_id.get(
                head_id
            )
        )

        head_word = (
            head_token.get(
                "text"
            )
            if head_token
            else "?"
        )

        print(
            f"{head_word} "
            f"--{relation}--> "
            f"{child_word}"
        )


# =====================================================================
# GLOBAL TOKEN ID DISPLAY
# =====================================================================

def print_global_token_mapping(
    parsed_sentences
):
    """
    Stanza token ID'leri her yeni cümlede tekrar 1'den başlar.

    Biz document-level scope resolution yaparken token ID çakışması
    olmaması için bunları global document ID'lerine dönüştürüyoruz.

    Bu fonksiyon bu dönüşümü açıklama amacıyla gösterir.
    """

    print(
        "\nDOCUMENT-LEVEL TOKEN ID MAPPING"
    )

    print(
        "-"
        * 110
    )

    print(
        f"{'SENT':<7}"
        f"{'LOCAL ID':<12}"
        f"{'GLOBAL ID':<12}"
        f"{'TOKEN':<25}"
        f"{'LOCAL HEAD':<13}"
        f"{'GLOBAL HEAD':<13}"
        f"DEPREL"
    )

    print(
        "-"
        * 110
    )

    offset = 0

    for sentence_number, sentence in enumerate(
        parsed_sentences,
        start=1
    ):

        tokens = (
            sentence[
                "tokens"
            ]
        )

        if not tokens:
            continue

        max_local_id = max(
            token["id"]
            for token
            in tokens
        )

        for token in tokens:

            local_id = (
                token[
                    "id"
                ]
            )

            local_head = (
                token.get(
                    "head",
                    0
                )
            )

            global_id = (
                local_id
                + offset
            )

            if local_head == 0:

                global_head = 0

            else:

                global_head = (
                    local_head
                    + offset
                )

            print(
                f"{sentence_number:<7}"
                f"{local_id:<12}"
                f"{global_id:<12}"
                f"{safe(token.get('text')):<25}"
                f"{local_head:<13}"
                f"{global_head:<13}"
                f"{safe(token.get('deprel'))}"
            )

        offset += (
            max_local_id
        )


# =====================================================================
# HISNET DISPLAY
# =====================================================================

def print_hisnet_result(
    result
):

    print(
        "\nHISNET BASELINE"
    )

    print(
        "-"
        * 110
    )

    print(
        "Prediction:",
        result[
            "prediction"
        ]
    )

    print(
        "Positive score:",
        fmt_score(
            result[
                "positive_score"
            ]
        )
    )

    print(
        "Negative score:",
        fmt_score(
            result[
                "negative_score"
            ]
        )
    )

    net_score = (
        result[
            "positive_score"
        ]
        - result[
            "negative_score"
        ]
    )

    print(
        "Net score (POS - NEG):",
        fmt_score(
            net_score
        )
    )

    print(
        "Neutral threshold:",
        NEUTRAL_THRESHOLD
    )

    print(
        "\nMorphological analyses:"
    )

    for analysis in result.get(
        "morphological_analyses",
        []
    ):

        print(
            "  "
            f"{safe(analysis.get('word')):<20}"
            " -> "
            f"root={safe(analysis.get('root')):<15}"
            f"analysis={safe(analysis.get('analysis'))}"
        )

    print(
        "\nHisNet sentiment candidates:"
    )

    sentiment_words = (
        result.get(
            "sentiment_words",
            []
        )
    )

    if not sentiment_words:

        print(
            "  No sentiment-bearing "
            "HisNet candidate."
        )

        return

    for index, word in enumerate(
        sentiment_words,
        start=1
    ):

        print(
            f"  [{index}] "
            f"surface={safe(word.get('word')):<20}"
            f"root={safe(word.get('root')):<15}"
            f"POS={fmt_score(word.get('positive')):<10}"
            f"NEG={fmt_score(word.get('negative'))}"
        )


# =====================================================================
# SCOPE DISPLAY
# =====================================================================

def get_scopes(
    dependency_result
):
    """
    Current implementation keeps canonical scopes at document level.

    Fallback is retained so the demo remains readable if scopes are
    grouped under sentence diagnostics.
    """

    scopes = (
        dependency_result.get(
            "scopes"
        )
    )

    if scopes is not None:

        return scopes

    scopes = []

    for sentence in dependency_result.get(
        "sentences",
        []
    ):

        scopes.extend(
            sentence.get(
                "scopes",
                []
            )
        )

    return scopes


def print_scope(
    index,
    scope
):

    print(
        "\n"
        f"SCOPE {index}"
    )

    print(
        "-"
        * 110
    )

    # -----------------------------------------------------------------
    # HisNet lexical candidate
    # -----------------------------------------------------------------

    print(
        "HisNet candidate:"
    )

    print(
        "  Surface:",
        safe(
            scope.get(
                "word"
            )
        )
    )

    print(
        "  Morphological root:",
        safe(
            scope.get(
                "root"
            )
        )
    )

    print(
        "  Original POS:",
        fmt_score(
            scope.get(
                "original_positive",
                scope.get(
                    "positive",
                    0.0
                )
            )
        )
    )

    print(
        "  Original NEG:",
        fmt_score(
            scope.get(
                "original_negative",
                scope.get(
                    "negative",
                    0.0
                )
            )
        )
    )

    # -----------------------------------------------------------------
    # Alignment
    # -----------------------------------------------------------------

    print(
        "\nAlignment:"
    )

    print(
        "  Status:",
        safe(
            scope.get(
                "alignment_status"
            )
        )
    )

    sentiment_token = (
        scope.get(
            "sentiment_token"
        )
    )

    if sentiment_token:

        print(
            "  Dependency token:",
            (
                f"id={safe(sentiment_token.get('id'))}, "
                f"word={safe(sentiment_token.get('word'))}, "
                f"lemma={safe(sentiment_token.get('lemma'))}, "
                f"upos={safe(sentiment_token.get('upos'))}, "
                f"head={safe(sentiment_token.get('head'))}, "
                f"deprel={safe(sentiment_token.get('deprel'))}"
            )
        )

    else:

        print(
            "  Dependency token: "
            "UNMATCHED"
        )

    # -----------------------------------------------------------------
    # Target
    # -----------------------------------------------------------------

    print(
        "\nTarget resolution:"
    )

    target = (
        scope.get(
            "target"
        )
    )

    if target:

        print(
            "  Target:",
            (
                f"id={safe(target.get('id'))}, "
                f"word={safe(target.get('word'))}, "
                f"root={safe(target.get('root'))}, "
                f"upos={safe(target.get('upos'))}, "
                f"deprel={safe(target.get('deprel'))}"
            )
        )

        print(
            "  Strategy:",
            safe(
                scope.get(
                    "target_strategy"
                )
            )
        )

    else:

        print(
            "  Target: None"
        )

        print(
            "  Strategy:",
            safe(
                scope.get(
                    "target_strategy"
                )
            )
        )

    # -----------------------------------------------------------------
    # Contextual role
    # -----------------------------------------------------------------

    print(
        "\nContextual sentiment role:"
    )

    print(
        "  Role:",
        safe(
            scope.get(
                "contextual_role"
            )
        )
    )

    print(
        "  Active sentiment:",
        safe(
            scope.get(
                "active_sentiment"
            )
        )
    )

    print(
        "  Suppression reason:",
        safe(
            scope.get(
                "suppression_reason"
            )
        )
    )

    # -----------------------------------------------------------------
    # Modifiers
    # -----------------------------------------------------------------

    modifiers = (
        scope.get(
            "modifiers",
            []
        )
    )

    print(
        "\nModifiers:"
    )

    if not modifiers:

        print(
            "  None"
        )

    else:

        for modifier in modifiers:

            print(
                "  "
                f"id={safe(modifier.get('id'))}, "
                f"word={safe(modifier.get('word'))}, "
                f"lemma={safe(modifier.get('lemma'))}, "
                f"deprel={safe(modifier.get('deprel'))}"
            )

    # -----------------------------------------------------------------
    # Negations
    # -----------------------------------------------------------------

    negations = (
        scope.get(
            "negations",
            []
        )
    )

    print(
        "\nNegations:"
    )

    if not negations:

        print(
            "  None"
        )

    else:

        for negation in negations:

            print(
                "  "
                f"id={safe(negation.get('id'))}, "
                f"word={safe(negation.get('word'))}, "
                f"lemma={safe(negation.get('lemma'))}, "
                f"deprel={safe(negation.get('deprel'))}, "
                f"feats={safe(negation.get('feats'))}"
            )

    # -----------------------------------------------------------------
    # Contrast
    # -----------------------------------------------------------------

    contrast = (
        scope.get(
            "contrast"
        )
    )

    print(
        "\nContrast:"
    )

    if contrast:

        print(
            "  "
            f"word={safe(contrast.get('word'))}, "
            f"lemma={safe(contrast.get('lemma'))}, "
            f"anchor_id={safe(contrast.get('anchor_id'))}"
        )

    else:

        print(
            "  None"
        )

    # -----------------------------------------------------------------
    # Dependency operations
    # -----------------------------------------------------------------

    operations = (
        scope.get(
            "operations",
            []
        )
    )

    print(
        "\nHOW DEPENDENCY IS USED:"
    )

    if not operations:

        print(
            "  No score-changing or "
            "diagnostic operation."
        )

    else:

        for operation in operations:

            if (
                operation
                == "negation_flip"
            ):

                explanation = (
                    "negation scope detected -> "
                    "POS and NEG scores are swapped"
                )

            elif (
                operation
                == "suppressed_as_target"
            ):

                explanation = (
                    "candidate functions as a target/entity -> "
                    "lexical sentiment contribution is suppressed"
                )

            elif (
                operation
                == "modifier_detected"
            ):

                explanation = (
                    "modifier detected -> "
                    "currently recorded; numeric multiplier "
                    "not yet applied"
                )

            elif (
                operation
                == "contrast_detected"
            ):

                explanation = (
                    "contrast marker detected -> "
                    "currently recorded; numeric weighting "
                    "not yet applied"
                )

            else:

                explanation = (
                    operation
                )

            print(
                f"  {operation}: "
                f"{explanation}"
            )

    print(
        "\nAdjusted contribution:"
    )

    print(
        "  POS:",
        fmt_score(
            scope.get(
                "adjusted_positive",
                scope.get(
                    "positive",
                    0.0
                )
            )
        )
    )

    print(
        "  NEG:",
        fmt_score(
            scope.get(
                "adjusted_negative",
                scope.get(
                    "negative",
                    0.0
                )
            )
        )
    )


# =====================================================================
# FINAL DEPENDENCY RESULT
# =====================================================================

def print_dependency_result(
    result
):

    scopes = (
        get_scopes(
            result
        )
    )

    print(
        "\n"
    )

    line()

    print(
        "DEPENDENCY-BASED SCOPE RESOLUTION"
    )

    line()

    if not scopes:

        print(
            "No resolved sentiment scopes."
        )

    else:

        for index, scope in enumerate(
            scopes,
            start=1
        ):

            print_scope(
                index,
                scope
            )

    print(
        "\n"
    )

    line()

    print(
        "FINAL HISNET + DEPENDENCY RESULT"
    )

    line()

    print(
        "Final POS:",
        fmt_score(
            result.get(
                "positive_score"
            )
        )
    )

    print(
        "Final NEG:",
        fmt_score(
            result.get(
                "negative_score"
            )
        )
    )

    positive_score = float(
        result.get(
            "positive_score",
            0.0
        )
    )

    negative_score = float(
        result.get(
            "negative_score",
            0.0
        )
    )

    net_score = (
        positive_score
        - negative_score
    )

    print(
        "Final net score "
        "(POS - NEG):",
        fmt_score(
            net_score
        )
    )

    print(
        "Neutral threshold:",
        NEUTRAL_THRESHOLD
    )

    print(
        "Final prediction:",
        result.get(
            "prediction"
        )
    )


# =====================================================================
# INITIALIZE ONCE
# =====================================================================

print(
    "Initializing HisNet..."
)

hisnet = (
    HisNetClassifier()
)


print(
    "Initializing HisNet + Dependency..."
)

dependency_classifier = (
    HisNetDependencyClassifier(
        hisnet_classifier=hisnet
    )
)


parser = (
    dependency_classifier
    .parser
)


# =====================================================================
# RUN DEMOS
# =====================================================================

for (
    example_name,
    text
) in EXAMPLES:

    print(
        "\n\n"
    )

    line(
        "=",
        120
    )

    print(
        f"EXAMPLE: {example_name}"
    )

    line(
        "=",
        120
    )

    print(
        "\nINPUT TEXT:"
    )

    print(
        text
    )

    # =============================================================
    # STEP 1 — STANZA PARSING
    # =============================================================

    parsed_sentences = (
        parser.parse(
            text
        )
    )

    print(
        "\n"
    )

    line()

    print(
        "STEP 1 — STANZA SENTENCE SPLITTING"
    )

    line()

    print(
        "Number of sentences:",
        len(
            parsed_sentences
        )
    )

    for index, sentence in enumerate(
        parsed_sentences,
        start=1
    ):

        print(
            f"  [{index}] "
            f"{sentence['text']}"
        )

    # =============================================================
    # STEP 2 — DEPENDENCY PARSE
    # =============================================================

    print(
        "\n"
    )

    line()

    print(
        "STEP 2 — DEPENDENCY PARSE"
    )

    line()

    for sentence_number, sentence in enumerate(
        parsed_sentences,
        start=1
    ):

        print_dependency_sentence(
            sentence_number,
            sentence
        )

    # =============================================================
    # STEP 3 — DOCUMENT LEVEL IDS
    # =============================================================

    print(
        "\n"
    )

    line()

    print(
        "STEP 3 — DOCUMENT-LEVEL TOKEN IDS"
    )

    line()

    print(
        "Stanza restarts token IDs for every sentence. "
        "For document-level sentiment scope resolution, "
        "we use globally unique token IDs."
    )

    print_global_token_mapping(
        parsed_sentences
    )

    # =============================================================
    # STEP 4 — HISNET
    # =============================================================

    print(
        "\n"
    )

    line()

    print(
        "STEP 4 — HISNET LEXICAL SENTIMENT"
    )

    line()

    hisnet_result = (
        hisnet.classify(
            text
        )
    )

    print_hisnet_result(
        hisnet_result
    )

    # =============================================================
    # STEP 5 — HISNET + DEPENDENCY
    # =============================================================

    print(
        "\n"
    )

    line()

    print(
        "STEP 5 — COMBINE HISNET WITH DEPENDENCY PARSE"
    )

    line()

    print(
        """
Pipeline used by the system:

    RAW TEXT
       |
       +------------------------------+
       |                              |
       v                              v
Morphological Analysis             Stanza
       |                              |
       v                              v
LongestRootFirstDisambiguation   Sentence Split
       |                              |
       v                              v
HisNet Lookup                  Dependency Parse
       |                              |
       +--------------+---------------+
                      |
                      v
             SentimentScopeResolver
                      |
          +-----------+-----------+
          |           |           |
        TARGET     NEGATION    MODIFIER
          |           |           |
          +-----------+-----------+
                      |
                  CONTRAST
                      |
                      v
          Dependency-aware composition
                      |
                      v
             POS score / NEG score
                      |
                      v
               POS - NEG
                      |
          +-----------+-----------+
          |                       |
     outside ±0.05            inside ±0.05
          |                       |
      POS / NEG                 NEUTRAL
"""
    )

    dependency_result = (
        dependency_classifier
        .classify(
            text,
            hisnet_result=
                hisnet_result
        )
    )

    print_dependency_result(
        dependency_result
    )


# =====================================================================
# DONE
# =====================================================================

print(
    "\n\n"
)

line(
    "=",
    120
)

print(
    "DEPENDENCY PIPELINE DEMO COMPLETED"
)

line(
    "=",
    120
)