from src.hisnet.dependency_hisnet_classifier import (
    DependencyHisNetClassifier
)


TESTS = [

    {
        "name":
            "BASIC TARGET",

        "text":
            "Film güzel.",

        "expected_prediction":
            "positive",

        "expected_candidates":
            {"güzel"},

        "forbidden_candidates":
            {"film"}
    },

    {
        "name":
            "EXPLICIT NEGATION",

        "text":
            "Film güzel değildi.",

        "expected_prediction":
            "negative",

        "expected_candidates":
            {"güzel"},

        "forbidden_candidates":
            {
                "film",
                "değildi"
            }
    },

    {
        "name":
            "LIGHT VERB NEGATION",

        "text":
            (
                "Bu ürünü kesinlikle "
                "tavsiye etmiyorum."
            ),

        "expected_prediction":
            "negative",

        "expected_candidates":
            {"tavsiye"},

        "forbidden_candidates":
            {
                "bu",
                "ürünü",
                "kesinlikle",
                "etmiyorum"
            }
    },

    {
        "name":
            "TARGET FILTERING",

        "text":
            "Yemekler kötüydü.",

        "expected_prediction":
            "negative",

        "expected_candidates":
            {"kötü"},

        "forbidden_candidates":
            {"yemekler"}
    },

    {
        "name":
            "TARGETLESS SENTIMENT",

        "text":
            "Harika!",

        "expected_prediction":
            "positive",

        "expected_candidates":
            {"harika"},

        "forbidden_candidates":
            set()
    },

    {
        "name":
            "MODIFIER IS NOT OPINION",

        "text":
            "Otel çok güzeldi.",

        "expected_prediction":
            "positive",

        "expected_candidates":
            {"güzeldi"},

        "forbidden_candidates":
            {
                "otel"
            }
    },

    {
        "name":
            "NEGATED ADJECTIVE WITH SHARED HEAD",

        "text":
            "Odalar kötü değildi.",

        "expected_prediction":
            "positive",

        "expected_candidates":
            {"kötü"},

        "forbidden_candidates":
            {
                "odalar",
                "değildi"
            }
    }
]


classifier = (
    DependencyHisNetClassifier()
)


def turkish_lower(
    text
):
    return (
        text
        .replace(
            "I",
            "ı"
        )
        .replace(
            "İ",
            "i"
        )
        .lower()
    )


for test in TESTS:

    print(
        "\n"
        + "=" * 100
    )

    print(
        test[
            "name"
        ]
    )

    print(
        "=" * 100
    )

    print(
        "TEXT:",
        test[
            "text"
        ]
    )

    result = (
        classifier.classify(
            test[
                "text"
            ]
        )
    )

    # =========================================================
    # DEPENDENCY CANDIDATES
    # =========================================================

    print(
        "\nDEPENDENCY CANDIDATES"
    )

    candidate_words = set()

    for candidate in result[
        "dependency_candidates"
    ]:

        word = (
            candidate.get(
                "word"
            )
        )

        normalized_word = (
            turkish_lower(
                word
            )
        )

        candidate_words.add(
            normalized_word
        )

        target = (
            candidate.get(
                "target"
            )
        )

        print(
            "  ",
            word,

            "| root =",
            candidate.get(
                "root"
            ),

            "| POS =",
            candidate[
                "sentiment_token"
            ].get(
                "upos"
            ),

            "| deprel =",
            candidate[
                "sentiment_token"
            ].get(
                "deprel"
            ),

            "| candidate strategy =",
            candidate.get(
                "candidate_strategy"
            ),

            "| target =",
            (
                target.get(
                    "word"
                )
                if target
                else None
            ),

            "| target strategy =",
            candidate.get(
                "target_strategy"
            ),

            "| negations =",
            [
                item.get(
                    "word"
                )

                for item
                in candidate.get(
                    "negations",
                    []
                )
            ],

            "| modifiers =",
            [
                item.get(
                    "word"
                )

                for item
                in candidate.get(
                    "modifiers",
                    []
                )
            ]
        )

    # =========================================================
    # CANDIDATE ASSERTIONS
    # =========================================================

    expected_candidates = {
        turkish_lower(
            item
        )
        for item
        in test[
            "expected_candidates"
        ]
    }

    forbidden_candidates = {
        turkish_lower(
            item
        )
        for item
        in test[
            "forbidden_candidates"
        ]
    }

    print(
        "\nCandidate words:",
        sorted(
            candidate_words
        )
    )

    print(
        "Expected candidates:",
        sorted(
            expected_candidates
        )
    )

    print(
        "Forbidden candidates:",
        sorted(
            forbidden_candidates
        )
    )

    missing_candidates = (
        expected_candidates
        - candidate_words
    )

    unexpected_forbidden = (
        forbidden_candidates
        & candidate_words
    )

    assert not missing_candidates, (
        f"{test['name']}: "
        f"missing expected candidates = "
        f"{missing_candidates}"
    )

    assert not unexpected_forbidden, (
        f"{test['name']}: "
        f"forbidden candidates selected = "
        f"{unexpected_forbidden}"
    )

    # =========================================================
    # HISNET LOOKUP
    # =========================================================

    print(
        "\nHISNET LOOKUP AFTER DEPENDENCY FILTER"
    )

    for candidate in result[
        "candidate_diagnostics"
    ]:

        print(
            "  ",
            candidate.get(
                "word"
            ),

            "| root =",
            candidate.get(
                "root"
            ),

            "| status =",
            candidate.get(
                "lexicon_status"
            ),

            "| POS =",
            candidate.get(
                "positive"
            ),

            "| NEG =",
            candidate.get(
                "negative"
            )
        )

    # =========================================================
    # FINAL SCOPES
    # =========================================================

    print(
        "\nFINAL SENTIMENT SCOPES"
    )

    for scope in result[
        "scopes"
    ]:

        target = (
            scope.get(
                "target"
            )
        )

        print(
            "  ",
            scope.get(
                "word"
            ),

            "| target =",
            (
                target.get(
                    "word"
                )
                if target
                else None
            ),

            "| original =",
            (
                scope.get(
                    "original_positive"
                ),
                scope.get(
                    "original_negative"
                )
            ),

            "| adjusted =",
            (
                scope.get(
                    "adjusted_positive"
                ),
                scope.get(
                    "adjusted_negative"
                )
            ),

            "| operations =",
            scope.get(
                "operations"
            )
        )

    # =========================================================
    # FINAL RESULT
    # =========================================================

    print(
        "\nRESULT"
    )

    print(
        "POS:",
        result[
            "positive_score"
        ]
    )

    print(
        "NEG:",
        result[
            "negative_score"
        ]
    )

    print(
        "NET:",
        result[
            "net_score"
        ]
    )

    print(
        "Prediction:",
        result[
            "prediction"
        ]
    )

    print(
        "Expected:",
        test[
            "expected_prediction"
        ]
    )

    print(
        "Dependency candidates:",
        result[
            "dependency_candidate_count"
        ]
    )

    print(
        "Sentiment candidates:",
        result[
            "sentiment_word_count"
        ]
    )

    assert (
        result[
            "prediction"
        ]
        == test[
            "expected_prediction"
        ]
    ), (
        f"{test['name']}: "
        f"expected="
        f"{test['expected_prediction']}, "
        f"got="
        f"{result['prediction']}"
    )


print(
    "\n"
    + "=" * 100
)

print(
    "ALL STRICT DEPENDENCY -> HISNET TESTS PASSED"
)

print(
    "=" * 100
)