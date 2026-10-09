from src.hisnet.hisnet_classifier import (
    HisNetClassifier
)

from src.hisnet.literature_weighted_hisnet_dependency_classifier import (
    LiteratureWeightedHisNetDependencyClassifier
)


TEST_SENTENCES = [

    "Otel çok güzel.",

    "Otel biraz kötü.",

    "Otel güzel ama odalar kötü.",

    "Otel kötü ama odalar güzel.",

    "Bu ürün son derece güzel.",

    "Bu ürün çok ama çok güzel.",

    "Film güzel değildi.",

    "Bu ürün çok daha kötü.",

    "Bu ürün daha da çok güzel.",

    "Bu ürün azıcık kötü.",

    "Büyük ekran güzel.",

    "Küçük oda çok güzel.",

    "Otel güzel fakat odalar kötü.",

    "Otel kötü ancak odalar güzel.",

    "Otel güzel ama odalar kötü değildi.",

    "Ürün güzel ama pahalı fakat kaliteli."
]


print(
    "=" * 100
)

print(
    "LITERATURE-WEIGHTED HISNET -> DEPENDENCY TEST"
)

print(
    "=" * 100
)


hisnet = (
    HisNetClassifier()
)


classifier = (
    LiteratureWeightedHisNetDependencyClassifier(
        hisnet_classifier=
            hisnet
    )
)


for index, text in enumerate(
    TEST_SENTENCES,
    start=1
):

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"TEST {index}"
    )

    print(
        "=" * 100
    )

    print(
        "Text:",
        text
    )


    hisnet_result = (
        hisnet.classify(
            text
        )
    )


    result = (
        classifier.classify(

            text,

            hisnet_result=
                hisnet_result
        )
    )


    print(
        "\nHisNet:"
    )

    print(
        "Prediction:",
        hisnet_result[
            "prediction"
        ]
    )

    print(
        "Positive:",
        hisnet_result[
            "positive_score"
        ]
    )

    print(
        "Negative:",
        hisnet_result[
            "negative_score"
        ]
    )


    print(
        "\nH -> D:"
    )

    print(
        "Prediction:",
        result[
            "dependency_prediction"
        ]
    )

    print(
        "Positive:",
        result[
            "dependency_positive_score"
        ]
    )

    print(
        "Negative:",
        result[
            "dependency_negative_score"
        ]
    )


    print(
        "\nH -> D -> Literature Weighted:"
    )

    print(
        "Prediction:",
        result[
            "prediction"
        ]
    )

    print(
        "Positive:",
        result[
            "positive_score"
        ]
    )

    print(
        "Negative:",
        result[
            "negative_score"
        ]
    )


    print(
        "\nScopes:"
    )


    for scope in result[
        "scopes"
    ]:

        print(
            "\n  Word:",
            scope.get(
                "word"
            )
        )

        print(
            "  Root:",
            scope.get(
                "root"
            )
        )

        print(
            "  Active:",
            scope.get(
                "active_sentiment"
            )
        )

        print(
            "  Dependency POS:",
            scope.get(
                "dependency_adjusted_positive"
            )
        )

        print(
            "  Dependency NEG:",
            scope.get(
                "dependency_adjusted_negative"
            )
        )

        print(
            "  Modifier phrase:",
            scope.get(
                "modifier_phrase"
            )
        )

        print(
            "  Modifier multiplier:",
            scope.get(
                "modifier_multiplier"
            )
        )

        print(
            "  Contrast marker:",
            scope.get(
                "contrast_marker"
            )
        )

        print(
            "  Contrast role:",
            scope.get(
                "contrast_role"
            )
        )

        print(
            "  Contrast multiplier:",
            scope.get(
                "contrast_multiplier"
            )
        )

        print(
            "  Final multiplier:",
            scope.get(
                "literature_multiplier"
            )
        )

        print(
            "  Final POS:",
            scope.get(
                "adjusted_positive"
            )
        )

        print(
            "  Final NEG:",
            scope.get(
                "adjusted_negative"
            )
        )

        print(
            "  Operations:",
            scope.get(
                "operations"
            )
        )

    print(
        "Contrast marker suppressed scopes:",
        result[
            "contrast_marker_suppressed_scope_count"
        ]
    )

    print(
        "\nModifier weighted scopes:",
        result[
            "modifier_weighted_scope_count"
        ]
    )

    print(
        "Contrast weighted scopes:",
        result[
            "contrast_weighted_scope_count"
        ]
    )

    print(
        "Contrast relations:",
        result[
            "contrast_relations"
        ]


    )


print(
    "\n"
    + "=" * 100
)

print(
    "LITERATURE-WEIGHTED TEST COMPLETED"
)

print(
    "=" * 100
)