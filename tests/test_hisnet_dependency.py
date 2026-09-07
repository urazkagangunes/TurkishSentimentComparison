from src.hisnet.hisnet_classifier import HisNetClassifier

from src.hisnet.hisnet_dependency_classifier import (
    HisNetDependencyClassifier
)


hisnet = HisNetClassifier()

hisnet_dependency = (
    HisNetDependencyClassifier()
)


sentences = [

    # Basic
    "Film güzel.",
    "Film kötü.",

    # Negation
    "Film güzel değildi.",
    "Film kötü değildi.",

    # Target + negation
    "Bu ürünü tavsiye etmiyorum.",

    # Missing lexical sentiment
    "Filmi beğenmedim.",

    # Multiple targets
    "Otel güzeldi ama odalar kötüydü.",

    # HisNet target ambiguity
    "Personel iyiydi fakat yemekler kötüydü.",

    # Modifier
    "Otel çok güzeldi.",
    "Personel biraz kötüydü.",

    # Known Stanza / lexical limitation
    "Otel aşırı güzel."
]


for sentence in sentences:

    baseline = hisnet.classify(
        sentence
    )

    dependency = (
        hisnet_dependency.classify(
            sentence
        )
    )

    print(
        "\n"
        + "=" * 110
    )

    print(
        "Sentence:",
        sentence
    )

    print(
        "HisNet:",
        baseline["prediction"],
        f"({baseline['positive_score']:.4f}, "
        f"{baseline['negative_score']:.4f})"
    )

    print(
        "HisNet + Dependency:",
        dependency["prediction"],
        f"({dependency['positive_score']:.4f}, "
        f"{dependency['negative_score']:.4f})"
    )

    for sentence_result in dependency[
        "sentences"
    ]:

        print(
            "\nResolved sentiment scopes:"
        )

        for scope in sentence_result[
            "scopes"
        ]:

            target = (
                scope["target"].get("root")
                if scope.get("target")
                else None
            )

            modifiers = [
                modifier.get("lemma")
                or modifier.get("word")

                for modifier
                in scope.get(
                    "modifiers",
                    []
                )
            ]

            negations = [
                negation.get("lemma")
                or negation.get("word")

                for negation
                in scope.get(
                    "negations",
                    []
                )
            ]

            contrast = (
                scope["contrast"].get(
                    "lemma"
                )

                if scope.get("contrast")
                else None
            )

            print(
                {
                    "opinion":
                        scope["root"],

                    "target":
                        target,

                    "role":
                        scope.get(
                            "contextual_role"
                        ),

                    "active":
                        scope.get(
                            "active_sentiment"
                        ),

                    "modifiers":
                        modifiers,

                    "negations":
                        negations,

                    "contrast":
                        contrast,

                    "original":
                        (
                            scope[
                                "original_positive"
                            ],
                            scope[
                                "original_negative"
                            ]
                        ),

                    "adjusted":
                        (
                            scope[
                                "adjusted_positive"
                            ],
                            scope[
                                "adjusted_negative"
                            ]
                        ),

                    "operations":
                        scope[
                            "operations"
                        ]
                }
            )