from src.hisnet.hisnet_classifier import HisNetClassifier

from src.hisnet.hisnet_dependency_classifier import (
    HisNetDependencyClassifier
)


hisnet = HisNetClassifier()
hisnet_dependency = HisNetDependencyClassifier()


sentences = [
    "Film güzel.",
    "Film kötü.",
    "Film güzel değildi.",
    "Film kötü değildi.",
    "Otel güzeldi ama odalar kötüydü.",
    "Filmi beğenmedim.",
    "Bu ürünü kesinlikle tavsiye etmiyorum."
]


for sentence in sentences:

    baseline = hisnet.classify(sentence)

    dependency = hisnet_dependency.classify(sentence)

    print("\n" + "=" * 100)

    print("Sentence:", sentence)

    print(
        "HisNet:",
        baseline["prediction"]
    )

    print(
        "HisNet + Dependency:",
        dependency["prediction"]
    )

    print(
        "HisNet Score:",
        baseline["positive_score"],
        "/",
        baseline["negative_score"]
    )

    print(
        "Dependency Score:",
        dependency["positive_score"],
        "/",
        dependency["negative_score"]
    )

    for sent_result in dependency["sentences"]:

        print("\nAdjustments:")

        for adjustment in sent_result["adjustments"]:
            print(" ", adjustment)

        print("\nUnresolved negations:")

        for negation in sent_result["unresolved_negations"]:
            print(" ", negation)