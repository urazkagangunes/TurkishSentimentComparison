from src.hisnet.hisnet_classifier import (
    HisNetClassifier,
    NEUTRAL_THRESHOLD
)


TESTS = [
    # positive, negative, expected
    (0.00, 0.00, "neutral"),
    (0.00, 0.02, "neutral"),
    (0.02, 0.00, "neutral"),
    (0.04, 0.08, "neutral"),
    (0.06, 0.05, "neutral"),

    (0.10, 0.03, "positive"),
    (0.4545, 0.00, "positive"),
    (1.00, 0.20, "positive"),

    (0.03, 0.10, "negative"),
    (0.00, 0.90, "negative"),
    (0.20, 1.00, "negative")
]


print(
    f"Neutral threshold: "
    f"{NEUTRAL_THRESHOLD}"
)

print(
    "=" * 70
)


for (
    positive,
    negative,
    expected
) in TESTS:

    prediction = (
        HisNetClassifier
        ._prediction_from_scores(
            positive,
            negative
        )
    )

    net = (
        positive
        - negative
    )

    print(
        f"POS={positive:<7} "
        f"NEG={negative:<7} "
        f"NET={net:<8.4f} "
        f"=> {prediction}"
    )

    assert (
        prediction
        == expected
    )


print(
    "=" * 70
)

print(
    "ALL THRESHOLD TESTS PASSED"
)