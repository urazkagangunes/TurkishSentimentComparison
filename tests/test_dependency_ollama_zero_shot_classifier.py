import os

from src.llm.dependency_ollama_zero_shot_classifier import (
    DependencyOllamaZeroShotSentimentClassifier
)


MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:32b"
)


REVIEWS = [

    "Film güzel.",

    "Film güzel değildi.",

    "Ürün bugün elime ulaştı.",

    "Bu ürünü kesinlikle tavsiye etmiyorum.",

    "Otel çok güzeldi.",

    "Ne iyi ne kötü, ortalama bir ürün.",

    "Harika bir ürün, tekrar satın alırım.",

    "Kargo hızlıydı fakat ürün beklentimi karşılamadı."
]


print(
    "=" * 100
)

print(
    "DEPENDENCY -> LLM ZERO-SHOT TEST"
)

print(
    "=" * 100
)

print(
    "Model:",
    MODEL
)


classifier = (
    DependencyOllamaZeroShotSentimentClassifier(

        model=
            MODEL,

        batch_size=
            8,

        temperature=
            0.1,

        top_p=
            1.0,

        num_ctx=
            8192,

        keep_alive=
            "30m",

        max_retries=
            2,

        use_gpu_parser=
            False
    )
)


result = (
    classifier.classify_batch(
        REVIEWS
    )
)


print(
    "\n"
    + "=" * 100
)

print(
    "DEPENDENCY INPUT"
)

print(
    "=" * 100
)


for index, (
    review,
    dependency_text
) in enumerate(

    zip(
        REVIEWS,
        result[
            "dependency_serializations"
        ]
    ),

    start=1
):

    print(
        f"\nREVIEW {index}"
    )

    print(
        "-" * 100
    )

    print(
        review
    )

    print(
        "\nDEPENDENCY PARSE:"
    )

    print(
        dependency_text
    )


print(
    "\n"
    + "=" * 100
)

print(
    "PREDICTIONS"
)

print(
    "=" * 100
)


for index, (
    review,
    prediction
) in enumerate(

    zip(
        REVIEWS,
        result[
            "predictions"
        ]
    ),

    start=1
):

    print(
        f"\n{index}."
    )

    print(
        "Review:",
        review
    )

    print(
        "Prediction:",
        prediction
    )


print(
    "\n"
    + "=" * 100
)

print(
    "DIAGNOSTICS"
)

print(
    "=" * 100
)


print(
    "Status:",
    result[
        "status"
    ]
)

print(
    "Batch size:",
    result[
        "batch_size"
    ]
)

print(
    "Dependency parse time:",
    round(
        result[
            "dependency_parse_seconds"
        ],
        3
    ),
    "seconds"
)

print(
    "LLM request time:",
    round(
        result[
            "request_seconds"
        ],
        3
    ),
    "seconds"
)

print(
    "Total pipeline time:",
    round(
        result[
            "total_pipeline_seconds"
        ],
        3
    ),
    "seconds"
)

print(
    "Prompt tokens:",
    result[
        "prompt_tokens"
    ]
)

print(
    "Output tokens:",
    result[
        "output_tokens"
    ]
)

print(
    "Attempts:",
    result[
        "attempt"
    ]
)


print(
    "\nCLASSIFIER STATS"
)


stats = (
    classifier.get_stats()
)


for key, value in stats.items():

    print(
        f"{key}: {value}"
    )


# ============================================================
# STRUCTURAL TESTS
#
# We intentionally do NOT assert expected sentiment labels.
# This is an integration/smoke test.
# ============================================================

assert (
    result[
        "status"
    ]
    == "success"
)


assert (
    len(
        result[
            "predictions"
        ]
    )
    == len(
        REVIEWS
    )
)


assert (
    len(
        result[
            "dependency_parses"
        ]
    )
    == len(
        REVIEWS
    )
)


for prediction in result[
    "predictions"
]:

    assert prediction in {
        "positive",
        "negative",
        "neutral"
    }


for parsed_review in result[
    "dependency_parses"
]:

    assert isinstance(
        parsed_review,
        list
    )

    assert len(
        parsed_review
    ) > 0


print(
    "\n"
    + "=" * 100
)

print(
    "DEPENDENCY -> LLM TEST PASSED"
)

print(
    "=" * 100
)