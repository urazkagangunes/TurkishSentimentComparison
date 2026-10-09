import os
import time

from src.llm.ollama_zero_shot_classifier import (
    OllamaZeroShotSentimentClassifier
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:32b"
)

BATCH_SIZE = 4


# ============================================================
# TEST REVIEWS
#
# IMPORTANT:
# We use the same 16 reviews from the previous batch-size
# experiments so that Batch 4 / 8 / 16 are comparable.
# ============================================================

REVIEWS = [

    "Ürün çok güzel, kesinlikle tavsiye ederim.",

    "Hiç beğenmedim, verdiğim paraya değmedi.",

    "Ürün bugün elime ulaştı.",

    "Kalitesi güzel ama boyutu biraz küçük.",

    "Beklediğimden çok daha kötü çıktı.",

    "Ne iyi ne kötü, ortalama bir ürün.",

    "Harika bir ürün, tekrar satın alırım.",

    "Kargo hızlıydı fakat ürün beklentimi karşılamadı.",

    "Ürün gayet başarılı, memnun kaldım.",

    "Maalesef hiç işe yaramadı.",

    "Paket bugün teslim edildi.",

    "Fiyatına göre fena değil.",

    "Çok kalitesiz, tekrar almam.",

    "Beklediğim kadar iyi değildi.",

    "Normal bir ürün, özel bir yanı yok.",

    "Mükemmel, kesinlikle yeniden alırım."
]


# ============================================================
# HEADER
# ============================================================

print(
    "=" * 100
)

print(
    "OLLAMA ZERO-SHOT BATCH SIZE 4 TEST"
)

print(
    "=" * 100
)

print(
    "Model:",
    MODEL
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Total reviews:",
    len(
        REVIEWS
    )
)

print(
    "Expected batches:",
    (
        len(REVIEWS)
        + BATCH_SIZE
        - 1
    )
    // BATCH_SIZE
)


# ============================================================
# INITIALIZE CLASSIFIER
# ============================================================

classifier = (
    OllamaZeroShotSentimentClassifier(

        model=
            MODEL,

        batch_size=
            BATCH_SIZE,

        temperature=
            0.1,

        top_p=
            1.0,

        num_ctx=
            4096,

        keep_alive=
            "30m",

        max_retries=
            2
    )
)


# ============================================================
# RUN
#
# classify_many_safe():
#
# 1. Sends reviews in batches of 4.
# 2. Retries failed batches.
# 3. Does NOT stop the experiment if a batch fails.
# 4. Failed examples are retried individually.
# 5. Never converts a technical failure to neutral.
# ============================================================

experiment_start = (
    time.perf_counter()
)


records = (
    classifier.classify_many_safe(
        REVIEWS
    )
)


experiment_seconds = (
    time.perf_counter()
    - experiment_start
)


# ============================================================
# PRINT PREDICTIONS
# ============================================================

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


for index, record in enumerate(
    records,
    start=1
):

    print(
        f"\n{index}."
    )

    print(
        "Review:",
        record[
            "text"
        ]
    )

    print(
        "Prediction:",
        record[
            "prediction"
        ]
    )

    print(
        "Status:",
        record[
            "status"
        ]
    )

    print(
        "Recovered:",
        record[
            "recovered"
        ]
    )

    if record[
        "error"
    ] is not None:

        print(
            "Error:",
            record[
                "error"
            ]
        )


# ============================================================
# CLASSIFIER STATISTICS
# ============================================================

stats = (
    classifier.get_stats()
)


print(
    "\n"
    + "=" * 100
)

print(
    "BATCH SIZE 4 DIAGNOSTICS"
)

print(
    "=" * 100
)


print(
    "Model:",
    stats[
        "model"
    ]
)

print(
    "Configured batch size:",
    stats[
        "configured_batch_size"
    ]
)

print(
    "Temperature:",
    stats[
        "temperature"
    ]
)

print(
    "Top-p:",
    stats[
        "top_p"
    ]
)

print(
    "Context size:",
    stats[
        "num_ctx"
    ]
)

print(
    "Max retries:",
    stats[
        "max_retries"
    ]
)


print(
    "\n--- REQUEST STATISTICS ---"
)


print(
    "Total requests:",
    stats[
        "total_requests"
    ]
)

print(
    "Total successful reviews:",
    stats[
        "total_reviews"
    ]
)

print(
    "Total retries:",
    stats[
        "total_retries"
    ]
)

print(
    "Total failed calls:",
    stats[
        "total_failures"
    ]
)


print(
    "\n--- TIMING ---"
)


print(
    "Classifier request time:",
    round(
        stats[
            "total_wall_seconds"
        ],
        3
    ),
    "seconds"
)


print(
    "Actual experiment wall time:",
    round(
        experiment_seconds,
        3
    ),
    "seconds"
)


actual_seconds_per_review = (
    experiment_seconds
    / len(
        REVIEWS
    )
)


actual_reviews_per_second = (
    len(
        REVIEWS
    )
    / experiment_seconds
)


print(
    "Actual seconds/review:",
    round(
        actual_seconds_per_review,
        4
    )
)


print(
    "Actual reviews/second:",
    round(
        actual_reviews_per_second,
        6
    )
)


print(
    "Classifier seconds/review:",
    round(
        stats[
            "seconds_per_review"
        ],
        4
    )
)


print(
    "Classifier reviews/second:",
    round(
        stats[
            "reviews_per_second"
        ],
        6
    )
)


print(
    "\n--- TOKEN STATISTICS ---"
)


print(
    "Total prompt tokens:",
    stats[
        "total_prompt_tokens"
    ]
)

print(
    "Total output tokens:",
    stats[
        "total_output_tokens"
    ]
)


# ============================================================
# STATUS COUNTS
# ============================================================

successful_count = sum(
    record[
        "status"
    ]
    == "success"

    for record in records
)


failed_count = sum(
    record[
        "status"
    ]
    == "failed"

    for record in records
)


recovered_count = sum(
    record[
        "recovered"
    ]

    for record in records
)


print(
    "\n--- RESULT STATUS ---"
)


print(
    "Successful:",
    successful_count
)

print(
    "Failed:",
    failed_count
)

print(
    "Recovered:",
    recovered_count
)


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

positive_count = sum(
    record[
        "prediction"
    ]
    == "positive"

    for record in records
)


negative_count = sum(
    record[
        "prediction"
    ]
    == "negative"

    for record in records
)


neutral_count = sum(
    record[
        "prediction"
    ]
    == "neutral"

    for record in records
)


print(
    "\n--- PREDICTION DISTRIBUTION ---"
)


print(
    "Positive:",
    positive_count
)

print(
    "Negative:",
    negative_count
)

print(
    "Neutral:",
    neutral_count
)


# ============================================================
# STRUCTURAL ASSERTIONS
#
# IMPORTANT:
#
# We intentionally do NOT assert expected sentiment labels.
#
# This test evaluates:
#
# - Ollama connection
# - batch splitting
# - structured output
# - failure handling
# - recovery
# - runtime
#
# Sentiment quality will be evaluated later using TRSAv1 gold
# labels.
# ============================================================

assert (
    len(
        records
    )
    == len(
        REVIEWS
    )
), (
    "Number of output records does not match "
    "number of input reviews."
)


for record in records:

    assert (
        record[
            "status"
        ]
        == "success"
    ), (
        "At least one test review could not be "
        "successfully classified."
    )

    assert (
        record[
            "prediction"
        ]
        in {
            "positive",
            "negative",
            "neutral"
        }
    ), (
        "Invalid sentiment prediction: "
        f"{record['prediction']}"
    )


# ============================================================
# FINISH
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "BATCH SIZE 4 TEST PASSED"
)

print(
    "=" * 100
)