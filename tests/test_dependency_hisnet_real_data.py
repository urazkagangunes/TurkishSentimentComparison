from pathlib import Path
from collections import Counter

from src.evaluation.dataset_loader import (
    load_sentiment_dataset
)

from src.hisnet.hisnet_classifier import (
    HisNetClassifier
)

from src.hisnet.dependency_hisnet_classifier import (
    DependencyHisNetClassifier
)


DATASET_PATH = Path(
    "datasets/Modified/sentence_sentiment/"
    "TRSAv1_fixed.csv"
)

SAMPLE_SIZE = 500
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

dataset = load_sentiment_dataset(
    file_path=DATASET_PATH,
    text_column="review",
    label_column="score",
    separator=";"
)


sample = (
    dataset
    .sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )
    .reset_index(
        drop=True
    )
)


print(
    "=" * 100
)

print(
    "DEPENDENCY -> HISNET REAL-DATA SANITY CHECK"
)

print(
    "=" * 100
)

print(
    "Examples:",
    len(sample)
)

print(
    "\nGold distribution:"
)

print(
    sample[
        "gold"
    ].value_counts()
)


# ============================================================
# CLASSIFIERS
# ============================================================

print(
    "\nInitializing HisNet..."
)

hisnet = (
    HisNetClassifier()
)


print(
    "Initializing Dependency -> HisNet..."
)

dependency_hisnet = (
    DependencyHisNetClassifier()
)


# ============================================================
# COUNTERS
# ============================================================

hisnet_prediction_counts = Counter()

dependency_prediction_counts = Counter()

candidate_strategy_counts = Counter()

target_strategy_counts = Counter()

dependency_candidate_total = 0

dependency_sentiment_total = 0

hisnet_sentiment_total = 0


no_dependency_candidate = []

dependency_candidate_but_no_sentiment = []

hisnet_has_sentiment_dependency_has_none = []

prediction_changes = []


# ============================================================
# RUN
# ============================================================

for index, row in sample.iterrows():

    text = row[
        "text"
    ]

    gold = row[
        "gold"
    ]

    hisnet_result = (
        hisnet.classify(
            text
        )
    )

    dependency_result = (
        dependency_hisnet.classify(
            text
        )
    )

    hisnet_prediction = (
        hisnet_result[
            "prediction"
        ]
    )

    dependency_prediction = (
        dependency_result[
            "prediction"
        ]
    )


    hisnet_prediction_counts[
        hisnet_prediction
    ] += 1

    dependency_prediction_counts[
        dependency_prediction
    ] += 1


    # --------------------------------------------------------
    # COUNTS
    # --------------------------------------------------------

    hisnet_sentiment_count = len(
        hisnet_result.get(
            "sentiment_words",
            []
        )
    )

    dependency_candidate_count = (
        dependency_result[
            "dependency_candidate_count"
        ]
    )

    dependency_sentiment_count = (
        dependency_result[
            "sentiment_word_count"
        ]
    )


    hisnet_sentiment_total += (
        hisnet_sentiment_count
    )

    dependency_candidate_total += (
        dependency_candidate_count
    )

    dependency_sentiment_total += (
        dependency_sentiment_count
    )


    # --------------------------------------------------------
    # STRATEGIES
    # --------------------------------------------------------

    for candidate in dependency_result.get(
        "dependency_candidates",
        []
    ):

        candidate_strategy_counts[
            candidate.get(
                "candidate_strategy"
            )
            or "None"
        ] += 1

        target_strategy_counts[
            candidate.get(
                "target_strategy"
            )
            or "None"
        ] += 1


    # --------------------------------------------------------
    # DIAGNOSTIC GROUPS
    # --------------------------------------------------------

    if dependency_candidate_count == 0:

        no_dependency_candidate.append({
            "id":
                index,

            "gold":
                gold,

            "text":
                text,

            "hisnet_prediction":
                hisnet_prediction,

            "hisnet_sentiment_count":
                hisnet_sentiment_count
        })


    if (
        dependency_candidate_count > 0
        and dependency_sentiment_count == 0
    ):

        dependency_candidate_but_no_sentiment.append({
            "id":
                index,

            "gold":
                gold,

            "text":
                text,

            "dependency_candidates":
                [
                    (
                        candidate.get(
                            "word"
                        ),
                        candidate.get(
                            "root"
                        ),
                        candidate.get(
                            "candidate_strategy"
                        )
                    )

                    for candidate
                    in dependency_result.get(
                        "dependency_candidates",
                        []
                    )
                ]
        })


    if (
        hisnet_sentiment_count > 0
        and dependency_sentiment_count == 0
    ):

        hisnet_has_sentiment_dependency_has_none.append({
            "id":
                index,

            "gold":
                gold,

            "text":
                text,

            "hisnet_prediction":
                hisnet_prediction,

            "dependency_prediction":
                dependency_prediction,

            "hisnet_words":
                [
                    (
                        item.get(
                            "word"
                        ),
                        item.get(
                            "root"
                        ),
                        item.get(
                            "positive"
                        ),
                        item.get(
                            "negative"
                        )
                    )

                    for item
                    in hisnet_result.get(
                        "sentiment_words",
                        []
                    )
                ],

            "dependency_candidates":
                [
                    (
                        item.get(
                            "word"
                        ),
                        item.get(
                            "root"
                        ),
                        item.get(
                            "candidate_strategy"
                        )
                    )

                    for item
                    in dependency_result.get(
                        "dependency_candidates",
                        []
                    )
                ]
        })


    if (
        hisnet_prediction
        != dependency_prediction
    ):

        prediction_changes.append({
            "id":
                index,

            "gold":
                gold,

            "text":
                text,

            "hisnet":
                hisnet_prediction,

            "dependency_hisnet":
                dependency_prediction,

            "hisnet_correct":
                hisnet_prediction
                == gold,

            "dependency_correct":
                dependency_prediction
                == gold
        })


    if (
        (index + 1)
        % 50
        == 0
    ):

        print(
            f"[{index + 1}/{SAMPLE_SIZE}]"
        )


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "SUMMARY"
)

print(
    "=" * 100
)


print(
    "\nHisNet prediction distribution:"
)

print(
    dict(
        hisnet_prediction_counts
    )
)


print(
    "\nDependency -> HisNet prediction distribution:"
)

print(
    dict(
        dependency_prediction_counts
    )
)


print(
    "\nTotal HisNet sentiment words:",
    hisnet_sentiment_total
)

print(
    "Total dependency candidates:",
    dependency_candidate_total
)

print(
    "Total D->H sentiment words:",
    dependency_sentiment_total
)


print(
    "\nAverage HisNet sentiment words / review:",
    round(
        hisnet_sentiment_total
        / SAMPLE_SIZE,
        4
    )
)

print(
    "Average dependency candidates / review:",
    round(
        dependency_candidate_total
        / SAMPLE_SIZE,
        4
    )
)

print(
    "Average D->H sentiment words / review:",
    round(
        dependency_sentiment_total
        / SAMPLE_SIZE,
        4
    )
)


print(
    "\nNo dependency candidate:",
    len(
        no_dependency_candidate
    )
)

print(
    "Dependency candidate exists but no HisNet sentiment:",
    len(
        dependency_candidate_but_no_sentiment
    )
)

print(
    "HisNet has sentiment but D->H has none:",
    len(
        hisnet_has_sentiment_dependency_has_none
    )
)

print(
    "Prediction changed:",
    len(
        prediction_changes
    )
)


print(
    "\nCandidate strategies:"
)

for key, value in (
    candidate_strategy_counts
    .most_common()
):

    print(
        f"  {key}: {value}"
    )


print(
    "\nTarget strategies:"
)

for key, value in (
    target_strategy_counts
    .most_common()
):

    print(
        f"  {key}: {value}"
    )


# ============================================================
# MOST IMPORTANT ERROR SAMPLE
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "HISNET HAS SENTIMENT BUT D->H HAS NONE"
)

print(
    "=" * 100
)


for item in (
    hisnet_has_sentiment_dependency_has_none[
        :20
    ]
):

    print(
        "\nID:",
        item[
            "id"
        ]
    )

    print(
        "Gold:",
        item[
            "gold"
        ]
    )

    print(
        "Text:",
        item[
            "text"
        ]
    )

    print(
        "HisNet:",
        item[
            "hisnet_prediction"
        ]
    )

    print(
        "D->H:",
        item[
            "dependency_prediction"
        ]
    )

    print(
        "HisNet sentiment words:",
        item[
            "hisnet_words"
        ]
    )

    print(
        "Dependency candidates:",
        item[
            "dependency_candidates"
        ]
    )


# ============================================================
# CHANGED PREDICTIONS
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "FIRST 20 CHANGED PREDICTIONS"
)

print(
    "=" * 100
)


for item in (
    prediction_changes[
        :20
    ]
):

    print(
        "\nGold:",
        item[
            "gold"
        ]
    )

    print(
        "Text:",
        item[
            "text"
        ]
    )

    print(
        "HisNet:",
        item[
            "hisnet"
        ],

        "| correct =",
        item[
            "hisnet_correct"
        ]
    )

    print(
        "D->H:",
        item[
            "dependency_hisnet"
        ],

        "| correct =",
        item[
            "dependency_correct"
        ]
    )


print(
    "\n"
    + "=" * 100
)

print(
    "REAL-DATA SANITY CHECK COMPLETED"
)

print(
    "=" * 100
)