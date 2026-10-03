from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

from src.evaluation.dataset_loader import (
    load_sentiment_dataset
)

from src.hisnet.hisnet_classifier import (
    HisNetClassifier
)

from src.hisnet.hisnet_dependency_classifier import (
    HisNetDependencyClassifier
)

from src.hisnet.dependency_hisnet_classifier import (
    DependencyHisNetClassifier
)


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = Path(
    "datasets/Modified/sentence_sentiment/"
    "TRSAv1_fixed.csv"
)

SAMPLE_SIZE = 1000
RANDOM_STATE = 42

LABELS = [
    "negative",
    "neutral",
    "positive"
]

OUTPUT_DIR = Path(
    "results/trsav1_three_systems_1000"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATASET
# ============================================================

dataset = load_sentiment_dataset(
    file_path=DATASET_PATH,
    text_column="review",
    label_column="score",
    separator=";"
)


print(
    "=" * 100
)

print(
    "TRSAv1 THREE-SYSTEM COMPARISON"
)

print(
    "=" * 100
)

print(
    "Full dataset size:",
    len(dataset)
)


# ============================================================
# STRATIFIED 1000-SAMPLE
#
# The same exact texts will be sent to all three systems.
# ============================================================

_, sample = train_test_split(
    dataset,
    test_size=SAMPLE_SIZE,
    random_state=RANDOM_STATE,
    stratify=dataset["gold"]
)

sample = (
    sample
    .reset_index(
        drop=True
    )
)


print(
    "\nSample size:",
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


# Save the exact sample so that every future comparison can
# reproduce the same 1000 examples.
sample.to_csv(
    OUTPUT_DIR / "sample_1000.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# INITIALIZE MODELS
# ============================================================

print(
    "\nInitializing HisNet..."
)

hisnet = (
    HisNetClassifier()
)


print(
    "Initializing HisNet -> Dependency..."
)

hisnet_dependency = (
    HisNetDependencyClassifier()
)


print(
    "Initializing Dependency -> HisNet..."
)

dependency_hisnet = (
    DependencyHisNetClassifier()
)


# ============================================================
# STORAGE
# ============================================================

rows = []


# ============================================================
# RUN ALL THREE SYSTEMS
# ============================================================

for index, row in sample.iterrows():

    text = row[
        "text"
    ]

    gold = row[
        "gold"
    ]


    # --------------------------------------------------------
    # 1. HISNET
    # --------------------------------------------------------

    hisnet_result = (
        hisnet.classify(
            text
        )
    )


    # --------------------------------------------------------
    # 2. HISNET -> DEPENDENCY
    #
    # IMPORTANT:
    # exact same HisNet result is reused.
    # --------------------------------------------------------

    hisnet_dependency_result = (
        hisnet_dependency.classify(
            text,
            hisnet_result=hisnet_result
        )
    )


    # --------------------------------------------------------
    # 3. DEPENDENCY -> HISNET
    # --------------------------------------------------------

    dependency_hisnet_result = (
        dependency_hisnet.classify(
            text
        )
    )


    # --------------------------------------------------------
    # OPERATIONS
    # --------------------------------------------------------

    hd_operations = []

    for scope in (
        hisnet_dependency_result.get(
            "scopes",
            []
        )
    ):

        hd_operations.extend(
            scope.get(
                "operations",
                []
            )
        )


    dh_operations = (
        dependency_hisnet_result.get(
            "operations",
            []
        )
    )


    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    rows.append({

        "id":
            index,

        "text":
            text,

        "gold":
            gold,

        # ====================================================
        # HISNET
        # ====================================================

        "hisnet_prediction":
            hisnet_result[
                "prediction"
            ],

        "hisnet_positive":
            hisnet_result[
                "positive_score"
            ],

        "hisnet_negative":
            hisnet_result[
                "negative_score"
            ],

        "hisnet_sentiment_words":
            len(
                hisnet_result.get(
                    "sentiment_words",
                    []
                )
            ),

        # ====================================================
        # HISNET -> DEPENDENCY
        # ====================================================

        "h_to_d_prediction":
            hisnet_dependency_result[
                "prediction"
            ],

        "h_to_d_positive":
            hisnet_dependency_result[
                "positive_score"
            ],

        "h_to_d_negative":
            hisnet_dependency_result[
                "negative_score"
            ],

        "h_to_d_negation":
            int(
                "negation_flip"
                in hd_operations
            ),

        "h_to_d_suppression":
            int(
                "suppressed_as_target"
                in hd_operations
            ),

        # ====================================================
        # DEPENDENCY -> HISNET
        # ====================================================

        "d_to_h_prediction":
            dependency_hisnet_result[
                "prediction"
            ],

        "d_to_h_positive":
            dependency_hisnet_result[
                "positive_score"
            ],

        "d_to_h_negative":
            dependency_hisnet_result[
                "negative_score"
            ],

        "d_to_h_dependency_candidates":
            dependency_hisnet_result[
                "dependency_candidate_count"
            ],

        "d_to_h_sentiment_words":
            dependency_hisnet_result[
                "sentiment_word_count"
            ],

        "d_to_h_negation":
            int(
                "negation_flip"
                in dh_operations
            ),

        "d_to_h_suppression":
            int(
                "suppressed_as_target"
                in dh_operations
            )
    })


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (
        (index + 1)
        % 50
        == 0
    ):

        print(
            f"[{index + 1}/{SAMPLE_SIZE}]"
        )


# ============================================================
# DATAFRAME
# ============================================================

results = pd.DataFrame(
    rows
)


results.to_csv(
    OUTPUT_DIR / "predictions_1000.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    gold,
    predictions
):

    accuracy = (
        accuracy_score(
            gold,
            predictions
        )
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _
    ) = (
        precision_recall_fscore_support(
            gold,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0
        )
    )

    (
        _,
        _,
        weighted_f1,
        _
    ) = (
        precision_recall_fscore_support(
            gold,
            predictions,
            labels=LABELS,
            average="weighted",
            zero_division=0
        )
    )

    return {
        "accuracy":
            accuracy,

        "macro_precision":
            macro_precision,

        "macro_recall":
            macro_recall,

        "macro_f1":
            macro_f1,

        "weighted_f1":
            weighted_f1
    }


gold = results[
    "gold"
]


system_columns = {

    "HisNet":
        "hisnet_prediction",

    "HisNet -> Dependency":
        "h_to_d_prediction",

    "Dependency -> HisNet":
        "d_to_h_prediction"
}


metrics_rows = []


for system_name, column in (
    system_columns.items()
):

    metrics = calculate_metrics(
        gold,
        results[
            column
        ]
    )

    metrics_rows.append({
        "system":
            system_name,

        **metrics
    })


metrics_df = pd.DataFrame(
    metrics_rows
)


# ============================================================
# PRINT MAIN RESULTS
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "MAIN METRICS"
)

print(
    "=" * 100
)


print(
    metrics_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


metrics_df.to_csv(
    OUTPUT_DIR / "metrics_1000.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# CONFUSION MATRICES
# ============================================================

for system_name, column in (
    system_columns.items()
):

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"CONFUSION MATRIX: {system_name}"
    )

    print(
        "Rows = Gold"
    )

    print(
        "Columns = Prediction"
    )

    print(
        "Label order:",
        LABELS
    )


    cm = confusion_matrix(
        gold,
        results[
            column
        ],
        labels=LABELS
    )


    cm_df = pd.DataFrame(
        cm,
        index=[
            f"gold_{label}"
            for label
            in LABELS
        ],
        columns=[
            f"pred_{label}"
            for label
            in LABELS
        ]
    )


    print(
        cm_df
    )


    safe_name = (
        system_name
        .lower()
        .replace(
            " ",
            "_"
        )
        .replace(
            "->",
            "to"
        )
    )


    cm_df.to_csv(
        OUTPUT_DIR
        / f"confusion_{safe_name}.csv",
        encoding="utf-8-sig"
    )


# ============================================================
# PAIRWISE EFFECT
# ============================================================

def pairwise_analysis(
    dataframe,
    baseline_column,
    comparison_column,
    name
):

    changed = 0

    improved = 0

    degraded = 0

    wrong_to_wrong = 0

    unchanged = 0


    for _, row in (
        dataframe.iterrows()
    ):

        gold_label = (
            row[
                "gold"
            ]
        )

        baseline = (
            row[
                baseline_column
            ]
        )

        comparison = (
            row[
                comparison_column
            ]
        )


        if baseline == comparison:

            unchanged += 1

            continue


        changed += 1


        baseline_correct = (
            baseline == gold_label
        )

        comparison_correct = (
            comparison == gold_label
        )


        if (
            not baseline_correct
            and comparison_correct
        ):

            improved += 1


        elif (
            baseline_correct
            and not comparison_correct
        ):

            degraded += 1


        else:

            wrong_to_wrong += 1


    print(
        "\n"
        + "=" * 100
    )

    print(
        f"PAIRWISE: {name}"
    )

    print(
        "=" * 100
    )

    print(
        "Changed:",
        changed
    )

    print(
        "Unchanged:",
        unchanged
    )

    print(
        "Improved:",
        improved
    )

    print(
        "Degraded:",
        degraded
    )

    print(
        "Wrong -> Wrong:",
        wrong_to_wrong
    )

    print(
        "Net correct change:",
        improved - degraded
    )


    return {
        "comparison":
            name,

        "changed":
            changed,

        "unchanged":
            unchanged,

        "improved":
            improved,

        "degraded":
            degraded,

        "wrong_to_wrong":
            wrong_to_wrong,

        "net_correct_change":
            improved - degraded
    }


pairwise_rows = []


# ------------------------------------------------------------
# H -> D effect relative to H
# ------------------------------------------------------------

pairwise_rows.append(
    pairwise_analysis(
        results,
        "hisnet_prediction",
        "h_to_d_prediction",
        "HisNet vs HisNet -> Dependency"
    )
)


# ------------------------------------------------------------
# D -> H effect relative to H
# ------------------------------------------------------------

pairwise_rows.append(
    pairwise_analysis(
        results,
        "hisnet_prediction",
        "d_to_h_prediction",
        "HisNet vs Dependency -> HisNet"
    )
)


# ------------------------------------------------------------
# Direct ordering comparison
# ------------------------------------------------------------

pairwise_rows.append(
    pairwise_analysis(
        results,
        "h_to_d_prediction",
        "d_to_h_prediction",
        "HisNet -> Dependency vs Dependency -> HisNet"
    )
)


pairwise_df = pd.DataFrame(
    pairwise_rows
)


pairwise_df.to_csv(
    OUTPUT_DIR / "pairwise_1000.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# PREDICTION DISTRIBUTIONS
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "PREDICTION DISTRIBUTIONS"
)

print(
    "=" * 100
)


for system_name, column in (
    system_columns.items()
):

    print(
        f"\n{system_name}:"
    )

    print(
        results[
            column
        ].value_counts()
    )


# ============================================================
# D -> H COVERAGE DIAGNOSTICS
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "DEPENDENCY -> HISNET COVERAGE"
)

print(
    "=" * 100
)


print(
    "Average HisNet sentiment words:",
    round(
        results[
            "hisnet_sentiment_words"
        ].mean(),
        4
    )
)


print(
    "Average D->H dependency candidates:",
    round(
        results[
            "d_to_h_dependency_candidates"
        ].mean(),
        4
    )
)


print(
    "Average D->H sentiment words:",
    round(
        results[
            "d_to_h_sentiment_words"
        ].mean(),
        4
    )
)


hisnet_has_sentiment_dh_none = (
    (
        results[
            "hisnet_sentiment_words"
        ] > 0
    )
    &
    (
        results[
            "d_to_h_sentiment_words"
        ] == 0
    )
).sum()


print(
    "HisNet has sentiment but D->H none:",
    hisnet_has_sentiment_dh_none
)


# ============================================================
# OPERATION COUNTS
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "DEPENDENCY OPERATION COUNTS"
)

print(
    "=" * 100
)


print(
    "H->D negation examples:",
    int(
        results[
            "h_to_d_negation"
        ].sum()
    )
)

print(
    "H->D suppression examples:",
    int(
        results[
            "h_to_d_suppression"
        ].sum()
    )
)

print(
    "D->H negation examples:",
    int(
        results[
            "d_to_h_negation"
        ].sum()
    )
)

print(
    "D->H suppression examples:",
    int(
        results[
            "d_to_h_suppression"
        ].sum()
    )
)


# ============================================================
# FINISH
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "1000-SAMPLE THREE-SYSTEM TEST COMPLETED"
)

print(
    "Results directory:"
)

print(
    OUTPUT_DIR
)

print(
    "=" * 100
)