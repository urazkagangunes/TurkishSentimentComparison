import json
from pathlib import Path

import pandas as pd


PREDICTIONS_PATH = Path(
    "results/beyazperde/"
    "beyazperde_full_predictions.csv"
)

OUTPUT_DIRECTORY = Path(
    "results/beyazperde/analysis"
)


LABELS = [
    "negative",
    "neutral",
    "positive"
]


# =============================================================
# HELPERS
# =============================================================

def parse_json_list(value):

    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    value = str(value).strip()

    if not value:
        return []

    try:
        parsed = json.loads(value)

        if isinstance(parsed, list):
            return parsed

    except json.JSONDecodeError:
        pass

    return []


def to_bool(value):

    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    return (
        str(value)
        .strip()
        .lower()
        == "true"
    )


def accuracy_from_column(
    dataframe,
    column
):

    if len(dataframe) == 0:
        return 0.0

    return float(
        (
            dataframe[column]
            == dataframe["gold"]
        ).mean()
    )


# =============================================================
# LOAD
# =============================================================

if not PREDICTIONS_PATH.exists():

    raise FileNotFoundError(
        f"Predictions file not found:\n"
        f"{PREDICTIONS_PATH}"
    )


OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


df = pd.read_csv(
    PREDICTIONS_PATH,
    sep=";",
    encoding="utf-8-sig"
)


print(
    "=" * 90
)

print(
    "BEYAZPERDE FULL DEPENDENCY EFFECT ANALYSIS"
)

print(
    "=" * 90
)

print(
    f"Total examples: {len(df)}"
)


# =============================================================
# NORMALIZE BOOLEAN COLUMNS
# =============================================================

boolean_columns = [
    "hisnet_correct",
    "dependency_correct",
    "prediction_changed",
    "improved",
    "degraded",
    "changed_but_still_wrong",
    "has_negation",
    "has_modifier",
    "has_contrast",
    "has_suppression"
]


for column in boolean_columns:

    if column in df.columns:

        df[column] = (
            df[column]
            .apply(to_bool)
        )


# =============================================================
# PARSE OPERATIONS
# =============================================================

df["operation_list"] = (
    df["operations"]
    .apply(parse_json_list)
)


df["has_negation_flip"] = (
    df["operation_list"]
    .apply(
        lambda operations:
        "negation_flip"
        in operations
    )
)


df["has_suppression_operation"] = (
    df["operation_list"]
    .apply(
        lambda operations:
        "suppressed_as_target"
        in operations
    )
)


df["has_modifier_operation"] = (
    df["operation_list"]
    .apply(
        lambda operations:
        "modifier_detected"
        in operations
    )
)


df["has_contrast_operation"] = (
    df["operation_list"]
    .apply(
        lambda operations:
        "contrast_detected"
        in operations
    )
)


# =============================================================
# SCORE-AFFECTING OPERATION GROUP
# =============================================================

def score_operation_group(row):

    negation = row[
        "has_negation_flip"
    ]

    suppression = row[
        "has_suppression_operation"
    ]

    if negation and suppression:
        return "negation + suppression"

    if negation:
        return "negation only"

    if suppression:
        return "suppression only"

    return "no score-changing operation"


df["score_operation_group"] = (
    df.apply(
        score_operation_group,
        axis=1
    )
)


# =============================================================
# 1. GLOBAL RESULT
# =============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "1. GLOBAL EFFECT"
)

print(
    "=" * 90
)


hisnet_accuracy = (
    accuracy_from_column(
        df,
        "hisnet_prediction"
    )
)

dependency_accuracy = (
    accuracy_from_column(
        df,
        "dependency_prediction"
    )
)


print(
    f"HisNet accuracy:      "
    f"{hisnet_accuracy:.4f}"
)

print(
    f"HisNet+Dep accuracy:  "
    f"{dependency_accuracy:.4f}"
)

print(
    f"Difference:           "
    f"{dependency_accuracy - hisnet_accuracy:+.4f}"
)

print()

print(
    "Prediction changed:",
    int(
        df[
            "prediction_changed"
        ].sum()
    )
)

print(
    "Improved:",
    int(
        df[
            "improved"
        ].sum()
    )
)

print(
    "Degraded:",
    int(
        df[
            "degraded"
        ].sum()
    )
)

print(
    "Wrong -> wrong:",
    int(
        df[
            "changed_but_still_wrong"
        ].sum()
    )
)


# =============================================================
# 2. EFFECT BY SCORE-CHANGING OPERATION
# =============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "2. EFFECT BY SCORE-CHANGING OPERATION"
)

print(
    "=" * 90
)


group_order = [
    "negation only",
    "suppression only",
    "negation + suppression",
    "no score-changing operation"
]


operation_rows = []


for group_name in group_order:

    subset = df[
        df[
            "score_operation_group"
        ]
        == group_name
    ]

    count = len(
        subset
    )

    changed = int(
        subset[
            "prediction_changed"
        ].sum()
    )

    improved = int(
        subset[
            "improved"
        ].sum()
    )

    degraded = int(
        subset[
            "degraded"
        ].sum()
    )

    wrong_to_wrong = int(
        subset[
            "changed_but_still_wrong"
        ].sum()
    )

    hisnet_acc = (
        accuracy_from_column(
            subset,
            "hisnet_prediction"
        )
    )

    dependency_acc = (
        accuracy_from_column(
            subset,
            "dependency_prediction"
        )
    )

    operation_rows.append({
        "group":
            group_name,

        "examples":
            count,

        "changed":
            changed,

        "improved":
            improved,

        "degraded":
            degraded,

        "wrong_to_wrong":
            wrong_to_wrong,

        "net_gain":
            improved
            - degraded,

        "hisnet_accuracy":
            hisnet_acc,

        "dependency_accuracy":
            dependency_acc,

        "accuracy_difference":
            dependency_acc
            - hisnet_acc
    })


operation_df = pd.DataFrame(
    operation_rows
)


print(
    operation_df.to_string(
        index=False,
        float_format=lambda value:
        f"{value:.4f}"
    )
)


operation_df.to_csv(
    OUTPUT_DIRECTORY
    / "operation_effect_summary.csv",
    index=False,
    encoding="utf-8-sig"
)


# =============================================================
# 3. EXACT OPERATION COMBINATIONS
# =============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "3. EXACT OPERATION COMBINATIONS"
)

print(
    "=" * 90
)


def operation_signature(
    operations
):

    if not operations:
        return "none"

    return " + ".join(
        sorted(
            set(operations)
        )
    )


df["operation_signature"] = (
    df["operation_list"]
    .apply(
        operation_signature
    )
)


signature_rows = []


for signature, subset in (
    df.groupby(
        "operation_signature"
    )
):

    signature_rows.append({
        "operations":
            signature,

        "examples":
            len(subset),

        "changed":
            int(
                subset[
                    "prediction_changed"
                ].sum()
            ),

        "improved":
            int(
                subset[
                    "improved"
                ].sum()
            ),

        "degraded":
            int(
                subset[
                    "degraded"
                ].sum()
            ),

        "wrong_to_wrong":
            int(
                subset[
                    "changed_but_still_wrong"
                ].sum()
            ),

        "net_gain":
            int(
                subset[
                    "improved"
                ].sum()
            )
            - int(
                subset[
                    "degraded"
                ].sum()
            )
    })


signature_df = (
    pd.DataFrame(
        signature_rows
    )
    .sort_values(
        by="examples",
        ascending=False
    )
)


print(
    signature_df.to_string(
        index=False
    )
)


signature_df.to_csv(
    OUTPUT_DIRECTORY
    / "exact_operation_combinations.csv",
    index=False,
    encoding="utf-8-sig"
)


# =============================================================
# 4. PREDICTION TRANSITIONS
# =============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "4. CHANGED PREDICTION TRANSITIONS"
)

print(
    "=" * 90
)


changed_df = df[
    df[
        "prediction_changed"
    ]
].copy()


transition_table = pd.crosstab(
    changed_df[
        "hisnet_prediction"
    ],
    changed_df[
        "dependency_prediction"
    ],
    rownames=[
        "HisNet"
    ],
    colnames=[
        "HisNet+Dependency"
    ],
    dropna=False
)


transition_table = (
    transition_table
    .reindex(
        index=LABELS,
        columns=LABELS,
        fill_value=0
    )
)


print(
    transition_table
)


transition_table.to_csv(
    OUTPUT_DIRECTORY
    / "prediction_transitions.csv",
    encoding="utf-8-sig"
)


print(
    "\nDetailed transitions:"
)


transition_counts = (
    changed_df
    .groupby(
        [
            "hisnet_prediction",
            "dependency_prediction"
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
    .sort_values(
        "count",
        ascending=False
    )
)


print(
    transition_counts.to_string(
        index=False
    )
)


transition_counts.to_csv(
    OUTPUT_DIRECTORY
    / "prediction_transition_counts.csv",
    index=False,
    encoding="utf-8-sig"
)


# =============================================================
# 5. TRANSITIONS + GOLD LABEL
# =============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "5. CHANGED TRANSITIONS BY GOLD LABEL"
)

print(
    "=" * 90
)


gold_transition_counts = (
    changed_df
    .groupby(
        [
            "gold",
            "hisnet_prediction",
            "dependency_prediction"
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
    .sort_values(
        [
            "gold",
            "count"
        ],
        ascending=[
            True,
            False
        ]
    )
)


print(
    gold_transition_counts.to_string(
        index=False
    )
)


gold_transition_counts.to_csv(
    OUTPUT_DIRECTORY
    / "transitions_by_gold.csv",
    index=False,
    encoding="utf-8-sig"
)


# =============================================================
# 6. NEGATION SUBSET
# =============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "6. NEGATION-FLIP SUBSET"
)

print(
    "=" * 90
)


negation_df = df[
    df[
        "has_negation_flip"
    ]
].copy()


print(
    "Examples with negation_flip:",
    len(
        negation_df
    )
)

print(
    "Changed:",
    int(
        negation_df[
            "prediction_changed"
        ].sum()
    )
)

print(
    "Improved:",
    int(
        negation_df[
            "improved"
        ].sum()
    )
)

print(
    "Degraded:",
    int(
        negation_df[
            "degraded"
        ].sum()
    )
)

print(
    "Wrong -> wrong:",
    int(
        negation_df[
            "changed_but_still_wrong"
        ].sum()
    )
)

print(
    "Net gain:",
    int(
        negation_df[
            "improved"
        ].sum()
    )
    - int(
        negation_df[
            "degraded"
        ].sum()
    )
)

print(
    "HisNet accuracy:",
    f"{accuracy_from_column(negation_df, 'hisnet_prediction'):.4f}"
)

print(
    "Dependency accuracy:",
    f"{accuracy_from_column(negation_df, 'dependency_prediction'):.4f}"
)


# =============================================================
# 7. SUPPRESSION SUBSET
# =============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "7. SUPPRESSION SUBSET"
)

print(
    "=" * 90
)


suppression_df = df[
    df[
        "has_suppression_operation"
    ]
].copy()


print(
    "Examples with suppression:",
    len(
        suppression_df
    )
)

print(
    "Changed:",
    int(
        suppression_df[
            "prediction_changed"
        ].sum()
    )
)

print(
    "Improved:",
    int(
        suppression_df[
            "improved"
        ].sum()
    )
)

print(
    "Degraded:",
    int(
        suppression_df[
            "degraded"
        ].sum()
    )
)

print(
    "Wrong -> wrong:",
    int(
        suppression_df[
            "changed_but_still_wrong"
        ].sum()
    )
)

print(
    "Net gain:",
    int(
        suppression_df[
            "improved"
        ].sum()
    )
    - int(
        suppression_df[
            "degraded"
        ].sum()
    )
)

print(
    "HisNet accuracy:",
    f"{accuracy_from_column(suppression_df, 'hisnet_prediction'):.4f}"
)

print(
    "Dependency accuracy:",
    f"{accuracy_from_column(suppression_df, 'dependency_prediction'):.4f}"
)


# =============================================================
# 8. MODIFIER / CONTRAST DIAGNOSTICS
# =============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "8. MODIFIER / CONTRAST DIAGNOSTICS"
)

print(
    "=" * 90
)


for title, column in [
    (
        "Modifier",
        "has_modifier_operation"
    ),
    (
        "Contrast",
        "has_contrast_operation"
    )
]:

    subset = df[
        df[
            column
        ]
    ]

    print(
        f"\n{title}:"
    )

    print(
        "  Examples:",
        len(
            subset
        )
    )

    print(
        "  Prediction changed:",
        int(
            subset[
                "prediction_changed"
            ].sum()
        )
    )

    print(
        "  Improved:",
        int(
            subset[
                "improved"
            ].sum()
        )
    )

    print(
        "  Degraded:",
        int(
            subset[
                "degraded"
            ].sum()
        )
    )


# =============================================================
# 9. FAIRNESS / INVARIANT CHECK
# =============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "9. SCORE INVARIANT CHECK"
)

print(
    "=" * 90
)


EPSILON = 1e-12


df[
    "positive_score_difference"
] = (
    df[
        "dependency_positive_score"
    ]
    - df[
        "hisnet_positive_score"
    ]
)


df[
    "negative_score_difference"
] = (
    df[
        "dependency_negative_score"
    ]
    - df[
        "hisnet_negative_score"
    ]
)


df[
    "score_changed"
] = (
    df[
        "positive_score_difference"
    ].abs()
    > EPSILON
) | (
    df[
        "negative_score_difference"
    ].abs()
    > EPSILON
)


no_score_operation = (
    ~df[
        "has_negation_flip"
    ]
    & ~df[
        "has_suppression_operation"
    ]
)


unexpected_score_change = df[
    no_score_operation
    & df[
        "score_changed"
    ]
].copy()


unexpected_prediction_change = df[
    no_score_operation
    & df[
        "prediction_changed"
    ]
].copy()


print(
    "Rows without negation/suppression "
    "but score changed:",
    len(
        unexpected_score_change
    )
)

print(
    "Rows without negation/suppression "
    "but prediction changed:",
    len(
        unexpected_prediction_change
    )
)


unexpected_score_change.to_csv(
    OUTPUT_DIRECTORY
    / "unexpected_score_changes.csv",
    sep=";",
    index=False,
    encoding="utf-8-sig"
)


unexpected_prediction_change.to_csv(
    OUTPUT_DIRECTORY
    / "unexpected_prediction_changes.csv",
    sep=";",
    index=False,
    encoding="utf-8-sig"
)


# =============================================================
# 10. SAVE CHANGED EXAMPLES
# =============================================================

changed_df.to_csv(
    OUTPUT_DIRECTORY
    / "all_changed_predictions.csv",
    sep=";",
    index=False,
    encoding="utf-8-sig"
)


df[
    df["improved"]
].to_csv(
    OUTPUT_DIRECTORY
    / "improved_examples.csv",
    sep=";",
    index=False,
    encoding="utf-8-sig"
)


df[
    df["degraded"]
].to_csv(
    OUTPUT_DIRECTORY
    / "degraded_examples.csv",
    sep=";",
    index=False,
    encoding="utf-8-sig"
)


df[
    df[
        "changed_but_still_wrong"
    ]
].to_csv(
    OUTPUT_DIRECTORY
    / "changed_but_still_wrong_examples.csv",
    sep=";",
    index=False,
    encoding="utf-8-sig"
)


# =============================================================
# DONE
# =============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "ANALYSIS COMPLETED"
)

print(
    "=" * 90
)

print(
    f"Files saved to: "
    f"{OUTPUT_DIRECTORY}"
)