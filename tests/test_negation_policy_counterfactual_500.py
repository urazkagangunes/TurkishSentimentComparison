import math
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from src.hisnet.hisnet_classifier import (
    HisNetClassifier
)

from src.hisnet.hisnet_dependency_classifier import (
    HisNetDependencyClassifier
)


# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = Path(
    "results/trsav1_four_classical_500_post_negation_freeze"
)

PREDICTIONS_FILE = (
    RESULTS_DIR
    / "predictions_500.csv"
)

OUTPUT_DIR = Path(
    "results/negation_policy_counterfactual_500"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

LABELS = [
    "negative",
    "neutral",
    "positive",
]

NEUTRAL_THRESHOLD = 0.05


# ============================================================
# HELPERS
# ============================================================

def predict_from_scores(
    positive,
    negative,
):

    net = (
        positive
        - negative
    )

    if (
        net
        > NEUTRAL_THRESHOLD
    ):
        return "positive"

    if (
        net
        < -NEUTRAL_THRESHOLD
    ):
        return "negative"

    return "neutral"


def calculate_metrics(
    dataframe,
    prediction_column,
):

    accuracy = accuracy_score(
        dataframe["gold"],
        dataframe[
            prediction_column
        ],
    )

    (
        precision,
        recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        dataframe["gold"],
        dataframe[
            prediction_column
        ],
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    (
        _,
        _,
        weighted_f1,
        _,
    ) = precision_recall_fscore_support(
        dataframe["gold"],
        dataframe[
            prediction_column
        ],
        labels=LABELS,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy":
            accuracy,

        "macro_precision":
            precision,

        "macro_recall":
            recall,

        "macro_f1":
            macro_f1,

        "weighted_f1":
            weighted_f1,
    }


def exact_mcnemar_p(
    a_only_correct,
    b_only_correct,
):

    b = int(
        a_only_correct
    )

    c = int(
        b_only_correct
    )

    n = (
        b + c
    )

    if n == 0:
        return 1.0

    k = min(
        b,
        c,
    )

    lower_tail = sum(

        math.comb(
            n,
            i,
        )

        for i in range(
            k + 1
        )

    ) / (
        2 ** n
    )

    return min(
        1.0,
        2.0 * lower_tail,
    )


# ============================================================
# NEGATION TYPE
# ============================================================

def is_self_morphological_negation(
    scope,
    negation,
):
    """
    True when the sentiment-bearing dependency token itself
    carries Polarity=Neg.

    Example:

        göremedim
        yaramıyor
        almayın
    """

    sentiment_token = (
        scope.get(
            "sentiment_token"
        )
        or {}
    )

    sentiment_id = (
        sentiment_token.get(
            "id"
        )
    )

    negation_id = (
        negation.get(
            "id"
        )
    )

    if (
        sentiment_id is None
        or negation_id is None
    ):
        return False

    if (
        int(sentiment_id)
        != int(negation_id)
    ):
        return False

    feats = (
        negation.get(
            "feats"
        )
        or ""
    )

    return (
        "Polarity=Neg"
        in feats
    )


# ============================================================
# COUNTERFACTUAL SCOPE SCORE
# ============================================================

def counterfactual_scope_score(
    scope,
    policy,
):
    """
    Policies
    --------

    CURRENT
        Keep all current negation evidence.

    NO_SELF
        Remove self morphological negation.
        Explicit değil and support/light-verb negation remain.

    POSITIVE_LEXICON_ONLY
        Keep self morphological negation only when the
        original HisNet lexical polarity is positive.

        This preserves cases such as:

            sev -> sevmedim

        while avoiding automatic inversion of already-negative
        lexical roots such as:

            gör -> göremedim
            yara -> yaramıyor
            al -> almayın
    """

    operations = (
        scope.get(
            "operations",
            []
        )
        or []
    )

    active = scope.get(
        "active_sentiment",
        True,
    )

    # Target suppression remains exactly as in frozen H->D.
    if (
        not active
        or
        "suppressed_as_target"
        in operations
    ):
        return (
            0.0,
            0.0,
        )


    original_positive = float(

        scope.get(
            "original_positive",
            scope.get(
                "positive",
                0.0,
            ),
        )

        or 0.0
    )


    original_negative = float(

        scope.get(
            "original_negative",
            scope.get(
                "negative",
                0.0,
            ),
        )

        or 0.0
    )


    negations = (
        scope.get(
            "negations",
            []
        )
        or []
    )


    retained_negations = []


    for negation in negations:

        self_morph = (
            is_self_morphological_negation(
                scope,
                negation,
            )
        )


        # ----------------------------------------------------
        # CURRENT
        # ----------------------------------------------------

        if policy == "CURRENT":

            retained_negations.append(
                negation
            )

            continue


        # ----------------------------------------------------
        # Non-self negation is preserved in every policy.
        #
        # This includes:
        #
        #   explicit değil
        #   compound:lvc support negation
        #   other structurally licensed negation
        # ----------------------------------------------------

        if not self_morph:

            retained_negations.append(
                negation
            )

            continue


        # ----------------------------------------------------
        # NO_SELF
        # ----------------------------------------------------

        if policy == "NO_SELF":

            continue


        # ----------------------------------------------------
        # POSITIVE_LEXICON_ONLY
        #
        # Only invert a self-negated lexical expression if
        # its original lexical polarity is positive.
        # ----------------------------------------------------

        if (
            policy
            ==
            "POSITIVE_LEXICON_ONLY"
        ):

            original_net = (
                original_positive
                - original_negative
            )

            if (
                original_net
                > 0
            ):
                retained_negations.append(
                    negation
                )

            continue


        raise ValueError(
            f"Unknown policy: {policy}"
        )


    # ========================================================
    # NEGATION COMPOSITION
    #
    # Same parity rule as H->D:
    # odd number of negations -> swap polarity
    # ========================================================

    if (
        len(
            retained_negations
        )
        % 2
        == 1
    ):

        return (
            original_negative,
            original_positive,
        )


    return (
        original_positive,
        original_negative,
    )


def counterfactual_result(
    hd_result,
    policy,
):

    positive = 0.0
    negative = 0.0

    self_morph_scope_count = 0


    for scope in (
        hd_result.get(
            "scopes",
            []
        )
        or []
    ):

        for negation in (
            scope.get(
                "negations",
                []
            )
            or []
        ):

            if (
                is_self_morphological_negation(
                    scope,
                    negation,
                )
            ):

                self_morph_scope_count += 1
                break


        (
            scope_positive,
            scope_negative,
        ) = counterfactual_scope_score(
            scope,
            policy,
        )


        positive += (
            scope_positive
        )

        negative += (
            scope_negative
        )


    prediction = predict_from_scores(
        positive,
        negative,
    )


    return {
        "prediction":
            prediction,

        "positive":
            positive,

        "negative":
            negative,

        "net":
            positive
            - negative,

        "self_morph_scope_count":
            self_morph_scope_count,
    }


# ============================================================
# LOAD FROZEN RESULTS
# ============================================================

if not PREDICTIONS_FILE.exists():

    raise FileNotFoundError(
        f"Missing: {PREDICTIONS_FILE}"
    )


frozen = pd.read_csv(
    PREDICTIONS_FILE
)


print(
    "=" * 100
)

print(
    "NEGATION POLICY COUNTERFACTUAL — 500 REVIEWS"
)

print(
    "=" * 100
)


print(
    "\nFull sample:",
    len(
        frozen
    ),
)


# ============================================================
# ONLY RERUN REVIEWS WHERE H->D USED NEGATION
# ============================================================

negation_cases = frozen[
    frozen[
        "h_to_d_negation_flips"
    ]
    > 0
].copy()


print(
    "Reviews with H->D negation flip:",
    len(
        negation_cases
    ),
)


# ============================================================
# INITIALIZE CURRENT FROZEN MODELS
# ============================================================

hisnet = (
    HisNetClassifier()
)


h_to_d = (
    HisNetDependencyClassifier(
        hisnet_classifier=
            hisnet
    )
)


# ============================================================
# START FROM EXISTING CURRENT PREDICTIONS
# ============================================================

evaluation = frozen[
    [
        "sample_id",
        "text",
        "gold",
        "h_to_d_prediction",
    ]
].copy()


evaluation = evaluation.rename(
    columns={
        "h_to_d_prediction":
            "CURRENT"
    }
)


evaluation[
    "NO_SELF"
] = evaluation[
    "CURRENT"
]


evaluation[
    "POSITIVE_LEXICON_ONLY"
] = evaluation[
    "CURRENT"
]


case_rows = []
scope_rows = []


# ============================================================
# RERUN ONLY 76 NEGATION-ACTIVE REVIEWS
# ============================================================

for run_index, row in enumerate(

    negation_cases.itertuples(
        index=False
    ),

    start=1,

):

    sample_id = int(
        row.sample_id
    )

    text = str(
        row.text
    )


    h_result = hisnet.classify(
        text
    )


    hd_result = h_to_d.classify(

        text,

        hisnet_result=
            h_result,
    )


    # --------------------------------------------------------
    # Current code must reproduce frozen 500 prediction.
    # --------------------------------------------------------

    if (
        hd_result[
            "prediction"
        ]
        !=
        row.h_to_d_prediction
    ):

        raise RuntimeError(

            f"sample_id={sample_id}: "
            f"current code no longer matches frozen run. "

            f"saved="
            f"{row.h_to_d_prediction}, "

            f"current="
            f"{hd_result['prediction']}"

        )


    policy_results = {}


    for policy in [

        "CURRENT",
        "NO_SELF",
        "POSITIVE_LEXICON_ONLY",

    ]:

        result = (
            counterfactual_result(
                hd_result,
                policy,
            )
        )


        policy_results[
            policy
        ] = result


    # --------------------------------------------------------
    # Verify our CURRENT recomposition matches actual H->D.
    # --------------------------------------------------------

    current_cf = (
        policy_results[
            "CURRENT"
        ]
    )


    if not math.isclose(

        current_cf[
            "positive"
        ],

        float(
            hd_result[
                "positive_score"
            ]
        ),

        rel_tol=1e-9,
        abs_tol=1e-9,

    ):

        raise RuntimeError(

            f"sample_id={sample_id}: "
            f"CURRENT positive recomposition mismatch. "

            f"cf={current_cf['positive']} "
            f"actual={hd_result['positive_score']}"

        )


    if not math.isclose(

        current_cf[
            "negative"
        ],

        float(
            hd_result[
                "negative_score"
            ]
        ),

        rel_tol=1e-9,
        abs_tol=1e-9,

    ):

        raise RuntimeError(

            f"sample_id={sample_id}: "
            f"CURRENT negative recomposition mismatch. "

            f"cf={current_cf['negative']} "
            f"actual={hd_result['negative_score']}"

        )


    # --------------------------------------------------------
    # Replace predictions in full 500 table.
    # --------------------------------------------------------

    mask = (
        evaluation[
            "sample_id"
        ]
        ==
        sample_id
    )


    evaluation.loc[
        mask,
        "NO_SELF"
    ] = (
        policy_results[
            "NO_SELF"
        ][
            "prediction"
        ]
    )


    evaluation.loc[
        mask,
        "POSITIVE_LEXICON_ONLY"
    ] = (
        policy_results[
            "POSITIVE_LEXICON_ONLY"
        ][
            "prediction"
        ]
    )


    # --------------------------------------------------------
    # Per-review output
    # --------------------------------------------------------

    case_rows.append({

        "sample_id":
            sample_id,

        "text":
            text,

        "gold":
            row.gold,

        "H":
            row.hisnet_prediction,

        "CURRENT":
            policy_results[
                "CURRENT"
            ][
                "prediction"
            ],

        "CURRENT_pos":
            policy_results[
                "CURRENT"
            ][
                "positive"
            ],

        "CURRENT_neg":
            policy_results[
                "CURRENT"
            ][
                "negative"
            ],

        "NO_SELF":
            policy_results[
                "NO_SELF"
            ][
                "prediction"
            ],

        "NO_SELF_pos":
            policy_results[
                "NO_SELF"
            ][
                "positive"
            ],

        "NO_SELF_neg":
            policy_results[
                "NO_SELF"
            ][
                "negative"
            ],

        "POSITIVE_LEXICON_ONLY":
            policy_results[
                "POSITIVE_LEXICON_ONLY"
            ][
                "prediction"
            ],

        "POS_ONLY_pos":
            policy_results[
                "POSITIVE_LEXICON_ONLY"
            ][
                "positive"
            ],

        "POS_ONLY_neg":
            policy_results[
                "POSITIVE_LEXICON_ONLY"
            ][
                "negative"
            ],

        "self_morph_scope_count":
            policy_results[
                "CURRENT"
            ][
                "self_morph_scope_count"
            ],
    })


    # --------------------------------------------------------
    # Scope-level self-negation diagnostics
    # --------------------------------------------------------

    for scope in (
        hd_result.get(
            "scopes",
            []
        )
        or []
    ):

        sentiment_token = (
            scope.get(
                "sentiment_token"
            )
            or {}
        )


        self_negations = [

            negation

            for negation
            in (
                scope.get(
                    "negations",
                    []
                )
                or []
            )

            if (
                is_self_morphological_negation(
                    scope,
                    negation,
                )
            )

        ]


        if not self_negations:
            continue


        scope_rows.append({

            "sample_id":
                sample_id,

            "gold":
                row.gold,

            "text":
                text,

            "scope_word":
                scope.get(
                    "word"
                ),

            "scope_root":
                scope.get(
                    "root"
                ),

            "token_text":
                sentiment_token.get(
                    "text"
                ),

            "token_lemma":
                sentiment_token.get(
                    "lemma"
                ),

            "token_upos":
                sentiment_token.get(
                    "upos"
                ),

            "token_deprel":
                sentiment_token.get(
                    "deprel"
                ),

            "token_feats":
                "; ".join(

                    str(
                        negation.get(
                            "feats"
                        )
                    )

                    for negation
                    in self_negations

                ),

            "original_positive":
                scope.get(
                    "original_positive",
                    scope.get(
                        "positive"
                    ),
                ),

            "original_negative":
                scope.get(
                    "original_negative",
                    scope.get(
                        "negative"
                    ),
                ),

            "current_adjusted_positive":
                scope.get(
                    "adjusted_positive"
                ),

            "current_adjusted_negative":
                scope.get(
                    "adjusted_negative"
                ),
        })


    if (
        run_index
        % 10
        == 0
    ):

        print(

            f"[NEGATION] "
            f"{run_index}/"
            f"{len(negation_cases)}"

        )


# ============================================================
# FULL 500 METRICS
# ============================================================

metric_rows = []


for policy in [

    "CURRENT",
    "NO_SELF",
    "POSITIVE_LEXICON_ONLY",

]:

    metrics = calculate_metrics(
        evaluation,
        policy,
    )


    metric_rows.append({

        "policy":
            policy,

        **metrics,
    })


metrics_df = pd.DataFrame(
    metric_rows
)


# ============================================================
# PAIRWISE CURRENT VS COUNTERFACTUAL
# ============================================================

pairwise_rows = []


for policy in [

    "NO_SELF",
    "POSITIVE_LEXICON_ONLY",

]:

    current_correct = (
        evaluation[
            "CURRENT"
        ]
        ==
        evaluation[
            "gold"
        ]
    )


    policy_correct = (
        evaluation[
            policy
        ]
        ==
        evaluation[
            "gold"
        ]
    )


    current_only = int(

        (
            current_correct
            &
            ~policy_correct
        )
        .sum()

    )


    policy_only = int(

        (
            ~current_correct
            &
            policy_correct
        )
        .sum()

    )


    changed = int(

        (
            evaluation[
                "CURRENT"
            ]
            !=
            evaluation[
                policy
            ]
        )
        .sum()

    )


    pairwise_rows.append({

        "current_vs":
            policy,

        "changed_predictions":
            changed,

        "current_only_correct":
            current_only,

        "policy_only_correct":
            policy_only,

        "net_correct_gain":
            (
                policy_only
                - current_only
            ),

        "mcnemar_exact_p":
            exact_mcnemar_p(
                current_only,
                policy_only,
            ),
    })


pairwise_df = pd.DataFrame(
    pairwise_rows
)


# ============================================================
# SAVE
# ============================================================

evaluation.to_csv(

    OUTPUT_DIR
    / "full_500_counterfactual_predictions.csv",

    index=False,
    encoding="utf-8-sig",
)


pd.DataFrame(
    case_rows
).to_csv(

    OUTPUT_DIR
    / "negation_cases.csv",

    index=False,
    encoding="utf-8-sig",
)


pd.DataFrame(
    scope_rows
).to_csv(

    OUTPUT_DIR
    / "self_morphological_negation_scopes.csv",

    index=False,
    encoding="utf-8-sig",
)


metrics_df.to_csv(

    OUTPUT_DIR
    / "policy_metrics.csv",

    index=False,
    encoding="utf-8-sig",
)


pairwise_df.to_csv(

    OUTPUT_DIR
    / "policy_pairwise.csv",

    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# PRINT
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "FULL 500 POLICY METRICS"
)

print(
    "=" * 100
)


print(

    metrics_df.to_string(

        index=False,

        float_format=lambda x:
            f"{x:.6f}"

    )

)


print(
    "\n"
    + "=" * 100
)

print(
    "CURRENT VS COUNTERFACTUAL"
)

print(
    "=" * 100
)


print(

    pairwise_df.to_string(

        index=False,

        float_format=lambda x:
            f"{x:.6f}"

    )

)


print(
    "\nResults:"
)

print(
    OUTPUT_DIR
)