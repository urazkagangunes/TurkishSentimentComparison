import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.hisnet.hisnet_classifier import HisNetClassifier
from src.hisnet.hisnet_dependency_classifier import HisNetDependencyClassifier
from src.hisnet.dependency_hisnet_classifier import DependencyHisNetClassifier


RESULTS_DIR = Path(
    "results/trsav1_four_classical_500_post_negation_freeze"
)

PREDICTIONS_FILE = (
    RESULTS_DIR
    / "predictions_500.csv"
)

OUTPUT_DIR = Path(
    "results/targeted_h_to_d_diagnostics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HELPERS
# ============================================================

def tr_lower(x):
    return (
        str(x or "")
        .replace("I", "ı")
        .replace("İ", "i")
        .lower()
    )


def j(x):
    return json.dumps(
        x,
        ensure_ascii=False,
        default=str,
    )


def token_word(token):
    token = token or {}

    return (
        token.get("word")
        or token.get("text")
    )


def flatten_tokens(hd_result):

    output = []

    for sentence in (
        hd_result.get(
            "sentences",
            []
        )
        or []
    ):

        for token in (
            sentence.get(
                "dependency_tokens",
                []
            )
            or []
        ):

            output.append({
                **token,

                "sentence_index":
                    token.get(
                        "sentence_index",
                        sentence.get(
                            "sentence_index"
                        ),
                    ),
            })

    return output


def alignment_score(
    sentiment_word,
    token,
):
    """
    Mirrors the current H->D alignment priority:

    exact surface             -> 100
    token surface == H root   -> 90
    Stanza lemma == H root    -> 80
    """

    surface = tr_lower(
        sentiment_word.get(
            "word"
        )
    )

    root = tr_lower(
        sentiment_word.get(
            "root"
        )
    )

    text = tr_lower(
        token_word(
            token
        )
    )

    lemma = tr_lower(
        token.get(
            "lemma"
        )
    )

    if text == surface:
        return (
            100,
            "exact_surface",
        )

    if text == root:
        return (
            90,
            "token_surface_equals_hisnet_root",
        )

    if lemma == root:
        return (
            80,
            "stanza_lemma_equals_hisnet_root",
        )

    return (
        0,
        None,
    )


def scope_token_id(
    scope,
):

    token = (
        scope.get(
            "sentiment_token"
        )
        or {}
    )

    if token.get(
        "id"
    ) is None:

        return None

    return int(
        token["id"]
    )


def has_op(
    scope,
    operation,
):

    return (
        operation
        in (
            scope.get(
                "operations",
                []
            )
            or []
        )
    )


# ============================================================
# LOAD FROZEN 500 RESULTS
# ============================================================

if not PREDICTIONS_FILE.exists():

    raise FileNotFoundError(
        f"Missing {PREDICTIONS_FILE}. "
        f"Run test_four_classical_systems_500_frozen first."
    )


pred = pd.read_csv(
    PREDICTIONS_FILE
)


required_columns = {

    "sample_id",
    "text",
    "gold",

    "hisnet_prediction",
    "h_to_d_prediction",
    "d_to_h_prediction",

    "h_to_d_unmatched_scope_count",
    "h_to_d_negation_flips",
}


missing_columns = (
    required_columns
    - set(
        pred.columns
    )
)


if missing_columns:

    raise RuntimeError(
        "Missing columns in predictions_500.csv: "
        + str(
            sorted(
                missing_columns
            )
        )
    )


# ============================================================
# SELECT TARGETED CASES
# ============================================================

pred[
    "unmatched"
] = (
    pred[
        "h_to_d_unmatched_scope_count"
    ]
    > 0
)


# H was correct, H->D made it wrong.
pred[
    "degraded"
] = (

    (
        pred[
            "hisnet_prediction"
        ]
        ==
        pred[
            "gold"
        ]
    )

    &

    (
        pred[
            "h_to_d_prediction"
        ]
        !=
        pred[
            "gold"
        ]
    )
)


# H was wrong, H->D fixed it.
pred[
    "improved"
] = (

    (
        pred[
            "hisnet_prediction"
        ]
        !=
        pred[
            "gold"
        ]
    )

    &

    (
        pred[
            "h_to_d_prediction"
        ]
        ==
        pred[
            "gold"
        ]
    )
)


pred[
    "degraded_neg"
] = (

    pred[
        "degraded"
    ]

    &

    (
        pred[
            "h_to_d_negation_flips"
        ]
        > 0
    )
)


pred[
    "improved_neg"
] = (

    pred[
        "improved"
    ]

    &

    (
        pred[
            "h_to_d_negation_flips"
        ]
        > 0
    )
)


# ============================================================
# TARGET SET
#
# We include:
#
#   unmatched
#   ALL degraded H->D examples
#   improved-negation examples as control group
# ============================================================

target = pred[

    pred[
        "unmatched"
    ]

    |

    pred[
        "degraded"
    ]

    |

    pred[
        "improved_neg"
    ]

].copy()


target = (
    target
    .sort_values(
        "sample_id"
    )
    .reset_index(
        drop=True
    )
)


def categories(
    row,
):

    output = []

    if row.unmatched:

        output.append(
            "UNMATCHED_SCOPE"
        )

    if row.degraded_neg:

        output.append(
            "H_TO_D_DEGRADED_NEGATION"
        )

    elif row.degraded:

        output.append(
            "H_TO_D_DEGRADED_NONNEGATION"
        )

    if row.improved_neg:

        output.append(
            "H_TO_D_IMPROVED_NEGATION_CONTROL"
        )

    return ";".join(
        output
    )


# ============================================================
# CATEGORY COUNTS
# ============================================================

counts = pd.DataFrame(
    [

        [
            "UNMATCHED_SCOPE",
            int(
                pred[
                    "unmatched"
                ].sum()
            ),
        ],

        [
            "H_TO_D_DEGRADED_ANY",
            int(
                pred[
                    "degraded"
                ].sum()
            ),
        ],

        [
            "H_TO_D_DEGRADED_NEGATION",
            int(
                pred[
                    "degraded_neg"
                ].sum()
            ),
        ],

        [
            "H_TO_D_IMPROVED_ANY",
            int(
                pred[
                    "improved"
                ].sum()
            ),
        ],

        [
            "H_TO_D_IMPROVED_NEGATION_CONTROL",
            int(
                pred[
                    "improved_neg"
                ].sum()
            ),
        ],

        [
            "UNION_TARGETED_REVIEWS",
            len(
                target
            ),
        ],

    ],

    columns=[
        "category",
        "reviews",
    ],
)


counts.to_csv(
    OUTPUT_DIR
    / "category_counts.csv",

    index=False,
    encoding="utf-8-sig",
)


print(
    "=" * 100
)

print(
    "TARGETED H->D DIAGNOSTIC"
)

print(
    "=" * 100
)

print(
    counts.to_string(
        index=False
    )
)


# ============================================================
# INITIALIZE SYSTEMS
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


d_to_h = (
    DependencyHisNetClassifier()
)


# ============================================================
# OUTPUT TABLES
# ============================================================

case_rows = []

token_rows = []

scope_rows = []

negation_rows = []

unmatched_rows = []

d_to_h_rows = []

hisnet_rows = []

morphology_rows = []


REPORT_FILE = (
    OUTPUT_DIR
    / "diagnostic_report.txt"
)


# ============================================================
# RUN TARGETED CASES
# ============================================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "TARGETED H->D DIAGNOSTIC REPORT\n"
    )

    report.write(
        "=" * 120
        + "\n\n"
    )

    report.write(
        counts.to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )


    for (
        case_number,
        row
    ) in enumerate(

        target.itertuples(
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

        gold = (
            row.gold
        )

        case_categories = (
            categories(
                row
            )
        )


        print(

            f"[{case_number}/"
            f"{len(target)}] "

            f"sample_id="
            f"{sample_id} | "

            f"{case_categories}"

        )


        # ====================================================
        # RERUN CURRENT FROZEN SYSTEMS
        # ====================================================

        h_result = (
            hisnet.classify(
                text
            )
        )


        hd_result = (
            h_to_d.classify(

                text,

                hisnet_result=
                    h_result,
            )
        )


        dh_result = (
            d_to_h.classify(
                text
            )
        )


        # ====================================================
        # VERIFY CURRENT CODE == SAVED 500 RUN
        # ====================================================

        saved = (
            pred.loc[
                pred[
                    "sample_id"
                ]
                ==
                sample_id
            ]
            .iloc[0]
        )


        mismatches = []


        checks = [

            (
                "H",

                saved[
                    "hisnet_prediction"
                ],

                h_result[
                    "prediction"
                ],
            ),

            (
                "H->D",

                saved[
                    "h_to_d_prediction"
                ],

                hd_result[
                    "prediction"
                ],
            ),

            (
                "D->H",

                saved[
                    "d_to_h_prediction"
                ],

                dh_result[
                    "prediction"
                ],
            ),
        ]


        for (
            system_name,
            saved_prediction,
            current_prediction
        ) in checks:

            if (
                saved_prediction
                !=
                current_prediction
            ):

                mismatches.append(

                    f"{system_name}: "
                    f"saved="
                    f"{saved_prediction}, "
                    f"current="
                    f"{current_prediction}"

                )


        if mismatches:

            raise RuntimeError(

                f"sample_id={sample_id}: "
                f"current code differs "
                f"from frozen 500 run:\n"

                +

                "\n".join(
                    mismatches
                )

            )


        # ====================================================
        # DEPENDENCY TOKENS
        # ====================================================

        dependency_tokens = (
            flatten_tokens(
                hd_result
            )
        )


        token_by_id = {

            int(
                token[
                    "id"
                ]
            ):
                token

            for token
            in dependency_tokens

            if token.get(
                "id"
            )
            is not None
        }


        scopes = (
            hd_result.get(
                "scopes",
                []
            )
            or []
        )


        unmatched_scopes = (
            hd_result.get(
                "unmatched_scopes",
                []
            )
            or []
        )


        # ====================================================
        # WHICH TOKEN HAS ALREADY BEEN USED BY WHICH SCOPE?
        # ====================================================

        token_owners = {}


        for scope in scopes:

            token_id = (
                scope_token_id(
                    scope
                )
            )

            if token_id is None:

                continue


            token_owners.setdefault(
                token_id,
                [],
            ).append(
                scope.get(
                    "word"
                )
            )


        # ====================================================
        # CASE SUMMARY
        # ====================================================

        case_rows.append({

            "sample_id":
                sample_id,

            "categories":
                case_categories,

            "text":
                text,

            "gold":
                gold,

            "H":
                h_result[
                    "prediction"
                ],

            "H_to_D":
                hd_result[
                    "prediction"
                ],

            "D_to_H":
                dh_result[
                    "prediction"
                ],

            "H_pos":
                h_result[
                    "positive_score"
                ],

            "H_neg":
                h_result[
                    "negative_score"
                ],

            "HD_pos":
                hd_result[
                    "positive_score"
                ],

            "HD_neg":
                hd_result[
                    "negative_score"
                ],

            "unmatched_scopes":
                len(
                    unmatched_scopes
                ),

            "negation_flip_scopes":
                sum(

                    has_op(
                        scope,
                        "negation_flip",
                    )

                    for scope
                    in scopes
                ),

            "D_to_H_candidates":
                dh_result.get(
                    "dependency_candidate_count"
                ),

            "D_to_H_sentiment_words":
                dh_result.get(
                    "sentiment_word_count"
                ),
        })


        # ====================================================
        # STANZA TOKEN TABLE
        # ====================================================

        for token in (
            dependency_tokens
        ):

            token_id = int(
                token[
                    "id"
                ]
            )

            head_id = int(
                token.get(
                    "head",
                    0
                )
                or 0
            )


            token_rows.append({

                "sample_id":
                    sample_id,

                "categories":
                    case_categories,

                "sentence_index":
                    token.get(
                        "sentence_index"
                    ),

                "token_id":
                    token_id,

                "local_id":
                    token.get(
                        "local_id"
                    ),

                "text":
                    token_word(
                        token
                    ),

                "lemma":
                    token.get(
                        "lemma"
                    ),

                "upos":
                    token.get(
                        "upos"
                    ),

                "feats":
                    token.get(
                        "feats"
                    ),

                "head_id":
                    head_id,

                "head_text":
                    (
                        "ROOT"

                        if head_id == 0

                        else token_word(
                            token_by_id.get(
                                head_id
                            )
                        )
                    ),

                "deprel":
                    token.get(
                        "deprel"
                    ),

                "used_by_h_to_d_scope":
                    (
                        token_id
                        in
                        token_owners
                    ),

                "scope_words":
                    ";".join(
                        token_owners.get(
                            token_id,
                            []
                        )
                    ),
            })


        # ====================================================
        # MORPHOLOGY
        # ====================================================

        for (
            index,
            morph
        ) in enumerate(

            h_result.get(
                "morphological_analyses",
                []
            )
            or []

        ):

            morphology_rows.append({

                "sample_id":
                    sample_id,

                "categories":
                    case_categories,

                "index":
                    index,

                "word":
                    morph.get(
                        "word"
                    ),

                "root":
                    morph.get(
                        "root"
                    ),

                "analysis":
                    str(
                        morph.get(
                            "analysis"
                        )
                    ),

                "full_record":
                    j(
                        morph
                    ),
            })


        # ====================================================
        # HISNET SENTIMENT WORDS
        # ====================================================

        for (
            index,
            sentiment_word
        ) in enumerate(

            h_result.get(
                "sentiment_words",
                []
            )
            or []

        ):

            hisnet_rows.append({

                "sample_id":
                    sample_id,

                "categories":
                    case_categories,

                "index":
                    index,

                "word":
                    sentiment_word.get(
                        "word"
                    ),

                "root":
                    sentiment_word.get(
                        "root"
                    ),

                "positive":
                    sentiment_word.get(
                        "positive"
                    ),

                "negative":
                    sentiment_word.get(
                        "negative"
                    ),

                "analysis":
                    str(
                        sentiment_word.get(
                            "analysis"
                        )
                    ),

                "full_record":
                    j(
                        sentiment_word
                    ),
            })


        # ====================================================
        # H->D SCOPES
        # ====================================================

        for (
            scope_index,
            scope
        ) in enumerate(
            scopes
        ):

            sentiment_token = (
                scope.get(
                    "sentiment_token"
                )
                or {}
            )

            target_data = (
                scope.get(
                    "target"
                )
                or {}
            )

            token_id = (
                scope_token_id(
                    scope
                )
            )


            full_sentiment_token = (

                token_by_id.get(
                    token_id
                )

                if token_id is not None

                else None
            )


            negations = (
                scope.get(
                    "negations",
                    []
                )
                or []
            )


            operations = (
                scope.get(
                    "operations",
                    []
                )
                or []
            )


            scope_rows.append({

                "sample_id":
                    sample_id,

                "categories":
                    case_categories,

                "scope_index":
                    scope_index,

                "word":
                    scope.get(
                        "word"
                    ),

                "root":
                    scope.get(
                        "root"
                    ),

                "positive":
                    scope.get(
                        "positive"
                    ),

                "negative":
                    scope.get(
                        "negative"
                    ),

                "alignment_status":
                    scope.get(
                        "alignment_status"
                    ),

                "active_sentiment":
                    scope.get(
                        "active_sentiment"
                    ),

                "token_id":
                    token_id,

                "token_text":
                    token_word(
                        sentiment_token
                    ),

                "token_lemma":
                    sentiment_token.get(
                        "lemma"
                    ),

                "token_upos":
                    sentiment_token.get(
                        "upos"
                    ),

                "token_feats":
                    (
                        full_sentiment_token
                        or {}
                    ).get(
                        "feats"
                    ),

                "token_head":
                    sentiment_token.get(
                        "head"
                    ),

                "token_deprel":
                    sentiment_token.get(
                        "deprel"
                    ),

                "target_word":
                    target_data.get(
                        "word"
                    ),

                "target_root":
                    target_data.get(
                        "root"
                    ),

                "target_strategy":
                    scope.get(
                        "target_strategy"
                    ),

                "negations":
                    j(
                        negations
                    ),

                "modifiers":
                    j(
                        scope.get(
                            "modifiers",
                            []
                        )
                    ),

                "contrast":
                    j(
                        scope.get(
                            "contrast"
                        )
                    ),

                "negation_scope_strategy":
                    scope.get(
                        "negation_scope_strategy"
                    ),

                "operations":
                    ";".join(
                        operations
                    ),

                "original_positive":
                    scope.get(
                        "original_positive"
                    ),

                "original_negative":
                    scope.get(
                        "original_negative"
                    ),

                "adjusted_positive":
                    scope.get(
                        "adjusted_positive"
                    ),

                "adjusted_negative":
                    scope.get(
                        "adjusted_negative"
                    ),
            })


            # =================================================
            # NEGATION FLIP DETAILS
            # =================================================

            if (
                "negation_flip"
                in
                operations
            ):

                negation_details = []

                negation_types = []


                for negation in (
                    negations
                ):

                    negation_id = (
                        negation.get(
                            "id"
                        )
                    )


                    negation_token = (

                        token_by_id.get(
                            int(
                                negation_id
                            )
                        )

                        if negation_id
                        is not None

                        else None
                    )


                    source = (
                        negation_token
                        or negation
                    )


                    lemma = tr_lower(
                        source.get(
                            "lemma"
                        )
                    )

                    surface = tr_lower(
                        token_word(
                            source
                        )
                    )


                    if (
                        lemma == "değil"

                        or

                        surface.startswith(
                            "değil"
                        )
                    ):

                        negation_type = (
                            "explicit_degil"
                        )


                    elif (

                        negation_id
                        is not None

                        and

                        token_id
                        is not None

                        and

                        int(
                            negation_id
                        )
                        ==
                        token_id
                    ):

                        negation_type = (
                            "self_morphological_negation"
                        )


                    else:

                        negation_type = (
                            "support_or_other_negation"
                        )


                    negation_types.append(
                        negation_type
                    )


                    negation_details.append({

                        "id":
                            negation_id,

                        "text":
                            token_word(
                                source
                            ),

                        "lemma":
                            source.get(
                                "lemma"
                            ),

                        "upos":
                            source.get(
                                "upos"
                            ),

                        "head":
                            source.get(
                                "head"
                            ),

                        "deprel":
                            source.get(
                                "deprel"
                            ),

                        "feats":
                            source.get(
                                "feats"
                            ),

                        "type":
                            negation_type,
                    })


                token_feats = (

                    (
                        full_sentiment_token
                        or {}
                    ).get(
                        "feats"
                    )

                    or ""
                )


                negation_rows.append({

                    "sample_id":
                        sample_id,

                    "categories":
                        case_categories,

                    "gold":
                        gold,

                    "text":
                        text,

                    "H":
                        h_result[
                            "prediction"
                        ],

                    "H_to_D":
                        hd_result[
                            "prediction"
                        ],

                    "scope_word":
                        scope.get(
                            "word"
                        ),

                    "scope_root":
                        scope.get(
                            "root"
                        ),

                    "token_id":
                        token_id,

                    "token_text":
                        token_word(
                            sentiment_token
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
                        token_feats,

                    "token_has_Polarity_Neg":
                        (
                            "Polarity=Neg"
                            in
                            token_feats
                        ),

                    "target_word":
                        target_data.get(
                            "word"
                        ),

                    "target_strategy":
                        scope.get(
                            "target_strategy"
                        ),

                    "negation_types":
                        ";".join(
                            negation_types
                        ),

                    "negations":
                        j(
                            negation_details
                        ),

                    "negation_scope_strategy":
                        scope.get(
                            "negation_scope_strategy"
                        ),

                    "original_positive":
                        scope.get(
                            "original_positive"
                        ),

                    "original_negative":
                        scope.get(
                            "original_negative"
                        ),

                    "adjusted_positive":
                        scope.get(
                            "adjusted_positive"
                        ),

                    "adjusted_negative":
                        scope.get(
                            "adjusted_negative"
                        ),
                })


        # ====================================================
        # UNMATCHED ALIGNMENT ANALYSIS
        # ====================================================

        for (
            unmatched_index,
            unmatched
        ) in enumerate(
            unmatched_scopes
        ):

            potential_candidates = []


            for token in (
                dependency_tokens
            ):

                (
                    score,
                    reason
                ) = (
                    alignment_score(
                        unmatched,
                        token
                    )
                )


                if score == 0:

                    continue


                token_id = int(
                    token[
                        "id"
                    ]
                )


                potential_candidates.append((

                    score,
                    reason,
                    token,

                    (
                        token_id
                        in
                        token_owners
                    ),

                    token_owners.get(
                        token_id,
                        []
                    ),
                ))


            potential_candidates.sort(

                key=lambda item: (

                    -item[0],

                    int(
                        item[2][
                            "id"
                        ]
                    ),
                )
            )


            # No token even satisfies the current matcher.
            if not potential_candidates:

                unmatched_rows.append({

                    "sample_id":
                        sample_id,

                    "categories":
                        case_categories,

                    "unmatched_index":
                        unmatched_index,

                    "unmatched_word":
                        unmatched.get(
                            "word"
                        ),

                    "unmatched_root":
                        unmatched.get(
                            "root"
                        ),

                    "positive":
                        unmatched.get(
                            "positive"
                        ),

                    "negative":
                        unmatched.get(
                            "negative"
                        ),

                    "candidate_token_id":
                        None,

                    "candidate_token_text":
                        None,

                    "candidate_token_lemma":
                        None,

                    "candidate_upos":
                        None,

                    "candidate_deprel":
                        None,

                    "alignment_score":
                        0,

                    "alignment_reason":
                        "NO_SURFACE_ROOT_OR_LEMMA_MATCH",

                    "token_already_used":
                        False,

                    "used_by_scope_words":
                        "",

                    "diagnosis":
                        "NO_POTENTIAL_MATCH",
                })


            else:

                for (
                    score,
                    reason,
                    token,
                    used,
                    used_by
                ) in potential_candidates:


                    unmatched_rows.append({

                        "sample_id":
                            sample_id,

                        "categories":
                            case_categories,

                        "unmatched_index":
                            unmatched_index,

                        "unmatched_word":
                            unmatched.get(
                                "word"
                            ),

                        "unmatched_root":
                            unmatched.get(
                                "root"
                            ),

                        "positive":
                            unmatched.get(
                                "positive"
                            ),

                        "negative":
                            unmatched.get(
                                "negative"
                            ),

                        "candidate_token_id":
                            token.get(
                                "id"
                            ),

                        "candidate_token_text":
                            token_word(
                                token
                            ),

                        "candidate_token_lemma":
                            token.get(
                                "lemma"
                            ),

                        "candidate_upos":
                            token.get(
                                "upos"
                            ),

                        "candidate_deprel":
                            token.get(
                                "deprel"
                            ),

                        "alignment_score":
                            score,

                        "alignment_reason":
                            reason,

                        "token_already_used":
                            used,

                        "used_by_scope_words":
                            ";".join(
                                used_by
                            ),

                        "diagnosis":
                            (
                                "POTENTIAL_MATCH_ALREADY_USED"

                                if used

                                else
                                "UNUSED_POTENTIAL_MATCH_EXISTS"
                            ),
                    })


        # ====================================================
        # D->H CANDIDATES FOR COMPARISON
        # ====================================================

        for (
            candidate_index,
            candidate
        ) in enumerate(

            dh_result.get(
                "candidate_diagnostics",
                []
            )
            or []

        ):

            token = (
                candidate.get(
                    "sentiment_token"
                )
                or {}
            )

            target_data = (
                candidate.get(
                    "target"
                )
                or {}
            )


            d_to_h_rows.append({

                "sample_id":
                    sample_id,

                "categories":
                    case_categories,

                "candidate_index":
                    candidate_index,

                "word":
                    candidate.get(
                        "word"
                    ),

                "root":
                    candidate.get(
                        "root"
                    ),

                "token_id":
                    token.get(
                        "id"
                    ),

                "token_text":
                    token_word(
                        token
                    ),

                "token_lemma":
                    token.get(
                        "lemma"
                    ),

                "upos":
                    token.get(
                        "upos"
                    ),

                "head":
                    token.get(
                        "head"
                    ),

                "deprel":
                    token.get(
                        "deprel"
                    ),

                "candidate_strategy":
                    candidate.get(
                        "candidate_strategy"
                    ),

                "target_word":
                    target_data.get(
                        "word"
                    ),

                "target_strategy":
                    candidate.get(
                        "target_strategy"
                    ),

                "lexicon_status":
                    candidate.get(
                        "lexicon_status"
                    ),

                "positive":
                    candidate.get(
                        "positive"
                    ),

                "negative":
                    candidate.get(
                        "negative"
                    ),

                "negations":
                    j(
                        candidate.get(
                            "negations",
                            []
                        )
                    ),

                "modifiers":
                    j(
                        candidate.get(
                            "modifiers",
                            []
                        )
                    ),
            })


        # ====================================================
        # HUMAN-READABLE REPORT
        # ====================================================

        report.write(
            "=" * 120
            + "\n"
        )


        report.write(

            f"sample_id={sample_id} | "
            f"{case_categories}\n"

        )


        report.write(

            f"Gold={gold} | "

            f"H="
            f"{h_result['prediction']} | "

            f"H->D="
            f"{hd_result['prediction']} | "

            f"D->H="
            f"{dh_result['prediction']}\n"

        )


        report.write(
            f"TEXT: {text}\n"
        )


        report.write(

            f"H scores: "
            f"+{h_result['positive_score']} "
            f"-{h_result['negative_score']} | "

            f"H->D scores: "
            f"+{hd_result['positive_score']} "
            f"-{hd_result['negative_score']}\n\n"

        )


        # ----------------------------------------------------
        # HisNet sentiment words
        # ----------------------------------------------------

        report.write(
            "HISNET SENTIMENT WORDS\n"
        )


        for word in (

            h_result.get(
                "sentiment_words",
                []
            )
            or []

        ):

            report.write(

                f"  "
                f"{word.get('word')} | "

                f"root="
                f"{word.get('root')} | "

                f"POS="
                f"{word.get('positive')} | "

                f"NEG="
                f"{word.get('negative')}\n"

            )


        # ----------------------------------------------------
        # Stanza
        # ----------------------------------------------------

        report.write(
            "\nSTANZA TOKENS\n"
        )


        for token in (
            dependency_tokens
        ):

            head_id = int(
                token.get(
                    "head",
                    0
                )
                or 0
            )


            head_text = (

                "ROOT"

                if head_id == 0

                else token_word(
                    token_by_id.get(
                        head_id
                    )
                )
            )


            report.write(

                f"  "
                f"{token.get('id')} "
                f"{token_word(token)} | "

                f"lemma="
                f"{token.get('lemma')} | "

                f"UPOS="
                f"{token.get('upos')} | "

                f"HEAD="
                f"{head_id}:"
                f"{head_text} | "

                f"DEPREL="
                f"{token.get('deprel')} | "

                f"FEATS="
                f"{token.get('feats')}\n"

            )


        # ----------------------------------------------------
        # H->D scopes
        # ----------------------------------------------------

        report.write(
            "\nH->D SCOPES\n"
        )


        for scope in (
            scopes
        ):

            sentiment_token = (
                scope.get(
                    "sentiment_token"
                )
                or {}
            )

            target_data = (
                scope.get(
                    "target"
                )
                or {}
            )


            report.write(

                f"  "
                f"{scope.get('word')} "

                f"root="
                f"{scope.get('root')} | "

                f"align="
                f"{scope.get('alignment_status')} | "

                f"token="
                f"{token_word(sentiment_token)}"
                f"#{sentiment_token.get('id')} | "

                f"target="
                f"{target_data.get('word')} | "

                f"neg="
                f"{[token_word(x) for x in (scope.get('negations', []) or [])]} | "

                f"ops="
                f"{scope.get('operations', [])} | "

                f"after=("
                f"{scope.get('adjusted_positive')},"
                f"{scope.get('adjusted_negative')})\n"

            )


        # ----------------------------------------------------
        # Unmatched details
        # ----------------------------------------------------

        if unmatched_scopes:

            report.write(
                "\nUNMATCHED ALIGNMENT CANDIDATES\n"
            )


            for item in [

                x
                for x
                in unmatched_rows

                if x[
                    "sample_id"
                ]
                ==
                sample_id

            ]:

                report.write(

                    f"  "
                    f"unmatched="
                    f"{item['unmatched_word']} "

                    f"root="
                    f"{item['unmatched_root']} -> "

                    f"token="
                    f"{item['candidate_token_text']}"
                    f"#{item['candidate_token_id']} "

                    f"lemma="
                    f"{item['candidate_token_lemma']} | "

                    f"score="
                    f"{item['alignment_score']} "

                    f"{item['alignment_reason']} | "

                    f"used="
                    f"{item['token_already_used']} "

                    f"by="
                    f"{item['used_by_scope_words']} | "

                    f"{item['diagnosis']}\n"

                )


        # ----------------------------------------------------
        # Negation flips
        # ----------------------------------------------------

        report.write(
            "\nNEGATION FLIPS\n"
        )


        for item in [

            x
            for x
            in negation_rows

            if x[
                "sample_id"
            ]
            ==
            sample_id

        ]:

            report.write(

                f"  "
                f"scope="
                f"{item['scope_word']} "

                f"root="
                f"{item['scope_root']} | "

                f"token="
                f"{item['token_text']} | "

                f"types="
                f"{item['negation_types']} | "

                f"FEATS="
                f"{item['token_feats']} | "

                f"neg="
                f"{item['negations']} | "

                f"before=("
                f"{item['original_positive']},"
                f"{item['original_negative']}) -> "

                f"after=("
                f"{item['adjusted_positive']},"
                f"{item['adjusted_negative']})\n"

            )


        # ----------------------------------------------------
        # D->H candidates
        # ----------------------------------------------------

        report.write(
            "\nD->H CANDIDATES\n"
        )


        for candidate in (

            dh_result.get(
                "candidate_diagnostics",
                []
            )
            or []

        ):

            token = (
                candidate.get(
                    "sentiment_token"
                )
                or {}
            )


            report.write(

                f"  "
                f"{candidate.get('word')} "

                f"root="
                f"{candidate.get('root')} | "

                f"{token.get('upos')}/"
                f"{token.get('deprel')} | "

                f"strategy="
                f"{candidate.get('candidate_strategy')} | "

                f"lexicon="
                f"{candidate.get('lexicon_status')} | "

                f"POS="
                f"{candidate.get('positive')} "

                f"NEG="
                f"{candidate.get('negative')}\n"

            )


        report.write(
            "\n\n"
        )


# ============================================================
# SAVE TABLES
# ============================================================

outputs = {

    "targeted_case_summary.csv":
        case_rows,

    "dependency_tokens.csv":
        token_rows,

    "morphology.csv":
        morphology_rows,

    "hisnet_sentiment_words.csv":
        hisnet_rows,

    "h_to_d_scopes.csv":
        scope_rows,

    "negation_flip_scopes.csv":
        negation_rows,

    "unmatched_alignment_candidates.csv":
        unmatched_rows,

    "d_to_h_candidates.csv":
        d_to_h_rows,
}


for (
    filename,
    rows
) in outputs.items():

    pd.DataFrame(
        rows
    ).to_csv(

        OUTPUT_DIR
        / filename,

        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# FINAL CONSOLE SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "TARGETED DIAGNOSTIC COMPLETED"
)

print(
    "=" * 100
)


print(
    "Selected reviews:",
    len(
        target
    ),
)


print(
    "Unmatched-scope reviews:",
    int(
        pred[
            "unmatched"
        ].sum()
    ),
)


print(
    "H->D degraded reviews:",
    int(
        pred[
            "degraded"
        ].sum()
    ),
)


print(
    "H->D degraded + negation:",
    int(
        pred[
            "degraded_neg"
        ].sum()
    ),
)


print(
    "H->D improved + negation controls:",
    int(
        pred[
            "improved_neg"
        ].sum()
    ),
)


# ============================================================
# UNMATCHED SUMMARY
# ============================================================

if unmatched_rows:

    print(
        "\nUnmatched diagnoses:"
    )


    unmatched_counter = Counter(

        item[
            "diagnosis"
        ]

        for item
        in unmatched_rows
    )


    for (
        name,
        count
    ) in (
        unmatched_counter
        .most_common()
    ):

        print(
            f"  {name}: {count}"
        )


# ============================================================
# NEGATION TYPE SUMMARY
# ============================================================

if negation_rows:

    negation_counter = (
        Counter()
    )


    for item in (
        negation_rows
    ):

        for negation_type in (

            str(
                item[
                    "negation_types"
                ]
            )
            .split(
                ";"
            )

        ):

            if negation_type:

                negation_counter[
                    negation_type
                ] += 1


    print(
        "\nNegation flip types:"
    )


    for (
        name,
        count
    ) in (
        negation_counter
        .most_common()
    ):

        print(
            f"  {name}: {count}"
        )


print(
    "\nResults directory:"
)

print(
    OUTPUT_DIR
)


print(
    "\nMost important files:"
)


print(
    " -",
    OUTPUT_DIR
    / "targeted_case_summary.csv",
)


print(
    " -",
    OUTPUT_DIR
    / "negation_flip_scopes.csv",
)


print(
    " -",
    OUTPUT_DIR
    / "unmatched_alignment_candidates.csv",
)


print(
    " -",
    REPORT_FILE,
)


print(
    "=" * 100
)