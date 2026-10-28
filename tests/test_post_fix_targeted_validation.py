from pathlib import Path

import pandas as pd

from src.hisnet.hisnet_classifier import HisNetClassifier
from src.hisnet.hisnet_dependency_classifier import (
    HisNetDependencyClassifier
)


# ============================================================
# PATHS
# ============================================================

OLD_RESULTS = Path(
    "results/trsav1_four_classical_500_post_negation_freeze/"
    "predictions_500.csv"
)

OUTPUT_DIR = Path(
    "results/post_fix_targeted_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD OLD 500 RUN
# ============================================================

if not OLD_RESULTS.exists():
    raise FileNotFoundError(
        f"Could not find old frozen predictions: {OLD_RESULTS}"
    )


old = pd.read_csv(
    OLD_RESULTS
)


required_columns = {
    "sample_id",
    "text",
    "gold",
    "hisnet_prediction",
    "h_to_d_prediction",
    "h_to_d_unmatched_scope_count",
}


missing = (
    required_columns
    - set(old.columns)
)


if missing:
    raise RuntimeError(
        f"Missing columns: {sorted(missing)}"
    )


# ============================================================
# SELECT PREVIOUSLY UNMATCHED REVIEWS
# ============================================================

unmatched_cases = old[
    old[
        "h_to_d_unmatched_scope_count"
    ]
    > 0
].copy()


print(
    "=" * 100
)

print(
    "POST-FIX TARGETED VALIDATION"
)

print(
    "=" * 100
)

print(
    "\nPreviously unmatched reviews:",
    len(unmatched_cases)
)


# ============================================================
# INITIALIZE CURRENT SYSTEM
# ============================================================

hisnet = HisNetClassifier()

h_to_d = HisNetDependencyClassifier(
    hisnet_classifier=hisnet
)


# ============================================================
# RERUN PREVIOUSLY UNMATCHED REVIEWS
# ============================================================

rows = []
scope_rows = []


for index, row in enumerate(
    unmatched_cases.itertuples(
        index=False
    ),
    start=1
):

    sample_id = int(
        row.sample_id
    )

    text = str(
        row.text
    )

    gold = row.gold


    h_result = hisnet.classify(
        text
    )


    hd_result = h_to_d.classify(
        text,
        hisnet_result=h_result,
    )


    new_unmatched = (
        hd_result.get(
            "unmatched_scopes",
            []
        )
        or []
    )


    scopes = (
        hd_result.get(
            "scopes",
            []
        )
        or []
    )


    fragment_scopes = [
        scope
        for scope in scopes
        if (
            str(
                scope.get(
                    "alignment_strategy"
                )
                or ""
            ).startswith(
                "fragmented_"
            )
            or
            str(
                scope.get(
                    "alignment_strategy"
                )
                or ""
            ).startswith(
                "edge_noise_"
            )
            or
            str(
                scope.get(
                    "alignment_strategy"
                )
                or ""
            ).startswith(
                "decorated_"
            )
        )
    ]


    rows.append({

        "sample_id":
            sample_id,

        "text":
            text,

        "gold":
            gold,

        "old_H":
            row.hisnet_prediction,

        "old_H_to_D":
            row.h_to_d_prediction,

        "new_H":
            h_result[
                "prediction"
            ],

        "new_H_to_D":
            hd_result[
                "prediction"
            ],

        "old_unmatched_scope_count":
            int(
                row.h_to_d_unmatched_scope_count
            ),

        "new_unmatched_scope_count":
            len(
                new_unmatched
            ),

        "fallback_matched_scope_count":
            len(
                fragment_scopes
            ),

        "prediction_changed":
            (
                row.h_to_d_prediction
                !=
                hd_result[
                    "prediction"
                ]
            ),
    })


    for scope in scopes:

        scope_rows.append({

            "sample_id":
                sample_id,

            "text":
                text,

            "word":
                scope.get(
                    "word"
                ),

            "root":
                scope.get(
                    "root"
                ),

            "alignment_status":
                scope.get(
                    "alignment_status"
                ),

            "alignment_strategy":
                scope.get(
                    "alignment_strategy"
                ),

            "fragment_token_ids":
                str(
                    scope.get(
                        "fragment_token_ids"
                    )
                ),

            "reconstructed_surface":
                scope.get(
                    "reconstructed_surface"
                ),

            "dependency_token":
                (
                    (
                        scope.get(
                            "sentiment_token"
                        )
                        or {}
                    ).get(
                        "word"
                    )
                ),

            "target":
                (
                    (
                        scope.get(
                            "target"
                        )
                        or {}
                    ).get(
                        "word"
                    )
                ),

            "negations":
                str([
                    neg.get(
                        "word"
                    )
                    for neg in (
                        scope.get(
                            "negations",
                            []
                        )
                        or []
                    )
                ]),

            "operations":
                str(
                    scope.get(
                        "operations",
                        []
                    )
                ),
        })


    print(
        f"[{index:02d}/{len(unmatched_cases):02d}] "
        f"sample={sample_id} | "
        f"old unmatched="
        f"{row.h_to_d_unmatched_scope_count} | "
        f"new unmatched="
        f"{len(new_unmatched)} | "
        f"fallback matched="
        f"{len(fragment_scopes)}"
    )


results = pd.DataFrame(
    rows
)

scopes_df = pd.DataFrame(
    scope_rows
)


results.to_csv(
    OUTPUT_DIR
    / "unmatched_before_after.csv",

    index=False,
    encoding="utf-8-sig",
)


scopes_df.to_csv(
    OUTPUT_DIR
    / "scope_alignment_details.csv",

    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "UNMATCHED SUMMARY"
)

print(
    "=" * 100
)


print(
    "Old unmatched scopes:",
    int(
        results[
            "old_unmatched_scope_count"
        ].sum()
    )
)


print(
    "New unmatched scopes:",
    int(
        results[
            "new_unmatched_scope_count"
        ].sum()
    )
)


print(
    "Fallback-matched scopes:",
    int(
        results[
            "fallback_matched_scope_count"
        ].sum()
    )
)


print(
    "Reviews whose H->D prediction changed:",
    int(
        results[
            "prediction_changed"
        ].sum()
    )
)


print(
    "\nAlignment strategies:"
)


print(
    scopes_df[
        "alignment_strategy"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)


# ============================================================
# SELF-MORPH NEGATION SMOKE TEST
# ============================================================

SELF_MORPH_SENTENCES = [

    "Hiçbir faydasını göremedim.",

    "Sivilce için işe yaramıyor.",

    "Bu ürünü almayın.",

    "Bende sivilce yaptı sevmedim.",

    "Ürünü alamadım.",

]


print(
    "\n"
    + "=" * 100
)

print(
    "SELF-MORPHOLOGICAL NEGATION SMOKE TEST"
)

print(
    "=" * 100
)


self_morph_rows = []


for text in SELF_MORPH_SENTENCES:

    h_result = hisnet.classify(
        text
    )


    hd_result = h_to_d.classify(
        text,
        hisnet_result=h_result,
    )


    print(
        f"\nTEXT: {text}"
    )

    print(
        "H:",
        h_result[
            "prediction"
        ],
        "| H->D:",
        hd_result[
            "prediction"
        ]
    )


    for scope in (
        hd_result.get(
            "scopes",
            []
        )
        or []
    ):

        token = (
            scope.get(
                "sentiment_token"
            )
            or {}
        )


        negations = (
            scope.get(
                "negations",
                []
            )
            or []
        )


        print(
            "  ",
            scope.get(
                "word"
            ),
            "| root=",
            scope.get(
                "root"
            ),
            "| token=",
            token.get(
                "word"
            ),
            "| neg=",
            [
                neg.get(
                    "word"
                )
                for neg
                in negations
            ],
            "| ops=",
            scope.get(
                "operations",
                []
            ),
        )


        self_morph_rows.append({

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

            "dependency_token":
                token.get(
                    "word"
                ),

            "negations":
                str([
                    neg.get(
                        "word"
                    )
                    for neg
                    in negations
                ]),

            "operations":
                str(
                    scope.get(
                        "operations",
                        []
                    )
                ),
        })


pd.DataFrame(
    self_morph_rows
).to_csv(

    OUTPUT_DIR
    / "self_morph_smoke_test.csv",

    index=False,
    encoding="utf-8-sig",
)


print(
    "\n"
    + "=" * 100
)

print(
    "DONE"
)

print(
    "=" * 100
)

print(
    "Output:",
    OUTPUT_DIR
)