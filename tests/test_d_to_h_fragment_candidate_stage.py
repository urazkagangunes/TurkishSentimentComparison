from collections import Counter
from pathlib import Path

import pandas as pd

from src.hisnet.hisnet_classifier import HisNetClassifier
from src.hisnet.hisnet_dependency_classifier import (
    HisNetDependencyClassifier,
)
from src.hisnet.dependency_hisnet_classifier import (
    DependencyHisNetClassifier,
)


OLD_RESULTS = Path(
    "results/trsav1_four_classical_500_post_negation_freeze/"
    "predictions_500.csv"
)

OUTPUT_DIR = Path(
    "results/d_to_h_fragment_candidate_stage"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD CASES
# ============================================================

old = pd.read_csv(
    OLD_RESULTS
)

cases = old[
    old[
        "h_to_d_unmatched_scope_count"
    ]
    > 0
].copy()


# ============================================================
# MODELS
# ============================================================

hisnet = HisNetClassifier()

h_to_d = HisNetDependencyClassifier(
    hisnet_classifier=hisnet
)

d_to_h = DependencyHisNetClassifier()


# ============================================================
# HELPERS
# ============================================================

def is_fallback(scope):

    strategy = str(
        scope.get(
            "alignment_strategy"
        )
        or ""
    )

    return (
        strategy.startswith("fragmented_")
        or strategy.startswith("edge_noise_")
        or strategy.startswith("decorated_")
    )


# ============================================================
# RUN
# ============================================================

rows = []

group_counter = Counter()


for row in cases.itertuples(
    index=False
):

    text = str(
        row.text
    )

    sample_id = int(
        row.sample_id
    )


    # --------------------------------------------------------
    # H + H->D
    # --------------------------------------------------------

    h_result = hisnet.classify(
        text
    )

    hd_result = h_to_d.classify(
        text,
        hisnet_result=h_result,
    )


    # --------------------------------------------------------
    # D->H
    # --------------------------------------------------------

    dh_result = d_to_h.classify(
        text
    )


    dependency_tokens = (
        dh_result.get(
            "dependency_tokens",
            []
        )
        or []
    )


    token_by_id = {

        int(token["id"]):
            token

        for token
        in dependency_tokens

        if token.get("id") is not None
    }


    candidates = (
        dh_result.get(
            "candidate_diagnostics",
            []
        )
        or []
    )


    candidate_by_token_id = {}

    for candidate in candidates:

        token = (
            candidate.get(
                "sentiment_token"
            )
            or {}
        )

        token_id = token.get(
            "id"
        )

        if token_id is None:
            continue

        candidate_by_token_id.setdefault(
            int(token_id),
            []
        ).append(
            candidate
        )


    # ========================================================
    # H->D RECOVERED SCOPES
    # ========================================================

    for scope in (
        hd_result.get(
            "scopes",
            []
        )
        or []
    ):

        if not is_fallback(
            scope
        ):
            continue


        fragment_ids = (
            scope.get(
                "fragment_token_ids"
            )
            or []
        )


        fragment_candidate_count = 0

        fragment_details = []


        for token_id in fragment_ids:

            token_id = int(
                token_id
            )

            token = token_by_id.get(
                token_id,
                {}
            )

            token_candidates = (
                candidate_by_token_id.get(
                    token_id,
                    []
                )
            )


            if token_candidates:

                fragment_candidate_count += 1


            fragment_details.append({

                "id":
                    token_id,

                "text":
                    token.get(
                        "text"
                    ),

                "lemma":
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

                "is_d_to_h_candidate":
                    bool(
                        token_candidates
                    ),

                "candidate_details": [

                    {
                        "word":
                            candidate.get(
                                "word"
                            ),

                        "root":
                            candidate.get(
                                "root"
                            ),

                        "candidate_strategy":
                            candidate.get(
                                "candidate_strategy"
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
                    }

                    for candidate
                    in token_candidates
                ],
            })


        # ====================================================
        # GROUP DIAGNOSIS
        # ====================================================

        if not fragment_ids:

            diagnosis = (
                "NO_FRAGMENT_IDS"
            )

        elif (
            fragment_candidate_count
            > 0
        ):

            diagnosis = (
                "FRAGMENT_HAS_DEPENDENCY_CANDIDATE"
            )

        else:

            diagnosis = (
                "NO_FRAGMENT_IS_DEPENDENCY_CANDIDATE"
            )


        group_counter[
            diagnosis
        ] += 1


        rows.append({

            "sample_id":
                sample_id,

            "gold":
                row.gold,

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

            "D_to_H":
                dh_result[
                    "prediction"
                ],

            "expected_word":
                scope.get(
                    "word"
                ),

            "expected_root":
                scope.get(
                    "root"
                ),

            "alignment_strategy":
                scope.get(
                    "alignment_strategy"
                ),

            "reconstructed_surface":
                scope.get(
                    "reconstructed_surface"
                ),

            "fragment_ids":
                str(
                    fragment_ids
                ),

            "fragment_candidate_count":
                fragment_candidate_count,

            "diagnosis":
                diagnosis,

            "fragment_details":
                str(
                    fragment_details
                ),
        })


# ============================================================
# SAVE
# ============================================================

result = pd.DataFrame(
    rows
)

result.to_csv(
    OUTPUT_DIR
    / "fragment_candidate_stage.csv",

    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# PRINT
# ============================================================

print(
    "=" * 110
)

print(
    "D->H FRAGMENT CANDIDATE-STAGE DIAGNOSTIC"
)

print(
    "=" * 110
)


print(
    "\nRecovered fallback scopes:",
    len(result)
)


print(
    "\nDiagnosis:"
)


for (
    diagnosis,
    count
) in group_counter.most_common():

    print(
        f"  {diagnosis}: {count}"
    )


print(
    "\n"
    + "=" * 110
)

print(
    "DETAIL"
)

print(
    "=" * 110
)


for row in result.itertuples(
    index=False
):

    print(
        f"\nSample {row.sample_id}"
    )

    print(
        f"  word: "
        f"{row.expected_word}"
    )

    print(
        f"  strategy: "
        f"{row.alignment_strategy}"
    )

    print(
        f"  diagnosis: "
        f"{row.diagnosis}"
    )

    print(
        f"  fragments: "
        f"{row.fragment_details}"
    )


print(
    "\nSaved:"
)

print(
    OUTPUT_DIR
    / "fragment_candidate_stage.csv"
)