from collections import Counter
from pathlib import Path

import pandas as pd

from src.dependency.sentiment_scope_resolver import (
    SentimentScopeResolver,
)
from src.hisnet.hisnet_classifier import (
    HisNetClassifier,
)
from src.hisnet.hisnet_dependency_classifier import (
    HisNetDependencyClassifier,
)
from src.hisnet.dependency_hisnet_classifier import (
    DependencyHisNetClassifier,
)


# ============================================================
# PATHS
# ============================================================

OLD_RESULTS = Path(
    "results/trsav1_four_classical_500_post_negation_freeze/"
    "predictions_500.csv"
)

OUTPUT_DIR = Path(
    "results/d_to_h_fragmentation_diagnostic"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HELPERS
# ============================================================

def tr_lower(text):
    return (
        str(text or "")
        .replace("I", "ı")
        .replace("İ", "i")
        .lower()
    )


def same_lexical_item(
    expected_word,
    expected_root,
    candidate_word,
    candidate_root,
):
    expected_word = tr_lower(
        expected_word
    )

    expected_root = tr_lower(
        expected_root
    )

    candidate_word = tr_lower(
        candidate_word
    )

    candidate_root = tr_lower(
        candidate_root
    )

    expected_forms = {
        form
        for form in [
            expected_word,
            expected_root,
        ]
        if form
    }

    candidate_forms = {
        form
        for form in [
            candidate_word,
            candidate_root,
        ]
        if form
    }

    return bool(
        expected_forms
        &
        candidate_forms
    )


def fallback_alignment(
    scope,
):
    strategy = str(
        scope.get(
            "alignment_strategy"
        )
        or ""
    )

    return (
        strategy.startswith(
            "fragmented_"
        )
        or strategy.startswith(
            "edge_noise_"
        )
        or strategy.startswith(
            "decorated_"
        )
    )


def reconstruct_dependency_fragments(
    expected_word,
    expected_root,
    dependency_tokens,
    resolver,
):
    """
    Diagnostic only.

    Checks whether adjacent D->H dependency tokens can reconstruct
    the H->D recovered lexical item.

    IMPORTANT:
    This does NOT query HisNet and does NOT alter D->H.
    """

    expected_word = tr_lower(
        expected_word
    )

    expected_root = tr_lower(
        expected_root
    )

    expected_forms = {
        form
        for form in [
            expected_word,
            expected_root,
        ]
        if form
    }

    ordered = sorted(
        [
            token
            for token in dependency_tokens
            if token.get(
                "id"
            ) is not None
        ],
        key=lambda token:
            int(
                token["id"]
            ),
    )

    matches = []

    for group_size in (
        2,
        3,
    ):

        for start in range(
            0,
            len(ordered)
            - group_size
            + 1,
        ):

            group = ordered[
                start:
                start + group_size
            ]

            ids = [
                int(
                    token[
                        "id"
                    ]
                )
                for token in group
            ]

            # Must be globally consecutive.
            if any(
                right
                != left + 1
                for left, right
                in zip(
                    ids,
                    ids[1:]
                )
            ):
                continue

            pieces = [
                resolver._normalize_alignment_piece(
                    token.get(
                        "text"
                    )
                )
                for token in group
            ]

            if any(
                not piece
                for piece in pieces
            ):
                continue

            reconstructed = "".join(
                pieces
            )

            if (
                reconstructed
                not in expected_forms
            ):
                continue

            sentence_indexes = [
                token.get(
                    "sentence_index"
                )
                for token in group
            ]

            matches.append({

                "token_ids":
                    ids,

                "token_texts":
                    [
                        token.get(
                            "text"
                        )
                        for token
                        in group
                    ],

                "pieces":
                    pieces,

                "reconstructed":
                    reconstructed,

                "sentence_indexes":
                    sentence_indexes,

                "cross_sentence":
                    (
                        len({
                            index
                            for index
                            in sentence_indexes
                            if index is not None
                        })
                        > 1
                    ),
            })

    return matches


# ============================================================
# LOAD OLD 500 RUN
# ============================================================

if not OLD_RESULTS.exists():

    raise FileNotFoundError(
        f"Could not find: {OLD_RESULTS}"
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
    "d_to_h_prediction",
    "h_to_d_unmatched_scope_count",
}


missing = (
    required_columns
    - set(
        old.columns
    )
)


if missing:

    raise RuntimeError(
        "Missing columns: "
        + str(
            sorted(
                missing
            )
        )
    )


# ============================================================
# ONLY REVIEWS THAT HAD OLD H->D UNMATCHED SCOPES
# ============================================================

cases = old[
    old[
        "h_to_d_unmatched_scope_count"
    ]
    > 0
].copy()


print(
    "=" * 110
)

print(
    "D->H FRAGMENTATION DIAGNOSTIC"
)

print(
    "=" * 110
)

print(
    "\nPreviously H->D-unmatched reviews:",
    len(
        cases
    ),
)


# ============================================================
# INITIALIZE CURRENT SYSTEMS
# ============================================================

resolver = (
    SentimentScopeResolver()
)

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
# OUTPUTS
# ============================================================

scope_rows = []

review_rows = []


REPORT_FILE = (
    OUTPUT_DIR
    / "console_report.txt"
)


# ============================================================
# RUN
# ============================================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "D->H FRAGMENTATION DIAGNOSTIC\n"
    )

    report.write(
        "=" * 120
        + "\n\n"
    )


    for (
        review_index,
        row
    ) in enumerate(

        cases.itertuples(
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


        print(

            f"[{review_index:02d}/"
            f"{len(cases):02d}] "

            f"sample_id="
            f"{sample_id}"

        )


        # ====================================================
        # CURRENT SYSTEMS
        # ====================================================

        h_result = hisnet.classify(
            text
        )


        hd_result = h_to_d.classify(

            text,

            hisnet_result=
                h_result,
        )


        dh_result = d_to_h.classify(
            text
        )


        # ====================================================
        # ONLY H->D SCOPES THAT WERE RECOVERED BY FALLBACK
        # ====================================================

        recovered_scopes = [

            scope

            for scope
            in (
                hd_result.get(
                    "scopes",
                    []
                )
                or []
            )

            if fallback_alignment(
                scope
            )

        ]


        dependency_tokens = (
            dh_result.get(
                "dependency_tokens",
                []
            )
            or []
        )


        candidate_diagnostics = (
            dh_result.get(
                "candidate_diagnostics",
                []
            )
            or []
        )


        dh_sentiment_words = (
            dh_result.get(
                "sentiment_words",
                []
            )
            or []
        )


        review_found = 0
        review_fragment_miss = 0
        review_other_miss = 0


        # ====================================================
        # HUMAN REPORT HEADER
        # ====================================================

        report.write(
            "=" * 120
            + "\n"
        )

        report.write(

            f"sample_id={sample_id}\n"

        )

        report.write(

            f"Gold={gold} | "

            f"H={h_result['prediction']} | "

            f"H->D="
            f"{hd_result['prediction']} | "

            f"D->H="
            f"{dh_result['prediction']}\n"

        )

        report.write(

            f"TEXT: {text}\n"

        )

        report.write(

            f"Recovered H->D fallback scopes: "
            f"{len(recovered_scopes)}\n\n"

        )


        # ====================================================
        # EACH RECOVERED H->D SCOPE
        # ====================================================

        for scope in (
            recovered_scopes
        ):

            expected_word = (
                scope.get(
                    "word"
                )
            )

            expected_root = (
                scope.get(
                    "root"
                )
            )

            alignment_strategy = (
                scope.get(
                    "alignment_strategy"
                )
            )

            fragment_ids = (
                scope.get(
                    "fragment_token_ids"
                )
            )

            reconstructed_surface = (
                scope.get(
                    "reconstructed_surface"
                )
            )


            # ------------------------------------------------
            # 1. Did D->H actually find it as sentiment?
            # ------------------------------------------------

            matching_sentiment = []

            for candidate in (
                dh_sentiment_words
            ):

                if same_lexical_item(

                    expected_word,
                    expected_root,

                    candidate.get(
                        "word"
                    ),
                    candidate.get(
                        "root"
                    ),

                ):

                    matching_sentiment.append(
                        candidate
                    )


            # ------------------------------------------------
            # 2. Was something equivalent at least generated
            #    as dependency candidate?
            # ------------------------------------------------

            matching_candidates = []

            for candidate in (
                candidate_diagnostics
            ):

                if same_lexical_item(

                    expected_word,
                    expected_root,

                    candidate.get(
                        "word"
                    ),
                    candidate.get(
                        "root"
                    ),

                ):

                    matching_candidates.append(
                        candidate
                    )


            # ------------------------------------------------
            # 3. Can D->H raw dependency tokens reconstruct
            #    the missing lexical item?
            # ------------------------------------------------

            fragment_matches = (
                reconstruct_dependency_fragments(

                    expected_word,
                    expected_root,

                    dependency_tokens,

                    resolver,
                )
            )


            # =================================================
            # CLASSIFICATION
            # =================================================

            if matching_sentiment:

                diagnosis = (
                    "D_TO_H_FOUND_SENTIMENT"
                )

                review_found += 1


            elif matching_candidates:

                diagnosis = (
                    "D_TO_H_CANDIDATE_BUT_NOT_SENTIMENT"
                )

                review_other_miss += 1


            elif fragment_matches:

                diagnosis = (
                    "D_TO_H_FRAGMENTATION_MISS"
                )

                review_fragment_miss += 1


            else:

                diagnosis = (
                    "D_TO_H_OTHER_MISS"
                )

                review_other_miss += 1


            # =================================================
            # CSV ROW
            # =================================================

            scope_rows.append({

                "sample_id":
                    sample_id,

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

                "D_to_H":
                    dh_result[
                        "prediction"
                    ],

                "expected_word":
                    expected_word,

                "expected_root":
                    expected_root,

                "h_to_d_alignment_strategy":
                    alignment_strategy,

                "h_to_d_fragment_token_ids":
                    str(
                        fragment_ids
                    ),

                "h_to_d_reconstructed_surface":
                    reconstructed_surface,

                "d_to_h_found_sentiment":
                    bool(
                        matching_sentiment
                    ),

                "d_to_h_matching_sentiment_words":
                    str([
                        {
                            "word":
                                x.get(
                                    "word"
                                ),

                            "root":
                                x.get(
                                    "root"
                                ),

                            "positive":
                                x.get(
                                    "positive"
                                ),

                            "negative":
                                x.get(
                                    "negative"
                                ),
                        }

                        for x
                        in matching_sentiment
                    ]),

                "d_to_h_matching_candidates":
                    str([
                        {
                            "word":
                                x.get(
                                    "word"
                                ),

                            "root":
                                x.get(
                                    "root"
                                ),

                            "lexicon_status":
                                x.get(
                                    "lexicon_status"
                                ),

                            "candidate_strategy":
                                x.get(
                                    "candidate_strategy"
                                ),
                        }

                        for x
                        in matching_candidates
                    ]),

                "fragment_reconstruction_found":
                    bool(
                        fragment_matches
                    ),

                "fragment_reconstruction_matches":
                    str(
                        fragment_matches
                    ),

                "diagnosis":
                    diagnosis,
            })


            # =================================================
            # HUMAN REPORT
            # =================================================

            report.write(
                "-" * 100
                + "\n"
            )

            report.write(

                f"H->D RECOVERED: "
                f"{expected_word} | "

                f"root="
                f"{expected_root} | "

                f"strategy="
                f"{alignment_strategy} | "

                f"fragment_ids="
                f"{fragment_ids} | "

                f"reconstructed="
                f"{reconstructed_surface}\n"

            )


            report.write(

                f"D->H diagnosis: "
                f"{diagnosis}\n"

            )


            if matching_sentiment:

                report.write(
                    "D->H sentiment match:\n"
                )

                for candidate in (
                    matching_sentiment
                ):

                    report.write(

                        f"  "
                        f"{candidate.get('word')} | "

                        f"root="
                        f"{candidate.get('root')} | "

                        f"POS="
                        f"{candidate.get('positive')} | "

                        f"NEG="
                        f"{candidate.get('negative')}\n"

                    )


            if matching_candidates:

                report.write(
                    "D->H candidate match:\n"
                )

                for candidate in (
                    matching_candidates
                ):

                    report.write(

                        f"  "
                        f"{candidate.get('word')} | "

                        f"root="
                        f"{candidate.get('root')} | "

                        f"lexicon="
                        f"{candidate.get('lexicon_status')} | "

                        f"strategy="
                        f"{candidate.get('candidate_strategy')}\n"

                    )


            if fragment_matches:

                report.write(

                    "D->H dependency-token "
                    "reconstruction:\n"

                )

                for fragment in (
                    fragment_matches
                ):

                    report.write(

                        f"  ids="
                        f"{fragment['token_ids']} | "

                        f"texts="
                        f"{fragment['token_texts']} | "

                        f"reconstructed="
                        f"{fragment['reconstructed']} | "

                        f"sentence_indexes="
                        f"{fragment['sentence_indexes']} | "

                        f"cross_sentence="
                        f"{fragment['cross_sentence']}\n"

                    )


            report.write(
                "\n"
            )


        # ====================================================
        # REVIEW SUMMARY
        # ====================================================

        review_rows.append({

            "sample_id":
                sample_id,

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

            "D_to_H":
                dh_result[
                    "prediction"
                ],

            "recovered_h_to_d_scopes":
                len(
                    recovered_scopes
                ),

            "d_to_h_found":
                review_found,

            "d_to_h_fragmentation_miss":
                review_fragment_miss,

            "d_to_h_other_miss":
                review_other_miss,
        })


        report.write(
            "\n"
        )


# ============================================================
# SAVE
# ============================================================

scope_df = pd.DataFrame(
    scope_rows
)

review_df = pd.DataFrame(
    review_rows
)


scope_df.to_csv(

    OUTPUT_DIR
    / "scope_diagnostic.csv",

    index=False,
    encoding="utf-8-sig",
)


review_df.to_csv(

    OUTPUT_DIR
    / "review_summary.csv",

    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 110
)

print(
    "SUMMARY"
)

print(
    "=" * 110
)


print(
    "\nRecovered H->D fallback scopes:",
    len(
        scope_df
    ),
)


if not scope_df.empty:

    counts = (
        scope_df[
            "diagnosis"
        ]
        .value_counts()
    )


    print(
        "\nD->H diagnoses:"
    )

    print(
        counts.to_string()
    )


    print(
        "\nDiagnosis rates:"
    )

    print(
        (
            counts
            / len(
                scope_df
            )
        )
        .round(
            4
        )
        .to_string()
    )


    fragmentation_misses = (
        scope_df[
            scope_df[
                "diagnosis"
            ]
            ==
            "D_TO_H_FRAGMENTATION_MISS"
        ]
    )


    print(
        "\nFragmentation misses:"
    )


    if (
        fragmentation_misses.empty
    ):

        print(
            "  NONE"
        )

    else:

        for row in (
            fragmentation_misses
            .itertuples(
                index=False
            )
        ):

            print(

                f"  sample="
                f"{row.sample_id} | "

                f"{row.expected_word} | "

                f"root="
                f"{row.expected_root} | "

                f"H->D="
                f"{row.H_to_D} | "

                f"D->H="
                f"{row.D_to_H}"

            )


print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)


print(
    "\nFiles:"
)

print(
    " -",
    OUTPUT_DIR
    / "scope_diagnostic.csv",
)

print(
    " -",
    OUTPUT_DIR
    / "review_summary.csv",
)

print(
    " -",
    REPORT_FILE,
)


print(
    "=" * 110
)