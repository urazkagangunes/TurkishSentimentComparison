from pathlib import Path

import pandas as pd

from src.dependency.sentiment_scope_resolver import (
    SentimentScopeResolver
)

from src.hisnet.hisnet_classifier import (
    HisNetClassifier
)

from src.hisnet.hisnet_dependency_classifier import (
    HisNetDependencyClassifier
)


OLD_RESULTS = Path(
    "results/trsav1_four_classical_500_post_negation_freeze/"
    "predictions_500.csv"
)

resolver = SentimentScopeResolver()

hisnet = HisNetClassifier()

h_to_d = HisNetDependencyClassifier(
    hisnet_classifier=hisnet
)


old = pd.read_csv(
    OLD_RESULTS
)


cases = old[
    old[
        "h_to_d_unmatched_scope_count"
    ]
    > 0
]


print(
    "=" * 120
)

print(
    "FRAGMENT ALIGNMENT DEBUG"
)

print(
    "=" * 120
)


for row in cases.itertuples(
    index=False
):

    text = str(
        row.text
    )

    h_result = hisnet.classify(
        text
    )

    hd_result = h_to_d.classify(
        text,
        hisnet_result=h_result
    )


    unmatched = (
        hd_result.get(
            "unmatched_scopes",
            []
        )
        or []
    )


    if not unmatched:
        continue


    print(
        "\n"
        + "=" * 120
    )

    print(
        f"SAMPLE {row.sample_id}"
    )

    print(
        text
    )


    # --------------------------------------------------------
    # Flatten dependency tokens exactly as current H->D sees them.
    # --------------------------------------------------------

    dependency_tokens = []

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

            dependency_tokens.append(
                token
            )


    print(
        "\nDEPENDENCY TOKENS"
    )


    for token in dependency_tokens:

        print(

            f"id={token.get('id'):>3} | "

            f"text={str(token.get('text')):<15} | "

            f"lemma={str(token.get('lemma')):<15} | "

            f"upos={str(token.get('upos')):<6} | "

            f"head={str(token.get('head')):<4} | "

            f"rel={str(token.get('deprel')):<12} | "

            f"sentence_index="
            f"{token.get('sentence_index')}"

        )


    # --------------------------------------------------------
    # Examine every unmatched HisNet word.
    # --------------------------------------------------------

    for scope in unmatched:

        word = scope[
            "word"
        ]

        root = scope[
            "root"
        ]


        sentiment_word = None


        for candidate in (
            h_result.get(
                "sentiment_words",
                []
            )
            or []
        ):

            if (
                candidate.get(
                    "word"
                )
                ==
                word
                and
                candidate.get(
                    "root"
                )
                ==
                root
            ):

                sentiment_word = candidate
                break


        print(
            "\n"
            + "-" * 120
        )

        print(
            f"UNMATCHED: "
            f"word={word}, "
            f"root={root}"
        )


        if sentiment_word is None:

            print(
                "Could not recover HisNet sentiment record."
            )

            continue


        # ====================================================
        # 1. Run matcher again with EMPTY used_token_ids.
        #
        # If this succeeds, original failure was caused by
        # token consumption/order.
        # ====================================================

        fresh_used = set()


        fresh_match = (
            resolver._match_sentiment_token(
                sentiment_word,
                dependency_tokens,
                fresh_used
            )
        )


        print(
            "\nFRESH MATCH:"
        )

        if fresh_match is None:

            print(
                "  NONE"
            )

        else:

            print(
                "  token=",
                fresh_match.get(
                    "text"
                )
            )

            print(
                "  id=",
                fresh_match.get(
                    "id"
                )
            )

            print(
                "  strategy=",
                fresh_match.get(
                    "_alignment_strategy"
                )
            )

            print(
                "  fragment_ids=",
                fresh_match.get(
                    "_fragment_token_ids"
                )
            )

            print(
                "  reconstructed=",
                fresh_match.get(
                    "_reconstructed_surface"
                )
            )


        # ====================================================
        # 2. Print every adjacent 2-token reconstruction.
        # ====================================================

        ordered = sorted(
            dependency_tokens,
            key=lambda token:
                int(
                    token[
                        "id"
                    ]
                )
        )


        print(
            "\nTWO-TOKEN RECONSTRUCTIONS:"
        )


        for i in range(
            len(ordered)
            - 1
        ):

            first = ordered[i]
            second = ordered[i + 1]


            first_id = int(
                first[
                    "id"
                ]
            )

            second_id = int(
                second[
                    "id"
                ]
            )


            if (
                second_id
                !=
                first_id + 1
            ):
                continue


            piece_1 = (
                resolver._normalize_alignment_piece(
                    first.get(
                        "text"
                    )
                )
            )


            piece_2 = (
                resolver._normalize_alignment_piece(
                    second.get(
                        "text"
                    )
                )
            )


            reconstructed = (
                piece_1
                + piece_2
            )


            interesting = (

                reconstructed
                ==
                resolver._turkish_lower(
                    word
                )

                or

                reconstructed
                ==
                resolver._turkish_lower(
                    root
                )
            )


            marker = (
                "  <<< MATCH"
                if interesting
                else ""
            )


            print(

                f"  [{first_id},{second_id}] "

                f"{repr(piece_1)} + "
                f"{repr(piece_2)} "

                f"= "
                f"{repr(reconstructed)} | "

                f"sentence_indexes="
                f"("
                f"{first.get('sentence_index')},"
                f"{second.get('sentence_index')}"
                f")"

                f"{marker}"

            )


        # ====================================================
        # 3. Print three-token reconstruction as well.
        # ====================================================

        print(
            "\nTHREE-TOKEN RECONSTRUCTIONS:"
        )


        for i in range(
            len(ordered)
            - 2
        ):

            group = ordered[
                i:
                i + 3
            ]


            ids = [
                int(
                    token[
                        "id"
                    ]
                )
                for token
                in group
            ]


            if not (
                ids[1]
                ==
                ids[0] + 1
                and
                ids[2]
                ==
                ids[1] + 1
            ):
                continue


            pieces = [

                resolver._normalize_alignment_piece(
                    token.get(
                        "text"
                    )
                )

                for token
                in group
            ]


            reconstructed = "".join(
                pieces
            )


            interesting = (

                reconstructed
                ==
                resolver._turkish_lower(
                    word
                )

                or

                reconstructed
                ==
                resolver._turkish_lower(
                    root
                )
            )


            marker = (
                "  <<< MATCH"
                if interesting
                else ""
            )


            print(

                f"  {ids} "

                f"{pieces} "

                f"= "
                f"{repr(reconstructed)} | "

                f"sentence_indexes="
                f"{[x.get('sentence_index') for x in group]}"

                f"{marker}"

            )


print(
    "\n"
    + "=" * 120
)

print(
    "DONE"
)

print(
    "=" * 120
)