from pathlib import Path
import csv


INPUT_PATH = Path(
    "datasets/Modified/sentence_sentiment/TRSAv1.csv"
)

OUTPUT_PATH = Path(
    "datasets/Modified/sentence_sentiment/TRSAv1_fixed.csv"
)


VALID_LABELS = {
    "Positive",
    "Negative",
    "Neutral",
    "positive",
    "negative",
    "neutral"
}


fixed_rows = []
bad_rows = []


with open(
    INPUT_PATH,
    "r",
    encoding="utf-8-sig"
) as file:

    # Skip header
    header = file.readline()

    for line_number, line in enumerate(
        file,
        start=2
    ):

        line = line.rstrip(
            "\r\n"
        )

        if not line:
            continue

        # =====================================================
        # IMPORTANT:
        #
        # Split ONLY on the LAST semicolon.
        #
        # Everything before it belongs to the review.
        # Everything after it must be the sentiment label.
        # =====================================================

        if ";" not in line:

            bad_rows.append(
                (
                    line_number,
                    line,
                    "no_semicolon"
                )
            )

            continue

        review, score = line.rsplit(
            ";",
            1
        )

        review = review.strip()
        score = score.strip()

        # Remove accidental outer quotes only.
        if (
            review.startswith('"')
            and review.endswith('"')
        ):
            review = review[
                1:-1
            ]

        review = (
            review
            .replace(
                '""',
                '"'
            )
            .strip()
        )

        score = (
            score
            .strip('"')
            .strip()
        )

        if score not in VALID_LABELS:

            bad_rows.append(
                (
                    line_number,
                    line,
                    f"unknown_label={score}"
                )
            )

            continue

        score_normalized = {
            "positive":
                "Positive",

            "negative":
                "Negative",

            "neutral":
                "Neutral"
        }.get(
            score.lower()
        )

        fixed_rows.append(
            (
                review,
                score_normalized
            )
        )


# =============================================================
# WRITE CORRECTLY QUOTED CSV
# =============================================================

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8-sig",
    newline=""
) as file:

    writer = csv.writer(
        file,
        delimiter=";",
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL
    )

    writer.writerow(
        [
            "review",
            "score"
        ]
    )

    writer.writerows(
        fixed_rows
    )


print(
    "=" * 80
)

print(
    "TRSAv1 REPAIR COMPLETED"
)

print(
    "=" * 80
)

print(
    "Valid rows:",
    len(
        fixed_rows
    )
)

print(
    "Bad rows:",
    len(
        bad_rows
    )
)

print(
    "Output:",
    OUTPUT_PATH
)


if bad_rows:

    print(
        "\nFirst bad rows:"
    )

    for row in bad_rows[
        :20
    ]:

        print(
            row
        )