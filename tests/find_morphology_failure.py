from src.common.morphology import MorphologicalRootExtractor
from src.evaluation.dataset_loader import load_sentiment_dataset


DATASET_PATH = (
    "datasets/Modified/"
    "sentence_sentiment/"
    "beyazperde_sentiment.csv"
)


dataset = load_sentiment_dataset(
    file_path=DATASET_PATH,
    text_column="Review",
    label_column="Score",
    separator=";",
    sample_per_class=100,
    random_state=42
)


print("Total examples:", len(dataset))


for index, row in dataset.iterrows():

    # Every dataset row gets a fresh analyzer for diagnosis.
    extractor = MorphologicalRootExtractor()

    text = row["text"]

    try:
        extractor.analyze_sentence(text)

    except Exception as e:

        print("\n" + "=" * 100)
        print("FAILED ROW:", index)
        print("GOLD:", row["gold"])
        print("TEXT:", text)
        print("ERROR:", repr(e))

        cleaned = extractor._prepare_sentence(text)

        print("\nCLEANED:")
        print(cleaned)

        tokens = cleaned.split()

        # =====================================================
        # IMPORTANT:
        # Use a NEW analyzer.
        #
        # The failed robust analysis may mutate the analyzer's
        # internal dictionary/trie, so reusing it can hide the
        # original failure.
        # =====================================================

        fresh_extractor = MorphologicalRootExtractor()

        print("\nFRESH SEQUENTIAL TOKEN ANALYSIS:")

        all_parses = []

        failed = False

        for position, token in enumerate(tokens):

            print(
                f"  [{position:02}] TRY  {repr(token)}"
            )

            try:

                parses = (
                    fresh_extractor.analyzer
                    .robustMorphologicalAnalysis(token)
                )

                all_parses.append(parses)

                print(
                    f"       PASS parses={parses.size()}"
                )

            except Exception as token_error:

                print(
                    f"       >>> FAIL TOKEN: {repr(token)}"
                )

                print(
                    f"       ERROR: {repr(token_error)}"
                )

                failed = True
                break

        # =====================================================
        # Only if every token analysis succeeds:
        # test disambiguation using the same fresh instance.
        # =====================================================

        if not failed:

            print(
                "\nAll token analyses passed "
                "with the fresh analyzer."
            )

            print(
                "Testing LongestRootFirstDisambiguation..."
            )

            try:

                selected = (
                    fresh_extractor.disambiguator
                    .disambiguate(all_parses)
                )

                print(
                    "DISAMBIGUATION PASS:",
                    len(selected),
                    "selected parses"
                )

                print(
                    "\nTesting selected parse conversion..."
                )

                for position, (
                    token,
                    parse
                ) in enumerate(
                    zip(tokens, selected)
                ):

                    try:

                        analysis = (
                            parse.transitionList()
                        )

                        root = analysis.split(
                            "+",
                            1
                        )[0]

                        print(
                            f"  [{position:02}] "
                            f"{token!r} -> "
                            f"{root!r} -> "
                            f"{analysis}"
                        )

                    except Exception as parse_error:

                        print(
                            f"  >>> SELECTED PARSE FAIL "
                            f"at {position}: {token!r}"
                        )

                        print(
                            "      ERROR:",
                            repr(parse_error)
                        )

                        break

            except Exception as dis_error:

                print(
                    ">>> DISAMBIGUATION FAIL"
                )

                print(
                    "ERROR:",
                    repr(dis_error)
                )

        break

else:

    print(
        "\nNo morphological-analysis failures found."
    )