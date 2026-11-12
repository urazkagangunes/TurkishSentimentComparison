from pathlib import Path

import pandas as pd


LABEL_MAP = {
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",

    "Positive": "positive",
    "Negative": "negative",
    "Neutral": "neutral",

    "POSITIVE": "positive",
    "NEGATIVE": "negative",
    "NEUTRAL": "neutral",
}


def load_sentiment_dataset(
    file_path,
    text_column,
    label_column,
    separator=";",
    sample_per_class=None,
    random_state=42
):
    """
    Loads a sentence/document sentiment dataset.

    Parameters
    ----------
    file_path : str or Path
        CSV dataset path.

    text_column : str
        Column containing raw text.

    label_column : str
        Column containing gold sentiment labels.

    separator : str
        CSV delimiter.

    sample_per_class : int or None
        If specified, draws the same number of examples
        from each sentiment class.

    random_state : int
        Reproducibility seed.

    Returns
    -------
    pd.DataFrame
        Columns:
            text
            gold
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        sep=separator,
        encoding="utf-8-sig"
    )

    if text_column not in df.columns:
        raise ValueError(
            f"Text column '{text_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    if label_column not in df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[
        [
            text_column,
            label_column
        ]
    ].copy()

    df.columns = [
        "text",
        "gold"
    ]

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------

    df = df.dropna(
        subset=[
            "text",
            "gold"
        ]
    )

    # ---------------------------------------------------------
    # Normalize text
    # ---------------------------------------------------------

    df["text"] = (
        df["text"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["text"] != ""
    ]

    # ---------------------------------------------------------
    # Normalize labels
    # ---------------------------------------------------------

    df["gold"] = (
        df["gold"]
        .astype(str)
        .str.strip()
    )

    unknown_labels = sorted(
        set(df["gold"])
        - set(LABEL_MAP.keys())
    )

    if unknown_labels:
        raise ValueError(
            "Unknown sentiment labels found: "
            f"{unknown_labels}"
        )

    df["gold"] = (
        df["gold"]
        .map(LABEL_MAP)
    )

    # ---------------------------------------------------------
    # Remove exact duplicates
    #
    # IMPORTANT:
    # We only remove duplicate text+label pairs.
    # Conflicting duplicate texts are preserved so they can
    # later be inspected as annotation inconsistency.
    # ---------------------------------------------------------

    df = df.drop_duplicates(
        subset=[
            "text",
            "gold"
        ]
    )

    df = df.reset_index(
        drop=True
    )

    # ---------------------------------------------------------
    # Optional stratified sample
    # ---------------------------------------------------------

    if sample_per_class is not None:

        sampled_frames = []

        for label, group in df.groupby(
            "gold"
        ):

            if len(group) < sample_per_class:
                raise ValueError(
                    f"Class '{label}' contains only "
                    f"{len(group)} examples, but "
                    f"{sample_per_class} requested."
                )

            sampled = group.sample(
                n=sample_per_class,
                random_state=random_state
            )

            sampled_frames.append(
                sampled
            )

        df = pd.concat(
            sampled_frames,
            ignore_index=True
        )

        # Shuffle after class-wise sampling.
        df = df.sample(
            frac=1.0,
            random_state=random_state
        ).reset_index(
            drop=True
        )

    return df