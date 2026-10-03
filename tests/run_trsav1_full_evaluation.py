from pathlib import Path

from src.evaluation.dataset_loader import (
    load_sentiment_dataset
)

from src.evaluation.sentiment_evaluator import (
    SentimentEvaluator
)


# =====================================================================
# CONFIGURATION
# =====================================================================

DATASET_PATH = Path(
    "datasets/Modified/sentence_sentiment/"
    "TRSAv1_fixed.csv"
)

OUTPUT_DIRECTORY = Path(
    "results/trsav1_full_conservative_negation"
)

DATASET_NAME = (
    "trsav1_full_threshold005_conservative_negation"
)


# =====================================================================
# LOAD FULL DATASET
# =====================================================================

print(
    "=" * 90
)

print(
    "LOADING FULL TRSAv1 DATASET"
)

print(
    "=" * 90
)


dataset = load_sentiment_dataset(
    file_path=DATASET_PATH,
    text_column="review",
    label_column="score",
    separator=";"

    # IMPORTANT:
    # sample_per_class kullanılmıyor.
    # Böylece tüm kullanılabilir dataset alınır.
)


print(
    "\nDataset loaded."
)

print(
    "Total evaluation examples:",
    len(dataset)
)

print(
    "\nClass distribution:"
)

print(
    dataset[
        "gold"
    ].value_counts()
)

print(
    "\nClass proportions:"
)

print(
    (
        dataset[
            "gold"
        ]
        .value_counts(
            normalize=True
        )
        .sort_index()
    )
)


# =====================================================================
# INITIALIZE
# =====================================================================

print(
    "\n"
    + "=" * 90
)

print(
    "INITIALIZING EVALUATOR"
)

print(
    "=" * 90
)


evaluator = (
    SentimentEvaluator()
)


# =====================================================================
# FULL EVALUATION
# =====================================================================

results, summary = (
    evaluator.evaluate(
        dataframe=dataset,

        dataset_name=
            DATASET_NAME,

        output_directory=
            OUTPUT_DIRECTORY,

        # Full dataset olduğu için her 500 örnekte
        # progress + checkpoint yeterli.
        progress_every=500,

        # Crash / elektrik / terminal kapanması durumunda
        # kaldığı yerden devam etsin.
        resume=True
    )
)


# =====================================================================
# CACHE STATISTICS
# =====================================================================

print(
    "\n"
    + "=" * 90
)

print(
    "CACHE STATISTICS"
)

print(
    "=" * 90
)


cache_stats = (
    evaluator
    .hisnet
    .get_cache_stats()
)


print(
    "\nHisNet:"
)

print(
    "Lookup requests:",
    cache_stats.get(
        "hisnet_lookup_requests"
    )
)

print(
    "Lookup cache hits:",
    cache_stats.get(
        "hisnet_lookup_cache_hits"
    )
)

print(
    "Lookup cache misses:",
    cache_stats.get(
        "hisnet_lookup_cache_misses"
    )
)

print(
    "Cached roots:",
    cache_stats.get(
        "hisnet_cached_roots"
    )
)


print(
    "\nMorphology:"
)


morphology_stats = (
    cache_stats.get(
        "morphology",
        {}
    )
)


for key, value in (
    morphology_stats.items()
):

    print(
        f"{key}: {value}"
    )


# =====================================================================
# DONE
# =====================================================================

print(
    "\n"
    + "=" * 90
)

print(
    "FULL TRSAv1 EVALUATION COMPLETED"
)

print(
    "=" * 90
)

print(
    "Dataset:",
    DATASET_NAME
)

print(
    "Examples:",
    len(dataset)
)

print(
    "Results directory:",
    OUTPUT_DIRECTORY
)