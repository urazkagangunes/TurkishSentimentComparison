from pathlib import Path

from src.evaluation.dataset_loader import (
    load_sentiment_dataset
)

from src.evaluation.sentiment_evaluator import (
    SentimentEvaluator
)


DATASET_PATH = Path(
    "datasets/Modified/sentence_sentiment/"
    "TRSAv1_fixed.csv"
)

OUTPUT_DIRECTORY = Path(
    "results/trsav1_fixed"
)


# =============================================================
# LOAD DATASET
# =============================================================

dataset = load_sentiment_dataset(
    file_path=DATASET_PATH,
    text_column="review",
    label_column="score",
    separator=";",
    sample_per_class=2000,
    random_state=42
)


print(
    "\nDataset loaded."
)

print(
    "Total examples:",
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


# =============================================================
# INITIALIZE EVALUATOR
# =============================================================

evaluator = (
    SentimentEvaluator()
)


# =============================================================
# EVALUATE
# =============================================================

results, summary = (
    evaluator.evaluate(
        dataframe=dataset,

        dataset_name=
            "trsav1_6000_threshold005",

        output_directory=
            OUTPUT_DIRECTORY,

        progress_every=100,

        resume=True
    )
)


# =============================================================
# CACHE / PERFORMANCE STATISTICS
# =============================================================

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
    "HisNet lookup requests:",
    cache_stats[
        "hisnet_lookup_requests"
    ]
)

print(
    "HisNet lookup cache hits:",
    cache_stats[
        "hisnet_lookup_cache_hits"
    ]
)

print(
    "HisNet lookup cache misses:",
    cache_stats[
        "hisnet_lookup_cache_misses"
    ]
)

print(
    "HisNet cached roots:",
    cache_stats[
        "hisnet_cached_roots"
    ]
)


morphology_stats = (
    cache_stats[
        "morphology"
    ]
)


print(
    "\nMorphology:"
)

for key, value in (
    morphology_stats.items()
):

    print(
        f"{key}: {value}"
    )