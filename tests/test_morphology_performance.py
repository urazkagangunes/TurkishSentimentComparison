from time import perf_counter
from pathlib import Path

import pandas as pd

from src.common.morphology import (
    MorphologicalRootExtractor
)


DATASET_PATH = Path(
    "datasets/Modified/sentence_sentiment/"
    "beyazperde_sentiment.csv"
)

SAMPLE_SIZE = 200


# =============================================================
# LOAD DATASET
# =============================================================

dataset = pd.read_csv(
    DATASET_PATH,
    sep=";",
    encoding="utf-8-sig"
)


if "Review" not in dataset.columns:

    raise ValueError(
        "Review column could not be found."
    )


dataset = (
    dataset[
        ["Review"]
    ]
    .dropna()
)


dataset["Review"] = (
    dataset["Review"]
    .astype(str)
    .str.strip()
)


dataset = dataset[
    dataset["Review"] != ""
]


dataset = (
    dataset
    .iloc[:SAMPLE_SIZE]
    .reset_index(drop=True)
)


print(
    "=" * 90
)

print(
    "MORPHOLOGY PERFORMANCE TEST"
)

print(
    "=" * 90
)

print(
    f"Dataset: {DATASET_PATH}"
)

print(
    f"Examples: {len(dataset)}"
)


# =============================================================
# INITIALIZE
# =============================================================

extractor = (
    MorphologicalRootExtractor(
        use_robust_cache=True
    )
)


# =============================================================
# FIRST PASS
# =============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "PASS 1 - COLD CACHE"
)

print(
    "=" * 90
)


start = perf_counter()


for number, text in enumerate(
    dataset["Review"],
    start=1
):

    extractor.analyze_sentence(
        text
    )

    if number % 25 == 0:

        print(
            f"[{number}/{len(dataset)}]"
        )


first_pass_time = (
    perf_counter()
    - start
)


first_stats = (
    extractor.get_cache_stats()
)


print(
    "\nPASS 1 COMPLETED"
)

print(
    f"Elapsed time: "
    f"{first_pass_time:.2f} seconds"
)


print(
    "\nCache statistics:"
)


for key, value in (
    first_stats.items()
):

    print(
        f"{key}: {value}"
    )


# =============================================================
# SAVE COUNTERS BEFORE PASS 2
# =============================================================

before_second = (
    extractor.get_cache_stats()
)


# =============================================================
# SECOND PASS
# =============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "PASS 2 - WARM CACHE"
)

print(
    "=" * 90
)


start = perf_counter()


for number, text in enumerate(
    dataset["Review"],
    start=1
):

    extractor.analyze_sentence(
        text
    )

    if number % 25 == 0:

        print(
            f"[{number}/{len(dataset)}]"
        )


second_pass_time = (
    perf_counter()
    - start
)


after_second = (
    extractor.get_cache_stats()
)


print(
    "\nPASS 2 COMPLETED"
)

print(
    f"Elapsed time: "
    f"{second_pass_time:.2f} seconds"
)


# =============================================================
# PASS 2 DELTAS
# =============================================================

second_pass_robust_requests = (
    after_second[
        "robust_requests"
    ]
    - before_second[
        "robust_requests"
    ]
)


second_pass_cache_hits = (
    after_second[
        "robust_cache_hits"
    ]
    - before_second[
        "robust_cache_hits"
    ]
)


second_pass_cache_misses = (
    after_second[
        "robust_cache_misses"
    ]
    - before_second[
        "robust_cache_misses"
    ]
)


second_pass_analyzer_builds = (
    after_second[
        "robust_analyzer_builds"
    ]
    - before_second[
        "robust_analyzer_builds"
    ]
)


# =============================================================
# SUMMARY
# =============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "PERFORMANCE SUMMARY"
)

print(
    "=" * 90
)


print(
    f"Pass 1 time: "
    f"{first_pass_time:.2f} sec"
)

print(
    f"Pass 2 time: "
    f"{second_pass_time:.2f} sec"
)


if second_pass_time > 0:

    speedup = (
        first_pass_time
        / second_pass_time
    )

    print(
        f"Warm-cache speedup: "
        f"{speedup:.2f}x"
    )


print(
    "\nFIRST PASS"
)

print(
    "-" * 55
)

print(
    "Normal requests:",
    first_stats[
        "normal_requests"
    ]
)

print(
    "Normal analyzer resets:",
    first_stats[
        "normal_analyzer_resets"
    ]
)

print(
    "Robust requests:",
    first_stats[
        "robust_requests"
    ]
)

print(
    "Robust cache hits:",
    first_stats[
        "robust_cache_hits"
    ]
)

print(
    "Robust cache misses:",
    first_stats[
        "robust_cache_misses"
    ]
)

print(
    "Actual robust analyzer builds:",
    first_stats[
        "robust_analyzer_builds"
    ]
)

print(
    "Robust failures:",
    first_stats[
        "robust_failures"
    ]
)

print(
    "Unique robust cached words:",
    first_stats[
        "robust_cached_words"
    ]
)


print(
    "\nSECOND PASS DELTA"
)

print(
    "-" * 55
)

print(
    "Robust requests:",
    second_pass_robust_requests
)

print(
    "Cache hits:",
    second_pass_cache_hits
)

print(
    "Cache misses:",
    second_pass_cache_misses
)

print(
    "New analyzer builds:",
    second_pass_analyzer_builds
)


# =============================================================
# ESTIMATED CONSTRUCTION SAVINGS
# =============================================================

old_build_count = (
    first_stats[
        "robust_requests"
    ]
)

new_build_count = (
    first_stats[
        "robust_analyzer_builds"
    ]
)


avoided_builds = (
    old_build_count
    - new_build_count
)


print(
    "\n"
    + "=" * 90
)

print(
    "ANALYZER CONSTRUCTION SAVINGS"
)

print(
    "=" * 90
)


print(
    "Old design analyzer builds "
    "(estimated):",
    old_build_count
)

print(
    "New design analyzer builds:",
    new_build_count
)

print(
    "Avoided analyzer builds:",
    avoided_builds
)


if old_build_count > 0:

    reduction = (
        avoided_builds
        / old_build_count
        * 100.0
    )

    print(
        f"Analyzer build reduction: "
        f"{reduction:.2f}%"
    )


# Using user's measured approximate build cost.
LOW_BUILD_SECONDS = 3.4
HIGH_BUILD_SECONDS = 3.8


saved_low = (
    avoided_builds
    * LOW_BUILD_SECONDS
)

saved_high = (
    avoided_builds
    * HIGH_BUILD_SECONDS
)


print(
    "\nEstimated construction time saved "
    "on these 200 reviews:"
)

print(
    f"{saved_low / 60:.2f} - "
    f"{saved_high / 60:.2f} minutes"
)


print(
    "\n"
    + "=" * 90
)

print(
    "TEST COMPLETED"
)

print(
    "=" * 90
)