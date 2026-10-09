import json
import os
import time
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

from src.evaluation.dataset_loader import (
    load_sentiment_dataset
)

from src.hisnet.hisnet_classifier import (
    HisNetClassifier
)

from src.hisnet.hisnet_dependency_classifier import (
    HisNetDependencyClassifier
)

from src.hisnet.dependency_hisnet_classifier import (
    DependencyHisNetClassifier
)

from src.llm.ollama_zero_shot_classifier import (
    OllamaZeroShotSentimentClassifier
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

DATASET_PATH = Path(
    "datasets/Modified/sentence_sentiment/"
    "TRSAv1_fixed.csv"
)

OUTPUT_DIR = Path(
    "results/trsav1_four_systems_500"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


SAMPLE_SIZE = 500

RANDOM_STATE = 42

LABELS = [
    "negative",
    "neutral",
    "positive"
]


# ============================================================
# LLM SETTINGS
# ============================================================

MODEL_NAME = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:32b"
)


LLM_BATCH_SIZES = [
    100,
    50,
    20,
    10
]


TEMPERATURE = 0.1

TOP_P = 1.0

SEED = 42


# ------------------------------------------------------------
# IMPORTANT:
#
# We use the SAME context size for every batch-size condition.
#
# 4096 may become too tight for 100 reviews + structured JSON
# output, so this benchmark uses 8192 consistently.
#
# If batch=100 exceeds available memory/context and fails,
# the failure is recorded rather than silently modifying the
# experiment.
# ------------------------------------------------------------

NUM_CTX = 8192


KEEP_ALIVE = "30m"

MAX_RETRIES = 2


# ============================================================
# FILES
# ============================================================

SAMPLE_FILE = (
    OUTPUT_DIR
    / "sample_500.csv"
)

CLASSICAL_FILE = (
    OUTPUT_DIR
    / "classical_predictions_500.csv"
)

CLASSICAL_METRICS_FILE = (
    OUTPUT_DIR
    / "classical_metrics_500.csv"
)

FINAL_METRICS_FILE = (
    OUTPUT_DIR
    / "all_metrics_500.csv"
)

LLM_BENCHMARK_FILE = (
    OUTPUT_DIR
    / "llm_batch_benchmark.csv"
)

LLM_AGREEMENT_FILE = (
    OUTPUT_DIR
    / "llm_batch_agreement.csv"
)


# ============================================================
# HELPERS
# ============================================================

def calculate_metrics(
    dataframe,
    prediction_column
):

    valid = dataframe[
        dataframe[
            prediction_column
        ].isin(
            LABELS
        )
    ].copy()

    if len(valid) != len(dataframe):

        return {
            "n":
                len(valid),

            "accuracy":
                None,

            "macro_precision":
                None,

            "macro_recall":
                None,

            "macro_f1":
                None,

            "weighted_f1":
                None
        }

    gold = valid[
        "gold"
    ]

    predictions = valid[
        prediction_column
    ]


    accuracy = (
        accuracy_score(
            gold,
            predictions
        )
    )


    (
        macro_precision,
        macro_recall,
        macro_f1,
        _
    ) = (
        precision_recall_fscore_support(
            gold,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0
        )
    )


    (
        _,
        _,
        weighted_f1,
        _
    ) = (
        precision_recall_fscore_support(
            gold,
            predictions,
            labels=LABELS,
            average="weighted",
            zero_division=0
        )
    )


    return {
        "n":
            len(valid),

        "accuracy":
            accuracy,

        "macro_precision":
            macro_precision,

        "macro_recall":
            macro_recall,

        "macro_f1":
            macro_f1,

        "weighted_f1":
            weighted_f1
    }


def save_confusion_matrix(
    dataframe,
    prediction_column,
    system_name,
    filename
):

    valid = dataframe[
        dataframe[
            prediction_column
        ].isin(
            LABELS
        )
    ]


    if len(valid) != len(dataframe):

        print(
            f"[SKIP] Confusion matrix for "
            f"{system_name}: unresolved predictions exist."
        )

        return


    cm = confusion_matrix(
        valid[
            "gold"
        ],
        valid[
            prediction_column
        ],
        labels=LABELS
    )


    cm_df = pd.DataFrame(
        cm,

        index=[
            f"gold_{label}"
            for label in LABELS
        ],

        columns=[
            f"pred_{label}"
            for label in LABELS
        ]
    )


    print(
        "\n"
        + "=" * 100
    )

    print(
        f"CONFUSION MATRIX: {system_name}"
    )

    print(
        "=" * 100
    )

    print(
        cm_df
    )


    cm_df.to_csv(
        OUTPUT_DIR
        / filename,

        encoding=
            "utf-8-sig"
    )


def atomic_save_csv(
    dataframe,
    path
):
    """
    Avoid corrupted checkpoints if Python/Windows stops
    while writing the file.
    """

    tmp_path = Path(
        str(path)
        + ".tmp"
    )

    dataframe.to_csv(
        tmp_path,
        index=False,
        encoding="utf-8-sig"
    )

    tmp_path.replace(
        path
    )


# ============================================================
# LOAD / CREATE EXACT SAMPLE
# ============================================================

print(
    "=" * 100
)

print(
    "TRSAv1 FOUR-SYSTEM 500-REVIEW EXPERIMENT"
)

print(
    "=" * 100
)


if SAMPLE_FILE.exists():

    print(
        "\nExisting 500-review sample found."
    )

    print(
        "Reusing:"
    )

    print(
        SAMPLE_FILE
    )

    sample = pd.read_csv(
        SAMPLE_FILE
    )

else:

    print(
        "\nLoading full TRSAv1..."
    )

    dataset = (
        load_sentiment_dataset(
            file_path=
                DATASET_PATH,

            text_column=
                "review",

            label_column=
                "score",

            separator=
                ";"
        )
    )


    print(
        "Full dataset:",
        len(
            dataset
        )
    )


    # --------------------------------------------------------
    # STRATIFIED RANDOM SAMPLE
    # --------------------------------------------------------

    _, sample = train_test_split(

        dataset,

        test_size=
            SAMPLE_SIZE,

        random_state=
            RANDOM_STATE,

        stratify=
            dataset[
                "gold"
            ]
    )


    sample = (
        sample
        .reset_index(
            drop=True
        )
    )


    sample.insert(
        0,
        "sample_id",
        range(
            len(sample)
        )
    )


    sample.to_csv(
        SAMPLE_FILE,
        index=False,
        encoding="utf-8-sig"
    )


print(
    "\nSample size:",
    len(
        sample
    )
)


print(
    "\nGold distribution:"
)

print(
    sample[
        "gold"
    ].value_counts()
)


assert (
    len(sample)
    == SAMPLE_SIZE
)


# ============================================================
# THREE NON-LLM SYSTEMS
#
# If already completed, do not waste time rerunning Stanza.
# ============================================================

if CLASSICAL_FILE.exists():

    print(
        "\n"
        + "=" * 100
    )

    print(
        "EXISTING CLASSICAL RESULTS FOUND"
    )

    print(
        "Reusing:"
    )

    print(
        CLASSICAL_FILE
    )

    print(
        "=" * 100
    )


    classical_results = (
        pd.read_csv(
            CLASSICAL_FILE
        )
    )

else:

    print(
        "\n"
        + "=" * 100
    )

    print(
        "RUNNING THREE NON-LLM SYSTEMS"
    )

    print(
        "=" * 100
    )


    print(
        "\nInitializing HisNet..."
    )

    hisnet = (
        HisNetClassifier()
    )


    print(
        "Initializing HisNet -> Dependency..."
    )

    h_to_d = (
        HisNetDependencyClassifier()
    )


    print(
        "Initializing Dependency -> HisNet..."
    )

    d_to_h = (
        DependencyHisNetClassifier()
    )


    rows = []


    for row_index, row in (
        sample.iterrows()
    ):

        text = (
            row[
                "text"
            ]
        )

        gold = (
            row[
                "gold"
            ]
        )


        # ====================================================
        # 1. HISNET
        # ====================================================

        h_result = (
            hisnet.classify(
                text
            )
        )


        # ====================================================
        # 2. HISNET -> DEPENDENCY
        #
        # Exact same HisNet result is reused.
        # ====================================================

        hd_result = (
            h_to_d.classify(
                text,
                hisnet_result=
                    h_result
            )
        )


        # ====================================================
        # 3. DEPENDENCY -> HISNET
        # ====================================================

        dh_result = (
            d_to_h.classify(
                text
            )
        )


        rows.append({

            "sample_id":
                row[
                    "sample_id"
                ],

            "text":
                text,

            "gold":
                gold,

            # -----------------------------------------------
            # H
            # -----------------------------------------------

            "hisnet_prediction":
                h_result[
                    "prediction"
                ],

            "hisnet_positive":
                h_result[
                    "positive_score"
                ],

            "hisnet_negative":
                h_result[
                    "negative_score"
                ],

            # -----------------------------------------------
            # H -> D
            # -----------------------------------------------

            "h_to_d_prediction":
                hd_result[
                    "prediction"
                ],

            "h_to_d_positive":
                hd_result[
                    "positive_score"
                ],

            "h_to_d_negative":
                hd_result[
                    "negative_score"
                ],

            # -----------------------------------------------
            # D -> H
            # -----------------------------------------------

            "d_to_h_prediction":
                dh_result[
                    "prediction"
                ],

            "d_to_h_positive":
                dh_result[
                    "positive_score"
                ],

            "d_to_h_negative":
                dh_result[
                    "negative_score"
                ],

            "d_to_h_candidates":
                dh_result[
                    "dependency_candidate_count"
                ],

            "d_to_h_sentiment_words":
                dh_result[
                    "sentiment_word_count"
                ]
        })


        if (
            (row_index + 1)
            % 25
            == 0
        ):

            print(
                f"[CLASSICAL] "
                f"{row_index + 1}/"
                f"{SAMPLE_SIZE}"
            )


            atomic_save_csv(
                pd.DataFrame(
                    rows
                ),
                CLASSICAL_FILE
            )


    classical_results = (
        pd.DataFrame(
            rows
        )
    )


    atomic_save_csv(
        classical_results,
        CLASSICAL_FILE
    )


# ============================================================
# CLASSICAL METRICS
# ============================================================

classical_systems = {

    "HisNet":
        "hisnet_prediction",

    "HisNet -> Dependency":
        "h_to_d_prediction",

    "Dependency -> HisNet":
        "d_to_h_prediction"
}


classical_metric_rows = []


for system_name, prediction_column in (
    classical_systems.items()
):

    metrics = (
        calculate_metrics(
            classical_results,
            prediction_column
        )
    )


    classical_metric_rows.append({

        "system":
            system_name,

        "batch_size":
            None,

        **metrics
    })


classical_metrics_df = (
    pd.DataFrame(
        classical_metric_rows
    )
)


classical_metrics_df.to_csv(
    CLASSICAL_METRICS_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\n"
    + "=" * 100
)

print(
    "THREE NON-LLM SYSTEMS"
)

print(
    "=" * 100
)


print(
    classical_metrics_df.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
            if pd.notna(x)
            else ""
        )
    )
)


# ============================================================
# LLM EXPERIMENT FUNCTION
# ============================================================

def run_llm_batch_condition(
    batch_size
):

    print(
        "\n"
        + "#" * 100
    )

    print(
        f"QWEN3-32B — BATCH SIZE {batch_size}"
    )

    print(
        "#" * 100
    )


    prediction_file = (
        OUTPUT_DIR
        / (
            f"qwen3_32b_"
            f"batch_{batch_size}_"
            f"predictions.csv"
        )
    )


    batch_log_file = (
        OUTPUT_DIR
        / (
            f"qwen3_32b_"
            f"batch_{batch_size}_"
            f"batch_log.csv"
        )
    )


    # ========================================================
    # LOAD EXISTING CHECKPOINT
    # ========================================================

    if prediction_file.exists():

        prediction_df = (
            pd.read_csv(
                prediction_file
            )
        )


        # Ensure None/NaN statuses are handled.
        completed_ids = set(

            prediction_df.loc[
                prediction_df[
                    "status"
                ]
                == "success",

                "sample_id"
            ].astype(
                int
            )
        )


        print(
            "Checkpoint found."
        )

        print(
            "Already completed:",
            len(
                completed_ids
            )
        )

    else:

        prediction_df = pd.DataFrame(
            columns=[
                "sample_id",
                "text",
                "gold",
                "prediction",
                "status",
                "recovered",
                "error"
            ]
        )

        completed_ids = set()


    # ========================================================
    # INITIALIZE LLM
    # ========================================================

    classifier = (
        OllamaZeroShotSentimentClassifier(

            model=
                MODEL_NAME,

            batch_size=
                batch_size,

            temperature=
                TEMPERATURE,

            top_p=
                TOP_P,

            num_ctx=
                NUM_CTX,

            keep_alive=
                KEEP_ALIVE,

            max_retries=
                MAX_RETRIES,

            seed=
                SEED
        )
    )


    pending = (
        sample[
            ~sample[
                "sample_id"
            ].isin(
                completed_ids
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    print(
        "Remaining reviews:",
        len(
            pending
        )
    )


    # ========================================================
    # RUN BATCHES
    # ========================================================

    initial_failed_ids = []

    batch_logs = []


    condition_start = (
        time.perf_counter()
    )


    total_batches = (

        (
            len(pending)
            + batch_size
            - 1
        )

        // batch_size

        if len(pending) > 0

        else 0
    )


    for batch_number, start in enumerate(

        range(
            0,
            len(pending),
            batch_size
        ),

        start=1
    ):

        batch_df = (
            pending.iloc[
                start:
                start + batch_size
            ]
            .copy()
        )


        texts = (
            batch_df[
                "text"
            ]
            .astype(str)
            .tolist()
        )


        batch_start = (
            time.perf_counter()
        )


        result = (
            classifier.classify_batch(
                texts
            )
        )


        batch_wall_seconds = (
            time.perf_counter()
            - batch_start
        )


        predictions = (
            result[
                "predictions"
            ]
        )


        batch_status = (
            result[
                "status"
            ]
        )


        # ====================================================
        # SAVE EACH REVIEW
        # ====================================================

        new_rows = []


        for local_index, (
            (_, source_row),
            prediction
        ) in enumerate(
            zip(
                batch_df.iterrows(),
                predictions
            )
        ):

            status = (
                "success"
                if prediction
                in LABELS
                else "failed"
            )


            sample_id = int(
                source_row[
                    "sample_id"
                ]
            )


            if status == "failed":

                initial_failed_ids.append(
                    sample_id
                )


            new_rows.append({

                "sample_id":
                    sample_id,

                "text":
                    source_row[
                        "text"
                    ],

                "gold":
                    source_row[
                        "gold"
                    ],

                "prediction":
                    prediction,

                "status":
                    status,

                "recovered":
                    False,

                "error":
                    result.get(
                        "error"
                    )
            })


        # ----------------------------------------------------
        # Remove old versions of these IDs, if any.
        # ----------------------------------------------------

        new_ids = {
            row[
                "sample_id"
            ]
            for row in new_rows
        }


        if len(prediction_df) > 0:

            prediction_df = (
                prediction_df[
                    ~prediction_df[
                        "sample_id"
                    ].isin(
                        new_ids
                    )
                ]
            )


        prediction_df = (
            pd.concat(
                [
                    prediction_df,
                    pd.DataFrame(
                        new_rows
                    )
                ],
                ignore_index=True
            )
        )


        prediction_df = (
            prediction_df
            .sort_values(
                "sample_id"
            )
            .reset_index(
                drop=True
            )
        )


        # ====================================================
        # BATCH LOG
        # ====================================================

        batch_logs.append({

            "batch_number":
                batch_number,

            "configured_batch_size":
                batch_size,

            "actual_batch_size":
                len(
                    texts
                ),

            "status":
                batch_status,

            "wall_seconds":
                batch_wall_seconds,

            "seconds_per_review":
                (
                    batch_wall_seconds
                    / len(texts)
                ),

            "prompt_tokens":
                result.get(
                    "prompt_tokens",
                    0
                ),

            "output_tokens":
                result.get(
                    "output_tokens",
                    0
                ),

            "attempt":
                result.get(
                    "attempt"
                ),

            "error":
                result.get(
                    "error"
                )
        })


        # ====================================================
        # CHECKPOINT AFTER EVERY BATCH
        # ====================================================

        atomic_save_csv(
            prediction_df,
            prediction_file
        )


        pd.DataFrame(
            batch_logs
        ).to_csv(
            batch_log_file,
            index=False,
            encoding="utf-8-sig"
        )


        print(
            f"[QWEN batch={batch_size}] "
            f"{batch_number}/"
            f"{total_batches} "
            f"| rows="
            f"{len(texts)} "
            f"| status="
            f"{batch_status} "
            f"| {batch_wall_seconds:.2f}s"
        )


    first_pass_seconds = (
        time.perf_counter()
        - condition_start
    )


    # ========================================================
    # RECOVERY PASS
    #
    # IMPORTANT:
    # Only unresolved reviews are retried individually.
    # No fake Neutral labels.
    # ========================================================

    unresolved = (
        prediction_df[
            prediction_df[
                "status"
            ]
            != "success"
        ]
        .copy()
    )


    recovery_start = (
        time.perf_counter()
    )


    recovered_count = 0


    if len(unresolved) > 0:

        print(
            "\n"
            + "=" * 100
        )

        print(
            f"RECOVERY — batch={batch_size}"
        )

        print(
            "Unresolved:",
            len(
                unresolved
            )
        )

        print(
            "=" * 100
        )


    for recovery_number, unresolved_row in enumerate(

        unresolved.itertuples(
            index=False
        ),

        start=1
    ):

        result = (
            classifier.classify_batch(
                [
                    unresolved_row.text
                ]
            )
        )


        prediction = (
            result[
                "predictions"
            ][0]
        )


        success = (
            prediction
            in LABELS
        )


        mask = (
            prediction_df[
                "sample_id"
            ]
            == unresolved_row.sample_id
        )


        prediction_df.loc[
            mask,
            "prediction"
        ] = (
            prediction
            if success
            else None
        )


        prediction_df.loc[
            mask,
            "status"
        ] = (
            "success"
            if success
            else "failed"
        )


        prediction_df.loc[
            mask,
            "recovered"
        ] = (
            success
        )


        prediction_df.loc[
            mask,
            "error"
        ] = (
            None
            if success
            else result.get(
                "error"
            )
        )


        if success:

            recovered_count += 1


        atomic_save_csv(
            prediction_df,
            prediction_file
        )


        print(
            f"[RECOVERY batch={batch_size}] "
            f"{recovery_number}/"
            f"{len(unresolved)} "
            f"| success="
            f"{success}"
        )


    recovery_seconds = (
        time.perf_counter()
        - recovery_start
    )


    total_condition_seconds = (
        first_pass_seconds
        + recovery_seconds
    )


    # ========================================================
    # FINAL CONDITION SUMMARY
    # ========================================================

    unresolved_count = int(

        (
            prediction_df[
                "status"
            ]
            != "success"
        )
        .sum()
    )


    stats = (
        classifier.get_stats()
    )


    benchmark = {

        "batch_size":
            batch_size,

        "model":
            MODEL_NAME,

        "num_ctx":
            NUM_CTX,

        "temperature":
            TEMPERATURE,

        "top_p":
            TOP_P,

        "seed":
            SEED,

        "reviews":
            len(
                prediction_df
            ),

        "initial_failed_reviews":
            len(
                set(
                    initial_failed_ids
                )
            ),

        "recovered_reviews":
            recovered_count,

        "unresolved_reviews":
            unresolved_count,

        "first_pass_seconds":
            first_pass_seconds,

        "recovery_seconds":
            recovery_seconds,

        "total_seconds":
            total_condition_seconds,

        "seconds_per_review":
            (
                total_condition_seconds
                / len(sample)
            ),

        "reviews_per_second":
            (
                len(sample)
                / total_condition_seconds
            ),

        "total_requests":
            stats[
                "total_requests"
            ],

        "total_retries":
            stats[
                "total_retries"
            ],

        "total_failed_calls":
            stats[
                "total_failures"
            ],

        "prompt_tokens":
            stats[
                "total_prompt_tokens"
            ],

        "output_tokens":
            stats[
                "total_output_tokens"
            ]
    }


    # --------------------------------------------------------
    # Only calculate final metrics if every review has a
    # real model prediction.
    # --------------------------------------------------------

    if unresolved_count == 0:

        metrics = (
            calculate_metrics(
                prediction_df,
                "prediction"
            )
        )

        benchmark.update(
            metrics
        )

    else:

        benchmark.update({

            "n":
                len(sample)
                - unresolved_count,

            "accuracy":
                None,

            "macro_precision":
                None,

            "macro_recall":
                None,

            "macro_f1":
                None,

            "weighted_f1":
                None
        })


        print(
            "\n"
            "[WARNING] "
            f"batch={batch_size}: "
            f"{unresolved_count} unresolved reviews remain."
        )

        print(
            "Metrics are intentionally NOT calculated."
        )


    return (
        prediction_df,
        benchmark
    )


# ============================================================
# RUN FOUR LLM BATCH CONDITIONS
# ============================================================

llm_results = {}

benchmark_rows = []


for batch_size in LLM_BATCH_SIZES:

    (
        prediction_df,
        benchmark
    ) = (
        run_llm_batch_condition(
            batch_size
        )
    )


    llm_results[
        batch_size
    ] = prediction_df


    benchmark_rows.append(
        benchmark
    )


    # Save after every completed condition.
    pd.DataFrame(
        benchmark_rows
    ).to_csv(
        LLM_BENCHMARK_FILE,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# LLM BATCH BENCHMARK TABLE
# ============================================================

benchmark_df = (
    pd.DataFrame(
        benchmark_rows
    )
)


print(
    "\n"
    + "=" * 100
)

print(
    "QWEN3-32B BATCH-SIZE BENCHMARK"
)

print(
    "=" * 100
)


columns_to_print = [

    "batch_size",

    "accuracy",

    "macro_f1",

    "weighted_f1",

    "total_seconds",

    "seconds_per_review",

    "reviews_per_second",

    "initial_failed_reviews",

    "recovered_reviews",

    "unresolved_reviews"
]


print(
    benchmark_df[
        columns_to_print
    ].to_string(
        index=False,

        float_format=lambda x: (
            f"{x:.4f}"
            if pd.notna(x)
            else ""
        )
    )
)


# ============================================================
# LLM BATCH-SIZE AGREEMENT
#
# Does batching itself change predictions?
# ============================================================

agreement_rows = []


for i in range(
    len(
        LLM_BATCH_SIZES
    )
):

    for j in range(
        i + 1,
        len(
            LLM_BATCH_SIZES
        )
    ):

        batch_a = (
            LLM_BATCH_SIZES[
                i
            ]
        )

        batch_b = (
            LLM_BATCH_SIZES[
                j
            ]
        )


        df_a = (
            llm_results[
                batch_a
            ][
                [
                    "sample_id",
                    "gold",
                    "prediction"
                ]
            ]
            .rename(
                columns={
                    "prediction":
                        "prediction_a"
                }
            )
        )


        df_b = (
            llm_results[
                batch_b
            ][
                [
                    "sample_id",
                    "prediction"
                ]
            ]
            .rename(
                columns={
                    "prediction":
                        "prediction_b"
                }
            )
        )


        merged = pd.merge(
            df_a,
            df_b,
            on="sample_id",
            how="inner"
        )


        valid = (
            merged[
                merged[
                    "prediction_a"
                ].isin(
                    LABELS
                )
                &
                merged[
                    "prediction_b"
                ].isin(
                    LABELS
                )
            ]
            .copy()
        )


        if len(valid) == 0:

            continue


        same = (
            valid[
                "prediction_a"
            ]
            ==
            valid[
                "prediction_b"
            ]
        )


        agreement_count = int(
            same.sum()
        )


        changed_count = (
            len(valid)
            - agreement_count
        )


        agreement_rate = (
            agreement_count
            / len(valid)
        )


        a_correct = (
            valid[
                "prediction_a"
            ]
            ==
            valid[
                "gold"
            ]
        )


        b_correct = (
            valid[
                "prediction_b"
            ]
            ==
            valid[
                "gold"
            ]
        )


        a_only_correct = int(
            (
                a_correct
                &
                ~b_correct
            )
            .sum()
        )


        b_only_correct = int(
            (
                ~a_correct
                &
                b_correct
            )
            .sum()
        )


        both_correct = int(
            (
                a_correct
                &
                b_correct
            )
            .sum()
        )


        both_wrong = int(
            (
                ~a_correct
                &
                ~b_correct
            )
            .sum()
        )


        agreement_rows.append({

            "batch_a":
                batch_a,

            "batch_b":
                batch_b,

            "n":
                len(valid),

            "agreement_count":
                agreement_count,

            "changed_count":
                changed_count,

            "agreement_rate":
                agreement_rate,

            "batch_a_only_correct":
                a_only_correct,

            "batch_b_only_correct":
                b_only_correct,

            "both_correct":
                both_correct,

            "both_wrong":
                both_wrong,

            "net_correct_b_minus_a":
                (
                    b_only_correct
                    - a_only_correct
                )
        })


agreement_df = (
    pd.DataFrame(
        agreement_rows
    )
)


agreement_df.to_csv(
    LLM_AGREEMENT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\n"
    + "=" * 100
)

print(
    "LLM BATCH-SIZE PREDICTION AGREEMENT"
)

print(
    "=" * 100
)


if len(
    agreement_df
) > 0:

    print(
        agreement_df.to_string(
            index=False,

            float_format=lambda x: (
                f"{x:.4f}"
                if pd.notna(x)
                else ""
            )
        )
    )


# ============================================================
# ALL SYSTEM METRICS
#
# Three non-LLM systems + Qwen under four batching conditions.
#
# Conceptually this is still four systems:
#
# H
# H->D
# D->H
# LLM
#
# but LLM is shown under four inference batching conditions.
# ============================================================

all_metric_rows = (
    classical_metric_rows.copy()
)


for benchmark in benchmark_rows:

    batch_size = (
        benchmark[
            "batch_size"
        ]
    )


    all_metric_rows.append({

        "system":
            "Qwen3-32B",

        "batch_size":
            batch_size,

        "n":
            benchmark.get(
                "n"
            ),

        "accuracy":
            benchmark.get(
                "accuracy"
            ),

        "macro_precision":
            benchmark.get(
                "macro_precision"
            ),

        "macro_recall":
            benchmark.get(
                "macro_recall"
            ),

        "macro_f1":
            benchmark.get(
                "macro_f1"
            ),

        "weighted_f1":
            benchmark.get(
                "weighted_f1"
            )
    })


all_metrics_df = (
    pd.DataFrame(
        all_metric_rows
    )
)


all_metrics_df.to_csv(
    FINAL_METRICS_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\n"
    + "=" * 100
)

print(
    "ALL SYSTEM RESULTS"
)

print(
    "=" * 100
)


print(
    all_metrics_df.to_string(

        index=False,

        float_format=lambda x: (
            f"{x:.4f}"
            if pd.notna(x)
            else ""
        )
    )
)


# ============================================================
# CONFUSION MATRICES — CLASSICAL
# ============================================================

save_confusion_matrix(
    classical_results,
    "hisnet_prediction",
    "HisNet",
    "confusion_hisnet.csv"
)


save_confusion_matrix(
    classical_results,
    "h_to_d_prediction",
    "HisNet -> Dependency",
    "confusion_h_to_d.csv"
)


save_confusion_matrix(
    classical_results,
    "d_to_h_prediction",
    "Dependency -> HisNet",
    "confusion_d_to_h.csv"
)


# ============================================================
# CONFUSION MATRICES — LLM CONDITIONS
# ============================================================

for batch_size, prediction_df in (
    llm_results.items()
):

    save_confusion_matrix(

        prediction_df,

        "prediction",

        f"Qwen3-32B batch={batch_size}",

        (
            f"confusion_qwen3_32b_"
            f"batch_{batch_size}.csv"
        )
    )


# ============================================================
# FINISH
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "500-REVIEW FOUR-SYSTEM EXPERIMENT COMPLETED"
)

print(
    "=" * 100
)

print(
    "Results directory:"
)

print(
    OUTPUT_DIR
)

print(
    "\nImportant files:"
)

print(
    "Sample:"
)

print(
    SAMPLE_FILE
)

print(
    "\nAll metrics:"
)

print(
    FINAL_METRICS_FILE
)

print(
    "\nLLM batch benchmark:"
)

print(
    LLM_BENCHMARK_FILE
)

print(
    "\nLLM batch agreement:"
)

print(
    LLM_AGREEMENT_FILE
)

print(
    "=" * 100
)