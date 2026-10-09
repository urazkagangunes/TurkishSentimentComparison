import json
import math
import os
import time

from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
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

from src.hisnet.literature_weighted_hisnet_dependency_classifier import (
    LiteratureWeightedHisNetDependencyClassifier
)

from src.llm.ollama_zero_shot_classifier import (
    OllamaZeroShotSentimentClassifier
)

from src.llm.dependency_ollama_zero_shot_classifier import (
    DependencyOllamaZeroShotSentimentClassifier
)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:32b"
)

LLM_BATCH_SIZE = 8

TEMPERATURE = 0.1

TOP_P = 1.0

NUM_CTX = 8192

SEED = 42

KEEP_ALIVE = "30m"

MAX_RETRIES = 2


LABELS = [
    "positive",
    "neutral",
    "negative"
]


SYSTEMS = [
    "H",
    "H->D",
    "D->H",
    "L",
    "D->L",
    "H->D-LW",
]


# ============================================================
# 16-SENTENCE SANITY SET
#
# IMPORTANT:
#
# These are manually specified expected labels for sanity
# testing. This is NOT a benchmark dataset and must NOT be
# used for parameter tuning.
# ============================================================

TEST_CASES = [

    {
        "id": 1,
        "text": "Otel çok güzel.",
        "gold": "positive",
        "focus": "single-word intensifier"
    },

    {
        "id": 2,
        "text": "Otel biraz kötü.",
        "gold": "negative",
        "focus": "mitigator"
    },

    {
        "id": 3,
        "text": "Otel güzel ama odalar kötü.",
        "gold": "negative",
        "focus": "contrast: negative second clause"
    },

    {
        "id": 4,
        "text": "Otel kötü ama odalar güzel.",
        "gold": "positive",
        "focus": "contrast: positive second clause"
    },

    {
        "id": 5,
        "text": "Bu ürün son derece güzel.",
        "gold": "positive",
        "focus": "multiword intensifier + modifier component suppression"
    },

    {
        "id": 6,
        "text": "Bu ürün çok ama çok güzel.",
        "gold": "positive",
        "focus": "multiword modifier containing contrast word"
    },

    {
        "id": 7,
        "text": "Film güzel değildi.",
        "gold": "negative",
        "focus": "negation"
    },

    {
        "id": 8,
        "text": "Bu ürün çok daha kötü.",
        "gold": "negative",
        "focus": "multiword modifier coverage"
    },

    {
        "id": 9,
        "text": "Bu ürün daha da çok güzel.",
        "gold": "positive",
        "focus": "multiword intensifier"
    },

    {
        "id": 10,
        "text": "Bu ürün azıcık kötü.",
        "gold": "negative",
        "focus": "strong mitigation"
    },

    {
        "id": 11,
        "text": "Büyük ekran güzel.",
        "gold": "positive",
        "focus": "modifier false-positive guard"
    },

    {
        "id": 12,
        "text": "Küçük oda çok güzel.",
        "gold": "positive",
        "focus": "modifier attachment guard"
    },

    {
        "id": 13,
        "text": "Otel güzel fakat odalar kötü.",
        "gold": "negative",
        "focus": "fakat contrast"
    },

    {
        "id": 14,
        "text": "Otel kötü ancak odalar güzel.",
        "gold": "positive",
        "focus": "ancak contrast + marker suppression"
    },

    {
        "id": 15,
        "text": "Otel güzel ama odalar kötü değildi.",
        "gold": "positive",
        "focus": "contrast + negation interaction"
    },

    {
        "id": 16,
        "text": "Ürün güzel ama pahalı fakat kaliteli.",
        "gold": "positive",
        "focus": "multiple contrast"
    },
]


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = Path(
    "results/six_systems_16_detailed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


REPORT_PATH = (
    OUTPUT_DIR
    / "console_report.txt"
)


# ============================================================
# REPORTER
#
# Prints both to terminal and a text file.
# ============================================================

class Reporter:

    def __init__(
        self,
        path
    ):

        self.file = open(
            path,
            "w",
            encoding="utf-8"
        )

    def log(
        self,
        *values
    ):

        text = " ".join(
            str(value)
            for value in values
        )

        print(
            text
        )

        self.file.write(
            text
            + "\n"
        )

        self.file.flush()

    def close(
        self
    ):

        self.file.close()


reporter = Reporter(
    REPORT_PATH
)


# ============================================================
# SERIALIZATION HELPERS
# ============================================================

def json_safe(
    value
):
    """
    Converts arbitrary nested diagnostic objects into
    JSON-serializable values.
    """

    if value is None:

        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool
        )
    ):

        return value

    if isinstance(
        value,
        dict
    ):

        return {

            str(key):
                json_safe(
                    item
                )

            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple
        )
    ):

        return [

            json_safe(
                item
            )

            for item
            in value
        ]

    if isinstance(
        value,
        set
    ):

        return sorted(
            json_safe(
                item
            )
            for item
            in value
        )

    # NumPy / pandas scalar support.
    try:

        return value.item()

    except Exception:

        pass

    return str(
        value
    )


def pretty_json(
    value
):

    return json.dumps(
        json_safe(
            value
        ),
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# FORMATTING
# ============================================================

def separator(
    character="=",
    width=120
):

    reporter.log(
        character
        * width
    )


def print_title(
    title
):

    reporter.log()

    separator()

    reporter.log(
        title
    )

    separator()


def print_subtitle(
    title
):

    reporter.log()

    separator(
        "-"
    )

    reporter.log(
        title
    )

    separator(
        "-"
    )


# ============================================================
# EXACT PAIRED BINOMIAL / MCNEMAR-LIKE TEST
# ============================================================

def exact_paired_p_value(
    a_only_correct,
    b_only_correct
):
    """
    Exact two-sided binomial test for paired discordant
    correctness counts.

    Equivalent to the exact McNemar/binomial test for the
    2x2 paired correctness table.

    No scipy dependency is required.
    """

    n = (
        a_only_correct
        + b_only_correct
    )

    if n == 0:

        return 1.0

    smaller = min(
        a_only_correct,
        b_only_correct
    )

    probability = sum(

        math.comb(
            n,
            k
        )

        for k in range(
            smaller + 1
        )
    )

    probability /= (
        2 ** n
    )

    return min(
        1.0,
        2.0
        * probability
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    system_name
):

    unresolved_count = sum(
        prediction is None
        for prediction in y_pred
    )

    result = {

        "system":
            system_name,

        "n":
            len(
                y_true
            ),

        "unresolved":
            unresolved_count
    }

    # --------------------------------------------------------
    # STRICT POLICY:
    #
    # Do not silently drop technical LLM failures.
    # --------------------------------------------------------

    if unresolved_count > 0:

        result.update({

            "accuracy":
                None,

            "macro_precision_present":
                None,

            "macro_recall_present":
                None,

            "macro_f1_present":
                None,

            "macro_precision_3class":
                None,

            "macro_recall_3class":
                None,

            "macro_f1_3class":
                None,

            "weighted_f1":
                None
        })

        return result

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    present_labels = [

        label

        for label in LABELS

        if label in set(
            y_true
        )
    ]

    (
        present_precision,
        present_recall,
        present_f1,
        _
    ) = precision_recall_fscore_support(

        y_true,
        y_pred,

        labels=
            present_labels,

        average=
            "macro",

        zero_division=
            0
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _
    ) = precision_recall_fscore_support(

        y_true,
        y_pred,

        labels=
            LABELS,

        average=
            "macro",

        zero_division=
            0
    )

    (
        _,
        _,
        weighted_f1,
        _
    ) = precision_recall_fscore_support(

        y_true,
        y_pred,

        labels=
            LABELS,

        average=
            "weighted",

        zero_division=
            0
    )

    result.update({

        "accuracy":
            accuracy,

        "macro_precision_present":
            present_precision,

        "macro_recall_present":
            present_recall,

        "macro_f1_present":
            present_f1,

        "macro_precision_3class":
            macro_precision,

        "macro_recall_3class":
            macro_recall,

        "macro_f1_3class":
            macro_f1,

        "weighted_f1":
            weighted_f1
    })

    return result


# ============================================================
# LLM INITIALIZATION
# ============================================================

def create_llm_classifier(
    classifier_class,
    dependency=False
):
    """
    Supports both the local seeded implementation and an older
    implementation where seed may not yet exist.
    """

    kwargs = {

        "model":
            MODEL,

        "batch_size":
            LLM_BATCH_SIZE,

        "temperature":
            TEMPERATURE,

        "top_p":
            TOP_P,

        "num_ctx":
            NUM_CTX,

        "keep_alive":
            KEEP_ALIVE,

        "max_retries":
            MAX_RETRIES,

        "seed":
            SEED
    }

    if dependency:

        kwargs[
            "use_gpu_parser"
        ] = False

    try:

        return classifier_class(
            **kwargs
        )

    except TypeError as error:

        if (
            "seed"
            not in str(
                error
            )
        ):

            raise

        reporter.log(
            "[WARNING] "
            "Current classifier constructor does not accept "
            "'seed'. Retrying without seed."
        )

        kwargs.pop(
            "seed",
            None
        )

        return classifier_class(
            **kwargs
        )


# ============================================================
# LLM BATCH RUNNER
# ============================================================

def run_llm_system(
    classifier,
    texts,
    system_name,
    dependency=False
):
    """
    Runs L or D->L in batches.

    Technical failures:
        - never become neutral
        - are recovered individually when possible
        - remain None if recovery also fails
    """

    predictions = [
        None
        for _ in texts
    ]

    review_metadata = [

        {
            "prediction":
                None,

            "batch_number":
                None,

            "recovered":
                False,

            "error":
                None
        }

        for _ in texts
    ]

    batch_results = []

    total_batches = math.ceil(
        len(
            texts
        )
        / LLM_BATCH_SIZE
    )

    system_wall_start = (
        time.perf_counter()
    )

    for batch_number, start in enumerate(

        range(
            0,
            len(
                texts
            ),
            LLM_BATCH_SIZE
        ),

        start=1
    ):

        batch_texts = (
            texts[
                start:
                start
                + LLM_BATCH_SIZE
            ]
        )

        print_subtitle(
            (
                f"{system_name} — "
                f"BATCH {batch_number}/{total_batches}"
            )
        )

        reporter.log(
            "Review indices:",
            (
                f"{start + 1}"
                f"-"
                f"{start + len(batch_texts)}"
            )
        )

        reporter.log(
            "Batch size:",
            len(
                batch_texts
            )
        )

        batch_wall_start = (
            time.perf_counter()
        )

        result = (
            classifier.classify_batch(
                batch_texts
            )
        )

        batch_wall_seconds = (
            time.perf_counter()
            - batch_wall_start
        )

        safe_result = (
            json_safe(
                result
            )
        )

        batch_results.append({

            "batch_number":
                batch_number,

            "start_index":
                start,

            "end_index":
                (
                    start
                    + len(
                        batch_texts
                    )
                    - 1
                ),

            "wall_clock_seconds":
                batch_wall_seconds,

            "result":
                safe_result
        })

        reporter.log(
            "Status:",
            result.get(
                "status"
            )
        )

        reporter.log(
            "Wall clock:",
            round(
                batch_wall_seconds,
                3
            ),
            "seconds"
        )

        reporter.log(
            "LLM request seconds:",
            round(
                float(
                    result.get(
                        "request_seconds",
                        0.0
                    )
                    or 0.0
                ),
                3
            )
        )

        if dependency:

            reporter.log(
                "Dependency parse seconds:",
                round(
                    float(
                        result.get(
                            "dependency_parse_seconds",
                            0.0
                        )
                        or 0.0
                    ),
                    3
                )
            )

            reporter.log(
                "Total pipeline seconds:",
                round(
                    float(
                        result.get(
                            "total_pipeline_seconds",
                            batch_wall_seconds
                        )
                        or batch_wall_seconds
                    ),
                    3
                )
            )

        reporter.log(
            "Prompt tokens:",
            result.get(
                "prompt_tokens"
            )
        )

        reporter.log(
            "Output tokens:",
            result.get(
                "output_tokens"
            )
        )

        reporter.log(
            "Attempts:",
            result.get(
                "attempt"
            )
        )

        reporter.log(
            "Error:",
            result.get(
                "error"
            )
        )

        batch_predictions = (
            result.get(
                "predictions"
            )
            or [
                None
                for _ in batch_texts
            ]
        )

        dependency_serializations = (
            result.get(
                "dependency_serializations"
            )
            or []
        )

        dependency_parses = (
            result.get(
                "dependency_parses"
            )
            or []
        )

        for local_index, prediction in enumerate(
            batch_predictions
        ):

            global_index = (
                start
                + local_index
            )

            predictions[
                global_index
            ] = prediction

            review_metadata[
                global_index
            ][
                "prediction"
            ] = prediction

            review_metadata[
                global_index
            ][
                "batch_number"
            ] = batch_number

            review_metadata[
                global_index
            ][
                "error"
            ] = result.get(
                "error"
            )

            if (
                dependency
                and local_index
                < len(
                    dependency_serializations
                )
            ):

                review_metadata[
                    global_index
                ][
                    "dependency_serialization"
                ] = (
                    dependency_serializations[
                        local_index
                    ]
                )

            if (
                dependency
                and local_index
                < len(
                    dependency_parses
                )
            ):

                review_metadata[
                    global_index
                ][
                    "dependency_parse"
                ] = (
                    json_safe(
                        dependency_parses[
                            local_index
                        ]
                    )
                )

        # ====================================================
        # INDIVIDUAL RECOVERY
        # ====================================================

        failed_local_indices = [

            index

            for index, prediction
            in enumerate(
                batch_predictions
            )

            if prediction is None
        ]

        if failed_local_indices:

            reporter.log()

            reporter.log(
                "[RECOVERY] "
                f"{len(failed_local_indices)} "
                "failed reviews will be retried individually."
            )

        for local_index in failed_local_indices:

            global_index = (
                start
                + local_index
            )

            text = (
                texts[
                    global_index
                ]
            )

            reporter.log(
                (
                    "[RECOVERY] "
                    f"{system_name} "
                    f"review {global_index + 1}"
                )
            )

            recovery_result = (
                classifier.classify(
                    text
                )
            )

            prediction = (
                recovery_result.get(
                    "prediction"
                )
            )

            predictions[
                global_index
            ] = prediction

            review_metadata[
                global_index
            ][
                "prediction"
            ] = prediction

            review_metadata[
                global_index
            ][
                "recovered"
            ] = (
                prediction
                is not None
            )

            review_metadata[
                global_index
            ][
                "recovery_result"
            ] = (
                json_safe(
                    recovery_result
                )
            )

            review_metadata[
                global_index
            ][
                "error"
            ] = (
                recovery_result.get(
                    "error"
                )
            )

            if dependency:

                dependency_serialization = (
                    recovery_result.get(
                        "dependency_serialization"
                    )
                )

                if dependency_serialization:

                    review_metadata[
                        global_index
                    ][
                        "dependency_serialization"
                    ] = (
                        dependency_serialization
                    )

                dependency_parse = (
                    recovery_result.get(
                        "dependency_parse"
                    )
                )

                if dependency_parse:

                    review_metadata[
                        global_index
                    ][
                        "dependency_parse"
                    ] = (
                        json_safe(
                            dependency_parse
                        )
                    )

    total_wall_seconds = (
        time.perf_counter()
        - system_wall_start
    )

    return {

        "predictions":
            predictions,

        "review_metadata":
            review_metadata,

        "batches":
            batch_results,

        "wall_seconds":
            total_wall_seconds,

        "classifier_stats":
            json_safe(
                classifier.get_stats()
            )
    }


# ============================================================
# START REPORT
# ============================================================

try:

    print_title(
        "SIX-SYSTEM TURKISH SENTIMENT SANITY TEST"
    )

    reporter.log(
        "Number of reviews:",
        len(
            TEST_CASES
        )
    )

    reporter.log(
        "Systems:",
        ", ".join(
            SYSTEMS
        )
    )

    reporter.log(
        "Ollama model:",
        MODEL
    )

    reporter.log(
        "LLM batch size:",
        LLM_BATCH_SIZE
    )

    reporter.log(
        "Temperature:",
        TEMPERATURE
    )

    reporter.log(
        "Top-p:",
        TOP_P
    )

    reporter.log(
        "Context:",
        NUM_CTX
    )

    reporter.log(
        "Seed:",
        SEED
    )


    # ========================================================
    # GOLD DISTRIBUTION
    # ========================================================

    gold_labels = [

        case[
            "gold"
        ]

        for case
        in TEST_CASES
    ]

    texts = [

        case[
            "text"
        ]

        for case
        in TEST_CASES
    ]

    gold_distribution = Counter(
        gold_labels
    )

    print_subtitle(
        "GOLD LABEL DISTRIBUTION"
    )

    for label in LABELS:

        reporter.log(
            label,
            ":",
            gold_distribution.get(
                label,
                0
            )
        )

    if (
        gold_distribution.get(
            "neutral",
            0
        )
        == 0
    ):

        reporter.log()

        reporter.log(
            "[IMPORTANT] "
            "This 16-example sanity set contains no neutral "
            "gold examples."
        )

        reporter.log(
            "Therefore macro-F1 over the classes PRESENT in "
            "the gold set is more informative than fixed "
            "3-class macro-F1 for this test."
        )


    # ========================================================
    # INITIALIZE CLASSICAL SYSTEMS
    # ========================================================

    print_title(
        "INITIALIZING CLASSICAL / LEXICON SYSTEMS"
    )

    initialization_times = {}


    start_time = (
        time.perf_counter()
    )

    hisnet = (
        HisNetClassifier()
    )

    initialization_times[
        "H"
    ] = (
        time.perf_counter()
        - start_time
    )


    start_time = (
        time.perf_counter()
    )

    hisnet_dependency = (
        HisNetDependencyClassifier(
            hisnet_classifier=
                hisnet
        )
    )

    initialization_times[
        "H->D"
    ] = (
        time.perf_counter()
        - start_time
    )


    start_time = (
        time.perf_counter()
    )

    dependency_hisnet = (
        DependencyHisNetClassifier()
    )

    initialization_times[
        "D->H"
    ] = (
        time.perf_counter()
        - start_time
    )


    start_time = (
        time.perf_counter()
    )

    literature_weighted = (
        LiteratureWeightedHisNetDependencyClassifier(
            hisnet_classifier=
                hisnet
        )
    )

    initialization_times[
        "H->D-LW"
    ] = (
        time.perf_counter()
        - start_time
    )


    for system_name, seconds in (
        initialization_times.items()
    ):

        reporter.log(
            system_name,
            "initialization:",
            round(
                seconds,
                3
            ),
            "seconds"
        )


    # ========================================================
    # RUN CLASSICAL SYSTEMS
    # ========================================================

    classical_results = {

        "H":
            [],

        "H->D":
            [],

        "D->H":
            [],

        "H->D-LW":
            []
    }


    classical_runtime = {

        "H":
            0.0,

        "H->D":
            0.0,

        "D->H":
            0.0,

        "H->D-LW":
            0.0
    }


    print_title(
        "RUNNING H / H->D / D->H / H->D-LW"
    )


    for index, case in enumerate(
        TEST_CASES,
        start=1
    ):

        text = case[
            "text"
        ]

        reporter.log()

        separator(
            "="
        )

        reporter.log(
            (
                f"REVIEW {index}/"
                f"{len(TEST_CASES)}"
            )
        )

        reporter.log(
            "Text:",
            text
        )

        reporter.log(
            "Gold:",
            case[
                "gold"
            ]
        )

        reporter.log(
            "Focus:",
            case[
                "focus"
            ]
        )

        # ----------------------------------------------------
        # H
        # ----------------------------------------------------

        start_time = (
            time.perf_counter()
        )

        h_result = (
            hisnet.classify(
                text
            )
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        classical_runtime[
            "H"
        ] += elapsed

        h_result[
            "_test_runtime_seconds"
        ] = elapsed

        classical_results[
            "H"
        ].append(
            h_result
        )


        # ----------------------------------------------------
        # H -> D
        #
        # IMPORTANT:
        # Reuse exact H result.
        # ----------------------------------------------------

        start_time = (
            time.perf_counter()
        )

        hd_result = (
            hisnet_dependency.classify(

                text,

                hisnet_result=
                    h_result
            )
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        classical_runtime[
            "H->D"
        ] += elapsed

        hd_result[
            "_test_runtime_seconds"
        ] = elapsed

        classical_results[
            "H->D"
        ].append(
            hd_result
        )


        # ----------------------------------------------------
        # D -> H
        # ----------------------------------------------------

        start_time = (
            time.perf_counter()
        )

        dh_result = (
            dependency_hisnet.classify(
                text
            )
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        classical_runtime[
            "D->H"
        ] += elapsed

        dh_result[
            "_test_runtime_seconds"
        ] = elapsed

        classical_results[
            "D->H"
        ].append(
            dh_result
        )


        # ----------------------------------------------------
        # H -> D -> Literature Weighted
        #
        # Reuse exact H baseline again.
        # ----------------------------------------------------

        start_time = (
            time.perf_counter()
        )

        lw_result = (
            literature_weighted.classify(

                text,

                hisnet_result=
                    h_result
            )
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        classical_runtime[
            "H->D-LW"
        ] += elapsed

        lw_result[
            "_test_runtime_seconds"
        ] = elapsed

        classical_results[
            "H->D-LW"
        ].append(
            lw_result
        )


        # ====================================================
        # BASIC STRUCTURAL VALIDATIONS
        # ====================================================

        assert (
            hd_result[
                "hisnet_prediction"
            ]
            == h_result[
                "prediction"
            ]
        )

        assert math.isclose(

            float(
                hd_result[
                    "hisnet_positive_score"
                ]
            ),

            float(
                h_result[
                    "positive_score"
                ]
            ),

            rel_tol=
                1e-9,

            abs_tol=
                1e-9
        )

        assert math.isclose(

            float(
                hd_result[
                    "hisnet_negative_score"
                ]
            ),

            float(
                h_result[
                    "negative_score"
                ]
            ),

            rel_tol=
                1e-9,

            abs_tol=
                1e-9
        )

        assert (
            lw_result[
                "dependency_prediction"
            ]
            == hd_result[
                "prediction"
            ]
        )

        assert math.isclose(

            float(
                lw_result[
                    "dependency_positive_score"
                ]
            ),

            float(
                hd_result[
                    "positive_score"
                ]
            ),

            rel_tol=
                1e-9,

            abs_tol=
                1e-9
        )

        assert math.isclose(

            float(
                lw_result[
                    "dependency_negative_score"
                ]
            ),

            float(
                hd_result[
                    "negative_score"
                ]
            ),

            rel_tol=
                1e-9,

            abs_tol=
                1e-9
        )


        reporter.log()

        reporter.log(
            "Predictions:"
        )

        reporter.log(
            "  H       :",
            h_result[
                "prediction"
            ]
        )

        reporter.log(
            "  H -> D  :",
            hd_result[
                "prediction"
            ]
        )

        reporter.log(
            "  D -> H  :",
            dh_result[
                "prediction"
            ]
        )

        reporter.log(
            "  H->D-LW :",
            lw_result[
                "prediction"
            ]
        )


    # ========================================================
    # INITIALIZE LLM SYSTEMS
    # ========================================================

    print_title(
        "INITIALIZING LLM SYSTEMS"
    )


    start_time = (
        time.perf_counter()
    )

    llm = (
        create_llm_classifier(
            OllamaZeroShotSentimentClassifier,
            dependency=False
        )
    )

    initialization_times[
        "L"
    ] = (
        time.perf_counter()
        - start_time
    )


    start_time = (
        time.perf_counter()
    )

    dependency_llm = (
        create_llm_classifier(
            DependencyOllamaZeroShotSentimentClassifier,
            dependency=True
        )
    )

    initialization_times[
        "D->L"
    ] = (
        time.perf_counter()
        - start_time
    )


    reporter.log(
        "L initialization:",
        round(
            initialization_times[
                "L"
            ],
            3
        ),
        "seconds"
    )

    reporter.log(
        "D->L initialization:",
        round(
            initialization_times[
                "D->L"
            ],
            3
        ),
        "seconds"
    )


    # ========================================================
    # RUN LLM
    # ========================================================

    print_title(
        "RUNNING LLM ONLY"
    )

    llm_run = (
        run_llm_system(
            llm,
            texts,
            system_name=
                "L",
            dependency=
                False
        )
    )


    # ========================================================
    # RUN DEPENDENCY -> LLM
    # ========================================================

    print_title(
        "RUNNING DEPENDENCY -> LLM"
    )

    dependency_llm_run = (
        run_llm_system(
            dependency_llm,
            texts,
            system_name=
                "D->L",
            dependency=
                True
        )
    )


    # ========================================================
    # COLLECT PREDICTIONS
    # ========================================================

    predictions_by_system = {

        "H":
            [
                result[
                    "prediction"
                ]
                for result
                in classical_results[
                    "H"
                ]
            ],

        "H->D":
            [
                result[
                    "prediction"
                ]
                for result
                in classical_results[
                    "H->D"
                ]
            ],

        "D->H":
            [
                result[
                    "prediction"
                ]
                for result
                in classical_results[
                    "D->H"
                ]
            ],

        "L":
            llm_run[
                "predictions"
            ],

        "D->L":
            dependency_llm_run[
                "predictions"
            ],

        "H->D-LW":
            [
                result[
                    "prediction"
                ]
                for result
                in classical_results[
                    "H->D-LW"
                ]
            ]
    }


    # ========================================================
    # PER-REVIEW MASTER TABLE
    # ========================================================

    prediction_rows = []


    for index, case in enumerate(
        TEST_CASES
    ):

        row = {

            "id":
                case[
                    "id"
                ],

            "text":
                case[
                    "text"
                ],

            "focus":
                case[
                    "focus"
                ],

            "gold":
                case[
                    "gold"
                ]
        }

        correct_count = 0

        wrong_systems = []

        for system_name in SYSTEMS:

            prediction = (
                predictions_by_system[
                    system_name
                ][
                    index
                ]
            )

            row[
                system_name
            ] = prediction

            is_correct = (
                prediction
                == case[
                    "gold"
                ]
            )

            row[
                f"{system_name}_correct"
            ] = is_correct

            if is_correct:

                correct_count += 1

            else:

                wrong_systems.append(
                    system_name
                )

        row[
            "correct_system_count"
        ] = correct_count

        row[
            "wrong_systems"
        ] = ", ".join(
            wrong_systems
        )

        prediction_rows.append(
            row
        )


    prediction_df = (
        pd.DataFrame(
            prediction_rows
        )
    )


    # ========================================================
    # SAVE MASTER PREDICTION TABLE
    # ========================================================

    prediction_csv_path = (
        OUTPUT_DIR
        / "predictions_16.csv"
    )

    prediction_df.to_csv(
        prediction_csv_path,
        index=False,
        encoding="utf-8-sig"
    )


    # ========================================================
    # SCORE DELTA TABLE FOR LEXICON SYSTEMS
    # ========================================================

    score_rows = []


    for index, case in enumerate(
        TEST_CASES
    ):

        h = (
            classical_results[
                "H"
            ][
                index
            ]
        )

        hd = (
            classical_results[
                "H->D"
            ][
                index
            ]
        )

        dh = (
            classical_results[
                "D->H"
            ][
                index
            ]
        )

        lw = (
            classical_results[
                "H->D-LW"
            ][
                index
            ]
        )

        score_rows.append({

            "id":
                case[
                    "id"
                ],

            "text":
                case[
                    "text"
                ],

            "gold":
                case[
                    "gold"
                ],

            "H_prediction":
                h[
                    "prediction"
                ],

            "H_pos":
                h[
                    "positive_score"
                ],

            "H_neg":
                h[
                    "negative_score"
                ],

            "H_net":
                (
                    h[
                        "positive_score"
                    ]
                    - h[
                        "negative_score"
                    ]
                ),

            "HD_prediction":
                hd[
                    "prediction"
                ],

            "HD_pos":
                hd[
                    "positive_score"
                ],

            "HD_neg":
                hd[
                    "negative_score"
                ],

            "HD_net":
                (
                    hd[
                        "positive_score"
                    ]
                    - hd[
                        "negative_score"
                    ]
                ),

            "DH_prediction":
                dh[
                    "prediction"
                ],

            "DH_pos":
                dh[
                    "positive_score"
                ],

            "DH_neg":
                dh[
                    "negative_score"
                ],

            "DH_net":
                dh[
                    "net_score"
                ],

            "LW_prediction":
                lw[
                    "prediction"
                ],

            "LW_pos":
                lw[
                    "positive_score"
                ],

            "LW_neg":
                lw[
                    "negative_score"
                ],

            "LW_net":
                (
                    lw[
                        "positive_score"
                    ]
                    - lw[
                        "negative_score"
                    ]
                ),

            "H_to_HD_changed":
                (
                    h[
                        "prediction"
                    ]
                    != hd[
                        "prediction"
                    ]
                ),

            "H_to_DH_changed":
                (
                    h[
                        "prediction"
                    ]
                    != dh[
                        "prediction"
                    ]
                ),

            "HD_to_LW_changed":
                (
                    hd[
                        "prediction"
                    ]
                    != lw[
                        "prediction"
                    ]
                ),

            "LW_modifier_weighted_scopes":
                lw.get(
                    "modifier_weighted_scope_count",
                    0
                ),

            "LW_modifier_components_suppressed":
                lw.get(
                    "modifier_component_suppressed_scope_count",
                    0
                ),

            "LW_contrast_markers_suppressed":
                lw.get(
                    "contrast_marker_suppressed_scope_count",
                    0
                ),

            "LW_contrast_weighted_scopes":
                lw.get(
                    "contrast_weighted_scope_count",
                    0
                )
        })


    score_df = (
        pd.DataFrame(
            score_rows
        )
    )


    score_csv_path = (
        OUTPUT_DIR
        / "lexicon_score_deltas_16.csv"
    )

    score_df.to_csv(
        score_csv_path,
        index=False,
        encoding="utf-8-sig"
    )


    # ========================================================
    # METRICS
    # ========================================================

    print_title(
        "FINAL METRICS"
    )


    metric_rows = []


    for system_name in SYSTEMS:

        metrics = (
            calculate_metrics(

                gold_labels,

                predictions_by_system[
                    system_name
                ],

                system_name
            )
        )

        metric_rows.append(
            metrics
        )


    metrics_df = (
        pd.DataFrame(
            metric_rows
        )
    )


    metrics_csv_path = (
        OUTPUT_DIR
        / "metrics_16.csv"
    )


    metrics_df.to_csv(
        metrics_csv_path,
        index=False,
        encoding="utf-8-sig"
    )


    reporter.log(
        metrics_df.to_string(
            index=False
        )
    )


    # ========================================================
    # CLASSIFICATION REPORT + CONFUSION MATRIX
    # ========================================================

    confusion_matrices = {}


    for system_name in SYSTEMS:

        print_subtitle(
            (
                f"{system_name} — "
                "CLASSIFICATION REPORT"
            )
        )

        predictions = (
            predictions_by_system[
                system_name
            ]
        )

        if any(
            prediction is None
            for prediction in predictions
        ):

            reporter.log(
                "Metrics not computed because unresolved "
                "predictions remain."
            )

            continue


        report_text = (
            classification_report(

                gold_labels,
                predictions,

                labels=
                    LABELS,

                digits=
                    4,

                zero_division=
                    0
            )
        )

        reporter.log(
            report_text
        )


        matrix = (
            confusion_matrix(

                gold_labels,
                predictions,

                labels=
                    LABELS
            )
        )


        matrix_df = pd.DataFrame(

            matrix,

            index=[
                f"GOLD_{label}"
                for label in LABELS
            ],

            columns=[
                f"PRED_{label}"
                for label in LABELS
            ]
        )


        confusion_matrices[
            system_name
        ] = (
            matrix.tolist()
        )


        reporter.log(
            "Confusion matrix:"
        )

        reporter.log(
            matrix_df.to_string()
        )


    # ========================================================
    # PAIRWISE SYSTEM COMPARISONS
    # ========================================================

    print_title(
        "ALL PAIRWISE SYSTEM COMPARISONS"
    )


    pairwise_rows = []


    for system_a, system_b in combinations(
        SYSTEMS,
        2
    ):

        predictions_a = (
            predictions_by_system[
                system_a
            ]
        )

        predictions_b = (
            predictions_by_system[
                system_b
            ]
        )


        # Skip strict statistics if unresolved predictions exist.
        valid = all(

            prediction_a is not None
            and prediction_b is not None

            for prediction_a, prediction_b
            in zip(
                predictions_a,
                predictions_b
            )
        )


        if not valid:

            pairwise_rows.append({

                "system_a":
                    system_a,

                "system_b":
                    system_b,

                "n":
                    len(
                        TEST_CASES
                    ),

                "agreement_count":
                    None,

                "agreement_rate":
                    None,

                "changed_count":
                    None,

                "both_correct":
                    None,

                "a_only_correct":
                    None,

                "b_only_correct":
                    None,

                "both_wrong":
                    None,

                "net_b_minus_a":
                    None,

                "exact_p":
                    None
            })

            continue


        agreement_count = sum(

            prediction_a
            == prediction_b

            for prediction_a, prediction_b
            in zip(
                predictions_a,
                predictions_b
            )
        )


        changed_count = (
            len(
                TEST_CASES
            )
            - agreement_count
        )


        both_correct = 0

        a_only_correct = 0

        b_only_correct = 0

        both_wrong = 0


        for gold, prediction_a, prediction_b in zip(

            gold_labels,
            predictions_a,
            predictions_b
        ):

            a_correct = (
                prediction_a
                == gold
            )

            b_correct = (
                prediction_b
                == gold
            )


            if (
                a_correct
                and b_correct
            ):

                both_correct += 1

            elif (
                a_correct
                and not b_correct
            ):

                a_only_correct += 1

            elif (
                not a_correct
                and b_correct
            ):

                b_only_correct += 1

            else:

                both_wrong += 1


        exact_p = (
            exact_paired_p_value(
                a_only_correct,
                b_only_correct
            )
        )


        pairwise_rows.append({

            "system_a":
                system_a,

            "system_b":
                system_b,

            "n":
                len(
                    TEST_CASES
                ),

            "agreement_count":
                agreement_count,

            "agreement_rate":
                (
                    agreement_count
                    / len(
                        TEST_CASES
                    )
                ),

            "changed_count":
                changed_count,

            "both_correct":
                both_correct,

            "a_only_correct":
                a_only_correct,

            "b_only_correct":
                b_only_correct,

            "both_wrong":
                both_wrong,

            "net_b_minus_a":
                (
                    b_only_correct
                    - a_only_correct
                ),

            "exact_p":
                exact_p
        })


    pairwise_df = (
        pd.DataFrame(
            pairwise_rows
        )
    )


    pairwise_csv_path = (
        OUTPUT_DIR
        / "pairwise_comparisons_16.csv"
    )


    pairwise_df.to_csv(
        pairwise_csv_path,
        index=False,
        encoding="utf-8-sig"
    )


    reporter.log(
        pairwise_df.to_string(
            index=False
        )
    )


    # ========================================================
    # FOCUSED ABLATION COMPARISONS
    # ========================================================

    print_title(
        "FOCUSED ABLATION COMPARISONS"
    )


    focused_pairs = [

        (
            "H",
            "H->D",
            "Effect of dependency AFTER HisNet"
        ),

        (
            "H",
            "D->H",
            "Effect of dependency BEFORE HisNet"
        ),

        (
            "H->D",
            "H->D-LW",
            "Effect of literature weighting"
        ),

        (
            "L",
            "D->L",
            "Effect of dependency input on LLM"
        ),

        (
            "H->D",
            "D->H",
            "Dependency ordering"
        )
    ]


    for (
        system_a,
        system_b,
        question
    ) in focused_pairs:

        row = pairwise_df[
            (
                pairwise_df[
                    "system_a"
                ]
                == system_a
            )
            &
            (
                pairwise_df[
                    "system_b"
                ]
                == system_b
            )
        ]

        # combinations() ordering may be reversed.
        if row.empty:

            row = pairwise_df[
                (
                    pairwise_df[
                        "system_a"
                    ]
                    == system_b
                )
                &
                (
                    pairwise_df[
                        "system_b"
                    ]
                    == system_a
                )
            ]

        reporter.log()

        reporter.log(
            question
        )

        reporter.log(
            f"{system_a} vs {system_b}"
        )

        if row.empty:

            reporter.log(
                "Pairwise record not found."
            )

        else:

            reporter.log(
                row.to_string(
                    index=False
                )
            )


    # ========================================================
    # PER-EXAMPLE COMPARISON TABLE
    # ========================================================

    print_title(
        "PER-EXAMPLE SIX-SYSTEM PREDICTIONS"
    )


    display_columns = [

        "id",
        "gold",
        "H",
        "H->D",
        "D->H",
        "L",
        "D->L",
        "H->D-LW",
        "correct_system_count",
        "wrong_systems",
        "text"
    ]


    reporter.log(
        prediction_df[
            display_columns
        ].to_string(
            index=False
        )
    )


    # ========================================================
    # CASES WHERE SYSTEMS DISAGREE
    # ========================================================

    print_title(
        "DISAGREEMENT / ERROR ANALYSIS"
    )


    for index, case in enumerate(
        TEST_CASES
    ):

        predictions = {

            system_name:
                predictions_by_system[
                    system_name
                ][
                    index
                ]

            for system_name
            in SYSTEMS
        }


        unique_predictions = set(
            predictions.values()
        )


        wrong_systems = [

            system_name

            for system_name, prediction
            in predictions.items()

            if prediction
            != case[
                "gold"
            ]
        ]


        if (
            len(
                unique_predictions
            )
            == 1
            and not wrong_systems
        ):

            continue


        separator(
            "-"
        )

        reporter.log(
            "Review:",
            case[
                "id"
            ]
        )

        reporter.log(
            "Text:",
            case[
                "text"
            ]
        )

        reporter.log(
            "Focus:",
            case[
                "focus"
            ]
        )

        reporter.log(
            "Gold:",
            case[
                "gold"
            ]
        )


        for system_name in SYSTEMS:

            prediction = (
                predictions[
                    system_name
                ]
            )

            marker = (
                "✓"
                if prediction
                == case[
                    "gold"
                ]
                else "✗"
            )

            reporter.log(
                (
                    f"{system_name:8s} "
                    f"{prediction!s:10s} "
                    f"{marker}"
                )
            )


    # ========================================================
    # DETAILED INTERNAL OUTPUT FOR EACH REVIEW
    # ========================================================

    print_title(
        "FULL INTERNAL DIAGNOSTICS — ALL 16 REVIEWS"
    )


    for index, case in enumerate(
        TEST_CASES
    ):

        separator(
            "="
        )

        reporter.log(
            (
                f"REVIEW {case['id']}: "
                f"{case['text']}"
            )
        )

        reporter.log(
            "Gold:",
            case[
                "gold"
            ]
        )

        reporter.log(
            "Focus:",
            case[
                "focus"
            ]
        )


        # ====================================================
        # H
        # ====================================================

        h = (
            classical_results[
                "H"
            ][
                index
            ]
        )


        print_subtitle(
            "SYSTEM 1 — HISNET ONLY (H)"
        )

        reporter.log(
            "Prediction:",
            h[
                "prediction"
            ]
        )

        reporter.log(
            "Positive score:",
            h[
                "positive_score"
            ]
        )

        reporter.log(
            "Negative score:",
            h[
                "negative_score"
            ]
        )

        reporter.log(
            "Net score:",
            h[
                "net_score"
            ]
        )

        reporter.log(
            "Lexicon coverage:",
            h[
                "lexicon_coverage"
            ]
        )

        reporter.log(
            "Sentiment coverage:",
            h[
                "sentiment_coverage"
            ]
        )

        reporter.log(
            "Runtime:",
            h[
                "_test_runtime_seconds"
            ]
        )

        reporter.log(
            "\nMorphological analyses:"
        )

        reporter.log(
            pretty_json(
                h[
                    "morphological_analyses"
                ]
            )
        )

        reporter.log(
            "\nSentiment words:"
        )

        reporter.log(
            pretty_json(
                h[
                    "sentiment_words"
                ]
            )
        )

        reporter.log(
            "\nUnmatched words:"
        )

        reporter.log(
            pretty_json(
                h[
                    "unmatched_words"
                ]
            )
        )


        # ====================================================
        # H -> D
        # ====================================================

        hd = (
            classical_results[
                "H->D"
            ][
                index
            ]
        )


        print_subtitle(
            "SYSTEM 2 — HISNET -> DEPENDENCY (H->D)"
        )

        reporter.log(
            "Prediction:",
            hd[
                "prediction"
            ]
        )

        reporter.log(
            "HisNet baseline prediction:",
            hd[
                "hisnet_prediction"
            ]
        )

        reporter.log(
            "HisNet POS:",
            hd[
                "hisnet_positive_score"
            ]
        )

        reporter.log(
            "HisNet NEG:",
            hd[
                "hisnet_negative_score"
            ]
        )

        reporter.log(
            "Dependency POS:",
            hd[
                "positive_score"
            ]
        )

        reporter.log(
            "Dependency NEG:",
            hd[
                "negative_score"
            ]
        )

        reporter.log(
            "Baseline reused:",
            hd[
                "baseline_reused"
            ]
        )

        reporter.log(
            "Runtime:",
            hd[
                "_test_runtime_seconds"
            ]
        )

        reporter.log(
            "\nResolved scopes:"
        )

        reporter.log(
            pretty_json(
                hd[
                    "scopes"
                ]
            )
        )

        reporter.log(
            "\nSentence/dependency diagnostics:"
        )

        reporter.log(
            pretty_json(
                hd[
                    "sentences"
                ]
            )
        )

        reporter.log(
            "\nUnmatched scopes:"
        )

        reporter.log(
            pretty_json(
                hd[
                    "unmatched_scopes"
                ]
            )
        )


        # ====================================================
        # D -> H
        # ====================================================

        dh = (
            classical_results[
                "D->H"
            ][
                index
            ]
        )


        print_subtitle(
            "SYSTEM 3 — DEPENDENCY -> HISNET (D->H)"
        )

        reporter.log(
            "Prediction:",
            dh[
                "prediction"
            ]
        )

        reporter.log(
            "Positive score:",
            dh[
                "positive_score"
            ]
        )

        reporter.log(
            "Negative score:",
            dh[
                "negative_score"
            ]
        )

        reporter.log(
            "Net score:",
            dh[
                "net_score"
            ]
        )

        reporter.log(
            "Token count:",
            dh[
                "token_count"
            ]
        )

        reporter.log(
            "Dependency candidate count:",
            dh[
                "dependency_candidate_count"
            ]
        )

        reporter.log(
            "Lexicon matched candidates:",
            dh[
                "lexicon_matched_candidate_count"
            ]
        )

        reporter.log(
            "Sentiment word count:",
            dh[
                "sentiment_word_count"
            ]
        )

        reporter.log(
            "Dependency candidate coverage:",
            dh[
                "dependency_candidate_coverage"
            ]
        )

        reporter.log(
            "Candidate lexicon coverage:",
            dh[
                "candidate_lexicon_coverage"
            ]
        )

        reporter.log(
            "Sentiment coverage:",
            dh[
                "sentiment_coverage"
            ]
        )

        reporter.log(
            "Operations:",
            dh[
                "operations"
            ]
        )

        reporter.log(
            "Runtime:",
            dh[
                "_test_runtime_seconds"
            ]
        )

        reporter.log(
            "\nDependency parse:"
        )

        reporter.log(
            pretty_json(
                dh[
                    "parsed_sentences"
                ]
            )
        )

        reporter.log(
            "\nDependency candidates BEFORE HisNet:"
        )

        reporter.log(
            pretty_json(
                dh[
                    "dependency_candidates"
                ]
            )
        )

        reporter.log(
            "\nCandidate HisNet diagnostics:"
        )

        reporter.log(
            pretty_json(
                dh[
                    "candidate_diagnostics"
                ]
            )
        )

        reporter.log(
            "\nFinal sentiment scopes:"
        )

        reporter.log(
            pretty_json(
                dh[
                    "scopes"
                ]
            )
        )


        # ====================================================
        # L
        # ====================================================

        print_subtitle(
            "SYSTEM 4 — LLM ONLY (L)"
        )


        llm_metadata = (
            llm_run[
                "review_metadata"
            ][
                index
            ]
        )


        reporter.log(
            "Prediction:",
            predictions_by_system[
                "L"
            ][
                index
            ]
        )

        reporter.log(
            "Batch:",
            llm_metadata.get(
                "batch_number"
            )
        )

        reporter.log(
            "Recovered:",
            llm_metadata.get(
                "recovered"
            )
        )

        reporter.log(
            "Error:",
            llm_metadata.get(
                "error"
            )
        )


        # ====================================================
        # D -> L
        # ====================================================

        print_subtitle(
            "SYSTEM 5 — DEPENDENCY -> LLM (D->L)"
        )


        dependency_llm_metadata = (
            dependency_llm_run[
                "review_metadata"
            ][
                index
            ]
        )


        reporter.log(
            "Prediction:",
            predictions_by_system[
                "D->L"
            ][
                index
            ]
        )

        reporter.log(
            "Batch:",
            dependency_llm_metadata.get(
                "batch_number"
            )
        )

        reporter.log(
            "Recovered:",
            dependency_llm_metadata.get(
                "recovered"
            )
        )

        reporter.log(
            "Error:",
            dependency_llm_metadata.get(
                "error"
            )
        )

        reporter.log(
            "\nDependency representation sent to LLM:"
        )

        reporter.log(
            dependency_llm_metadata.get(
                "dependency_serialization",
                "N/A"
            )
        )

        reporter.log(
            "\nFull dependency parse:"
        )

        reporter.log(
            pretty_json(
                dependency_llm_metadata.get(
                    "dependency_parse",
                    []
                )
            )
        )


        # ====================================================
        # H -> D -> LW
        # ====================================================

        lw = (
            classical_results[
                "H->D-LW"
            ][
                index
            ]
        )


        print_subtitle(
            "SYSTEM 6 — H->D LITERATURE WEIGHTED"
        )

        reporter.log(
            "Dependency baseline prediction:",
            lw[
                "dependency_prediction"
            ]
        )

        reporter.log(
            "Final prediction:",
            lw[
                "prediction"
            ]
        )

        reporter.log(
            "Dependency POS:",
            lw[
                "dependency_positive_score"
            ]
        )

        reporter.log(
            "Dependency NEG:",
            lw[
                "dependency_negative_score"
            ]
        )

        reporter.log(
            "Weighted POS:",
            lw[
                "positive_score"
            ]
        )

        reporter.log(
            "Weighted NEG:",
            lw[
                "negative_score"
            ]
        )

        reporter.log(
            "Modifier weighted scopes:",
            lw.get(
                "modifier_weighted_scope_count"
            )
        )

        reporter.log(
            "Modifier components suppressed:",
            lw.get(
                "modifier_component_suppressed_scope_count"
            )
        )

        reporter.log(
            "Contrast markers suppressed:",
            lw.get(
                "contrast_marker_suppressed_scope_count"
            )
        )

        reporter.log(
            "Contrast weighted scopes:",
            lw.get(
                "contrast_weighted_scope_count"
            )
        )

        reporter.log(
            "Runtime:",
            lw[
                "_test_runtime_seconds"
            ]
        )

        reporter.log(
            "\nWeighted scopes:"
        )

        reporter.log(
            pretty_json(
                lw[
                    "scopes"
                ]
            )
        )

        reporter.log(
            "\nLicensed contrast markers:"
        )

        reporter.log(
            pretty_json(
                lw.get(
                    "licensed_contrast_markers"
                )
            )
        )

        reporter.log(
            "\nContrast relations:"
        )

        reporter.log(
            pretty_json(
                lw.get(
                    "contrast_relations"
                )
            )
        )

        reporter.log(
            "\nLiterature weighting config:"
        )

        reporter.log(
            pretty_json(
                lw.get(
                    "literature_weighting"
                )
            )
        )


    # ========================================================
    # LLM BATCH-LEVEL DETAILS
    # ========================================================

    print_title(
        "LLM BATCH-LEVEL DIAGNOSTICS"
    )


    print_subtitle(
        "L — BATCH DETAILS"
    )


    for batch in llm_run[
        "batches"
    ]:

        result = batch[
            "result"
        ]

        reporter.log(
            pretty_json({

                "batch_number":
                    batch[
                        "batch_number"
                    ],

                "start_index":
                    batch[
                        "start_index"
                    ],

                "end_index":
                    batch[
                        "end_index"
                    ],

                "wall_clock_seconds":
                    batch[
                        "wall_clock_seconds"
                    ],

                "status":
                    result.get(
                        "status"
                    ),

                "request_seconds":
                    result.get(
                        "request_seconds"
                    ),

                "prompt_tokens":
                    result.get(
                        "prompt_tokens"
                    ),

                "output_tokens":
                    result.get(
                        "output_tokens"
                    ),

                "attempt":
                    result.get(
                        "attempt"
                    ),

                "error":
                    result.get(
                        "error"
                    ),

                "raw_response":
                    result.get(
                        "raw_response"
                    )
            })
        )


    print_subtitle(
        "D->L — BATCH DETAILS"
    )


    for batch in dependency_llm_run[
        "batches"
    ]:

        result = batch[
            "result"
        ]

        reporter.log(
            pretty_json({

                "batch_number":
                    batch[
                        "batch_number"
                    ],

                "start_index":
                    batch[
                        "start_index"
                    ],

                "end_index":
                    batch[
                        "end_index"
                    ],

                "wall_clock_seconds":
                    batch[
                        "wall_clock_seconds"
                    ],

                "status":
                    result.get(
                        "status"
                    ),

                "dependency_parse_seconds":
                    result.get(
                        "dependency_parse_seconds"
                    ),

                "request_seconds":
                    result.get(
                        "request_seconds"
                    ),

                "total_pipeline_seconds":
                    result.get(
                        "total_pipeline_seconds"
                    ),

                "prompt_tokens":
                    result.get(
                        "prompt_tokens"
                    ),

                "output_tokens":
                    result.get(
                        "output_tokens"
                    ),

                "attempt":
                    result.get(
                        "attempt"
                    ),

                "error":
                    result.get(
                        "error"
                    ),

                "raw_response":
                    result.get(
                        "raw_response"
                    )
            })
        )


    # ========================================================
    # RUNTIME SUMMARY
    # ========================================================

    print_title(
        "RUNTIME SUMMARY"
    )


    runtime_rows = [

        {
            "system":
                "H",

            "wall_seconds":
                classical_runtime[
                    "H"
                ],

            "seconds_per_review":
                (
                    classical_runtime[
                        "H"
                    ]
                    / len(
                        TEST_CASES
                    )
                )
        },

        {
            "system":
                "H->D",

            "wall_seconds":
                classical_runtime[
                    "H->D"
                ],

            "seconds_per_review":
                (
                    classical_runtime[
                        "H->D"
                    ]
                    / len(
                        TEST_CASES
                    )
                )
        },

        {
            "system":
                "D->H",

            "wall_seconds":
                classical_runtime[
                    "D->H"
                ],

            "seconds_per_review":
                (
                    classical_runtime[
                        "D->H"
                    ]
                    / len(
                        TEST_CASES
                    )
                )
        },

        {
            "system":
                "L",

            "wall_seconds":
                llm_run[
                    "wall_seconds"
                ],

            "seconds_per_review":
                (
                    llm_run[
                        "wall_seconds"
                    ]
                    / len(
                        TEST_CASES
                    )
                )
        },

        {
            "system":
                "D->L",

            "wall_seconds":
                dependency_llm_run[
                    "wall_seconds"
                ],

            "seconds_per_review":
                (
                    dependency_llm_run[
                        "wall_seconds"
                    ]
                    / len(
                        TEST_CASES
                    )
                )
        },

        {
            "system":
                "H->D-LW",

            "wall_seconds":
                classical_runtime[
                    "H->D-LW"
                ],

            "seconds_per_review":
                (
                    classical_runtime[
                        "H->D-LW"
                    ]
                    / len(
                        TEST_CASES
                    )
                )
        }
    ]


    runtime_df = (
        pd.DataFrame(
            runtime_rows
        )
    )


    runtime_csv_path = (
        OUTPUT_DIR
        / "runtime_16.csv"
    )


    runtime_df.to_csv(
        runtime_csv_path,
        index=False,
        encoding="utf-8-sig"
    )


    reporter.log(
        runtime_df.to_string(
            index=False
        )
    )


    reporter.log()

    reporter.log(
        "NOTE: Classical runtime values are diagnostic only."
    )

    reporter.log(
        "Caches and model warm-up mean this 16-example run "
        "should not be treated as a rigorous runtime benchmark."
    )


    # ========================================================
    # LLM STATS
    # ========================================================

    print_title(
        "LLM CLASSIFIER STATISTICS"
    )


    reporter.log(
        "LLM only:"
    )

    reporter.log(
        pretty_json(
            llm_run[
                "classifier_stats"
            ]
        )
    )


    reporter.log()

    reporter.log(
        "Dependency -> LLM:"
    )

    reporter.log(
        pretty_json(
            dependency_llm_run[
                "classifier_stats"
            ]
        )
    )


    # ========================================================
    # FULL JSON RESULT
    # ========================================================

    detailed_output = {

        "configuration": {

            "model":
                MODEL,

            "llm_batch_size":
                LLM_BATCH_SIZE,

            "temperature":
                TEMPERATURE,

            "top_p":
                TOP_P,

            "num_ctx":
                NUM_CTX,

            "seed":
                SEED,

            "neutral_threshold":
                0.05,

            "test_type":
                "synthetic_sanity_test",

            "parameter_tuning_allowed":
                False
        },

        "test_cases":
            TEST_CASES,

        "initialization_seconds":
            initialization_times,

        "predictions":
            prediction_rows,

        "metrics":
            metric_rows,

        "pairwise_comparisons":
            pairwise_rows,

        "confusion_matrices":
            confusion_matrices,

        "classical_results":
            {

                system_name:
                    json_safe(
                        results
                    )

                for system_name, results
                in classical_results.items()
            },

        "llm_only":
            json_safe(
                llm_run
            ),

        "dependency_llm":
            json_safe(
                dependency_llm_run
            ),

        "runtime":
            runtime_rows
    }


    detailed_json_path = (
        OUTPUT_DIR
        / "detailed_results_16.json"
    )


    with open(
        detailed_json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            detailed_output,
            file,
            ensure_ascii=False,
            indent=2
        )


    # ========================================================
    # FINAL OUTPUT FILES
    # ========================================================

    print_title(
        "OUTPUT FILES"
    )

    reporter.log(
        "Predictions:",
        prediction_csv_path
    )

    reporter.log(
        "Metrics:",
        metrics_csv_path
    )

    reporter.log(
        "Pairwise comparisons:",
        pairwise_csv_path
    )

    reporter.log(
        "Lexicon score deltas:",
        score_csv_path
    )

    reporter.log(
        "Runtime:",
        runtime_csv_path
    )

    reporter.log(
        "Full JSON:",
        detailed_json_path
    )

    reporter.log(
        "Console report:",
        REPORT_PATH
    )


    # ========================================================
    # FINAL STRUCTURAL CHECKS
    # ========================================================

    print_title(
        "FINAL STRUCTURAL CHECKS"
    )


    assert (
        len(
            prediction_df
        )
        == 16
    )


    for system_name in [
        "H",
        "H->D",
        "D->H",
        "H->D-LW"
    ]:

        assert all(

            prediction
            in LABELS

            for prediction
            in predictions_by_system[
                system_name
            ]
        )


    reporter.log(
        "Classical system label validation: PASSED"
    )


    reporter.log(
        "H -> D baseline reuse checks: PASSED"
    )


    reporter.log(
        "H -> D-LW dependency baseline checks: PASSED"
    )


    unresolved_llm = sum(
        prediction is None

        for prediction
        in predictions_by_system[
            "L"
        ]
    )


    unresolved_dependency_llm = sum(
        prediction is None

        for prediction
        in predictions_by_system[
            "D->L"
        ]
    )


    reporter.log(
        "LLM unresolved predictions:",
        unresolved_llm
    )


    reporter.log(
        "D->LLM unresolved predictions:",
        unresolved_dependency_llm
    )


    if (
        unresolved_llm == 0
        and unresolved_dependency_llm == 0
    ):

        reporter.log(
            "All six systems produced complete predictions."
        )

    else:

        reporter.log(
            "[WARNING] At least one LLM prediction remains "
            "unresolved. Do NOT interpret final LLM metrics "
            "until this is resolved."
        )


    print_title(
        "SIX-SYSTEM SANITY TEST COMPLETED"
    )


finally:

    reporter.close()