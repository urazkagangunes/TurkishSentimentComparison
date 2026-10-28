import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from src.evaluation.dataset_loader import load_sentiment_dataset
from src.hisnet.hisnet_classifier import HisNetClassifier
from src.hisnet.hisnet_dependency_classifier import HisNetDependencyClassifier
from src.hisnet.dependency_hisnet_classifier import DependencyHisNetClassifier
from src.hisnet.literature_weighted_hisnet_dependency_classifier import (
    LiteratureWeightedHisNetDependencyClassifier,
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================
#
# PURPOSE:
#   Frozen 500-review evaluation of the four NON-LLM systems:
#
#       H
#       H -> D
#       D -> H
#       H -> D -> Literature Weighting
#
# IMPORTANT:
#   - This script does NOT tune any threshold or coefficient.
#   - It reuses the exact existing 500-review sample when present.
#   - It writes to a NEW result directory so old 3-system results
#     cannot be silently reused after the code changes.
# ============================================================

DATASET_PATH = Path(
    "datasets/Modified/sentence_sentiment/"
    "TRSAv1_fixed.csv"
)

# Reuse the SAME 500 sample used by the previous experiment.
CANONICAL_SAMPLE_DIR = Path(
    "results/trsav1_four_systems_500"
)

SAMPLE_FILE = (
    CANONICAL_SAMPLE_DIR
    / "sample_500.csv"
)

# Fresh output directory for the frozen four-classical run.
OUTPUT_DIR = Path(
    "results/trsav1_four_classical_500_post_negation_freeze"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CANONICAL_SAMPLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SAMPLE_SIZE = 500
RANDOM_STATE = 42

LABELS = [
    "negative",
    "neutral",
    "positive",
]

SYSTEM_COLUMNS = {
    "H": "hisnet_prediction",
    "H->D": "h_to_d_prediction",
    "D->H": "d_to_h_prediction",
    "H->D-LW": "lw_prediction",
}

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "predictions_500.csv"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "metrics_500.csv"
)

PER_CLASS_FILE = (
    OUTPUT_DIR
    / "per_class_metrics_500.csv"
)

PAIRWISE_FILE = (
    OUTPUT_DIR
    / "pairwise_comparisons_500.csv"
)

TRANSITIONS_FILE = (
    OUTPUT_DIR
    / "prediction_transitions_500.csv"
)

OPERATION_SUMMARY_FILE = (
    OUTPUT_DIR
    / "operation_summary_500.csv"
)

TIMING_FILE = (
    OUTPUT_DIR
    / "timing_summary_500.csv"
)

INTERESTING_CASES_FILE = (
    OUTPUT_DIR
    / "interesting_cases_500.csv"
)

DIAGNOSTIC_FLAGS_FILE = (
    OUTPUT_DIR
    / "diagnostic_flags_500.csv"
)


# ============================================================
# GENERIC HELPERS
# ============================================================

def atomic_save_csv(
    dataframe,
    path,
):
    """
    Avoids a partially written CSV if execution is interrupted.
    """

    tmp_path = Path(
        str(path)
        + ".tmp"
    )

    dataframe.to_csv(
        tmp_path,
        index=False,
        encoding="utf-8-sig",
    )

    tmp_path.replace(
        path
    )


def calculate_metrics(
    dataframe,
    prediction_column,
):
    valid = dataframe[
        dataframe[
            prediction_column
        ].isin(
            LABELS
        )
    ].copy()

    if len(valid) != len(dataframe):
        raise RuntimeError(
            f"{prediction_column} contains unresolved/invalid predictions."
        )

    gold = valid["gold"]
    predictions = valid[
        prediction_column
    ]

    accuracy = accuracy_score(
        gold,
        predictions,
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        gold,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    (
        _,
        _,
        weighted_f1,
        _,
    ) = precision_recall_fscore_support(
        gold,
        predictions,
        labels=LABELS,
        average="weighted",
        zero_division=0,
    )

    return {
        "n": len(valid),
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def save_confusion_matrix(
    dataframe,
    prediction_column,
    filename,
):
    cm = confusion_matrix(
        dataframe["gold"],
        dataframe[
            prediction_column
        ],
        labels=LABELS,
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
        ],
    )

    cm_df.to_csv(
        OUTPUT_DIR / filename,
        encoding="utf-8-sig",
    )

    return cm_df


def operation_count(
    result,
    operation_name,
):
    """
    Counts an exact operation across canonical scopes.
    """

    count = 0

    for scope in (
        result.get("scopes", [])
        or []
    ):
        operations = (
            scope.get("operations", [])
            or []
        )

        count += sum(
            1
            for operation in operations
            if operation == operation_name
        )

    return count


def operation_prefix_count(
    result,
    prefix,
):
    """
    Counts operations such as:
        literature_modifier_weight:1.2
        literature_contrast_weight:1.85
    """

    count = 0

    for scope in (
        result.get("scopes", [])
        or []
    ):
        operations = (
            scope.get("operations", [])
            or []
        )

        count += sum(
            1
            for operation in operations
            if str(operation).startswith(
                prefix
            )
        )

    return count


def count_shared_negation_groups(
    result,
):
    """
    Diagnostic only.

    A shared negation can be CORRECT, e.g.:
        iyi ve güzel değildi

    Therefore this is counted but NEVER treated as an error.
    """

    owners = defaultdict(
        set
    )

    for scope in (
        result.get("scopes", [])
        or []
    ):
        if not scope.get(
            "active_sentiment",
            True,
        ):
            continue

        sentiment_token = (
            scope.get(
                "sentiment_token"
            )
            or {}
        )

        sentiment_id = (
            sentiment_token.get("id")
        )

        if sentiment_id is None:
            continue

        for negation in (
            scope.get("negations", [])
            or []
        ):
            negation_id = (
                negation.get("id")
            )

            if negation_id is None:
                continue

            owners[
                int(negation_id)
            ].add(
                int(sentiment_id)
            )

    return sum(
        1
        for sentiment_ids
        in owners.values()
        if len(sentiment_ids) > 1
    )


def count_duplicate_contrast_pairs(
    lw_result,
):
    """
    Counts sentence-level cases where multiple contrast markers
    collapse onto the same reconstructed pre/post head pair.

    This is a known diagnostic limitation, not automatically
    treated as a classification bug.
    """

    duplicate_count = 0

    relations_by_sentence = (
        lw_result.get(
            "contrast_relations",
            {}
        )
        or {}
    )

    for relations in (
        relations_by_sentence.values()
    ):
        pair_counts = Counter()

        for relation in (
            relations
            or []
        ):
            pair = (
                relation.get(
                    "pre_head_id"
                ),
                relation.get(
                    "post_head_id"
                ),
            )

            pair_counts[pair] += 1

        duplicate_count += sum(
            1
            for count
            in pair_counts.values()
            if count > 1
        )

    return duplicate_count


def turkish_lower(
    text,
):
    return (
        str(text)
        .replace("I", "ı")
        .replace("İ", "i")
        .lower()
    )


def contiguous_phrase_hits(
    text,
    phrases,
):
    """
    Coarse TEXT-LEVEL diagnostic only.

    A hit does NOT mean the expression truly modifies a sentiment
    word. It only tells us that a literature-listed expression
    occurs somewhere in the review.
    """

    normalized = turkish_lower(
        text
    )

    tokens = re.findall(
        r"[a-zçğıöşüâîû]+",
        normalized,
    )

    hits = []

    for phrase in phrases:
        phrase = tuple(
            turkish_lower(
                token
            )
            for token in phrase
        )

        n = len(phrase)

        for i in range(
            0,
            len(tokens) - n + 1,
        ):
            if tuple(
                tokens[
                    i:i + n
                ]
            ) == phrase:
                hits.append(
                    " ".join(
                        phrase
                    )
                )
                break

    return sorted(
        set(
            hits
        )
    )


# ============================================================
# EXACT MCNEMAR TEST
# ============================================================

def exact_mcnemar_p(
    a_only_correct,
    b_only_correct,
):
    """
    Two-sided exact McNemar/binomial test.

    Under H0, among discordant correctness pairs,
    each direction has probability 0.5.

    Implemented without scipy so the experiment does not
    gain a new dependency.
    """

    b = int(
        a_only_correct
    )

    c = int(
        b_only_correct
    )

    n = b + c

    if n == 0:
        return 1.0

    k = min(
        b,
        c,
    )

    lower_tail = sum(
        math.comb(
            n,
            i,
        )
        for i in range(
            k + 1
        )
    ) / (
        2 ** n
    )

    return min(
        1.0,
        2.0 * lower_tail,
    )


# ============================================================
# LOAD / CREATE THE EXACT 500 SAMPLE
# ============================================================

print(
    "=" * 110
)

print(
    "TRSAv1 — FOUR CLASSICAL SYSTEMS — FROZEN 500-REVIEW EVALUATION"
)

print(
    "=" * 110
)

if SAMPLE_FILE.exists():

    print(
        "\nReusing canonical 500-review sample:"
    )

    print(
        SAMPLE_FILE
    )

    sample = pd.read_csv(
        SAMPLE_FILE
    )

else:

    print(
        "\nCanonical sample not found; creating it once with random_state=42."
    )

    dataset = load_sentiment_dataset(
        file_path=DATASET_PATH,
        text_column="review",
        label_column="score",
        separator=";",
    )

    _, sample = train_test_split(
        dataset,
        test_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=dataset["gold"],
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
        ),
    )

    sample.to_csv(
        SAMPLE_FILE,
        index=False,
        encoding="utf-8-sig",
    )


# ------------------------------------------------------------
# Sample integrity checks
# ------------------------------------------------------------

assert (
    len(sample)
    == SAMPLE_SIZE
)

assert (
    sample["sample_id"]
    .nunique()
    == SAMPLE_SIZE
)

assert (
    sample["sample_id"]
    .isna()
    .sum()
    == 0
)

assert set(
    sample["gold"]
    .unique()
).issubset(
    set(
        LABELS
    )
)

print(
    "\nSample size:",
    len(sample),
)

print(
    "\nGold distribution:"
)

print(
    sample["gold"]
    .value_counts()
    .reindex(
        LABELS,
        fill_value=0,
    )
)


# ============================================================
# INITIALIZE FOUR CLASSICAL SYSTEMS
# ============================================================

print(
    "\n"
    + "=" * 110
)

print(
    "INITIALIZING SYSTEMS"
)

print(
    "=" * 110
)

print(
    "H: HisNet"
)

hisnet = HisNetClassifier()

print(
    "H->D: HisNet -> Dependency"
)

h_to_d = HisNetDependencyClassifier(
    hisnet_classifier=hisnet
)

print(
    "D->H: Dependency -> HisNet"
)

d_to_h = DependencyHisNetClassifier()

print(
    "H->D-LW: HisNet -> Dependency -> Literature Weighting"
)

lw = (
    LiteratureWeightedHisNetDependencyClassifier(
        hisnet_classifier=hisnet
    )
)


# ============================================================
# RUN ALL 500 REVIEWS
# ============================================================

rows = []
diagnostic_flags = []

modifier_phrases = list(
    lw.MODIFIER_WEIGHTS.keys()
)

experiment_start = (
    time.perf_counter()
)


for row_index, row in (
    sample.iterrows()
):

    sample_id = int(
        row["sample_id"]
    )

    text = str(
        row["text"]
    )

    gold = row[
        "gold"
    ]

    # ========================================================
    # 1. H
    # ========================================================

    start = (
        time.perf_counter()
    )

    h_result = hisnet.classify(
        text
    )

    h_seconds = (
        time.perf_counter()
        - start
    )


    # ========================================================
    # 2. H -> D
    #
    # Exact same H result is injected.
    # ========================================================

    start = (
        time.perf_counter()
    )

    hd_result = h_to_d.classify(
        text,
        hisnet_result=h_result,
    )

    hd_seconds = (
        time.perf_counter()
        - start
    )


    # ========================================================
    # 3. D -> H
    # ========================================================

    start = (
        time.perf_counter()
    )

    dh_result = d_to_h.classify(
        text
    )

    dh_seconds = (
        time.perf_counter()
        - start
    )


    # ========================================================
    # 4. H -> D -> LITERATURE WEIGHTING
    # ========================================================

    start = (
        time.perf_counter()
    )

    lw_result = lw.classify(
        text,
        hisnet_result=h_result,
    )

    lw_seconds = (
        time.perf_counter()
        - start
    )


    # ========================================================
    # HARD ARCHITECTURAL INVARIANTS
    # ========================================================

    if not hd_result.get(
        "baseline_reused",
        False,
    ):
        raise RuntimeError(
            f"sample_id={sample_id}: "
            f"H->D did not reuse supplied H baseline."
        )

    if not math.isclose(
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
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            f"sample_id={sample_id}: "
            f"H/H->D positive baseline mismatch."
        )

    if not math.isclose(
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
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            f"sample_id={sample_id}: "
            f"H/H->D negative baseline mismatch."
        )

    # LW must begin from exact frozen H->D result.
    if (
        lw_result.get(
            "dependency_prediction"
        )
        != hd_result.get(
            "prediction"
        )
    ):
        raise RuntimeError(
            f"sample_id={sample_id}: "
            f"LW frozen H->D prediction mismatch."
        )

    if not math.isclose(
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
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            f"sample_id={sample_id}: "
            f"LW frozen H->D positive score mismatch."
        )

    if not math.isclose(
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
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            f"sample_id={sample_id}: "
            f"LW frozen H->D negative score mismatch."
        )


    # ========================================================
    # DIAGNOSTIC COUNTS
    # ========================================================

    hd_suppressed = operation_count(
        hd_result,
        "suppressed_as_target",
    )

    hd_negation_flips = operation_count(
        hd_result,
        "negation_flip",
    )

    hd_modifier_detected = operation_count(
        hd_result,
        "modifier_detected",
    )

    hd_contrast_detected = operation_count(
        hd_result,
        "contrast_detected",
    )

    dh_suppressed = operation_count(
        dh_result,
        "suppressed_as_target",
    )

    dh_negation_flips = operation_count(
        dh_result,
        "negation_flip",
    )

    dh_modifier_detected = operation_count(
        dh_result,
        "modifier_detected",
    )

    dh_contrast_detected = operation_count(
        dh_result,
        "contrast_detected",
    )

    lw_modifier_operations = (
        operation_prefix_count(
            lw_result,
            "literature_modifier_weight:",
        )
    )

    lw_contrast_operations = (
        operation_prefix_count(
            lw_result,
            "literature_contrast_weight:",
        )
    )

    lw_duplicate_contrast_pairs = (
        count_duplicate_contrast_pairs(
            lw_result
        )
    )

    text_modifier_hits = (
        contiguous_phrase_hits(
            text,
            modifier_phrases,
        )
    )


    # ========================================================
    # DIAGNOSTIC FLAGS
    # ========================================================

    if len(
        hd_result.get(
            "unmatched_scopes",
            [],
        )
    ) > 0:
        diagnostic_flags.append({
            "sample_id":
                sample_id,

            "flag":
                "H_TO_D_UNMATCHED_SCOPE",

            "text":
                text,

            "detail":
                str(
                    len(
                        hd_result[
                            "unmatched_scopes"
                        ]
                    )
                ),
        })


    if (
        lw_duplicate_contrast_pairs
        > 0
    ):
        diagnostic_flags.append({
            "sample_id":
                sample_id,

            "flag":
                "LW_DUPLICATE_CONTRAST_PAIR",

            "text":
                text,

            "detail":
                str(
                    lw_duplicate_contrast_pairs
                ),
        })


    # Coarse coverage diagnostic only.
    if (
        len(
            text_modifier_hits
        ) > 0
        and int(
            lw_result.get(
                "modifier_weighted_scope_count",
                0,
            )
        ) == 0
    ):
        diagnostic_flags.append({
            "sample_id":
                sample_id,

            "flag":
                "TEXT_MODIFIER_PRESENT_BUT_NO_LW_WEIGHT",

            "text":
                text,

            "detail":
                "; ".join(
                    text_modifier_hits
                ),
        })


    # ========================================================
    # SAVE ONE REVIEW
    # ========================================================

    rows.append({

        "sample_id":
            sample_id,

        "text":
            text,

        "gold":
            gold,

        # ----------------------------------------------------
        # H
        # ----------------------------------------------------

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

        "hisnet_sentiment_words":
            len(
                h_result.get(
                    "sentiment_words",
                    [],
                )
            ),

        # ----------------------------------------------------
        # H -> D
        # ----------------------------------------------------

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

        "h_to_d_scope_count":
            len(
                hd_result.get(
                    "scopes",
                    [],
                )
            ),

        "h_to_d_unmatched_scope_count":
            len(
                hd_result.get(
                    "unmatched_scopes",
                    [],
                )
            ),

        "h_to_d_suppressed":
            hd_suppressed,

        "h_to_d_negation_flips":
            hd_negation_flips,

        "h_to_d_modifier_detected":
            hd_modifier_detected,

        "h_to_d_contrast_detected":
            hd_contrast_detected,

        "h_to_d_shared_negation_groups":
            count_shared_negation_groups(
                hd_result
            ),

        # ----------------------------------------------------
        # D -> H
        # ----------------------------------------------------

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

        "d_to_h_lexicon_matched_candidates":
            dh_result[
                "lexicon_matched_candidate_count"
            ],

        "d_to_h_sentiment_words":
            dh_result[
                "sentiment_word_count"
            ],

        "d_to_h_suppressed":
            dh_suppressed,

        "d_to_h_negation_flips":
            dh_negation_flips,

        "d_to_h_modifier_detected":
            dh_modifier_detected,

        "d_to_h_contrast_detected":
            dh_contrast_detected,

        "d_to_h_shared_negation_groups":
            count_shared_negation_groups(
                dh_result
            ),

        # ----------------------------------------------------
        # H -> D -> LW
        # ----------------------------------------------------

        "lw_prediction":
            lw_result[
                "prediction"
            ],

        "lw_positive":
            lw_result[
                "positive_score"
            ],

        "lw_negative":
            lw_result[
                "negative_score"
            ],

        "lw_modifier_weighted_scopes":
            lw_result.get(
                "modifier_weighted_scope_count",
                0,
            ),

        "lw_modifier_component_suppressed":
            lw_result.get(
                "modifier_component_suppressed_scope_count",
                0,
            ),

        "lw_contrast_marker_suppressed":
            lw_result.get(
                "contrast_marker_suppressed_scope_count",
                0,
            ),

        "lw_contrast_weighted_scopes":
            lw_result.get(
                "contrast_weighted_scope_count",
                0,
            ),

        "lw_modifier_operations":
            lw_modifier_operations,

        "lw_contrast_operations":
            lw_contrast_operations,

        "lw_duplicate_contrast_pairs":
            lw_duplicate_contrast_pairs,

        "lw_shared_negation_groups":
            count_shared_negation_groups(
                lw_result
            ),

        "text_literature_modifier_hits":
            "; ".join(
                text_modifier_hits
            ),

        # ----------------------------------------------------
        # TIMING
        # ----------------------------------------------------

        "hisnet_seconds":
            h_seconds,

        "h_to_d_seconds":
            hd_seconds,

        "d_to_h_seconds":
            dh_seconds,

        "lw_seconds":
            lw_seconds,
    })


    # Checkpoint every 25 reviews.
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
            PREDICTIONS_FILE,
        )

        atomic_save_csv(
            pd.DataFrame(
                diagnostic_flags
            ),
            DIAGNOSTIC_FLAGS_FILE,
        )


results = pd.DataFrame(
    rows
)

atomic_save_csv(
    results,
    PREDICTIONS_FILE,
)

atomic_save_csv(
    pd.DataFrame(
        diagnostic_flags
    ),
    DIAGNOSTIC_FLAGS_FILE,
)

total_experiment_seconds = (
    time.perf_counter()
    - experiment_start
)


# ============================================================
# SYSTEM METRICS
# ============================================================

metric_rows = []

for system_name, prediction_column in (
    SYSTEM_COLUMNS.items()
):

    metrics = calculate_metrics(
        results,
        prediction_column,
    )

    metric_rows.append({
        "system":
            system_name,

        **metrics,
    })


metrics_df = pd.DataFrame(
    metric_rows
)

metrics_df.to_csv(
    METRICS_FILE,
    index=False,
    encoding="utf-8-sig",
)


print(
    "\n"
    + "=" * 110
)

print(
    "SYSTEM METRICS"
)

print(
    "=" * 110
)

print(
    metrics_df.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
            if pd.notna(x)
            else ""
        ),
    )
)


# ============================================================
# PER-CLASS METRICS
# ============================================================

per_class_rows = []

for system_name, prediction_column in (
    SYSTEM_COLUMNS.items()
):

    (
        precision,
        recall,
        f1,
        support,
    ) = (
        precision_recall_fscore_support(
            results["gold"],
            results[
                prediction_column
            ],
            labels=LABELS,
            average=None,
            zero_division=0,
        )
    )

    for (
        label,
        p,
        r,
        f,
        s,
    ) in zip(
        LABELS,
        precision,
        recall,
        f1,
        support,
    ):

        per_class_rows.append({
            "system":
                system_name,

            "label":
                label,

            "precision":
                p,

            "recall":
                r,

            "f1":
                f,

            "support":
                int(
                    s
                ),
        })


per_class_df = pd.DataFrame(
    per_class_rows
)

per_class_df.to_csv(
    PER_CLASS_FILE,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# CONFUSION MATRICES
# ============================================================

for system_name, prediction_column in (
    SYSTEM_COLUMNS.items()
):

    safe_name = (
        system_name
        .replace("->", "_to_")
        .replace("-", "_")
        .replace(" ", "")
    )

    cm_df = save_confusion_matrix(
        results,
        prediction_column,
        f"confusion_{safe_name}.csv",
    )

    print(
        "\n"
        + "-" * 110
    )

    print(
        f"CONFUSION MATRIX: {system_name}"
    )

    print(
        cm_df
    )


# ============================================================
# PAIRWISE COMPARISONS
# ============================================================

PAIRWISE_SYSTEMS = [
    ("H", "H->D"),
    ("H", "D->H"),
    ("H->D", "D->H"),
    ("H->D", "H->D-LW"),
    ("D->H", "H->D-LW"),
    ("H", "H->D-LW"),
]

pairwise_rows = []
transition_rows = []


for system_a, system_b in (
    PAIRWISE_SYSTEMS
):

    column_a = (
        SYSTEM_COLUMNS[
            system_a
        ]
    )

    column_b = (
        SYSTEM_COLUMNS[
            system_b
        ]
    )

    pred_a = results[
        column_a
    ]

    pred_b = results[
        column_b
    ]

    gold = results[
        "gold"
    ]

    correct_a = (
        pred_a
        == gold
    )

    correct_b = (
        pred_b
        == gold
    )

    changed = (
        pred_a
        != pred_b
    )

    a_only_correct = int(
        (
            correct_a
            &
            ~correct_b
        ).sum()
    )

    b_only_correct = int(
        (
            ~correct_a
            &
            correct_b
        ).sum()
    )

    both_correct = int(
        (
            correct_a
            &
            correct_b
        ).sum()
    )

    both_wrong = int(
        (
            ~correct_a
            &
            ~correct_b
        ).sum()
    )

    wrong_to_wrong_changed = int(
        (
            changed
            &
            ~correct_a
            &
            ~correct_b
        ).sum()
    )

    same_prediction = int(
        (
            pred_a
            == pred_b
        ).sum()
    )

    changed_prediction = int(
        changed.sum()
    )

    accuracy_a = float(
        correct_a.mean()
    )

    accuracy_b = float(
        correct_b.mean()
    )

    p_value = exact_mcnemar_p(
        a_only_correct,
        b_only_correct,
    )

    pairwise_rows.append({

        "system_a":
            system_a,

        "system_b":
            system_b,

        "n":
            len(results),

        "same_prediction":
            same_prediction,

        "changed_prediction":
            changed_prediction,

        "agreement_rate":
            (
                same_prediction
                / len(results)
            ),

        "both_correct":
            both_correct,

        "both_wrong":
            both_wrong,

        "a_only_correct":
            a_only_correct,

        "b_only_correct":
            b_only_correct,

        "net_correct_b_minus_a":
            (
                b_only_correct
                - a_only_correct
            ),

        "wrong_to_wrong_changed":
            wrong_to_wrong_changed,

        "accuracy_a":
            accuracy_a,

        "accuracy_b":
            accuracy_b,

        "accuracy_delta_b_minus_a":
            (
                accuracy_b
                - accuracy_a
            ),

        "mcnemar_exact_p":
            p_value,
    })


    transition_df = pd.DataFrame({
        "gold":
            gold,

        "prediction_a":
            pred_a,

        "prediction_b":
            pred_b,
    })

    grouped = (
        transition_df
        .groupby(
            [
                "gold",
                "prediction_a",
                "prediction_b",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    for transition in (
        grouped.to_dict(
            "records"
        )
    ):
        transition_rows.append({
            "system_a":
                system_a,

            "system_b":
                system_b,

            **transition,
        })


pairwise_df = pd.DataFrame(
    pairwise_rows
)

pairwise_df.to_csv(
    PAIRWISE_FILE,
    index=False,
    encoding="utf-8-sig",
)

pd.DataFrame(
    transition_rows
).to_csv(
    TRANSITIONS_FILE,
    index=False,
    encoding="utf-8-sig",
)


print(
    "\n"
    + "=" * 110
)

print(
    "PAIRWISE COMPARISONS"
)

print(
    "=" * 110
)

print(
    pairwise_df.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.6f}"
            if pd.notna(x)
            else ""
        ),
    )
)


# ============================================================
# OPERATION / COVERAGE SUMMARY
# ============================================================

operation_columns = [
    "h_to_d_suppressed",
    "h_to_d_negation_flips",
    "h_to_d_modifier_detected",
    "h_to_d_contrast_detected",
    "h_to_d_shared_negation_groups",
    "h_to_d_unmatched_scope_count",

    "d_to_h_candidates",
    "d_to_h_lexicon_matched_candidates",
    "d_to_h_sentiment_words",
    "d_to_h_suppressed",
    "d_to_h_negation_flips",
    "d_to_h_modifier_detected",
    "d_to_h_contrast_detected",
    "d_to_h_shared_negation_groups",

    "lw_modifier_weighted_scopes",
    "lw_modifier_component_suppressed",
    "lw_contrast_marker_suppressed",
    "lw_contrast_weighted_scopes",
    "lw_duplicate_contrast_pairs",
    "lw_shared_negation_groups",
]

operation_rows = []

for column in operation_columns:

    operation_rows.append({

        "diagnostic":
            column,

        "total_count":
            float(
                results[
                    column
                ].sum()
            ),

        "reviews_with_event":
            int(
                (
                    results[
                        column
                    ]
                    > 0
                ).sum()
            ),

        "review_rate":
            float(
                (
                    results[
                        column
                    ]
                    > 0
                ).mean()
            ),

        "mean_per_review":
            float(
                results[
                    column
                ].mean()
            ),
    })


operation_summary_df = (
    pd.DataFrame(
        operation_rows
    )
)

operation_summary_df.to_csv(
    OPERATION_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig",
)


print(
    "\n"
    + "=" * 110
)

print(
    "OPERATION / COVERAGE SUMMARY"
)

print(
    "=" * 110
)

print(
    operation_summary_df.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
            if pd.notna(x)
            else ""
        ),
    )
)


# ============================================================
# TIMING SUMMARY
# ============================================================

timing_mapping = {
    "H":
        "hisnet_seconds",

    "H->D":
        "h_to_d_seconds",

    "D->H":
        "d_to_h_seconds",

    "H->D-LW":
        "lw_seconds",
}

timing_rows = []

for system_name, column in (
    timing_mapping.items()
):

    timing_rows.append({

        "system":
            system_name,

        "total_seconds":
            float(
                results[
                    column
                ].sum()
            ),

        "mean_seconds_per_review":
            float(
                results[
                    column
                ].mean()
            ),

        "median_seconds_per_review":
            float(
                results[
                    column
                ].median()
            ),

        "max_seconds_per_review":
            float(
                results[
                    column
                ].max()
            ),
    })


timing_rows.append({

    "system":
        "FULL_EXPERIMENT_WALL",

    "total_seconds":
        total_experiment_seconds,

    "mean_seconds_per_review":
        (
            total_experiment_seconds
            / SAMPLE_SIZE
        ),

    "median_seconds_per_review":
        None,

    "max_seconds_per_review":
        None,
})


timing_df = pd.DataFrame(
    timing_rows
)

timing_df.to_csv(
    TIMING_FILE,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# INTERESTING CASES
# ============================================================

prediction_columns = list(
    SYSTEM_COLUMNS.values()
)

any_wrong = pd.Series(
    False,
    index=results.index,
)

for prediction_column in (
    prediction_columns
):

    any_wrong = (
        any_wrong
        |
        (
            results[
                prediction_column
            ]
            != results[
                "gold"
            ]
        )
    )


prediction_nunique = (
    results[
        prediction_columns
    ]
    .nunique(
        axis=1
    )
)

any_disagreement = (
    prediction_nunique
    > 1
)


interesting = results[
    any_wrong
    |
    any_disagreement
].copy()


interesting.to_csv(
    INTERESTING_CASES_FILE,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 110
)

print(
    "FROZEN 500-REVIEW FOUR-CLASSICAL EXPERIMENT COMPLETED"
)

print(
    "=" * 110
)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)

print(
    "\nImportant files:"
)

for path in [
    PREDICTIONS_FILE,
    METRICS_FILE,
    PER_CLASS_FILE,
    PAIRWISE_FILE,
    TRANSITIONS_FILE,
    OPERATION_SUMMARY_FILE,
    TIMING_FILE,
    INTERESTING_CASES_FILE,
    DIAGNOSTIC_FLAGS_FILE,
]:
    print(
        " -",
        path,
    )


print(
    "\nTotal wall seconds:",
    f"{total_experiment_seconds:.2f}",
)

print(
    "=" * 110
)