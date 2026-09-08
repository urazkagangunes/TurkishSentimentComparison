import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

from src.hisnet.hisnet_classifier import (
    HisNetClassifier
)

from src.hisnet.hisnet_dependency_classifier import (
    HisNetDependencyClassifier
)


LABELS = [
    "negative",
    "neutral",
    "positive"
]


class SentimentEvaluator:

    def __init__(self):

        print(
            "Initializing HisNet..."
        )

        self.hisnet = (
            HisNetClassifier()
        )

        print(
            "Initializing HisNet + Dependency..."
        )

        self.hisnet_dependency = (
            HisNetDependencyClassifier()
        )

    # =========================================================
    # CHECKPOINT / RESUME
    # =========================================================

    @staticmethod
    def _dataset_fingerprint(
        dataframe
    ):
        """
        Creates a deterministic fingerprint from text + gold
        labels in their current order.

        This prevents accidentally resuming a checkpoint from
        another dataset or from the same dataset in a different
        order.
        """

        hasher = hashlib.sha256()

        for text, gold in dataframe[
            ["text", "gold"]
        ].itertuples(
            index=False,
            name=None
        ):

            hasher.update(
                str(text).encode(
                    "utf-8"
                )
            )

            hasher.update(
                b"\x00"
            )

            hasher.update(
                str(gold).encode(
                    "utf-8"
                )
            )

            hasher.update(
                b"\x01"
            )

        return hasher.hexdigest()

    @staticmethod
    def _json_default(
        value
    ):
        """
        Converts NumPy/Pandas scalar values into native Python
        values when writing checkpoint JSON.
        """

        if hasattr(
            value,
            "item"
        ):
            return value.item()

        raise TypeError(
            f"Object of type "
            f"{type(value).__name__} "
            f"is not JSON serializable"
        )

    @classmethod
    def _save_checkpoint(
        cls,
        checkpoint_path,
        dataset_name,
        dataset_fingerprint,
        total_examples,
        rows
    ):
        """
        Saves evaluation progress atomically.

        A temporary file is written first. Only after the JSON
        has been written successfully does it replace the
        previous checkpoint.
        """

        checkpoint_data = {
            "checkpoint_version":
                1,

            "dataset":
                dataset_name,

            "fingerprint":
                dataset_fingerprint,

            "total_examples":
                total_examples,

            "processed_examples":
                len(rows),

            "rows":
                rows
        }

        temporary_path = Path(
            str(checkpoint_path)
            + ".tmp"
        )

        with open(
            temporary_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                checkpoint_data,
                file,
                ensure_ascii=False,
                default=cls._json_default
            )

        temporary_path.replace(
            checkpoint_path
        )

    @staticmethod
    def _load_checkpoint(
        checkpoint_path,
        dataset_name,
        dataset_fingerprint,
        total_examples
    ):
        """
        Loads and validates an existing checkpoint.
        """

        with open(
            checkpoint_path,
            "r",
            encoding="utf-8"
        ) as file:

            checkpoint = (
                json.load(file)
            )

        if (
            checkpoint.get(
                "checkpoint_version"
            )
            != 1
        ):
            raise ValueError(
                "Unsupported checkpoint version."
            )

        if (
            checkpoint.get(
                "dataset"
            )
            != dataset_name
        ):
            raise ValueError(
                "Checkpoint dataset name does not "
                "match the current dataset."
            )

        if (
            checkpoint.get(
                "total_examples"
            )
            != total_examples
        ):
            raise ValueError(
                "Checkpoint example count does not "
                "match the current dataset."
            )

        if (
            checkpoint.get(
                "fingerprint"
            )
            != dataset_fingerprint
        ):
            raise ValueError(
                "Checkpoint dataset fingerprint does "
                "not match the current dataset/order."
            )

        rows = checkpoint.get(
            "rows",
            []
        )

        processed_examples = (
            checkpoint.get(
                "processed_examples",
                len(rows)
            )
        )

        if (
            processed_examples
            != len(rows)
        ):
            raise ValueError(
                "Checkpoint is inconsistent: "
                "processed_examples does not match "
                "the number of stored rows."
            )

        if (
            processed_examples
            > total_examples
        ):
            raise ValueError(
                "Checkpoint contains more examples "
                "than the current dataset."
            )

        return rows

    # =========================================================
    # DEPENDENCY DIAGNOSTICS
    # =========================================================

    @staticmethod
    def _extract_dependency_analysis(
        dependency_result
    ):
        """
        Extracts compact linguistic information from
        HisNet + Dependency output.

        This is stored for later error analysis.
        """

        opinions = []
        targets = []
        modifiers = []
        negations = []
        contrasts = []
        operations = []

        scope_count = 0
        active_opinion_count = 0
        suppressed_count = 0

        scopes = dependency_result.get(
            "scopes"
        )

        # Backward compatibility with older classifier outputs.
        if scopes is None:

            scopes = []

            for sentence_result in dependency_result.get(
                    "sentences",
                    []
            ):
                scopes.extend(
                    sentence_result.get(
                        "scopes",
                        []
                    )
                )

        for scope in scopes:

                scope_count += 1

                root = scope.get(
                    "root"
                )

                if root is not None:
                    opinions.append(
                        root
                    )

                target = scope.get(
                    "target"
                )

                if target is not None:

                    target_root = (
                        target.get("root")
                        or target.get("lemma")
                        or target.get("word")
                    )

                    if target_root:
                        targets.append(
                            target_root
                        )

                if scope.get(
                    "active_sentiment",
                    True
                ):

                    active_opinion_count += 1

                else:

                    suppressed_count += 1

                for modifier in scope.get(
                    "modifiers",
                    []
                ):

                    value = (
                        modifier.get("lemma")
                        or modifier.get("word")
                    )

                    if value:
                        modifiers.append(
                            value
                        )

                for negation in scope.get(
                    "negations",
                    []
                ):

                    value = (
                        negation.get("lemma")
                        or negation.get("word")
                    )

                    if value:
                        negations.append(
                            value
                        )

                contrast = scope.get(
                    "contrast"
                )

                if contrast:

                    value = (
                        contrast.get("lemma")
                        or contrast.get("word")
                    )

                    if value:
                        contrasts.append(
                            value
                        )

                operations.extend(
                    scope.get(
                        "operations",
                        []
                    )
                )

        return {
            "opinions":
                opinions,

            "targets":
                targets,

            "modifiers":
                modifiers,

            "negations":
                negations,

            "contrasts":
                contrasts,

            "operations":
                operations,

            "scope_count":
                scope_count,

            "active_opinion_count":
                active_opinion_count,

            "suppressed_count":
                suppressed_count,

            "has_modifier":
                len(modifiers) > 0,

            "has_negation":
                len(negations) > 0,

            "has_contrast":
                len(contrasts) > 0,

            "has_suppression":
                suppressed_count > 0
        }

    # =========================================================
    # METRICS
    # =========================================================

    @staticmethod
    def _calculate_metrics(
        gold,
        predictions
    ):

        accuracy = accuracy_score(
            gold,
            predictions
        )

        precision_macro, recall_macro, f1_macro, _ = (
            precision_recall_fscore_support(
                gold,
                predictions,
                labels=LABELS,
                average="macro",
                zero_division=0
            )
        )

        precision_weighted, recall_weighted, f1_weighted, _ = (
            precision_recall_fscore_support(
                gold,
                predictions,
                labels=LABELS,
                average="weighted",
                zero_division=0
            )
        )

        report = classification_report(
            gold,
            predictions,
            labels=LABELS,
            output_dict=True,
            zero_division=0
        )

        matrix = confusion_matrix(
            gold,
            predictions,
            labels=LABELS
        )

        return {
            "accuracy":
                float(accuracy),

            "macro_precision":
                float(precision_macro),

            "macro_recall":
                float(recall_macro),

            "macro_f1":
                float(f1_macro),

            "weighted_precision":
                float(precision_weighted),

            "weighted_recall":
                float(recall_weighted),

            "weighted_f1":
                float(f1_weighted),

            "classification_report":
                report,

            "confusion_matrix":
                matrix.tolist()
        }

    # =========================================================
    # MAIN EVALUATION
    # =========================================================

    def evaluate(
        self,
        dataframe,
        dataset_name,
        output_directory="results",
        progress_every=25,
        resume=True
    ):

        output_directory = Path(
            output_directory
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        total = len(
            dataframe
        )

        if total == 0:
            raise ValueError(
                "Dataset is empty."
            )

        # =====================================================
        # CHECKPOINT INITIALIZATION
        # =====================================================

        checkpoint_path = (
            output_directory
            / f"{dataset_name}_checkpoint.json"
        )

        dataset_fingerprint = (
            self._dataset_fingerprint(
                dataframe
            )
        )

        rows = []

        # -----------------------------------------------------
        # Resume existing evaluation
        # -----------------------------------------------------

        if (
            resume
            and checkpoint_path.exists()
        ):

            print(
                "\nCheckpoint found:"
            )

            print(
                checkpoint_path
            )

            rows = (
                self._load_checkpoint(
                    checkpoint_path=
                        checkpoint_path,

                    dataset_name=
                        dataset_name,

                    dataset_fingerprint=
                        dataset_fingerprint,

                    total_examples=
                        total
                )
            )

            start_position = len(
                rows
            )

            print(
                f"Resuming from "
                f"{start_position}/{total} "
                f"completed examples."
            )

        else:

            # If the user explicitly starts without resume,
            # an old checkpoint must not remain in place.
            if (
                not resume
                and checkpoint_path.exists()
            ):

                checkpoint_path.unlink()

                print(
                    "\nExisting checkpoint removed "
                    "because resume=False."
                )

            start_position = 0

            print(
                "\nNo checkpoint found. "
                "Starting from the beginning."
            )

        print(
            "\n"
            + "=" * 90
        )

        print(
            f"DATASET: {dataset_name}"
        )

        print(
            f"Examples: {total}"
        )

        print(
            "=" * 90
        )

        # =====================================================
        # EVALUATION LOOP
        # =====================================================

        for position in range(
            start_position,
            total
        ):

            row = dataframe.iloc[
                position
            ]

            original_index = (
                dataframe.index[
                    position
                ]
            )

            text = row["text"]
            gold = row["gold"]

            try:

                # =============================================
                # HISNET
                # =============================================

                hisnet_result = (
                    self.hisnet.classify(
                        text
                    )
                )

                # =============================================
                # HISNET + DEPENDENCY
                # =============================================

                dependency_result = (
                    self.hisnet_dependency.classify(
                        text,
                        hisnet_result=hisnet_result
                    )
                )

                hisnet_prediction = (
                    hisnet_result[
                        "prediction"
                    ]
                )

                dependency_prediction = (
                    dependency_result[
                        "prediction"
                    ]
                )

                # =============================================
                # CORRECTNESS
                # =============================================

                hisnet_correct = (
                    hisnet_prediction
                    == gold
                )

                dependency_correct = (
                    dependency_prediction
                    == gold
                )

                changed = (
                    hisnet_prediction
                    != dependency_prediction
                )

                improved = (
                    not hisnet_correct
                    and dependency_correct
                )

                degraded = (
                    hisnet_correct
                    and not dependency_correct
                )

                changed_but_still_wrong = (
                    changed
                    and not hisnet_correct
                    and not dependency_correct
                )

                # =============================================
                # DEPENDENCY ANALYSIS
                # =============================================

                linguistic = (
                    self._extract_dependency_analysis(
                        dependency_result
                    )
                )

                # =============================================
                # STORE RESULT
                # =============================================

                rows.append({
                    "id":
                        original_index,

                    "text":
                        text,

                    "gold":
                        gold,

                    # -----------------------------------------
                    # HisNet
                    # -----------------------------------------

                    "hisnet_prediction":
                        hisnet_prediction,

                    "hisnet_positive_score":
                        hisnet_result[
                            "positive_score"
                        ],

                    "hisnet_negative_score":
                        hisnet_result[
                            "negative_score"
                        ],

                    "hisnet_correct":
                        hisnet_correct,

                    "lexicon_coverage":
                        hisnet_result[
                            "lexicon_coverage"
                        ],

                    "sentiment_coverage":
                        hisnet_result[
                            "sentiment_coverage"
                        ],

                    # -----------------------------------------
                    # HisNet + Dependency
                    # -----------------------------------------

                    "dependency_prediction":
                        dependency_prediction,

                    "dependency_positive_score":
                        dependency_result[
                            "positive_score"
                        ],

                    "dependency_negative_score":
                        dependency_result[
                            "negative_score"
                        ],

                    "dependency_correct":
                        dependency_correct,

                    # -----------------------------------------
                    # Comparison
                    # -----------------------------------------

                    "prediction_changed":
                        changed,

                    "improved":
                        improved,

                    "degraded":
                        degraded,

                    "changed_but_still_wrong":
                        changed_but_still_wrong,

                    # -----------------------------------------
                    # Linguistic diagnostics
                    # -----------------------------------------

                    "has_negation":
                        linguistic[
                            "has_negation"
                        ],

                    "has_modifier":
                        linguistic[
                            "has_modifier"
                        ],

                    "has_contrast":
                        linguistic[
                            "has_contrast"
                        ],

                    "has_suppression":
                        linguistic[
                            "has_suppression"
                        ],

                    "scope_count":
                        linguistic[
                            "scope_count"
                        ],

                    "active_opinion_count":
                        linguistic[
                            "active_opinion_count"
                        ],

                    "suppressed_count":
                        linguistic[
                            "suppressed_count"
                        ],

                    # Lists are serialized so that the CSV
                    # remains easy to inspect later.
                    "opinions":
                        json.dumps(
                            linguistic[
                                "opinions"
                            ],
                            ensure_ascii=False
                        ),

                    "targets":
                        json.dumps(
                            linguistic[
                                "targets"
                            ],
                            ensure_ascii=False
                        ),

                    "modifiers":
                        json.dumps(
                            linguistic[
                                "modifiers"
                            ],
                            ensure_ascii=False
                        ),

                    "negations":
                        json.dumps(
                            linguistic[
                                "negations"
                            ],
                            ensure_ascii=False
                        ),

                    "contrasts":
                        json.dumps(
                            linguistic[
                                "contrasts"
                            ],
                            ensure_ascii=False
                        ),

                    "operations":
                        json.dumps(
                            linguistic[
                                "operations"
                            ],
                            ensure_ascii=False
                        )
                })

            except Exception as error:

                # ---------------------------------------------
                # Save every successfully completed example
                # before propagating the error.
                # ---------------------------------------------

                self._save_checkpoint(
                    checkpoint_path=
                        checkpoint_path,

                    dataset_name=
                        dataset_name,

                    dataset_fingerprint=
                        dataset_fingerprint,

                    total_examples=
                        total,

                    rows=
                        rows
                )

                print(
                    "\n"
                    + "!" * 90
                )

                print(
                    "EVALUATION ERROR"
                )

                print(
                    "!" * 90
                )

                print(
                    f"Failed position: "
                    f"{position + 1}/{total}"
                )

                print(
                    f"Dataset index: "
                    f"{original_index}"
                )

                print(
                    f"Gold: {gold}"
                )

                print(
                    f"Text: {text}"
                )

                print(
                    f"Error: {repr(error)}"
                )

                print(
                    f"\nCheckpoint saved at "
                    f"{len(rows)}/{total} "
                    f"completed examples."
                )

                print(
                    "Run the same command again "
                    "to resume."
                )

                raise

            # =================================================
            # PERIODIC CHECKPOINT
            # =================================================

            processed = (
                position + 1
            )

            should_save_checkpoint = (
                processed == total
                or (
                    progress_every
                    and processed
                    % progress_every
                    == 0
                )
            )

            if should_save_checkpoint:

                self._save_checkpoint(
                    checkpoint_path=
                        checkpoint_path,

                    dataset_name=
                        dataset_name,

                    dataset_fingerprint=
                        dataset_fingerprint,

                    total_examples=
                        total,

                    rows=
                        rows
                )

                print(
                    f"[{processed}/{total}] "
                    f"completed "
                    f"(checkpoint saved)"
                )

        # =====================================================
        # RESULT DATAFRAME
        # =====================================================

        result_df = pd.DataFrame(
            rows
        )

        # =====================================================
        # METRICS
        # =====================================================

        gold = result_df[
            "gold"
        ].tolist()

        hisnet_predictions = result_df[
            "hisnet_prediction"
        ].tolist()

        dependency_predictions = result_df[
            "dependency_prediction"
        ].tolist()

        hisnet_metrics = (
            self._calculate_metrics(
                gold,
                hisnet_predictions
            )
        )

        dependency_metrics = (
            self._calculate_metrics(
                gold,
                dependency_predictions
            )
        )

        # =====================================================
        # COMPARISON STATISTICS
        # =====================================================

        comparison = {

            "total_examples":
                total,

            "prediction_changed":
                int(
                    result_df[
                        "prediction_changed"
                    ].sum()
                ),

            "improved":
                int(
                    result_df[
                        "improved"
                    ].sum()
                ),

            "degraded":
                int(
                    result_df[
                        "degraded"
                    ].sum()
                ),

            "changed_but_still_wrong":
                int(
                    result_df[
                        "changed_but_still_wrong"
                    ].sum()
                ),

            "negation_examples":
                int(
                    result_df[
                        "has_negation"
                    ].sum()
                ),

            "modifier_examples":
                int(
                    result_df[
                        "has_modifier"
                    ].sum()
                ),

            "contrast_examples":
                int(
                    result_df[
                        "has_contrast"
                    ].sum()
                ),

            "suppression_examples":
                int(
                    result_df[
                        "has_suppression"
                    ].sum()
                ),

            "mean_lexicon_coverage":
                float(
                    result_df[
                        "lexicon_coverage"
                    ].mean()
                ),

            "mean_sentiment_coverage":
                float(
                    result_df[
                        "sentiment_coverage"
                    ].mean()
                )
        }

        # =====================================================
        # OPERATION COUNTS
        # =====================================================

        operation_counter = Counter()

        for operations_json in result_df[
            "operations"
        ]:

            operations = json.loads(
                operations_json
            )

            operation_counter.update(
                operations
            )

        comparison[
            "operation_counts"
        ] = dict(
            operation_counter
        )

        # =====================================================
        # SAVE PER-EXAMPLE RESULTS
        # =====================================================

        result_path = (
            output_directory
            / f"{dataset_name}_predictions.csv"
        )

        result_df.to_csv(
            result_path,
            sep=";",
            index=False,
            encoding="utf-8-sig"
        )

        # =====================================================
        # SAVE SUMMARY
        # =====================================================

        summary = {
            "dataset":
                dataset_name,

            "labels":
                LABELS,

            "hisnet":
                hisnet_metrics,

            "hisnet_dependency":
                dependency_metrics,

            "comparison":
                comparison
        }

        summary_path = (
            output_directory
            / f"{dataset_name}_summary.json"
        )

        with open(
            summary_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                summary,
                file,
                ensure_ascii=False,
                indent=2
            )

        # =====================================================
        # CONFUSION MATRICES
        # =====================================================

        hisnet_cm = pd.DataFrame(
            hisnet_metrics[
                "confusion_matrix"
            ],
            index=[
                f"gold_{label}"
                for label in LABELS
            ],
            columns=[
                f"pred_{label}"
                for label in LABELS
            ]
        )

        dependency_cm = pd.DataFrame(
            dependency_metrics[
                "confusion_matrix"
            ],
            index=[
                f"gold_{label}"
                for label in LABELS
            ],
            columns=[
                f"pred_{label}"
                for label in LABELS
            ]
        )

        hisnet_cm.to_csv(
            output_directory
            / f"{dataset_name}_hisnet_confusion.csv",
            encoding="utf-8-sig"
        )

        dependency_cm.to_csv(
            output_directory
            / f"{dataset_name}_dependency_confusion.csv",
            encoding="utf-8-sig"
        )

        # =====================================================
        # PRINT SUMMARY
        # =====================================================

        self._print_summary(
            dataset_name,
            hisnet_metrics,
            dependency_metrics,
            comparison
        )

        # =====================================================
        # REMOVE CHECKPOINT ONLY AFTER EVERYTHING SUCCEEDED
        # =====================================================

        if checkpoint_path.exists():

            checkpoint_path.unlink()

            print(
                "\nEvaluation completed successfully."
            )

            print(
                "Checkpoint removed:"
            )

            print(
                checkpoint_path
            )

        return result_df, summary

    # =========================================================
    # PRINT SUMMARY
    # =========================================================

    @staticmethod
    def _print_summary(
        dataset_name,
        hisnet_metrics,
        dependency_metrics,
        comparison
    ):

        print(
            "\n"
            + "=" * 90
        )

        print(
            f"EVALUATION SUMMARY: {dataset_name}"
        )

        print(
            "=" * 90
        )

        print(
            f"{'Metric':<25}"
            f"{'HisNet':>15}"
            f"{'HisNet+Dep':>15}"
        )

        print(
            "-" * 55
        )

        print(
            f"{'Accuracy':<25}"
            f"{hisnet_metrics['accuracy']:>15.4f}"
            f"{dependency_metrics['accuracy']:>15.4f}"
        )

        print(
            f"{'Macro Precision':<25}"
            f"{hisnet_metrics['macro_precision']:>15.4f}"
            f"{dependency_metrics['macro_precision']:>15.4f}"
        )

        print(
            f"{'Macro Recall':<25}"
            f"{hisnet_metrics['macro_recall']:>15.4f}"
            f"{dependency_metrics['macro_recall']:>15.4f}"
        )

        print(
            f"{'Macro F1':<25}"
            f"{hisnet_metrics['macro_f1']:>15.4f}"
            f"{dependency_metrics['macro_f1']:>15.4f}"
        )

        print(
            "\nDEPENDENCY EFFECT"
        )

        print(
            "-" * 55
        )

        print(
            "Prediction changed:",
            comparison[
                "prediction_changed"
            ]
        )

        print(
            "Improved:",
            comparison[
                "improved"
            ]
        )

        print(
            "Degraded:",
            comparison[
                "degraded"
            ]
        )

        print(
            "Changed but still wrong:",
            comparison[
                "changed_but_still_wrong"
            ]
        )

        print(
            "\nLINGUISTIC EVENTS"
        )

        print(
            "-" * 55
        )

        print(
            "Negation examples:",
            comparison[
                "negation_examples"
            ]
        )

        print(
            "Modifier examples:",
            comparison[
                "modifier_examples"
            ]
        )

        print(
            "Contrast examples:",
            comparison[
                "contrast_examples"
            ]
        )

        print(
            "Suppression examples:",
            comparison[
                "suppression_examples"
            ]
        )

        print(
            "\nCOVERAGE"
        )

        print(
            "-" * 55
        )

        print(
            "Mean lexicon coverage:",
            f"{comparison['mean_lexicon_coverage']:.4f}"
        )

        print(
            "Mean sentiment coverage:",
            f"{comparison['mean_sentiment_coverage']:.4f}"
        )

        print(
            "\nOperation counts:",
            comparison[
                "operation_counts"
            ]
        )