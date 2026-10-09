import json
import os
import time
from typing import List, Optional

import httpx
import ollama


class OllamaZeroShotSentimentClassifier:
    """
    Zero-shot Turkish sentiment classifier using Ollama.

    Experimental configuration
    --------------------------
    Task:
        Turkish 3-class sentiment classification

    Labels:
        positive
        negative
        neutral

    Prompting:
        zero-shot

    Inference:
        batched

    Thinking:
        disabled

    Final intended model:
        qwen3:32b

    Development model:
        qwen3:4b-instruct

    Important experimental principles
    ---------------------------------
    1. No labelled examples are included in the prompt.
    2. Reviews inside a batch are instructed to be classified
       independently.
    3. Technical failures are NEVER converted to "neutral".
    4. Failed batches do NOT stop the experiment.
    5. Failed reviews can be recovered individually.
    6. A fixed random seed is used for reproducibility.
    """

    LABELS = {
        "positive",
        "negative",
        "neutral"
    }

    def __init__(
        self,
        model: Optional[str] = None,
        batch_size: int = 8,
        host: str = "http://localhost:11434",
        temperature: float = 0.1,
        top_p: float = 1.0,
        num_ctx: int = 4096,
        keep_alive: str = "30m",
        max_retries: int = 2,
        seed: int = 42
    ):

        # =====================================================
        # MODEL
        #
        # PowerShell development:
        #
        # $env:OLLAMA_MODEL="qwen3:4b-instruct"
        #
        # Final experiment:
        #
        # $env:OLLAMA_MODEL="qwen3:32b"
        # =====================================================

        self.model = (
            model
            or os.getenv(
                "OLLAMA_MODEL",
                "qwen3:32b"
            )
        )

        self.batch_size = batch_size

        self.temperature = temperature
        self.top_p = top_p
        self.num_ctx = num_ctx

        self.keep_alive = keep_alive
        self.max_retries = max_retries

        # Fixed seed for experimental reproducibility.
        self.seed = seed

        self.client = ollama.Client(
            host=host
        )

        # =====================================================
        # RUNTIME STATISTICS
        # =====================================================

        self.total_requests = 0
        self.total_reviews = 0

        self.total_retries = 0

        # Number of calls that ultimately failed
        # after all retry attempts.
        self.total_failures = 0

        self.total_wall_seconds = 0.0

        self.total_prompt_tokens = 0
        self.total_output_tokens = 0

    # =========================================================
    # PROMPT
    # =========================================================

    @staticmethod
    def _build_prompt(
        texts: List[str]
    ) -> str:
        """
        Build one zero-shot batch prompt.

        No labelled examples are included.
        Therefore the experiment remains zero-shot.
        """

        review_blocks = []

        for index, text in enumerate(
            texts,
            start=1
        ):

            review_blocks.append(
                (
                    f"[REVIEW {index}]\n"
                    f"{text}\n"
                    f"[/REVIEW {index}]"
                )
            )

        reviews = "\n\n".join(
            review_blocks
        )

        return (
            "Aşağıda Türkçe kullanıcı yorumları verilmiştir.\n\n"

            "Görevin, HER yorumu diğer yorumlardan TAMAMEN "
            "BAĞIMSIZ olarak genel duygu açısından "
            "sınıflandırmaktır.\n\n"

            "Her yorum için yalnızca şu üç etiketten birini seç:\n"
            "- positive\n"
            "- negative\n"
            "- neutral\n\n"

            "Bir yorumun sınıflandırılmasında diğer yorumların "
            "içeriğini kullanma.\n\n"

            "Her REVIEW için tam olarak bir sonuç üret.\n"

            "Ek açıklama, gerekçe veya yorum yazma.\n\n"

            f"{reviews}"
        )

    # =========================================================
    # STRUCTURED OUTPUT SCHEMA
    # =========================================================

    @staticmethod
    def _build_output_schema(
        batch_size: int
    ) -> dict:
        """
        Require exactly one prediction for every review.
        """

        return {
            "type": "object",

            "properties": {

                "predictions": {

                    "type": "array",

                    "minItems":
                        batch_size,

                    "maxItems":
                        batch_size,

                    "items": {

                        "type": "object",

                        "properties": {

                            "id": {
                                "type":
                                    "integer"
                            },

                            "label": {

                                "type":
                                    "string",

                                "enum": [
                                    "positive",
                                    "negative",
                                    "neutral"
                                ]
                            }
                        },

                        "required": [
                            "id",
                            "label"
                        ],

                        "additionalProperties":
                            False
                    }
                }
            },

            "required": [
                "predictions"
            ],

            "additionalProperties":
                False
        }

    # =========================================================
    # OUTPUT VALIDATION
    # =========================================================

    def _validate_predictions(
        self,
        raw_predictions,
        expected_count: int
    ) -> List[str]:
        """
        Validate:
            - prediction list exists
            - correct number of predictions
            - IDs are unique
            - IDs are exactly 1..N
            - labels are valid

        Returns predictions in original review order.
        """

        if not isinstance(
            raw_predictions,
            list
        ):

            raise ValueError(
                "predictions must be a list."
            )

        if len(
            raw_predictions
        ) != expected_count:

            raise ValueError(
                "Prediction count mismatch: "
                f"expected={expected_count}, "
                f"got={len(raw_predictions)}"
            )

        prediction_by_id = {}

        for item in raw_predictions:

            if not isinstance(
                item,
                dict
            ):

                raise ValueError(
                    "Prediction item is not an object."
                )

            prediction_id = (
                item.get(
                    "id"
                )
            )

            label = (
                str(
                    item.get(
                        "label",
                        ""
                    )
                )
                .strip()
                .lower()
            )

            if prediction_id in prediction_by_id:

                raise ValueError(
                    "Duplicate prediction ID: "
                    f"{prediction_id}"
                )

            if label not in self.LABELS:

                raise ValueError(
                    "Invalid sentiment label: "
                    f"{label}"
                )

            prediction_by_id[
                prediction_id
            ] = label

        expected_ids = set(
            range(
                1,
                expected_count + 1
            )
        )

        actual_ids = set(
            prediction_by_id.keys()
        )

        if actual_ids != expected_ids:

            raise ValueError(
                "Prediction IDs mismatch. "
                f"Expected={sorted(expected_ids)}, "
                f"Got={sorted(actual_ids)}"
            )

        # Return predictions in exact input order.
        return [
            prediction_by_id[index]

            for index in range(
                1,
                expected_count + 1
            )
        ]

    # =========================================================
    # CLASSIFY ONE BATCH
    # =========================================================

    def classify_batch(
        self,
        texts: List[str]
    ) -> dict:
        """
        Classify one batch.

        If all retry attempts fail:
            - experiment does NOT crash
            - prediction is NOT changed to neutral
            - None predictions are returned
            - status="failed"
        """

        # =====================================================
        # EMPTY INPUT
        # =====================================================

        if not texts:

            return {
                "predictions":
                    [],

                "model":
                    self.model,

                "batch_size":
                    0,

                "request_seconds":
                    0.0,

                "prompt_tokens":
                    0,

                "output_tokens":
                    0,

                "attempt":
                    0,

                "status":
                    "success",

                "error":
                    None,

                "raw_response":
                    None
            }

        # =====================================================
        # BATCH SIZE CHECK
        # =====================================================

        if len(texts) > self.batch_size:

            raise ValueError(
                "classify_batch received more reviews "
                "than configured batch_size. "
                f"batch_size={self.batch_size}, "
                f"received={len(texts)}"
            )

        prompt = (
            self._build_prompt(
                texts
            )
        )

        output_schema = (
            self._build_output_schema(
                len(texts)
            )
        )

        last_error = None

        failed_attempt_seconds = 0.0

        # =====================================================
        # FIRST ATTEMPT + RETRIES
        #
        # max_retries=2:
        #
        # Attempt 1
        # Retry 1
        # Retry 2
        #
        # Total = 3 attempts
        # =====================================================

        for attempt in range(
            self.max_retries + 1
        ):

            if attempt > 0:

                self.total_retries += 1

                print(
                    "[LLM RETRY] "
                    f"{attempt}/{self.max_retries}"
                )

            start_time = (
                time.perf_counter()
            )

            try:

                self.total_requests += 1

                response = (
                    self.client.chat(

                        model=
                            self.model,

                        messages=[
                            {
                                "role":
                                    "user",

                                "content":
                                    prompt
                            }
                        ],

                        # -------------------------------------
                        # Qwen3 reasoning disabled.
                        # -------------------------------------

                        think=False,

                        # -------------------------------------
                        # Ollama structured JSON output.
                        # -------------------------------------

                        format=
                            output_schema,

                        stream=False,

                        keep_alive=
                            self.keep_alive,

                        options={

                            "temperature":
                                self.temperature,

                            "top_p":
                                self.top_p,

                            "num_ctx":
                                self.num_ctx,

                            # Fixed seed for reproducibility.
                            "seed":
                                self.seed
                        }
                    )
                )

                request_seconds = (
                    time.perf_counter()
                    - start_time
                )

                # =================================================
                # RESPONSE CONTENT
                # =================================================

                content = (
                    response.message.content
                )

                # =================================================
                # JSON PARSE
                # =================================================

                parsed = (
                    json.loads(
                        content
                    )
                )

                # =================================================
                # STRUCTURAL VALIDATION
                # =================================================

                predictions = (
                    self._validate_predictions(

                        parsed.get(
                            "predictions"
                        ),

                        len(texts)
                    )
                )

                # =================================================
                # TOKEN DIAGNOSTICS
                # =================================================

                prompt_eval_count = (
                    getattr(
                        response,
                        "prompt_eval_count",
                        None
                    )
                    or 0
                )

                eval_count = (
                    getattr(
                        response,
                        "eval_count",
                        None
                    )
                    or 0
                )

                # =================================================
                # RUNTIME STATISTICS
                # =================================================

                self.total_wall_seconds += (
                    request_seconds
                )

                self.total_reviews += (
                    len(texts)
                )

                self.total_prompt_tokens += (
                    prompt_eval_count
                )

                self.total_output_tokens += (
                    eval_count
                )

                # =================================================
                # SUCCESS
                # =================================================

                return {
                    "predictions":
                        predictions,

                    "model":
                        self.model,

                    "batch_size":
                        len(texts),

                    "request_seconds":
                        request_seconds,

                    "prompt_tokens":
                        prompt_eval_count,

                    "output_tokens":
                        eval_count,

                    "attempt":
                        attempt + 1,

                    "status":
                        "success",

                    "error":
                        None,

                    "raw_response":
                        content
                }

            # =================================================
            # EXPECTED RUNTIME / OUTPUT FAILURES
            # =================================================

            except (
                json.JSONDecodeError,
                ValueError,
                ollama.ResponseError,
                httpx.HTTPError,
                OSError,
                TimeoutError
            ) as error:

                attempt_seconds = (
                    time.perf_counter()
                    - start_time
                )

                failed_attempt_seconds += (
                    attempt_seconds
                )

                last_error = error

                print(
                    "\n"
                    "[LLM WARNING] "
                    f"Batch attempt "
                    f"{attempt + 1}/"
                    f"{self.max_retries + 1} failed."
                )

                print(
                    "Error type:",
                    type(error).__name__
                )

                print(
                    "Error:",
                    str(error)
                )

        # =====================================================
        # ALL ATTEMPTS FAILED
        #
        # IMPORTANT:
        #
        # We DO NOT:
        #
        # prediction = "neutral"
        #
        # and we DO NOT:
        #
        # raise RuntimeError(...)
        #
        # The experiment continues safely.
        # =====================================================

        self.total_failures += 1

        self.total_wall_seconds += (
            failed_attempt_seconds
        )

        error_message = (
            (
                f"{type(last_error).__name__}: "
                f"{last_error}"
            )

            if last_error is not None

            else
                "Unknown LLM error"
        )

        print(
            "\n"
            "[LLM FAILED BATCH]"
        )

        print(
            "Reviews in failed batch:",
            len(texts)
        )

        print(
            "Error:",
            error_message
        )

        return {
            "predictions":
                [None] * len(texts),

            "model":
                self.model,

            "batch_size":
                len(texts),

            "request_seconds":
                failed_attempt_seconds,

            "prompt_tokens":
                0,

            "output_tokens":
                0,

            "attempt":
                self.max_retries + 1,

            "status":
                "failed",

            "error":
                error_message,

            "raw_response":
                None
        }

    # =========================================================
    # SIMPLE MANY-REVIEW CLASSIFICATION
    # =========================================================

    def classify_many(
        self,
        texts: List[str]
    ) -> List[Optional[str]]:
        """
        Split reviews into batches.

        Technical failures do not crash the experiment.

        Failed reviews are returned as None.

        For experiments, classify_many_safe() is preferred,
        because it also performs an individual recovery pass.
        """

        all_predictions = []

        total = len(
            texts
        )

        for start in range(
            0,
            total,
            self.batch_size
        ):

            end = min(
                start + self.batch_size,
                total
            )

            batch = texts[
                start:end
            ]

            result = (
                self.classify_batch(
                    batch
                )
            )

            all_predictions.extend(
                result[
                    "predictions"
                ]
            )

        return all_predictions

    # =========================================================
    # SAFE EXPERIMENT CLASSIFICATION
    # =========================================================

    def classify_many_safe(
        self,
        texts: List[str]
    ) -> List[dict]:
        """
        Robust experiment-oriented inference.

        PASS 1
        ------
        Classify normally in batches.

        If a batch fails:
            mark its reviews as failed
            but continue the experiment.

        PASS 2
        ------
        Retry each failed review individually.

        No technical error is ever converted to a sentiment
        label.

        Returns one record for every original review:

        {
            "index": 0,
            "text": "...",
            "prediction": "positive" | "negative" |
                          "neutral" | None,
            "status": "success" | "failed",
            "recovered": True | False,
            "error": None | "..."
        }
        """

        total = len(
            texts
        )

        records = []

        for index, text in enumerate(
            texts
        ):

            records.append({

                "index":
                    index,

                "text":
                    text,

                "prediction":
                    None,

                "status":
                    "pending",

                "recovered":
                    False,

                "error":
                    None
            })

        failed_indices = []

        # =====================================================
        # FIRST PASS — NORMAL BATCHING
        # =====================================================

        total_batches = (
            (
                total
                + self.batch_size
                - 1
            )
            // self.batch_size
        )

        batch_number = 0

        for start in range(
            0,
            total,
            self.batch_size
        ):

            batch_number += 1

            end = min(
                start + self.batch_size,
                total
            )

            batch = texts[
                start:end
            ]

            result = (
                self.classify_batch(
                    batch
                )
            )

            if (
                result.get(
                    "status"
                )
                == "success"
            ):

                for offset, prediction in enumerate(
                    result[
                        "predictions"
                    ]
                ):

                    global_index = (
                        start + offset
                    )

                    records[
                        global_index
                    ][
                        "prediction"
                    ] = prediction

                    records[
                        global_index
                    ][
                        "status"
                    ] = "success"

                    records[
                        global_index
                    ][
                        "error"
                    ] = None

            else:

                print(
                    "\n"
                    "[LLM FAILED BATCH]"
                )

                print(
                    "Rows:",
                    start,
                    "-",
                    end - 1
                )

                print(
                    "Error:",
                    result.get(
                        "error"
                    )
                )

                for global_index in range(
                    start,
                    end
                ):

                    records[
                        global_index
                    ][
                        "status"
                    ] = "failed"

                    records[
                        global_index
                    ][
                        "prediction"
                    ] = None

                    records[
                        global_index
                    ][
                        "error"
                    ] = result.get(
                        "error"
                    )

                    failed_indices.append(
                        global_index
                    )

            # =================================================
            # PROGRESS
            # =================================================

            print(
                "[LLM BATCH] "
                f"{batch_number}/{total_batches} "
                f"| reviews {start + 1}-{end}/{total}"
            )

        # =====================================================
        # INDIVIDUAL RECOVERY PASS
        # =====================================================

        if failed_indices:

            print(
                "\n"
                + "=" * 100
            )

            print(
                "LLM INDIVIDUAL RECOVERY PASS"
            )

            print(
                "Failed reviews:",
                len(
                    failed_indices
                )
            )

            print(
                "=" * 100
            )

        for recovery_number, index in enumerate(
            failed_indices,
            start=1
        ):

            text = texts[
                index
            ]

            result = (
                self.classify_batch(
                    [text]
                )
            )

            if (
                result.get(
                    "status"
                )
                == "success"
            ):

                prediction = (
                    result[
                        "predictions"
                    ][0]
                )

                records[
                    index
                ][
                    "prediction"
                ] = prediction

                records[
                    index
                ][
                    "status"
                ] = "success"

                records[
                    index
                ][
                    "recovered"
                ] = True

                records[
                    index
                ][
                    "error"
                ] = None

            else:

                # Individual recovery also failed.
                #
                # Still do NOT invent a label.

                records[
                    index
                ][
                    "prediction"
                ] = None

                records[
                    index
                ][
                    "status"
                ] = "failed"

                records[
                    index
                ][
                    "recovered"
                ] = False

                records[
                    index
                ][
                    "error"
                ] = result.get(
                    "error"
                )

            print(
                "[LLM RECOVERY] "
                f"{recovery_number}/"
                f"{len(failed_indices)}"
            )

        # =====================================================
        # FINAL SUMMARY
        # =====================================================

        successful_count = sum(
            record[
                "status"
            ]
            == "success"

            for record in records
        )

        failed_count = sum(
            record[
                "status"
            ]
            == "failed"

            for record in records
        )

        recovered_count = sum(
            record[
                "recovered"
            ]

            for record in records
        )

        print(
            "\n"
            + "=" * 100
        )

        print(
            "LLM SAFE CLASSIFICATION SUMMARY"
        )

        print(
            "=" * 100
        )

        print(
            "Total reviews:",
            total
        )

        print(
            "Successful:",
            successful_count
        )

        print(
            "Failed:",
            failed_count
        )

        print(
            "Recovered:",
            recovered_count
        )

        print(
            "=" * 100
        )

        return records

    # =========================================================
    # SINGLE REVIEW
    # =========================================================

    def classify(
        self,
        text: str
    ) -> dict:
        """
        Convenience method for one review.

        Technical failure:

            prediction=None
            status="failed"

        The experiment does not crash.
        """

        result = (
            self.classify_batch(
                [text]
            )
        )

        prediction = (
            result[
                "predictions"
            ][0]
        )

        return {
            "text":
                text,

            "prediction":
                prediction,

            "status":
                result.get(
                    "status"
                ),

            "error":
                result.get(
                    "error"
                ),

            "model":
                result[
                    "model"
                ],

            "request_seconds":
                result[
                    "request_seconds"
                ],

            "attempt":
                result.get(
                    "attempt"
                )
        }

    # =========================================================
    # RUNTIME STATISTICS
    # =========================================================

    def get_stats(
        self
    ) -> dict:

        reviews_per_second = (
            self.total_reviews
            / self.total_wall_seconds

            if self.total_wall_seconds > 0

            else 0.0
        )

        seconds_per_review = (
            self.total_wall_seconds
            / self.total_reviews

            if self.total_reviews > 0

            else 0.0
        )

        return {
            "model":
                self.model,

            "configured_batch_size":
                self.batch_size,

            "temperature":
                self.temperature,

            "top_p":
                self.top_p,

            "num_ctx":
                self.num_ctx,

            "seed":
                self.seed,

            "max_retries":
                self.max_retries,

            "total_requests":
                self.total_requests,

            "total_reviews":
                self.total_reviews,

            "total_retries":
                self.total_retries,

            "total_failures":
                self.total_failures,

            "total_wall_seconds":
                self.total_wall_seconds,

            "reviews_per_second":
                reviews_per_second,

            "seconds_per_review":
                seconds_per_review,

            "total_prompt_tokens":
                self.total_prompt_tokens,

            "total_output_tokens":
                self.total_output_tokens
        }