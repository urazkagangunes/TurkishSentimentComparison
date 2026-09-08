from src.evaluation.dataset_loader import (
    load_sentiment_dataset
)

from src.evaluation.sentiment_evaluator import (
    SentimentEvaluator
)


DATASET_PATH = (
    "datasets/Modified/"
    "sentence_sentiment/"
    "beyazperde_sentiment.csv"
)


# =============================================================
# LOAD
# =============================================================

dataset = load_sentiment_dataset(
    file_path=DATASET_PATH,
    text_column="Review",
    label_column="Score",
    separator=";",
    sample_per_class=None,
    random_state=42
)


print("\nDataset loaded.")
print("Total examples:", len(dataset))

print("\nClass distribution:")
print(
    dataset["gold"].value_counts()
)


# =============================================================
# EVALUATE
# =============================================================

evaluator = SentimentEvaluator()

results, summary = evaluator.evaluate(
    dataframe=dataset,
    dataset_name="beyazperde_full",
    output_directory="results/beyazperde",
    progress_every=100,
    resume=True
)