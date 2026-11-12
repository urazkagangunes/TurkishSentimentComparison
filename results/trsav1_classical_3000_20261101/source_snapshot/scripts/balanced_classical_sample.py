"""Deterministic, balanced TRSAv1 development samples disjoint by review text."""
from pathlib import Path

import pandas as pd

from scripts.run_six_systems import digest
from src.evaluation.dataset_loader import load_sentiment_dataset
from tests.run_trsav1_classical_1200 import SOURCE


def load_disjoint_sample(per_class, exclude_paths, source=SOURCE, seed=42):
    if per_class < 1:
        raise ValueError("sample_per_class must be positive")
    dataset = load_sentiment_dataset(source, "review", "score")
    excluded, exclusions = set(), []
    for path in exclude_paths:
        path = Path(path).resolve()
        old = pd.read_csv(path, encoding="utf-8-sig")
        if "text" not in old or old["text"].isna().any():
            raise ValueError(f"Exclusion sample requires nonmissing text: {path}")
        texts = set(old["text"].astype(str).str.strip())
        excluded.update(texts)
        exclusions.append(dict(path=str(path), sha256=digest(path), rows=len(old), unique_texts=len(texts)))
    eligible = dataset.loc[~dataset["text"].isin(excluded)]
    chosen, used, skipped = [], set(), []
    # Each class has a seeded random permutation. Conflicting texts retain
    # their source label; a text already selected in an earlier class is
    # skipped, never relabelled. Class order and skipped IDs are recorded.
    for label in ("negative", "neutral", "positive"):
        group = eligible.loc[eligible["gold"] == label].sample(frac=1, random_state=seed)
        selected = []
        for index, text in group["text"].items():
            if text in used:
                skipped.append(int(index))
                continue
            selected.append(index)
            used.add(text)
            if len(selected) == per_class:
                break
        if len(selected) != per_class:
            raise ValueError(f"Not enough eligible unique texts in {label}: {len(selected)} < {per_class}")
        chosen.extend(selected)
    sample = dataset.loc[chosen].sample(frac=1, random_state=seed)
    source_ids = sample.index.tolist()
    sample = sample.reset_index(drop=True)
    conflicts = dataset.groupby("text")["gold"].nunique()
    conflicts = set(conflicts[conflicts > 1].index)
    info = dict(
        origin="exclude_exact_stripped_text_then_classwise_seeded_unique_sampling_then_seeded_shuffle",
        random_state=seed, sample_size=len(sample), source=str(Path(source).resolve()),
        source_sha256=digest(source), dataset_size=len(dataset),
        dataset_distribution=dataset["gold"].value_counts().to_dict(),
        eligible_size=len(eligible), excluded_source_rows=len(dataset)-len(eligible),
        exclusions=exclusions, excluded_unique_texts=len(excluded), overlap_with_exclusions=0,
        class_selection_order=["negative", "neutral", "positive"],
        skipped_cross_label_duplicate_source_ids=skipped,
        sample_distribution=sample["gold"].value_counts().to_dict(), matches_seed42_reference=seed == 42,
        reference_protocol="this disjoint sampling protocol, not the original 1200-row draw",
        conflicting_source_label_row_ids=[i for i, text in enumerate(sample["text"]) if text in conflicts],
        sample_conflicting_texts=0, source_cleaned_row_ids=source_ids,
        purpose="development_diagnostic_not_independent_final_test",
    )
    if sample["text"].duplicated().any() or set(sample["text"]) & excluded:
        raise AssertionError("Sample texts must be unique and disjoint from exclusions")
    return sample, sample.to_dict("records"), info
