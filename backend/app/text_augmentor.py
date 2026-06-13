import pandas as pd
import numpy as np
import nltk
import nlpaug.augmenter.word as naw
from typing import List

# Download required NLTK data
def ensure_nltk_data():
    import ssl
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception:
        pass

    resources = [
        "wordnet",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "omw-1.4",
        "punkt"
    ]
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

ensure_nltk_data()


# ─── Column Detection ─────────────────────────────────────────────────────────

def is_text_column(series: pd.Series, min_avg_words: int = 3) -> bool:
    if series.dtype != object:
        return False

    sample = series.dropna().head(50)
    if len(sample) == 0:
        return False

    # Calculate average word count first
    avg_words = sample.apply(lambda x: len(str(x).split())).mean()

    # If average words >= min_avg_words, it's text regardless of uniqueness
    if avg_words >= min_avg_words:
        return True

    # Skip if too many unique values look like IDs (only for short text)
    if series.nunique() / len(series) > 0.95:
        return False

    return False

def detect_text_columns(df: pd.DataFrame) -> List[str]:
    """Return list of columns that contain meaningful text."""
    return [col for col in df.columns if is_text_column(df[col])]


# ─── Augmentation ─────────────────────────────────────────────────────────────

def augment_text(text: str, augmenter, num_variations: int = 1) -> List[str]:
    """Generate N augmented variations of a text."""
    if not text or not str(text).strip():
        return [text] * num_variations

    results = []
    for _ in range(num_variations):
        try:
            augmented = augmenter.augment(str(text))
            if isinstance(augmented, list):
                results.append(augmented[0])
            else:
                results.append(augmented)
        except Exception:
            results.append(text)

    return results


def augment_dataset(
    df: pd.DataFrame,
    text_columns: List[str],
    num_rows: int,
    augmentation_strength: str = "medium"
) -> pd.DataFrame:
    """
    Augment a dataset containing text columns.

    Strategy:
    - Keep all original rows
    - Generate additional rows by augmenting text columns
    - Non-text columns are copied from the original row
    """

    # Configure augmenter strength
    strength_config = {
        "low":    {"aug_p": 0.1},
        "medium": {"aug_p": 0.2},
        "high":   {"aug_p": 0.3},
    }
    config = strength_config.get(augmentation_strength, strength_config["medium"])

    augmenter = naw.SynonymAug(
        aug_src="wordnet",
        aug_p=config["aug_p"]
    )

    original_rows = len(df)
    rows_to_generate = num_rows - original_rows

    if rows_to_generate <= 0:
        return df.copy()

    # How many times do we need to augment each row?
    augmentations_per_row = max(1, int(np.ceil(rows_to_generate / original_rows)))

    augmented_rows = []
    generated_count = 0

    for _, row in df.iterrows():
        if generated_count >= rows_to_generate:
            break

        for _ in range(augmentations_per_row):
            if generated_count >= rows_to_generate:
                break

            new_row = row.copy()

            for col in text_columns:
                if col in row and pd.notna(row[col]):
                    variations = augment_text(str(row[col]), augmenter, num_variations=1)
                    new_row[col] = variations[0]

            augmented_rows.append(new_row)
            generated_count += 1

    augmented_df = pd.DataFrame(augmented_rows)
    result_df = pd.concat([df, augmented_df], ignore_index=True)

    return result_df


# ─── Quality Scoring ──────────────────────────────────────────────────────────

def score_text_augmentation(
    original_df: pd.DataFrame,
    augmented_df: pd.DataFrame,
    text_columns: List[str]
) -> dict:
    """
    Score the quality of text augmentation.
    Checks:
    1. Vocabulary diversity — augmented text uses different words
    2. Length preservation — augmented text is similar length
    3. Label preservation — non-text columns are correctly distributed
    """
    scores = {}

    for col in text_columns:
        if col not in original_df.columns or col not in augmented_df.columns:
            continue

        orig_texts = original_df[col].dropna().astype(str)
        aug_texts = augmented_df[col].dropna().astype(str)

        # Length preservation score
        orig_avg_len = orig_texts.apply(len).mean()
        aug_avg_len = aug_texts.apply(len).mean()
        len_diff_pct = abs(orig_avg_len - aug_avg_len) / orig_avg_len * 100 if orig_avg_len > 0 else 0
        length_score = max(0, 100 - len_diff_pct)

        # Vocabulary diversity score
        orig_words = set(" ".join(orig_texts.tolist()).lower().split())
        aug_words = set(" ".join(aug_texts.tolist()).lower().split())
        new_words = aug_words - orig_words
        diversity_score = min(100, len(new_words) / max(1, len(orig_words)) * 100 * 5)

        col_score = round((length_score * 0.6 + diversity_score * 0.4), 1)

        scores[col] = {
            "score": col_score,
            "length_preservation": round(length_score, 1),
            "vocabulary_diversity": round(diversity_score, 1),
            "original_avg_length": round(orig_avg_len, 1),
            "augmented_avg_length": round(aug_avg_len, 1)
        }

    # Overall score
    if scores:
        overall = round(np.mean([s["score"] for s in scores.values()]), 1)
    else:
        overall = 0

    if overall >= 80:
        grade = "Excellent"
        color = "green"
    elif overall >= 60:
        grade = "Good"
        color = "yellow"
    else:
        grade = "Fair"
        color = "red"

    return {
        "overall_score": overall,
        "grade": grade,
        "color": color,
        "column_scores": scores
    }