"""
SyntheticRows — dataset usability guard.

check_dataset_usable(df, text_columns) -> (ok: bool, reason: str | None)

Refuses ONLY genuinely impossible datasets (nothing to learn from).
Datasets that are merely hard — tiny (2+ rows), imbalanced, zero-inflated,
all-text-with-a-usable-signal, or a mix of junk and usable columns — are
allowed through to the normal pipeline.

Refusal floor for row count is 2, matching bootstrap_tiny_dataset: with a
single row, pandas std() is NaN so bootstrap can only copy that row, which
would produce identical synthetic rows (garbage).
"""
import pandas as pd

_HINT = ("Tip: SyntheticRows works best with columns that have repeating values — "
         "categories, numbers, or labels.")


def check_dataset_usable(df: pd.DataFrame, text_columns: list | None = None):
    text_columns = text_columns or []
    n_rows = len(df)
    n_cols = df.shape[1]

    # 1. Too few rows to learn any distribution (bootstrap needs >= 2).
    if n_rows < 2:
        return False, (
            "This dataset has only one row. SyntheticRows needs at least a couple of "
            f"rows to learn patterns from. {_HINT}"
        )

    # 2. A single column that is entirely constant — nothing variable to model.
    if n_cols == 1:
        only_col = df.columns[0]
        if df[only_col].nunique(dropna=False) <= 1:
            return False, (
                "This dataset has a single column with the same value in every row, "
                f"so there's nothing variable to synthesize. {_HINT}"
            )

    # 3. Every column is all-unique (IDs / free text) AND there's no text column
    #    to fall back on. If text columns exist, the dataset is usable on the
    #    text-augmentation path, so we must NOT refuse it here.
    #    Only meaningful when there are enough rows that uniqueness is a real
    #    signal — in a tiny dataset every column is trivially all-unique, and
    #    bootstrap handles those fine, so we skip this check below 10 rows.
    if not text_columns and n_rows >= 10:
        all_unique = all(
            df[c].nunique(dropna=False) == n_rows for c in df.columns
        )
        if all_unique:
            return False, (
                "Every column in this dataset has all-unique values (like IDs or "
                "free text), so there's no recurring pattern to learn from. "
                f"{_HINT}"
            )

    return True, None