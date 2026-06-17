import pandas as pd
import numpy as np
from typing import List, Tuple
from .models import Fix, ColumnType, DatasetProfile


def _safe_int(series: pd.Series, fill_value: int = 0) -> pd.Series:
    """
    Round and cast a numeric Series to int, safely handling NaN/inf.

    A plain .astype(int) raises IntCastingNaNError if the column contains any
    NaN or inf (which can happen when a column was all-empty, or when the
    synthesizer produced non-finite values). This replaces inf with NaN, fills
    NaN with the column median (or a fallback), then rounds and casts — so the
    conversion can never crash the generation pipeline.
    """
    s = series.replace([np.inf, -np.inf], np.nan)
    if s.notna().any():
        fill = s.median()
    else:
        fill = fill_value
    if pd.isna(fill):
        fill = fill_value
    return s.fillna(fill).round().astype(int)


def detect_skewed_columns(df: pd.DataFrame, threshold: float = 1.0) -> List[str]:
    """
    Detect numerical columns with high skewness that benefit from a log transform.
    Skips binary/low-cardinality columns (e.g. 0/1 labels and flags): a log
    transform is only meaningful for continuous quantities, and transforming a
    label corrupts its values (1 -> ln(2)), breaking class matching downstream.
    """
    skewed = []
    for col in df.select_dtypes(include=[np.number]).columns:
        # Binary or near-constant columns are never "skewed" in a useful sense.
        if df[col].nunique() <= 2:
            continue
        skewness = abs(df[col].skew())
        if skewness > threshold and df[col].min() >= 0:
            skewed.append(col)
    return skewed


def get_integer_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns that should be integers."""
    int_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        if non_null.apply(lambda x: x == int(x)).all():
            int_cols.append(col)
    return int_cols


def restore_dtypes(synthetic_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    synthetic_df = synthetic_df.copy()
    for col in original_df.select_dtypes(include=[np.number]).columns:
        if col not in synthetic_df.columns:
            continue

        col_min = original_df[col].min()
        col_max = original_df[col].max()

        # Clip to original range
        synthetic_df[col] = synthetic_df[col].clip(lower=col_min, upper=col_max)

        # Determine whether the original column is integer-like
        non_null = original_df[col].dropna()
        is_int_like = pd.api.types.is_integer_dtype(original_df[col]) or (
            len(non_null) > 0 and non_null.apply(lambda x: x == int(x)).all()
        )

        if is_int_like:
            synthetic_df[col] = _safe_int(synthetic_df[col])
        else:
            # Round floats to same decimal places as original
            sample_vals = original_df[col].dropna().head(100)
            if len(sample_vals) == 0:
                continue
            avg_decimals = sample_vals.apply(
                lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0
            ).median()
            decimal_places = min(int(avg_decimals), 4) if not pd.isna(avg_decimals) else 2
            synthetic_df[col] = synthetic_df[col].round(decimal_places)

    return synthetic_df


def log_transform(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Apply log1p transform to skewed columns."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    return df


def reverse_log_transform(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Reverse log1p transform after generation."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = np.expm1(df[col])
    return df


def apply_fixes(df: pd.DataFrame, fixes: List[Fix]) -> pd.DataFrame:
    df = df.copy()

    for fix in fixes:
        if not fix.approved:
            continue

        col = fix.column
        if col not in df.columns:
            continue

        if fix.fix_type == "fill_missing":
            if pd.api.types.is_numeric_dtype(df[col]):
                median = df[col].median()
                # If the whole column is empty there is no median — drop it,
                # since an all-NaN column can't be filled or synthesized.
                if pd.isna(median):
                    df = df.drop(columns=[col])
                else:
                    df[col] = df[col].fillna(median)
            else:
                mode = df[col].mode()
                if len(mode) > 0:
                    df[col] = df[col].fillna(mode[0])
                else:
                    df = df.drop(columns=[col])

        elif fix.fix_type == "fix_zeros":
            median_val = df[col].replace(0, np.nan).median()
            # If every value was zero there is no non-zero median — leave the
            # column as-is rather than turning it entirely into NaN.
            if not pd.isna(median_val):
                df[col] = df[col].replace(0, median_val)

        elif fix.fix_type == "cap_outliers":
            # IQR-based capping — robust to extreme values (mean/std gets
            # corrupted by the very outliers we're trying to cap)
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if not pd.isna(iqr) and iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        elif fix.fix_type == "drop_column":
            df = df.drop(columns=[col])

    return df


def auto_preprocess(df: pd.DataFrame, profile: DatasetProfile) -> pd.DataFrame:
    df = df.copy()

    for col_profile in profile.columns:
        col = col_profile.name

        if col not in df.columns:
            continue

        col_type = col_profile.col_type

        # Always drop ID columns and high cardinality columns
        if col_type in [ColumnType.ID_COLUMN, ColumnType.CATEGORICAL_HIGH]:
            df = df.drop(columns=[col])
            continue

        # Drop any column that is entirely empty — nothing to synthesize, and
        # it would crash downstream integer/dtype handling.
        if df[col].isna().all():
            df = df.drop(columns=[col])
            continue

        # Handle datetime columns — extract useful features
        if col_type == ColumnType.DATETIME:
            try:
                parsed = pd.to_datetime(df[col])
                df[f"{col}_year"] = parsed.dt.year
                df[f"{col}_month"] = parsed.dt.month
                df[f"{col}_day"] = parsed.dt.day
                df[f"{col}_dayofweek"] = parsed.dt.dayofweek
                df = df.drop(columns=[col])
            except Exception:
                df = df.drop(columns=[col])
            continue

        # Fill missing values based on column type
        if col_profile.num_missing > 0:
            if col_type in [ColumnType.NUMERICAL_CONTINUOUS, ColumnType.NUMERICAL_DISCRETE]:
                median = df[col].median()
                if pd.isna(median):
                    df = df.drop(columns=[col])
                    continue
                df[col] = df[col].fillna(median)
            else:
                mode = df[col].mode()
                if len(mode) > 0:
                    df[col] = df[col].fillna(mode[0])
                else:
                    df = df.drop(columns=[col])
                    continue

        # Drop constant columns
        if df[col].nunique() == 1:
            df = df.drop(columns=[col])
            continue

        # Handle rare categories — group into "Other"
        if col_type == ColumnType.CATEGORICAL_LOW:
            freq = df[col].value_counts(normalize=True)
            rare = freq[freq < 0.01].index
            if len(rare) > 0:
                df[col] = df[col].apply(lambda x: "Other" if x in rare else x)

    return df


def apply_domain_constraints(synthetic_df: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce minimal, safe domain constraints on synthetic data.
    Philosophy: only enforce what we know for certain.
    Don't over-constrain — that hurts distributions more than it helps.
    """
    synthetic_df = synthetic_df.copy()

    age_keywords = ["age", "years", "yr", "yrs"]
    probability_keywords = ["probability", "prob", "ratio", "proportion", "fraction"]
    positive_keywords = [
        "price", "cost", "salary", "income", "revenue", "sales",
        "amount", "fee", "wage", "spend", "budget"
    ]
    count_keywords = ["count", "quantity", "qty", "frequency", "occurrences"]

    for col in synthetic_df.columns:
        if col not in real_df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(synthetic_df[col]):
            continue

        col_lower = col.lower()
        real_col = real_df[col].dropna()
        if len(real_col) == 0:
            continue
        real_min = real_col.min()
        real_max = real_col.max()
        real_std = real_col.std() if real_col.std() > 0 else 1.0

        # ── Rule 1: Binary/label columns — never touch ────────────────────────
        is_binary = real_col.nunique() == 2
        if is_binary:
            synthetic_df[col] = synthetic_df[col].clip(lower=real_min, upper=real_max)
            continue

        # ── Rule 2: Columns where 0-1 range is the definition ─────────────────
        if real_min >= 0 and real_max <= 1.0:
            synthetic_df[col] = synthetic_df[col].clip(lower=0.0, upper=1.0)
            continue

        # ── Rule 3: Age — integer, bounded to real range with small buffer ─────
        if any(kw in col_lower for kw in age_keywords):
            synthetic_df[col] = synthetic_df[col].clip(
                lower=max(1, real_min),
                upper=min(150, real_max + real_std)
            )
            synthetic_df[col] = _safe_int(synthetic_df[col])
            continue

        # ── Rule 4: Probability columns — 0 to 1 ─────────────────────────────
        if any(kw in col_lower for kw in probability_keywords) and real_max <= 1.0:
            synthetic_df[col] = synthetic_df[col].clip(lower=0.0, upper=1.0)
            continue

        # ── Rule 5: Always-positive business columns ──────────────────────────
        if any(kw in col_lower for kw in positive_keywords) and real_min >= 0:
            synthetic_df[col] = synthetic_df[col].clip(lower=0.0)
            continue

        # ── Rule 6: Count columns — non-negative integers ─────────────────────
        if any(kw in col_lower for kw in count_keywords):
            synthetic_df[col] = synthetic_df[col].clip(lower=0)
            synthetic_df[col] = _safe_int(synthetic_df[col])
            continue

        # ── Rule 7: Universal safety net — generous 3 std buffer ──────────────
        # Only prevents extreme extrapolation, doesn't fight distributions
        hard_lower = real_min - (real_std * 3)
        hard_upper = real_max + (real_std * 3)

        # Never negative if real data was non-negative
        if real_min >= 0:
            hard_lower = max(0, hard_lower)

        synthetic_df[col] = synthetic_df[col].clip(lower=hard_lower, upper=hard_upper)

        # ── Rule 8: Integer restoration ───────────────────────────────────────
        if pd.api.types.is_integer_dtype(real_df[col]):
            synthetic_df[col] = _safe_int(synthetic_df[col])

    return synthetic_df


def bootstrap_tiny_dataset(df: pd.DataFrame, num_rows: int, noise_factor: float = 0.05) -> pd.DataFrame:
    """
    For tiny datasets (<50 rows), use statistical bootstrapping with small noise
    instead of generative models. This preserves correlations and distributions
    much better than neural networks on tiny data.
    """
    import random as rnd

    rows = []
    for _ in range(num_rows):
        # Sample a random real row
        base_row = df.sample(n=1, random_state=rnd.randint(0, 10000)).iloc[0].copy()

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                col_std = df[col].std()
                if col_std > 0:
                    # Add small Gaussian noise proportional to std
                    noise = np.random.normal(0, col_std * noise_factor)
                    base_row[col] = base_row[col] + noise

                    # Keep within real data range
                    base_row[col] = np.clip(
                        base_row[col],
                        df[col].min() - col_std,
                        df[col].max() + col_std
                    )

                    # Restore integer if needed
                    if pd.api.types.is_integer_dtype(df[col]):
                        base_row[col] = int(round(base_row[col]))

        rows.append(base_row)

    return pd.DataFrame(rows).reset_index(drop=True)