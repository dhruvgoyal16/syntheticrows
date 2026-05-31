import pandas as pd
import numpy as np
from typing import List, Tuple
from .models import Fix, ColumnType, DatasetProfile


def detect_skewed_columns(df: pd.DataFrame, threshold: float = 1.0) -> List[str]:
    """Detect numerical columns with high skewness."""
    skewed = []
    for col in df.select_dtypes(include=[np.number]).columns:
        skewness = abs(df[col].skew())
        if skewness > threshold and df[col].min() >= 0:
            skewed.append(col)
    return skewed

def get_integer_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns that should be integers."""
    int_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].dropna().apply(lambda x: x == int(x)).all():
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

        # Round integers
        if pd.api.types.is_integer_dtype(original_df[col]):
            synthetic_df[col] = synthetic_df[col].round().astype(int)
        elif original_df[col].dropna().apply(lambda x: x == int(x)).all():
            synthetic_df[col] = synthetic_df[col].round().astype(int)
        else:
            # Round floats to same decimal places as original
            sample_vals = original_df[col].dropna().head(100)
            avg_decimals = sample_vals.apply(
                lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0
            ).median()
            decimal_places = min(int(avg_decimals), 4)
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
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

        elif fix.fix_type == "fix_zeros":
            median_val = df[col].replace(0, np.nan).median()
            df[col] = df[col].replace(0, median_val)

        elif fix.fix_type == "cap_outliers":
            mean = df[col].mean()
            std = df[col].std()
            df[col] = df[col].clip(lower=mean - 3 * std, upper=mean + 3 * std)

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
                df[col] = df[col].fillna(df[col].median())
            else:
                mode = df[col].mode()
                if len(mode) > 0:
                    df[col] = df[col].fillna(mode[0])

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