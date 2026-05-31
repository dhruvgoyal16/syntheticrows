import pandas as pd
import numpy as np
from typing import Tuple
from .models import (
    ColumnType, DatasetType, DatasetSize,
    ColumnIssue, ColumnProfile, DatasetProfile
)

# ─── Column Type Detection ─────────────────────────────────────────────────────

def detect_column_type(col: pd.Series, col_name: str) -> ColumnType:
    # DateTime
    if pd.api.types.is_datetime64_any_dtype(col):
        return ColumnType.DATETIME
    if col.dtype == object:
        try:
            pd.to_datetime(col)
            return ColumnType.DATETIME
        except Exception:
            pass

    # Try to parse as datetime by name
    if any(kw in col_name.lower() for kw in ["date", "time", "timestamp", "year", "month", "day"]):
        try:
            pd.to_datetime(col)
            return ColumnType.DATETIME
        except Exception:
            pass

    # Boolean
    if col.nunique() == 2 and set(col.dropna().unique()).issubset({0, 1, True, False, "yes", "no", "true", "false"}):
        return ColumnType.BOOLEAN

    # ID column — unique per row
    if col.nunique() / len(col) > 0.95 and len(col) > 20:
        return ColumnType.ID_COLUMN

    # Categorical
    if col.dtype == object or col.dtype.name == "category":
        unique_pct = col.nunique() / len(col)
        if unique_pct > 0.5:
            return ColumnType.CATEGORICAL_HIGH
        return ColumnType.CATEGORICAL_LOW

    # Numerical
    if pd.api.types.is_integer_dtype(col):
        if col.nunique() <= 20:
            return ColumnType.NUMERICAL_DISCRETE
        return ColumnType.NUMERICAL_CONTINUOUS

    if pd.api.types.is_float_dtype(col):
        return ColumnType.NUMERICAL_CONTINUOUS

    return ColumnType.CATEGORICAL_LOW


# ─── Column Issue Detection ────────────────────────────────────────────────────

def detect_column_issues(col: pd.Series, col_name: str, col_type: ColumnType) -> list:
    issues = []
    total = len(col)

    # Missing values
    missing = col.isna().sum()
    if missing > 0:
        pct = round((missing / total) * 100, 1)
        issues.append(ColumnIssue(
            issue=f"{missing} missing values ({pct}% of rows)",
            fix_type="fill_missing",
            recommendation="Fill with column median" if col_type in [
                ColumnType.NUMERICAL_CONTINUOUS,
                ColumnType.NUMERICAL_DISCRETE
            ] else "Fill with most frequent value",
            severity="high" if pct > 20 else "medium"
        ))

    # Zero inflation — only for continuous numerical
    if col_type == ColumnType.NUMERICAL_CONTINUOUS:
        skip_zero_check = col_name.lower() in [
            "outcome", "target", "label", "class",
            "count", "quantity", "flag"
        ]
        if not skip_zero_check:
            zero_count = (col == 0).sum()
            zero_pct = (zero_count / total) * 100
            if zero_pct > 10:
                median_val = round(col.replace(0, np.nan).median(), 2)
                issues.append(ColumnIssue(
                    issue=f"{zero_count} zero values ({round(zero_pct, 1)}% of rows) — may indicate missing data",
                    fix_type="fix_zeros",
                    recommendation=f"Replace zeros with column median ({median_val})",
                    severity="high" if zero_pct > 30 else "medium"
                ))

    # Outliers — only for numerical
    if col_type in [ColumnType.NUMERICAL_CONTINUOUS, ColumnType.NUMERICAL_DISCRETE]:
        numeric_col = pd.to_numeric(col, errors="coerce").dropna()
        if len(numeric_col) > 0:
            mean = numeric_col.mean()
            std = numeric_col.std()
            if std > 0:
                outlier_count = ((numeric_col - mean).abs() > 3 * std).sum()
                if outlier_count > 0:
                    issues.append(ColumnIssue(
                        issue=f"{outlier_count} outliers detected (beyond 3 std deviations)",
                        fix_type="cap_outliers",
                        recommendation="Cap values at ±3 standard deviations",
                        severity="medium"
                    ))

    # Constant column
    if col.nunique() == 1:
        issues.append(ColumnIssue(
            issue="Column has only one unique value — useless for training",
            fix_type="drop_column",
            recommendation="Drop this column before generation",
            severity="high"
        ))

    # High cardinality categorical
    if col_type == ColumnType.CATEGORICAL_HIGH:
        issues.append(ColumnIssue(
            issue=f"High cardinality — {col.nunique()} unique values ({round(col.nunique()/total*100, 1)}% of rows)",
            fix_type="drop_column",
            recommendation="Drop this column — too many unique values to synthesize meaningfully",
            severity="high"
        ))

    # ID column
    if col_type == ColumnType.ID_COLUMN:
        issues.append(ColumnIssue(
            issue="Likely an ID column — unique identifier per row",
            fix_type="drop_column",
            recommendation="Drop this column — ID columns should never be synthesized",
            severity="high"
        ))

    return issues


# ─── Dataset Type Detection ────────────────────────────────────────────────────

def detect_dataset_type(df: pd.DataFrame, col_types: dict) -> DatasetType:
    # Time series — has datetime column
    datetime_cols = [c for c, t in col_types.items() if t == ColumnType.DATETIME]
    if datetime_cols:
        return DatasetType.TIME_SERIES

    # Imbalanced — check if any column looks like a binary target
    for col in df.columns:
        if df[col].nunique() == 2:
            counts = df[col].value_counts(normalize=True)
            if counts.iloc[0] > 0.80:
                return DatasetType.IMBALANCED

    return DatasetType.STANDARD


def detect_target_column(df: pd.DataFrame) -> Tuple[bool, str]:
    target_keywords = ["target", "label", "class", "outcome", "y", "output", "result"]
    for col in df.columns:
        if col.lower() in target_keywords:
            return True, col
    return False, None


def detect_imbalance(df: pd.DataFrame, target_col: str = None) -> Tuple[bool, float]:
    check_col = target_col
    if check_col is None:
        # Find most likely target column
        for col in df.columns:
            if df[col].nunique() == 2:
                check_col = col
                break
    if check_col is None:
        return False, None

    counts = df[check_col].value_counts(normalize=True)
    ratio = round(float(counts.iloc[0]), 3)
    # 60/40 split or worse is considered imbalanced for ML purposes
    return ratio > 0.60, ratio

# ─── Size Category ─────────────────────────────────────────────────────────────

def get_size_category(num_rows: int) -> DatasetSize:
    if num_rows < 100:
        return DatasetSize.TINY
    elif num_rows < 500:
        return DatasetSize.SMALL
    elif num_rows < 5000:
        return DatasetSize.MEDIUM
    return DatasetSize.LARGE


# ─── Main Profiler ─────────────────────────────────────────────────────────────

def profile_dataset(df: pd.DataFrame, filename: str = "dataset.csv") -> DatasetProfile:
    col_types = {}
    column_profiles = []
    columns_to_drop = []

    for col_name in df.columns:
        col = df[col_name]
        col_type = detect_column_type(col, col_name)
        col_types[col_name] = col_type

        issues = detect_column_issues(col, col_name, col_type)

        # Track columns that should be dropped
        drop_issues = [i for i in issues if i.fix_type == "drop_column"]
        if drop_issues:
            columns_to_drop.append(col_name)

        numeric_col = pd.to_numeric(col, errors="coerce")
        zero_count = int((col == 0).sum()) if pd.api.types.is_numeric_dtype(col) else 0
        zero_pct = round((zero_count / len(col)) * 100, 1) if len(col) > 0 else 0.0

        mean = numeric_col.mean() if not numeric_col.isna().all() else 0
        std = numeric_col.std() if not numeric_col.isna().all() else 0
        outlier_count = 0
        if std and std > 0:
            outlier_count = int(((numeric_col - mean).abs() > 3 * std).sum())

        column_profiles.append(ColumnProfile(
            name=col_name,
            dtype=str(col.dtype),
            col_type=col_type,
            num_missing=int(col.isna().sum()),
            missing_pct=round((col.isna().sum() / len(col)) * 100, 1),
            num_zeros=zero_count,
            zero_pct=zero_pct,
            num_outliers=outlier_count,
            num_unique=int(col.nunique()),
            unique_pct=round((col.nunique() / len(col)) * 100, 1),
            issues=issues
        ))

    dataset_type = detect_dataset_type(df, col_types)
    has_target, target_col = detect_target_column(df)
    is_imbalanced, imbalance_ratio = detect_imbalance(df, target_col)

    return DatasetProfile(
        filename=filename,
        num_rows=len(df),
        num_cols=len(df.columns),
        size_category=get_size_category(len(df)),
        dataset_type=dataset_type,
        has_datetime=any(t == ColumnType.DATETIME for t in col_types.values()),
        has_target_column=has_target,
        target_column=target_col,
        is_imbalanced=is_imbalanced,
        imbalance_ratio=imbalance_ratio,
        columns=column_profiles,
        columns_to_drop=columns_to_drop
    )