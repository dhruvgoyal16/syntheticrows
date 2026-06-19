import pandas as pd
import numpy as np
from typing import Tuple,Optional
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

    # Try to parse as datetime by name — only if values look like actual dates
    # (not just columns that happen to contain the word "year" or "day")
    if any(kw in col_name.lower() for kw in ["date", "time", "timestamp"]):
        try:
            pd.to_datetime(col)
            return ColumnType.DATETIME
        except Exception:
            pass

    # Boolean
    if col.nunique() == 2 and set(col.dropna().unique()).issubset({0, 1, True, False, "yes", "no", "true", "false"}):
        return ColumnType.BOOLEAN

    # ID column — unique per row
    # ID column — unique per row, but ONLY for integers or strings.
    # Continuous floats are naturally near-unique (every measurement differs),
    # so all-unique floats are normal data, NOT identifiers — never flag them.
    if col.nunique() / len(col) > 0.95 and len(col) > 20:
        is_float = pd.api.types.is_float_dtype(col)
        if not is_float:
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
    # Missing values
    missing = col.isna().sum()
    if missing > 0:
        pct = round((missing / total) * 100, 1)
        if missing == total:
            # Column is 100% empty — nothing to fill from, so it can't be
            # synthesized. Drop it instead of trying (and failing) to fill.
            issues.append(ColumnIssue(
                issue=f"Column is completely empty ({pct}% missing) — no values to learn from",
                fix_type="drop_column",
                recommendation="Drop this column — it has no data to synthesize",
                severity="high"
            ))
        else:
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

    # Outliers — only for numerical (IQR method: robust to extreme values,
    # unlike mean/std which the outliers themselves corrupt)
    if col_type in [ColumnType.NUMERICAL_CONTINUOUS, ColumnType.NUMERICAL_DISCRETE]:
        numeric_col = pd.to_numeric(col, errors="coerce").dropna()
        if len(numeric_col) >= 4:  # need enough points for meaningful quartiles
            q1 = numeric_col.quantile(0.25)
            q3 = numeric_col.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outlier_count = ((numeric_col < lower_bound) | (numeric_col > upper_bound)).sum()
                if outlier_count > 0:
                    issues.append(ColumnIssue(
                        issue=f"{outlier_count} outliers detected (outside the typical value range)",
                        fix_type="cap_outliers",
                        recommendation="Cap extreme values to the normal range (IQR method)",
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

# Date columns named like these are almost always an attribute of an entity
# (a customer, an order) rather than a true time axis — so they don't make a
# dataset "time-series".
_EVENT_DATE_HINTS = [
    "signup", "sign_up", "created", "updated", "modified", "registered",
    "joined", "birth", "dob", "order", "purchase", "last_", "_at",
    "expiry", "expiration", "hire", "enroll"
]


def _looks_like_event_date(col_name: str) -> bool:
    n = col_name.lower()
    return any(h in n for h in _EVENT_DATE_HINTS)


def _is_true_time_series(df: pd.DataFrame, datetime_cols: list) -> bool:
    """
    Conservative time-series detection. Only returns True when the evidence is
    strong, because misrouting ordinary tabular data (with an incidental date
    like signup_date) to the time-series model breaks class balancing and other
    tabular features. When unsure we default to tabular; the user can override.

    A dataset is treated as time-series only if ALL hold:
      1. Exactly one datetime column (multiple date cols => entity table).
      2. That column isn't named like an event/attribute date (signup_date etc.).
      3. The datetime values are highly unique (~one row per timestamp).
      4. The dataset is narrow (date + a few measured columns).
    """
    n_rows = len(df)
    if n_rows == 0 or len(datetime_cols) != 1:
        return False

    dt_col = datetime_cols[0]

    if _looks_like_event_date(dt_col):
        return False

    uniqueness = df[dt_col].nunique(dropna=True) / n_rows
    if uniqueness < 0.95:
        return False

    non_dt_cols = df.shape[1] - len(datetime_cols)
    if non_dt_cols > 4:
        return False

    return True


def detect_dataset_type(df: pd.DataFrame, col_types: dict) -> DatasetType:
    # Time series — only when the evidence is strong (see _is_true_time_series).
    # An incidental date column (signup_date, created_at, ...) does NOT make a
    # dataset time-series; that data should route through the tabular models.
    datetime_cols = [c for c, t in col_types.items() if t == ColumnType.DATETIME]
    if datetime_cols and _is_true_time_series(df, datetime_cols):
        return DatasetType.TIME_SERIES

    # Imbalanced — check if any column looks like a binary target
    for col in df.columns:
        if df[col].nunique() == 2:
            counts = df[col].value_counts(normalize=True)
            if counts.iloc[0] > 0.80:
                return DatasetType.IMBALANCED

    return DatasetType.STANDARD


# Name hints suggesting a CONTINUOUS (regression) target
_REGRESSION_HINTS = [
    "price", "charges", "charge", "cost", "amount", "salary", "wage", "revenue",
    "sales", "income", "value", "amt", "total", "spend", "spending", "fare", "rent",
    "premium", "balance", "duration", "length", "weight", "height", "temperature",
    "temp", "consumption", "demand", "yield", "profit", "mpg",
]

# Name hints suggesting a CLASSIFICATION target
_CLASSIFICATION_HINTS = [
    "default", "churn", "fraud", "survived", "outcome", "target", "label", "class",
    "result", "approved", "converted", "clicked", "purchased", "cancelled", "failed",
    "success", "win", "loss", "y", "output", "response", "event", "flag", "status",
    "disease", "died", "dead", "active", "inactive", "valid", "invalid", "sentiment",
    "category", "grade", "type", "tier", "level",
]


def _is_continuous_target(df: pd.DataFrame, col: str, n_rows: int) -> bool:
    """Numeric column with high cardinality -> plausible regression target.
    Rejects low-cardinality (classification) columns and perfect-ID integers."""
    if not pd.api.types.is_numeric_dtype(df[col]):
        return False
    nu = df[col].nunique(dropna=True)
    if nu <= 20:
        return False
    if pd.api.types.is_float_dtype(df[col]):
        return True
    uniq_ratio = nu / n_rows
    return 0.05 < uniq_ratio < 0.999


def _is_class_candidate(df: pd.DataFrame, col: str, n_rows: int) -> bool:
    """2-20 recurring distinct values -> plausible classification target."""
    nu = df[col].nunique(dropna=True)
    return 2 <= nu <= 20 and (nu / n_rows) < 0.5


def classify_target_type(df: pd.DataFrame, col: str) -> Optional[str]:
    """Label a chosen target column as 'regression' or 'classification'.
    Used both for the detected target and for a user-confirmed target."""
    if col is None or col not in df.columns:
        return None
    return "regression" if _is_continuous_target(df, col, len(df)) else "classification"


def detect_target_column(df: pd.DataFrame) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Suggest a likely target column AND its type ('classification' | 'regression').
    Suggestion only — the user confirms/overrides in the UI.

    Strategy (position is the strongest real-world signal):
      1. The LAST column, if it's a plausible target, judged by its own type.
      2. Name hints (classification names first, then regression), as fallback.
      3. Existing classification structural logic.
      4. Any continuous column as a last resort.
    """
    n_rows = len(df)
    if n_rows == 0 or len(df.columns) == 0:
        return False, None, None

    cols = list(df.columns)
    last = cols[-1]

    # Priority 1 — the conventional target position: the last column.
    if _is_continuous_target(df, last, n_rows):
        return True, last, "regression"
    if _is_class_candidate(df, last, n_rows):
        return True, last, "classification"

    # Priority 2 — name hints (only if the last column wasn't a clear target).
    for col in reversed(cols):
        if any(h in col.lower() for h in _CLASSIFICATION_HINTS) and _is_class_candidate(df, col, n_rows):
            return True, col, "classification"
    for col in reversed(cols):
        if any(h in col.lower() for h in _REGRESSION_HINTS) and _is_continuous_target(df, col, n_rows):
            return True, col, "regression"

    # Priority 3 — classification structural logic (unchanged from before).
    class_cols = [c for c in cols if _is_class_candidate(df, c, n_rows)]
    if class_cols:
        for col in reversed(class_cols):
            if pd.api.types.is_numeric_dtype(df[col]):
                vals = set(df[col].dropna().unique())
                if vals.issubset({0, 1, 0.0, 1.0}):
                    return True, col, "classification"
        binary = [c for c in class_cols if df[c].nunique(dropna=True) == 2]
        if binary:
            return True, binary[-1], "classification"
        return True, class_cols[-1], "classification"

    # Priority 4 — any continuous column as a last resort.
    cont = [c for c in cols if _is_continuous_target(df, c, n_rows)]
    if cont:
        return True, cont[-1], "regression"

    return False, None, None

def detect_imbalance(df: pd.DataFrame, target_col: str = None) -> Tuple[bool, float]:
    check_col = target_col
    if check_col is None:
        for col in df.columns:
            if df[col].nunique() == 2:
                check_col = col
                break
    if check_col is None:
        return False, None

    counts = df[check_col].value_counts(normalize=True)
    ratio = round(float(counts.iloc[0]), 3)

    # Also check string binary columns like "0"/"1" or "yes"/"no"
    if ratio <= 0.60 and check_col is None:
        for col in df.columns:
            col_vals = df[col].astype(str).str.lower().unique()
            if set(col_vals).issubset({"0", "1", "yes", "no", "true", "false"}):
                counts = df[col].value_counts(normalize=True)
                ratio = round(float(counts.iloc[0]), 3)
                if ratio > 0.60:
                    check_col = col
                    break

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
    has_target, target_col, target_type = detect_target_column(df)
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
        target_type=target_type,
        is_imbalanced=is_imbalanced,
        imbalance_ratio=imbalance_ratio,
        columns=column_profiles,
        columns_to_drop=columns_to_drop
    )