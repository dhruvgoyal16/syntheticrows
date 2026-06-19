from pydantic import BaseModel
from typing import List, Optional
from enum import Enum


# ─── Enums ────────────────────────────────────────────────────────────────────

class ColumnType(str, Enum):
    NUMERICAL_CONTINUOUS = "numerical_continuous"
    NUMERICAL_DISCRETE   = "numerical_discrete"
    CATEGORICAL_LOW      = "categorical_low"
    CATEGORICAL_HIGH     = "categorical_high"
    BOOLEAN              = "boolean"
    DATETIME             = "datetime"
    ID_COLUMN            = "id_column"

class DatasetType(str, Enum):
    STANDARD      = "standard"
    TIME_SERIES   = "time_series"
    IMBALANCED    = "imbalanced"
    UNSUPERVISED  = "unsupervised"

class DatasetSize(str, Enum):
    TINY   = "tiny"    # <100 rows
    SMALL  = "small"   # 100-500 rows
    MEDIUM = "medium"  # 500-5000 rows
    LARGE  = "large"   # 5000+ rows

class ModelType(str, Enum):
    GAUSSIAN_COPULA = "GaussianCopula"
    CTGAN           = "CTGAN"
    TVAE            = "TVAE"
    PAR             = "PAR"


# ─── Column Profile ────────────────────────────────────────────────────────────

class ColumnIssue(BaseModel):
    issue: str
    fix_type: str
    recommendation: str
    severity: str  # "high", "medium", "low"

class ColumnProfile(BaseModel):
    name: str
    dtype: str
    col_type: ColumnType
    num_missing: int
    missing_pct: float
    num_zeros: int
    zero_pct: float
    num_outliers: int
    num_unique: int
    unique_pct: float
    issues: List[ColumnIssue]


# ─── Dataset Profile ───────────────────────────────────────────────────────────

class DatasetProfile(BaseModel):
    filename: str
    num_rows: int
    num_cols: int
    size_category: DatasetSize
    dataset_type: DatasetType
    has_datetime: bool
    has_target_column: bool
    target_column: Optional[str]
    target_type: Optional[str] = None  # "classification" | "regression" | None
    is_imbalanced: bool
    imbalance_ratio: Optional[float]
    columns: List[ColumnProfile]
    columns_to_drop: List[str]


# ─── Fix (user controlled) ─────────────────────────────────────────────────────

class Fix(BaseModel):
    column: str
    issue: str
    fix_type: str
    approved: bool


# ─── Validation Results ────────────────────────────────────────────────────────

class ColumnQuality(BaseModel):
    column: str
    score: float
    grade: str   # "Excellent", "Good", "Fair", "Poor"
    mean_diff_pct: float
    std_diff_pct: float

class ValidationResult(BaseModel):
    distinguishability_score: float
    statistical_score: float
    coverage_score: float
    final_score: float
    grade: str
    color: str
    column_quality: List[ColumnQuality]
    tstr: Optional[dict] = None