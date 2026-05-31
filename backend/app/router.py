from .models import DatasetProfile, DatasetSize, DatasetType, ColumnType, ModelType


def select_model(profile: DatasetProfile) -> tuple[ModelType, dict]:
    """
    Returns (model_type, model_kwargs) based on dataset profile.
    """

    # Time series — always use PAR
    if profile.dataset_type == DatasetType.TIME_SERIES:
        return ModelType.PAR, {}

    num_rows = profile.num_rows

    # Check if dataset is purely numerical and clean
    non_datetime_cols = [
        c for c in profile.columns
        if c.col_type not in [ColumnType.DATETIME, ColumnType.ID_COLUMN, ColumnType.CATEGORICAL_HIGH]
    ]
    all_numerical = all(
        c.col_type in [ColumnType.NUMERICAL_CONTINUOUS, ColumnType.NUMERICAL_DISCRETE, ColumnType.BOOLEAN]
        for c in non_datetime_cols
    )
    total_issues = sum(len(c.issues) for c in profile.columns)
    is_clean = total_issues == 0

    # Imbalanced — always use CTGAN with conditional sampling
    if profile.is_imbalanced:
        epochs = 300 if num_rows < 2000 else 200
        return ModelType.CTGAN, {"epochs": epochs, "conditional": True}

    # Tiny datasets — TVAE handles small data best
    if num_rows < 100:
        return ModelType.TVAE, {}

    # Small datasets
    if num_rows < 500:
        if all_numerical and is_clean:
            return ModelType.GAUSSIAN_COPULA, {}
        return ModelType.TVAE, {}

    # Medium datasets
    if num_rows < 5000:
        if all_numerical and is_clean:
            return ModelType.GAUSSIAN_COPULA, {}
        epochs = 300 if num_rows < 2000 else 200
        return ModelType.CTGAN, {"epochs": epochs}

    # Large datasets
    return ModelType.CTGAN, {"epochs": 100}