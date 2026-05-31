import pandas as pd
import numpy as np
import random
from .models import ModelType, DatasetProfile
from .router import select_model
from .preprocessor import detect_skewed_columns, log_transform, reverse_log_transform, restore_dtypes
from sdv.metadata import Metadata
from sdv.single_table import (
    GaussianCopulaSynthesizer,
    CTGANSynthesizer,
    TVAESynthesizer,
)


def build_synthesizer(model_type: ModelType, metadata: Metadata, kwargs: dict):
    if model_type == ModelType.GAUSSIAN_COPULA:
        return GaussianCopulaSynthesizer(metadata)
    elif model_type == ModelType.CTGAN:
        epochs = kwargs.get("epochs", 300)
        return CTGANSynthesizer(metadata, epochs=epochs, verbose=False, cuda=False)
    elif model_type == ModelType.TVAE:
        return TVAESynthesizer(metadata)
    else:
        return CTGANSynthesizer(metadata, epochs=200, verbose=False, cuda=False)


def generate(
    df: pd.DataFrame,
    profile: DatasetProfile,
    num_rows: int
) -> tuple[pd.DataFrame, str]:
    """
    Returns (synthetic_df, model_name_used)
    """
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    model_type, model_kwargs = select_model(profile)

    # Detect and apply log transform for skewed columns
    skewed_cols = detect_skewed_columns(df)
    if skewed_cols:
        df_transformed = log_transform(df, skewed_cols)
    else:
        df_transformed = df.copy()

    metadata = Metadata.detect_from_dataframe(df_transformed)
    synthesizer = build_synthesizer(model_type, metadata, model_kwargs)

    synthesizer.fit(df_transformed)
    synthetic_transformed = synthesizer.sample(num_rows=num_rows)

    # Reverse the log transform on synthetic data
    if skewed_cols:
        synthetic_df = reverse_log_transform(synthetic_transformed, skewed_cols)
    else:
        synthetic_df = synthetic_transformed

    # Restore integer types and clip to original ranges
    synthetic_df = restore_dtypes(synthetic_df, df)

    return synthetic_df, model_type.value